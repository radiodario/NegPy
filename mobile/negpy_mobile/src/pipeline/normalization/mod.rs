//! Normalization, the base stage's second half: `NormalizationProcessor.process` and the
//! helpers of `features/exposure/normalization.py` it runs, plus the Contrast Mask and
//! Diffusion planes the engine builds from the source between base and exposure (`planes`).
//!
//! Bounds and meter readings are `f64` here, as the desktop holds them as Python floats; the
//! kernels read the bounds as float32, as `normalize_log_image` casts them. `Metrics` stores
//! the readings as float32, so `normalize_detailed` also hands back the full-precision values.

mod axis;
mod bounds;
mod planes;

use std::borrow::Cow;

use negpy_core::analysis;
use negpy_core::meter::{self, Activity, ChannelNorm, NeutralConsts};

use crate::config::{BwFilter, ExposureConfig, ProcessConfig, ProcessMode, Rgb};
use crate::context::{LogNegativeBounds, NeutralAxis, PipelineContext};
use crate::image::{Image, Roi};
use crate::pipeline::constants::EXPOSURE_CONSTANTS;

pub use axis::{blend_neutral_axis, pooled_neutral_axis};
pub use bounds::{
    analyze_log_exposure_bounds_from_log, density_above_base, luma_source_bounds,
    luminance_density_range, mix_luma_color_bounds, resolve_bounds_detailed,
};
pub use planes::{
    analysis_grid, build_contrast_mask, build_diffusion, contrast_mask_plane, diffusion_grid,
    diffusion_plane, expand_mask_plane, geometry_kwargs, normalized_roi,
};

/// Rec. 709 luma as Python floats (`domain/types.py`), for the float64 math on the bounds.
pub(crate) const LUMA: [f64; 3] = [0.2126, 0.7152, 0.0722];

/// Built-in crosstalk matrix, row-major (`process/models.py`).
const DEFAULT_CROSSTALK_MATRIX: [f64; 9] = [1.0, -0.05, -0.02, -0.04, 1.0, -0.08, -0.01, -0.1, 1.0];

/// Detection resolution of the clear-base read (`geometry/logic.py`).
const AUTOCROP_DETECT_RES: usize = 1800;

/// The B&W print's channel weights on the normalized log channels (`exposure/bw_filter.py`).
pub fn bw_filter_weights(filter: BwFilter) -> [f64; 3] {
    match filter {
        BwFilter::None => [0.31, 0.39, 0.30],
        BwFilter::Yellow8 => [0.47, 0.57, -0.04],
        BwFilter::DeepYellow15 => [0.56, 0.50, -0.06],
        BwFilter::Orange21 => [0.72, 0.32, -0.04],
        BwFilter::Red25 => [1.06, -0.06, 0.00],
        BwFilter::YellowGreen11 => [0.25, 0.78, -0.03],
        BwFilter::Green58 => [0.04, 1.08, -0.12],
    }
}

/// `collapse_weights`: a filter acts only on a color source.
pub fn collapse_weights(process: &ProcessConfig, exposure: &ExposureConfig) -> [f64; 3] {
    if process.process_mode == ProcessMode::Bw && process.source_color == Some(true) {
        bw_filter_weights(exposure.bw_filter)
    } else {
        bw_filter_weights(BwFilter::None)
    }
}

/// `collapse_matrix`: every row is `weights`, so `matmul_3x3` gives three equal channels.
pub fn collapse_matrix(weights: [f64; 3]) -> [[f32; 3]; 3] {
    let row = weights.map(|w| w as f32);
    [row, row, row]
}

/// `plane_weights`: the weights the mask and diffusion planes collapse with; None outside B&W.
pub fn plane_weights(process: &ProcessConfig, exposure: &ExposureConfig) -> Option<[f64; 3]> {
    (process.process_mode == ProcessMode::Bw).then(|| collapse_weights(process, exposure))
}

/// D-min / D-max per channel in raw log10, as Python floats (`LogNegativeBounds`).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Bounds {
    pub floors: [f64; 3],
    pub ceils: [f64; 3],
}

impl Bounds {
    pub fn new(floors: [f64; 3], ceils: [f64; 3]) -> Self {
        Self { floors, ceils }
    }

    pub fn from_rgb(floors: Rgb, ceils: Rgb) -> Self {
        Self::new(
            [floors.0 as f64, floors.1 as f64, floors.2 as f64],
            [ceils.0 as f64, ceils.1 as f64, ceils.2 as f64],
        )
    }

    pub fn from_metric(b: &LogNegativeBounds) -> Self {
        Self::from_rgb(b.floors, b.ceils)
    }

    pub fn to_metric(self) -> LogNegativeBounds {
        let [f0, f1, f2] = self.floors32();
        let [c0, c1, c2] = self.ceils32();
        LogNegativeBounds {
            floors: (f0, f1, f2),
            ceils: (c0, c1, c2),
        }
    }

    /// The floors as the normalization kernel reads them.
    pub fn floors32(&self) -> [f32; 3] {
        self.floors.map(|v| v as f32)
    }

    pub fn ceils32(&self) -> [f32; 3] {
        self.ceils.map(|v| v as f32)
    }
}

/// The neutral axis in the meter's own precision: `measure_neutral_axis_from_log`'s tuple,
/// with a pooled axis's offset weight (`pool_neutral_axis`).
#[derive(Clone, Debug, PartialEq)]
pub struct Axis {
    pub mid: [f64; 3],
    pub shadow: [f64; 3],
    pub highlight: Option<[f64; 3]>,
    pub confidence: f64,
    pub offset_weight: Option<f64>,
}

impl Axis {
    pub fn to_metric(&self) -> NeutralAxis {
        let rgb = |r: [f64; 3]| (r[0] as f32, r[1] as f32, r[2] as f32);
        NeutralAxis {
            mid: rgb(self.mid),
            shadow: rgb(self.shadow),
            highlight: self.highlight.map(rgb),
            confidence: self.confidence as f32,
            offset_weight: self.offset_weight.map(|w| w as f32),
        }
    }
}

/// The effective crosstalk matrix, row-major, as `resolve_crosstalk_matrix` builds it.
pub type Unmix = [[f64; 3]; 3];

/// `resolve_crosstalk_matrix`: identity-to-calibration blend by `strength`, row-normalized;
/// None when off.
pub fn resolve_crosstalk_matrix(strength: f64, matrix: Option<&[f32]>) -> Option<Unmix> {
    if strength <= 0.0 {
        return None;
    }
    let cal: [f64; 9] = match matrix {
        Some(m) if m.len() == 9 => std::array::from_fn(|i| m[i] as f64),
        _ => DEFAULT_CROSSTALK_MATRIX,
    };
    let mut applied: Unmix = std::array::from_fn(|i| {
        std::array::from_fn(|j| {
            let eye = if i == j { 1.0 } else { 0.0 };
            eye * (1.0 - strength) + cal[i * 3 + j] * strength
        })
    });
    for row in applied.iter_mut() {
        let sum = (row[0] + row[1] + row[2]).max(1e-6);
        for v in row.iter_mut() {
            *v /= sum;
        }
    }
    Some(applied)
}

/// `effective_crosstalk_matrix`: the configured unmix only for the film it was derived for.
pub fn effective_crosstalk_matrix(
    process: &ProcessConfig,
    mode: Option<ProcessMode>,
) -> Option<Unmix> {
    if mode.is_some_and(|m| process.crosstalk_process != m) {
        return None;
    }
    resolve_crosstalk_matrix(
        process.crosstalk_strength as f64,
        process.crosstalk_matrix.as_deref(),
    )
}

/// `unmix_log_image` on one pixel: the matrix in float32, each row's products summed in order,
/// as `np.einsum` accumulates them.
fn unmix_pixel(p: [f32; 3], m: &[[f32; 3]; 3]) -> [f32; 3] {
    m.map(|row| p[0] * row[0] + p[1] * row[1] + p[2] * row[2])
}

/// `unmix_log_image`: the unmix matrix over an HxWx3 log image; the image itself when None.
pub fn unmix_log_image(img: Image, unmix: Option<&Unmix>) -> Image {
    let Some(m) = unmix else {
        return img;
    };
    let m32 = m.map(|row| row.map(|v| v as f32));
    let mut out = img;
    for px in out.data.as_chunks_mut::<3>().0 {
        *px = unmix_pixel(*px, &m32);
    }
    out
}

/// `to_log_density`: log10 of the scan clamped to 1e-6..1, the domain every meter reads.
pub fn to_log_density(image: &Image) -> Image {
    let mut out = vec![0.0f32; image.data.len()];
    analysis::to_log_density(&image.data, &mut out);
    Image::new(image.height, image.width, out)
}

/// The normalization's own log density: clamped below only, as `normalization.wgsl` does.
pub fn log_density_open(image: &Image) -> Image {
    let mut out = vec![0.0f32; image.data.len()];
    analysis::log_density_open(&image.data, &mut out);
    Image::new(image.height, image.width, out)
}

/// `normalize_log_image`: the linear stretch `(v - floor) / (ceil - floor)`, unclamped.
pub fn normalize_log_image(img_log: &Image, bounds: &Bounds) -> Image {
    let mut out = vec![0.0f32; img_log.data.len()];
    analysis::normalize_log_image(&img_log.data, bounds.floors32(), bounds.ceils32(), &mut out);
    Image::new(img_log.height, img_log.width, out)
}

/// `negpy_core.matmul_3x3` over an image.
pub(crate) fn matmul_3x3(img: &Image, m: [[f32; 3]; 3]) -> Image {
    let mut out = vec![0.0f32; img.data.len()];
    negpy_core::color::matmul_3x3(&img.data, &mut out, m);
    Image::new(img.height, img.width, out)
}

/// `get_analysis_crop`: the center crop with `buffer` of each border left out.
fn analysis_crop(img: Cow<'_, Image>, buffer: f64) -> Cow<'_, Image> {
    if buffer <= 0.0 {
        return img;
    }
    let safe = buffer.clamp(0.0, 0.3);
    let (h, w) = img.shape();
    let cut_h = (h as f64 * safe) as usize;
    let cut_w = (w as f64 * safe) as usize;
    Cow::Owned(img.crop((cut_h, h - cut_h, cut_w, w - cut_w)))
}

/// The ROI crop and the analysis buffer every `*_from_log` meter applies first.
fn metered(img: &Image, roi: Option<Roi>, buffer: f64) -> Cow<'_, Image> {
    let cur = match roi {
        Some(r) => Cow::Owned(img.crop(r)),
        None => Cow::Borrowed(img),
    };
    analysis_crop(cur, buffer)
}

/// `_block_median_grid`: the block median onto the fixed analysis grid; the image itself when
/// it already fits.
pub fn block_median_grid(img: Cow<'_, Image>) -> Cow<'_, Image> {
    let grid = EXPOSURE_CONSTANTS.analysis_grid;
    let (h, w) = img.shape();
    let b = (h.max(w) as f64 / grid as f64).ceil() as usize;
    if b <= 1 || h < b || w < b {
        return img;
    }
    let (hb, wb) = ((h / b) * b, (w / b) * b);
    let whole = img.crop((0, hb, 0, wb));
    let data = analysis::block_median_grid(&whole.data, (hb, wb, 3), b);
    Cow::Owned(Image::new(hb / b, wb / b, data))
}

/// `prefilter_log_grid`: log10, crop, buffer, block median, once for every meter.
pub fn prefilter_log_grid(image: &Image, roi: Option<Roi>, buffer: f64) -> Image {
    let log: Cow<'_, Image> = Cow::Owned(to_log_density(image));
    let cur = match roi {
        Some(r) => Cow::Owned(log.crop(r)),
        None => log,
    };
    block_median_grid(analysis_crop(cur, buffer)).into_owned()
}

/// `_rect_roi`: a normalized (x1, y1, x2, y2) rect as a pixel ROI; None when degenerate.
fn rect_roi((h, w): (usize, usize), rect: (f32, f32, f32, f32)) -> Option<Roi> {
    let (x1, y1, x2, y2) = (rect.0 as f64, rect.1 as f64, rect.2 as f64, rect.3 as f64);
    let roi = (
        (y1.min(y2) * h as f64) as usize,
        (y1.max(y2) * h as f64) as usize,
        (x1.min(x2) * w as f64) as usize,
        (x1.max(x2) * w as f64) as usize,
    );
    (roi.1 >= roi.0 + 2 && roi.3 >= roi.2 + 2).then_some(roi)
}

/// `resolve_analysis_region`: the (roi, buffer) the meters read. A freehand rect wins over the
/// crop ROI and the centered buffer; with no crop the meter rect stands in for one.
pub fn resolve_analysis_region(
    shape: (usize, usize),
    active_roi: Option<Roi>,
    analysis_buffer: f64,
    analysis_rect: Option<(f32, f32, f32, f32)>,
    meter_rect: Option<(f32, f32, f32, f32)>,
) -> (Option<Roi>, f64) {
    if let Some(roi) = analysis_rect.and_then(|r| rect_roi(shape, r)) {
        return (Some(roi), 0.0);
    }
    if active_roi.is_none()
        && let Some(rect) = meter_rect
    {
        return (rect_roi(shape, rect), analysis_buffer);
    }
    (active_roi, analysis_buffer)
}

/// `measure_clip_fractions`: per channel, the share of the metered window at or above the
/// sensor-white clip level.
pub fn measure_clip_fractions(image: &Image, roi: Option<Roi>, buffer: f64) -> [f64; 3] {
    let (h, w) = image.shape();
    let (y1, y2, x1, x2) = roi.unwrap_or((0, h, 0, w));
    let (y2, x2) = (y2.min(h), x2.min(w));
    let (mut y1, mut x1) = (y1.min(y2), x1.min(x2));
    let (mut y2, mut x2) = (y2, x2);
    if buffer > 0.0 {
        let b = buffer.clamp(0.0, 0.3);
        let cut_h = ((y2 - y1) as f64 * b) as usize;
        let cut_w = ((x2 - x1) as f64 * b) as usize;
        y1 += cut_h;
        y2 -= cut_h;
        x1 += cut_w;
        x2 -= cut_w;
    }
    let level = EXPOSURE_CONSTANTS.scan_clip_level as f32;
    analysis::clip_fractions(&image.data, w, (y1, y2, x1, x2), level)
}

/// `_meter_norm`: per channel floor and span. The bounds reach the meters as Python floats, so
/// NumPy keeps that step in float32 (NEP 50).
fn meter_norm(bounds: &Bounds) -> [ChannelNorm; 3] {
    std::array::from_fn(|ch| {
        let f = bounds.floors[ch];
        let mut span = bounds.ceils[ch] - f;
        if span.abs() < 1e-6 {
            span = if span >= 0.0 { 1e-6 } else { -1e-6 };
        }
        ChannelNorm {
            floor: f,
            floor_wide: false,
            span,
            span_wide: false,
        }
    })
}

/// `_activity_gate`.
fn activity_gate() -> Activity {
    let c = &EXPOSURE_CONSTANTS;
    Activity {
        block: c.activity_block,
        gate: c.activity_gate_density as f32,
        min_fraction: c.activity_min_fraction,
    }
}

/// The textured luma cells of a grid, normalized by `bounds` when given.
fn textured_luma(grid: &Image, bounds: Option<&Bounds>) -> Vec<f32> {
    let norm = bounds.map(meter_norm);
    let lum = meter::luma(&grid.data, norm.as_ref());
    meter::textured_cells(lum, grid.height, grid.width, activity_gate())
}

/// `measure_anchor_from_log`: the frame's midtone in [0, 1], pulled part way from `assumed`
/// toward the metered value and held within the meter band.
pub fn measure_anchor_from_log(
    img_log: &Image,
    bounds: &Bounds,
    roi: Option<Roi>,
    buffer: f64,
    assumed: Option<f64>,
) -> f64 {
    let c = &EXPOSURE_CONSTANTS;
    let grid = block_median_grid(metered(img_log, roi, buffer));
    let measured = meter::anchor_measure(&textured_luma(&grid, Some(bounds)), c.anchor_trim_clip);
    let a = assumed.unwrap_or(c.assumed_anchor);
    let anchor = a + c.anchor_meter_strength * (measured - a);
    anchor.clamp(a - c.anchor_meter_band, a + c.anchor_meter_band)
}

/// `measure_textural_range_from_log`: the textured luma spread in log density.
pub fn measure_textural_range_from_log(img_log: &Image, roi: Option<Roi>, buffer: f64) -> f64 {
    let grid = block_median_grid(metered(img_log, roi, buffer));
    meter::textural_range(
        &textured_luma(&grid, None),
        EXPOSURE_CONSTANTS.textural_range_clip,
    )
}

/// `measure_shadow_point_from_log`: normalized luma of the textured dark tail.
pub fn measure_shadow_point_from_log(
    img_log: &Image,
    bounds: &Bounds,
    roi: Option<Roi>,
    buffer: f64,
) -> f32 {
    let grid = block_median_grid(metered(img_log, roi, buffer));
    meter::percentile(
        &textured_luma(&grid, Some(bounds)),
        EXPOSURE_CONSTANTS.shadow_reach_percentile,
    )
}

/// `measure_highlight_point_from_log`: normalized luma of the textured bright tail.
pub fn measure_highlight_point_from_log(
    img_log: &Image,
    bounds: &Bounds,
    roi: Option<Roi>,
    buffer: f64,
) -> f32 {
    let grid = block_median_grid(metered(img_log, roi, buffer));
    meter::percentile(
        &textured_luma(&grid, Some(bounds)),
        EXPOSURE_CONSTANTS.highlight_hold_percentile,
    )
}

fn neutral_consts() -> NeutralConsts {
    let c = &EXPOSURE_CONSTANTS;
    let band = |(lo, hi): (f64, f64)| (lo as f32, hi as f32);
    let sector = |chroma: f64, (lo, hi): (f64, f64)| (chroma as f32, lo as f32, hi as f32);
    NeutralConsts {
        quantile: c.neutral_axis_chroma_quantile,
        cap: c.neutral_axis_chroma_cap,
        first_pass_cap: c.neutral_axis_first_pass_cap,
        min_pixels: c.neutral_axis_min_pixels,
        highlight: band(c.neutral_axis_highlight_band),
        mid: band(c.neutral_axis_mid_band),
        shadow: band(c.neutral_axis_shadow_band),
        subjects: [
            sector(c.neutral_axis_foliage_chroma, c.neutral_axis_foliage_hue),
            sector(c.neutral_axis_sky_chroma, c.neutral_axis_sky_hue),
        ],
        n0: c.neutral_axis_confidence_n0,
        agreement: (
            c.neutral_axis_agreement_deadzone,
            c.neutral_axis_agreement_scale,
        ),
        width: (c.neutral_axis_width_deadzone, c.neutral_axis_width_scale),
    }
}

/// `measure_neutral_axis_from_log`: per-channel neutral refs in the highlight, midtone and
/// shadow bands with a confidence; None without a midtone or shadow set.
pub fn measure_neutral_axis_from_log(
    img_log: &Image,
    bounds: &Bounds,
    roi: Option<Roi>,
    buffer: f64,
    exclude_subjects: bool,
) -> Option<Axis> {
    let grid = block_median_grid(metered(img_log, roi, buffer));
    let axis = meter::neutral_axis(
        &grid.data,
        (bounds.floors32(), bounds.ceils32()),
        (bounds.floors, bounds.ceils),
        exclude_subjects,
        &neutral_consts(),
    )?;
    Some(Axis {
        mid: axis.mid,
        shadow: axis.shadow,
        highlight: axis.highlight,
        confidence: axis.confidence,
        offset_weight: None,
    })
}

/// `measure_base_log`: per-channel log10 transmittance of the clear base, read from the film
/// box's margin on the uncropped frame at the crop detector's resolution; None without one.
pub fn measure_base_log(image: &Image) -> Option<[f64; 3]> {
    let (h, w) = image.shape();
    let det_scale = (AUTOCROP_DETECT_RES as f64 / h.max(w) as f64).min(1.0);
    if det_scale >= 1.0 {
        return analysis::measure_base_log(&image.data, h, w);
    }
    let dw = ((w as f64 * det_scale).round_ties_even() as usize).max(1);
    let dh = ((h as f64 * det_scale).round_ties_even() as usize).max(1);
    let det = negpy_core::resize::resize_area(&image.data, h, w, 3, dh, dw)?;
    analysis::measure_base_log(&det, dh, dw)
}

/// `measure_base_log` as the context stores it.
pub fn base_log(image: &Image) -> Option<Rgb> {
    measure_base_log(image).map(|b| (b[0] as f32, b[1] as f32, b[2] as f32))
}

/// `base_scan_density`: the clear base's per-channel scan density after the unmix, the space
/// the scan Parade plots. None without a base.
pub fn base_scan_density(base_log: Option<[f64; 3]>, unmix: Option<&Unmix>) -> Option<[f64; 3]> {
    let base = base_log?.map(|v| v as f32);
    let unmixed = match unmix {
        Some(m) => unmix_pixel(base, &m.map(|row| row.map(|v| v as f32))),
        None => base,
    };
    Some(unmixed.map(|v| -v as f64))
}

/// `per_channel_point_offsets` as Python floats: global plus per-layer trim.
fn per_channel_point_offsets(process: &ProcessConfig) -> ([f64; 3], [f64; 3]) {
    let wp = process.white_point_offset as f64;
    let bp = process.black_point_offset as f64;
    (
        [
            wp + process.white_point_trim_red as f64,
            wp + process.white_point_trim_green as f64,
            wp + process.white_point_trim_blue as f64,
        ],
        [
            bp + process.black_point_trim_red as f64,
            bp + process.black_point_trim_green as f64,
            bp + process.black_point_trim_blue as f64,
        ],
    )
}

/// What `NormalizationProcessor.process` writes into `context.metrics`, in its own precision.
#[derive(Clone, Debug, PartialEq)]
pub struct Normalized {
    pub image: Image,
    /// The analyzed bounds; None when a roll or per-frame lock made the analysis unnecessary.
    pub log_bounds: Option<Bounds>,
    pub log_bounds_base: Bounds,
    pub pre_trim_bounds: Bounds,
    pub final_bounds: Bounds,
    pub norm_density_range: f64,
    pub density_above_base: Option<[f64; 3]>,
    pub neutral_axis_refs: Option<Axis>,
    pub metered_anchor: f64,
    pub textural_range: f64,
    pub shadow_point: f32,
    pub highlight_point: f32,
    pub scan_clip_fractions: [f64; 3],
}

/// `NormalizationProcessor.process`: linear scan to normalized log density, with every meter
/// reading. `base_log` is the clear base the engine read for the source, if any.
pub fn normalize_detailed(
    image: &Image,
    process: &ProcessConfig,
    mode: ProcessMode,
    active_roi: Option<Roi>,
    base_log: Option<[f64; 3]>,
) -> Normalized {
    let img_log = log_density_open(image);
    let (an_roi, an_buffer) = resolve_analysis_region(
        image.shape(),
        active_roi,
        process.analysis_buffer as f64,
        process.analysis_rect,
        process.meter_rect,
    );
    let prefiltered = prefilter_log_grid(image, an_roi, an_buffer);
    let scan_clip_fractions = measure_clip_fractions(image, an_roi, an_buffer);

    let unmix = effective_crosstalk_matrix(process, Some(mode));
    let img_log = unmix_log_image(img_log, unmix.as_ref());
    let prefiltered = unmix_log_image(prefiltered, unmix.as_ref());

    let mut analyzed = None;
    let (bounds, base_bounds) = resolve_bounds_detailed(process, || {
        let b = analyze_log_exposure_bounds_from_log(
            &prefiltered,
            None,
            0.0,
            process.luma_range_clip as f64,
            process.color_range_clip as f64,
        );
        analyzed = Some(b);
        b
    });
    // Cast Removal meters the film's own cast, so the neutral axis reads the pre-trim bounds.
    let pre_trim_bounds = bounds;
    let norm_density_range = luminance_density_range(&bounds);
    let density_above_base =
        density_above_base(base_log, &pre_trim_bounds, &prefiltered, unmix.as_ref());

    let (wp3, bp3) = per_channel_point_offsets(process);
    let bounds = if wp3.iter().chain(&bp3).any(|&v| v != 0.0) {
        Bounds::new(
            std::array::from_fn(|ch| bounds.floors[ch] + wp3[ch]),
            std::array::from_fn(|ch| bounds.ceils[ch] + bp3[ch]),
        )
    } else {
        bounds
    };

    let res = normalize_log_image(&img_log, &bounds);

    let neutral_axis_refs = if mode == ProcessMode::Bw {
        None
    } else {
        let own = measure_neutral_axis_from_log(&prefiltered, &pre_trim_bounds, None, 0.0, true);
        match pooled_neutral_axis(process) {
            Some(pooled) => Some(blend_neutral_axis(own.as_ref(), &pooled)),
            None => own,
        }
    };

    let anchor_bounds = luma_source_bounds(process, &base_bounds);
    Normalized {
        metered_anchor: measure_anchor_from_log(&prefiltered, &anchor_bounds, None, 0.0, None),
        textural_range: measure_textural_range_from_log(&prefiltered, None, 0.0),
        shadow_point: measure_shadow_point_from_log(&prefiltered, &anchor_bounds, None, 0.0),
        highlight_point: measure_highlight_point_from_log(&prefiltered, &anchor_bounds, None, 0.0),
        image: res,
        log_bounds: analyzed,
        log_bounds_base: base_bounds,
        pre_trim_bounds,
        final_bounds: bounds,
        norm_density_range,
        density_above_base,
        neutral_axis_refs,
        scan_clip_fractions,
    }
}

/// The base stage's normalization on the context: `normalize_detailed` with its readings
/// stored in `ctx.metrics`. An analyzed `log_bounds` replaces the stored one; a locked frame
/// keeps it, as the desktop's metrics dict does.
pub fn normalize(img: &Image, cfg: &ProcessConfig, ctx: &mut PipelineContext) -> Image {
    let base_log = ctx
        .metrics
        .base_log
        .map(|b| [b.0 as f64, b.1 as f64, b.2 as f64]);
    let n = normalize_detailed(img, cfg, ctx.process_mode, ctx.active_roi, base_log);
    let m = &mut ctx.metrics;
    if let Some(b) = n.log_bounds {
        m.log_bounds = Some(b.to_metric());
    }
    m.log_bounds_base = Some(n.log_bounds_base.to_metric());
    m.pre_trim_bounds = Some(n.pre_trim_bounds.to_metric());
    m.final_bounds = Some(n.final_bounds.to_metric());
    m.norm_density_range = Some(n.norm_density_range as f32);
    m.neutral_axis_refs = n.neutral_axis_refs.as_ref().map(Axis::to_metric);
    m.metered_anchor = Some(n.metered_anchor as f32);
    m.textural_range = Some(n.textural_range as f32);
    m.shadow_point = Some(n.shadow_point);
    m.highlight_point = Some(n.highlight_point);
    let f = n.scan_clip_fractions;
    m.scan_clip_fractions = Some((f[0] as f32, f[1] as f32, f[2] as f32));
    n.image
}
