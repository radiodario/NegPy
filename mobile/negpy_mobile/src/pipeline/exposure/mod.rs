//! The exposure stage (`DarkroomEngine.run_exposure`): the print's H&D curve, or the Flat
//! master, with the dodge/burn maps, the Contrast Mask, Diffusion and the B&W collapse, then
//! Hue Trim (`features/exposure/processor.py::PhotometricProcessor`). The transparency
//! transfer (an E-6 print) is not this stage's: the caller routes a slide elsewhere.
//!
//! The scalar inputs (`ExposureInputs`) are the desktop's Python floats, at f64.

pub mod bw_filter;
pub mod curve;
pub mod hue;
pub mod local;
pub mod placement;

use crate::config::{ExposureConfig, LocalMask, PipelineConfig, ProcessMode, RenderIntent};
use crate::context::{Metrics, PipelineContext};
use crate::image::Image;
use crate::pipeline::constants::EXPOSURE_CONSTANTS;
use crate::pipeline::papers::{UnsupportedStock, effective_paper_profile, resolve_print_color};

pub use bw_filter::{collapse_matrix, collapse_weights, plane_weights, printed_exposure};
pub use curve::*;
pub use hue::apply_hue_trim;
pub use local::{LocalMaps, compute_local_maps};
pub use placement::limited_mask_params;

/// The base stage's scalar metrics the curve is solved from.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct ExposureMetrics {
    pub final_bounds: Option<Bounds64>,
    pub log_bounds: Option<Bounds64>,
    pub norm_density_range: Option<f64>,
    pub metered_anchor: Option<f64>,
    pub textural_range: Option<f64>,
    pub shadow_point: Option<f64>,
    pub highlight_point: Option<f64>,
    pub neutral_axis_refs: Option<NeutralAxis64>,
    pub contrast_mask_centre: f64,
}

impl From<&Metrics> for ExposureMetrics {
    fn from(m: &Metrics) -> Self {
        Self {
            final_bounds: m.final_bounds.map(Into::into),
            log_bounds: m.log_bounds.map(Into::into),
            norm_density_range: m.norm_density_range,
            metered_anchor: m.metered_anchor,
            textural_range: m.textural_range,
            shadow_point: m.shadow_point,
            highlight_point: m.highlight_point,
            neutral_axis_refs: m.neutral_axis_refs.as_ref().map(Into::into),
            contrast_mask_centre: m.contrast_mask_centre,
        }
    }
}

/// Everything the stage reads beside the pixels and the context's planes.
#[derive(Clone, Debug, PartialEq)]
pub struct ExposureInputs {
    pub exposure: ExposureConfig,
    pub process_mode: ProcessMode,
    pub source_color: Option<bool>,
    pub hue_trim: f64,
    pub masks: Vec<LocalMask>,
    pub metrics: ExposureMetrics,
}

impl ExposureInputs {
    pub fn new(cfg: &PipelineConfig, ctx: &PipelineContext) -> Self {
        Self {
            exposure: cfg.exposure.clone(),
            process_mode: cfg.process.process_mode,
            source_color: cfg.process.source_color,
            hue_trim: cfg.process.hue_trim,
            masks: cfg.local.masks.clone(),
            metrics: (&ctx.metrics).into(),
        }
    }
}

/// `run_exposure` for the print path and the Flat intent; sets `ctx.metrics.print_slopes`.
pub fn exposure(
    img: &Image,
    cfg: &PipelineConfig,
    ctx: &mut PipelineContext,
) -> Result<Image, UnsupportedStock> {
    let inputs = ExposureInputs::new(cfg, ctx);
    exposure_with(img, &inputs, ctx)
}

/// `exposure` from its scalar inputs.
pub fn exposure_with(
    img: &Image,
    inputs: &ExposureInputs,
    ctx: &mut PipelineContext,
) -> Result<Image, UnsupportedStock> {
    let ex = printed_exposure(inputs.process_mode, &inputs.exposure);
    let weights = collapse_weights(
        inputs.process_mode,
        inputs.source_color,
        inputs.exposure.bw_filter,
    );
    let out = if ex.render_intent == RenderIntent::Flat {
        process_flat(img, &ex, weights, ctx)?
    } else {
        process_print(img, &ex, weights, &inputs.masks, &inputs.metrics, ctx)?
    };
    Ok(apply_hue_trim(&out, inputs.hue_trim))
}

fn collapse(img: &Image, weights: (f64, f64, f64)) -> Image {
    let mut out = vec![0.0f32; img.data.len()];
    negpy_core::color::matmul_3x3(&img.data, &mut out, collapse_matrix(weights));
    Image::new(img.height, img.width, out)
}

/// `gray_to_rgb(get_luminance(img))`.
fn luminance_rgb(img: &Image) -> Image {
    let mut lum = vec![0.0f32; img.height * img.width];
    negpy_core::analysis::luminance(&img.data, &mut lum);
    let mut out = vec![0.0f32; img.data.len()];
    negpy_core::convert::gray_to_rgb(&lum, &mut out);
    Image::new(img.height, img.width, out)
}

fn process_print(
    img: &Image,
    ex: &ExposureConfig,
    weights: (f64, f64, f64),
    masks: &[LocalMask],
    m: &ExposureMetrics,
    ctx: &mut PipelineContext,
) -> Result<Image, UnsupportedStock> {
    let mode = ctx.process_mode;
    let paper = effective_paper_profile(&ex.paper_profile, mode);
    let d_min = if ex.paper_dmin { paper.d_min } else { 0.0 };
    let anchor = if ex.auto_exposure {
        m.metered_anchor
    } else {
        None
    };
    let lum_range = m.norm_density_range;
    let final_bounds = m.final_bounds;
    let (strength, neutral_axis_norm) = cast_solve_inputs(
        final_bounds.as_ref(),
        m.neutral_axis_refs.as_ref(),
        ex.cast_removal_strength,
    );
    let mg = effective_midtone_gamma(Some(paper), ex.midtone_gamma);
    let params = per_channel_curve_params(
        ex.grade,
        ex.density,
        ex.auto_normalize_contrast,
        strength,
        lum_range,
        m.textural_range,
        d_min,
        anchor,
        Some(paper),
        neutral_axis_norm.as_ref(),
        [ex.grade_trim_red, ex.grade_trim_green, ex.grade_trim_blue],
        m.shadow_point,
        Some(mg),
    );
    let (slopes, pivots) = (params.slopes, params.pivots);
    ctx.metrics.print_slopes = Some((slopes[0], slopes[1], slopes[2]));
    let hl_auto = match m.highlight_point {
        Some(point) if ex.auto_normalize_contrast => highlight_hold_offset(
            slopes[1],
            pivots[1],
            point,
            d_min,
            Some(paper),
            ex.preflash,
            ex.grade,
            Some(mg),
        ),
        _ => 0.0,
    };
    let sh_auto = match m.shadow_point {
        Some(point) if ex.auto_normalize_contrast => {
            shadow_hold_offset(slopes[1], pivots[1], point, d_min, Some(paper), Some(mg))
        }
        _ => 0.0,
    };

    let cmy_max = EXPOSURE_CONSTANTS.cmy_max_density;
    let cmy_offsets = filtration_offsets(
        [ex.wb_cyan, ex.wb_magenta, ex.wb_yellow],
        final_bounds.as_ref(),
    );
    let shadow_cmy = [
        ex.shadow_cyan * cmy_max,
        ex.shadow_magenta * cmy_max,
        ex.shadow_yellow * cmy_max,
    ];
    let highlight_cmy = [
        ex.highlight_cyan * cmy_max,
        ex.highlight_magenta * cmy_max,
        ex.highlight_yellow * cmy_max,
    ];
    let (toe_eff, shoulder_eff) = grade_coupled_shape(slopes[1], ex.toe, ex.shoulder);
    let (sg_deltas, hg_deltas) = split_grade_deltas(
        ex.grade,
        ex.shadow_grade,
        ex.highlight_grade,
        [
            ex.shadow_grade_trim_red,
            ex.shadow_grade_trim_green,
            ex.shadow_grade_trim_blue,
        ],
        [
            ex.highlight_grade_trim_red,
            ex.highlight_grade_trim_green,
            ex.highlight_grade_trim_blue,
        ],
    );

    let is_bw = mode == ProcessMode::Bw;
    let collapsed;
    let image = if is_bw {
        collapsed = collapse(img, weights);
        &collapsed
    } else {
        img
    };
    let (h, w) = image.shape();

    let local_maps = (!masks.is_empty())
        .then(|| compute_local_maps(masks, h, w, ctx.original_size, &ctx.metrics.geometry));
    let mut ev_map = local_maps.as_ref().map(|maps| maps.plane(0));
    let mask_coeffs = contrast_mask_coeffs(
        ex.contrast_mask_strength,
        m.contrast_mask_centre,
        slopes,
        pivots,
        cmy_offsets,
        lum_range.unwrap_or(0.0),
    );
    let mask_ev = contrast_mask_ev(
        ctx.metrics.contrast_mask_plane.as_ref(),
        mask_coeffs,
        (h, w),
        ctx.metrics.contrast_mask_roi,
    );
    if let Some(mask_ev) = mask_ev {
        ev_map = Some(match ev_map {
            None => mask_ev.data,
            Some(ev) => {
                let mut sum = vec![0.0f32; ev.len()];
                negpy_core::convert::add(&ev, &mask_ev.data, &mut sum);
                sum
            }
        });
    }
    let grade_map = local_maps
        .as_ref()
        .filter(|maps| maps.channel_any(1))
        .map(|maps| local_grade_factor_map(&maps.plane(1), ex.grade));
    let flash_map = local_maps
        .as_ref()
        .filter(|maps| maps.channel_any(2))
        .map(|maps| maps.plane(2));
    let diffusion_plane = ctx
        .metrics
        .diffusion_plane
        .as_ref()
        .filter(|_| ex.diffusion > 0.0);
    let diffusion_rect = diffusion_plane.map(|_| {
        let (y1, y2, x1, x2) = ctx.metrics.diffusion_roi.unwrap_or((0, h, 0, w));
        [x1 as f64, y1 as f64, (x2 - x1) as f64, (y2 - y1) as f64]
    });
    let keyed = local_maps.as_ref().filter(|maps| maps.planes > 3);
    let key_alpha = keyed.map(|maps| maps.key_alphas());
    let key_params = keyed.and_then(|_| limited_mask_params(masks, ex, mode, m));
    let grade_deltas = keyed.map(|maps| maps.plane(1));

    let args = CurveArgs {
        params,
        toe: toe_eff,
        toe_width: ex.toe_width,
        shoulder: shoulder_eff,
        shoulder_width: ex.shoulder_width,
        shadow_cmy,
        highlight_cmy,
        cmy_offsets,
        d_min,
        midtone_gamma: Some(mg),
        paper: Some(paper),
        ev_map: ev_map.as_deref(),
        ev_scale: local_ev_scale(final_bounds.as_ref()),
        grade_map: grade_map.as_deref(),
        bpc: !ex.paper_black,
        toe_trims: [ex.toe_trim_red, ex.toe_trim_green, ex.toe_trim_blue],
        shoulder_trims: [
            ex.shoulder_trim_red,
            ex.shoulder_trim_green,
            ex.shoulder_trim_blue,
        ],
        snap_trims: [
            ex.midtone_gamma_trim_red,
            ex.midtone_gamma_trim_green,
            ex.midtone_gamma_trim_blue,
        ],
        toe_width_trims: [
            ex.toe_width_trim_red,
            ex.toe_width_trim_green,
            ex.toe_width_trim_blue,
        ],
        shoulder_width_trims: [
            ex.shoulder_width_trim_red,
            ex.shoulder_width_trim_green,
            ex.shoulder_width_trim_blue,
        ],
        shadow_density: ex.shadow_density + sh_auto,
        highlight_density: ex.highlight_density + hl_auto,
        shadow_grade_deltas: sg_deltas,
        highlight_grade_deltas: hg_deltas,
        // B&W collapsed to one density above, so the separation matrix is a no-op there.
        dye_separation: if is_bw { 1.0 } else { ex.dye_separation },
        dye_separation_trims: if is_bw {
            [0.0; 3]
        } else {
            [
                ex.dye_separation_trim_red,
                ex.dye_separation_trim_green,
                ex.dye_separation_trim_blue,
            ]
        },
        separation_damping: if is_bw { 0.0 } else { ex.separation_damping },
        key_alpha: key_alpha.as_deref(),
        key_params: key_params.as_deref(),
        grade_deltas: grade_deltas.as_deref(),
        frame_grade: ex.grade,
        preflash: ex.preflash,
        flash_map: flash_map.as_deref(),
        diffusion: if diffusion_plane.is_some() {
            ex.diffusion
        } else {
            0.0
        },
        diffusion_plane,
        diffusion_rect,
        diffusion_range: if diffusion_plane.is_some() {
            channel_density_ranges(final_bounds.as_ref())
        } else {
            [1.0; 3]
        },
        black_hold: black_hold_taper(ex.hold_black, Some(paper)),
    };
    let img_pos = apply_characteristic_curve(image, &args)?;
    Ok(if is_bw {
        luminance_rgb(&img_pos)
    } else {
        img_pos
    })
}

/// The flat log master: the normalized log signal with the manual WB and the stock's matrix.
fn process_flat(
    img: &Image,
    ex: &ExposureConfig,
    weights: (f64, f64, f64),
    ctx: &PipelineContext,
) -> Result<Image, UnsupportedStock> {
    let (gain, lift) = flat_curve_params();
    let final_bounds = ctx.metrics.final_bounds.map(Bounds64::from);
    let cmy_offsets = filtration_offsets(
        [ex.wb_cyan, ex.wb_magenta, ex.wb_yellow],
        final_bounds.as_ref(),
    );
    let is_bw = ctx.process_mode == ProcessMode::Bw;
    let collapsed;
    let image = if is_bw {
        collapsed = collapse(img, weights);
        &collapsed
    } else {
        img
    };
    let stock_matrix = if is_bw {
        None
    } else {
        resolve_print_color(Some(effective_paper_profile(
            &ex.paper_profile,
            ctx.process_mode,
        )))?
    };
    let img_pos = apply_flat_curve(image, gain, lift, cmy_offsets, stock_matrix);
    Ok(if is_bw {
        luminance_rgb(&img_pos)
    } else {
        img_pos
    })
}
