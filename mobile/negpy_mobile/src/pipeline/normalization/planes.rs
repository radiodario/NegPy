//! The planes the engine builds from the source between base and exposure: the Contrast
//! Mask's zero-mean blurred luma and the Diffusion's blurred light, both on `analysis_grid`
//! with the frame's geometry replayed (`contrast_mask_plane`, `diffusion_grid`,
//! `diffusion_plane` and `expand_mask_plane` on the desktop).

use std::borrow::Cow;

use negpy_core::{analysis, resize};

use super::{
    Bounds, Unmix, collapse_matrix, effective_crosstalk_matrix, luminance_density_range,
    matmul_3x3, normalize_log_image, plane_weights, prefilter_log_grid, unmix_log_image,
};
use crate::config::{GeometryConfig, PipelineConfig, RenderIntent, RenderPath};
use crate::context::{GeometryParams, PipelineContext};
use crate::image::{Image, Plane, Roi};
use crate::pipeline::constants::EXPOSURE_CONSTANTS;
use crate::pipeline::geometry;

/// Contrast Mask blur sigma, per-cent of the analysis grid's short side.
const MASK_BLUR_PERCENT: f64 = 4.0;
/// Diffusion's spread: (weight, sigma in per-cent of the frame's short side) Gaussians.
const DIFFUSION_PSF: [(f64, f64); 5] = [
    (0.0421, 0.213),
    (0.1576, 0.449),
    (0.2723, 0.926),
    (0.3540, 2.017),
    (0.1740, 5.48),
];

/// (y1, y2, x1, x2) as fractions of a frame.
pub type RoiNorm = (f64, f64, f64, f64);

/// `normalized_roi`: a pixel ROI as fractions of `shape`; None for none and for the full frame.
pub fn normalized_roi(roi: Option<Roi>, (h, w): (usize, usize)) -> Option<RoiNorm> {
    let (y1, y2, x1, x2) = roi?;
    if (y1, x1, y2, x2) == (0, 0, h, w) {
        return None;
    }
    let (hf, wf) = (h as f64, w as f64);
    Some((
        y1 as f64 / hf,
        y2 as f64 / hf,
        x1 as f64 / wf,
        x2 as f64 / wf,
    ))
}

/// `geometry_kwargs`: the geometry a plane replays, from the config as the engine passes it.
pub fn geometry_kwargs(cfg: &GeometryConfig) -> GeometryParams {
    geometry::geometry_params(cfg)
}

/// `analysis_grid`: the frame before geometry, downsampled to the analysis grid, the geometry
/// replayed and cropped to `roi_norm`.
pub fn analysis_grid(image: &Image, g: &GeometryParams, roi_norm: Option<RoiNorm>) -> Image {
    let (h, w) = image.shape();
    let grid = EXPOSURE_CONSTANTS.analysis_grid;
    let mut cur = Cow::Borrowed(image);
    if h.max(w) > grid {
        let scale = grid as f64 / h.max(w) as f64;
        let dw = ((w as f64 * scale).round_ties_even() as usize).max(1);
        let dh = ((h as f64 * scale).round_ties_even() as usize).max(1);
        let data = resize::resize_area(&image.data, h, w, 3, dh, dw).expect("a non-empty frame");
        cur = Cow::Owned(Image::new(dh, dw, data));
    }
    let warped = geometry::apply_geometry(&cur, g);
    let Some((ny1, ny2, nx1, nx2)) = roi_norm else {
        return warped;
    };
    let (gh, gw) = warped.shape();
    let at = |f: f64, n: usize| (f * n as f64).round_ties_even() as i64;
    let y1 = at(ny1, gh).clamp(0, gh as i64 - 1) as usize;
    let y2 = (at(ny2, gh).min(gh as i64) as usize).max(y1 + 1);
    let x1 = at(nx1, gw).clamp(0, gw as i64 - 1) as usize;
    let x2 = (at(nx2, gw).min(gw as i64) as usize).max(x1 + 1);
    warped.crop((y1, y2, x1, x2))
}

/// `contrast_mask_plane`: the zero-mean blurred luma plane on `analysis_grid` and the val it
/// was centred on. `bw_weights` collapses the val first, as a B&W print does the pixel. None
/// for an empty grid.
pub fn contrast_mask_plane(
    image: &Image,
    bounds: &Bounds,
    unmix: Option<&Unmix>,
    g: &GeometryParams,
    roi_norm: Option<RoiNorm>,
    bw_weights: Option<[f64; 3]>,
) -> Option<(Plane, f32)> {
    let image = analysis_grid(image, g, roi_norm);
    let (h, w) = image.shape();
    // A frame that never metered normalizes to huge values; no stretch, no mask.
    if luminance_density_range(bounds) < 1e-6 {
        return Some((Plane::new(h, w, vec![0.0; h * w]), 0.5));
    }
    let mut val = normalize_log_image(
        &unmix_log_image(prefilter_log_grid(&image, None, 0.0), unmix),
        bounds,
    );
    if let Some(wts) = bw_weights {
        val = matmul_3x3(&val, collapse_matrix(wts));
    }
    let (vh, vw) = val.shape();
    let sigma = MASK_BLUR_PERCENT * 0.01 * vh.min(vw) as f64;
    let (plane, centre) = analysis::contrast_mask_tail(&val.data, vh, vw, sigma)?;
    Some((Plane::new(vh, vw, plane), centre))
}

/// `expand_mask_plane`: the analysis-grid plane at render size, landing on `roi` with the
/// edges replicated outside it. None for an empty ROI.
pub fn expand_mask_plane(plane: &Plane, (h, w): (usize, usize), roi: Option<Roi>) -> Option<Plane> {
    let (y1, y2, x1, x2) = roi.unwrap_or((0, h, 0, w));
    let (y2, x2) = (y2.min(h), x2.min(w));
    if y2 < y1 + 1 || x2 < x1 + 1 {
        return None;
    }
    let mut out = vec![0.0f32; h * w];
    resize::expand_plane(
        &plane.data,
        (plane.height, plane.width),
        (h, w),
        (y1, y2, x1, x2),
        1.0,
        &mut out,
    )?;
    Some(Plane::new(h, w, out))
}

/// `diffusion_grid`: the unmixed light blurred by the diffusion spread, as log density on
/// `analysis_grid`. None for an empty grid.
pub fn diffusion_grid(
    image: &Image,
    unmix: Option<&Unmix>,
    g: &GeometryParams,
    roi_norm: Option<RoiNorm>,
) -> Option<Image> {
    let image = analysis_grid(image, g, roi_norm);
    let unmixed = unmix_log_image(prefilter_log_grid(&image, None, 0.0), unmix);
    let (h, w) = unmixed.shape();
    let short = 0.01 * h.min(w) as f64;
    let psf: Vec<(f32, f64)> = DIFFUSION_PSF
        .iter()
        .map(|&(wt, s)| (wt as f32, s * short))
        .collect();
    let mut out = vec![0.0f32; unmixed.data.len()];
    analysis::diffusion_tail(&unmixed.data, h, w, &psf, &mut out)?;
    Some(Image::new(h, w, out))
}

/// `diffusion_plane`: `diffusion_grid` in the print's normalization, collapsed by `bw_weights`
/// for a B&W print. None when the frame never metered.
pub fn diffusion_plane(
    grid: &Image,
    bounds: &Bounds,
    bw_weights: Option<[f64; 3]>,
) -> Option<Image> {
    if luminance_density_range(bounds) < 1e-6 {
        return None;
    }
    let val = normalize_log_image(grid, bounds);
    Some(match bw_weights {
        Some(wts) => matmul_3x3(&val, collapse_matrix(wts)),
        None => val,
    })
}

/// Both planes are print-curve inputs: the transfer curve and the Flat master take none.
fn takes_planes(settings: &PipelineConfig) -> bool {
    settings.process.render_path() == RenderPath::Print
        && settings.exposure.render_intent != RenderIntent::Flat
}

/// The engine's Contrast Mask block after base: from the source `img` and the context's
/// pre-trim bounds, the plane expanded to `render_shape` (the base output's size) into
/// `metrics.contrast_mask_plane`, its centre and a cleared `contrast_mask_roi`.
pub fn build_contrast_mask(
    img: &Image,
    settings: &PipelineConfig,
    ctx: &mut PipelineContext,
    render_shape: (usize, usize),
) {
    let mask_bounds = ctx.metrics.pre_trim_bounds.or(ctx.metrics.final_bounds);
    let Some(mask_bounds) = mask_bounds else {
        return;
    };
    if !(settings.exposure.contrast_mask_strength > 0.0 && takes_planes(settings)) {
        return;
    }
    let roi = ctx.active_roi;
    let unmix = effective_crosstalk_matrix(&settings.process, Some(settings.process.process_mode));
    let built = contrast_mask_plane(
        img,
        &Bounds::from_metric(&mask_bounds),
        unmix.as_ref(),
        &geometry_kwargs(&settings.geometry),
        normalized_roi(roi, render_shape),
        plane_weights(&settings.process, &settings.exposure),
    );
    let Some((plane, centre)) = built else {
        return;
    };
    if let Some(expanded) = expand_mask_plane(&plane, render_shape, roi) {
        ctx.metrics.contrast_mask_plane = Some(expanded);
        ctx.metrics.contrast_mask_centre = f64::from(centre);
        ctx.metrics.contrast_mask_roi = None;
    }
}

/// The engine's Diffusion block after base: the grid in the final bounds' normalization into
/// `metrics.diffusion_plane`, with the crop ROI it covers in `diffusion_roi`.
pub fn build_diffusion(
    img: &Image,
    settings: &PipelineConfig,
    ctx: &mut PipelineContext,
    render_shape: (usize, usize),
) {
    let Some(diff_bounds) = ctx.metrics.final_bounds else {
        return;
    };
    if !(settings.exposure.diffusion > 0.0 && takes_planes(settings)) {
        return;
    }
    let roi = ctx.active_roi;
    let unmix = effective_crosstalk_matrix(&settings.process, Some(settings.process.process_mode));
    let grid = diffusion_grid(
        img,
        unmix.as_ref(),
        &geometry_kwargs(&settings.geometry),
        normalized_roi(roi, render_shape),
    );
    let plane = grid.and_then(|g| {
        diffusion_plane(
            &g,
            &Bounds::from_metric(&diff_bounds),
            plane_weights(&settings.process, &settings.exposure),
        )
    });
    if let Some(plane) = plane {
        ctx.metrics.diffusion_plane = Some(plane);
        ctx.metrics.diffusion_roi = roi;
    }
}
