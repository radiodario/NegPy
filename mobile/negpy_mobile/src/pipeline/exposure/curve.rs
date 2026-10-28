//! The curve glue of `features/exposure/logic.py`: the straight line, Auto Density and Auto
//! Grade, Cast Removal's per-channel solve, the Contrast Mask's stops and the print kernel's
//! argument build. Scalars are f64, as the desktop's Python floats; `libm` gives one result on
//! every CPU.

use negpy_core::print_curve::{self, Diffusion, Keys, Plane as CorePlane, PrintCurve};

use crate::context::{LogNegativeBounds, NeutralAxis};
use crate::image::{Image, Plane, Roi};
use crate::pipeline::constants::{EXPOSURE_CONSTANTS, ExposureConstants};
use crate::pipeline::papers::{
    PaperProfile, Rows3, UnsupportedStock, compose_density_matrices, effective_constants,
    resolve_print_color, resolve_saturation_matrix, slope_gain,
};

pub type Rgb64 = [f64; 3];

/// `LogNegativeBounds` at the desktop's float64.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Bounds64 {
    pub floors: Rgb64,
    pub ceils: Rgb64,
}

pub fn rgb64(v: (f32, f32, f32)) -> Rgb64 {
    [v.0 as f64, v.1 as f64, v.2 as f64]
}

impl From<LogNegativeBounds> for Bounds64 {
    fn from(b: LogNegativeBounds) -> Self {
        Self {
            floors: rgb64(b.floors),
            ceils: rgb64(b.ceils),
        }
    }
}

/// `metrics["neutral_axis_refs"]` at float64.
#[derive(Clone, Debug, PartialEq)]
pub struct NeutralAxis64 {
    pub mid: Rgb64,
    pub shadow: Rgb64,
    pub highlight: Option<Rgb64>,
    pub confidence: f64,
}

impl From<&NeutralAxis> for NeutralAxis64 {
    fn from(a: &NeutralAxis) -> Self {
        Self {
            mid: rgb64(a.mid),
            shadow: rgb64(a.shadow),
            highlight: a.highlight.map(rgb64),
            confidence: a.confidence as f64,
        }
    }
}

/// (midtone, shadow, highlight) neutral refs normalized against the bounds.
pub type NeutralAxisNorm = (Rgb64, Rgb64, Option<Rgb64>);

/// Per-channel (slopes, pivots, curvatures): `v = slope * (u - pivot) + curv * u^2`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CurveParams {
    pub slopes: Rgb64,
    pub pivots: Rgb64,
    pub curvatures: Rgb64,
}

/// Python's `min(max(v, lo), hi)`.
fn clamp(v: f64, lo: f64, hi: f64) -> f64 {
    v.max(lo).min(hi)
}

/// `_fast_sigmoid`: the logistic sigmoid, overflow-safe.
pub fn fast_sigmoid(x: f64) -> f64 {
    if x >= 0.0 {
        1.0 / (1.0 + libm::exp(-x))
    } else {
        let z = libm::exp(x);
        z / (1.0 + z)
    }
}

/// `_inv_softplus_np`: log(exp(y) - 1), stable for y > 0.
pub fn inv_softplus(y: f64) -> f64 {
    if y > 20.0 {
        y
    } else {
        libm::log(libm::expm1(y.max(1e-12)))
    }
}

/// Fallback density range when none is measured: a normal negative's, scaled by K.
pub fn default_grade_range() -> f64 {
    let c = &EXPOSURE_CONSTANTS;
    c.auto_grade_target * c.auto_grade_nominal_ratio * c.auto_grade_nominal_range
}

/// Straight-line slope k from the grade as an ISO R paper exposure range.
pub fn grade_to_slope(grade: f64, density_range: Option<f64>, gain: f64) -> f64 {
    let c = &EXPOSURE_CONSTANTS;
    let rng_in = density_range.unwrap_or_else(default_grade_range);
    let er = clamp(grade, c.iso_r_min, c.iso_r_max) / 100.0;
    let rng = clamp(rng_in.abs(), 0.3, 3.5);
    let k = c.grade_contrast_scale * rng / er * gain;
    clamp(k, c.slope_min, c.slope_max)
}

/// Inverse of `grade_to_slope`, clamped to the slider's R range.
pub fn slope_to_grade(slope: f64, density_range: Option<f64>, gain: f64) -> f64 {
    let c = &EXPOSURE_CONSTANTS;
    let rng_in = density_range.unwrap_or_else(default_grade_range);
    let rng = clamp(rng_in.abs(), 0.3, 3.5);
    if slope <= 0.0 {
        return c.iso_r_max;
    }
    let er = c.grade_contrast_scale * rng * gain / slope;
    clamp(er * 100.0, c.iso_r_min, c.iso_r_max)
}

/// Range fed to `grade_to_slope`: the measured one, or Auto Grade's textural normalization.
pub fn effective_grade_range(
    auto_normalize_contrast: bool,
    floor_ceil_range: Option<f64>,
    textural_range: Option<f64>,
) -> Option<f64> {
    let c = &EXPOSURE_CONSTANTS;
    if !auto_normalize_contrast {
        return floor_ceil_range;
    }
    let (Some(textural), Some(floor_ceil)) = (textural_range, floor_ceil_range) else {
        return Some(default_grade_range());
    };
    let measured = textural.abs();
    if measured < 1e-6 {
        return Some(3.5);
    }
    let k = c.auto_grade_target;
    let q = c.auto_grade_nominal_range / measured;
    let strength = c.auto_grade_strength;
    let factor = ((1.0 - strength) + strength * q).min(q * c.auto_grade_max_overfill);
    Some(k * floor_ceil.abs() * factor)
}

/// Straight-line value v* that the base knees map onto `target` (reference_print_density).
pub fn reference_linear_value(
    d_min: f64,
    paper: Option<&PaperProfile>,
    target: Option<f64>,
) -> f64 {
    let c = effective_constants(paper);
    let t = target.unwrap_or(c.reference_print_density);
    let d_max = c.d_max;
    let a_hl = c.shoulder_sharpness_base;
    let a_sh = c.toe_sharpness_base;
    let v1 = d_max - inv_softplus(a_sh * (d_max - t)) / a_sh;
    d_min + inv_softplus(a_hl * (v1 - d_min)) / a_hl
}

/// (midtone gamma, its width, v*).
pub fn midtone_terms(
    d_min: f64,
    paper: Option<&PaperProfile>,
    midtone_gamma: Option<f64>,
) -> (f64, f64, f64) {
    let c = effective_constants(paper);
    let mg = midtone_gamma.unwrap_or(c.paper_midtone_gamma);
    (
        mg,
        c.paper_gamma_width,
        reference_linear_value(d_min, paper, None),
    )
}

/// Straight-line value `v` after the variable midtone gamma, as the kernels apply it.
pub fn through_midtone(
    v: f64,
    d_min: f64,
    paper: Option<&PaperProfile>,
    midtone_gamma: Option<f64>,
) -> f64 {
    let (mg, w, v_star) = midtone_terms(d_min, paper, midtone_gamma);
    v + mg * w * libm::tanh((v - v_star) / w)
}

/// Straight-line value that prints `target` through the midtone gamma: bisection.
pub fn straight_line_target(
    d_min: f64,
    paper: Option<&PaperProfile>,
    target: Option<f64>,
    midtone_gamma: Option<f64>,
) -> f64 {
    let u = reference_linear_value(d_min, paper, target);
    let (mg, w, v_star) = midtone_terms(d_min, paper, midtone_gamma);
    let (mut lo, mut hi) = (u - mg.abs() * w, u + mg.abs() * w);
    for _ in 0..60 {
        let mid = 0.5 * (lo + hi);
        if mid + mg * w * libm::tanh((mid - v_star) / w) < u {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    0.5 * (lo + hi)
}

/// Straight-line value the anchor prints from.
pub fn anchor_linear_value(
    d_min: f64,
    paper: Option<&PaperProfile>,
    metered: bool,
    midtone_gamma: Option<f64>,
) -> f64 {
    if !metered {
        return reference_linear_value(d_min, paper, None);
    }
    let target = effective_constants(paper).anchor_target_density;
    straight_line_target(d_min, paper, Some(target), midtone_gamma)
}

/// (v_th, gamma) for `preflash_value`.
pub fn preflash_params(
    grade: f64,
    d_min: f64,
    paper: Option<&PaperProfile>,
    midtone_gamma: Option<f64>,
) -> (f64, f64) {
    let c = effective_constants(paper);
    let e = &EXPOSURE_CONSTANTS;
    let v_th = straight_line_target(
        d_min,
        paper,
        Some(d_min + c.preflash_threshold_density),
        midtone_gamma,
    );
    let er = clamp(grade, e.iso_r_min, e.iso_r_max) / 100.0;
    (v_th, e.grade_contrast_scale / er * slope_gain(paper))
}

/// Straight-line value `v` after a uniform flash of `flash` times the threshold exposure.
pub fn preflash_value(v: f64, v_th: f64, flash: f64, gamma: f64) -> f64 {
    v_th + gamma * libm::log10(libm::pow(10.0, (v - v_th) / gamma) + flash)
}

/// Auto Grade's floor on the slope: the textured dark tail prints at least shadow_reach_density.
#[allow(clippy::too_many_arguments)]
pub fn shadow_reach_slope(
    slope: f64,
    anchor: f64,
    shadow_point: f64,
    d_min: f64,
    paper: Option<&PaperProfile>,
    midtone_gamma: Option<f64>,
    metered: bool,
) -> f64 {
    let c = effective_constants(paper);
    let span = shadow_point - anchor;
    if span <= 1e-6 {
        return slope;
    }
    let v_black = straight_line_target(d_min, paper, Some(c.shadow_reach_density), midtone_gamma);
    let needed = (v_black - anchor_linear_value(d_min, paper, metered, midtone_gamma)) / span;
    clamp(slope, needed, c.slope_max)
}

/// Auto Grade's highlight burn: a highlight_density term; 0 when the tone holds.
#[allow(clippy::too_many_arguments)]
pub fn highlight_hold_offset(
    slope: f64,
    pivot: f64,
    highlight_point: f64,
    d_min: f64,
    paper: Option<&PaperProfile>,
    preflash: f64,
    grade: f64,
    midtone_gamma: Option<f64>,
) -> f64 {
    let c = effective_constants(paper);
    let target = c.highlight_hold_density;
    if target <= 0.0 {
        return 0.0;
    }
    let mut v = slope * (highlight_point - pivot);
    if preflash > 0.0 {
        let (v_th, gamma) = preflash_params(grade, d_min, paper, midtone_gamma);
        v = preflash_value(v, v_th, preflash, gamma);
    }
    let v = through_midtone(v, d_min, paper, midtone_gamma);
    let v_hold = reference_linear_value(d_min, paper, Some(target));
    if v >= v_hold {
        return 0.0;
    }
    let z_hi = c.reference_print_density + c.zone_density_highlight_offset;
    let w = 1.0 - fast_sigmoid(c.zone_density_sharpness * (v - z_hi));
    ((v_hold - v) / w.max(1e-6)).min(c.highlight_hold_max)
}

/// Auto Grade's shadow lift (a negative shadow_density); 0 when the tail holds.
pub fn shadow_hold_offset(
    slope: f64,
    pivot: f64,
    shadow_point: f64,
    d_min: f64,
    paper: Option<&PaperProfile>,
    midtone_gamma: Option<f64>,
) -> f64 {
    let c = effective_constants(paper);
    let strength = c.shadow_hold_strength;
    if strength <= 0.0 {
        return 0.0;
    }
    let v = through_midtone(slope * (shadow_point - pivot), d_min, paper, midtone_gamma);
    let v_start = reference_linear_value(d_min, paper, Some(c.shadow_reach_density))
        + c.shadow_hold_overshoot;
    if v <= v_start {
        return 0.0;
    }
    -(strength * (v - v_start)).min(c.shadow_hold_max)
}

/// The pivot that prints the anchor at its target for `slope`, offset by the density slider.
pub fn compute_pivot(
    slope: f64,
    density: f64,
    d_min: f64,
    anchor: Option<f64>,
    paper: Option<&PaperProfile>,
    midtone_gamma: Option<f64>,
) -> f64 {
    let c = effective_constants(paper);
    let r = anchor.unwrap_or(c.assumed_anchor);
    let v = anchor_linear_value(d_min, paper, anchor.is_some(), midtone_gamma);
    let base = r - v / slope;
    base + (1.0 - density) * c.density_multiplier
}

/// Per-channel reference densities to their normalized position in the floor-ceil stretch.
pub fn normalize_refs(refs: Rgb64, floors: Rgb64, ceils: Rgb64) -> Rgb64 {
    let epsilon = 1e-6;
    std::array::from_fn(|ch| {
        let mut denom = ceils[ch] - floors[ch];
        if denom.abs() < epsilon {
            denom = if denom >= 0.0 { epsilon } else { -epsilon };
        }
        (refs[ch] - floors[ch]) / denom
    })
}

/// (midtone, shadow, highlight) refs normalized against `bounds`; None without both.
pub fn normalized_neutral_axis(
    bounds: Option<&Bounds64>,
    refs: Option<&NeutralAxis64>,
) -> Option<NeutralAxisNorm> {
    let (b, r) = (bounds?, refs?);
    let norm = |v: Rgb64| normalize_refs(v, b.floors, b.ceils);
    Some((norm(r.mid), norm(r.shadow), r.highlight.map(norm)))
}

/// Applied cast-removal strength: the neutral-reference confidence biases the slider.
pub fn effective_cast_strength(strength: f64, confidence: Option<f64>) -> f64 {
    match confidence {
        Some(c) => c * strength,
        None => strength,
    }
}

/// (effective strength, neutral_axis_norm) from the raw metrics.
pub fn cast_solve_inputs(
    bounds: Option<&Bounds64>,
    refs: Option<&NeutralAxis64>,
    slider_strength: f64,
) -> (f64, Option<NeutralAxisNorm>) {
    let norm = normalized_neutral_axis(bounds, refs);
    let confidence = refs.map(|r| r.confidence);
    (effective_cast_strength(slider_strength, confidence), norm)
}

/// Shared straight-line slope before the per-layer gammas.
#[allow(clippy::too_many_arguments)]
pub fn auto_base_slope(
    grade: f64,
    auto_normalize_contrast: bool,
    lum_range: Option<f64>,
    textural_range: Option<f64>,
    shadow_point: Option<f64>,
    anchor: Option<f64>,
    d_min: f64,
    paper: Option<&PaperProfile>,
    midtone_gamma: Option<f64>,
) -> f64 {
    let mut slope = grade_to_slope(
        grade,
        effective_grade_range(auto_normalize_contrast, lum_range, textural_range),
        slope_gain(paper),
    );
    if let (true, Some(point)) = (auto_normalize_contrast, shadow_point) {
        let c = effective_constants(paper);
        let r = anchor.unwrap_or(c.assumed_anchor);
        slope = shadow_reach_slope(
            slope,
            r,
            point,
            d_min,
            paper,
            midtone_gamma,
            anchor.is_some(),
        );
    }
    slope
}

/// `_grade_trim_mult`: a per-layer ISO-R trim as a slope ratio.
pub fn grade_trim_mult(grade: f64, trim: f64, c: &ExposureConstants) -> f64 {
    let r0 = clamp(grade, c.iso_r_min, c.iso_r_max);
    let r1 = clamp(r0 + trim, c.iso_r_min, c.iso_r_max);
    r0 / r1
}

/// Per-pixel slope multiplier for the local-grade map, in float32 as NumPy computes it.
pub fn local_grade_factor_map(grade_deltas: &[f32], grade: f64) -> Vec<f32> {
    let c = &EXPOSURE_CONSTANTS;
    let r0 = clamp(grade, c.iso_r_min, c.iso_r_max) as f32;
    let (lo, hi) = (c.iso_r_min as f32, c.iso_r_max as f32);
    grade_deltas
        .iter()
        .map(|&d| r0 / (r0 + d).max(lo).min(hi))
        .collect()
}

/// Split-grade ISO-R trims as per-layer local contrast gains (multiplier - 1).
pub fn split_grade_deltas(
    grade: f64,
    shadow_grade: f64,
    highlight_grade: f64,
    shadow_trims: Rgb64,
    highlight_trims: Rgb64,
) -> (Rgb64, Rgb64) {
    let c = &EXPOSURE_CONSTANTS;
    (
        std::array::from_fn(|ch| grade_trim_mult(grade, shadow_grade + shadow_trims[ch], c) - 1.0),
        std::array::from_fn(|ch| {
            grade_trim_mult(grade, highlight_grade + highlight_trims[ch], c) - 1.0
        }),
    )
}

/// Grade-coupled baseline toe and shoulder.
pub fn grade_coupled_shape(slope_g: f64, toe: f64, shoulder: f64) -> (f64, f64) {
    let c = &EXPOSURE_CONSTANTS;
    let slope_norm = (slope_g - c.slope_min) / (c.slope_max - c.slope_min);
    let slope_norm = clamp(slope_norm, 0.0, 1.0);
    (
        toe + c.toe_grade_strength * slope_norm,
        shoulder + c.shoulder_grade_strength * slope_norm,
    )
}

/// Paper's variable midtone gamma plus the user's additive trim.
pub fn effective_midtone_gamma(paper: Option<&PaperProfile>, trim: f64) -> f64 {
    effective_constants(paper).paper_midtone_gamma + trim
}

/// Per-layer midtone gamma: paper baseline + global trim + Snap trim, unclamped.
pub fn per_channel_midtone_gamma(
    paper: Option<&PaperProfile>,
    trim: f64,
    snap_trims: Rgb64,
) -> Rgb64 {
    let base = effective_midtone_gamma(paper, trim);
    snap_trims.map(|t| base + t)
}

/// Per-layer effective toe/shoulder: global value + trim, clamped to the slider domain.
pub fn per_channel_toe_shoulder(
    toe_eff: f64,
    shoulder_eff: f64,
    toe_trims: Rgb64,
    shoulder_trims: Rgb64,
) -> (Rgb64, Rgb64) {
    (
        toe_trims.map(|t| clamp(toe_eff + t, -1.0, 1.0)),
        shoulder_trims.map(|t| clamp(shoulder_eff + t, -1.0, 1.0)),
    )
}

/// Per-layer effective widths: global width + trim, clamped to the width slider domain.
pub fn per_channel_widths(
    toe_width: f64,
    shoulder_width: f64,
    toe_width_trims: Rgb64,
    shoulder_width_trims: Rgb64,
) -> (Rgb64, Rgb64) {
    (
        toe_width_trims.map(|t| clamp(toe_width + t, 0.1, 5.0)),
        shoulder_width_trims.map(|t| clamp(shoulder_width + t, 0.1, 5.0)),
    )
}

/// Per-layer dye-separation k: global value + trim, clamped.
pub fn per_channel_dye_separation(dye_separation: f64, trims: Rgb64) -> Rgb64 {
    trims.map(|t| clamp(dye_separation + t, 0.0, 3.0))
}

/// The density black point compensation maps to black, per channel.
pub fn black_density_rgb(d_max: f64, toe: Rgb64, toe_height: f64) -> Rgb64 {
    toe.map(|t| {
        if t < 0.0 {
            d_max + t * toe_height
        } else {
            d_max
        }
    })
}

/// Per-channel paper-white floor: d_min plus the paper's base tint.
pub fn paper_dmin_rgb(d_min: f64, paper: Option<&PaperProfile>) -> Rgb64 {
    match paper {
        Some(p) if d_min > 0.0 => {
            let (c, m, y) = p.base_tint_cmy;
            [
                (d_min + c).max(0.0),
                (d_min + m).max(0.0),
                (d_min + y).max(0.0),
            ]
        }
        _ => [d_min.max(0.0); 3],
    }
}

/// Hold Black's taper span for the kernels; 0 turns it off.
pub fn black_hold_taper(hold_black: bool, paper: Option<&PaperProfile>) -> f64 {
    if hold_black {
        effective_constants(paper).zone_black_hold_taper
    } else {
        0.0
    }
}

/// WB sliders as normalized-space offsets.
pub fn filtration_offsets(wb_cmy: Rgb64, bounds: Option<&Bounds64>) -> Rgb64 {
    let cmy_max = EXPOSURE_CONSTANTS.cmy_max_density;
    std::array::from_fn(|ch| {
        let d = wb_cmy[ch] * cmy_max;
        match bounds {
            Some(b) => d / (b.ceils[ch] - b.floors[ch]).abs().max(1e-6),
            None => d,
        }
    })
}

/// Each channel's stretch range in log10 units; 1 without bounds.
pub fn channel_density_ranges(bounds: Option<&Bounds64>) -> Rgb64 {
    match bounds {
        Some(b) => std::array::from_fn(|ch| (b.ceils[ch] - b.floors[ch]).abs().max(1e-6)),
        None => [1.0; 3],
    }
}

/// A real diffuser passes most of the light unscattered: the slider's top is a quarter.
pub const DIFFUSION_MIX_MAX: f64 = 0.25;

/// The fraction of blurred light the paper sees at a Diffusion slider value.
pub fn diffusion_mix(diffusion: f64) -> f64 {
    DIFFUSION_MIX_MAX * clamp(diffusion, 0.0, 1.0)
}

/// Normalized-space size of one dodge/burn stop per channel.
pub fn local_ev_scale(bounds: Option<&Bounds64>) -> Rgb64 {
    let step = libm::log10(2.0);
    match bounds {
        Some(b) => std::array::from_fn(|ch| step / (b.ceils[ch] - b.floors[ch]).abs().max(1e-6)),
        None => [step; 3],
    }
}

/// The analysis-grid mask plane at render size, landed back at `roi`, edge-replicated.
pub fn expand_mask_plane(
    plane: &Plane,
    out_shape: (usize, usize),
    roi: Option<Roi>,
) -> Option<Plane> {
    let (h, w) = out_shape;
    let (y1, y2, x1, x2) = roi.unwrap_or((0, h, 0, w));
    let (y2, x2) = (y2.min(h), x2.min(w));
    if y2 < y1 + 1 || x2 < x1 + 1 {
        return None;
    }
    let mut out = vec![0.0f32; h * w];
    negpy_core::resize::expand_plane(
        &plane.data,
        (plane.height, plane.width),
        (h, w),
        (y1, y2, x1, x2),
        1.0,
        &mut out,
    )?;
    Some(Plane::new(h, w, out))
}

/// Contrast Mask tone codes, after the Frontier's HyperTone.
pub const MASK_MID_CODE: f64 = 412.0;
pub const MASK_AIM_CODE: f64 = 450.0;
pub const MASK_CODES_PER_DENSITY: f64 = 145.0;
pub const MASK_CODE_MAX: f64 = 1023.0;
pub const MASK_TABLE_CAP: f64 = 819.0;

/// The (a, b, lo, hi, g_lo, g_hi, lim) of `contrast_mask_stops`, or None when off.
pub fn contrast_mask_coeffs(
    strength: f64,
    centre: f64,
    slopes: Rgb64,
    pivots: Rgb64,
    offsets: Rgb64,
    density_range: f64,
) -> Option<[f64; 7]> {
    let slope = (slopes[0] + slopes[1] + slopes[2]) / 3.0;
    if strength <= 0.0 || slope <= 0.0 || slopes[1] <= 0.0 || density_range <= 0.0 {
        return None;
    }
    let c = MASK_CODES_PER_DENSITY;
    let r = EXPOSURE_CONSTANTS.reference_print_density;
    let term = |ch: usize| slopes[ch] * (offsets[ch] - pivots[ch]);
    let shift = (term(0) + term(1) + term(2)) / 3.0;
    let stops_per_code = density_range / (c * slopes[1] * libm::log10(2.0));
    let gain = strength * stops_per_code / (4.0 * 256.0);
    Some([
        c * slope,
        MASK_AIM_CODE + c * (slope * centre + shift - r) - MASK_MID_CODE,
        -MASK_MID_CODE,
        MASK_CODE_MAX - MASK_MID_CODE,
        gain,
        gain,
        MASK_TABLE_CAP / 4.0 * stops_per_code,
    ])
}

/// The contrast mask as print-exposure stops, ready to add to the dodge/burn map.
pub fn contrast_mask_ev(
    plane: Option<&Plane>,
    coeffs: Option<[f64; 7]>,
    out_shape: (usize, usize),
    roi: Option<Roi>,
) -> Option<Plane> {
    let expanded = expand_mask_plane(plane?, out_shape, roi)?;
    let coeffs = coeffs?;
    let mut out = vec![0.0f32; expanded.data.len()];
    negpy_core::analysis::contrast_mask_stops(&expanded.data, &mut out, coeffs.map(|v| v as f32));
    Some(Plane::new(expanded.height, expanded.width, out))
}

/// Fixed (gain, lift) for the flat log master.
pub fn flat_curve_params() -> (f64, f64) {
    (
        EXPOSURE_CONSTANTS.flat_log_gain,
        EXPOSURE_CONSTANTS.flat_log_lift,
    )
}

/// The flat log master: the normalized log signal as the code value, WB as a log shift, the
/// stock's density matrix after it.
pub fn apply_flat_curve(
    img: &Image,
    gain: f64,
    lift: f64,
    cmy_offsets: Rgb64,
    stock_matrix: Option<Rows3>,
) -> Image {
    let m = stock_matrix.map(|rows| rows.map(|r| r.map(|v| v as f32)));
    let mut out = vec![0.0f32; img.data.len()];
    negpy_core::color::flat_curve(
        &img.data,
        &mut out,
        cmy_offsets.map(|o| o as f32),
        m,
        gain as f32,
        lift as f32,
    );
    Image::new(img.height, img.width, out)
}

/// `apply_characteristic_curve`'s keyword arguments; planes are HxW over the image.
#[derive(Clone, Copy, Debug)]
pub struct CurveArgs<'a> {
    pub params: CurveParams,
    pub toe: f64,
    pub toe_width: f64,
    pub shoulder: f64,
    pub shoulder_width: f64,
    pub shadow_cmy: Rgb64,
    pub highlight_cmy: Rgb64,
    pub cmy_offsets: Rgb64,
    pub d_min: f64,
    pub midtone_gamma: Option<f64>,
    pub paper: Option<&'a PaperProfile>,
    pub ev_map: Option<&'a [f32]>,
    pub ev_scale: Rgb64,
    pub grade_map: Option<&'a [f32]>,
    pub bpc: bool,
    pub toe_trims: Rgb64,
    pub shoulder_trims: Rgb64,
    pub snap_trims: Rgb64,
    pub toe_width_trims: Rgb64,
    pub shoulder_width_trims: Rgb64,
    pub shadow_density: f64,
    pub highlight_density: f64,
    pub shadow_grade_deltas: Rgb64,
    pub highlight_grade_deltas: Rgb64,
    pub dye_separation: f64,
    pub dye_separation_trims: Rgb64,
    pub separation_damping: f64,
    /// HxWxK shape alphas with K (stops, ISO-R delta, e0, e1) rows and the ISO-R plane.
    pub key_alpha: Option<&'a [f32]>,
    pub key_params: Option<&'a [[f64; 4]]>,
    pub grade_deltas: Option<&'a [f32]>,
    pub frame_grade: f64,
    pub preflash: f64,
    pub flash_map: Option<&'a [f32]>,
    pub diffusion: f64,
    pub diffusion_plane: Option<&'a Image>,
    /// (x, y, width, height) in pixels of the image the plane covers.
    pub diffusion_rect: Option<[f64; 4]>,
    pub diffusion_range: Rgb64,
    pub black_hold: f64,
}

impl Default for CurveArgs<'_> {
    fn default() -> Self {
        Self {
            params: CurveParams {
                slopes: [0.0; 3],
                pivots: [0.0; 3],
                curvatures: [0.0; 3],
            },
            toe: 0.0,
            toe_width: 2.5,
            shoulder: 0.0,
            shoulder_width: 2.5,
            shadow_cmy: [0.0; 3],
            highlight_cmy: [0.0; 3],
            cmy_offsets: [0.0; 3],
            d_min: 0.0,
            midtone_gamma: None,
            paper: None,
            ev_map: None,
            ev_scale: [0.0; 3],
            grade_map: None,
            bpc: false,
            toe_trims: [0.0; 3],
            shoulder_trims: [0.0; 3],
            snap_trims: [0.0; 3],
            toe_width_trims: [0.0; 3],
            shoulder_width_trims: [0.0; 3],
            shadow_density: 0.0,
            highlight_density: 0.0,
            shadow_grade_deltas: [0.0; 3],
            highlight_grade_deltas: [0.0; 3],
            dye_separation: 1.0,
            dye_separation_trims: [0.0; 3],
            separation_damping: 0.0,
            key_alpha: None,
            key_params: None,
            grade_deltas: None,
            frame_grade: 115.0,
            preflash: 0.0,
            flash_map: None,
            diffusion: 0.0,
            diffusion_plane: None,
            diffusion_rect: None,
            diffusion_range: [1.0; 3],
            black_hold: 0.0,
        }
    }
}

/// The asymmetric H&D print curve per channel in log-density space, through the negpy_core
/// kernel. A LUT stock has no matrix here, so it is an error.
pub fn apply_characteristic_curve(
    img: &Image,
    a: &CurveArgs<'_>,
) -> Result<Image, UnsupportedStock> {
    let c = effective_constants(a.paper);
    let ts = c.toe_shoulder_strength;
    let midtone_gamma = a.midtone_gamma.unwrap_or(c.paper_midtone_gamma);
    let v_star = reference_linear_value(a.d_min, a.paper, None);
    let dye = resolve_print_color(a.paper)?;
    let sat_k3 = per_channel_dye_separation(a.dye_separation, a.dye_separation_trims);
    let use_sep_damping = a.separation_damping > 0.0 && sat_k3 != [1.0, 1.0, 1.0];
    let sat = if use_sep_damping {
        None
    } else {
        resolve_saturation_matrix((sat_k3[0], sat_k3[1], sat_k3[2]))
    };
    let composed = compose_density_matrices(dye, sat);
    let (h, w) = img.shape();
    let rect = a.diffusion_rect.unwrap_or([0.0, 0.0, w as f64, h as f64]);
    let (toe3, sh3) = per_channel_toe_shoulder(a.toe, a.shoulder, a.toe_trims, a.shoulder_trims);
    let (tw3, sw3) = per_channel_widths(
        a.toe_width,
        a.shoulder_width,
        a.toe_width_trims,
        a.shoulder_width_trims,
    );
    let (flash_v, flash_gamma) =
        preflash_params(a.frame_grade, a.d_min, a.paper, Some(midtone_gamma));
    fn plane<'p>(data: &'p [f32], w: usize) -> CorePlane<'p> {
        CorePlane { data, w }
    }
    let keys = match (a.key_alpha, a.key_params, a.grade_deltas) {
        (Some(alpha), Some(params), Some(deltas)) if !params.is_empty() => Some(Keys {
            alpha,
            params,
            deltas: plane(deltas, w),
        }),
        _ => None,
    };
    let diffusion = match a.diffusion_plane {
        Some(p) if a.diffusion > 0.0 => Some(Diffusion {
            plane: &p.data,
            gh: p.height,
            gw: p.width,
            rect,
            mix: diffusion_mix(a.diffusion),
            range: a.diffusion_range,
        }),
        _ => None,
    };
    let p = PrintCurve {
        pivots: a.params.pivots.map(|v| v as f32),
        slopes: a.params.slopes.map(|v| v as f32),
        curvatures: a.params.curvatures.map(|v| v as f32),
        toe: toe3.map(|t| t * ts),
        shoulder: sh3.map(|s| s * ts),
        toe_width: tw3,
        shoulder_width: sw3,
        cmy_offsets: a.cmy_offsets.map(|v| v as f32),
        shadow_cmy: a.shadow_cmy.map(|v| v as f32),
        highlight_cmy: a.highlight_cmy.map(|v| v as f32),
        d_min_rgb: paper_dmin_rgb(a.d_min, a.paper),
        d_max: c.d_max,
        a_toe_base: c.toe_sharpness_base,
        a_sh_base: c.shoulder_sharpness_base,
        width_ref: c.toeshoulder_width_ref,
        toe_height: c.toe_height,
        sh_height: c.shoulder_height,
        zone_center: c.reference_print_density,
        shadow_density: a.shadow_density,
        highlight_density: a.highlight_density,
        shadow_grade: a.shadow_grade_deltas,
        highlight_grade: a.highlight_grade_deltas,
        zone_sh_center: c.reference_print_density + c.zone_density_shadow_offset,
        zone_hi_center: c.reference_print_density + c.zone_density_highlight_offset,
        zone_k: c.zone_density_sharpness,
        v_star,
        midtone_gamma: a.snap_trims.map(|t| midtone_gamma + t),
        gamma_width: c.paper_gamma_width,
        dye_mix: composed,
        sep: use_sep_damping.then_some((
            sat_k3,
            a.separation_damping,
            c.separation_damping_ref_spread,
        )),
        ev: a.ev_map.map(|d| plane(d, w)),
        ev_scale: a.ev_scale.map(|v| v as f32),
        grade: if keys.is_none() {
            a.grade_map.map(|d| plane(d, w))
        } else {
            None
        },
        keys,
        r0: clamp(a.frame_grade, c.iso_r_min, c.iso_r_max),
        r_min: c.iso_r_min,
        r_max: c.iso_r_max,
        flash: a.preflash.max(0.0),
        flash_v,
        flash_gamma,
        flash_map: a.flash_map.map(|d| plane(d, w)),
        diffusion,
        lut: None,
        bpc: a.bpc,
        black_hold: a.black_hold,
    };
    let mut out = vec![0.0f32; img.data.len()];
    print_curve::apply_print_curve(&img.data, w, &p, &mut out);
    Ok(Image::new(h, w, out))
}

/// `np.linalg.solve` of a 3x3 system: LU with partial pivoting; None when singular.
fn solve3(mut a: [[f64; 3]; 3], mut b: [f64; 3]) -> Option<[f64; 3]> {
    for j in 0..3 {
        let mut p = j;
        for i in j + 1..3 {
            if a[i][j].abs() > a[p][j].abs() {
                p = i;
            }
        }
        if a[p][j] == 0.0 {
            return None;
        }
        a.swap(p, j);
        b.swap(p, j);
        let r = 1.0 / a[j][j];
        for row in a.iter_mut().skip(j + 1) {
            row[j] *= r;
        }
        for i in j + 1..3 {
            for k in j + 1..3 {
                a[i][k] -= a[i][j] * a[j][k];
            }
        }
    }
    for k in 0..3 {
        if b[k] != 0.0 {
            for i in k + 1..3 {
                b[i] -= b[k] * a[i][k];
            }
        }
    }
    for k in (0..3).rev() {
        if b[k] != 0.0 {
            b[k] /= a[k][k];
            for i in 0..k {
                b[i] -= b[k] * a[i][k];
            }
        }
    }
    Some(b)
}

/// Per-channel (slopes, pivots, curvatures). Cast Removal fits R and B to green's neutral
/// axis: a line through the midtone and shadow, a quadratic when a highlight ref is present.
#[allow(clippy::too_many_arguments)]
pub fn per_channel_curve_params(
    grade: f64,
    density: f64,
    auto_normalize_contrast: bool,
    strength: f64,
    lum_range: Option<f64>,
    textural_range: Option<f64>,
    d_min: f64,
    anchor: Option<f64>,
    paper: Option<&PaperProfile>,
    neutral_axis_norm: Option<&NeutralAxisNorm>,
    grade_trims: Rgb64,
    shadow_point: Option<f64>,
    midtone_gamma: Option<f64>,
) -> CurveParams {
    let c = effective_constants(paper);
    let mut cg: Rgb64 = paper.map_or([1.0; 3], |p| {
        [p.channel_gamma.0, p.channel_gamma.1, p.channel_gamma.2]
    });
    if grade_trims != [0.0, 0.0, 0.0] {
        cg = std::array::from_fn(|ch| cg[ch] * grade_trim_mult(grade, grade_trims[ch], &c));
    }
    let (slope_min, slope_max) = (c.slope_min, c.slope_max);
    let base_slope = auto_base_slope(
        grade,
        auto_normalize_contrast,
        lum_range,
        textural_range,
        shadow_point,
        anchor,
        d_min,
        paper,
        midtone_gamma,
    );
    let epsilon = 1e-6;

    if let (true, Some((mid_norm, sh_norm, hl_norm))) = (strength > 0.0, neutral_axis_norm) {
        let limit = c.midtone_cast_max_offset;
        let curv_lim = c.neutral_axis_curv_max_ratio;
        let (m_g, s_g) = (mid_norm[1], sh_norm[1]);
        let slope_g = clamp(base_slope * cg[1], slope_min, slope_max);
        let pivot_g = compute_pivot(slope_g, density, d_min, anchor, paper, midtone_gamma);
        let target = |g: f64| slope_g * (g - pivot_g);
        let (t_m, t_s) = (target(m_g), target(s_g));
        let h_g = hl_norm.map(|h| h[1]);
        let clamp_dev = |g: f64, v: f64| g + clamp(strength * (v - g), -limit, limit);

        let mut out = CurveParams {
            slopes: [0.0; 3],
            pivots: [0.0; 3],
            curvatures: [0.0; 3],
        };
        for ch in 0..3 {
            if ch == 1 {
                out.slopes[ch] = slope_g;
                out.pivots[ch] = pivot_g;
                continue;
            }
            let u_m = clamp_dev(m_g, mid_norm[ch]);
            let u_s = clamp_dev(s_g, sh_norm[ch]);
            let mut curv = 0.0;
            if let (Some(h_g), Some(hl)) = (h_g, hl_norm) {
                let u_h = clamp_dev(h_g, hl[ch]);
                let m = [
                    [1.0, u_h, u_h * u_h],
                    [1.0, u_m, u_m * u_m],
                    [1.0, u_s, u_s * u_s],
                ];
                curv = solve3(m, [target(h_g), t_m, t_s]).map_or(0.0, |x| x[2]);
                curv = clamp(curv, -curv_lim * slope_g, curv_lim * slope_g);
            }
            let du = u_m - u_s;
            let slope_ch = if du.abs() < epsilon {
                slope_g
            } else {
                ((t_m - t_s) - curv * (u_m * u_m - u_s * u_s)) / du
            };
            let slope_ch = clamp(slope_ch * cg[ch], slope_min, slope_max);
            let curv_ch = curv * cg[ch];
            let pivot_ch = if slope_ch.abs() > epsilon {
                u_m - (t_m - curv_ch * u_m * u_m) / slope_ch
            } else {
                pivot_g
            };
            out.slopes[ch] = slope_ch;
            out.pivots[ch] = pivot_ch;
            out.curvatures[ch] = curv_ch;
        }
        return out;
    }

    let slopes: Rgb64 = std::array::from_fn(|ch| clamp(base_slope * cg[ch], slope_min, slope_max));
    let pivots: Rgb64 =
        slopes.map(|s| compute_pivot(s, density, d_min, anchor, paper, midtone_gamma));
    CurveParams {
        slopes,
        pivots,
        curvatures: [0.0; 3],
    }
}
