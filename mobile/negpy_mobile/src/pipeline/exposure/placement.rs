//! `features/exposure/placement.py`'s tone-limited mask edges: `key_edges` inverts the chart's
//! forward model (`curve_params_from_metrics` -> `print_curve` -> `print_curve_output` ->
//! `zone_of_encoded`) on a grid, as the desktop does. The zone solver itself is UI.

use crate::config::{MaskKey, ProcessMode};
use crate::pipeline::papers::{PaperProfile, effective_constants, effective_paper_profile};
use crate::pipeline::working::WORKING_GAMMA;

use super::curve::{
    CurveParams, black_hold_taper, cast_solve_inputs, effective_midtone_gamma, grade_coupled_shape,
    highlight_hold_offset, per_channel_curve_params, preflash_params, preflash_value,
    reference_linear_value, shadow_hold_offset, split_grade_deltas,
};
use super::local::{LocalMask64, limited_masks};
use super::{Exposure, ExposureMetrics};

pub const ZONE_MID_REFLECTANCE: f32 = 0.18;
const KEY_GRID_N: usize = 512;
const KEY_MIN_SPAN: f64 = 1e-4;

/// `np.logaddexp(x, y)`.
fn logaddexp(x: f64, y: f64) -> f64 {
    if x == y {
        return x + std::f64::consts::LN_2;
    }
    let tmp = x - y;
    if tmp > 0.0 {
        x + libm::log1p(libm::exp(-tmp))
    } else if tmp <= 0.0 {
        y + libm::log1p(libm::exp(tmp))
    } else {
        tmp
    }
}

/// `_expit`: exp(-logaddexp(0, -x)).
fn expit(x: f64) -> f64 {
    libm::exp(-logaddexp(0.0, -x))
}

/// `CharacteristicCurve`'s constructor arguments.
#[derive(Clone, Copy, Debug)]
pub struct CurveSpec<'a> {
    pub contrast: f64,
    pub pivot: f64,
    pub d_min: f64,
    pub toe: f64,
    pub toe_width: f64,
    pub shoulder: f64,
    pub shoulder_width: f64,
    pub paper: Option<&'a PaperProfile>,
    pub midtone_gamma: Option<f64>,
    pub bpc: bool,
    pub shadow_density: f64,
    pub highlight_density: f64,
    pub shadow_grade_delta: f64,
    pub highlight_grade_delta: f64,
    pub curvature: f64,
    pub preflash: f64,
    pub grade: f64,
    pub black_hold: f64,
}

impl Default for CurveSpec<'_> {
    fn default() -> Self {
        Self {
            contrast: 2.0,
            pivot: 0.0,
            d_min: 0.0,
            toe: 0.0,
            toe_width: 2.5,
            shoulder: 0.0,
            shoulder_width: 2.5,
            paper: None,
            midtone_gamma: None,
            bpc: false,
            shadow_density: 0.0,
            highlight_density: 0.0,
            shadow_grade_delta: 0.0,
            highlight_grade_delta: 0.0,
            curvature: 0.0,
            preflash: 0.0,
            grade: 115.0,
            black_hold: 0.0,
        }
    }
}

/// The achromatic H&D print curve in density space: the scalar mirror of the print kernel,
/// for the chart and the tone-limited mask edges.
#[derive(Clone, Copy, Debug)]
pub struct CharacteristicCurve {
    k: f64,
    x0: f64,
    curvature: f64,
    flash: f64,
    midtone_gamma: f64,
    flash_v: f64,
    flash_gamma: f64,
    v_star: f64,
    gamma_width: f64,
    zone_sh_center: f64,
    zone_hi_center: f64,
    zone_k: f64,
    shadow_density: f64,
    highlight_density: f64,
    black_hold: f64,
    shadow_grade_delta: f64,
    highlight_grade_delta: f64,
    bpc: bool,
    bpc_black: f64,
    a_hl: f64,
    a_sh: f64,
    d_min_eff: f64,
    d_max_eff: f64,
}

impl CharacteristicCurve {
    pub fn new(s: &CurveSpec<'_>) -> Self {
        let c = effective_constants(s.paper);
        let ts = c.toe_shoulder_strength;
        let midtone_gamma = s.midtone_gamma.unwrap_or(c.paper_midtone_gamma);
        let (flash_v, flash_gamma) =
            preflash_params(s.grade, s.d_min, s.paper, Some(midtone_gamma));
        let d_max = c.d_max;
        let mut db = d_max;
        if s.toe * ts < 0.0 {
            db = d_max + s.toe * ts * c.toe_height;
        }
        let wr = c.toeshoulder_width_ref;
        let a_hl = c.shoulder_sharpness_base * wr / s.shoulder_width.max(1e-6);
        let a_sh_base = c.toe_sharpness_base * wr / s.toe_width.max(1e-6);
        let d_min_eff = (s.d_min + s.shoulder * ts * c.shoulder_height).max(0.0);
        let toe_eff = s.toe * ts;
        let (mut d_max_eff, a_sh) = if toe_eff >= 0.0 {
            (d_max - toe_eff * c.toe_height, a_sh_base)
        } else {
            (d_max, a_sh_base * (1.0 - toe_eff * 4.0))
        };
        if d_max_eff < d_min_eff + 0.1 {
            d_max_eff = d_min_eff + 0.1;
        }
        Self {
            k: s.contrast,
            x0: s.pivot,
            curvature: s.curvature,
            flash: s.preflash.max(0.0),
            midtone_gamma,
            flash_v,
            flash_gamma,
            v_star: reference_linear_value(s.d_min, s.paper, None),
            gamma_width: c.paper_gamma_width,
            zone_sh_center: c.reference_print_density + c.zone_density_shadow_offset,
            zone_hi_center: c.reference_print_density + c.zone_density_highlight_offset,
            zone_k: c.zone_density_sharpness,
            shadow_density: s.shadow_density,
            highlight_density: s.highlight_density,
            black_hold: s.black_hold,
            shadow_grade_delta: s.shadow_grade_delta,
            highlight_grade_delta: s.highlight_grade_delta,
            bpc: s.bpc,
            bpc_black: libm::pow(10.0, -db),
            a_hl,
            a_sh,
            d_min_eff,
            d_max_eff,
        }
    }

    /// Print density of the normalized-log value `x`.
    pub fn density(&self, x: f64) -> f64 {
        let mut xv = x;
        if self.curvature != 0.0 {
            let vertex = -self.k / (2.0 * self.curvature);
            xv = if self.curvature > 0.0 {
                xv.max(vertex)
            } else {
                xv.min(vertex)
            };
        }
        let mut v = self.k * (xv - self.x0) + self.curvature * xv * xv;
        if self.flash > 0.0 {
            v = preflash_value(v, self.flash_v, self.flash, self.flash_gamma);
        }
        if self.midtone_gamma != 0.0 {
            v += self.midtone_gamma
                * self.gamma_width
                * libm::tanh((v - self.v_star) / self.gamma_width);
        }
        if self.shadow_grade_delta != 0.0 || self.highlight_grade_delta != 0.0 {
            let w_gsh = expit(self.zone_k * (v - self.zone_sh_center));
            let w_ghi = 1.0 - expit(self.zone_k * (v - self.zone_hi_center));
            v = v
                + self.shadow_grade_delta * w_gsh * (v - self.zone_sh_center)
                + self.highlight_grade_delta * w_ghi * (v - self.zone_hi_center);
        }
        if self.shadow_density != 0.0 || self.highlight_density != 0.0 {
            let mut w_zsh = expit(self.zone_k * (v - self.zone_sh_center));
            if self.black_hold > 0.0 && self.shadow_density < 0.0 {
                let t = ((self.d_max_eff - v) / self.black_hold).clamp(0.0, 1.0);
                w_zsh = w_zsh * t * t * (3.0 - 2.0 * t);
            }
            let w_zhi = 1.0 - expit(self.zone_k * (v - self.zone_hi_center));
            v = v + self.shadow_density * w_zsh + self.highlight_density * w_zhi;
        }
        let v1 = self.d_min_eff + logaddexp(0.0, self.a_hl * (v - self.d_min_eff)) / self.a_hl;
        let mut res =
            self.d_max_eff - logaddexp(0.0, self.a_sh * (self.d_max_eff - v1)) / self.a_sh;
        if self.bpc {
            let t = libm::pow(10.0, -res);
            let t = (t - self.bpc_black) / (1.0 - self.bpc_black);
            res = -libm::log10(t.max(1e-12));
        }
        res
    }
}

/// The achromatic print curve for `exposure` at `slope`/`pivot`; a None density takes the
/// exposure's own.
pub fn print_curve(
    exposure: &Exposure,
    slope: f64,
    pivot: f64,
    mode: ProcessMode,
    curvature: f64,
    shadow_density: Option<f64>,
    highlight_density: Option<f64>,
) -> CharacteristicCurve {
    let profile = effective_paper_profile(&exposure.paper_profile, mode);
    let d_min = if exposure.paper_dmin {
        profile.d_min
    } else {
        0.0
    };
    let (toe_eff, shoulder_eff) = grade_coupled_shape(slope, exposure.toe, exposure.shoulder);
    let (sg, hg) = split_grade_deltas(
        exposure.grade,
        exposure.shadow_grade,
        exposure.highlight_grade,
        [0.0; 3],
        [0.0; 3],
    );
    CharacteristicCurve::new(&CurveSpec {
        contrast: slope,
        pivot,
        d_min,
        toe: toe_eff,
        toe_width: exposure.toe_width,
        shoulder: shoulder_eff,
        shoulder_width: exposure.shoulder_width,
        paper: Some(profile),
        midtone_gamma: Some(effective_midtone_gamma(
            Some(profile),
            exposure.midtone_gamma,
        )),
        bpc: !exposure.paper_black,
        shadow_density: shadow_density.unwrap_or(exposure.shadow_density),
        highlight_density: highlight_density.unwrap_or(exposure.highlight_density),
        shadow_grade_delta: sg[0],
        highlight_grade_delta: hg[0],
        curvature,
        preflash: exposure.preflash,
        grade: exposure.grade,
        black_hold: black_hold_taper(exposure.hold_black, Some(profile)),
    })
}

/// `WORKING_INV_GAMMA`: float32 of the float64 reciprocal.
fn working_inv_gamma() -> f32 {
    (1.0 / WORKING_GAMMA as f64) as f32
}

/// Curve density of a float32 input to the display-encoded output: 10^-D in float32, then
/// the working OETF.
pub fn print_curve_output(curve: &CharacteristicCurve, x_log_exp: f64) -> f64 {
    let d = curve.density((x_log_exp as f32) as f64) as f32;
    let t = libm::powf(10.0, -d);
    let mut out = [0.0f32];
    negpy_core::color::oetf_encode(&[t], &mut out, working_inv_gamma());
    out[0] as f64
}

fn mid_gray_encoded() -> f64 {
    let mut out = [0.0f32];
    negpy_core::color::oetf_encode(&[ZONE_MID_REFLECTANCE], &mut out, working_inv_gamma());
    out[0] as f64
}

/// Print zone (0..10) of a display-encoded lightness.
pub fn zone_of_encoded(enc: f64) -> f64 {
    let enc = enc.clamp(0.0, 1.0);
    let mid = mid_gray_encoded();
    if enc <= mid {
        5.0 * enc / mid
    } else {
        5.0 + 5.0 * (enc - mid) / (1.0 - mid)
    }
}

/// (slopes, pivots, curvatures) the render solved for, from the metrics it published.
pub fn curve_params_from_metrics(
    exposure: &Exposure,
    mode: ProcessMode,
    metrics: &ExposureMetrics,
) -> CurveParams {
    let profile = effective_paper_profile(&exposure.paper_profile, mode);
    let d_min = if exposure.paper_dmin {
        profile.d_min
    } else {
        0.0
    };
    let anchor = if exposure.auto_exposure {
        metrics.metered_anchor
    } else {
        None
    };
    let bounds = metrics.final_bounds.or(metrics.log_bounds);
    let (strength, norm) = cast_solve_inputs(
        bounds.as_ref(),
        metrics.neutral_axis_refs.as_ref(),
        exposure.cast_removal_strength,
    );
    per_channel_curve_params(
        exposure.grade,
        exposure.density,
        exposure.auto_normalize_contrast,
        strength,
        metrics.norm_density_range,
        metrics.textural_range,
        d_min,
        anchor,
        Some(profile),
        norm.as_ref(),
        [
            exposure.grade_trim_red,
            exposure.grade_trim_green,
            exposure.grade_trim_blue,
        ],
        metrics.shadow_point,
        Some(effective_midtone_gamma(
            Some(profile),
            exposure.midtone_gamma,
        )),
    )
}

/// The shadow hold lift the render applied, from the published metrics.
pub fn auto_shadow_from_metrics(
    exposure: &Exposure,
    mode: ProcessMode,
    metrics: &ExposureMetrics,
) -> f64 {
    let (true, Some(point)) = (exposure.auto_normalize_contrast, metrics.shadow_point) else {
        return 0.0;
    };
    let profile = effective_paper_profile(&exposure.paper_profile, mode);
    let d_min = if exposure.paper_dmin {
        profile.d_min
    } else {
        0.0
    };
    let p = curve_params_from_metrics(exposure, mode, metrics);
    let mg = effective_midtone_gamma(Some(profile), exposure.midtone_gamma);
    shadow_hold_offset(
        p.slopes[1],
        p.pivots[1],
        point,
        d_min,
        Some(profile),
        Some(mg),
    )
}

/// The highlight hold burn the render applied, from the published metrics.
pub fn auto_highlight_from_metrics(
    exposure: &Exposure,
    mode: ProcessMode,
    metrics: &ExposureMetrics,
) -> f64 {
    let (true, Some(point)) = (exposure.auto_normalize_contrast, metrics.highlight_point) else {
        return 0.0;
    };
    let profile = effective_paper_profile(&exposure.paper_profile, mode);
    let d_min = if exposure.paper_dmin {
        profile.d_min
    } else {
        0.0
    };
    let p = curve_params_from_metrics(exposure, mode, metrics);
    highlight_hold_offset(
        p.slopes[1],
        p.pivots[1],
        point,
        d_min,
        Some(profile),
        exposure.preflash,
        exposure.grade,
        Some(effective_midtone_gamma(
            Some(profile),
            exposure.midtone_gamma,
        )),
    )
}

/// Zones the achromatic print curve puts normalized-log `vals` on under `exposure`.
fn zones_of(
    exposure: &Exposure,
    mode: ProcessMode,
    metrics: &ExposureMetrics,
    vals: &[f64],
) -> Vec<f64> {
    let p = curve_params_from_metrics(exposure, mode, metrics);
    let hl = exposure.highlight_density + auto_highlight_from_metrics(exposure, mode, metrics);
    let sh = exposure.shadow_density + auto_shadow_from_metrics(exposure, mode, metrics);
    let curve = print_curve(
        exposure,
        p.slopes[1],
        p.pivots[1],
        mode,
        p.curvatures[1],
        Some(sh),
        Some(hl),
    );
    vals.iter()
        .map(|&v| zone_of_encoded(print_curve_output(&curve, v)))
        .collect()
}

/// Zone the achromatic print curve puts `val_luma` on under `exposure`.
pub fn predicted_zone(
    exposure: &Exposure,
    mode: ProcessMode,
    metrics: &ExposureMetrics,
    val_luma: f64,
) -> f64 {
    zones_of(exposure, mode, metrics, &[val_luma])[0]
}

/// `np.interp(x, xp, fp)` for a rising `xp`.
fn interp(x: f64, xp: &[f64], fp: &[f64]) -> f64 {
    let n = xp.len();
    if x.is_nan() {
        return x;
    }
    if x < xp[0] {
        return fp[0];
    }
    if x > xp[n - 1] {
        return fp[n - 1];
    }
    let j = xp.partition_point(|&v| v <= x) - 1;
    if j == n - 1 || xp[j] == x {
        return fp[j];
    }
    let slope = (fp[j + 1] - fp[j]) / (xp[j + 1] - xp[j]);
    let mut r = slope * (x - xp[j]) + fp[j];
    if r.is_nan() {
        r = slope * (x - xp[j + 1]) + fp[j + 1];
        if r.is_nan() && fp[j] == fp[j + 1] {
            r = fp[j];
        }
    }
    r
}

/// `np.linspace(-0.5, 1.5, 512)`.
fn key_grid() -> Vec<f64> {
    let (start, stop) = (-0.5, 1.5);
    let step = (stop - start) / (KEY_GRID_N - 1) as f64;
    let mut grid: Vec<f64> = (0..KEY_GRID_N).map(|i| i as f64 * step + start).collect();
    grid[KEY_GRID_N - 1] = stop;
    grid
}

/// A tone-limited mask's weight edges in normalized-log luma: weight 0 at e0, 1 at e1.
pub fn key_edges(
    mask: &LocalMask64,
    exposure: &Exposure,
    mode: ProcessMode,
    metrics: &ExposureMetrics,
) -> (f64, f64) {
    let grid = key_grid();
    let mut zones = zones_of(exposure, mode, metrics, &grid);
    zones.reverse();
    let mut vals = grid;
    vals.reverse();
    let half = 0.5 * mask.key_softness;
    let lighter = interp(mask.key_zone + half, &zones, &vals);
    let darker = interp(mask.key_zone - half, &zones, &vals).max(lighter + KEY_MIN_SPAN);
    if mask.key == MaskKey::Highlights {
        (darker, lighter)
    } else {
        (lighter, darker)
    }
}

/// (stops, ISO-R delta, e0, e1) per tone-limited mask, in plane order; None when there is none.
pub fn limited_mask_params(
    masks: &[LocalMask64],
    exposure: &Exposure,
    mode: ProcessMode,
    metrics: &ExposureMetrics,
) -> Option<Vec<[f64; 4]>> {
    let limited = limited_masks(masks);
    if limited.is_empty() {
        return None;
    }
    Some(
        limited
            .iter()
            .map(|m| {
                let (e0, e1) = key_edges(m, exposure, mode, metrics);
                [m.stops, m.grade, e0, e1]
            })
            .collect(),
    )
}
