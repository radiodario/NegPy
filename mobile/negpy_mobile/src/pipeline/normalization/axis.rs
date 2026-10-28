//! The roll's pooled neutral axis (`pooled_neutral_axis`, `process/models.py`) and the blend
//! of a frame's own midtone offset onto it (`blend_neutral_axis`).

use serde_json::Value;

use super::Axis;
use crate::config::{ProcessConfig, ProcessMode};

/// The Frontier's cast-hue strength, piecewise linear in the cast's hue.
const CAST_HUE_DEGREES: [f64; 9] = [-180.0, -120.0, -90.0, -30.0, 0.0, 30.0, 100.0, 140.0, 180.0];
const CAST_HUE_STRENGTH: [f64; 9] = [0.3, 0.3, 0.6, 0.6, 0.0, 0.0, 1.0, 1.0, 0.3];

fn triplet(v: &Value) -> Option<[f64; 3]> {
    let a = v.as_array()?;
    Some([
        a.first()?.as_f64()?,
        a.get(1)?.as_f64()?,
        a.get(2)?.as_f64()?,
    ])
}

/// `neutral_axis_tuple`: an axis read back from JSON; a pooled one carries its offset weight.
pub fn axis_of_json(v: &Value) -> Option<Axis> {
    let a = v.as_array()?;
    Some(Axis {
        mid: triplet(a.first()?)?,
        shadow: triplet(a.get(1)?)?,
        highlight: match a.get(2) {
            Some(h) if !h.is_null() => Some(triplet(h)?),
            _ => None,
        },
        confidence: a.get(3)?.as_f64()?,
        offset_weight: a.get(4).and_then(Value::as_f64),
    })
}

/// `pooled_neutral_axis`: the roll's axis this frame renders Cast Removal with, or None to
/// meter its own. Color negative only.
pub fn pooled_neutral_axis(process: &ProcessConfig) -> Option<Axis> {
    if process.use_cast_average && process.process_mode == ProcessMode::C41 {
        process.locked_neutral_axis.as_ref().and_then(axis_of_json)
    } else {
        None
    }
}

/// `np.polyfit(x, y, len(x) - 1)` through two or three points, highest power first. The exact
/// interpolant in closed form; `polyfit`'s scaled least squares rounds its last bits differently.
fn interpolant(x: &[f64], y: &[f64]) -> Vec<f64> {
    match x.len() {
        2 => {
            let slope = (y[1] - y[0]) / (x[1] - x[0]);
            vec![slope, y[0] - slope * x[0]]
        }
        3 => {
            let d0 = (x[0] - x[1]) * (x[0] - x[2]);
            let d1 = (x[1] - x[0]) * (x[1] - x[2]);
            let d2 = (x[2] - x[0]) * (x[2] - x[1]);
            if d0 == 0.0 || d1 == 0.0 || d2 == 0.0 {
                // Two bands at one green: the line through the two distinct points.
                let (j, k) = if x[0] != x[1] { (0, 1) } else { (0, 2) };
                return interpolant(&[x[j], x[k]], &[y[j], y[k]]);
            }
            let (l0, l1, l2) = (y[0] / d0, y[1] / d1, y[2] / d2);
            vec![
                l0 + l1 + l2,
                -(l0 * (x[1] + x[2]) + l1 * (x[0] + x[2]) + l2 * (x[0] + x[1])),
                l0 * x[1] * x[2] + l1 * x[0] * x[2] + l2 * x[0] * x[1],
            ]
        }
        _ => vec![y[0]],
    }
}

/// `np.polyval`: Horner from the highest power.
fn polyval(coef: &[f64], x: f64) -> f64 {
    coef.iter().fold(0.0, |acc, &c| acc * x + c)
}

/// `_axis_curve_residuals`: (mid, shadow) x (R, B) distance of `axis` from `pooled`'s R/B
/// against G curve, each read at `axis`'s own green.
fn axis_curve_residuals(axis: &Axis, pooled: &Axis) -> [[f64; 2]; 2] {
    let bands: Vec<[f64; 3]> = [Some(pooled.mid), Some(pooled.shadow), pooled.highlight]
        .into_iter()
        .flatten()
        .collect();
    let g: Vec<f64> = bands.iter().map(|b| b[1]).collect();
    let ptp = g.iter().copied().fold(f64::NEG_INFINITY, f64::max)
        - g.iter().copied().fold(f64::INFINITY, f64::min);
    let own = [axis.mid, axis.shadow];
    let mut out = [[0.0; 2]; 2];
    for (j, ch) in [0usize, 2].into_iter().enumerate() {
        let coef = (ptp > 1e-6).then(|| {
            let y: Vec<f64> = bands.iter().map(|b| b[ch]).collect();
            interpolant(&g, &y)
        });
        for (i, band) in own.iter().enumerate() {
            let ref_g = band[1];
            let fit = match &coef {
                Some(c) => polyval(c, ref_g),
                None => bands[0][ch] - bands[0][1] + ref_g,
            };
            out[i][j] = band[ch] - fit;
        }
    }
    out
}

/// `np.interp` of one value over increasing `xp`.
fn interp(x: f64, xp: &[f64], fp: &[f64]) -> f64 {
    let last = xp.len() - 1;
    if x.is_nan() {
        return x;
    }
    if x < xp[0] {
        return fp[0];
    }
    if x >= xp[last] {
        return fp[last];
    }
    let j = xp.partition_point(|&v| v <= x) - 1;
    if xp[j] == x {
        return fp[j];
    }
    let slope = (fp[j + 1] - fp[j]) / (xp[j + 1] - xp[j]);
    slope * (x - xp[j]) + fp[j]
}

/// `_cast_hue_strength`: the share of a frame's own cast that Cast Removal removes, from its
/// (R, B) log offsets at equal green.
fn cast_hue_strength(dr: f64, db: f64) -> f64 {
    let hue =
        libm::atan2(0.5 * dr - db, -0.5 * 3.0f64.sqrt() * dr) * (180.0 / std::f64::consts::PI);
    interp(hue, &CAST_HUE_DEGREES, &CAST_HUE_STRENGTH)
}

/// `blend_neutral_axis`: the pooled axis shifted by this frame's own midtone offset from the
/// pooled curve, scaled by the pool's offset weight, the frame's confidence and the cast-hue
/// strength. No own axis, or a zero weight, is the pooled axis itself.
pub fn blend_neutral_axis(own: Option<&Axis>, pooled: &Axis) -> Axis {
    let weight = pooled.offset_weight.unwrap_or(0.0);
    let mut k = own.map_or(0.0, |o| weight * o.confidence);
    let pooled_only = Axis {
        offset_weight: None,
        ..pooled.clone()
    };
    let Some(own) = own.filter(|_| k > 0.0) else {
        return pooled_only;
    };
    let [dr, db] = axis_curve_residuals(own, pooled)[0];
    k *= cast_hue_strength(dr, db);
    let shift = |b: [f64; 3]| [b[0] + k * dr, b[1], b[2] + k * db];
    Axis {
        mid: shift(pooled.mid),
        shadow: shift(pooled.shadow),
        highlight: pooled.highlight.map(shift),
        confidence: pooled.confidence,
        offset_weight: None,
    }
}
