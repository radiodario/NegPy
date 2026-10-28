//! `features/local/logic.py`: the dodge/burn, grade and flash planes from the masks' vertices,
//! mapped through the frame's geometry (`geometry/logic.py::map_coords_to_geometry`).

use negpy_core::shapes::{self, MaskAlpha};
use serde::Deserialize;

use crate::config::{LocalMask, MaskKey, MaskShape};
use crate::context::GeometryParams;

pub const MAX_KEYED_MASKS: usize = 4;
const OVAL_SAMPLES: usize = 64;
const KEYSTONE_EPS: f64 = 1e-4;
const DISTORT_EPS: f64 = 1e-6;

/// `LocalMask` at the desktop's float64.
#[derive(Clone, Debug, PartialEq, Deserialize)]
#[serde(default)]
pub struct LocalMask64 {
    pub vertices: Vec<(f64, f64)>,
    pub stops: f64,
    pub feather: f64,
    pub grade: f64,
    pub flash: f64,
    pub shape: MaskShape,
    pub invert: bool,
    pub enabled: bool,
    pub key: MaskKey,
    pub key_zone: f64,
    pub key_softness: f64,
}

impl From<&LocalMask> for LocalMask64 {
    fn from(m: &LocalMask) -> Self {
        Self {
            vertices: m
                .vertices
                .iter()
                .map(|&(x, y)| (x as f64, y as f64))
                .collect(),
            stops: m.stops as f64,
            feather: m.feather as f64,
            grade: m.grade as f64,
            flash: m.flash as f64,
            shape: m.shape,
            invert: m.invert,
            enabled: m.enabled,
            key: m.key,
            key_zone: m.key_zone as f64,
            key_softness: m.key_softness as f64,
        }
    }
}

impl Default for LocalMask64 {
    fn default() -> Self {
        (&LocalMask::default()).into()
    }
}

/// The minimum number of vertices that this shape needs.
pub fn min_points(shape: MaskShape) -> usize {
    if shape == MaskShape::Gradient { 2 } else { 3 }
}

/// Catmull-Rom densification of a polyline through its points; fewer than 3 are unchanged.
pub fn smooth_polyline(
    pts: &[(f64, f64)],
    closed: bool,
    samples_per_seg: usize,
) -> Vec<(f64, f64)> {
    let n = pts.len();
    if n < 3 {
        return pts.to_vec();
    }
    let step = 1.0 / samples_per_seg as f64;
    let n_seg = if closed { n } else { n - 1 };
    let mut out = Vec::with_capacity(n_seg * samples_per_seg + 1);
    for i in 0..n_seg {
        let at = |k: isize| -> (f64, f64) {
            if closed {
                pts[k.rem_euclid(n as isize) as usize]
            } else {
                pts[k.clamp(0, n as isize - 1) as usize]
            }
        };
        let i = i as isize;
        let (p0, p1, p2, p3) = (at(i - 1), at(i), at(i + 1), at(i + 2));
        for s in 0..samples_per_seg {
            let t = s as f64 * step;
            let t2 = t * t;
            let t3 = t * t * t;
            let axis = |p0: f64, p1: f64, p2: f64, p3: f64| {
                0.5 * (2.0 * p1
                    + (p2 - p0) * t
                    + (2.0 * p0 - 5.0 * p1 + 4.0 * p2 - p3) * t2
                    + (-p0 + 3.0 * p1 - 3.0 * p2 + p3) * t3)
            };
            out.push((axis(p0.0, p1.0, p2.0, p3.0), axis(p0.1, p1.1, p2.1, p3.1)));
        }
    }
    if !closed {
        out.push(pts[n - 1]);
    }
    out
}

/// The closed mask outline in the space of `pts`; empty for a gradient.
pub fn outline_points(shape: MaskShape, pts: &[(f64, f64)]) -> Vec<(f64, f64)> {
    match shape {
        MaskShape::Gradient => Vec::new(),
        MaskShape::Oval => {
            let ((cx, cy), (px, py), (qx, qy)) = (pts[0], pts[1], pts[2]);
            let (ux, uy) = (px - cx, py - cy);
            let (vx, vy) = (qx - cx, qy - cy);
            (0..OVAL_SAMPLES)
                .map(|i| {
                    let t = 2.0 * std::f64::consts::PI * i as f64 / OVAL_SAMPLES as f64;
                    let (c, s) = (libm::cos(t), libm::sin(t));
                    (cx + ux * c + vx * s, cy + uy * c + vy * s)
                })
                .collect()
        }
        MaskShape::Polygon => smooth_polyline(pts, true, 16),
    }
}

/// The alpha half of a mask spec: the outline in pixels (float32, truncated), or a gradient's
/// card-edge ramp.
fn mask_spec(
    shape: MaskShape,
    pts: &[(f64, f64)],
    h: usize,
    w: usize,
    feather_sigma: f64,
) -> MaskAlpha {
    let (wf, hf) = (w as f64, h as f64);
    if shape == MaskShape::Gradient {
        let (a, b) = (pts[0], pts[1]);
        let (ax, ay) = (a.0 * wf, a.1 * hf);
        let (dx, dy) = (b.0 * wf - ax, b.1 * hf - ay);
        let denom = dx * dx + dy * dy;
        if denom <= 1e-9 {
            return MaskAlpha::Empty;
        }
        return MaskAlpha::Ramp {
            ax: ax as f32,
            ay: ay as f32,
            kx: (dx / denom) as f32,
            ky: (dy / denom) as f32,
        };
    }
    let outline = outline_points(shape, pts)
        .into_iter()
        .map(|(x, y)| {
            (
                ((x * wf) as f32) as i32 as i64,
                ((y * hf) as f32) as i32 as i64,
            )
        })
        .collect();
    MaskAlpha::Fill {
        outline,
        sigma: feather_sigma,
    }
}

/// Indices of the tone-limited masks that get a shape plane, in plane order.
pub fn limited_indices(masks: &[LocalMask64]) -> Vec<usize> {
    masks
        .iter()
        .enumerate()
        .filter(|(_, m)| {
            m.enabled && m.key != MaskKey::Off && m.vertices.len() >= min_points(m.shape)
        })
        .map(|(i, _)| i)
        .take(MAX_KEYED_MASKS)
        .collect()
}

/// The tone-limited masks in plane order.
pub fn limited_masks(masks: &[LocalMask64]) -> Vec<&LocalMask64> {
    limited_indices(masks)
        .into_iter()
        .map(|i| &masks[i])
        .collect()
}

/// `compute_local_maps`' output: h x w x planes float32. Planes 0-2 sum the unlimited masks'
/// stops, grade delta and flash; plane 3 + k is the k-th limited mask's alpha.
#[derive(Clone, Debug, PartialEq)]
pub struct LocalMaps {
    pub height: usize,
    pub width: usize,
    pub planes: usize,
    pub data: Vec<f32>,
}

impl LocalMaps {
    pub fn plane(&self, c: usize) -> Vec<f32> {
        let mut out = vec![0.0f32; self.height * self.width];
        negpy_core::convert::extract_channel(&self.data, self.planes, c, &mut out);
        out
    }

    pub fn channel_any(&self, c: usize) -> bool {
        negpy_core::convert::channel_any(&self.data, self.planes, c)
    }

    /// Planes 3.. as an h x w x K array.
    pub fn key_alphas(&self) -> Vec<f32> {
        self.data
            .chunks_exact(self.planes)
            .flat_map(|p| p[3..].iter().copied())
            .collect()
    }
}

/// Per-pixel dodge/burn maps for a render of `h` x `w` from masks drawn on the raw frame.
pub fn compute_local_maps(
    masks: &[LocalMask64],
    h: usize,
    w: usize,
    orig_shape: (usize, usize),
    geo: &GeometryParams,
) -> LocalMaps {
    let limited = limited_indices(masks);
    let planes = 3 + limited.len();
    let mut data = vec![0.0f32; h * w * planes];
    if !masks.is_empty() {
        let short_side = h.min(w) as f64;
        let specs: Vec<shapes::LocalMask> = masks
            .iter()
            .enumerate()
            .filter(|(_, m)| m.enabled && m.vertices.len() >= min_points(m.shape))
            .map(|(i, m)| {
                let transformed: Vec<(f64, f64)> = m
                    .vertices
                    .iter()
                    .map(|&(rx, ry)| map_coords_to_geometry(rx, ry, orig_shape, geo))
                    .collect();
                shapes::LocalMask {
                    alpha: mask_spec(m.shape, &transformed, h, w, m.feather * short_side),
                    invert: m.invert,
                    plane: limited.iter().position(|&k| k == i),
                    stops: m.stops as f32,
                    grade: m.grade as f32,
                    flash: m.flash as f32,
                }
            })
            .collect();
        shapes::local_maps(&specs, (h, w), planes, &mut data).expect("mask specs are well formed");
    }
    LocalMaps {
        height: h,
        width: w,
        planes,
        data,
    }
}

/// Maps a raw-normalized point to the geometry-transformed frame, normalized; not clamped.
pub fn map_coords_to_geometry(
    nx: f64,
    ny: f64,
    orig_shape: (usize, usize),
    geo: &GeometryParams,
) -> (f64, f64) {
    let (px, py, w, h) = pre_keystone_point(nx, ny, orig_shape, geo);
    let (px, py) = if geo.converge_v != 0.0 || geo.converge_h != 0.0 {
        map_point_keystone(px, py, geo.converge_v as f64, geo.converge_h as f64, w, h)
    } else {
        (px, py)
    };
    (px / w.max(1.0), py / h.max(1.0))
}

/// A raw-normalized point in pixel space before keystone correction, with the frame size there.
fn pre_keystone_point(
    nx: f64,
    ny: f64,
    orig_shape: (usize, usize),
    geo: &GeometryParams,
) -> (f64, f64, f64, f64) {
    let (h_orig, w_orig) = (orig_shape.0 as f64, orig_shape.1 as f64);
    let (mut px, mut py) = (nx * w_orig, ny * h_orig);
    let (mut h, mut w) = (h_orig, w_orig);
    match geo.rotation.rem_euclid(4) {
        1 => {
            (px, py) = (py, w - px);
            (h, w) = (w, h);
        }
        2 => (px, py) = (w - px, h - py),
        3 => {
            (px, py) = (h - py, px);
            (h, w) = (w, h);
        }
        _ => {}
    }
    if geo.flip_horizontal {
        px = w - px;
    }
    if geo.flip_vertical {
        py = h - py;
    }
    if geo.fine_rotation != 0.0 {
        let center = ((w / 2.0) as f32, (h / 2.0) as f32);
        let m = negpy_core::warp::rotation_matrix(center, geo.fine_rotation as f64, 1.0);
        (px, py) = (m[0] * px + m[1] * py + m[2], m[3] * px + m[4] * py + m[5]);
    }
    if geo.distortion_k1 != 0.0 {
        (px, py) = map_point_radial(px, py, geo.distortion_k1 as f64, w, h);
    }
    (px, py, w, h)
}

fn radial_center(w: f64, h: f64) -> (f64, f64, f64) {
    ((w - 1.0) * 0.5, (h - 1.0) * 0.5, 0.5 * libm::hypot(w, h))
}

/// Largest scale at which the corrected frame still maps fully inside the source.
pub fn compute_distortion_scale(k1: f64, w: f64, h: f64) -> f64 {
    if k1.abs() < 1e-9 {
        return 1.0;
    }
    let (hw, hh, halfdiag) = radial_center(w, h);
    let inv_hd2 = 1.0 / (halfdiag * halfdiag);
    let edge = 32usize;
    let mut pts = Vec::with_capacity(edge * 4);
    for i in 0..edge {
        let t = i as f64 / edge as f64;
        pts.push((-hw + 2.0 * hw * t, -hh));
        pts.push((-hw + 2.0 * hw * t, hh));
        pts.push((-hw, -hh + 2.0 * hh * t));
        pts.push((hw, -hh + 2.0 * hh * t));
    }
    let max_ratio = |s: f64| -> f64 {
        let mut worst = 0.0f64;
        for &(px, py) in &pts {
            let (pxs, pys) = (px * s, py * s);
            let f = 1.0 + k1 * (pxs * pxs + pys * pys) * inv_hd2;
            if f <= 0.0 {
                return f64::INFINITY;
            }
            worst = worst.max((pxs * f).abs() / hw).max((pys * f).abs() / hh);
        }
        worst
    };
    let (mut lo, mut hi) = (1e-3, 1.0);
    while max_ratio(hi) < 1.0 && hi < 1e3 {
        hi *= 2.0;
    }
    for _ in 0..60 {
        let mid = 0.5 * (lo + hi);
        if max_ratio(mid) < 1.0 {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    0.5 * (lo + hi)
}

/// Where an undistorted point lands in the corrected output. The desktop takes the cubic's
/// root from an eigenvalue solve; Newton from `r_in` reaches the same root within its tolerance.
fn map_point_radial(px: f64, py: f64, k1: f64, w: f64, h: f64) -> (f64, f64) {
    if k1.abs() < DISTORT_EPS {
        return (px, py);
    }
    let (cx, cy, halfdiag) = radial_center(w, h);
    let s = compute_distortion_scale(k1, w, h);
    let (ix, iy) = (px - cx, py - cy);
    let r_in = libm::hypot(ix, iy);
    if r_in < 1e-9 {
        return (px, py);
    }
    let a = k1 / (halfdiag * halfdiag);
    let mut t = r_in;
    for _ in 0..64 {
        let f = a * t * t * t + t - r_in;
        let df = 3.0 * a * t * t + 1.0;
        if df == 0.0 {
            break;
        }
        let next = t - f / df;
        if (next - t).abs() <= 1e-12 * t.abs().max(1.0) {
            t = next;
            break;
        }
        t = next;
    }
    if t.is_nan() || t <= 0.0 {
        t = r_in;
    }
    let scale = (t / s) / r_in;
    (cx + ix * scale, cy + iy * scale)
}

/// The normalized source quad of the keystone correction, as float32.
fn keystone_quad(converge_v: f64, converge_h: f64) -> [[f32; 2]; 4] {
    let (a, b) = (converge_v * 0.005, converge_h * 0.005);
    [[a, b], [1.0 - a, -b], [1.0 + a, 1.0 + b], [-a, 1.0 - b]].map(|p| p.map(|v| v as f32))
}

fn matmul3(a: &[[f64; 3]; 3], b: &[[f64; 3]; 3]) -> [[f64; 3]; 3] {
    std::array::from_fn(|i| {
        std::array::from_fn(|j| a[i][0] * b[0][j] + a[i][1] * b[1][j] + a[i][2] * b[2][j])
    })
}

/// The keystone map in pixel-index coordinates.
fn keystone_matrix(converge_v: f64, converge_h: f64, w: f64, h: f64) -> Option<[[f64; 3]; 3]> {
    let dst = [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]];
    let k = negpy_core::warp::perspective_transform(&keystone_quad(converge_v, converge_h), &dst)?;
    let k = [[k[0], k[1], k[2]], [k[3], k[4], k[5]], [k[6], k[7], k[8]]];
    let a = [[w, 0.0, -0.5], [0.0, h, -0.5], [0.0, 0.0, 1.0]];
    let a_inv = [
        [1.0 / w, 0.0, 0.5 / w],
        [0.0, 1.0 / h, 0.5 / h],
        [0.0, 0.0, 1.0],
    ];
    Some(matmul3(&matmul3(&a, &k), &a_inv))
}

/// Where a pre-correction point lands in the corrected output.
fn map_point_keystone(
    px: f64,
    py: f64,
    converge_v: f64,
    converge_h: f64,
    w: f64,
    h: f64,
) -> (f64, f64) {
    if converge_v.abs() < KEYSTONE_EPS && converge_h.abs() < KEYSTONE_EPS {
        return (px, py);
    }
    let Some(m) = keystone_matrix(converge_v, converge_h, w, h) else {
        return (px, py);
    };
    let den = m[2][0] * px + m[2][1] * py + m[2][2];
    if den.abs() < 1e-12 {
        return (px, py);
    }
    (
        (m[0][0] * px + m[0][1] * py + m[0][2]) / den,
        (m[1][0] * px + m[1][1] * py + m[1][2]) / den,
    )
}
