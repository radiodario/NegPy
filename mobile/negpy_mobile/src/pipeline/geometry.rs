//! The geometry stage (`features/geometry/processor.py` and the warps of `geometry/logic.py`):
//! quarter turns, flips, fine rotation, lens distortion and keystone on the frame, the crop ROI
//! into the context, and the final crop. The detection of an auto crop runs before the engine;
//! the rect is already in `crop_rect` here.

use std::borrow::Cow;

use negpy_core::warp::{self, Border, Interp};

use crate::config::GeometryConfig;
use crate::context::{GeometryParams, PipelineContext};
use crate::image::{Image, Roi};

const DISTORT_EPS: f64 = 1e-6;
/// Per-cent of the frame.
const KEYSTONE_EPS: f64 = 1e-4;
const DISTORTION_SAMPLES: usize = 128;
const CROP_TO_VALID_ITERS: usize = 30;
const CROP_TO_VALID_TERNARY_ITERS: usize = 25;

const BILINEAR_REPLICATE: (Interp, Border) = (Interp::Linear, Border::Replicate);

type Mat3 = [f64; 9];
const IDENTITY: Mat3 = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];

fn warp_image(img: &Image) -> warp::Image<'_> {
    warp::Image {
        data: &img.data,
        h: img.height,
        w: img.width,
        cn: 3,
    }
}

/// `np.rot90(img, k)`: `k` counter-clockwise quarter turns.
pub fn rot90(img: &Image, k: i32) -> Image {
    let (h, w) = img.shape();
    let src = |y: usize, x: usize| &img.data[(y * w + x) * 3..(y * w + x) * 3 + 3];
    match k.rem_euclid(4) {
        0 => img.clone(),
        1 => {
            let mut out = Vec::with_capacity(img.data.len());
            for i in 0..w {
                for j in 0..h {
                    out.extend_from_slice(src(j, w - 1 - i));
                }
            }
            Image::new(w, h, out)
        }
        2 => {
            let mut out = Vec::with_capacity(img.data.len());
            for i in 0..h {
                for j in 0..w {
                    out.extend_from_slice(src(h - 1 - i, w - 1 - j));
                }
            }
            Image::new(h, w, out)
        }
        _ => {
            let mut out = Vec::with_capacity(img.data.len());
            for i in 0..w {
                for j in 0..h {
                    out.extend_from_slice(src(h - 1 - j, i));
                }
            }
            Image::new(w, h, out)
        }
    }
}

/// `np.fliplr`: mirrors each row.
pub fn fliplr(img: &Image) -> Image {
    let (h, w) = img.shape();
    let mut out = Vec::with_capacity(img.data.len());
    for y in 0..h {
        let row = &img.data[y * w * 3..(y + 1) * w * 3];
        for px in row.as_chunks::<3>().0.iter().rev() {
            out.extend_from_slice(px);
        }
    }
    Image::new(h, w, out)
}

/// `np.flipud`: reverses the rows.
pub fn flipud(img: &Image) -> Image {
    let (h, w) = img.shape();
    let mut out = Vec::with_capacity(img.data.len());
    for row in img.data.chunks_exact(w * 3).rev() {
        out.extend_from_slice(row);
    }
    Image::new(h, w, out)
}

/// Sub-degree rotation about the frame center, bilinear with replicated edges.
pub fn apply_fine_rotation(img: &Image, angle: f64) -> Image {
    if angle == 0.0 {
        return img.clone();
    }
    let (h, w) = img.shape();
    let center = ((w as f64 / 2.0) as f32, (h as f64 / 2.0) as f32);
    let m = warp::rotation_matrix(center, angle, 1.0);
    let mut out = vec![0.0f32; img.data.len()];
    warp::warp_affine(&warp_image(img), BILINEAR_REPLICATE, m, false, w, &mut out);
    Image::new(h, w, out)
}

/// The center in pixel-index convention and the half-diagonal, the radius unit.
fn radial_center(w: usize, h: usize) -> (f64, f64, f64) {
    let (wf, hf) = (w as f64, h as f64);
    // `math.hypot` of two whole numbers is the correctly rounded root of their exact sum.
    let halfdiag = 0.5 * (wf * wf + hf * hf).sqrt();
    ((wf - 1.0) * 0.5, (hf - 1.0) * 0.5, halfdiag)
}

/// Largest scale at which the corrected frame still maps fully inside the source, found by
/// bisection, so a barrel and a pincushion bind alike.
pub fn compute_distortion_scale(k1: f64, w: usize, h: usize) -> f64 {
    if k1.abs() < 1e-9 {
        return 1.0;
    }
    let (hw, hh, halfdiag) = radial_center(w, h);
    let inv_hd2 = 1.0 / (halfdiag * halfdiag);

    let edge = (DISTORTION_SAMPLES / 4).max(1);
    let mut pts = Vec::with_capacity(4 * edge);
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

    let (mut lo, mut hi) = (1e-3f64, 1.0f64);
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

/// Radial lens-distortion correction (Brown-Conrady k1), scaled to fill the frame.
pub fn apply_radial_distortion(img: &Image, k1: f64) -> Image {
    if k1.abs() < DISTORT_EPS {
        return img.clone();
    }
    let (h, w) = img.shape();
    let (cx, cy, halfdiag) = radial_center(w, h);
    let s = compute_distortion_scale(k1, w, h);
    let inv_hd2 = 1.0 / (halfdiag * halfdiag);
    let mut out = vec![0.0f32; img.data.len()];
    warp::remap_radial(
        &warp_image(img),
        (cx as f32, cy as f32),
        s as f32,
        k1 as f32,
        inv_hd2 as f32,
        &mut out,
    );
    Image::new(h, w, out)
}

/// The normalized source quad of the keystone correction, in float32 as the desktop builds it.
pub fn keystone_quad(converge_v: f64, converge_h: f64) -> [[f32; 2]; 4] {
    let (a, b) = (converge_v * 0.005, converge_h * 0.005);
    [
        [a as f32, b as f32],
        [(1.0 - a) as f32, (-b) as f32],
        [(1.0 + a) as f32, (1.0 + b) as f32],
        [(-a) as f32, (1.0 - b) as f32],
    ]
}

/// Forward map (source to corrected) in normalized coordinates, `u = (index + 0.5) / dims`.
pub fn keystone_matrix_normalized(converge_v: f64, converge_h: f64) -> Mat3 {
    let src = keystone_quad(converge_v, converge_h);
    let dst = [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]];
    warp::perspective_transform(&src, &dst).expect("the keystone quad is convex")
}

/// Normalized [0, 1] to pixel-index coordinates, where pixel i sits at index i.
fn norm_to_index(w: usize, h: usize) -> Mat3 {
    [w as f64, 0.0, -0.5, 0.0, h as f64, -0.5, 0.0, 0.0, 1.0]
}

/// The same map in pixel-index coordinates, for `warp_perspective`.
pub fn keystone_matrix(converge_v: f64, converge_h: f64, w: usize, h: usize) -> Mat3 {
    let a = norm_to_index(w, h);
    // `np.linalg.inv` back-substitutes the upper-triangular index map to these values exactly.
    let (wf, hf) = (w as f64, h as f64);
    let a_inv = [
        1.0 / wf,
        0.0,
        0.5 / wf,
        0.0,
        1.0 / hf,
        0.5 / hf,
        0.0,
        0.0,
        1.0,
    ];
    matmul3(
        &matmul3(&a, &keystone_matrix_normalized(converge_v, converge_h)),
        &a_inv,
    )
}

/// Perspective correction on the same canvas with replicated edges: the wedge stays for the
/// user to crop, as fine rotation leaves it.
pub fn apply_keystone(img: &Image, converge_v: f64, converge_h: f64) -> Image {
    if converge_v.abs() < KEYSTONE_EPS && converge_h.abs() < KEYSTONE_EPS {
        return img.clone();
    }
    let (h, w) = img.shape();
    let m = keystone_matrix(converge_v, converge_h, w, h);
    let mut out = vec![0.0f32; img.data.len()];
    warp::warp_perspective(
        &warp_image(img),
        BILINEAR_REPLICATE,
        m,
        false,
        (w, h),
        &mut out,
    );
    Image::new(h, w, out)
}

fn matmul3(a: &Mat3, b: &Mat3) -> Mat3 {
    std::array::from_fn(|i| {
        let (r, c) = (i / 3, i % 3);
        a[r * 3] * b[c] + a[r * 3 + 1] * b[3 + c] + a[r * 3 + 2] * b[6 + c]
    })
}

/// `np.linalg.inv` of a 3x3: LU with partial pivoting, then the columns of the identity.
fn inv3(m: &Mat3) -> Option<Mat3> {
    let mut lu = [[0.0f64; 3]; 3];
    for (r, row) in lu.iter_mut().enumerate() {
        row.copy_from_slice(&m[r * 3..r * 3 + 3]);
    }
    let mut perm = [0usize, 1, 2];
    for j in 0..3 {
        // The first row of the largest pivot, as `idamax` picks it.
        let mut p = j;
        for i in j + 1..3 {
            if lu[i][j].abs() > lu[p][j].abs() {
                p = i;
            }
        }
        if lu[p][j] == 0.0 {
            return None;
        }
        lu.swap(j, p);
        perm.swap(j, p);
        let inv_pivot = 1.0 / lu[j][j];
        for row in lu.iter_mut().skip(j + 1) {
            row[j] *= inv_pivot;
        }
        for i in j + 1..3 {
            for k in j + 1..3 {
                lu[i][k] -= lu[i][j] * lu[j][k];
            }
        }
    }
    let mut out = [0.0f64; 9];
    for col in 0..3 {
        let mut b = [0.0f64; 3];
        for (i, &p) in perm.iter().enumerate() {
            b[i] = if p == col { 1.0 } else { 0.0 };
        }
        for k in 0..3 {
            if b[k] != 0.0 {
                for i in k + 1..3 {
                    b[i] -= b[k] * lu[i][k];
                }
            }
        }
        for k in (0..3).rev() {
            if b[k] != 0.0 {
                b[k] /= lu[k][k];
                for i in 0..k {
                    b[i] -= b[k] * lu[i][k];
                }
            }
        }
        for i in 0..3 {
            out[i * 3 + col] = b[i];
        }
    }
    Some(out)
}

/// Largest axis-aligned, frame-centered rect, normalized to the frame, that avoids the
/// replicated edges fine rotation and keystone leave behind. The forward-mapped frame is a
/// convex quad, so a rect is valid when its four corners are; the area at a fixed half-width
/// is unimodal, so a ternary search on it nests a bisection on the half-height.
pub fn compute_geometry_crop_rect(
    fine_rotation: f64,
    converge_v: f64,
    converge_h: f64,
    w: usize,
    h: usize,
) -> (f64, f64, f64, f64) {
    if fine_rotation.abs() < 1e-9
        && converge_v.abs() < KEYSTONE_EPS
        && converge_h.abs() < KEYSTONE_EPS
    {
        return (0.0, 0.0, 1.0, 1.0);
    }
    let (wf, hf) = (w as f64, h as f64);
    let (cx, cy) = (wf / 2.0, hf / 2.0);
    let rot = warp::rotation_matrix((cx as f32, cy as f32), fine_rotation, 1.0);
    let m_rot = [
        rot[0], rot[1], rot[2], rot[3], rot[4], rot[5], 0.0, 0.0, 1.0,
    ];
    let m_combined = matmul3(&keystone_matrix(converge_v, converge_h, w, h), &m_rot);
    let m_inv = inv3(&m_combined).unwrap_or(IDENTITY);

    let source_valid = |x: f64, y: f64| -> bool {
        let p: [f64; 3] =
            std::array::from_fn(|r| m_inv[r * 3] * x + m_inv[r * 3 + 1] * y + m_inv[r * 3 + 2]);
        if p[2].abs() < 1e-12 {
            return false;
        }
        let (sx, sy) = (p[0] / p[2], p[1] / p[2]);
        (-1e-6..=(wf - 1.0) + 1e-6).contains(&sx) && (-1e-6..=(hf - 1.0) + 1e-6).contains(&sy)
    };
    let corners_valid = |hw: f64, hh: f64| -> bool {
        [-hw, hw]
            .iter()
            .all(|dx| [-hh, hh].iter().all(|dy| source_valid(cx + dx, cy + dy)))
    };

    let (max_hw, max_hh) = (cx, cy);
    let hh_max = |hw: f64| -> f64 {
        let (mut lo, mut hi) = (0.0f64, max_hh);
        for _ in 0..CROP_TO_VALID_ITERS {
            let mid = 0.5 * (lo + hi);
            if corners_valid(hw, mid) {
                lo = mid;
            } else {
                hi = mid;
            }
        }
        lo
    };

    let (mut lo_w, mut hi_w) = (0.0f64, max_hw);
    for _ in 0..CROP_TO_VALID_TERNARY_ITERS {
        let m1 = lo_w + (hi_w - lo_w) / 3.0;
        let m2 = hi_w - (hi_w - lo_w) / 3.0;
        if m1 * hh_max(m1) < m2 * hh_max(m2) {
            lo_w = m1;
        } else {
            hi_w = m2;
        }
    }
    let hw = 0.5 * (lo_w + hi_w);
    let hh = hh_max(hw);
    (
        (cx - hw) / wf,
        (cy - hh) / hf,
        (cx + hw) / wf,
        (cy + hh) / hf,
    )
}

/// `apply_margin_to_roi` on a ROI that may still carry negative corners (an off-frame rect).
fn inset_roi((y1, y2, x1, x2): (i64, i64, i64, i64), h: usize, w: usize, margin_px: f64) -> Roi {
    // An inset larger than half the crop would turn the ROI inside out.
    let margin = margin_px
        .min((y2 - y1 - 1) as f64 / 2.0)
        .min((x2 - x1 - 1) as f64 / 2.0);
    let (y1, y2, x1, x2) = (y1 as f64, y2 as f64, x1 as f64, x2 as f64);
    // `int()` truncates toward zero, as `as i64` does.
    let clamp = |v: f64| (v as i64).max(0) as usize;
    (
        clamp((y1 + margin).max(0.0)),
        clamp((y2 - margin).min(h as f64)),
        clamp((x1 + margin).max(0.0)),
        clamp((x2 - margin).min(w as f64)),
    )
}

/// Insets (or, negative, grows) a ROI by `margin_px`, clamped to the frame.
pub fn apply_margin_to_roi(roi: Roi, h: usize, w: usize, margin_px: f64) -> Roi {
    let (y1, y2, x1, x2) = roi;
    inset_roi(
        (y1 as i64, y2 as i64, x1 as i64, x2 as i64),
        h,
        w,
        margin_px,
    )
}

/// Maps a normalized rect `(x1, y1, x2, y2)`, drawn on the transformed frame, to a pixel ROI
/// of that frame, inset by `offset_px` scaled to the buffer.
pub fn get_manual_rect_coords(
    (h, w): (usize, usize),
    manual_rect: (f64, f64, f64, f64),
    offset_px: i32,
    scale_factor: f64,
) -> Roi {
    let (x1_n, y1_n, x2_n, y2_n) = manual_rect;
    let (wf, hf) = (w as f64, h as f64);
    let xs = (x1_n * wf, x2_n * wf);
    let ys = (y1_n * hf, y2_n * hf);
    let (ix1, ix2) = (xs.0.min(xs.1) as i64, xs.0.max(xs.1) as i64);
    let (iy1, iy2) = (ys.0.min(ys.1) as i64, ys.0.max(ys.1) as i64);
    let margin = offset_px as f64 * scale_factor;
    inset_roi((iy1, iy2, ix1, ix2), h, w, margin)
}

/// The forward chain the stage and `analysis_grid` both replay: quarter turns, flips, fine
/// rotation, distortion, then keystone, which needs a frame free of barrel distortion.
pub fn apply_geometry(img: &Image, p: &GeometryParams) -> Image {
    let mut cur = Cow::Borrowed(img);
    if p.rotation != 0 {
        cur = Cow::Owned(rot90(&cur, p.rotation));
    }
    if p.flip_horizontal {
        cur = Cow::Owned(fliplr(&cur));
    }
    if p.flip_vertical {
        cur = Cow::Owned(flipud(&cur));
    }
    if p.fine_rotation != 0.0 {
        cur = Cow::Owned(apply_fine_rotation(&cur, p.fine_rotation));
    }
    if p.distortion_k1 != 0.0 {
        cur = Cow::Owned(apply_radial_distortion(&cur, p.distortion_k1));
    }
    apply_keystone(&cur, p.converge_v, p.converge_h)
}

/// The geometry the config asks for, as the context records it.
pub fn geometry_params(cfg: &GeometryConfig) -> GeometryParams {
    GeometryParams {
        rotation: cfg.rotation,
        fine_rotation: cfg.fine_rotation,
        flip_horizontal: cfg.flip_horizontal,
        flip_vertical: cfg.flip_vertical,
        converge_v: cfg.converge_v,
        converge_h: cfg.converge_h,
        distortion_k1: cfg.distortion_k1,
    }
}

/// `GeometryProcessor.process`: warps the frame and sets the crop ROI on the context.
pub fn geometry(img: &Image, cfg: &GeometryConfig, ctx: &mut PipelineContext) -> Image {
    let params = geometry_params(cfg);
    let out = apply_geometry(img, &params);
    ctx.metrics.geometry = params;

    let (h, w) = out.shape();
    let scale_factor = ctx.scale_factor;
    ctx.active_roi = if let Some(rect) = cfg.crop_rect {
        Some(get_manual_rect_coords(
            (h, w),
            rect,
            cfg.autocrop_offset,
            scale_factor,
        ))
    } else if cfg.crop_to_valid && !cfg.crop_from_auto {
        let valid =
            compute_geometry_crop_rect(cfg.fine_rotation, cfg.converge_v, cfg.converge_h, w, h);
        Some(get_manual_rect_coords(
            (h, w),
            valid,
            cfg.autocrop_offset,
            scale_factor,
        ))
    } else if cfg.autocrop_offset > 0 {
        let margin = cfg.autocrop_offset as f64 * scale_factor;
        Some(apply_margin_to_roi((0, h, 0, w), h, w, margin))
    } else {
        None
    };
    ctx.metrics.active_roi = ctx.active_roi;
    out
}

/// `CropProcessor.process`: the frame inside the context's ROI, or the whole frame.
pub fn crop(img: &Image, ctx: &PipelineContext) -> Image {
    match ctx.active_roi {
        Some(roi) => img.crop(roi),
        None => img.clone(),
    }
}
