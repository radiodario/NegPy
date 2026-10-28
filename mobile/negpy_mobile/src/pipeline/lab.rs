//! The Lab stage (`features/lab/processor.py` and `logic.py`): CLAHE, chroma denoise,
//! saturation, sharpening, glow and halation on the linear print, around the `negpy_core`
//! kernels, in the float32 order the desktop evaluates them.

use std::borrow::Cow;

use negpy_core::color::{self, LabSpace};
use negpy_core::lab as core;
use negpy_core::mathf::np_expf;
use negpy_core::stats::pairwise_sum_f32;

use crate::config::{LabConfig, ProcessMode};
use crate::context::PipelineContext;
use crate::image::Image;
use crate::pipeline::working::WORKING_GAMMA;

/// The desktop's float of a config field: the f32 widened through its shortest decimal, the
/// value the user typed, where Python keeps a float64.
pub fn wide(v: f32) -> f64 {
    v.to_string().parse().unwrap_or_else(|_| f64::from(v))
}

// CIELAB in the working space (`kernel/image/logic.py`), on OpenCV's float Lab scale. The
// literals are the desktop's, trailing zeros included.
#[allow(clippy::excessive_precision)]
pub const WORKING_TO_XYZ: [[f32; 3]; 3] = [
    [0.5767309, 0.1855540, 0.1881852],
    [0.2973769, 0.6273491, 0.0752741],
    [0.0270343, 0.0706872, 0.9911085],
];
#[allow(clippy::excessive_precision)]
pub const XYZ_TO_WORKING: [[f32; 3]; 3] = [
    [2.0413690, -0.5649464, -0.3446944],
    [-0.9692660, 1.8760108, 0.0415560],
    [0.0134474, -0.1183897, 1.0154096],
];
pub const WORKING_WHITE: [f32; 3] = [0.95047, 1.00000, 1.08883];
pub const LAB_EPS: f32 = 0.008856;
pub const LAB_KAPPA: f32 = 7.787;
/// `np.float32(1.0 / _WORKING_GAMMA)`.
pub const WORKING_INV_GAMMA: f32 = (1.0 / WORKING_GAMMA as f64) as f32;

/// Adobe RGB (1998) -> XYZ D65 luminance row, as the desktop's Python floats and float32s.
pub const LUM: [f64; 3] = [0.2973769, 0.6273491, 0.0752741];
pub const LUM32: [f32; 3] = [0.2973769, 0.6273491, 0.0752741];

/// `WORKING_LAB` as the kernels take it: (RGB to Lab, Lab to RGB).
pub fn working_lab() -> (LabSpace, LabSpace) {
    (
        LabSpace::new(WORKING_TO_XYZ, WORKING_WHITE, LAB_EPS, LAB_KAPPA),
        LabSpace::new(XYZ_TO_WORKING, WORKING_WHITE, LAB_EPS, LAB_KAPPA),
    )
}

/// `gaussian_kernel_1d(sigma)`: the blur taps in float32, `np.exp` as NumPy's SIMD code
/// rounds it, normalized by NumPy's pairwise sum.
pub fn gaussian_kernel_1d(sigma: f64) -> Vec<f32> {
    let r = ((2.5 * sigma).ceil() as i64).clamp(1, 255);
    let s2 = (2.0 * sigma * sigma) as f32;
    let k: Vec<f32> = (-r..=r)
        .map(|x| {
            let x = x as f32;
            np_expf(-(x * x) / s2)
        })
        .collect();
    let sum = pairwise_sum_f32(&k);
    k.iter().map(|v| v / sum).collect()
}

fn filled(img: &Image, fill: impl FnOnce(&[f32], (usize, usize), &mut [f32])) -> Image {
    let mut out = vec![0.0f32; img.data.len()];
    fill(&img.data, img.shape(), &mut out);
    Image::new(img.height, img.width, out)
}

/// `apply_clahe(img, strength)`: the L* channel equalized per tile; unchanged at strength 0.
pub fn clahe(img: &Image, strength: f32) -> Image {
    if strength <= 0.0 {
        return img.clone();
    }
    filled(img, |src, dims, dst| {
        core::clahe(src, dims, wide(strength), working_lab(), dst)
    })
}

/// `np.clip(img, 0, 1)`.
pub fn clip01(img: &Image) -> Image {
    filled(img, |src, _, dst| negpy_core::convert::clip_unit(src, dst))
}

/// `apply_saturation`: chroma scaled in CIELAB with the gamut knee, then skin protection.
/// None where the desktop returns the input itself.
pub fn saturation(img: &Image, saturation: f32, skin_protection: f32) -> Option<Image> {
    if saturation == 1.0 && skin_protection <= 0.0 {
        return None;
    }
    let (to_lab, to_rgb) = working_lab();
    let n = img.data.len();
    let mut lab = vec![0.0f32; n];
    color::rgb_to_lab(&img.data, &mut lab, to_lab);
    if saturation != 1.0 {
        let mut out = vec![0.0f32; n];
        color::gamut_aware_chroma_scale(&lab, &mut out, saturation, 10, to_rgb);
        lab = out;
    }
    if skin_protection > 0.0 {
        let mut out = vec![0.0f32; n];
        color::skin_chroma_rein(&lab, &mut out, skin_protection);
        lab = out;
    }
    let mut rgb = vec![0.0f32; n];
    color::lab_to_rgb(&lab, &mut rgb, to_rgb);
    let mut out = vec![0.0f32; n];
    negpy_core::convert::clip_unit(&rgb, &mut out);
    Some(Image::new(img.height, img.width, out))
}

/// Auto Masking: the masking that holds back the grain is this gain times the measured grain.
pub const AUTO_MASKING_GAIN: f64 = 0.27;
/// `LabConfig().sharpen_masking`.
pub const NEUTRAL_MASKING: f64 = 0.1;

/// `auto_masking(img, config)`: the masking Auto Masking picks, or None when it is off.
pub fn auto_masking(img: &Image, cfg: &LabConfig) -> Option<f64> {
    if !cfg.sharpen_auto_masking {
        return None;
    }
    Some(match core::sharpen_grain(&img.data, img.shape(), LUM32) {
        None => NEUTRAL_MASKING,
        Some(grain) => (AUTO_MASKING_GAIN * f64::from(grain)).min(1.0),
    })
}

/// `effective_masking(config, auto)`.
pub fn effective_masking(cfg: &LabConfig, auto: Option<f64>) -> f64 {
    let masking = wide(cfg.sharpen_masking);
    match auto {
        None => masking,
        Some(auto) => (auto + masking - NEUTRAL_MASKING).clamp(0.0, 1.0),
    }
}

/// `rl_iterations(radius)`: Python's `round` ties to even.
pub fn rl_iterations(radius: f64) -> usize {
    ((10.0 * radius).round_ties_even() as i64).clamp(5, 20) as usize
}

/// `apply_usm_sharpening(img, amount, radius, masking)`; None at amount 0.
pub fn usm_sharpening(img: &Image, amount: f64, radius: f64, masking: f64) -> Option<Image> {
    if amount <= 0.0 {
        return None;
    }
    let k = gaussian_kernel_1d(radius);
    Some(filled(img, |src, dims, dst| {
        core::usm_sharpen(
            src,
            dims,
            ((amount * 2.5) as f32, masking),
            &k,
            working_lab(),
            dst,
        )
        .expect("the image is too small for the blur")
    }))
}

/// `apply_rl_sharpening(img, amount, radius, masking)`; None at amount 0.
pub fn rl_sharpening(img: &Image, amount: f64, radius: f64, masking: f64) -> Option<Image> {
    if amount <= 0.0 {
        return None;
    }
    let k = gaussian_kernel_1d(radius);
    let params = core::Rl {
        amount: amount as f32,
        masking,
        iterations: rl_iterations(radius),
        lum: LUM32,
        eps: 1e-6,
    };
    Some(filled(img, |src, dims, dst| {
        core::rl_sharpen(src, dims, &params, &k, dst).expect("the image is too small for the blur")
    }))
}

/// `apply_lab_sharpening(img, config, auto)`: the method the config names, at its amount,
/// radius and effective masking.
pub fn sharpening(img: &Image, cfg: &LabConfig, auto: Option<f64>) -> Option<Image> {
    let masking = effective_masking(cfg, auto);
    let (amount, radius) = (wide(cfg.sharpen), wide(cfg.sharpen_radius));
    if cfg.sharpen_method == "rl" {
        rl_sharpening(img, amount, radius, masking)
    } else {
        usm_sharpening(img, amount, radius, masking)
    }
}

/// Below this blur sigma (px) both engines skip the stage.
pub const CHROMA_DENOISE_MIN_SIGMA: f64 = 0.25;

/// `chroma_denoise_sigma(radius, scale_factor)`: 0.0 when the stage does not run.
pub fn chroma_denoise_sigma(radius: f64, scale_factor: f64) -> f64 {
    let sigma = radius * scale_factor;
    if sigma >= CHROMA_DENOISE_MIN_SIGMA {
        sigma
    } else {
        0.0
    }
}

/// `apply_chroma_denoise(img, radius, scale_factor)`; None when the sigma is below the floor.
pub fn chroma_denoise(img: &Image, radius: f64, scale_factor: f64) -> Option<Image> {
    let sigma = chroma_denoise_sigma(radius, scale_factor);
    if sigma == 0.0 {
        return None;
    }
    let k = gaussian_kernel_1d(sigma);
    Some(filled(img, |src, dims, dst| {
        core::chroma_denoise(src, dims, sigma, &k, working_lab(), dst)
            .expect("the image is too small for the blur")
    }))
}

// Base-reflection bounces, (weight, sigma over HALATION_SIGMA).
pub const HALATION_SIGMA: f64 = 0.0027;
pub const HALATION_BOUNCES: [(f64, f64); 3] = [
    (4.0 / 7.0, 1.0),
    (2.0 / 7.0, std::f64::consts::SQRT_2),
    (1.0 / 7.0, 1.732_050_807_568_877_2),
];
pub const HALATION_AMOUNT_MAX: f64 = 0.2;
pub const HALATION_TINT: [f64; 3] = [1.0, 0.3, 0.05];

// Glow: a mist filter's scatter, as (weight, sigma over the short side).
pub const GLOW_PSF: [(f64, f64); 4] = [
    (0.1104, 0.001582),
    (0.3559, 0.004122),
    (0.4014, 0.010181),
    (0.1323, 0.024798),
];
pub const GLOW_AMOUNT_MAX: f64 = 0.25;
pub const GLOW_THRESHOLD: f64 = 0.5;

/// `halation_amounts(strength, bw)`: per-channel mix; neutral, same luminance, on B&W.
pub fn halation_amounts(strength: f64, bw: bool) -> [f32; 3] {
    let [r, g, b] = HALATION_TINT.map(|t| HALATION_AMOUNT_MAX * strength * t);
    if bw {
        let y = r * LUM[0] + g * LUM[1] + b * LUM[2];
        return [y as f32; 3];
    }
    [r as f32, g as f32, b as f32]
}

/// `halation_sigma_px(shape)`.
pub fn halation_sigma_px((h, w): (usize, usize)) -> f64 {
    HALATION_SIGMA * h.min(w) as f64
}

/// `apply_glow_and_halation(img, glow_amount, halation_strength, bw)`; None when both are 0.
pub fn glow_and_halation(
    img: &Image,
    glow_amount: f64,
    halation_strength: f64,
    bw: bool,
) -> Option<Image> {
    if glow_amount == 0.0 && halation_strength == 0.0 {
        return None;
    }
    let short = img.height.min(img.width) as f64;
    let glow = (glow_amount > 0.0).then(|| {
        let a = (GLOW_AMOUNT_MAX * glow_amount) as f32;
        let psf = GLOW_PSF
            .iter()
            .map(|&(w, s)| (w as f32, s * short))
            .collect();
        (
            core::Spread {
                amounts: [a; 3],
                psf,
            },
            GLOW_THRESHOLD,
        )
    });
    let halation = (halation_strength > 0.0).then(|| {
        let sigma = halation_sigma_px(img.shape());
        let psf = HALATION_BOUNCES
            .iter()
            .map(|&(w, k)| (w as f32, k * sigma))
            .collect();
        core::Spread {
            amounts: halation_amounts(halation_strength, bw),
            psf,
        }
    });
    Some(filled(img, |src, dims, dst| {
        core::glow_and_halation(src, dims, (glow, halation), (LUM32, WORKING_INV_GAMMA), dst)
            .expect("a blur failed")
    }))
}

/// `PhotoLabProcessor.process`: the effects in sequence, then the clip to 0..1. Records
/// `auto_masking` in the metrics when Auto Masking is on.
pub fn lab(img: &Image, cfg: &LabConfig, ctx: &mut PipelineContext) -> Image {
    let mut cur = Cow::Borrowed(img);
    if cfg.chroma_denoise > 0.0
        && let Some(out) = chroma_denoise(&cur, wide(cfg.chroma_denoise), wide(ctx.scale_factor))
    {
        cur = Cow::Owned(out);
    }
    if (cfg.saturation != 1.0 || cfg.skin_protection > 0.0)
        && let Some(out) = saturation(&cur, cfg.saturation, cfg.skin_protection)
    {
        cur = Cow::Owned(out);
    }
    if cfg.sharpen > 0.0 {
        let auto = auto_masking(&cur, cfg);
        if let Some(auto) = auto {
            ctx.metrics.auto_masking = Some(auto as f32);
        }
        if let Some(out) = sharpening(&cur, cfg, auto) {
            cur = Cow::Owned(out);
        }
    }
    if (cfg.glow_amount > 0.0 || cfg.halation_strength > 0.0)
        && let Some(out) = glow_and_halation(
            &cur,
            wide(cfg.glow_amount),
            wide(cfg.halation_strength),
            ctx.process_mode == ProcessMode::Bw,
        )
    {
        cur = Cow::Owned(out);
    }
    clip01(&cur)
}
