//! The finish stage (`features/finish/processor.py` and `logic.py`) and the output encode:
//! the vignette, the filed carrier with its paper margin, then the working TRC.

use std::borrow::Cow;
use std::sync::OnceLock;

use negpy_core::finish::{self as core, Carrier};

use crate::config::{FinishConfig, PipelineConfig, ToningConfig};
use crate::context::PipelineContext;
use crate::image::Image;
use crate::pipeline::toning::toned_paper_white;
use crate::pipeline::working;

/// Profile samples per edge.
pub const CARRIER_SAMPLES: usize = 2048;
/// Tone table sampled at t = u**3, dense in the toe.
pub const CARRIER_TONE_SAMPLES: usize = 64;

static CARRIER_PROFILES: OnceLock<Vec<f32>> = OnceLock::new();

/// `carrier_profiles()`: the (8, CARRIER_SAMPLES) profiles of the one filed carrier, as the
/// desktop draws them from its fixed seed; `carrier_profiles.f32` holds them little-endian.
pub fn carrier_profiles() -> &'static [f32] {
    CARRIER_PROFILES.get_or_init(|| {
        include_bytes!("carrier_profiles.f32")
            .as_slice()
            .as_chunks::<4>()
            .0
            .iter()
            .map(|b| f32::from_le_bytes(*b))
            .collect()
    })
}

/// `carrier_width_px(carrier_width_mm, print_size_cm, long_edge_px)`.
pub fn carrier_width_px(carrier_width_mm: f64, print_size_cm: f64, long_edge_px: f64) -> f64 {
    (carrier_width_mm / (print_size_cm * 10.0).max(1.0)) * long_edge_px
}

/// `carrier_tone_exposures()`: exposure fraction of each tone-table entry, 0 to 1. The table
/// is the desktop's `carrier_tone_exposures().tolist()`, NumPy's float32 `linspace ** POWER`.
pub fn carrier_tone_exposures() -> [f32; CARRIER_TONE_SAMPLES] {
    [
        0.0,
        3.9992487e-06,
        3.199399e-05,
        0.00010797971,
        0.00025595192,
        0.0004999061,
        0.00086383766,
        0.0013717421,
        0.0020476154,
        0.0029154525,
        0.0039992486,
        0.005323,
        0.0069107013,
        0.008786349,
        0.010973937,
        0.0134974625,
        0.016380923,
        0.019648308,
        0.02332362,
        0.027430847,
        0.03199399,
        0.037037037,
        0.042584,
        0.048658855,
        0.05528561,
        0.062488254,
        0.07029079,
        0.0787172,
        0.087791495,
        0.09753767,
        0.1079797,
        0.1191416,
        0.13104738,
        0.143721,
        0.15718646,
        0.1714678,
        0.18658896,
        0.20257394,
        0.21944678,
        0.23723143,
        0.2559519,
        0.2756322,
        0.2962963,
        0.31796825,
        0.340672,
        0.3644315,
        0.38927084,
        0.41521397,
        0.44228488,
        0.47050756,
        0.49990603,
        0.5305043,
        0.5623263,
        0.59539604,
        0.6297376,
        0.66537493,
        0.70233196,
        0.7406328,
        0.78030133,
        0.8213616,
        0.8638376,
        0.90775335,
        0.9531328,
        1.0,
    ]
}

/// `np.log10` of `carrier_tone_exposures()` clamped to 1e-6, the ramp `rebate_tone` prints, as
/// NumPy's float32 `log10` rounds it on the desktop (one ulp off `mathf::log10f` on some entries).
pub fn carrier_tone_log_exposures() -> [f32; CARRIER_TONE_SAMPLES] {
    [
        -6.0000005,
        -5.3980217,
        -4.4949317,
        -3.966658,
        -3.5918417,
        -3.3011117,
        -3.063568,
        -2.8627276,
        -2.6887517,
        -2.535294,
        -2.3980217,
        -2.2738438,
        -2.160478,
        -2.0561917,
        -1.9596378,
        -1.8697479,
        -1.7856617,
        -1.7066748,
        -1.632204,
        -1.5617609,
        -1.4949317,
        -1.4313638,
        -1.3707536,
        -1.3128382,
        -1.257388,
        -1.2042017,
        -1.1531017,
        -1.1039304,
        -1.0565476,
        -1.0108277,
        -0.9666579,
        -0.9239366,
        -0.88257164,
        -0.84247977,
        -0.8035849,
        -0.76581746,
        -0.72911406,
        -0.6934165,
        -0.65867084,
        -0.6248278,
        -0.59184164,
        -0.5596701,
        -0.5282738,
        -0.49761626,
        -0.4676636,
        -0.43838412,
        -0.40974814,
        -0.38172805,
        -0.3542979,
        -0.3274334,
        -0.30111164,
        -0.2753111,
        -0.2500116,
        -0.22519407,
        -0.20084038,
        -0.17693357,
        -0.15345758,
        -0.13039708,
        -0.10773765,
        -0.0854656,
        -0.0635679,
        -0.04203214,
        -0.020846581,
        0.0,
    ]
}

/// `linear_carrier_tone()`: plain light, paper to black.
pub fn linear_carrier_tone() -> Vec<[f32; 3]> {
    carrier_tone_exposures()
        .iter()
        .map(|&t| [1.0 - t; 3])
        .collect()
}

/// `apply_vignette(img, stops, size, roundness)`; None at 0 stops.
pub fn apply_vignette(img: &Image, stops: f64, size: f64, roundness: f64) -> Option<Image> {
    if stops == 0.0 {
        return None;
    }
    let mut out = vec![0.0f32; img.data.len()];
    core::apply_vignette(
        &img.data, img.height, img.width, 3, stops, size, roundness, &mut out,
    );
    Some(Image::new(img.height, img.width, out))
}

/// `apply_carrier(img, width_px, rough, flare, corner, paper, tone)`; None at no width.
/// `tone` is `rebate_tone` of the frame, or plain light when None.
pub fn apply_carrier(
    img: &Image,
    width_px: f64,
    (rough, flare, corner): (f64, f64, f64),
    paper: (f32, f32, f32),
    tone: Option<&[[f32; 3]]>,
) -> Option<Image> {
    if width_px <= 0.0 {
        return None;
    }
    let linear;
    let tone = match tone {
        Some(t) => t,
        None => {
            linear = linear_carrier_tone();
            &linear
        }
    };
    assert_eq!(
        tone.len(),
        CARRIER_TONE_SAMPLES,
        "the tone table has 64 rows"
    );
    let c = Carrier {
        width_px,
        rough,
        flare,
        corner,
        paper: [paper.0, paper.1, paper.2],
        tone,
        profiles: carrier_profiles(),
    };
    let mut out = vec![0.0f32; img.data.len()];
    core::apply_carrier(&img.data, img.shape(), &c, &mut out);
    Some(Image::new(img.height, img.width, out))
}

/// `PrintService.effective_paper_linear`: the mat color in scene-linear, for the carrier's
/// paper margin. A malformed hex reads as white.
pub fn effective_paper_linear(finish: &FinishConfig, toning: &ToningConfig) -> (f32, f32, f32) {
    if finish.border_match_paper {
        return toned_paper_white(toning);
    }
    let hex = finish.border_color.trim_start_matches('#');
    let channel = |i: usize| {
        hex.get(i..i + 2)
            .and_then(|s| u8::from_str_radix(s, 16).ok())
            .map_or(1.0, |v| (f64::from(v) / 255.0) as f32)
    };
    let srgb = Image::new(1, 1, vec![channel(0), channel(2), channel(4)]);
    let [r, g, b] = working::oetf_decode(&srgb).pixel(0, 0);
    (r, g, b)
}

/// `FinishProcessor.process` with the config's print size and mat color.
pub fn finish(
    img: &Image,
    cfg: &PipelineConfig,
    tone: Option<&[[f32; 3]]>,
    ctx: &PipelineContext,
) -> Image {
    let f = &cfg.finish;
    let mut cur = Cow::Borrowed(img);
    if f.vignette_stops != 0.0
        && let Some(out) = apply_vignette(
            &cur,
            f.vignette_stops,
            f.vignette_size,
            f.vignette_roundness,
        )
    {
        cur = Cow::Owned(out);
    }
    // The crop tool's uncropped preview has no frame edge for the carrier to file.
    if f.carrier_width > 0.0 && !ctx.crop_preview_full {
        let paper = effective_paper_linear(f, &cfg.toning);
        let long_edge = cur.height.max(cur.width) as f64;
        let width = carrier_width_px(f.carrier_width, cfg.export.export_print_size, long_edge);
        if let Some(out) = apply_carrier(
            &cur,
            width,
            (f.carrier_rough, f.carrier_flare, f.carrier_corner),
            paper,
            tone,
        ) {
            cur = Cow::Owned(out);
        }
    }
    cur.into_owned()
}

/// The output transform: scene-linear to display-encoded code values (`ensure_image` only
/// casts, so the encode is the whole step).
pub fn encode_output(img: &Image) -> Image {
    working::oetf_encode(img)
}
