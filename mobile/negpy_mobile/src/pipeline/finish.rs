//! The finish stage (`features/finish/processor.py` and `logic.py`) and the output encode:
//! the vignette, the filed carrier with its paper margin, then the working TRC.

use std::borrow::Cow;
use std::sync::OnceLock;

use negpy_core::finish::{self as core, Carrier};

use crate::config::{FinishConfig, PipelineConfig, ToningConfig};
use crate::context::PipelineContext;
use crate::image::Image;
use crate::pipeline::lab::wide;
use crate::pipeline::toning::toned_paper_white;
use crate::pipeline::working;

/// Profile samples per edge.
pub const CARRIER_SAMPLES: usize = 2048;
/// Tone table sampled at t = u**POWER, dense in the toe.
pub const CARRIER_TONE_SAMPLES: usize = 64;
pub const CARRIER_TONE_POWER: f32 = 3.0;

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

/// `carrier_tone_exposures()`: exposure fraction of each tone-table entry, 0 to 1. The power
/// is libm's `powf`; NumPy on an AVX-512 desktop runs SVML's, off by an ulp on some entries.
pub fn carrier_tone_exposures() -> [f32; CARRIER_TONE_SAMPLES] {
    let last = CARRIER_TONE_SAMPLES - 1;
    let step = 1.0 / last as f64;
    std::array::from_fn(|i| {
        let u = if i == last {
            1.0
        } else {
            (i as f64 * step) as f32
        };
        libm::powf(u, CARRIER_TONE_POWER)
    })
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
            wide(f.vignette_stops),
            wide(f.vignette_size),
            wide(f.vignette_roundness),
        )
    {
        cur = Cow::Owned(out);
    }
    // The crop tool's uncropped preview has no frame edge for the carrier to file.
    if f.carrier_width > 0.0 && !ctx.crop_preview_full {
        let paper = effective_paper_linear(f, &cfg.toning);
        let long_edge = cur.height.max(cur.width) as f64;
        let width = carrier_width_px(
            wide(f.carrier_width),
            wide(cfg.export.export_print_size),
            long_edge,
        );
        if let Some(out) = apply_carrier(
            &cur,
            width,
            (
                wide(f.carrier_rough),
                wide(f.carrier_flare),
                wide(f.carrier_corner),
            ),
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
