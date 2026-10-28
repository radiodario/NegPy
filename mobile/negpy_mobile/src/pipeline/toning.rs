//! Toning (`features/toning/processor.py` and `logic.py`): the six chemical baths on a B&W
//! print, then the Lab split tone, on the `negpy_core` kernels.

use std::borrow::Cow;

use negpy_core::toning::{self as core, Bath, Toning};

use crate::config::{AltProcess, ProcessMode, ToningConfig};
use crate::context::PipelineContext;
use crate::image::Image;
use crate::pipeline::lab::working_lab;

/// One bath's constants: reference density, exponent, per-channel covering-power gain. The
/// gains are float32 as the desktop passes them.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BathConstants {
    pub d_ref: f64,
    pub power: f64,
    pub gain: [f32; 3],
}

/// `TONING_CONSTANTS`, with `LITH_TONING_CONSTANTS` over it on a lith print.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ToningConstants {
    pub selenium: BathConstants,
    pub sepia: BathConstants,
    pub gold: BathConstants,
    pub gold_sepia_gain: [f32; 3],
    pub blue: BathConstants,
    pub copper: BathConstants,
    pub vanadium: BathConstants,
}

pub const TONING_CONSTANTS: ToningConstants = ToningConstants {
    selenium: BathConstants {
        d_ref: 2.0,
        power: 1.5,
        gain: [1.04, 1.10, 1.02],
    },
    sepia: BathConstants {
        d_ref: 1.8,
        power: 2.0,
        gain: [0.82, 0.94, 1.12],
    },
    gold: BathConstants {
        d_ref: 1.6,
        power: 1.5,
        gain: [1.08, 1.03, 1.00],
    },
    gold_sepia_gain: [0.80, 0.95, 1.20],
    blue: BathConstants {
        d_ref: 0.9,
        power: 0.85,
        gain: [1.30, 1.00, 0.80],
    },
    copper: BathConstants {
        d_ref: 0.9,
        power: 0.6,
        gain: [0.72, 0.94, 1.18],
    },
    vanadium: BathConstants {
        d_ref: 1.8,
        power: 1.2,
        gain: [1.12, 0.85, 1.03],
    },
};

/// Lith silver moves further in selenium and gold; the other baths keep their constants.
pub const LITH_TONING_CONSTANTS: ToningConstants = ToningConstants {
    selenium: BathConstants {
        d_ref: 1.2,
        power: 1.0,
        gain: [1.10, 1.24, 1.12],
    },
    gold: BathConstants {
        gain: [1.16, 1.06, 0.96],
        ..TONING_CONSTANTS.gold
    },
    ..TONING_CONSTANTS
};

fn bath(strength: f64, c: BathConstants) -> Bath {
    Bath {
        strength,
        d_ref: c.d_ref,
        power: c.power,
        gain: c.gain.map(f64::from),
    }
}

/// `apply_chemical_toning` with the config's six strengths; None when every bath is off.
pub fn apply_chemical_toning(img: &Image, cfg: &ToningConfig, lith_active: bool) -> Option<Image> {
    let strengths = [
        cfg.selenium_strength,
        cfg.sepia_strength,
        cfg.gold_strength,
        cfg.blue_strength,
        cfg.copper_strength,
        cfg.vanadium_strength,
    ];
    if strengths.iter().all(|&s| s == 0.0) {
        return None;
    }
    let [sel, sep, gold, blue, copper, van] = strengths;
    let c = if lith_active {
        LITH_TONING_CONSTANTS
    } else {
        TONING_CONSTANTS
    };
    let t = Toning {
        selenium: bath(sel, c.selenium),
        sepia: bath(sep, c.sepia),
        gold: bath(gold, c.gold),
        gold_sepia_gain: c.gold_sepia_gain.map(f64::from),
        blue: bath(blue, c.blue),
        copper: bath(copper, c.copper),
        vanadium: bath(van, c.vanadium),
        gold_flat: lith_active,
    };
    let mut out = vec![0.0f32; img.data.len()];
    core::apply_chemical_toning(&img.data, &mut out, &t);
    Some(Image::new(img.height, img.width, out))
}

/// `apply_split_toning(img, shadow_hue, shadow_strength, highlight_hue, highlight_strength)`:
/// a* and b* pushed toward each hue by its strength under a smoothstep of L*. None when both
/// strengths are 0.
pub fn apply_split_toning(
    img: &Image,
    (shadow_hue, shadow_strength): (f64, f64),
    (highlight_hue, highlight_strength): (f64, f64),
) -> Option<Image> {
    if shadow_strength == 0.0 && highlight_strength == 0.0 {
        return None;
    }
    let tint = |hue: f64, strength: f64| {
        (strength > 0.0).then(|| {
            let rad = hue.to_radians();
            [
                libm::cos(rad) * 20.0 * strength,
                libm::sin(rad) * 20.0 * strength,
            ]
        })
    };
    let tints = (
        tint(shadow_hue, shadow_strength),
        tint(highlight_hue, highlight_strength),
    );
    let mut out = vec![0.0f32; img.data.len()];
    negpy_core::lab::split_toning(&img.data, tints, working_lab(), &mut out);
    Some(Image::new(img.height, img.width, out))
}

/// The config's split tone.
pub fn split_toning(img: &Image, cfg: &ToningConfig) -> Option<Image> {
    apply_split_toning(
        img,
        (cfg.shadow_tint_hue, cfg.shadow_tint_strength),
        (cfg.highlight_tint_hue, cfg.highlight_tint_strength),
    )
}

/// `ToningProcessor.process`: the baths on a B&W print that holds silver (not a cyanotype),
/// then the split tone on any print. None where the desktop returns the input itself.
pub fn toning_stage(
    img: &Image,
    cfg: &ToningConfig,
    alt_process: AltProcess,
    ctx: &PipelineContext,
) -> Option<Image> {
    let mut cur = Cow::Borrowed(img);
    if ctx.process_mode == ProcessMode::Bw
        && alt_process != AltProcess::Cyanotype
        && let Some(out) = apply_chemical_toning(&cur, cfg, alt_process == AltProcess::Lith)
    {
        cur = Cow::Owned(out);
    }
    if let Some(out) = split_toning(&cur, cfg) {
        cur = Cow::Owned(out);
    }
    match cur {
        Cow::Borrowed(_) => None,
        Cow::Owned(out) => Some(out),
    }
}

/// `PrintService._toned_paper_white`: paper white through the baths and the split tone.
pub fn toned_paper_white(cfg: &ToningConfig) -> (f32, f32, f32) {
    let white = Image::new(1, 1, vec![1.0; 3]);
    let tinted = apply_chemical_toning(&white, cfg, false).unwrap_or_else(|| white.clone());
    let tinted = split_toning(&tinted, cfg).unwrap_or(tinted);
    let [r, g, b] = tinted.pixel(0, 0);
    (r, g, b)
}
