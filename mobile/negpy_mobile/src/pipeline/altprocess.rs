//! The alternative processes (`features/lith`, `features/cyanotype`, `features/sabattier`)
//! and `print_stages` of `engine.py`: each process gates on the B&W mode and the
//! `alt_process` choice, then toning follows.

use std::borrow::Cow;

use negpy_core::altprocess::{self as core, Cyanotype, Interp, Lith, Sabattier};

use crate::config::{AltProcess, AltProcessConfig, PipelineConfig, ProcessMode};
use crate::context::PipelineContext;
use crate::image::Image;
use crate::pipeline::lab::{gaussian_kernel_1d, wide, working_lab};
use crate::pipeline::papers::{
    LithPath, PaperProfile, effective_constants, effective_paper_profile,
};
use crate::pipeline::toning::toning_stage;

// `LITH_CONSTANTS`.
pub const LITH_FOOT_MAX: f64 = 0.70;
pub const LITH_FOOT_RATE: f64 = 0.60;
pub const LITH_FOOT_VEIL: f64 = 0.60;
pub const LITH_KNEE_LO: f64 = 1.25;
pub const LITH_KNEE_SPAN: f64 = 0.70;
pub const LITH_ABRUPT_LO: f64 = 0.30;
pub const LITH_ABRUPT_SPAN: f64 = 0.27;
pub const LITH_PATH_U: [f32; 4] = [0.10, 0.35, 0.65, 1.00];

// `CYANOTYPE_CONSTANTS`.
pub const CYANO_MID_COMPRESS: f64 = 0.45;
pub const CYANO_BLEACH_FLOOR: f64 = 0.15;
pub const CYANO_TANNIN_RESTORE: f64 = 1.05;
pub const CYANO_TANNIN_DMAX_GAIN: f64 = 0.15;
pub const CYANO_BROWN_DIR: [f32; 2] = [9.0, 20.0];
pub const CYANO_PATH_U: [f32; 4] = [0.00, 0.15, 0.55, 1.00];

// `SABATTIER_CONSTANTS`.
pub const SABATTIER_FOLD_WIDTH: f64 = 0.08;
pub const SABATTIER_EDGE_WIDTH: f64 = 0.10;
pub const SABATTIER_STILL_REACH: f64 = 1.0;

/// A cyanotype sensitizer's Dmax and (a*, b*) hue path (`SENSITIZERS`).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SensitizerConstants {
    pub d_max: f64,
    pub path: [(f32, f32); 4],
}

pub const SENSITIZER_CLASSIC: SensitizerConstants = SensitizerConstants {
    d_max: 0.95,
    path: [(0.5, 3.0), (-4.0, 6.0), (-7.0, -17.0), (-6.0, -25.0)],
};
pub const SENSITIZER_NEW: SensitizerConstants = SensitizerConstants {
    d_max: 1.40,
    path: [(0.3, 2.0), (-2.0, 2.0), (-9.0, -24.0), (-8.0, -34.0)],
};

/// `sensitizer_constants(sensitizer)`; an unknown name reads as classic.
pub fn sensitizer_constants(sensitizer: &str) -> SensitizerConstants {
    if sensitizer == "new" {
        SENSITIZER_NEW
    } else {
        SENSITIZER_CLASSIC
    }
}

fn path(xp: [f32; 4], anchors: [(f32, f32); 4]) -> (Interp, Interp) {
    (
        Interp::new(xp, anchors.map(|a| a.0)),
        Interp::new(xp, anchors.map(|a| a.1)),
    )
}

fn filled(img: &Image, fill: impl FnOnce(&[f32], (usize, usize), &mut [f32])) -> Image {
    let mut out = vec![0.0f32; img.data.len()];
    fill(&img.data, img.shape(), &mut out);
    Image::new(img.height, img.width, out)
}

/// `apply_lith(img, path, d_max, enabled, exposure, snatch, abruptness)`; None when off.
pub fn apply_lith(
    img: &Image,
    path_ab: &LithPath,
    d_max: f64,
    enabled: bool,
    (exposure, snatch, abruptness): (f64, f64, f64),
) -> Option<Image> {
    if !enabled {
        return None;
    }
    let anchors = path_ab.map(|(a, b)| (a as f32, b as f32));
    let (a_path, b_path) = path(LITH_PATH_U, anchors);
    let (to_lab, to_rgb) = working_lab();
    let c = Lith {
        d_max: d_max as f32,
        over: (0.301 * exposure) as f32,
        foot_max: LITH_FOOT_MAX as f32,
        foot_veil: LITH_FOOT_VEIL as f32,
        foot_rate: LITH_FOOT_RATE as f32,
        knee: (d_max * (LITH_KNEE_LO - LITH_KNEE_SPAN * snatch)) as f32,
        width: (LITH_ABRUPT_LO - LITH_ABRUPT_SPAN * abruptness).max(0.01) as f32,
        a_path,
        b_path,
        lab_eps: to_lab.eps as f32,
        lab_kappa: to_lab.kappa as f32,
    };
    Some(filled(img, |src, _, dst| core::lith(src, &c, to_rgb, dst)))
}

/// `apply_cyanotype(img, sensitizer, enabled, exposure, scale, bleach, tannin)`; None when off.
pub fn apply_cyanotype(
    img: &Image,
    sensitizer: &str,
    enabled: bool,
    (exposure, scale, bleach, tannin): (f64, f64, f64, f64),
) -> Option<Image> {
    if !enabled {
        return None;
    }
    let sc = sensitizer_constants(sensitizer);
    let (a_path, b_path) = path(CYANO_PATH_U, sc.path);
    let (to_lab, to_rgb) = working_lab();
    let c = Cyanotype {
        d_max: sc.d_max as f32,
        over: (0.301 * exposure) as f32,
        scale: scale.max(0.1) as f32,
        mid_compress: CYANO_MID_COMPRESS as f32,
        bleach: bleach as f32,
        bleach_floor: CYANO_BLEACH_FLOOR as f32,
        tannin: tannin as f32,
        tannin_restore: CYANO_TANNIN_RESTORE as f32,
        tannin_dmax_gain: CYANO_TANNIN_DMAX_GAIN as f32,
        brown: CYANO_BROWN_DIR,
        a_path,
        b_path,
        lab_eps: to_lab.eps as f32,
        lab_kappa: to_lab.kappa as f32,
    };
    Some(filled(img, |src, _, dst| {
        core::cyanotype(src, &c, to_rgb, dst)
    }))
}

/// `line_sigma_px(agitation, shape)`: the Mackie line's blur sigma in render pixels.
pub fn line_sigma_px(agitation: f64, (h, w): (usize, usize)) -> f64 {
    let reach = SABATTIER_STILL_REACH * (1.0 - agitation).clamp(0.0, 1.0);
    reach * 0.01 * h.min(w) as f64
}

/// `apply_sabattier(img, d_max, enabled, strength, reexposure, line_sigma_px)`; None when off.
pub fn apply_sabattier(
    img: &Image,
    d_max: f64,
    enabled: bool,
    (strength, reexposure, line_sigma_px): (f64, f64, f64),
) -> Option<Image> {
    if !enabled {
        return None;
    }
    let s = Sabattier {
        re: (reexposure * d_max) as f32,
        fold_width: SABATTIER_FOLD_WIDTH as f32,
        edge_width: SABATTIER_EDGE_WIDTH as f32,
        strength: strength as f32,
        d_max: d_max as f32,
    };
    let k = if line_sigma_px > 0.0 {
        gaussian_kernel_1d(line_sigma_px)
    } else {
        Vec::new()
    };
    Some(filled(img, |src, dims, dst| {
        core::sabattier(src, dims, &s, &k, dst).expect("the image is too small for the blur")
    }))
}

fn paper_d_max(paper: &PaperProfile) -> f64 {
    effective_constants(Some(paper)).d_max
}

/// `LithProcessor.process`; None where the desktop returns the input itself.
pub fn lith_stage(
    img: &Image,
    cfg: &AltProcessConfig,
    paper: &PaperProfile,
    ctx: &PipelineContext,
) -> Option<Image> {
    if ctx.process_mode != ProcessMode::Bw {
        return None;
    }
    apply_lith(
        img,
        &paper.lith_path,
        paper_d_max(paper),
        cfg.alt_process == AltProcess::Lith,
        (
            wide(cfg.lith_exposure),
            wide(cfg.lith_snatch),
            wide(cfg.lith_abruptness),
        ),
    )
}

/// `CyanotypeProcessor.process`; None where the desktop returns the input itself.
pub fn cyanotype_stage(
    img: &Image,
    cfg: &AltProcessConfig,
    ctx: &PipelineContext,
) -> Option<Image> {
    if ctx.process_mode != ProcessMode::Bw {
        return None;
    }
    apply_cyanotype(
        img,
        &cfg.cyano_sensitizer,
        cfg.alt_process == AltProcess::Cyanotype,
        (
            wide(cfg.cyano_exposure),
            wide(cfg.cyano_scale),
            wide(cfg.cyano_bleach),
            wide(cfg.cyano_tannin),
        ),
    )
}

/// `SabattierProcessor.process`; None where the desktop returns the input itself.
pub fn sabattier_stage(
    img: &Image,
    cfg: &AltProcessConfig,
    paper: &PaperProfile,
    ctx: &PipelineContext,
) -> Option<Image> {
    if ctx.process_mode != ProcessMode::Bw {
        return None;
    }
    apply_sabattier(
        img,
        paper_d_max(paper),
        cfg.alt_process == AltProcess::Sabattier,
        (
            wide(cfg.sabattier_strength),
            wide(cfg.sabattier_reexposure),
            line_sigma_px(wide(cfg.sabattier_agitation), img.shape()),
        ),
    )
}

/// `print_stages`: alternative process, then toning, on the linear print.
pub fn print_stages(img: &Image, cfg: &PipelineConfig, ctx: &PipelineContext) -> Image {
    let paper = effective_paper_profile(&cfg.exposure.paper_profile, cfg.process.process_mode);
    let mut cur = Cow::Borrowed(img);
    if let Some(out) = lith_stage(&cur, &cfg.altproc, paper, ctx) {
        cur = Cow::Owned(out);
    }
    if let Some(out) = cyanotype_stage(&cur, &cfg.altproc, ctx) {
        cur = Cow::Owned(out);
    }
    if let Some(out) = sabattier_stage(&cur, &cfg.altproc, paper, ctx) {
        cur = Cow::Owned(out);
    }
    if let Some(out) = toning_stage(&cur, &cfg.toning, cfg.altproc.alt_process, ctx) {
        cur = Cow::Owned(out);
    }
    cur.into_owned()
}
