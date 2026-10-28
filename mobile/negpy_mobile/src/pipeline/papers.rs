//! Paper profiles (`features/exposure/papers.py`): per-paper overrides of the print
//! character. A color stock's dyes are a density matrix or a baked LUT; the LUT files are not
//! bundled yet, so a LUT stock is reported to the caller as unsupported.

use crate::config::ProcessMode;
use crate::pipeline::constants::{EXPOSURE_CONSTANTS, ExposureConstants};

pub type Rows3 = [[f64; 3]; 3];
pub type LithPath = [(f64, f64); 4];

const IDENTITY_DYE: Rows3 = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
const DEFAULT_LITH_PATH: LithPath = [(4.0, 8.0), (2.0, 8.0), (-1.0, 4.0), (0.0, 0.0)];

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum PrintColor {
    Matrix(Rows3),
    Lut(&'static str),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PaperKind {
    Default,
    Bw,
    Ra4,
}

#[derive(Clone, Debug, PartialEq)]
pub struct PaperProfile {
    pub key: &'static str,
    pub label: &'static str,
    pub kind: PaperKind,
    pub d_max: f64,
    pub d_min: f64,
    pub toe_sharpness_base: f64,
    pub shoulder_sharpness_base: f64,
    pub toe_height: f64,
    pub shoulder_height: f64,
    pub paper_midtone_gamma: f64,
    pub paper_gamma_width: f64,
    pub channel_gamma: (f64, f64, f64),
    pub slope_gain: f64,
    pub base_tint_cmy: (f64, f64, f64),
    pub print_color: PrintColor,
    pub lith_path: LithPath,
}

const fn base(key: &'static str, label: &'static str, kind: PaperKind) -> PaperProfile {
    PaperProfile {
        key,
        label,
        kind,
        d_max: EXPOSURE_CONSTANTS.d_max,
        d_min: EXPOSURE_CONSTANTS.d_min,
        toe_sharpness_base: EXPOSURE_CONSTANTS.toe_sharpness_base,
        shoulder_sharpness_base: EXPOSURE_CONSTANTS.shoulder_sharpness_base,
        toe_height: EXPOSURE_CONSTANTS.toe_height,
        shoulder_height: EXPOSURE_CONSTANTS.shoulder_height,
        paper_midtone_gamma: EXPOSURE_CONSTANTS.paper_midtone_gamma,
        paper_gamma_width: EXPOSURE_CONSTANTS.paper_gamma_width,
        channel_gamma: (1.0, 1.0, 1.0),
        slope_gain: 1.0,
        base_tint_cmy: (0.0, 0.0, 0.0),
        print_color: PrintColor::Matrix(IDENTITY_DYE),
        lith_path: DEFAULT_LITH_PATH,
    }
}

pub const DEFAULT_PROFILE_KEY: &str = "neutral";

pub static PAPER_PROFILES: &[PaperProfile] = &[
    base("neutral", "Digital (Adobe RGB)", PaperKind::Default),
    PaperProfile {
        print_color: PrintColor::Lut("pakon"),
        ..base("pakon", "Digital (Pakon)", PaperKind::Ra4)
    },
    PaperProfile {
        d_max: 2.52,
        d_min: 0.005,
        toe_sharpness_base: 3.53,
        shoulder_sharpness_base: 1.21,
        paper_midtone_gamma: 0.25,
        paper_gamma_width: 2.0,
        slope_gain: 1.2,
        print_color: PrintColor::Lut("frontier"),
        ..base("frontier", "Digital (Frontier)", PaperKind::Ra4)
    },
    PaperProfile {
        toe_sharpness_base: 4.0,
        d_max: 2.10,
        d_min: 0.04,
        paper_midtone_gamma: 0.15,
        ..base("ilford_mg_rc", "Ilford Multigrade RC", PaperKind::Bw)
    },
    PaperProfile {
        d_max: 2.15,
        d_min: 0.04,
        toe_sharpness_base: 5.0,
        paper_midtone_gamma: 0.15,
        ..base(
            "ilford_fb_classic",
            "Ilford Multigrade FB Classic",
            PaperKind::Bw,
        )
    },
    PaperProfile {
        d_max: 2.0,
        d_min: 0.05,
        toe_sharpness_base: 3.5,
        paper_midtone_gamma: 0.10,
        lith_path: [(14.0, 22.0), (7.0, 26.0), (-8.0, 14.0), (-2.0, 2.0)],
        ..base("foma_fomatone", "Foma Fomatone MG Classic", PaperKind::Bw)
    },
    PaperProfile {
        toe_sharpness_base: 4.0,
        d_max: 2.0,
        d_min: 0.04,
        paper_midtone_gamma: 0.15,
        lith_path: [(6.0, 18.0), (2.0, 18.0), (-9.0, 10.0), (-2.0, 1.0)],
        ..base("foma_fomabrom", "Foma Fomabrom Variant", PaperKind::Bw)
    },
    PaperProfile {
        d_max: 2.13,
        d_min: 0.062,
        toe_sharpness_base: 3.58,
        shoulder_sharpness_base: 3.07,
        paper_midtone_gamma: -0.111,
        paper_gamma_width: 0.47,
        ..base("kodak_polymax_ii", "Kodak Polymax II RC", PaperKind::Bw)
    },
    PaperProfile {
        d_max: 2.15,
        d_min: 0.062,
        toe_sharpness_base: 5.99,
        shoulder_sharpness_base: 2.96,
        paper_midtone_gamma: -0.148,
        paper_gamma_width: 0.85,
        ..base(
            "kodak_polycontrast_iv",
            "Kodak Polycontrast IV RC",
            PaperKind::Bw,
        )
    },
    PaperProfile {
        d_max: 2.55,
        d_min: 0.06,
        toe_sharpness_base: 3.5,
        paper_midtone_gamma: 0.22,
        channel_gamma: (1.04, 1.0, 0.98),
        print_color: PrintColor::Lut("kodak_e4070"),
        ..base("kodak_endura", "Kodak Endura Premier", PaperKind::Ra4)
    },
    PaperProfile {
        d_max: 2.42,
        d_min: 0.083,
        toe_sharpness_base: 5.96,
        shoulder_sharpness_base: 2.89,
        paper_midtone_gamma: 0.22,
        base_tint_cmy: (0.008, 0.007, -0.016),
        print_color: PrintColor::Lut("fuji_ca_dpii"),
        ..base("fuji_crystal", "Fujicolor Crystal Archive", PaperKind::Ra4)
    },
    PaperProfile {
        d_max: 2.14,
        d_min: 0.071,
        toe_sharpness_base: 5.96,
        shoulder_sharpness_base: 2.89,
        paper_midtone_gamma: 0.22,
        base_tint_cmy: (0.015, 0.011, -0.026),
        print_color: PrintColor::Lut("fuji_ca_hd"),
        ..base(
            "fuji_crystal_hd",
            "Fujicolor Crystal Archive HD",
            PaperKind::Ra4,
        )
    },
    PaperProfile {
        d_max: 2.65,
        d_min: 0.093,
        toe_sharpness_base: 1.78,
        shoulder_sharpness_base: 2.81,
        paper_midtone_gamma: 0.25,
        paper_gamma_width: 0.89,
        channel_gamma: (0.968, 1.0, 1.013),
        base_tint_cmy: (-0.002, 0.012, -0.009),
        print_color: PrintColor::Lut("kodak_e4021"),
        ..base("kodak_portra_endura", "Kodak Portra Endura", PaperKind::Ra4)
    },
    PaperProfile {
        d_max: 2.60,
        d_min: 0.095,
        toe_sharpness_base: 7.99,
        shoulder_sharpness_base: 3.23,
        paper_midtone_gamma: 0.165,
        paper_gamma_width: 0.62,
        channel_gamma: (1.008, 1.0, 1.0),
        base_tint_cmy: (0.011, -0.009, -0.003),
        print_color: PrintColor::Lut("kodak_e4021"),
        ..base("kodak_supra_endura", "Kodak Supra Endura", PaperKind::Ra4)
    },
    PaperProfile {
        d_max: 2.61,
        d_min: 0.096,
        toe_sharpness_base: 6.04,
        shoulder_sharpness_base: 5.74,
        paper_midtone_gamma: 0.197,
        paper_gamma_width: 0.60,
        channel_gamma: (1.109, 1.0, 1.051),
        base_tint_cmy: (-0.003, -0.003, 0.005),
        print_color: PrintColor::Lut("kodak_e4070"),
        ..base("kodak_ultra_endura", "Kodak Ultra Endura", PaperKind::Ra4)
    },
    PaperProfile {
        d_max: 2.25,
        d_min: 0.083,
        toe_sharpness_base: 2.30,
        shoulder_sharpness_base: 2.34,
        paper_midtone_gamma: 0.250,
        paper_gamma_width: 0.82,
        channel_gamma: (1.013, 1.0, 1.004),
        base_tint_cmy: (0.011, 0.011, -0.022),
        print_color: PrintColor::Lut("kodak_e4070"),
        ..base(
            "kodak_ektacolor_edge",
            "Kodak Ektacolor Edge",
            PaperKind::Ra4,
        )
    },
    PaperProfile {
        d_max: 2.29,
        d_min: 0.065,
        toe_sharpness_base: 2.65,
        shoulder_sharpness_base: 2.86,
        paper_midtone_gamma: -0.019,
        paper_gamma_width: 0.31,
        channel_gamma: (1.034, 1.0, 0.989),
        print_color: PrintColor::Lut("kodak_2383"),
        ..base("kodak_2383", "Kodak Vision 2383", PaperKind::Ra4)
    },
    PaperProfile {
        d_max: 2.25,
        d_min: 0.025,
        toe_sharpness_base: 3.22,
        shoulder_sharpness_base: 2.28,
        paper_midtone_gamma: 0.074,
        paper_gamma_width: 0.58,
        channel_gamma: (1.075, 1.0, 1.049),
        print_color: PrintColor::Lut("kodak_2393"),
        ..base("kodak_2393", "Kodak Vision 2393", PaperKind::Ra4)
    },
];

pub fn default_profile() -> &'static PaperProfile {
    &PAPER_PROFILES[0]
}

/// Profile for `key`, the Digital default on an unknown key.
pub fn resolve_paper(key: &str) -> &'static PaperProfile {
    PAPER_PROFILES
        .iter()
        .find(|p| p.key == key)
        .unwrap_or_else(default_profile)
}

fn mode_kind(mode: ProcessMode) -> Option<PaperKind> {
    match mode {
        ProcessMode::C41 => Some(PaperKind::Ra4),
        ProcessMode::Bw => Some(PaperKind::Bw),
        ProcessMode::E6 => None,
    }
}

/// The stored profile only when its kind matches the mode, else the Digital default.
pub fn effective_paper_profile(key: &str, mode: ProcessMode) -> &'static PaperProfile {
    let paper = resolve_paper(key);
    if paper.kind == PaperKind::Default || mode_kind(mode) == Some(paper.kind) {
        paper
    } else {
        default_profile()
    }
}

/// The paper's straight-line slope over the one its Grade gives.
pub fn slope_gain(paper: Option<&PaperProfile>) -> f64 {
    paper.map_or(1.0, |p| p.slope_gain)
}

/// `EXPOSURE_CONSTANTS` with the profile's tonal overrides.
pub fn effective_constants(paper: Option<&PaperProfile>) -> ExposureConstants {
    let mut c = EXPOSURE_CONSTANTS;
    if let Some(p) = paper.filter(|p| p.kind != PaperKind::Default) {
        c.d_max = p.d_max;
        c.d_min = p.d_min;
        c.toe_sharpness_base = p.toe_sharpness_base;
        c.shoulder_sharpness_base = p.shoulder_sharpness_base;
        c.toe_height = p.toe_height;
        c.shoulder_height = p.shoulder_height;
        c.paper_midtone_gamma = p.paper_midtone_gamma;
        c.paper_gamma_width = p.paper_gamma_width;
    }
    c
}

/// The stock's dye matrix, row-normalized; None for identity, and for a LUT stock, which has
/// no matrix (`Err` tells the caller the stock is unsupported here).
pub fn resolve_print_color(
    paper: Option<&PaperProfile>,
) -> Result<Option<Rows3>, UnsupportedStock> {
    match paper.map(|p| p.print_color) {
        None => Ok(None),
        Some(PrintColor::Lut(key)) => Err(UnsupportedStock(key)),
        Some(PrintColor::Matrix(rows)) if rows == IDENTITY_DYE => Ok(None),
        Some(PrintColor::Matrix(rows)) => {
            let mut m = rows;
            for row in m.iter_mut() {
                let s = (row[0] + row[1] + row[2]).max(1e-6);
                for v in row.iter_mut() {
                    *v /= s;
                }
            }
            Ok(Some(m))
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct UnsupportedStock(pub &'static str);

impl std::fmt::Display for UnsupportedStock {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "print stock LUT '{}' is not available on this device",
            self.0
        )
    }
}

impl std::error::Error for UnsupportedStock {}

/// Density-domain saturation matrix: row ch is `k[ch] * I[ch] + (1 - k[ch]) * J[ch]`, J the
/// achromatic projection. None at k = 1 on every channel.
pub fn resolve_saturation_matrix(k: (f64, f64, f64)) -> Option<Rows3> {
    if k == (1.0, 1.0, 1.0) {
        return None;
    }
    let ks = [k.0, k.1, k.2];
    let mut m = [[0.0; 3]; 3];
    for (r, row) in m.iter_mut().enumerate() {
        for (c, v) in row.iter_mut().enumerate() {
            *v = (if r == c { ks[r] } else { 0.0 }) + (1.0 - ks[r]) / 3.0;
        }
    }
    Some(m)
}

/// `sat @ dye`, or None when both are identity.
pub fn compose_density_matrices(dye: Option<Rows3>, sat: Option<Rows3>) -> Option<Rows3> {
    if dye.is_none() && sat.is_none() {
        return None;
    }
    let d = dye.unwrap_or(IDENTITY_DYE);
    let s = sat.unwrap_or(IDENTITY_DYE);
    let mut out = [[0.0; 3]; 3];
    for (i, row) in out.iter_mut().enumerate() {
        for (j, v) in row.iter_mut().enumerate() {
            *v = (0..3).map(|k| s[i][k] * d[k][j]).sum();
        }
    }
    Some(out)
}
