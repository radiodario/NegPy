//! `features/exposure/bw_filter.py`: the weights a B&W print collapses the three channels with,
//! a Wratten filter's on a color source.

use crate::config::{BwFilter, ExposureConfig, ProcessMode};

/// R, G, B weights on the normalized log channels; each row sums to exactly 1.
pub fn bw_filter_weights(filter: BwFilter) -> (f64, f64, f64) {
    match filter {
        BwFilter::None => (0.31, 0.39, 0.30),
        BwFilter::Yellow8 => (0.47, 0.57, -0.04),
        BwFilter::DeepYellow15 => (0.56, 0.50, -0.06),
        BwFilter::Orange21 => (0.72, 0.32, -0.04),
        BwFilter::Red25 => (1.06, -0.06, 0.00),
        BwFilter::YellowGreen11 => (0.25, 0.78, -0.03),
        BwFilter::Green58 => (0.04, 1.08, -0.12),
    }
}

/// The B&W print's channel weights. A filter acts only on a color source.
pub fn collapse_weights(
    mode: ProcessMode,
    source_color: Option<bool>,
    filter: BwFilter,
) -> (f64, f64, f64) {
    if mode == ProcessMode::Bw && source_color == Some(true) {
        bw_filter_weights(filter)
    } else {
        bw_filter_weights(BwFilter::None)
    }
}

/// 3x3 float32 with every row `weights`: `matmul_3x3` then gives three equal channels.
pub fn collapse_matrix(weights: (f64, f64, f64)) -> [[f32; 3]; 3] {
    let row = [weights.0 as f32, weights.1 as f32, weights.2 as f32];
    [row; 3]
}

/// The weights the mask and diffusion planes collapse with, or None outside B&W.
pub fn plane_weights(
    mode: ProcessMode,
    source_color: Option<bool>,
    filter: BwFilter,
) -> Option<(f64, f64, f64)> {
    (mode == ProcessMode::Bw).then(|| collapse_weights(mode, source_color, filter))
}

/// The exposure the print renders: B&W drops the color filtration.
pub fn printed_exposure(mode: ProcessMode, exposure: &ExposureConfig) -> ExposureConfig {
    let mut ex = exposure.clone();
    if mode == ProcessMode::Bw {
        ex.wb_cyan = 0.0;
        ex.wb_magenta = 0.0;
        ex.wb_yellow = 0.0;
        ex.shadow_cyan = 0.0;
        ex.shadow_magenta = 0.0;
        ex.shadow_yellow = 0.0;
        ex.highlight_cyan = 0.0;
        ex.highlight_magenta = 0.0;
        ex.highlight_yellow = 0.0;
    }
    ex
}
