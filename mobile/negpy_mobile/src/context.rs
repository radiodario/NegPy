//! What the stages pass to each other beside the pixels: the desktop's `PipelineContext`
//! and the entries of its `metrics` dict that a later stage reads, as typed fields.

use crate::config::{ProcessMode, Rgb};
use crate::image::{Image, Plane, Roi};

/// D-min / D-max per channel in raw log10 (`normalization.LogNegativeBounds`).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LogNegativeBounds {
    pub floors: Rgb,
    pub ceils: Rgb,
}

/// The geometry the frame was rendered with, for the mappers that follow it
/// (`metrics["geometry_params"]` and `metrics["distortion_k1"]`).
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct GeometryParams {
    pub rotation: i32,
    pub fine_rotation: f32,
    pub flip_horizontal: bool,
    pub flip_vertical: bool,
    pub converge_v: f32,
    pub converge_h: f32,
    pub distortion_k1: f32,
}

/// The neutral axis Cast Removal balances on (`measure_neutral_axis_from_log`): per-channel
/// refs at the midtone, shadow and highlight bands, a confidence, and a pooled axis's offset
/// weight.
#[derive(Clone, Debug, PartialEq)]
pub struct NeutralAxis {
    pub mid: Rgb,
    pub shadow: Rgb,
    pub highlight: Option<Rgb>,
    pub confidence: f32,
    pub offset_weight: Option<f32>,
}

#[derive(Clone, Debug, Default)]
pub struct Metrics {
    pub geometry: GeometryParams,
    pub active_roi: Option<Roi>,
    /// The clear base in raw log (the roll's picked one or the source's rebate read).
    pub base_log: Option<Rgb>,
    pub log_bounds: Option<LogNegativeBounds>,
    pub log_bounds_base: Option<LogNegativeBounds>,
    pub pre_trim_bounds: Option<LogNegativeBounds>,
    pub final_bounds: Option<LogNegativeBounds>,
    pub norm_density_range: Option<f32>,
    pub neutral_axis_refs: Option<NeutralAxis>,
    pub metered_anchor: Option<f32>,
    pub textural_range: Option<f32>,
    pub shadow_point: Option<f32>,
    pub highlight_point: Option<f32>,
    pub print_slopes: Option<Rgb>,
    pub contrast_mask_plane: Option<Plane>,
    pub contrast_mask_centre: f32,
    pub contrast_mask_roi: Option<Roi>,
    pub diffusion_plane: Option<Image>,
    pub diffusion_roi: Option<Roi>,
    pub scan_clip_fractions: Option<Rgb>,
    pub auto_masking: Option<f32>,
}

#[derive(Clone, Debug)]
pub struct PipelineContext {
    pub original_size: (usize, usize),
    /// The buffer's long edge over the preview render size: the unit of the pixel settings.
    pub scale_factor: f32,
    pub process_mode: ProcessMode,
    pub active_roi: Option<Roi>,
    pub crop_preview_full: bool,
    pub metrics: Metrics,
}

impl PipelineContext {
    pub fn new(
        original_size: (usize, usize),
        scale_factor: f32,
        process_mode: ProcessMode,
    ) -> Self {
        Self {
            original_size,
            scale_factor,
            process_mode,
            active_roi: None,
            crop_preview_full: false,
            metrics: Metrics::default(),
        }
    }
}
