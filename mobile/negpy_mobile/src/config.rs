//! The edit, as the desktop stores it: one flat JSON object (`WorkspaceConfig.to_dict()`),
//! keyed by field name across every section. `Edit` keeps the whole object, so a field this
//! crate does not read (export, metadata, scanning) survives a round trip to the desktop
//! database untouched, and reads the sections the pipeline uses into typed structs.
//!
//! Defaults mirror `negpy/features/<name>/models.py`; `tests/config_defaults.rs` checks them
//! against the desktop's own defaults.

use serde::{Deserialize, Deserializer, Serialize};
use serde_json::{Map, Value};

/// Deserializes a flat JSON object; unknown keys are ignored, missing ones take the default.
fn section<T: for<'de> Deserialize<'de> + Default>(flat: &Map<String, Value>) -> T {
    serde_json::from_value(Value::Object(flat.clone())).unwrap_or_default()
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Default, Serialize, Deserialize)]
pub enum ProcessMode {
    #[serde(rename = "Color Negative", alias = "C41")]
    #[default]
    C41,
    #[serde(rename = "B&W Negative", alias = "B&W")]
    Bw,
    #[serde(rename = "Transparency", alias = "E-6")]
    E6,
}

/// A stale or hand-edited mode reads as color negative, as `ProcessMode._missing_` does.
fn process_mode<'de, D: Deserializer<'de>>(d: D) -> Result<ProcessMode, D::Error> {
    Ok(ProcessMode::deserialize(d).unwrap_or_default())
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum RenderIntent {
    #[default]
    Print,
    Flat,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BwFilter {
    #[default]
    None,
    Yellow8,
    DeepYellow15,
    Orange21,
    Red25,
    YellowGreen11,
    Green58,
}

fn bw_filter<'de, D: Deserializer<'de>>(d: D) -> Result<BwFilter, D::Error> {
    Ok(BwFilter::deserialize(d).unwrap_or_default())
}

/// Which renderer a frame goes through (`features/process/path.py`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RenderPath {
    Print,
    Transfer,
    Positive,
}

pub type Rgb = (f32, f32, f32);

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct ProcessConfig {
    #[serde(deserialize_with = "process_mode")]
    pub process_mode: ProcessMode,
    pub source_color: Option<bool>,
    pub linear_raw: bool,
    pub positive_source: bool,
    pub highlight_reconstruction: i32,
    pub analysis_buffer: f32,
    pub analysis_rect: Option<(f32, f32, f32, f32)>,
    pub meter_rect: Option<(f32, f32, f32, f32)>,
    pub meter_key: String,
    pub film_base: Option<Rgb>,
    pub luma_range_clip: f32,
    pub color_range_clip: f32,
    pub use_luma_average: bool,
    pub use_color_average: bool,
    pub locked_floors: Rgb,
    pub locked_ceils: Rgb,
    pub local_floors: Rgb,
    pub local_ceils: Rgb,
    pub use_cast_average: bool,
    /// (mid, shadow, highlight or null, confidence[, offset weight]); kept as JSON.
    pub locked_neutral_axis: Option<Value>,
    pub white_point_offset: f32,
    pub black_point_offset: f32,
    pub white_point_trim_red: f32,
    pub white_point_trim_green: f32,
    pub white_point_trim_blue: f32,
    pub black_point_trim_red: f32,
    pub black_point_trim_green: f32,
    pub black_point_trim_blue: f32,
    pub crosstalk_strength: f32,
    pub crosstalk_matrix: Option<Vec<f32>>,
    pub crosstalk_profile: String,
    #[serde(deserialize_with = "process_mode")]
    pub crosstalk_process: ProcessMode,
    pub sensor_matrix: Option<Vec<f32>>,
    pub sensor_profile: String,
    pub sensor_unmix: String,
    pub hue_trim: f32,
    pub lock_bounds: bool,
    pub roll_name: Option<String>,
    pub baseline_source: String,
}

impl Default for ProcessConfig {
    fn default() -> Self {
        Self {
            process_mode: ProcessMode::C41,
            source_color: None,
            linear_raw: false,
            positive_source: false,
            highlight_reconstruction: 0,
            analysis_buffer: 0.05,
            analysis_rect: None,
            meter_rect: None,
            meter_key: String::new(),
            film_base: None,
            luma_range_clip: 0.0,
            color_range_clip: 1.0,
            use_luma_average: false,
            use_color_average: false,
            locked_floors: (0.0, 0.0, 0.0),
            locked_ceils: (0.0, 0.0, 0.0),
            local_floors: (0.0, 0.0, 0.0),
            local_ceils: (0.0, 0.0, 0.0),
            use_cast_average: false,
            locked_neutral_axis: None,
            white_point_offset: 0.0,
            black_point_offset: 0.0,
            white_point_trim_red: 0.0,
            white_point_trim_green: 0.0,
            white_point_trim_blue: 0.0,
            black_point_trim_red: 0.0,
            black_point_trim_green: 0.0,
            black_point_trim_blue: 0.0,
            crosstalk_strength: 0.0,
            crosstalk_matrix: None,
            crosstalk_profile: "Generic C41".into(),
            crosstalk_process: ProcessMode::C41,
            sensor_matrix: None,
            sensor_profile: "None".into(),
            sensor_unmix: "two_scale".into(),
            hue_trim: 0.0,
            lock_bounds: false,
            roll_name: None,
            baseline_source: String::new(),
        }
    }
}

impl ProcessConfig {
    pub fn render_path(&self) -> RenderPath {
        if self.process_mode != ProcessMode::E6 {
            RenderPath::Print
        } else if self.positive_source {
            RenderPath::Positive
        } else {
            RenderPath::Transfer
        }
    }

    pub fn is_local_initialized(&self) -> bool {
        let (r, g, b) = self.local_floors;
        r != 0.0 || g != 0.0 || b != 0.0
    }

    pub fn is_locked_initialized(&self) -> bool {
        let (r, g, b) = self.locked_floors;
        r != 0.0 || g != 0.0 || b != 0.0
    }

    /// Per-channel white and black point offsets: global plus per-layer trim
    /// (`process.models.per_channel_point_offsets`).
    pub fn per_channel_point_offsets(&self) -> (Rgb, Rgb) {
        (
            (
                self.white_point_offset + self.white_point_trim_red,
                self.white_point_offset + self.white_point_trim_green,
                self.white_point_offset + self.white_point_trim_blue,
            ),
            (
                self.black_point_offset + self.black_point_trim_red,
                self.black_point_offset + self.black_point_trim_green,
                self.black_point_offset + self.black_point_trim_blue,
            ),
        )
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct ExposureConfig {
    pub density: f32,
    pub grade: f32,
    pub grade_trim_red: f32,
    pub grade_trim_green: f32,
    pub grade_trim_blue: f32,
    pub wb_cyan: f32,
    pub wb_magenta: f32,
    pub wb_yellow: f32,
    pub shadow_cyan: f32,
    pub shadow_magenta: f32,
    pub shadow_yellow: f32,
    pub highlight_cyan: f32,
    pub highlight_magenta: f32,
    pub highlight_yellow: f32,
    pub shadow_density: f32,
    pub highlight_density: f32,
    pub hold_black: bool,
    pub contrast_mask_strength: f32,
    pub diffusion: f32,
    pub shadow_grade: f32,
    pub highlight_grade: f32,
    pub shadow_grade_trim_red: f32,
    pub shadow_grade_trim_green: f32,
    pub shadow_grade_trim_blue: f32,
    pub highlight_grade_trim_red: f32,
    pub highlight_grade_trim_green: f32,
    pub highlight_grade_trim_blue: f32,
    pub toe: f32,
    pub toe_width: f32,
    pub shoulder: f32,
    pub shoulder_width: f32,
    pub toe_trim_red: f32,
    pub toe_trim_green: f32,
    pub toe_trim_blue: f32,
    pub shoulder_trim_red: f32,
    pub shoulder_trim_green: f32,
    pub shoulder_trim_blue: f32,
    pub toe_width_trim_red: f32,
    pub toe_width_trim_green: f32,
    pub toe_width_trim_blue: f32,
    pub shoulder_width_trim_red: f32,
    pub shoulder_width_trim_green: f32,
    pub shoulder_width_trim_blue: f32,
    pub paper_dmin: bool,
    pub paper_black: bool,
    pub midtone_gamma: f32,
    pub midtone_gamma_trim_red: f32,
    pub midtone_gamma_trim_green: f32,
    pub midtone_gamma_trim_blue: f32,
    pub preflash: f32,
    pub cast_removal_strength: f32,
    pub auto_exposure: bool,
    pub auto_normalize_contrast: bool,
    pub render_intent: RenderIntent,
    pub paper_profile: String,
    pub dye_separation: f32,
    pub dye_separation_trim_red: f32,
    pub dye_separation_trim_green: f32,
    pub dye_separation_trim_blue: f32,
    pub separation_damping: f32,
    #[serde(deserialize_with = "bw_filter")]
    pub bw_filter: BwFilter,
}

impl Default for ExposureConfig {
    fn default() -> Self {
        Self {
            density: 1.0,
            grade: 115.0,
            grade_trim_red: 0.0,
            grade_trim_green: 0.0,
            grade_trim_blue: 0.0,
            wb_cyan: 0.0,
            wb_magenta: 0.0,
            wb_yellow: 0.0,
            shadow_cyan: 0.0,
            shadow_magenta: 0.0,
            shadow_yellow: 0.0,
            highlight_cyan: 0.0,
            highlight_magenta: 0.0,
            highlight_yellow: 0.0,
            shadow_density: 0.0,
            highlight_density: 0.0,
            hold_black: false,
            contrast_mask_strength: 0.0,
            diffusion: 0.0,
            shadow_grade: 0.0,
            highlight_grade: 0.0,
            shadow_grade_trim_red: 0.0,
            shadow_grade_trim_green: 0.0,
            shadow_grade_trim_blue: 0.0,
            highlight_grade_trim_red: 0.0,
            highlight_grade_trim_green: 0.0,
            highlight_grade_trim_blue: 0.0,
            toe: 0.0,
            toe_width: 2.5,
            shoulder: 0.0,
            shoulder_width: 2.5,
            toe_trim_red: 0.0,
            toe_trim_green: 0.0,
            toe_trim_blue: 0.0,
            shoulder_trim_red: 0.0,
            shoulder_trim_green: 0.0,
            shoulder_trim_blue: 0.0,
            toe_width_trim_red: 0.0,
            toe_width_trim_green: 0.0,
            toe_width_trim_blue: 0.0,
            shoulder_width_trim_red: 0.0,
            shoulder_width_trim_green: 0.0,
            shoulder_width_trim_blue: 0.0,
            paper_dmin: false,
            paper_black: false,
            midtone_gamma: 0.0,
            midtone_gamma_trim_red: 0.0,
            midtone_gamma_trim_green: 0.0,
            midtone_gamma_trim_blue: 0.0,
            preflash: 0.0,
            cast_removal_strength: 0.5,
            auto_exposure: true,
            auto_normalize_contrast: true,
            render_intent: RenderIntent::Print,
            paper_profile: "neutral".into(),
            dye_separation: 1.0,
            dye_separation_trim_red: 0.0,
            dye_separation_trim_green: 0.0,
            dye_separation_trim_blue: 0.0,
            separation_damping: 0.0,
            bw_filter: BwFilter::None,
        }
    }
}

impl ExposureConfig {
    /// A grade stored on the legacy 0-5 paper ladder converts as `ExposureConfig.__post_init__` does.
    fn fix_legacy(mut self) -> Self {
        if self.grade <= 5.0 {
            self.grade = 150.0 - 20.0 * self.grade;
        }
        self
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct GeometryConfig {
    pub lens_distortion_from_metadata: bool,
    pub lens_ca_from_metadata: bool,
    pub rotation: i32,
    pub fine_rotation: f32,
    pub flip_horizontal: bool,
    pub flip_vertical: bool,
    pub converge_v: f32,
    pub converge_h: f32,
    pub distortion_k1: f32,
    pub crop_to_valid: bool,
    pub autocrop_offset: i32,
    pub autocrop_ratio: String,
    pub autocrop_mode: String,
    pub autocrop_rebate_trim: f32,
    pub autocrop_square: bool,
    pub autocrop_squared: bool,
    pub crop_rect: Option<(f32, f32, f32, f32)>,
    pub crop_from_auto: bool,
    pub crop_detect_key: String,
}

impl Default for GeometryConfig {
    fn default() -> Self {
        Self {
            lens_distortion_from_metadata: false,
            lens_ca_from_metadata: false,
            rotation: 0,
            fine_rotation: 0.0,
            flip_horizontal: false,
            flip_vertical: false,
            converge_v: 0.0,
            converge_h: 0.0,
            distortion_k1: 0.0,
            crop_to_valid: false,
            autocrop_offset: 0,
            autocrop_ratio: "Free".into(),
            autocrop_mode: "image".into(),
            autocrop_rebate_trim: 1.0,
            autocrop_square: false,
            autocrop_squared: false,
            crop_rect: None,
            crop_from_auto: false,
            crop_detect_key: String::new(),
        }
    }
}

impl GeometryConfig {
    fn fix_legacy(mut self) -> Self {
        if self.lens_distortion_from_metadata {
            self.distortion_k1 = 0.0;
        }
        self
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct LabConfig {
    pub saturation: f32,
    pub skin_protection: f32,
    pub clahe_strength: f32,
    pub sharpen: f32,
    pub sharpen_method: String,
    pub sharpen_radius: f32,
    pub sharpen_masking: f32,
    pub sharpen_auto_masking: bool,
    pub chroma_denoise: f32,
    pub glow_amount: f32,
    pub halation_strength: f32,
}

impl Default for LabConfig {
    fn default() -> Self {
        Self {
            saturation: 1.0,
            skin_protection: 0.5,
            clahe_strength: 0.0,
            sharpen: 0.25,
            sharpen_method: "usm".into(),
            sharpen_radius: 1.0,
            sharpen_masking: 0.1,
            sharpen_auto_masking: false,
            chroma_denoise: 0.0,
            glow_amount: 0.0,
            halation_strength: 0.0,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum MaskShape {
    Polygon,
    Oval,
    Gradient,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum MaskKey {
    Off,
    Highlights,
    Shadows,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct LocalMask {
    pub vertices: Vec<(f32, f32)>,
    pub stops: f32,
    pub feather: f32,
    pub grade: f32,
    pub flash: f32,
    pub shape: MaskShape,
    pub invert: bool,
    pub enabled: bool,
    pub key: MaskKey,
    pub key_zone: f32,
    pub key_softness: f32,
}

impl Default for LocalMask {
    fn default() -> Self {
        Self {
            vertices: Vec::new(),
            stops: 0.0,
            feather: 0.04,
            grade: 0.0,
            flash: 0.0,
            shape: MaskShape::Polygon,
            invert: false,
            enabled: true,
            key: MaskKey::Off,
            key_zone: 6.0,
            key_softness: 1.0,
        }
    }
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct LocalAdjustmentsConfig {
    pub masks: Vec<LocalMask>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum AltProcess {
    #[default]
    None,
    Lith,
    Cyanotype,
    Sabattier,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct AltProcessConfig {
    pub alt_process: AltProcess,
    pub lith_exposure: f32,
    pub lith_snatch: f32,
    pub lith_abruptness: f32,
    pub cyano_sensitizer: String,
    pub cyano_exposure: f32,
    pub cyano_scale: f32,
    pub cyano_bleach: f32,
    pub cyano_tannin: f32,
    pub sabattier_strength: f32,
    pub sabattier_reexposure: f32,
    pub sabattier_agitation: f32,
}

impl Default for AltProcessConfig {
    fn default() -> Self {
        Self {
            alt_process: AltProcess::None,
            lith_exposure: 2.0,
            lith_snatch: 0.55,
            lith_abruptness: 0.6,
            cyano_sensitizer: "classic".into(),
            cyano_exposure: 0.0,
            cyano_scale: 1.4,
            cyano_bleach: 0.0,
            cyano_tannin: 0.0,
            sabattier_strength: 1.3,
            sabattier_reexposure: 0.45,
            sabattier_agitation: 0.7,
        }
    }
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct ToningConfig {
    pub selenium_strength: f32,
    pub sepia_strength: f32,
    pub gold_strength: f32,
    pub blue_strength: f32,
    pub copper_strength: f32,
    pub vanadium_strength: f32,
    pub shadow_tint_hue: f32,
    pub shadow_tint_strength: f32,
    pub highlight_tint_hue: f32,
    pub highlight_tint_strength: f32,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct FinishConfig {
    pub vignette_stops: f32,
    pub vignette_size: f32,
    pub vignette_roundness: f32,
    pub carrier_width: f32,
    pub carrier_rough: f32,
    pub carrier_corner: f32,
    pub carrier_flare: f32,
    pub border_size: f32,
    pub border_color: String,
    pub border_bottom_weight: f32,
    pub border_match_paper: bool,
}

impl Default for FinishConfig {
    fn default() -> Self {
        Self {
            vignette_stops: 0.0,
            vignette_size: 0.5,
            vignette_roundness: 0.0,
            carrier_width: 0.0,
            carrier_rough: 0.25,
            carrier_corner: 0.25,
            carrier_flare: 0.5,
            border_size: 0.0,
            border_color: "#ffffff".into(),
            border_bottom_weight: 1.0,
            border_match_paper: false,
        }
    }
}

/// The export fields the pipeline reads (the print size the carrier scales with).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct ExportFields {
    pub export_print_size: f32,
}

impl Default for ExportFields {
    fn default() -> Self {
        Self {
            export_print_size: 30.0,
        }
    }
}

/// The sections the pipeline reads, typed.
#[derive(Clone, Debug, PartialEq)]
pub struct PipelineConfig {
    pub process: ProcessConfig,
    pub exposure: ExposureConfig,
    pub geometry: GeometryConfig,
    pub lab: LabConfig,
    pub local: LocalAdjustmentsConfig,
    pub altproc: AltProcessConfig,
    pub toning: ToningConfig,
    pub finish: FinishConfig,
    pub export: ExportFields,
}

impl Default for PipelineConfig {
    fn default() -> Self {
        Self::from_flat(&Map::new())
    }
}

impl PipelineConfig {
    /// Reads the typed sections out of a flat dict. Local masks sit nested under `local_masks`.
    pub fn from_flat(flat: &Map<String, Value>) -> Self {
        let local = flat
            .get("local_masks")
            .cloned()
            .map(serde_json::from_value)
            .and_then(Result::ok)
            .unwrap_or_default();
        let mut process: ProcessConfig = section(flat);
        if process.positive_source && process.process_mode != ProcessMode::E6 {
            process.positive_source = false;
        }
        Self {
            process,
            exposure: section::<ExposureConfig>(flat).fix_legacy(),
            geometry: section::<GeometryConfig>(flat).fix_legacy(),
            lab: section(flat),
            local,
            altproc: section(flat),
            toning: section(flat),
            finish: section(flat),
            export: section(flat),
        }
    }
}

/// One frame's edit: the flat dict as stored, plus the typed view the pipeline renders with.
#[derive(Clone, Debug, PartialEq)]
pub struct Edit {
    pub flat: Map<String, Value>,
    pub config: PipelineConfig,
}

impl Edit {
    pub fn from_flat(flat: Map<String, Value>) -> Self {
        let config = PipelineConfig::from_flat(&flat);
        Self { flat, config }
    }

    pub fn from_json(text: &str) -> Result<Self, serde_json::Error> {
        let value: Value = serde_json::from_str(text)?;
        match value {
            Value::Object(flat) => Ok(Self::from_flat(flat)),
            _ => Err(serde::de::Error::custom("an edit is a JSON object")),
        }
    }

    /// Writes `value` under `key` in the flat dict and re-reads the typed view.
    pub fn set(&mut self, key: &str, value: Value) {
        self.flat.insert(key.to_string(), value);
        self.config = PipelineConfig::from_flat(&self.flat);
    }

    pub fn to_json(&self) -> String {
        serde_json::to_string(&Value::Object(self.flat.clone())).expect("flat dict serializes")
    }
}
