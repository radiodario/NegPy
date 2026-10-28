//! Loader for the parity fixtures `mobile/scripts/make_fixtures.py` writes: a JSON file per
//! case, with the flat config, the context and every stage's output as a raw little-endian
//! float32 file next to it.

use std::path::{Path, PathBuf};

use serde_json::Value;

use crate::image::{Image, Plane, Roi};

pub struct Fixture {
    pub dir: PathBuf,
    pub json: Value,
}

impl Fixture {
    pub fn load(name: &str) -> Fixture {
        let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures");
        let text = std::fs::read_to_string(dir.join(format!("{name}.json"))).unwrap_or_else(|e| {
            panic!("fixture {name}: {e}; run `uv run python mobile/scripts/make_fixtures.py`")
        });
        Fixture {
            dir,
            json: serde_json::from_str(&text).expect("fixture json"),
        }
    }

    /// The flat config dict the desktop wrote for this case.
    pub fn config(&self) -> &Value {
        &self.json["config"]
    }

    pub fn context(&self) -> &Value {
        &self.json["context"]
    }

    pub fn metrics(&self) -> &Value {
        &self.json["metrics"]
    }

    fn raw_f32(&self, entry: &Value) -> (Vec<usize>, Vec<f32>) {
        let shape: Vec<usize> = entry["shape"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_u64().unwrap() as usize)
            .collect();
        let bytes =
            std::fs::read(self.dir.join(entry["file"].as_str().unwrap())).expect("fixture data");
        let data: Vec<f32> = bytes
            .as_chunks::<4>()
            .0
            .iter()
            .map(|b| f32::from_le_bytes(*b))
            .collect();
        assert_eq!(
            data.len(),
            shape.iter().product::<usize>(),
            "fixture data does not match its shape"
        );
        (shape, data)
    }

    /// Has the stage `key` (a case that bypasses a stage leaves it out).
    pub fn has(&self, key: &str) -> bool {
        self.json["arrays"].get(key).is_some()
    }

    /// An `(h, w, 3)` array under `arrays[key]`.
    pub fn image(&self, key: &str) -> Image {
        let (shape, data) = self.raw_f32(&self.json["arrays"][key]);
        assert_eq!(shape.len(), 3, "{key} is not an image");
        assert_eq!(shape[2], 3, "{key} is not RGB");
        Image::new(shape[0], shape[1], data)
    }

    /// An `(h, w)` array under `arrays[key]`.
    pub fn plane(&self, key: &str) -> Plane {
        let (shape, data) = self.raw_f32(&self.json["arrays"][key]);
        assert_eq!(shape.len(), 2, "{key} is not a plane");
        Plane::new(shape[0], shape[1], data)
    }

    /// Any float array under `arrays[key]`, with its shape.
    pub fn array(&self, key: &str) -> (Vec<usize>, Vec<f32>) {
        self.raw_f32(&self.json["arrays"][key])
    }
}

/// Every fixture name, from the directory listing.
pub fn fixture_names() -> Vec<String> {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures");
    let mut names: Vec<String> = std::fs::read_dir(dir)
        .map(|rd| {
            rd.filter_map(|e| e.ok())
                .filter_map(|e| e.file_name().to_str().map(String::from))
                .filter_map(|n| n.strip_suffix(".json").map(String::from))
                .collect()
        })
        .unwrap_or_default();
    names.sort();
    names
}

use crate::config::{Edit, ProcessMode, Rgb};
use crate::context::{GeometryParams, LogNegativeBounds, Metrics, NeutralAxis, PipelineContext};

fn rgb_of(v: &Value) -> Option<Rgb> {
    let a = v.as_array()?;
    Some((
        a[0].as_f64()? as f32,
        a[1].as_f64()? as f32,
        a[2].as_f64()? as f32,
    ))
}

fn bounds_of(v: &Value) -> Option<LogNegativeBounds> {
    Some(LogNegativeBounds {
        floors: rgb_of(v.get("floors")?)?,
        ceils: rgb_of(v.get("ceils")?)?,
    })
}

fn roi_of(v: &Value) -> Option<Roi> {
    let a = v.as_array()?;
    Some((
        a[0].as_u64()? as usize,
        a[1].as_u64()? as usize,
        a[2].as_u64()? as usize,
        a[3].as_u64()? as usize,
    ))
}

fn f32_of(v: &Value) -> Option<f32> {
    v.as_f64().map(|x| x as f32)
}

/// The desktop's `neutral_axis_refs`: (mid, shadow, highlight or null, confidence[, weight]).
pub fn neutral_axis_of(v: &Value) -> Option<NeutralAxis> {
    let a = v.as_array()?;
    Some(NeutralAxis {
        mid: rgb_of(&a[0])?,
        shadow: rgb_of(&a[1])?,
        highlight: rgb_of(&a[2]),
        confidence: f32_of(&a[3])?,
        offset_weight: a.get(4).and_then(f32_of),
    })
}

impl Fixture {
    /// The edit this case rendered with.
    pub fn edit(&self) -> Edit {
        Edit::from_flat(self.config().as_object().expect("flat config").clone())
    }

    pub fn process_mode(&self) -> ProcessMode {
        self.edit().config.process.process_mode
    }

    /// A context as the engine would hold it before the first stage: size, scale factor, mode.
    pub fn fresh_context(&self) -> PipelineContext {
        let ctx = self.context();
        let size = ctx["original_size"].as_array().unwrap();
        PipelineContext::new(
            (
                size[0].as_u64().unwrap() as usize,
                size[1].as_u64().unwrap() as usize,
            ),
            ctx["scale_factor"].as_f64().unwrap() as f32,
            self.process_mode(),
        )
    }

    /// A context as the engine held it after the base stage: the recorded metrics, with the
    /// planes the engine built between base and exposure.
    pub fn context_after_base(&self) -> PipelineContext {
        let mut ctx = self.fresh_context();
        let m = self.metrics();
        ctx.active_roi = m.get("active_roi").and_then(roi_of);
        let gp = &m["geometry_params"];
        let metrics = Metrics {
            geometry: GeometryParams {
                rotation: gp["rotation"].as_i64().unwrap_or(0) as i32,
                fine_rotation: f32_of(&gp["fine_rotation"]).unwrap_or(0.0),
                flip_horizontal: gp["flip_horizontal"].as_bool().unwrap_or(false),
                flip_vertical: gp["flip_vertical"].as_bool().unwrap_or(false),
                converge_v: f32_of(&gp["converge_v"]).unwrap_or(0.0),
                converge_h: f32_of(&gp["converge_h"]).unwrap_or(0.0),
                distortion_k1: m.get("distortion_k1").and_then(f32_of).unwrap_or(0.0),
            },
            active_roi: ctx.active_roi,
            base_log: m.get("base_log").and_then(rgb_of),
            log_bounds: m.get("log_bounds").and_then(bounds_of),
            log_bounds_base: m.get("log_bounds_base").and_then(bounds_of),
            pre_trim_bounds: m.get("pre_trim_bounds").and_then(bounds_of),
            final_bounds: m.get("final_bounds").and_then(bounds_of),
            norm_density_range: m.get("norm_density_range").and_then(f32_of),
            neutral_axis_refs: m.get("neutral_axis_refs").and_then(neutral_axis_of),
            metered_anchor: m.get("metered_anchor").and_then(f32_of),
            textural_range: m.get("textural_range").and_then(f32_of),
            shadow_point: m.get("shadow_point").and_then(f32_of),
            highlight_point: m.get("highlight_point").and_then(f32_of),
            print_slopes: None,
            contrast_mask_plane: if self.has("metrics.contrast_mask_plane") {
                Some(self.plane("metrics.contrast_mask_plane"))
            } else {
                None
            },
            contrast_mask_centre: m
                .get("contrast_mask_centre")
                .and_then(f32_of)
                .unwrap_or(0.5),
            contrast_mask_roi: m.get("contrast_mask_roi").and_then(roi_of),
            diffusion_plane: if self.has("metrics.diffusion_plane") {
                Some(self.image("metrics.diffusion_plane"))
            } else {
                None
            },
            diffusion_roi: m.get("diffusion_roi").and_then(roi_of),
            scan_clip_fractions: m.get("scan_clip_fractions").and_then(rgb_of),
            auto_masking: None,
        };
        ctx.metrics = metrics;
        ctx
    }
}
