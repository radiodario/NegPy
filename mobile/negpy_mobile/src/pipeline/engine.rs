//! `DarkroomEngine.process()` (`negpy/services/rendering/engine.py`) for the print path:
//! base (geometry + normalization), the mask and diffusion planes, exposure, clahe, lab,
//! the alt process and toning, crop, finish, the output encode. No stage cache: a preview
//! re-renders whole, and the phone's proxy is small enough for that.

use crate::config::{AltProcess, PipelineConfig, ProcessMode, RenderIntent, RenderPath};
use crate::context::PipelineContext;
use crate::image::Image;
use crate::pipeline::papers::UnsupportedStock;
use crate::pipeline::{altprocess, exposure, finish, geometry, lab, normalization};

#[derive(Debug, Clone, PartialEq)]
pub enum EngineError {
    UnsupportedStock(String),
    UnsupportedPath(String),
}

impl std::fmt::Display for EngineError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            EngineError::UnsupportedStock(m) | EngineError::UnsupportedPath(m) => f.write_str(m),
        }
    }
}

impl std::error::Error for EngineError {}

impl From<UnsupportedStock> for EngineError {
    fn from(e: UnsupportedStock) -> Self {
        EngineError::UnsupportedStock(e.to_string())
    }
}

pub struct RenderOutput {
    pub image: Image,
    pub metrics_json: String,
}

/// The desktop's preview size: the unit of the pixel settings (`render_scale_factor`).
pub const PREVIEW_RENDER_SIZE: f32 = 1600.0;

pub fn render_scale_factor(shape: (usize, usize)) -> f32 {
    shape.0.max(shape.1) as f32 / PREVIEW_RENDER_SIZE
}

/// The engine on `img`, a scene-linear source, with a fresh context.
pub fn process(
    img: &Image,
    cfg: &PipelineConfig,
    ctx: &mut PipelineContext,
) -> Result<Image, EngineError> {
    if cfg.process.render_path() != RenderPath::Print {
        return Err(EngineError::UnsupportedPath(
            "slides render on the desktop only".into(),
        ));
    }
    let flat = cfg.exposure.render_intent == RenderIntent::Flat;

    // Base: the clear base read once per source, before geometry, then geometry and normalization.
    ctx.metrics.base_log = normalization::base_log(img);
    let current = geometry::geometry(img, &cfg.geometry, ctx);
    let current = normalization::normalize(&current, &cfg.process, ctx);

    // The planes the exposure stage mixes in, built from the source with the geometry re-applied.
    if !flat {
        let render_shape = current.shape();
        normalization::build_contrast_mask(img, cfg, ctx, render_shape);
        normalization::build_diffusion(img, cfg, ctx, render_shape);
    }

    let mut current = exposure::exposure(&current, cfg, ctx)?;

    if !flat {
        current = lab::clahe(&current, cfg.lab.clahe_strength);
        current = lab::lab(&current, &cfg.lab, ctx);
        current = altprocess::print_stages(&current, cfg, ctx);
    }
    if !ctx.crop_preview_full {
        current = geometry::crop(&current, ctx);
    }
    if !flat {
        let carrier_live = cfg.finish.carrier_width > 0.0 && !ctx.crop_preview_full;
        let tone = if carrier_live {
            Some(rebate_tone(cfg, ctx))
        } else {
            None
        };
        current = finish::finish(&current, cfg, tone.as_deref(), ctx);
        current = finish::encode_output(&current);
    }
    Ok(current)
}

/// `finish/processor.py::rebate_tone`: the print color of a neutral ramp off the film base,
/// through this frame's curves, saturation and toning, relative to its lightest step.
pub fn rebate_tone(cfg: &PipelineConfig, ctx: &PipelineContext) -> Vec<[f32; 3]> {
    const REBATE_BASE_MARGIN: f32 = 0.15;
    let mode = cfg.process.process_mode;
    let Some(bounds) = ctx.metrics.final_bounds else {
        return finish::linear_carrier_tone();
    };
    if !(mode == ProcessMode::C41 || mode == ProcessMode::Bw)
        || cfg.process.positive_source
        || cfg.exposure.render_intent == RenderIntent::Flat
        || !matches!(
            cfg.altproc.alt_process,
            AltProcess::None | AltProcess::Sabattier
        )
    {
        return finish::linear_carrier_tone();
    }
    let floors = [bounds.floors.0, bounds.floors.1, bounds.floors.2];
    let ceils = [bounds.ceils.0, bounds.ceils.1, bounds.ceils.2];
    let t = finish::carrier_tone_exposures();
    let mut strip = Vec::with_capacity(t.len() * 3);
    for &e in &t {
        let lg = negpy_core::mathf::log10f(e.max(1e-6));
        for c in 0..3 {
            strip.push(1.0 + (REBATE_BASE_MARGIN + lg) / (ceils[c] - floors[c]).max(1e-6));
        }
    }
    let strip = Image::new(1, t.len(), strip);
    let mut strip_ctx = PipelineContext::new((1, t.len()), 1.0, mode);
    strip_ctx.metrics = ctx.metrics.clone();
    let mut tone_cfg = cfg.clone();
    tone_cfg.exposure.contrast_mask_strength = 0.0;
    tone_cfg.exposure.diffusion = 0.0;
    tone_cfg.local.masks.clear();
    let mut inputs = exposure::ExposureInputs::new(&tone_cfg, &strip_ctx);
    // `PhotometricProcessor` alone: the hue trim rides the exposure stage, not the ramp.
    inputs.hue_trim = 0.0;
    let mut printed = exposure::exposure_with(&strip, &inputs, &mut strip_ctx).unwrap_or(strip);
    if (cfg.lab.saturation != 1.0 || cfg.lab.skin_protection > 0.0)
        && let Some(out) = lab::saturation(&printed, cfg.lab.saturation, cfg.lab.skin_protection)
    {
        printed = out;
    }
    if let Some(out) = crate::pipeline::toning::toning_stage(
        &printed,
        &cfg.toning,
        cfg.altproc.alt_process,
        &strip_ctx,
    ) {
        printed = out;
    }
    let first = [
        printed.data[0].max(1e-6),
        printed.data[1].max(1e-6),
        printed.data[2].max(1e-6),
    ];
    printed
        .data
        .as_chunks::<3>()
        .0
        .iter()
        .map(|p| [p[0] / first[0], p[1] / first[1], p[2] / first[2]])
        .collect()
}

/// A preview: the source scaled to `max_edge` on its long side, then the engine.
pub fn render_preview(
    source: &Image,
    cfg: &PipelineConfig,
    max_edge: usize,
) -> Result<RenderOutput, EngineError> {
    let (h, w) = source.shape();
    let long = h.max(w);
    let img = if max_edge > 0 && long > max_edge {
        let scale = max_edge as f64 / long as f64;
        let (dh, dw) = (
            ((h as f64 * scale).round() as usize).max(1),
            ((w as f64 * scale).round() as usize).max(1),
        );
        let data = negpy_core::resize::resize_area(&source.data, h, w, 3, dh, dw)
            .expect("resize of a non-empty image");
        Image::new(dh, dw, data)
    } else {
        source.clone()
    };
    let mut ctx = PipelineContext::new(
        img.shape(),
        render_scale_factor(img.shape()),
        cfg.process.process_mode,
    );
    let image = process(&img, cfg, &mut ctx)?;
    Ok(RenderOutput {
        image,
        metrics_json: metrics_json(&ctx),
    })
}

fn metrics_json(ctx: &PipelineContext) -> String {
    let m = &ctx.metrics;
    let bounds = |b: &Option<crate::context::LogNegativeBounds>| {
        b.map(|b| serde_json::json!({"floors": [b.floors.0, b.floors.1, b.floors.2], "ceils": [b.ceils.0, b.ceils.1, b.ceils.2]}))
    };
    serde_json::json!({
        "final_bounds": bounds(&m.final_bounds),
        "pre_trim_bounds": bounds(&m.pre_trim_bounds),
        "norm_density_range": m.norm_density_range,
        "metered_anchor": m.metered_anchor,
        "textural_range": m.textural_range,
        "shadow_point": m.shadow_point,
        "highlight_point": m.highlight_point,
        "print_slopes": m.print_slopes.map(|s| [s.0, s.1, s.2]),
        "scan_clip_fractions": m.scan_clip_fractions.map(|s| [s.0, s.1, s.2]),
        "active_roi": m.active_roi.map(|r| [r.0, r.1, r.2, r.3]),
    })
    .to_string()
}

/// The desktop's default edit as flat JSON.
pub fn default_edit_json() -> String {
    crate::config::default_edit_json()
}
