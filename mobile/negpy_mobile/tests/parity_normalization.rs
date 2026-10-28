//! The normalization stage reproduces the desktop's base output and every meter reading, and
//! the Contrast Mask and Diffusion planes the engine builds after it.

use negpy_mobile::context::{LogNegativeBounds, PipelineContext};
use negpy_mobile::fixture::{Fixture, fixture_names};
use negpy_mobile::image::Roi;
use negpy_mobile::pipeline::normalization::{
    Axis, Bounds, Normalized, base_log, build_contrast_mask, build_diffusion, normalize,
    normalize_detailed,
};
use serde_json::Value;

fn roi_of(v: &Value) -> Option<Roi> {
    let a = v.as_array()?;
    Some((
        a[0].as_u64()? as usize,
        a[1].as_u64()? as usize,
        a[2].as_u64()? as usize,
        a[3].as_u64()? as usize,
    ))
}

fn triplet(v: &Value) -> [f64; 3] {
    let a = v.as_array().expect("a triplet");
    [
        a[0].as_f64().unwrap(),
        a[1].as_f64().unwrap(),
        a[2].as_f64().unwrap(),
    ]
}

fn bounds_of(v: &Value) -> Bounds {
    Bounds::new(triplet(&v["floors"]), triplet(&v["ceils"]))
}

fn assert_exact(name: &str, actual: f64, expected: f64) {
    assert!(
        actual.to_bits() == expected.to_bits(),
        "{name}: {actual:?} != {expected:?} (diff {:e})",
        actual - expected
    );
}

fn assert_within(name: &str, actual: f64, expected: f64, rel: f64) {
    let tol = rel * expected.abs().max(1.0);
    assert!(
        (actual - expected).abs() <= tol,
        "{name}: {actual:?} != {expected:?} (diff {:e}, tol {tol:e})",
        actual - expected
    );
}

fn assert_bounds(name: &str, actual: &Bounds, expected: &Bounds, rel: f64) {
    for ch in 0..3 {
        let f = format!("{name}.floors[{ch}]");
        let c = format!("{name}.ceils[{ch}]");
        if rel == 0.0 {
            assert_exact(&f, actual.floors[ch], expected.floors[ch]);
            assert_exact(&c, actual.ceils[ch], expected.ceils[ch]);
        } else {
            assert_within(&f, actual.floors[ch], expected.floors[ch], rel);
            assert_within(&c, actual.ceils[ch], expected.ceils[ch], rel);
        }
    }
}

/// The context the engine hands normalization: fresh, with the geometry stage's crop ROI.
fn context_before_normalization(f: &Fixture) -> PipelineContext {
    let mut ctx = f.fresh_context();
    ctx.active_roi = f.metrics().get("active_roi").and_then(roi_of);
    ctx
}

fn run(name: &str) -> (Fixture, Normalized, PipelineContext) {
    let f = Fixture::load(name);
    let geometry = f.image("geometry");
    let ctx = context_before_normalization(&f);
    let cfg = f.edit().config;
    let n = normalize_detailed(
        &geometry,
        &cfg.process,
        ctx.process_mode,
        ctx.active_roi,
        None,
    );
    let mut ctx_after = ctx.clone();
    let out = normalize(&geometry, &cfg.process, &mut ctx_after);
    assert_eq!(
        out, n.image,
        "{name}: normalize and normalize_detailed disagree"
    );
    (f, n, ctx_after)
}

#[test]
fn every_fixture_reproduces_the_base_stage_pixels() {
    for name in fixture_names() {
        if name == "defaults" {
            continue;
        }
        let (f, n, _) = run(&name);
        let expected = f.image("base");
        let diff = n.image.max_abs_diff(&expected);
        assert_eq!(diff, 0.0, "{name}: base pixels differ by {diff}");
    }
}

#[test]
fn every_fixture_reproduces_the_bounds() {
    for name in fixture_names() {
        if name == "defaults" {
            continue;
        }
        let (f, n, _) = run(&name);
        let m = f.metrics();
        let analyzed = n
            .log_bounds
            .expect("the fixtures hold no roll lock, so every frame analyzes");
        assert_bounds(
            &format!("{name}.log_bounds"),
            &analyzed,
            &bounds_of(&m["log_bounds"]),
            0.0,
        );
        assert_bounds(
            &format!("{name}.log_bounds_base"),
            &n.log_bounds_base,
            &bounds_of(&m["log_bounds_base"]),
            0.0,
        );
        assert_bounds(
            &format!("{name}.pre_trim_bounds"),
            &n.pre_trim_bounds,
            &bounds_of(&m["pre_trim_bounds"]),
            0.0,
        );
        // ProcessConfig holds the White/Black Point offsets as f32; the desktop adds the Python
        // float, so a non-dyadic offset moves the trimmed floors in their last f64 bits.
        let trimmed = f.edit().config.process.white_point_offset != 0.0
            || f.edit().config.process.black_point_offset != 0.0;
        let rel = if trimmed { 1e-8 } else { 0.0 };
        assert_bounds(
            &format!("{name}.final_bounds"),
            &n.final_bounds,
            &bounds_of(&m["final_bounds"]),
            rel,
        );
        assert_exact(
            &format!("{name}.norm_density_range"),
            n.norm_density_range,
            m["norm_density_range"].as_f64().unwrap(),
        );
    }
}

#[test]
fn every_fixture_reproduces_the_meter_readings() {
    for name in fixture_names() {
        if name == "defaults" {
            continue;
        }
        let (f, n, _) = run(&name);
        let m = f.metrics();
        assert_exact(
            &format!("{name}.metered_anchor"),
            n.metered_anchor,
            m["metered_anchor"].as_f64().unwrap(),
        );
        assert_exact(
            &format!("{name}.textural_range"),
            n.textural_range,
            m["textural_range"].as_f64().unwrap(),
        );
        // The shadow and highlight points are float32 percentiles, written to JSON widened.
        assert_exact(
            &format!("{name}.shadow_point"),
            n.shadow_point as f64,
            m["shadow_point"].as_f64().unwrap(),
        );
        assert_exact(
            &format!("{name}.highlight_point"),
            n.highlight_point as f64,
            m["highlight_point"].as_f64().unwrap(),
        );
        let clip = triplet(&m["scan_clip_fractions"]);
        for (ch, (got, want)) in n.scan_clip_fractions.iter().zip(&clip).enumerate() {
            assert_exact(&format!("{name}.scan_clip_fractions[{ch}]"), *got, *want);
        }
        assert!(
            n.density_above_base.is_none(),
            "{name}: no base was read, so no densities above it"
        );
    }
}

#[test]
fn every_fixture_reproduces_the_neutral_axis() {
    for name in fixture_names() {
        if name == "defaults" {
            continue;
        }
        let (f, n, _) = run(&name);
        let expected = &f.metrics()["neutral_axis_refs"];
        match (&n.neutral_axis_refs, expected.as_array()) {
            (None, None) => assert!(expected.is_null(), "{name}: expected an axis"),
            (Some(axis), Some(e)) => {
                let Axis {
                    mid,
                    shadow,
                    highlight,
                    confidence,
                    offset_weight,
                } = axis;
                for (band, got, want) in [("mid", mid, &e[0]), ("shadow", shadow, &e[1])] {
                    let want = triplet(want);
                    for ch in 0..3 {
                        assert_exact(
                            &format!("{name}.neutral_axis.{band}[{ch}]"),
                            got[ch],
                            want[ch],
                        );
                    }
                }
                match (highlight, e[2].is_null()) {
                    (None, true) => {}
                    (Some(got), false) => {
                        let want = triplet(&e[2]);
                        for ch in 0..3 {
                            assert_exact(
                                &format!("{name}.neutral_axis.highlight[{ch}]"),
                                got[ch],
                                want[ch],
                            );
                        }
                    }
                    _ => panic!("{name}: highlight band presence differs"),
                }
                assert_exact(
                    &format!("{name}.neutral_axis.confidence"),
                    *confidence,
                    e[3].as_f64().unwrap(),
                );
                assert!(
                    offset_weight.is_none(),
                    "{name}: a metered axis carries no offset weight"
                );
            }
            (got, _) => panic!("{name}: neutral axis {got:?} against {expected}"),
        }
    }
}

#[test]
fn the_context_metrics_match_the_desktop_metrics() {
    for name in fixture_names() {
        if name == "defaults" {
            continue;
        }
        let (f, _, ctx) = run(&name);
        let want = f.context_after_base().metrics;
        let got = ctx.metrics;
        let same = |a: Option<LogNegativeBounds>, b: Option<LogNegativeBounds>| a == b;
        assert!(same(got.log_bounds, want.log_bounds), "{name}: log_bounds");
        assert!(
            same(got.log_bounds_base, want.log_bounds_base),
            "{name}: log_bounds_base"
        );
        assert!(
            same(got.pre_trim_bounds, want.pre_trim_bounds),
            "{name}: pre_trim_bounds"
        );
        assert_eq!(
            got.norm_density_range, want.norm_density_range,
            "{name}: norm_density_range"
        );
        assert_eq!(
            got.neutral_axis_refs, want.neutral_axis_refs,
            "{name}: neutral_axis_refs"
        );
        assert_eq!(
            got.metered_anchor, want.metered_anchor,
            "{name}: metered_anchor"
        );
        assert_eq!(
            got.textural_range, want.textural_range,
            "{name}: textural_range"
        );
        assert_eq!(got.shadow_point, want.shadow_point, "{name}: shadow_point");
        assert_eq!(
            got.highlight_point, want.highlight_point,
            "{name}: highlight_point"
        );
        assert_eq!(
            got.scan_clip_fractions, want.scan_clip_fractions,
            "{name}: scan_clip_fractions"
        );
    }
}

#[test]
fn the_synthetic_frames_show_no_clear_base() {
    for name in fixture_names() {
        if name == "defaults" {
            continue;
        }
        let f = Fixture::load(&name);
        let read = base_log(&f.image("source"));
        assert!(
            f.metrics()["base_log"].is_null(),
            "{name}: the desktop read a base"
        );
        assert_eq!(read, None, "{name}: base read");
    }
}

#[test]
fn c41_masks_reproduces_the_contrast_mask_and_diffusion_planes() {
    let name = "c41_masks";
    let (f, _, mut ctx) = run(name);
    let source = f.image("source");
    let settings = f.edit().config;
    let render_shape = f.image("geometry").shape();
    build_contrast_mask(&source, &settings, &mut ctx, render_shape);
    build_diffusion(&source, &settings, &mut ctx, render_shape);

    let plane = ctx.metrics.contrast_mask_plane.expect("a mask plane");
    let expected = f.plane("metrics.contrast_mask_plane");
    assert_eq!(plane.height, expected.height);
    assert_eq!(plane.width, expected.width);
    let diff = plane
        .data
        .iter()
        .zip(&expected.data)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0, f32::max);
    assert_eq!(diff, 0.0, "{name}: contrast mask plane differs by {diff}");
    assert_exact(
        "contrast_mask_centre",
        ctx.metrics.contrast_mask_centre as f64,
        f.metrics()["contrast_mask_centre"].as_f64().unwrap(),
    );
    assert_eq!(ctx.metrics.contrast_mask_roi, None);

    let diffusion = ctx.metrics.diffusion_plane.expect("a diffusion plane");
    let expected = f.image("metrics.diffusion_plane");
    let diff = diffusion.max_abs_diff(&expected);
    assert_eq!(diff, 0.0, "{name}: diffusion plane differs by {diff}");
    assert_eq!(ctx.metrics.diffusion_roi, None);
}

#[test]
fn a_frame_without_the_sliders_builds_no_planes() {
    let (f, _, mut ctx) = run("c41_default");
    let settings = f.edit().config;
    let source = f.image("source");
    build_contrast_mask(&source, &settings, &mut ctx, source.shape());
    build_diffusion(&source, &settings, &mut ctx, source.shape());
    assert!(ctx.metrics.contrast_mask_plane.is_none());
    assert!(ctx.metrics.diffusion_plane.is_none());
}
