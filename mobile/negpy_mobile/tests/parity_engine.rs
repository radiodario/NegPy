//! The whole engine, source in and print out, against the desktop's `final` array.

use negpy_mobile::fixture::{Fixture, fixture_names};
use negpy_mobile::pipeline::engine;

#[test]
fn every_fixture_renders_as_the_desktop_did() {
    let mut worst = Vec::new();
    for name in fixture_names() {
        if name == "defaults" {
            continue;
        }
        let f = Fixture::load(&name);
        let cfg = f.edit().config;
        let mut ctx = f.fresh_context();
        let out = engine::process(&f.image("source"), &cfg, &mut ctx).unwrap();
        let expected = f.image("final");
        assert_eq!(out.shape(), expected.shape(), "{name}: shape");
        worst.push((name.clone(), out.max_abs_diff(&expected)));
    }
    assert!(!worst.is_empty());
    let off: Vec<_> = worst.iter().filter(|(_, d)| *d != 0.0).collect();
    assert!(off.is_empty(), "max abs diff against the desktop: {off:?}");
}

/// The engine's stages, one at a time on the fixture's own source, against each intermediate
/// array the desktop wrote.
#[test]
fn every_stage_of_the_engine_matches_the_fixture() {
    use negpy_mobile::config::RenderIntent;
    use negpy_mobile::pipeline::{altprocess, exposure, finish, geometry, lab, normalization};
    let mut off = Vec::new();
    for name in fixture_names() {
        if name == "defaults" {
            continue;
        }
        let f = Fixture::load(&name);
        let cfg = f.edit().config;
        let img = f.image("source");
        let mut ctx = f.fresh_context();
        let mut note = |stage: &str, d: f32| {
            if d != 0.0 {
                off.push((name.clone(), stage.to_string(), d));
            }
        };
        ctx.metrics.base_log = normalization::base_log(&img);
        let cur = geometry::geometry(&img, &cfg.geometry, &mut ctx);
        note("geometry", cur.max_abs_diff(&f.image("geometry")));
        let cur = normalization::normalize(&cur, &cfg.process, &mut ctx);
        note("base", cur.max_abs_diff(&f.image("base")));
        let flat = cfg.exposure.render_intent == RenderIntent::Flat;
        if !flat {
            normalization::build_contrast_mask(&img, &cfg, &mut ctx, cur.shape());
            normalization::build_diffusion(&img, &cfg, &mut ctx, cur.shape());
        }
        let mut cur = exposure::exposure(&cur, &cfg, &mut ctx).unwrap();
        note("exposure", cur.max_abs_diff(&f.image("exposure")));
        if flat {
            continue;
        }
        cur = lab::clahe(&cur, cfg.lab.clahe_strength);
        note("clahe", cur.max_abs_diff(&f.image("clahe")));
        cur = lab::lab(&cur, &cfg.lab, &mut ctx);
        note("lab", cur.max_abs_diff(&f.image("lab")));
        cur = altprocess::print_stages(&cur, &cfg, &ctx);
        note("print", cur.max_abs_diff(&f.image("print")));
        cur = geometry::crop(&cur, &ctx);
        note("crop", cur.max_abs_diff(&f.image("crop")));
        let tone = (cfg.finish.carrier_width > 0.0).then(|| engine::rebate_tone(&cfg, &ctx));
        if f.has("rebate_tone") {
            let (_, want) = f.array("rebate_tone");
            let got: Vec<f32> = tone.iter().flatten().flatten().copied().collect();
            note("rebate_tone", if got == want { 0.0 } else { f32::NAN });
        }
        cur = finish::finish(&cur, &cfg, tone.as_deref(), &ctx);
        note("finish", cur.max_abs_diff(&f.image("finish")));
        cur = finish::encode_output(&cur);
        note("encoded", cur.max_abs_diff(&f.image("encoded")));
    }
    assert!(off.is_empty(), "stages off the desktop: {off:?}");
}
