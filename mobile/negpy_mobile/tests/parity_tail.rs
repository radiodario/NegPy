//! The tail of the pipeline against the desktop's fixtures: each stage from clahe to the
//! output encode reproduces the desktop CPU engine's array bit for bit.

use negpy_mobile::config::AltProcess;
use negpy_mobile::fixture::{Fixture, fixture_names};
use negpy_mobile::pipeline::{altprocess, finish, lab};

/// The cases that ran the creative stages (the flat intent skips them).
fn tail_cases() -> Vec<(String, Fixture)> {
    let cases: Vec<(String, Fixture)> = fixture_names()
        .into_iter()
        .filter(|n| n != "defaults")
        .map(|n| {
            let f = Fixture::load(&n);
            (n, f)
        })
        .filter(|(_, f)| f.has("lab"))
        .collect();
    assert!(cases.len() >= 8, "the tail fixtures are missing");
    cases
}

fn case(name: &str) -> Fixture {
    Fixture::load(name)
}

#[test]
fn clahe_matches_the_desktop() {
    for (name, f) in tail_cases() {
        let strength = f.edit().config.lab.clahe_strength;
        let out = lab::clahe(&f.image("exposure"), strength);
        assert_eq!(out.max_abs_diff(&f.image("clahe")), 0.0, "{name}");
        if strength <= 0.0 {
            assert_eq!(
                out,
                f.image("exposure"),
                "{name}: strength 0 passes through"
            );
        }
    }
    let f = case("c41_manual");
    assert_eq!(f.edit().config.lab.clahe_strength, 0.3);
    assert!(f.image("clahe").max_abs_diff(&f.image("exposure")) > 0.0);
}

#[test]
fn lab_matches_the_desktop() {
    for (name, f) in tail_cases() {
        let cfg = f.edit().config;
        let mut ctx = f.context_after_base();
        let out = lab::lab(&f.image("clahe"), &cfg.lab, &mut ctx);
        assert_eq!(out.max_abs_diff(&f.image("lab")), 0.0, "{name}");
        assert_eq!(
            ctx.metrics.auto_masking, None,
            "{name}: Auto Masking is off"
        );
    }
    let f = case("c41_manual");
    let cfg = f.edit().config.lab;
    assert_eq!(
        (cfg.saturation, cfg.sharpen, cfg.skin_protection),
        (1.2, 0.5, 0.5)
    );
}

#[test]
fn print_stages_match_the_desktop() {
    for (name, f) in tail_cases() {
        let cfg = f.edit().config;
        let ctx = f.context_after_base();
        let out = altprocess::print_stages(&f.image("lab"), &cfg, &ctx);
        assert_eq!(out.max_abs_diff(&f.image("print")), 0.0, "{name}");
    }
    for (name, alt) in [
        ("bw_lith", AltProcess::Lith),
        ("bw_cyanotype", AltProcess::Cyanotype),
        ("bw_sabattier", AltProcess::Sabattier),
    ] {
        let f = case(name);
        assert_eq!(f.edit().config.altproc.alt_process, alt);
        assert!(
            f.image("print").max_abs_diff(&f.image("lab")) > 0.0,
            "{name}"
        );
    }
    let toning = case("bw_default").edit().config.toning;
    assert_eq!(
        (toning.sepia_strength, toning.selenium_strength),
        (0.5, 0.3)
    );
    let toning = case("c41_manual").edit().config.toning;
    assert_eq!(
        (toning.shadow_tint_hue, toning.shadow_tint_strength),
        (200.0, 0.3)
    );
}

#[test]
fn finish_matches_the_desktop() {
    for (name, f) in tail_cases() {
        let cfg = f.edit().config;
        let ctx = f.context_after_base();
        let tone = f.has("rebate_tone").then(|| {
            let (shape, data) = f.array("rebate_tone");
            assert_eq!(shape, [finish::CARRIER_TONE_SAMPLES, 3]);
            data.as_chunks::<3>().0.to_vec()
        });
        let out = finish::finish(&f.image("crop"), &cfg, tone.as_deref(), &ctx);
        assert_eq!(out.max_abs_diff(&f.image("finish")), 0.0, "{name}");
    }
    let f = case("c41_manual");
    assert_eq!(f.edit().config.finish.vignette_stops, 0.5);
    assert!(f.image("finish").max_abs_diff(&f.image("crop")) > 0.0);
    let f = case("c41_carrier");
    assert_eq!(f.edit().config.finish.carrier_width, 2.0);
    assert!(f.has("rebate_tone"));
    assert!(f.image("finish").max_abs_diff(&f.image("crop")) > 0.0);
}

#[test]
fn output_encode_matches_the_desktop() {
    for (name, f) in tail_cases() {
        let out = finish::encode_output(&f.image("finish"));
        assert_eq!(out.max_abs_diff(&f.image("encoded")), 0.0, "{name}");
        assert_eq!(
            out,
            f.image("final"),
            "{name}: the encode is the engine's output"
        );
    }
}

#[test]
fn the_paper_margin_is_the_mat_color() {
    let cfg = case("c41_carrier").edit().config;
    assert_eq!(
        finish::effective_paper_linear(&cfg.finish, &cfg.toning),
        (1.0, 1.0, 1.0)
    );
}
