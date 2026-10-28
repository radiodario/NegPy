//! The exposure stage renders the desktop engine's `exposure` array from its `base` array.
//!
//! The context holds the config and the base stage's metrics at float32; the desktop solved
//! the curve from Python floats. `exact` rebuilds those inputs at float64 from the fixture's
//! JSON, which is where the port is held to the bit.

use negpy_mobile::config::ProcessMode;
use negpy_mobile::fixture::{Fixture, fixture_names, neutral_axis_of};
use negpy_mobile::pipeline::exposure::{
    Bounds64, Exposure, ExposureInputs, ExposureMetrics, LocalMask64, NeutralAxis64, exposure,
    exposure_with, limited_mask_params,
};
use serde_json::Value;

fn cases() -> Vec<String> {
    fixture_names()
        .into_iter()
        .filter(|n| n != "defaults")
        .collect()
}

fn bounds64(v: Option<&Value>) -> Option<Bounds64> {
    let v = v?;
    let rgb = |k: &str| -> Option<[f64; 3]> {
        let a = v.get(k)?.as_array()?;
        Some([a[0].as_f64()?, a[1].as_f64()?, a[2].as_f64()?])
    };
    Some(Bounds64 {
        floors: rgb("floors")?,
        ceils: rgb("ceils")?,
    })
}

/// The stage's scalar inputs at the precision the desktop rendered with.
fn exact_inputs(f: &Fixture) -> ExposureInputs {
    let flat = f.config().as_object().expect("flat config");
    let m = f.metrics();
    let f64_of = |k: &str| m.get(k).and_then(Value::as_f64);
    let refs = m.get("neutral_axis_refs").and_then(|v| {
        let a = v.as_array()?;
        let rgb = |x: &Value| -> Option<[f64; 3]> {
            let a = x.as_array()?;
            Some([a[0].as_f64()?, a[1].as_f64()?, a[2].as_f64()?])
        };
        neutral_axis_of(v)?;
        Some(NeutralAxis64 {
            mid: rgb(&a[0])?,
            shadow: rgb(&a[1])?,
            highlight: rgb(&a[2]),
            confidence: a[3].as_f64()?,
        })
    });
    let masks: Vec<LocalMask64> = flat
        .get("local_masks")
        .and_then(|v| v.get("masks"))
        .cloned()
        .map(serde_json::from_value)
        .and_then(Result::ok)
        .unwrap_or_default();
    let cfg = &f.edit().config;
    ExposureInputs {
        exposure: Exposure::from_flat(flat),
        process_mode: cfg.process.process_mode,
        source_color: cfg.process.source_color,
        hue_trim: flat.get("hue_trim").and_then(Value::as_f64).unwrap_or(0.0),
        masks,
        metrics: ExposureMetrics {
            final_bounds: bounds64(m.get("final_bounds")),
            log_bounds: bounds64(m.get("log_bounds")),
            norm_density_range: f64_of("norm_density_range"),
            metered_anchor: f64_of("metered_anchor"),
            textural_range: f64_of("textural_range"),
            shadow_point: f64_of("shadow_point"),
            highlight_point: f64_of("highlight_point"),
            neutral_axis_refs: refs,
            contrast_mask_centre: f64_of("contrast_mask_centre").unwrap_or(0.5),
        },
    }
}

#[test]
fn exact_from_the_desktop_precision_inputs() {
    for name in cases() {
        let f = Fixture::load(&name);
        let mut ctx = f.context_after_base();
        let inputs = exact_inputs(&f);
        let out = exposure_with(&f.image("base"), &inputs, &mut ctx).expect("a supported stock");
        let want = f.image("exposure");
        assert_eq!(out.shape(), want.shape(), "{name}: shape");
        let diff = out.max_abs_diff(&want);
        assert_eq!(diff, 0.0, "{name}: max abs diff {diff}");
    }
}

#[test]
fn print_slopes_match_the_desktop() {
    for name in cases() {
        let f = Fixture::load(&name);
        let mut ctx = f.context_after_base();
        let inputs = exact_inputs(&f);
        exposure_with(&f.image("base"), &inputs, &mut ctx).expect("a supported stock");
        let want = f
            .metrics()
            .get("print_slopes")
            .and_then(Value::as_array)
            .cloned();
        match (want, ctx.metrics.print_slopes) {
            (None, got) => assert_eq!(got, None, "{name}: no slopes on the flat master"),
            (Some(w), Some(got)) => {
                let w: Vec<f32> = w.iter().map(|v| v.as_f64().unwrap() as f32).collect();
                assert_eq!(
                    (got.0, got.1, got.2),
                    (w[0], w[1], w[2]),
                    "{name}: print_slopes"
                );
            }
            (Some(_), None) => panic!("{name}: print_slopes not set"),
        }
    }
}

#[test]
fn from_the_context_within_float32_of_the_desktop() {
    for name in cases() {
        let f = Fixture::load(&name);
        let mut ctx = f.context_after_base();
        let out =
            exposure(&f.image("base"), &f.edit().config, &mut ctx).expect("a supported stock");
        let want = f.image("exposure");
        let diff = out.max_abs_diff(&want);
        // The context and the typed config hold the curve's scalar inputs at float32, so the
        // kernel's float64 parameters land a few ulps off the desktop's.
        assert!(diff <= 2.5e-7, "{name}: max abs diff {diff}");
    }
}

#[test]
fn a_lut_stock_is_unsupported() {
    let f = Fixture::load("c41_default");
    let mut ctx = f.context_after_base();
    let mut edit = f.edit();
    edit.set("paper_profile", Value::from("kodak_endura"));
    assert_eq!(edit.config.process.process_mode, ProcessMode::C41);
    let err = exposure(&f.image("base"), &edit.config, &mut ctx).unwrap_err();
    assert_eq!(err.0, "kodak_e4070");
    edit.set("render_intent", Value::from("flat"));
    assert!(exposure(&f.image("base"), &edit.config, &mut ctx).is_err());
}

/// Every shape, a tone limit of each kind, and an inverted mask, on the `c41_masks` frame.
fn every_mask_kind() -> Vec<LocalMask64> {
    serde_json::from_str(
        r#"[
        {"vertices": [[0.1, 0.2], [0.6, 0.15], [0.7, 0.6], [0.4, 0.9], [0.05, 0.5]], "stops": -0.7, "grade": 12.0, "flash": 0.2, "shape": "polygon", "feather": 0.03},
        {"vertices": [[0.2, 0.3], [0.9, 0.7]], "stops": 0.4, "shape": "gradient"},
        {"vertices": [[0.5, 0.5], [0.8, 0.5], [0.8, 0.8]], "stops": 0.5, "shape": "oval", "feather": 0.05, "invert": true, "key": "highlights", "key_zone": 6.0, "key_softness": 1.0},
        {"vertices": [[0.3, 0.3], [0.5, 0.3], [0.5, 0.6]], "stops": 0.25, "grade": -10.0, "shape": "oval", "feather": 0.02, "key": "shadows", "key_zone": 4.5, "key_softness": 1.5}
    ]"#,
    )
    .expect("mask json")
}

#[test]
fn tone_limited_mask_edges_match_the_desktop() {
    let f = Fixture::load("c41_masks");
    let inputs = exact_inputs(&f);
    let params = limited_mask_params(
        &every_mask_kind(),
        &inputs.exposure,
        inputs.process_mode,
        &inputs.metrics,
    )
    .expect("two limited masks");
    let want = [
        [0.5, 0.0, 0.41329294899435565, 0.3053682330084299],
        [0.25, -10.0, 0.4413680406316057, 0.6069894113162804],
    ];
    assert_eq!(params.len(), want.len());
    for (got, want) in params.iter().zip(&want) {
        for (g, w) in got.iter().zip(want) {
            // The edges come off a grid the desktop runs through NumPy's SIMD exp and pow.
            assert!((g - w).abs() <= 1e-9, "{got:?} vs {want:?}");
        }
    }
}

#[test]
fn tone_limited_masks_render() {
    let f = Fixture::load("c41_masks");
    let mut ctx = f.context_after_base();
    let mut inputs = exact_inputs(&f);
    let plain = exposure_with(&f.image("base"), &inputs, &mut ctx).expect("a supported stock");
    inputs.masks = every_mask_kind();
    let keyed = exposure_with(&f.image("base"), &inputs, &mut ctx).expect("a supported stock");
    assert!(keyed.max_abs_diff(&plain) > 0.0);
    assert!(keyed.data.iter().all(|v| v.is_finite()));
}
