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
        let diff = out.max_abs_diff(&expected);
        worst.push((name.clone(), diff));
        // The config and metrics scalars are f32 where the desktop solves from Python floats.
        assert!(diff <= 5.0e-7, "{name}: max abs diff {diff}");
    }
    assert!(!worst.is_empty());
}
