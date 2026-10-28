//! The geometry stage renders each fixture's `geometry` array from `source` bit for bit and
//! records the crop ROI the desktop recorded; the crop stage slices `print` into `crop`.

use negpy_mobile::context::GeometryParams;
use negpy_mobile::fixture::{Fixture, fixture_names};
use negpy_mobile::image::{Image, Roi};
use negpy_mobile::pipeline::geometry::{
    apply_geometry, compute_distortion_scale, compute_geometry_crop_rect, crop, fliplr, flipud,
    geometry, keystone_matrix, rot90,
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

fn rendered_fixtures() -> Vec<(String, Fixture)> {
    fixture_names()
        .into_iter()
        .map(|n| (n.clone(), Fixture::load(&n)))
        .filter(|(_, f)| f.has("source"))
        .collect()
}

#[test]
fn geometry_matches_every_fixture() {
    for (name, f) in rendered_fixtures() {
        let cfg = f.edit().config.geometry;
        let mut ctx = f.fresh_context();
        let out = geometry(&f.image("source"), &cfg, &mut ctx);
        let want = f.image("geometry");
        assert_eq!(out.shape(), want.shape(), "{name}");
        assert_eq!(out.max_abs_diff(&want), 0.0, "{name}");

        let m = f.metrics();
        let roi = roi_of(&m["active_roi"]);
        assert_eq!(ctx.active_roi, roi, "{name}: active_roi");
        assert_eq!(ctx.metrics.active_roi, roi, "{name}: metrics.active_roi");

        let gp = &m["geometry_params"];
        let want_params = GeometryParams {
            rotation: gp["rotation"].as_i64().unwrap() as i32,
            fine_rotation: gp["fine_rotation"].as_f64().unwrap() as f32,
            flip_horizontal: gp["flip_horizontal"].as_bool().unwrap(),
            flip_vertical: gp["flip_vertical"].as_bool().unwrap(),
            converge_v: gp["converge_v"].as_f64().unwrap() as f32,
            converge_h: gp["converge_h"].as_f64().unwrap() as f32,
            distortion_k1: m["distortion_k1"].as_f64().unwrap() as f32,
        };
        assert_eq!(ctx.metrics.geometry, want_params, "{name}: geometry_params");
    }
}

#[test]
fn crop_matches_every_fixture() {
    for (name, f) in rendered_fixtures() {
        if !f.has("print") {
            continue;
        }
        let ctx = f.context_after_base();
        let out = crop(&f.image("print"), &ctx);
        let want = f.image("crop");
        assert_eq!(out.shape(), want.shape(), "{name}");
        assert_eq!(out.max_abs_diff(&want), 0.0, "{name}");
    }
}

#[test]
fn the_manual_fixture_crops_its_rect() {
    let f = Fixture::load("c41_manual");
    let ctx = f.context_after_base();
    assert_eq!(ctx.active_roi, Some((7, 68, 8, 97)));
    let out = crop(&f.image("print"), &ctx);
    assert_eq!(out.shape(), (61, 89));
}

#[test]
fn the_fixture_geometry_replays_through_the_shared_chain() {
    let f = Fixture::load("c41_carrier");
    let mut ctx = f.fresh_context();
    let staged = geometry(&f.image("source"), &f.edit().config.geometry, &mut ctx);
    let replayed = apply_geometry(&f.image("source"), &ctx.metrics.geometry);
    assert_eq!(staged, replayed);
}

/// `np.rot90` of `[[0, 1, 2], [3, 4, 5]]` for each k, pixel values as their index.
#[test]
fn rot90_turns_counterclockwise() {
    let px = |v: usize| [v as f32; 3];
    let img = Image::new(2, 3, (0..6).flat_map(px).collect());
    let values = |img: &Image| -> Vec<Vec<f32>> {
        (0..img.height)
            .map(|y| (0..img.width).map(|x| img.pixel(y, x)[0]).collect())
            .collect()
    };
    assert_eq!(
        values(&rot90(&img, 1)),
        vec![vec![2.0, 5.0], vec![1.0, 4.0], vec![0.0, 3.0]]
    );
    assert_eq!(
        values(&rot90(&img, 2)),
        vec![vec![5.0, 4.0, 3.0], vec![2.0, 1.0, 0.0]]
    );
    assert_eq!(
        values(&rot90(&img, 3)),
        vec![vec![3.0, 0.0], vec![4.0, 1.0], vec![5.0, 2.0]]
    );
    assert_eq!(rot90(&img, -1), rot90(&img, 3));
    assert_eq!(rot90(&img, 4), img);
    assert_eq!(
        values(&fliplr(&img)),
        vec![vec![2.0, 1.0, 0.0], vec![5.0, 4.0, 3.0]]
    );
    assert_eq!(
        values(&flipud(&img)),
        vec![vec![3.0, 4.0, 5.0], vec![0.0, 1.0, 2.0]]
    );
}

/// Values the desktop's `geometry.logic` computes for the same arguments.
#[test]
fn scalar_helpers_match_the_desktop() {
    assert_eq!(compute_distortion_scale(0.05, 108, 72), 0.9570950398753579);
    assert_eq!(compute_distortion_scale(-0.08, 108, 72), 1.0258403030877727);

    let close = |a: f64, b: f64| (a - b).abs() <= 1e-12 * b.abs().max(1.0);
    // NumPy multiplies and inverts these matrices on BLAS, whose summation order is its own.
    let m = keystone_matrix(3.0, -2.0, 108, 72);
    let want = [
        0.9900379590205854,
        0.04324867023286163,
        -1.5554722594320172,
        -0.014113855202069403,
        1.0399944579486269,
        0.7852075811095496,
        -0.00037478103837298163,
        0.0008424144620660549,
        1.0002338167118465,
    ];
    for (i, (a, b)) in m.iter().zip(want).enumerate() {
        assert!(close(*a, b), "keystone_matrix[{i}]: {a} vs {b}");
    }
    assert_eq!(
        keystone_matrix(0.0, 0.0, 108, 72),
        [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
    );

    let rect = compute_geometry_crop_rect(1.5, 0.0, 0.0, 108, 72);
    let want = (
        0.017252340868372486,
        0.03268402861431241,
        0.9827476591316275,
        0.9673159713856876,
    );
    for (a, b) in [
        (rect.0, want.0),
        (rect.1, want.1),
        (rect.2, want.2),
        (rect.3, want.3),
    ] {
        assert!(close(a, b), "crop rect: {a} vs {b}");
    }
    let rect = compute_geometry_crop_rect(-2.0, 3.0, -2.0, 108, 72);
    let want = (
        0.0334750107530013,
        0.04479167843237519,
        0.9665249892469987,
        0.9552083215676248,
    );
    for (a, b) in [
        (rect.0, want.0),
        (rect.1, want.1),
        (rect.2, want.2),
        (rect.3, want.3),
    ] {
        assert!(close(a, b), "crop rect: {a} vs {b}");
    }
    assert_eq!(
        compute_geometry_crop_rect(0.0, 0.0, 0.0, 108, 72),
        (0.0, 0.0, 1.0, 1.0)
    );
}
