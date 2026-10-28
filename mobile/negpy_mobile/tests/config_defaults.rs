//! The typed defaults match the desktop's `WorkspaceConfig()` field for field, and a flat dict
//! survives a round trip untouched.

use negpy_mobile::config::{Edit, PipelineConfig};
use negpy_mobile::fixture::Fixture;
use serde_json::Value;

fn defaults() -> Value {
    let f = Fixture::load("defaults");
    f.json["workspace"].clone()
}

#[test]
fn typed_defaults_match_the_desktop() {
    let flat = defaults();
    let from_desktop = PipelineConfig::from_flat(flat.as_object().unwrap());
    assert_eq!(from_desktop, PipelineConfig::default());
}

#[test]
fn every_typed_field_names_a_desktop_key() {
    let flat = defaults();
    let keys = flat.as_object().unwrap();
    for section in [
        serde_json::to_value(negpy_mobile::config::ProcessConfig::default()).unwrap(),
        serde_json::to_value(negpy_mobile::config::ExposureConfig::default()).unwrap(),
        serde_json::to_value(negpy_mobile::config::GeometryConfig::default()).unwrap(),
        serde_json::to_value(negpy_mobile::config::LabConfig::default()).unwrap(),
        serde_json::to_value(negpy_mobile::config::AltProcessConfig::default()).unwrap(),
        serde_json::to_value(negpy_mobile::config::ToningConfig::default()).unwrap(),
        serde_json::to_value(negpy_mobile::config::FinishConfig::default()).unwrap(),
        serde_json::to_value(negpy_mobile::config::ExportFields::default()).unwrap(),
    ] {
        for (key, value) in section.as_object().unwrap() {
            let desktop = keys
                .get(key)
                .unwrap_or_else(|| panic!("{key} is not a desktop field"));
            match (desktop.as_f64(), value.as_f64()) {
                (Some(a), Some(b)) => assert_eq!(a, b, "{key}: desktop default differs"),
                _ => assert_eq!(desktop, value, "{key}: desktop default differs"),
            }
        }
    }
}

#[test]
fn a_flat_dict_round_trips_untouched() {
    let text = serde_json::to_string(&defaults()).unwrap();
    let edit = Edit::from_json(&text).unwrap();
    let back: Value = serde_json::from_str(&edit.to_json()).unwrap();
    assert_eq!(back, defaults());
}

#[test]
fn legacy_values_read_as_the_desktop_reads_them() {
    let mut flat = defaults().as_object().unwrap().clone();
    flat.insert("process_mode".into(), Value::from("B&W"));
    flat.insert("grade".into(), Value::from(2.0));
    let cfg = PipelineConfig::from_flat(&flat);
    assert_eq!(
        cfg.process.process_mode,
        negpy_mobile::config::ProcessMode::Bw
    );
    assert_eq!(cfg.exposure.grade, 110.0);
}

#[test]
fn the_default_edit_reads_back_as_the_defaults() {
    let text = negpy_mobile::config::default_edit_json();
    let edit = Edit::from_json(&text).unwrap();
    assert_eq!(edit.config, PipelineConfig::default());
    let desktop = defaults();
    for (key, value) in edit.flat.iter() {
        if key == "local_masks" {
            continue;
        }
        let d = desktop
            .get(key)
            .unwrap_or_else(|| panic!("{key} is not a desktop field"));
        match (d.as_f64(), value.as_f64()) {
            (Some(a), Some(b)) => assert_eq!(a, b, "{key}"),
            _ => assert_eq!(d, value, "{key}"),
        }
    }
}
