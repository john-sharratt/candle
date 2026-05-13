mod harness;

use serde_json::json;

#[test]
fn convert_fahrenheit_to_celsius() {
    let resp = harness::expect_success(harness::invoke(
        "unit_convert",
        json!({"value": 32.0, "from": "fahrenheit", "to": "celsius"}),
    ));
    let result = resp["result"].as_f64().unwrap();
    assert!((result - 0.0).abs() < 0.01, "got {result}");
}

#[test]
fn convert_kelvin_to_celsius() {
    let resp = harness::expect_success(harness::invoke(
        "unit_convert",
        json!({"value": 273.15, "from": "kelvin", "to": "celsius"}),
    ));
    let result = resp["result"].as_f64().unwrap();
    assert!((result - 0.0).abs() < 0.01, "got {result}");
}

#[test]
fn convert_hours_to_seconds() {
    let resp = harness::expect_success(harness::invoke(
        "unit_convert",
        json!({"value": 1.0, "from": "hour", "to": "second"}),
    ));
    let result = resp["result"].as_f64().unwrap();
    assert!((result - 3600.0).abs() < 0.01, "got {result}");
}

#[test]
fn convert_mib_to_bytes() {
    let resp = harness::expect_success(harness::invoke(
        "unit_convert",
        json!({"value": 1.0, "from": "mib", "to": "byte"}),
    ));
    let result = resp["result"].as_f64().unwrap();
    assert!((result - 1_048_576.0).abs() < 0.5, "got {result}");
}

#[test]
fn convert_gib_vs_gb() {
    let gib = harness::expect_success(harness::invoke(
        "unit_convert",
        json!({"value": 1.0, "from": "gib", "to": "byte"}),
    ));
    let gb = harness::expect_success(harness::invoke(
        "unit_convert",
        json!({"value": 1.0, "from": "gb", "to": "byte"}),
    ));
    let gib_bytes = gib["result"].as_f64().unwrap();
    let gb_bytes = gb["result"].as_f64().unwrap();
    assert!(
        (gib_bytes - gb_bytes).abs() > 100.0,
        "GiB ({gib_bytes}) should differ from GB ({gb_bytes})"
    );
}

#[test]
fn convert_feet_to_meters() {
    let resp = harness::expect_success(harness::invoke(
        "unit_convert",
        json!({"value": 1.0, "from": "foot", "to": "meter"}),
    ));
    let result = resp["result"].as_f64().unwrap();
    assert!((result - 0.3048).abs() < 0.001, "got {result}");
}

#[test]
fn convert_same_unit() {
    let resp = harness::expect_success(harness::invoke(
        "unit_convert",
        json!({"value": 42.0, "from": "km", "to": "km"}),
    ));
    let result = resp["result"].as_f64().unwrap();
    assert!((result - 42.0).abs() < 0.001, "got {result}");
}

#[test]
fn convert_km_to_miles() {
    let resp = harness::expect_success(harness::invoke("unit_convert", json!({
        "value": 1.0, "from": "km", "to": "mile"
    })));
    let result = resp["result"].as_f64().unwrap();
    assert!((result - 0.621371).abs() < 0.001, "got {result}");
}

#[test]
fn convert_celsius_to_fahrenheit() {
    let resp = harness::expect_success(harness::invoke("unit_convert", json!({
        "value": 100.0, "from": "celsius", "to": "fahrenheit"
    })));
    let result = resp["result"].as_f64().unwrap();
    assert!((result - 212.0).abs() < 0.01, "got {result}");
}

#[test]
fn convert_celsius_to_kelvin() {
    let resp = harness::expect_success(harness::invoke("unit_convert", json!({
        "value": 0.0, "from": "celsius", "to": "kelvin"
    })));
    let result = resp["result"].as_f64().unwrap();
    assert!((result - 273.15).abs() < 0.01);
}

#[test]
fn convert_kg_to_lb() {
    let resp = harness::expect_success(harness::invoke("unit_convert", json!({
        "value": 1.0, "from": "kg", "to": "lb"
    })));
    let result = resp["result"].as_f64().unwrap();
    assert!((result - 2.20462).abs() < 0.001);
}

#[test]
fn convert_dimension_mismatch() {
    let resp = harness::invoke("unit_convert", json!({"value": 1.0, "from": "km", "to": "kg"}));
    harness::expect_error(&resp, "dimension_mismatch");
}

#[test]
fn convert_unknown_unit() {
    let resp = harness::invoke("unit_convert", json!({"value": 1.0, "from": "parsec", "to": "km"}));
    harness::expect_error(&resp, "unknown_unit");
}

#[test]
fn convert_bytes_to_mib() {
    let resp = harness::expect_success(harness::invoke("unit_convert", json!({
        "value": 1048576.0, "from": "byte", "to": "mib"
    })));
    let result = resp["result"].as_f64().unwrap();
    assert!((result - 1.0).abs() < 0.0001);
}
