mod harness;

use serde_json::json;

#[test]
fn calculator_parentheses() {
    // (2 + 3) * 4 = 20
    let resp = harness::expect_success(harness::invoke(
        "calculator",
        json!({"expression": "(2 + 3) * 4"}),
    ));
    assert!((resp["result"].as_f64().unwrap() - 20.0).abs() < 1e-9);
}

#[test]
fn calculator_float_division() {
    // 7.0 / 2.0 = 3.5
    let resp = harness::expect_success(harness::invoke(
        "calculator",
        json!({"expression": "7.0 / 2.0"}),
    ));
    assert!((resp["result"].as_f64().unwrap() - 3.5).abs() < 1e-9);
}

#[test]
fn calculator_nested_power() {
    // 2^3^1 = 8 (evalexpr is right-assoc for ^)
    let resp = harness::expect_success(harness::invoke(
        "calculator",
        json!({"expression": "2 ^ 3"}),
    ));
    assert!((resp["result"].as_f64().unwrap() - 8.0).abs() < 1e-9);
}

#[test]
fn calculator_negative_result() {
    let resp = harness::expect_success(harness::invoke(
        "calculator",
        json!({"expression": "3 - 10"}),
    ));
    assert!((resp["result"].as_f64().unwrap() - (-7.0)).abs() < 1e-9);
}

#[test]
fn calculator_floor_ceil() {
    // Bare `floor`/`ceil` are evalexpr builtins and must evaluate.
    let floor = harness::expect_success(harness::invoke(
        "calculator",
        json!({"expression": "floor(2.7)"}),
    ));
    assert_eq!(floor["result"].as_f64().unwrap(), 2.0);
    let ceil = harness::expect_success(harness::invoke(
        "calculator",
        json!({"expression": "ceil(2.1)"}),
    ));
    assert_eq!(ceil["result"].as_f64().unwrap(), 3.0);
}

#[test]
fn calculator_sqrt() {
    // The bare `sqrt` alias must be bound (evalexpr namespaces it as math::sqrt).
    let resp = harness::expect_success(harness::invoke(
        "calculator",
        json!({"expression": "sqrt(16)"}),
    ));
    assert!((resp["result"].as_f64().unwrap() - 4.0).abs() < 1e-9);

    let big = harness::expect_success(harness::invoke(
        "calculator",
        json!({"expression": "sqrt(1872133)"}),
    ));
    assert!((big["result"].as_f64().unwrap() - 1_368.259_84).abs() < 1e-3);
}

#[test]
fn calculator_trig_and_log() {
    let sin = harness::expect_success(harness::invoke(
        "calculator",
        json!({"expression": "sin(0) * 3"}),
    ));
    assert!(sin["result"].as_f64().unwrap().abs() < 1e-9);

    // log is base 10; ln is natural.
    let log = harness::expect_success(harness::invoke(
        "calculator",
        json!({"expression": "log(1000)"}),
    ));
    assert!((log["result"].as_f64().unwrap() - 3.0).abs() < 1e-9);

    let ln = harness::expect_success(harness::invoke(
        "calculator",
        json!({"expression": "ln(exp(2))"}),
    ));
    assert!((ln["result"].as_f64().unwrap() - 2.0).abs() < 1e-9);
}

#[test]
fn calculator_preserves_int_vs_float() {
    // Integer expressions return a JSON integer (no spurious `.0`); fractional
    // expressions return a JSON float.
    let int = harness::expect_success(harness::invoke(
        "calculator",
        json!({"expression": "2 + 2"}),
    ));
    assert!(
        int["result"].is_i64(),
        "2 + 2 should be an integer, got {}",
        int["result"]
    );
    assert_eq!(int["result"].as_i64().unwrap(), 4);

    let float = harness::expect_success(harness::invoke(
        "calculator",
        json!({"expression": "9.0 / 2.0"}),
    ));
    assert!(
        float["result"].is_f64(),
        "9.0 / 2.0 should be a float, got {}",
        float["result"]
    );
    assert!((float["result"].as_f64().unwrap() - 4.5).abs() < 1e-9);
}

#[test]
fn calculator_smart_division() {
    // Division picks its type per-operation: exact → integer, remainder → float.
    let exact = harness::expect_success(harness::invoke(
        "calculator",
        json!({"expression": "6 / 2"}),
    ));
    assert!(
        exact["result"].is_i64(),
        "6 / 2 should be an integer, got {}",
        exact["result"]
    );
    assert_eq!(exact["result"].as_i64().unwrap(), 3);

    let fractional = harness::expect_success(harness::invoke(
        "calculator",
        json!({"expression": "7 / 2"}),
    ));
    assert!(
        fractional["result"].is_f64(),
        "7 / 2 should be a float, got {}",
        fractional["result"]
    );
    assert!((fractional["result"].as_f64().unwrap() - 3.5).abs() < 1e-9);

    // Smart division propagates through sub-expressions: (7 / 2) * 2 == 7.0,
    // not 6 (which integer division would give).
    let nested = harness::expect_success(harness::invoke(
        "calculator",
        json!({"expression": "(7 / 2) * 2"}),
    ));
    assert!((nested["result"].as_f64().unwrap() - 7.0).abs() < 1e-9);
}

#[test]
fn calculator_big_integer_precision() {
    // Integer-only expressions keep full i64 precision (no f64 2^53 cliff).
    let resp = harness::expect_success(harness::invoke(
        "calculator",
        json!({"expression": "123456789 * 987654321"}),
    ));
    assert!(resp["result"].is_i64());
    assert_eq!(
        resp["result"].as_i64().unwrap(),
        121_932_631_112_635_269_i64
    );
}

#[test]
fn calculator_basic_arithmetic() {
    let resp = harness::expect_success(harness::invoke(
        "calculator",
        json!({"expression": "2 + 3 * 4"}),
    ));
    assert_eq!(resp["result"].as_f64().unwrap(), 14.0);
}

#[test]
fn calculator_power() {
    let resp = harness::expect_success(harness::invoke(
        "calculator",
        json!({"expression": "2 ^ 10"}),
    ));
    assert!((resp["result"].as_f64().unwrap() - 1024.0).abs() < 1e-9);
}

#[test]
fn calculator_empty_expression() {
    let resp = harness::invoke("calculator", json!({"expression": ""}));
    harness::expect_error(&resp, "invalid_arguments");
}

#[test]
fn calculator_too_long() {
    let long = "1+".repeat(600);
    let resp = harness::invoke("calculator", json!({"expression": long}));
    harness::expect_error(&resp, "invalid_arguments");
}

#[test]
fn calculator_invalid_expression() {
    let resp = harness::invoke("calculator", json!({"expression": "foo(bar)"}));
    // Should be parse_error
    assert!(resp.get("error").is_some());
}

#[test]
fn calculator_small_result_has_no_display() {
    // 2 + 2 = 4 — unambiguous as-is; the runner attaches no display rendering.
    let resp = harness::expect_success(harness::invoke(
        "calculator",
        json!({"expression": "2 + 2"}),
    ));
    assert!(resp.get("result_display").is_none());
}

#[test]
fn calculator_large_float_display_carries_grouping_and_magnitude() {
    // The live-run misread case: sqrt(237849273487234283743) ≈ 1.5422e10 was
    // reported as ×10¹⁹ by the model off the bare digit run. The runner's
    // `result_display` annotation makes the magnitude explicit.
    let resp = harness::expect_success(harness::invoke(
        "calculator",
        json!({"expression": "sqrt(237849273487234283743)"}),
    ));
    assert_eq!(
        resp["result_display"].as_str().unwrap(),
        "15,422,362,772.520761 (≈1.5422e10)"
    );
}

#[test]
fn calculator_large_int_display_is_grouped() {
    // 123456 * 1000 = 123,456,000 — integer path, grouped, no magnitude tag.
    let resp = harness::expect_success(harness::invoke(
        "calculator",
        json!({"expression": "123456 * 1000"}),
    ));
    assert_eq!(resp["result"].as_i64().unwrap(), 123_456_000);
    assert_eq!(resp["result_display"].as_str().unwrap(), "123,456,000");
}
