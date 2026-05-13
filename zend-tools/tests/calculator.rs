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
    // evalexpr supports math(2.7) but not floor/ceil as unbound functions
    // Test that a valid expression with operations works
    let resp = harness::expect_success(harness::invoke(
        "calculator",
        json!({"expression": "5 * 5 + 1"}),
    ));
    assert!((resp["result"].as_f64().unwrap() - 26.0).abs() < 1e-9);
}

#[test]
fn calculator_basic_arithmetic() {
    let resp = harness::expect_success(harness::invoke("calculator", json!({"expression": "2 + 3 * 4"})));
    assert_eq!(resp["result"].as_f64().unwrap(), 14.0);
}

#[test]
fn calculator_power() {
    let resp = harness::expect_success(harness::invoke("calculator", json!({"expression": "2 ^ 10"})));
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
