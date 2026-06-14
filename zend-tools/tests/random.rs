mod harness;

use serde_json::json;

#[test]
fn random_integer_in_range() {
    let resp = harness::expect_success(harness::invoke(
        "random",
        json!({"kind": "integer", "min": 1.0, "max": 6.0}),
    ));
    let val = resp["result"].as_i64().unwrap();
    assert!(val >= 1 && val <= 6, "got {val}");
}

#[test]
fn random_float_in_range() {
    let resp = harness::expect_success(harness::invoke(
        "random",
        json!({"kind": "float", "min": 0.0, "max": 1.0}),
    ));
    let f = resp["result"].as_f64().unwrap();
    assert!(f >= 0.0 && f < 1.0, "got {f}");
}

#[test]
fn random_choice_from_list() {
    let resp = harness::expect_success(harness::invoke(
        "random",
        json!({"kind": "choice", "choices": ["x", "y", "z"]}),
    ));
    let s = resp["result"].as_str().unwrap();
    assert!(["x", "y", "z"].contains(&s));
}

#[test]
fn random_shuffle_length() {
    let resp = harness::expect_success(harness::invoke(
        "random",
        json!({"kind": "shuffle", "choices": ["a", "b", "c", "d"]}),
    ));
    let arr = resp["result"].as_array().unwrap();
    assert_eq!(arr.len(), 4);
}

#[test]
fn random_dice_2d6() {
    let resp = harness::expect_success(harness::invoke(
        "random",
        json!({"kind": "dice", "sides": 6, "count": 2}),
    ));
    let arr = resp["result"].as_array().unwrap();
    assert_eq!(arr.len(), 2);
    for v in arr {
        let n = v.as_u64().unwrap();
        assert!(n >= 1 && n <= 6, "got {n}");
    }
}

#[test]
fn random_integer_default() {
    let resp = harness::expect_success(harness::invoke("random", json!({"kind": "integer"})));
    let val = resp["result"].as_i64().unwrap();
    assert!(val >= 0 && val <= 100);
}

#[test]
fn random_integer_range() {
    let resp = harness::expect_success(harness::invoke(
        "random",
        json!({
            "kind": "integer", "min": 10.0, "max": 20.0, "count": 5
        }),
    ));
    let arr = resp["result"].as_array().unwrap();
    assert_eq!(arr.len(), 5);
    for v in arr {
        let n = v.as_i64().unwrap();
        assert!(n >= 10 && n <= 20);
    }
}

#[test]
fn random_float() {
    let resp = harness::expect_success(harness::invoke(
        "random",
        json!({
            "kind": "float", "min": 0.0, "max": 1.0
        }),
    ));
    let f = resp["result"].as_f64().unwrap();
    assert!(f >= 0.0 && f < 1.0);
}

#[test]
fn random_choice() {
    let resp = harness::expect_success(harness::invoke(
        "random",
        json!({
            "kind": "choice", "choices": ["a", "b", "c"]
        }),
    ));
    let s = resp["result"].as_str().unwrap();
    assert!(["a", "b", "c"].contains(&s));
}

#[test]
fn random_shuffle() {
    let resp = harness::expect_success(harness::invoke(
        "random",
        json!({
            "kind": "shuffle", "choices": ["x", "y", "z"]
        }),
    ));
    let arr = resp["result"].as_array().unwrap();
    assert_eq!(arr.len(), 3);
}

#[test]
fn random_dice() {
    let resp = harness::expect_success(harness::invoke(
        "random",
        json!({
            "kind": "dice", "sides": 6
        }),
    ));
    let n = resp["result"].as_u64().unwrap();
    assert!(n >= 1 && n <= 6);
}

#[test]
fn random_invalid_kind() {
    let resp = harness::invoke("random", json!({"kind": "bogus"}));
    harness::expect_error(&resp, "invalid_kind");
}

#[test]
fn random_count_too_large() {
    let resp = harness::invoke(
        "random",
        json!({
            "kind": "integer", "count": 1001
        }),
    );
    harness::expect_error(&resp, "invalid_arguments");
}
