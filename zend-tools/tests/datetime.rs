mod harness;

use serde_json::json;

#[test]
fn datetime_utc_fields() {
    let resp = harness::expect_success(harness::invoke("datetime", json!({})));
    assert!(resp["iso8601"].as_str().is_some());
    assert!(resp["unix"].as_i64().is_some());
    assert!(resp["weekday"].as_str().is_some());
    assert!(resp["timezone"].as_str().is_some());
}

#[test]
fn datetime_tokyo() {
    let resp = harness::expect_success(harness::invoke(
        "datetime",
        json!({"timezone": "Asia/Tokyo"}),
    ));
    let iso = resp["iso8601"].as_str().unwrap();
    assert!(iso.contains("+09:00"), "expected +09:00 offset in {iso}");
}

#[test]
fn datetime_unix_reasonable() {
    let resp = harness::expect_success(harness::invoke("datetime", json!({})));
    let unix = resp["unix"].as_i64().unwrap();
    assert!(
        unix > 1_700_000_000,
        "unix timestamp {unix} is before Nov 2023"
    );
}

#[test]
fn datetime_weekday_present() {
    let resp = harness::expect_success(harness::invoke("datetime", json!({})));
    let weekday = resp["weekday"].as_str().unwrap();
    assert!(!weekday.is_empty());
    let valid_days = [
        "Monday",
        "Tuesday",
        "Wednesday",
        "Thursday",
        "Friday",
        "Saturday",
        "Sunday",
    ];
    assert!(valid_days.contains(&weekday), "unknown weekday: {weekday}");
}

#[test]
fn datetime_utc_default() {
    let resp = harness::expect_success(harness::invoke("datetime", json!({})));
    assert_eq!(resp["timezone"], "UTC");
    assert!(resp["unix"].as_i64().unwrap() > 0);
    assert!(resp["iso8601"].as_str().unwrap().contains('T'));
    let weekday = resp["weekday"].as_str().unwrap();
    let valid_days = [
        "Monday",
        "Tuesday",
        "Wednesday",
        "Thursday",
        "Friday",
        "Saturday",
        "Sunday",
    ];
    assert!(
        valid_days.contains(&weekday),
        "unexpected weekday: {weekday}"
    );
}

#[test]
fn datetime_named_timezone() {
    let resp = harness::expect_success(harness::invoke(
        "datetime",
        json!({"timezone": "America/New_York"}),
    ));
    assert_eq!(resp["timezone"], "America/New_York");
}

#[test]
fn datetime_invalid_timezone() {
    let resp = harness::invoke("datetime", json!({"timezone": "Not/ATimezone"}));
    harness::expect_error(&resp, "invalid_timezone");
}

#[test]
fn datetime_schema_registered() {
    let schema = harness::schema("datetime");
    assert!(schema["properties"].is_object());
}
