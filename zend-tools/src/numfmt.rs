//! Human-readable numeric renderings for tool responses.
//!
//! A bare long number in a tool response (`15422362772.520761`) forces the
//! model to count digits to grasp its magnitude — and miscounting there is
//! exactly how a correct tool result gets reported wrong (observed live:
//! sqrt → ~1.54e10 reported as ×10¹⁹). Tools that return numbers attach a
//! `display` rendering — digit-grouped, and magnitude-annotated for large or
//! tiny floats — so the model quotes instead of re-deriving.

/// Insert `,` thousands separators into a plain decimal rendering (no
/// e-notation). Handles a leading `-` and an optional fractional part; only
/// the integer digits are grouped.
pub(crate) fn group_digits(raw: &str) -> String {
    let (sign, rest) = raw.strip_prefix('-').map_or(("", raw), |r| ("-", r));
    let (int_part, frac_part) = match rest.split_once('.') {
        Some((i, f)) => (i, Some(f)),
        None => (rest, None),
    };
    let mut grouped = String::with_capacity(raw.len() + int_part.len() / 3);
    for (i, c) in int_part.chars().enumerate() {
        if i > 0 && (int_part.len() - i) % 3 == 0 {
            grouped.push(',');
        }
        grouped.push(c);
    }
    match frac_part {
        Some(f) => format!("{sign}{grouped}.{f}"),
        None => format!("{sign}{grouped}"),
    }
}

/// The `display` rendering for an integer result, or `None` when the bare
/// number already reads unambiguously (fewer than 5 digits).
pub(crate) fn display_i64(i: i64) -> Option<String> {
    if i.unsigned_abs() >= 10_000 {
        Some(group_digits(&i.to_string()))
    } else {
        None
    }
}

/// [`display_i64`] for values above `i64::MAX` (JSON numbers are `u64` there).
pub(crate) fn display_u64(u: u64) -> Option<String> {
    if u >= 10_000 {
        Some(group_digits(&u.to_string()))
    } else {
        None
    }
}

/// Recursively annotate a tool-response JSON value: every object field whose
/// number is large or tiny enough to misread gains a sibling
/// `<key>_display` rendering. Running once at the dispatch chokepoint covers
/// every tool — a 10-digit `unix` timestamp, a `random` draw, a file `size` —
/// without each tool remembering to opt in. Fields already named `*_display`
/// are left alone, and an existing sibling is never overwritten.
pub(crate) fn annotate_json(v: &mut serde_json::Value) {
    match v {
        serde_json::Value::Object(map) => {
            let mut additions: Vec<(String, String)> = Vec::new();
            for (k, val) in map.iter_mut() {
                match val {
                    serde_json::Value::Number(n) => {
                        if k.ends_with("_display") {
                            continue;
                        }
                        let display = if let Some(i) = n.as_i64() {
                            display_i64(i)
                        } else if let Some(u) = n.as_u64() {
                            display_u64(u)
                        } else {
                            n.as_f64().and_then(display_f64)
                        };
                        if let Some(d) = display {
                            additions.push((format!("{k}_display"), d));
                        }
                    }
                    _ => annotate_json(val),
                }
            }
            for (k, d) in additions {
                map.entry(k).or_insert_with(|| serde_json::Value::String(d));
            }
        }
        serde_json::Value::Array(items) => {
            for item in items {
                annotate_json(item);
            }
        }
        _ => {}
    }
}

/// The `display` rendering for a float result: digit-grouped with a magnitude
/// annotation for large values, magnitude-only for tiny ones, `None` when the
/// bare number already reads unambiguously.
pub(crate) fn display_f64(f: f64) -> Option<String> {
    let a = f.abs();
    let tiny = a > 0.0 && a < 1e-4;
    let big = a >= 10_000.0;
    if !tiny && !big {
        return None;
    }
    // `{}` on f64 never uses e-notation, so the grouped form is a plain
    // decimal; cap the grouped rendering at 21 integer digits (beyond f64's
    // exact range there is no value in a wall of digits) and fall back to the
    // magnitude annotation alone.
    let raw = format!("{f}");
    let int_digits = raw
        .trim_start_matches('-')
        .split('.')
        .next()
        .map_or(0, str::len);
    let sci = format!("≈{f:.4e}");
    if tiny || int_digits > 21 {
        Some(sci)
    } else {
        Some(format!("{} ({sci})", group_digits(&raw)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn grouping_plain_integers() {
        assert_eq!(group_digits("4"), "4");
        assert_eq!(group_digits("1234"), "1,234");
        assert_eq!(group_digits("15422362772"), "15,422,362,772");
        assert_eq!(group_digits("-1234567"), "-1,234,567");
    }

    #[test]
    fn grouping_preserves_fraction_ungrouped() {
        assert_eq!(group_digits("15422362772.520761"), "15,422,362,772.520761");
        assert_eq!(group_digits("-0.25"), "-0.25");
    }

    #[test]
    fn small_values_have_no_display() {
        assert_eq!(display_i64(4), None);
        assert_eq!(display_i64(-9999), None);
        assert_eq!(display_f64(3.5), None);
        assert_eq!(display_f64(0.001), None);
        assert_eq!(display_f64(0.0), None);
    }

    #[test]
    fn large_int_display_is_grouped() {
        assert_eq!(display_i64(10_000).as_deref(), Some("10,000"));
        assert_eq!(
            display_i64(-1_234_567_890).as_deref(),
            Some("-1,234,567,890")
        );
    }

    #[test]
    fn the_live_sqrt_case_is_unambiguous() {
        // The exact result the model misread as ×10¹⁹ in a live run: the
        // display carries both the grouped digits and the explicit magnitude.
        assert_eq!(
            display_f64(15422362772.520761).as_deref(),
            Some("15,422,362,772.520761 (≈1.5422e10)")
        );
    }

    #[test]
    fn tiny_floats_get_magnitude_only() {
        assert_eq!(display_f64(0.000012).as_deref(), Some("≈1.2000e-5"));
    }

    #[test]
    fn huge_floats_skip_the_digit_wall() {
        assert_eq!(display_f64(1e300).as_deref(), Some("≈1.0000e300"));
    }

    #[test]
    fn annotate_adds_display_siblings_recursively_and_skips_small_values() {
        let mut v = serde_json::json!({
            "unix": 1783825140i64,
            "weekday": "Sunday",
            "count": 3,
            "nested": { "result": 15422362772.520761f64 },
            "items": [ { "size": 123456u64 }, { "size": 12u64 } ],
        });
        annotate_json(&mut v);
        assert_eq!(v["unix_display"], "1,783,825,140");
        assert!(v.get("weekday_display").is_none());
        assert!(v.get("count_display").is_none());
        assert_eq!(
            v["nested"]["result_display"],
            "15,422,362,772.520761 (≈1.5422e10)"
        );
        assert_eq!(v["items"][0]["size_display"], "123,456");
        assert!(v["items"][1].get("size_display").is_none());
    }

    #[test]
    fn annotate_never_overwrites_or_chains_display_fields() {
        let mut v = serde_json::json!({
            "result": 123456,
            "result_display": "already here",
        });
        annotate_json(&mut v);
        // The existing sibling wins, and the numeric-looking suffix field
        // itself gains no `_display_display`.
        assert_eq!(v["result_display"], "already here");
        assert!(v.get("result_display_display").is_none());
        assert_eq!(v.as_object().unwrap().len(), 2);
    }
}
