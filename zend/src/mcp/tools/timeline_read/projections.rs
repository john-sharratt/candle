//! `view: projections`: one turn's projection points, from the record each
//! turn keeps of them (`GET /v1/conversations/{id}/projections/{turn}/{event}`):
//! the opening point, each reprojection as it decoded, and the point after it
//! sealed. For each: where in the reply it was taken, the tools whose
//! signatures it showed with their scores against the bar in force there, the
//! strongest it left out, and its tokens by bucket — what answers why a tool's
//! signature was or was not in front of the model when it called.
//!
//! The bar is the `tools` collection's (`zend/src/prompts/projection.yaml`):
//! a score of 800 to be shown, held down to 250 over the first 64 tokens of a
//! reply. A point that selected no tool records no candidates at all, and says
//! so.

use host::substrate_views::TimelineDetail;
use serde_json::Value;

use super::{head, no_turn};
use crate::projection_event::ProjectionEventOut;

/// The tools collection's bar to be shown, and its early-reply bar.
const MIN_SCORE: f64 = 800.0;
const EARLY_MIN_SCORE: f64 = 250.0;
/// The last reply token the early bar holds for.
const EARLY_WINDOW_TOKENS: u64 = 64;

/// How many tools left out a point names, strongest first.
const LEFT_OUT: usize = 5;

/// The bar a point taken at reply token `start` held a tool to: the early
/// one over tokens `1..=64`, the full one at the opening and after.
pub fn bar_at(start: u64) -> f64 {
    if (1..=EARLY_WINDOW_TOKENS).contains(&start) {
        EARLY_MIN_SCORE
    } else {
        MIN_SCORE
    }
}

/// Where turn `index` is in the projection record: its place among the
/// conversation's turns that are no summary and have an assistant half.
pub fn record_of(t: &TimelineDetail, index: u32) -> Option<usize> {
    t.turns
        .iter()
        .filter(|turn| turn.kind == "normal" && !turn.assistant.is_empty())
        .position(|turn| turn.index == index)
}

/// One point of `of`, as its lines. The first is the opening the prompt was
/// built from; the last of several is recomputed after the turn sealed, and
/// governs nothing the reply read.
pub fn point(n: usize, of: usize, p: &Value) -> String {
    let start = p.get("start_token").and_then(Value::as_u64).unwrap_or(0);
    let bar = bar_at(start);
    let mut out = format!(
        "point {n} at reply token {start} ({}): {} tokens materialized; the substrate holds {}",
        match n {
            0 => "opening — the prompt the reply began on",
            n if n + 1 == of => "after the turn sealed",
            _ => "reprojected while decoding",
        },
        p.get("materialized_tokens")
            .and_then(Value::as_u64)
            .unwrap_or(0),
        p.get("substrate_tokens")
            .and_then(Value::as_u64)
            .unwrap_or(0),
    );
    let tools = p
        .pointer("/selection/system")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .find(|i| {
            i.get("kind").and_then(Value::as_str) == Some("collection")
                && i.get("name").and_then(Value::as_str) == Some("tools")
        });
    match tools
        .and_then(|t| t.get("sections"))
        .and_then(Value::as_array)
    {
        Some(sections) => {
            let mut members: Vec<(&str, f64, bool, bool)> = sections
                .iter()
                .map(|s| {
                    (
                        s.get("name").and_then(Value::as_str).unwrap_or("?"),
                        s.get("score").and_then(Value::as_f64).unwrap_or(0.0),
                        s.get("selected").and_then(Value::as_bool).unwrap_or(false),
                        s.get("qualified").and_then(Value::as_bool).unwrap_or(false),
                    )
                })
                .collect();
            members.sort_by(|a, b| b.1.total_cmp(&a.1));
            // A member shown with no score is a mandatory one: it is in every
            // prompt, whatever the belief.
            let shown: Vec<String> = members
                .iter()
                .filter(|m| m.2 && m.1 > 0.0)
                .map(|(name, score, _, _)| format!("{name} {score:.0}"))
                .collect();
            let always: Vec<&str> = members
                .iter()
                .filter(|m| m.2 && m.1 <= 0.0)
                .map(|m| m.0)
                .collect();
            out.push_str(&format!(
                "\n  signatures shown by score (bar {bar:.0}): {}",
                if shown.is_empty() {
                    "none — no tool scored".to_string()
                } else {
                    shown.join(", ")
                }
            ));
            if !always.is_empty() {
                out.push_str(&format!("\n  always shown: {}", always.join(", ")));
            }
            let left: Vec<String> = members
                .iter()
                .filter(|m| !m.2 && m.1 > 0.0)
                .take(LEFT_OUT)
                .map(|(name, score, _, qualified)| {
                    format!(
                        "{name} {score:.0}{}",
                        if *qualified { " (cleared the bar)" } else { "" }
                    )
                })
                .collect();
            out.push_str(&format!(
                "\n  strongest left out: {}",
                if left.is_empty() {
                    "none scored".to_string()
                } else {
                    left.join(", ")
                }
            ));
        }
        None => out.push_str(&format!(
            "\n  signatures shown by score (bar {bar:.0}): none — no tool reached the bar, so \
             the prompt carried the tool names alone and the point recorded no candidates"
        )),
    }
    let buckets: Vec<String> = p
        .get("buckets")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .map(|b| {
            format!(
                "{} {}",
                b.get("label").and_then(Value::as_str).unwrap_or("?"),
                b.get("tokens").and_then(Value::as_u64).unwrap_or(0)
            )
        })
        .collect();
    if !buckets.is_empty() {
        out.push_str(&format!("\n  tokens by bucket: {}", buckets.join(", ")));
    }
    let turns = p
        .pointer("/selection/turns")
        .and_then(Value::as_array)
        .map_or(0, |t| {
            t.iter()
                .filter(|s| s.get("selected").and_then(Value::as_bool).unwrap_or(true))
                .count()
        });
    out.push_str(&format!("\n  dialogue turns in context: {turns}"));
    out
}

/// Turn `index`'s points, from `points` — every turn's, in record order.
pub fn render(
    t: &TimelineDetail,
    index: u32,
    points: &[Vec<ProjectionEventOut>],
) -> Result<String, String> {
    if !t.turns.iter().any(|turn| turn.index == index) {
        return Err(no_turn(t, index));
    }
    let Some(at) = record_of(t, index) else {
        return Err(format!(
            "turn {index} of timeline {} is a summary or has no reply, so it has no projections",
            t.timeline
        ));
    };
    let Some(record) = points.get(at).filter(|r| !r.is_empty()) else {
        return Err(format!(
            "turn {index} of timeline {} kept no projection record — it was decoded before \
             records were kept, or its record was lost",
            t.timeline
        ));
    };
    let mut out = head(t);
    out.push_str(&format!(
        "\nturn #{index}: {} projection points",
        record.len()
    ));
    for (n, p) in record.iter().enumerate() {
        let v = serde_json::to_value(p).unwrap_or(Value::Null);
        out.push('\n');
        out.push_str(&point(n, record.len(), &v));
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::super::tests::{timeline, turn_view};
    use super::*;

    #[test]
    fn the_bar_is_held_down_over_a_replys_first_64_tokens() {
        assert_eq!(bar_at(0), 800.0);
        assert_eq!(bar_at(1), 250.0);
        assert_eq!(bar_at(64), 250.0);
        assert_eq!(bar_at(65), 800.0);
    }

    /// **A turn's record is its place among the turns that are no summary
    /// and have a reply.**
    #[test]
    fn a_turns_record_is_its_place_among_the_replied_turns() {
        let mut t = timeline();
        let mut summary = turn_view(1, "s", "summary");
        summary.kind = "sot";
        t.turns = vec![
            turn_view(0, "a", "one"),
            summary,
            turn_view(2, "b", ""),
            turn_view(3, "c", "three"),
        ];
        assert_eq!(record_of(&t, 0), Some(0));
        assert_eq!(record_of(&t, 1), None);
        assert_eq!(record_of(&t, 2), None);
        assert_eq!(record_of(&t, 3), Some(1));
    }

    /// **A point says what it showed against the bar, and the strongest it
    /// left out**; one with no tools collection says none reached the bar.
    #[test]
    fn a_point_says_what_it_showed_and_what_it_left_out() {
        let p = json!({
            "start_token": 0, "materialized_tokens": 9000, "substrate_tokens": 40000,
            "buckets": [{"label": "tools", "kind": "section", "tokens": 1200}],
            "selection": {"system": [{"kind": "collection", "name": "tools", "sections": [
                {"name": "file_list", "tokens": 300, "selected": true, "score": 990.0, "qualified": true},
                {"name": "job_status", "tokens": 300, "selected": false, "score": 640.0, "qualified": false},
                {"name": "run_command", "tokens": 300, "selected": true, "score": 1000.0, "qualified": true},
                {"name": "datetime", "tokens": 90, "selected": false, "score": 0.0, "qualified": false},
                {"name": "file_read", "tokens": 300, "selected": true, "score": 0.0, "qualified": false}
            ]}], "turns": [{"selected": true}, {"selected": false}]}
        });
        assert_eq!(
            point(0, 3, &p),
            "point 0 at reply token 0 (opening — the prompt the reply began on): 9000 tokens \
             materialized; the substrate holds 40000\n  signatures shown by score (bar 800): \
             run_command 1000, file_list 990\n  always shown: file_read\n  strongest left out: \
             job_status 640\n  tokens by bucket: tools 1200\n  dialogue turns in context: 1"
        );
        let none = json!({"start_token": 40, "selection": {"system": [], "turns": []}});
        assert!(
            point(1, 3, &none).contains("point 1 at reply token 40 (reprojected while decoding)")
        );
        assert!(point(1, 3, &none)
            .contains("signatures shown by score (bar 250): none — no tool reached the bar"));
        assert!(point(2, 3, &none).contains("(after the turn sealed)"));
    }
}
