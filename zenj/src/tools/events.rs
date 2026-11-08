//! `zenj_events`: what happened since a cursor — tracks on air, choices and
//! their reasons, requests, transitions, mode changes, faults — with the
//! cursor to ask from next.

use std::fmt::Write;

use serde_json::{json, Value};

use super::{Answer, Dj, Tool};
use crate::events::Event;

pub struct EventsTool;

/// The most one call returns.
const LIMIT: usize = 100;

impl Tool for EventsTool {
    const NAME: &'static str = "zenj_events";

    fn definition() -> Value {
        json!({
            "name": Self::NAME,
            "description": "What happened at the party since `since` (0 for the start, else the cursor the last call gave): tracks on air, ZenJ's choices and why, requests and where they went, transitions, mode changes and faults, oldest first, with the cursor to ask from next.",
            "inputSchema": {"type": "object", "properties": {
                "since": {"type": "integer", "minimum": 0},
                "limit": {"type": "integer", "minimum": 1, "maximum": 100}
            }},
            "annotations": {"readOnlyHint": true},
        })
    }
}

pub fn say(events: &[Event], last: u64) -> String {
    let mut s = String::new();
    for e in events {
        let kind = serde_json::to_value(e.kind)
            .ok()
            .and_then(|v| v.as_str().map(str::to_string))
            .unwrap_or_default();
        let _ = writeln!(
            s,
            "#{} {} {kind}: {}",
            e.id,
            e.at.format("%H:%M:%S"),
            e.text
        );
    }
    let cursor = events.last().map_or(last, |e| e.id);
    let _ = write!(s, "cursor: {cursor}");
    s
}

pub fn call(dj: &Dj, args: &Value) -> Answer {
    let since = args.get("since").and_then(Value::as_u64).unwrap_or(0);
    let limit = args
        .get("limit")
        .and_then(Value::as_u64)
        .map_or(LIMIT, |l| (l as usize).min(LIMIT));
    let got = dj.events.since(since, limit);
    if got.is_empty() {
        return Answer::ok(format!(
            "nothing since {since}; cursor: {}",
            dj.events.last().max(since)
        ));
    }
    Answer::ok(say(&got, dj.events.last()))
}

#[cfg(test)]
mod tests {
    use chrono::{Local, TimeZone};

    use super::*;
    use crate::events::Kind;
    use crate::tools::fixture::dj;

    #[test]
    fn events_are_said_with_the_cursor_to_ask_from() {
        let (dj, _) = dj();
        let at = Local.with_ymd_and_hms(2026, 12, 31, 22, 41, 5).unwrap();
        dj.events
            .push_at(at, Kind::Track, "on air, deck 1: ABBA — Waterloo");
        dj.events
            .push_at(at, Kind::Request, "yen's request: placed");
        assert_eq!(
            call(&dj, &json!({"since": 0})).text,
            "#1 22:41:05 track: on air, deck 1: ABBA — Waterloo\n#2 22:41:05 request: yen's request: placed\ncursor: 2"
        );
        assert_eq!(
            call(&dj, &json!({"since": 2})).text,
            "nothing since 2; cursor: 2"
        );
        assert_eq!(
            call(&dj, &json!({"since": 0, "limit": 1}))
                .text
                .lines()
                .last(),
            Some("cursor: 1")
        );
    }
}
