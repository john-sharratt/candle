//! `GET /v1/events` — the engine's high-rate events (`candle::engine_events`):
//! each kind's count since start, a recent instance and how long ago it was
//! taken, most frequent first. What those events were logged for, without
//! writing a line each.

use axum::Json;
use candle::engine_events::{snapshot, EventCount};
use serde::Serialize;

#[derive(Serialize, Debug, PartialEq)]
pub struct Event {
    kind: &'static str,
    count: u64,
    last: String,
    age_s: f64,
}

#[derive(Serialize, Debug, PartialEq)]
pub struct Events {
    events: Vec<Event>,
}

fn of(e: EventCount) -> Event {
    Event {
        kind: e.kind,
        count: e.count,
        last: e.last,
        age_s: (e.age_s * 100.0).round() / 100.0,
    }
}

pub async fn events() -> Json<Events> {
    Json(Events {
        events: snapshot().into_iter().map(of).collect(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A kind reaches the page by name, its age to the hundredth of a second.
    #[test]
    fn a_kind_reaches_the_page_by_name() {
        let e = of(EventCount {
            kind: "reproject",
            count: 7,
            last: "from_view=2 to_view=3".into(),
            age_s: 1.23456,
        });
        assert_eq!(
            serde_json::to_value(e).unwrap(),
            serde_json::json!({"kind": "reproject", "count": 7,
                               "last": "from_view=2 to_view=3", "age_s": 1.23})
        );
    }
}
