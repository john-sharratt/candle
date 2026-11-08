//! `zenj_queue`: the next track and the requests waiting; with `pin`, make a
//! track the next, past the shortlist.

use serde_json::{json, Value};

use super::{now, Answer, Dj, Tool};
use crate::brain::run::Ask;

pub struct Queue;

impl Tool for Queue {
    const NAME: &'static str = "zenj_queue";

    fn definition() -> Value {
        json!({
            "name": Self::NAME,
            "description": "The next track and the requests waiting. With `pin` — a track's artist and title, or its ISRC — that track becomes the next, ahead of ZenJ's own choice.",
            "inputSchema": {"type": "object", "properties": {
                "pin": {"type": "string", "description": "A track to play next: words of its artist and title, or its ISRC."}
            }},
        })
    }
}

pub async fn call(dj: &Dj, args: &Value) -> Answer {
    match args
        .get("pin")
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|s| !s.is_empty())
    {
        Some(q) => {
            let query = q.to_string();
            dj.ask(|reply| Ask::Pin { query, reply }).await
        }
        None => Answer::ok(now::say(&dj.live.load())),
    }
}
