//! `zenj_request`: ask for a track, with an optional note and urgency. It is
//! placed where it fits, usually within two tracks; `now` mixes it in at the
//! next phrase.

use serde_json::{json, Value};

use super::{text, Answer, Dj, Tool};
use crate::brain::run::Ask;

pub struct RequestTool;

impl Tool for RequestTool {
    const NAME: &'static str = "zenj_request";

    fn definition() -> Value {
        json!({
            "name": Self::NAME,
            "description": "Request a track at the party. ZenJ places it where it fits the mix, usually within two tracks, and says where; `now` mixes it in at the next phrase instead. A track not in tonight's library is answered with the closest that is.",
            "inputSchema": {"type": "object", "properties": {
                "track": {"type": "string", "description": "Its artist and title, or its ISRC."},
                "note": {"type": "string", "description": "Why, or who it is for: Zen reads it when choosing."},
                "urgency": {"type": "number", "minimum": 0, "maximum": 1, "description": "0 whenever it fits, 1 as soon as it can."},
                "now": {"type": "boolean", "description": "Play it next, at the coming phrase."},
                "by": {"type": "string", "description": "Who asked, when it is not the caller."}
            }, "required": ["track"]},
        })
    }
}

pub async fn call(dj: &Dj, caller: &str, args: &Value) -> Answer {
    let query = match text(args, "track") {
        Ok(t) => t.to_string(),
        Err(a) => return a,
    };
    let note = args.get("note").and_then(Value::as_str).unwrap_or("").trim().to_string();
    let urgency = args.get("urgency").and_then(Value::as_f64).unwrap_or(0.5) as f32;
    let now = args.get("now").and_then(Value::as_bool).unwrap_or(false);
    let by = args
        .get("by")
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .unwrap_or(caller)
        .to_string();
    dj.ask(|reply| Ask::Request { query, by, note, urgency, now, reply }).await
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tools::fixture::dj;

    /// **A request reaches the brain as asked**, from the caller unless it
    /// names someone else, and the brain's line is the answer.
    #[tokio::test]
    async fn a_request_reaches_the_brain_as_asked() {
        let (dj, mut rx) = dj();
        let brain = tokio::spawn(async move {
            match rx.recv().await.unwrap() {
                Ask::Request { query, by, note, urgency, now, reply } => {
                    assert_eq!((query.as_str(), by.as_str(), note.as_str(), urgency, now), ("ABBA dancing queen", "yen", "for mum", 1.0, false));
                    reply.send("placed".into()).unwrap();
                }
                _ => panic!("not a request"),
            }
        });
        let a = call(&dj, "john", &json!({"track": " ABBA dancing queen ", "note": "for mum", "urgency": 1.0, "by": "yen"})).await;
        assert_eq!(a, Answer::ok("placed"));
        brain.await.unwrap();
        assert_eq!(call(&dj, "john", &json!({})).await, Answer::error("`track` is required"));
    }
}
