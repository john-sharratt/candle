//! `zenj_said`: a line someone in the room said — "more energy", "something
//! for Yen's mum" — kept for the next choice, which Zen makes hearing it.

use serde_json::{json, Value};

use super::{text, Answer, Dj, Tool};
use crate::brain::run::Ask;

pub struct Said;

impl Tool for Said {
    const NAME: &'static str = "zenj_said";

    fn definition() -> Value {
        json!({
            "name": Self::NAME,
            "description": "Pass on what someone at the party said about the music — a mood, a wish, a name — so ZenJ's next choice hears it. Use zenj_request for a specific track and zenj_energy for more or less energy.",
            "inputSchema": {"type": "object", "properties": {
                "text": {"type": "string"},
                "who": {"type": "string", "description": "Who said it, when it is not the caller."}
            }, "required": ["text"]},
        })
    }
}

pub async fn call(dj: &Dj, caller: &str, args: &Value) -> Answer {
    let line = match text(args, "text") {
        Ok(t) => t.to_string(),
        Err(a) => return a,
    };
    let who = args
        .get("who")
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .unwrap_or(caller)
        .to_string();
    if dj
        .asks
        .send(Ask::Said {
            who: who.clone(),
            text: line,
        })
        .await
        .is_err()
    {
        return Answer::error("ZenJ's brain is not running");
    }
    Answer::ok(format!("heard; the next choice hears {who} too"))
}
