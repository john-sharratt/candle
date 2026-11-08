//! `zenj_mode`: who has the decks. Back to a person is immediate; toward
//! ZenJ needs a person's yes in the chat first (§6.3), which the caller
//! states with `confirmed`.

use serde_json::{json, Value};

use super::{text, Answer, Dj, Tool};
use crate::brain::run::Ask;
use crate::hands::Mode;

pub struct ModeTool;

impl Tool for ModeTool {
    const NAME: &'static str = "zenj_mode";

    fn definition() -> Value {
        json!({
            "name": Self::NAME,
            "description": "Hand the decks between a person and ZenJ: `human` (a person mixes; ZenJ only keeps the next track ready), `assist` (ZenJ also loads and cues the next on the idle deck), `zenj` (ZenJ mixes). Moving to assist or zenj needs a person at the party to have said yes in this chat: set `confirmed` only then.",
            "inputSchema": {"type": "object", "properties": {
                "mode": {"type": "string", "enum": ["human", "assist", "zenj"]},
                "confirmed": {"type": "boolean", "description": "A person at the party said yes to this, in this chat."}
            }, "required": ["mode"]},
        })
    }
}

pub async fn call(dj: &Dj, args: &Value) -> Answer {
    let mode = match text(args, "mode").map(Mode::parse) {
        Ok(Some(m)) => m,
        Ok(None) => return Answer::error("`mode` is human, assist or zenj"),
        Err(a) => return a,
    };
    let confirmed = args.get("confirmed").and_then(Value::as_bool).unwrap_or(false);
    if mode != Mode::Human && !confirmed {
        return Answer::error(format!(
            "moving to {} needs a person at the party to say yes first: ask them, then call again with confirmed",
            mode.name()
        ));
    }
    dj.ask(|reply| Ask::Mode { mode, reply }).await
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tools::fixture::dj;

    #[tokio::test]
    async fn more_zenj_needs_a_yes_and_back_to_a_person_does_not() {
        let (dj, mut rx) = dj();
        let refused = call(&dj, &json!({"mode": "zenj"})).await;
        assert!(refused.is_error && refused.text.starts_with("moving to zenj needs"));
        assert!(call(&dj, &json!({"mode": "loud"})).await.is_error);
        let brain = tokio::spawn(async move {
            for want in [Mode::Human, Mode::Zenj] {
                match rx.recv().await.unwrap() {
                    Ask::Mode { mode, reply } => {
                        assert_eq!(mode, want);
                        reply.send(mode.name().into()).unwrap();
                    }
                    _ => panic!(),
                }
            }
        });
        assert_eq!(call(&dj, &json!({"mode": "human"})).await, Answer::ok("human"));
        assert_eq!(call(&dj, &json!({"mode": "zenj", "confirmed": true})).await, Answer::ok("zenj"));
        brain.await.unwrap();
    }
}
