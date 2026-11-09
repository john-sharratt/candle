//! `zenj_rekordbox`: open or close rekordbox (`docs/house_modes.md` §3). The
//! house's Party mode does this itself; the tool is for doing it alone.
//! While rekordbox is open this machine is reserved for the decks.

use serde_json::{json, Value};

use super::{text, Answer, Dj, Tool};
use crate::rekordbox::control::Act;

pub struct RekordboxTool;

impl Tool for RekordboxTool {
    const NAME: &'static str = "zenj_rekordbox";

    fn definition() -> Value {
        json!({
            "name": Self::NAME,
            "description": "Open or close rekordbox on BlackWorld, the machine the decks are cabled to. While it is open the machine is reserved for the decks: nothing is deployed or built there, it reads nothing in, and its model member steps aside while another serves Zen, until ten minutes after it closes. Closing is refused while a deck is playing unless `force` is set, which only a person at the party asks for.",
            "inputSchema": {"type": "object", "properties": {
                "act": {"type": "string", "enum": ["open", "close"]},
                "force": {"type": "boolean", "description": "Close it even with a deck playing: a person asked for exactly that."}
            }, "required": ["act"]},
        })
    }
}

pub async fn call(dj: &Dj, args: &Value) -> Answer {
    let force = args.get("force").and_then(Value::as_bool).unwrap_or(false);
    let act = match text(args, "act").map(|a| Act::parse(a, force)) {
        Ok(Some(a)) => a,
        Ok(None) => return Answer::error("`act` is open or close"),
        Err(a) => return a,
    };
    match dj.rekordbox.act(act).await {
        Ok(line) => Answer::ok(line),
        Err(why) => Answer::error(why),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tools::fixture::dj;

    #[tokio::test]
    async fn an_act_is_open_or_close() {
        let (dj, _) = dj();
        assert_eq!(
            call(&dj, &json!({"act": "dance"})).await,
            Answer::error("`act` is open or close")
        );
        assert!(call(&dj, &json!({})).await.is_error);
        assert_eq!(
            call(&dj, &json!({"act": "close"})).await,
            Answer::ok("rekordbox is already closed")
        );
    }
}
