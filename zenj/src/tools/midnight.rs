//! `zenj_midnight`: the track whose drop lands on 00:00:00 (§5.1). ZenJ
//! schedules the run-in back from it; from 23:40 only a person changes it.

use serde_json::{json, Value};

use super::{text, Answer, Dj, Tool};
use crate::brain::run::Ask;

pub struct Midnight;

impl Tool for Midnight {
    const NAME: &'static str = "zenj_midnight";

    fn definition() -> Value {
        json!({
            "name": Self::NAME,
            "description": "Choose the midnight track: ZenJ times the mix into it so its drop — its first chorus — lands on 00:00:00. Set it only with John's yes.",
            "inputSchema": {"type": "object", "properties": {
                "track": {"type": "string", "description": "Artist and title, or ISRC."}
            }, "required": ["track"]},
        })
    }
}

pub async fn call(dj: &Dj, args: &Value) -> Answer {
    let query = match text(args, "track") {
        Ok(t) => t.to_string(),
        Err(a) => return a,
    };
    dj.ask(|reply| Ask::Midnight { query, reply }).await
}
