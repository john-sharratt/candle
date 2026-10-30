//! `journal_accept {journal}`: accept a Zen worker's proposal
//! (`docs/memory_backbone.md` §7.2) — the note stays, no longer proposed.

use journals::ops::Op;
use serde_json::{json, Value};

use super::{mem, string, Answer, Ctx};

pub const NAME: &str = "journal_accept";

pub fn definition() -> Value {
    json!({
        "name": NAME,
        "description": "Accept a note a Zen worker proposed for the user's memory — journal_read shows it as a \
            `proposal` — once the user agrees it is true; `journal` is its address, `<path>#<line id>`. A \
            proposal they reject is forgotten with journal_forget.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "journal": {"type": "string", "description": "The proposal's address, `<path>#<line id>`."}
            },
            "required": ["journal"],
            "additionalProperties": false
        }
    })
}

pub fn call(ctx: &Ctx, args: &Value) -> Answer {
    let journal = match string(args, "journal") {
        Ok(j) => j,
        Err(e) => return Answer::error(e),
    };
    let m = match mem::open(ctx) {
        Ok(m) => m,
        Err(e) => return Answer::error(e),
    };
    let address = match m.address(journal) {
        Ok(a) => a,
        Err(e) => return Answer::error(e),
    };
    match m.memory.write(
        &m.actor(),
        &Op::Accept {
            address: address.clone(),
        },
    ) {
        Ok(_) => Answer::ok(format!("Accepted {address}.")),
        Err(e) => Answer::error(e),
    }
}
