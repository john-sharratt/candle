//! `get_user_context`: who the user is, what they asked to be kept, and what
//! was done for them lately — through this client first
//! (`docs/mcp_user_context.md` §2.2). The aggregator puts the client's guide
//! in front of the answer, and gives the same answer to the first call of a
//! conversation, whatever tool it is (§2.5).

use serde_json::{json, Value};

use super::{Answer, Ctx};
use crate::mcp::memory::{self, Context};
use crate::mcp::workers::now_ms;

pub const NAME: &str = "get_user_context";

pub fn definition() -> Value {
    json!({
        "name": NAME,
        "annotations": {"readOnlyHint": true},
        "description": "Who this user is, what they asked you to remember, and what you did together in \
            recent conversations — pictures drawn, bots given missions, work started — in this app first, \
            then in their others; and how to work with them here. Call this before answering anything from \
            this user, especially anything that assumes shared history: \"again\", \"last time\", \"like \
            before\", \"my\". It answers at once.",
        "inputSchema": {"type": "object", "properties": {}, "additionalProperties": false}
    })
}

pub fn call(ctx: &Ctx, _args: &Value) -> Answer {
    let Some(records) = ctx.session.shared_substrate() else {
        return Answer::error("the model is still loading; try again shortly");
    };
    let notes = memory::entries(&records, &ctx.caller.key());
    Answer::ok(
        Context::of(
            &ctx.caller.person.name,
            ctx.mcp.client.name(),
            &notes,
            now_ms(),
        )
        .render(),
    )
}
