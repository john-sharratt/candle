//! `conversation_ref_seen {used_ms?}`: when the call's conversation ref was
//! last used, as the record both aggregators read says
//! (`docs/private_mode.md` §3.2) — told a use first when `used_ms` is given.
//! The aggregators call it by name, telling a use in the background and
//! asking only for a ref older than they can vouch for; it is in no listing,
//! and the aggregator refuses it from a client.

use serde_json::{json, Value};

use super::{Answer, Ctx, LOADING};
use crate::mcp::records::Records;
use crate::mcp::ref_records;
use crate::mcp::tools::mem::NO_CONVERSATION;
use crate::mcp::workers::now_ms;

pub const NAME: &str = "conversation_ref_seen";

pub fn call(ctx: &Ctx, args: &Value) -> Answer {
    let Some(r) = ctx.conversation else {
        return Answer::error(NO_CONVERSATION);
    };
    let Some(records) = ctx.session.shared_substrate() else {
        return Answer::error(LOADING);
    };
    let conversation = r.conversation();
    // A use is never later than now: told one in the future, a ref would
    // outlive its four idle hours.
    let told = args
        .get("used_ms")
        .and_then(Value::as_u64)
        .map(|at| at.min(now_ms()));
    let used = match told {
        Some(at) => match ref_records::touch(&records as &dyn Records, &conversation, at) {
            Ok(at) => Some(at),
            Err(e) => return Answer::error(e.to_string()),
        },
        None => ref_records::used(&records as &dyn Records, &conversation),
    };
    Answer::ok(json!({"used_ms": used}).to_string())
}
