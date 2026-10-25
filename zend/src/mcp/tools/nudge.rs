//! `zend_nudge {worker, message}`: guidance for a running worker, taken at its
//! next tool round (`docs/claude_zend_gateway.md` §6.2).

use serde_json::{json, Value};

use super::{string, Answer, Ctx};
use crate::mcp::names::Path;
use crate::mcp::workers;

pub const NAME: &str = "zend_nudge";

pub fn definition() -> Value {
    json!({
        "name": NAME,
        "description": "Send guidance to a zend worker that is still running, unasked — when its journal or \
            its question shows it heading the wrong way, or repeating a step. The message joins its next \
            tool round's results, so it is read at the next point the worker looks up, never mid-thought; a \
            worker that ends its turn without another tool round never reads it, and the thread's journal \
            says so. For a worker that asked or is stuck, use `zend_answer`.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "worker": {"type": "string", "description": "The worker's path, `T:<thread>/<name>`."},
                "message": {"type": "string", "description": "The guidance, in a sentence or two."}
            },
            "required": ["worker", "message"],
            "additionalProperties": false
        }
    })
}

pub fn call(ctx: &Ctx, args: &Value) -> Answer {
    let (path, text) = match (string(args, "worker"), string(args, "message")) {
        (Ok(p), Ok(t)) => (p, t),
        (Err(e), _) | (_, Err(e)) => return Answer::error(e),
    };
    let (thread, name) = match Path::parse(path) {
        Ok(Path::Worker { thread, worker }) => (thread, worker),
        Ok(_) => {
            return Answer::error(format!(
                "`{path}` is not a worker; a worker is `T:<thread>/<name>`"
            ))
        }
        Err(e) => return Answer::error(e),
    };
    match workers::nudge(&ctx.session, &ctx.caller, &thread, &name, text) {
        Ok(w) => Answer::ok(format!(
            "Queued for {}: it reads it when its next tool round ends.",
            w.path()
        )),
        Err(e) => Answer::error(e),
    }
}
