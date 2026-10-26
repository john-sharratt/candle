//! `repo_job_log {node, job, from?, grep?}`: read an agent job's log
//! (`docs/claude_zend_gateway.md` §8.3), a page at a time by byte offset — the
//! agent hands out at most 256 KiB per read — and, with `grep`, only the lines
//! of the page that hold it. Admins only.

use std::fmt::Display;

use agent::wire::{JobView, Staged};
use serde_json::{json, Value};

use super::{string, Answer, Ctx};
use crate::mcp::ops;

pub const NAME: &str = "repo_job_log";

pub fn definition() -> Value {
    json!({
        "name": NAME,
        "description": "Read a job `repo_job_start` started: how it stands — running or ended, its exit code, and \
            for a job that staged a build, where the build is — then its log from byte `from` (0 by \
            default), a page of up to 256 KiB with the offset to read on from. With `grep`, only the page's \
            lines that hold it (ignoring case), numbered as in the page. Read the last page for a test \
            summary or a build error. Admins only.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "node": {"type": "string", "description": "The agent's node id, as `repo_job_start` answered it."},
                "job": {"type": "string", "description": "The job's id."},
                "from": {"type": "integer", "minimum": 0, "description": "The byte to read from."},
                "grep": {"type": "string", "description": "Only lines holding this text, ignoring case."}
            },
            "required": ["node", "job"],
            "additionalProperties": false
        }
    })
}

/// A page of log as the tool answers it: where it ends and where to read on,
/// then its text — every line, or with `grep` only those holding it.
pub fn render(
    text: &str,
    from: u64,
    read: u64,
    size: u64,
    ended: bool,
    grep: Option<&str>,
) -> String {
    let to = from + read;
    let mut out = format!(
        "bytes {from}–{to} of {size}{}",
        if ended {
            ", the job has ended"
        } else {
            ", the job is still running"
        }
    );
    if to < size {
        out.push_str(&format!("; read on with `from: {to}`"));
    }
    match grep {
        Some(pattern) => {
            let needle = pattern.to_lowercase();
            let hits: Vec<String> = text
                .lines()
                .enumerate()
                .filter(|(_, l)| l.to_lowercase().contains(&needle))
                .map(|(i, l)| format!("{}: {l}", i + 1))
                .collect();
            out.push_str(&format!("\n{} lines hold `{pattern}`", hits.len()));
            for h in hits {
                out.push('\n');
                out.push_str(&h);
            }
        }
        None => {
            out.push('\n');
            out.push_str(text);
        }
    }
    out
}

/// Why `job` on `node` could not be read: `e`, and what that means.
fn lost(job: &str, node: u64, e: &impl Display) -> String {
    format!(
        "job {job} on {node:016x}: {e} — a job lives in its agent's process, and one lost to a \
         restart is gone: run it again"
    )
}

pub async fn call(ctx: &Ctx, args: &Value) -> Answer {
    if let Err(e) = ops::admin_only(&ctx.caller, NAME) {
        return Answer::error(e);
    }
    let node = match string(args, "node").and_then(ops::node) {
        Ok(n) => u64::from_str_radix(&n, 16).expect("sixteen hex digits"),
        Err(e) => return Answer::error(e),
    };
    let job = match string(args, "job") {
        Ok(j) => j,
        Err(e) => return Answer::error(e),
    };
    let from = args.get("from").and_then(Value::as_u64).unwrap_or(0);
    let Some(agents) = ctx.session.agents() else {
        return Answer::error(
            "this member reaches no agent: it has no cluster and no agent of its own",
        );
    };
    // The member's own agent is reached in process while it is cut off from
    // every cluster, and through the gateway otherwise.
    let at = agents.at(node);
    let head = match agents.view(at, job).await {
        Ok(view) => headline(&view),
        Err(e) => return Answer::error(lost(job, node, &e)),
    };
    let chunk = match agents.tail(at, job, from).await {
        Ok(c) => c,
        Err(e) => return Answer::error(lost(job, node, &e)),
    };
    let bytes = match chunk.bytes() {
        Ok(b) => b,
        Err(e) => return Answer::error(e.to_string()),
    };
    Answer::ok(format!(
        "{head}\n{}",
        render(
            &String::from_utf8_lossy(&bytes),
            chunk.from,
            bytes.len() as u64,
            chunk.size,
            chunk.ended,
            args.get("grep").and_then(Value::as_str),
        )
    ))
}

/// How a job stands, in a line: its state, its exit code once it has one,
/// and what it staged.
pub fn headline(view: &JobView) -> String {
    let mut out = format!("job {}: {:?}", view.id, view.status);
    if let Some(o) = &view.outcome {
        out.push_str(&format!(
            "; exit code {}{}",
            o.exit_code.map_or_else(|| "none".into(), |c| c.to_string()),
            if o.timed_out { ", timed out" } else { "" }
        ));
    }
    match &view.staged {
        Some(Staged::At(path)) => out.push_str(&format!("; staged at {path}")),
        Some(Staged::Not(why)) => out.push_str(&format!("; nothing staged: {why}")),
        None => {}
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_page_says_where_it_ends_and_where_to_read_on() {
        assert_eq!(
            render("a\nb", 0, 3, 10, false, None),
            "bytes 0–3 of 10, the job is still running; read on with `from: 3`\na\nb"
        );
        assert_eq!(
            render("x", 9, 1, 10, true, None),
            "bytes 9–10 of 10, the job has ended\nx"
        );
    }

    #[test]
    fn grep_keeps_the_lines_that_hold_it_numbered_in_the_page() {
        assert_eq!(
            render(
                "ok\nFAILED a\nok\nfailed b",
                0,
                24,
                24,
                true,
                Some("failed")
            ),
            "bytes 0–24 of 24, the job has ended\n2 lines hold `failed`\n2: FAILED a\n4: failed b"
        );
    }
}
