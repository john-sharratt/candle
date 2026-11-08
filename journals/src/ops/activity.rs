//! Calls that changed something, kept on their conversation's `journal.yaml`
//! (`docs/zen_memory_and_guides.md` D3): what the server records of such
//! calls, in place of episode lines, so a conversation's own journal says what
//! it did without filling its window or its group's provenance. The server
//! gathers a conversation's calls and writes them together, each at the time
//! it was made.

use crate::error::{Error, Result};
use crate::ops::Applied;
use crate::path::JournalPath;
use crate::time::Stamp;
use crate::tree::Tree;

/// One call: when it was made, its tool, and what it was asked.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Call {
    pub at: Stamp,
    pub tool: String,
    pub subject: String,
}

/// Keep `calls` on the conversation `path`, in their order.
pub fn record(tree: &mut Tree, path: &JournalPath, calls: &[Call]) -> Result<Applied> {
    if calls.is_empty() {
        return Err(Error::invalid("an activity record holds at least one call"));
    }
    if calls.iter().any(|c| c.tool.trim().is_empty()) {
        return Err(Error::invalid("a call names its tool"));
    }
    let j = tree.need_mut(path)?;
    if !j.place.is_conversation() {
        return Err(Error::invalid(format!(
            "{path} is no conversation: only a conversation keeps what its calls did"
        )));
    }
    j.writable(&path.show())?;
    let meta = j
        .meta
        .as_mut()
        .ok_or_else(|| Error::invalid(format!("{path} has no journal.yaml to keep it on")))?;
    for c in calls {
        meta.activity.push(c.at, c.tool.trim(), c.subject.trim());
    }
    Ok(Applied::touching(
        format!("Activity on {path}"),
        vec![path.clone()],
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::actor::Actor;
    use crate::ops::fixture::{now, p, zen};
    use crate::ops::{apply, OnTaken, Op};
    use crate::scope::Channel;
    use crate::testing::slug;

    fn server() -> Actor {
        Actor::Server {
            person: slug("john"),
            channel: Channel::ClaudeCode,
        }
    }

    /// **A call is kept on its conversation's `journal.yaml`, not as a line**;
    /// anywhere else is refused.
    #[test]
    fn a_call_is_kept_on_its_conversation_not_as_a_line() {
        let mut t = zen();
        apply(
            &mut t,
            &server(),
            &Op::Create {
                under: p("conversations"),
                name: slug("2026-11-08-0308-claude-code"),
                cue: "open for the conversation through Claude Code begun 2026-11-08 03:08".into(),
                on_taken: OnTaken::Number,
            },
            now(),
        )
        .unwrap();
        let conv = p("conversations/2026-11-08-0308-claude-code");
        let calls = vec![
            Call {
                at: now(),
                tool: "log_grep".into(),
                subject: " wave too wide ".into(),
            },
            Call {
                at: now().plus(20),
                tool: "job_status".into(),
                subject: "task:x".into(),
            },
        ];
        let op = |path: &JournalPath| Op::Activity {
            path: path.clone(),
            calls: calls.clone(),
        };
        let a = apply(&mut t, &server(), &op(&conv), now().plus(60)).unwrap();
        assert_eq!(a.subject, format!("Activity on {conv}"));
        let j = t.get(&conv).unwrap();
        assert!(j.window.is_empty(), "no line was written");
        let done = &j.meta.as_ref().unwrap().activity.done;
        assert_eq!(
            done.iter()
                .map(|d| (d.at, d.tool.as_str(), d.subject.as_str()))
                .collect::<Vec<_>>(),
            [
                (now(), "log_grep", "wave too wide"),
                (now().plus(20), "job_status", "task:x")
            ],
            "each call at its own time, not the write's"
        );
        assert!(apply(&mut t, &server(), &op(&p("topics")), now())
            .unwrap_err()
            .to_string()
            .contains("record a conversation's calls there"));
        // Past the permit, the op itself holds the same line.
        assert!(record(&mut t, &p("topics"), &calls)
            .unwrap_err()
            .to_string()
            .contains("no conversation"));
        assert!(record(&mut t, &conv, &[]).is_err());
    }
}
