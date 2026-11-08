//! A call that changed something, kept on its conversation's `journal.yaml`
//! (`docs/zen_memory_and_guides.md` D3): what the server records after each
//! such call, in place of an episode line, so a conversation's own journal
//! says what it did without filling its window or its group's provenance.

use crate::error::{Error, Result};
use crate::ops::Applied;
use crate::path::JournalPath;
use crate::time::Stamp;
use crate::tree::Tree;

/// Keep `tool`, asked `subject`, on the conversation `path`.
pub fn record(
    tree: &mut Tree,
    path: &JournalPath,
    tool: &str,
    subject: &str,
    now: Stamp,
) -> Result<Applied> {
    let tool = tool.trim();
    if tool.is_empty() {
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
    meta.activity.push(now, tool, subject.trim());
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
        let op = |path: &JournalPath| Op::Activity {
            path: path.clone(),
            tool: "log_grep".into(),
            subject: " wave too wide ".into(),
        };
        let a = apply(&mut t, &server(), &op(&conv), now()).unwrap();
        assert_eq!(a.subject, format!("Activity on {conv}"));
        let j = t.get(&conv).unwrap();
        assert!(j.window.is_empty(), "no line was written");
        let done = &j.meta.as_ref().unwrap().activity.done;
        assert_eq!((done[0].tool.as_str(), done[0].subject.as_str()), ("log_grep", "wave too wide"));
        assert!(apply(&mut t, &server(), &op(&p("topics")), now())
            .unwrap_err()
            .to_string()
            .contains("record a conversation's calls there"));
        // Past the permit, the op itself holds the same line.
        assert!(record(&mut t, &p("topics"), "log_grep", "x", now())
            .unwrap_err()
            .to_string()
            .contains("no conversation"));
    }
}
