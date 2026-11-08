//! What a note says about itself beside its text
//! (`docs/zen_memory_and_guides.md` Phase 2): whether the assistant writing
//! it concluded it rather than heard it, where it came from, when it stops
//! being true, and which line it supersedes.
//!
//! A line past its `expires`, and a line another line `replaces`, is hidden:
//! kept in its file and in history, never shown by a read, a search or a
//! frame.

use std::collections::HashSet;

use crate::error::Problem;
use crate::line::{Line, LineId};
use crate::time::Stamp;
use crate::yaml::{scalar, Fields};

/// The longest a `source` may be, in characters.
pub const MAX_SOURCE: usize = 300;

/// A note's terms. Every field is absent on a line that states none.
#[derive(Clone, Debug, Default, PartialEq, Eq, Hash)]
pub struct Terms {
    /// The assistant writing it concluded it — from the code, a document, its
    /// own reasoning — rather than heard the speaker say it.
    pub concluded: bool,
    /// Where it came from: a document's section, a file and line, a commit, a
    /// URL.
    pub source: Option<String>,
    /// When it stops being true.
    pub expires: Option<Stamp>,
    /// The line of the same journal it supersedes.
    pub replaces: Option<LineId>,
}

impl Terms {
    /// The terms' own rules, apart from the line they are on.
    pub fn check(&self) -> Result<(), String> {
        if let Some(s) = &self.source {
            let n = s.chars().count();
            if s.trim().is_empty() {
                return Err("its `source` is empty".into());
            }
            if n > MAX_SOURCE {
                return Err(format!("its `source` is {n} characters, over {MAX_SOURCE}"));
            }
        }
        Ok(())
    }

    /// The terms' fields of a line's file, in order, each on its own line.
    pub(crate) fn render(&self) -> String {
        let mut out = String::new();
        if self.concluded {
            out.push_str("concluded: true\n");
        }
        if let Some(s) = &self.source {
            out.push_str(&format!("source: {}\n", scalar(s)));
        }
        if let Some(e) = self.expires {
            out.push_str(&format!("expires: {e}\n"));
        }
        if let Some(r) = &self.replaces {
            out.push_str(&format!("replaces: {}\n", scalar(r.as_str())));
        }
        out
    }

    /// The terms read back from a line's file.
    pub(crate) fn read(f: &mut Fields<'_>) -> Result<Terms, Problem> {
        Ok(Terms {
            concluded: f.opt_bool("concluded")?.unwrap_or(false),
            source: f.opt_str("source")?,
            expires: f.opt_parsed("expires", Stamp::parse)?,
            replaces: f.opt_parsed("replaces", LineId::parse)?,
        })
    }
}

/// The lines of `window` shown at `now`: those not past their `expires` and
/// not superseded by another line's `replaces`, in their order.
pub fn shown(window: &[Line], now: Stamp) -> Vec<&Line> {
    let replaced: HashSet<&LineId> = window
        .iter()
        .filter_map(|l| l.terms.replaces.as_ref())
        .collect();
    window
        .iter()
        .filter(|l| !replaced.contains(&l.id) && is_current(l, now))
        .collect()
}

/// Whether `line` is not yet past its `expires` at `now`.
pub fn is_current(line: &Line, now: Stamp) -> bool {
    line.terms.expires.is_none_or(|e| e > now)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testing::note;

    fn at(s: &str) -> Stamp {
        Stamp::parse(s).unwrap()
    }

    /// **Each term is written as its own field and read back as it was**, and
    /// a line with none writes none.
    #[test]
    fn terms_are_written_exactly_and_read_back() {
        let mut line = note("2026-11-08T05:00:00Z", "The ingest key version is 4.");
        assert_eq!(
            line.render(),
            "at: 2026-11-08T05:00:00Z\nkind: note\nby: claude-code\naccount: john\n\
             text: The ingest key version is 4.\n"
        );
        line.terms = Terms {
            concluded: true,
            source: Some("zend/src/branch_ingest/keys.rs:12".into()),
            expires: Some(at("2026-12-01T00:00:00Z")),
            replaces: Some(LineId::parse("2026-10-31-0034-7c25").unwrap()),
        };
        assert_eq!(
            line.render(),
            "at: 2026-11-08T05:00:00Z\nkind: note\nby: claude-code\naccount: john\n\
             concluded: true\nsource: zend/src/branch_ingest/keys.rs:12\n\
             expires: 2026-12-01T00:00:00Z\nreplaces: 2026-10-31-0034-7c25\n\
             text: The ingest key version is 4.\n"
        );
    }

    /// **A line past its expiry, and one another line replaces, are not
    /// shown**; the rest keep their order.
    #[test]
    fn expired_and_replaced_lines_are_not_shown() {
        let old = note("2026-10-31T00:34:00Z", "The key version is 2.");
        let mut reminder = note("2026-10-29T20:03:00Z", "Service at 9am today.");
        reminder.terms.expires = Some(at("2026-10-31T00:00:00Z"));
        let mut new = note("2026-11-08T05:00:00Z", "The key version is 4.");
        new.terms.replaces = Some(old.id.clone());
        let lasting = note("2026-11-01T00:00:00Z", "John works on Sydney time.");
        let window = vec![reminder.clone(), old, lasting.clone(), new.clone()];
        let texts = |now: &str| -> Vec<String> {
            shown(&window, at(now))
                .into_iter()
                .map(|l| l.text.clone())
                .collect()
        };
        assert_eq!(
            texts("2026-11-08T06:00:00Z"),
            ["John works on Sydney time.", "The key version is 4."]
        );
        assert_eq!(
            texts("2026-10-30T12:00:00Z"),
            [
                "Service at 9am today.",
                "John works on Sydney time.",
                "The key version is 4."
            ],
            "before its expiry a reminder is shown"
        );
    }

    #[test]
    fn an_empty_or_overlong_source_is_wrong() {
        let empty = Terms {
            source: Some("  ".into()),
            ..Terms::default()
        };
        assert_eq!(empty.check().unwrap_err(), "its `source` is empty");
        let long = Terms {
            source: Some("x".repeat(MAX_SOURCE + 1)),
            ..Terms::default()
        };
        assert!(long.check().unwrap_err().contains("over 300"));
        assert!(Terms::default().check().is_ok());
    }
}
