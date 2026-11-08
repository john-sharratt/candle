//! What a conversation's calls did, kept on its `journal.yaml`
//! (`docs/zen_memory_and_guides.md` D3): each call that changed something, its
//! tool and what it was asked, newest last — never a line of its own, so no
//! provenance reads it and no window fills with it. Past [`MAX_DONE`] the
//! oldest are let go into a count per tool.

use std::collections::BTreeMap;

use serde_yaml::Value;

use crate::error::Problem;
use crate::line::LineId;
use crate::time::Stamp;
use crate::yaml::{item_line, scalar, Fields};

/// The most calls a conversation keeps, newest last.
pub const MAX_DONE: usize = 200;

/// The longest a call's subject is kept, in characters.
pub const MAX_SUBJECT: usize = 120;

/// One call that changed something.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Done {
    pub at: Stamp,
    pub tool: String,
    /// What it was asked, cut to [`MAX_SUBJECT`].
    pub subject: String,
    /// The id of the episode line it was kept as, when one was moved in from
    /// its entry file: the line's id is kept with it.
    pub id: Option<LineId>,
}

/// A conversation's calls.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Activity {
    /// The newest [`MAX_DONE`], oldest first.
    pub done: Vec<Done>,
    /// How many calls of each tool were let go past the cap.
    pub folded: BTreeMap<String, u64>,
}

impl Activity {
    pub fn is_empty(&self) -> bool {
        self.done.is_empty() && self.folded.is_empty()
    }

    /// Keep a call, letting the oldest go past the cap.
    pub fn push(&mut self, at: Stamp, tool: &str, subject: &str) {
        self.done.push(Done {
            at,
            tool: tool.to_string(),
            subject: subject.chars().take(MAX_SUBJECT).collect(),
            id: None,
        });
        while self.done.len() > MAX_DONE {
            let gone = self.done.remove(0);
            *self.folded.entry(gone.tool).or_insert(0) += 1;
        }
    }

    /// The newest `n` calls, newest first.
    pub fn newest(&self, n: usize) -> impl Iterator<Item = &Done> {
        self.done.iter().rev().take(n)
    }

    /// Its fields of a `journal.yaml`. Each call is a block mapping, a field
    /// to a line, so a subject is quoted by the rules of a plain value — in a
    /// flow mapping a comma would end it.
    pub(crate) fn render(&self, out: &mut String) {
        if !self.done.is_empty() {
            out.push_str("activity:\n");
            for d in &self.done {
                out.push_str(&format!("  - at: {}\n", d.at));
                out.push_str(&format!("    tool: {}\n", scalar(&d.tool)));
                out.push_str(&format!("    subject: {}\n", scalar(&d.subject)));
                if let Some(id) = &d.id {
                    out.push_str(&format!("    id: {}\n", scalar(id.as_str())));
                }
            }
        }
        if !self.folded.is_empty() {
            out.push_str("activity_folded:\n");
            for (tool, n) in &self.folded {
                out.push_str(&format!("  {}: {n}\n", scalar(tool)));
            }
        }
    }

    /// Its fields read back from a `journal.yaml`.
    pub(crate) fn read(f: &mut Fields<'_>, file: &str, text: &str) -> Result<Activity, Problem> {
        let mut done = Vec::new();
        for (i, dv) in f.opt_seq("activity")?.into_iter().flatten().enumerate() {
            let mut df = Fields::of(dv, file, item_line(text, "activity", i), None)?;
            done.push(Done {
                at: df.parsed("at", Stamp::parse)?,
                tool: df.str("tool")?,
                subject: df.opt_str("subject")?.unwrap_or_default(),
                id: df.opt_parsed("id", LineId::parse)?,
            });
            df.done()?;
        }
        if done.len() > MAX_DONE {
            return Err(f.problem(
                "activity",
                format!("{} calls kept: at most {MAX_DONE}", done.len()),
            ));
        }
        let mut folded = BTreeMap::new();
        if let Some(Value::Mapping(m)) = f.opt_map("activity_folded")? {
            for (k, v) in m {
                let (Some(tool), Some(n)) = (k.as_str(), v.as_u64()) else {
                    return Err(f.problem(
                        "activity_folded",
                        "`activity_folded` counts each tool's calls: `tool: n`",
                    ));
                };
                folded.insert(tool.to_string(), n);
            }
        }
        Ok(Activity { done, folded })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn at(s: &str) -> Stamp {
        Stamp::parse(s).unwrap()
    }

    /// **Past the cap the oldest go into a count per tool**, the newest kept in
    /// order, and a long subject is cut.
    #[test]
    fn past_the_cap_the_oldest_are_counted() {
        let mut a = Activity::default();
        a.push(at("2026-11-08T03:00:00Z"), "log_grep", &"x".repeat(300));
        assert_eq!(a.done[0].subject.chars().count(), MAX_SUBJECT);
        for i in 0..MAX_DONE {
            a.push(
                at("2026-11-08T03:00:00Z").plus(i as i64 + 1),
                "job_status",
                "a job",
            );
        }
        assert_eq!(a.done.len(), MAX_DONE);
        assert_eq!(a.folded, BTreeMap::from([("log_grep".to_string(), 1)]));
        assert_eq!(a.newest(1).next().unwrap().at, at("2026-11-08T03:03:20Z"));
    }

    /// **Its fields are written exactly**: one call to a line, the counts
    /// after; nothing when it is empty.
    #[test]
    fn its_fields_are_written_exactly() {
        let mut out = String::new();
        Activity::default().render(&mut out);
        assert_eq!(out, "");
        let mut a = Activity::default();
        a.push(
            at("2026-11-08T03:27:00Z"),
            "log_grep",
            "^2026-11-08T0: wave",
        );
        a.push(
            at("2026-11-08T03:28:00Z"),
            "log_grep",
            "repo_task_start, job_status",
        );
        a.done[1].id = Some(LineId::parse("2026-11-08-0328-7e3a").unwrap());
        a.folded.insert("service_status".into(), 3);
        a.render(&mut out);
        assert_eq!(
            out,
            "activity:\n  - at: 2026-11-08T03:27:00Z\n    tool: log_grep\n    \
             subject: \"^2026-11-08T0: wave\"\n  - at: 2026-11-08T03:28:00Z\n    \
             tool: log_grep\n    subject: repo_task_start, job_status\n    \
             id: 2026-11-08-0328-7e3a\nactivity_folded:\n  service_status: 3\n"
        );
    }
}
