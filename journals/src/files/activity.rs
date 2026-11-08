//! What a conversation's calls did, kept on its `journal.yaml`
//! (`docs/zen_memory_and_guides.md` D3): each call that changed something, its
//! tool and what it was asked, newest last — never a line of its own, so no
//! provenance reads it and no window fills with it. Past [`MAX_DONE`] the
//! oldest are let go into a count per tool.

use std::collections::BTreeMap;

use serde_yaml::Value;

use crate::error::Problem;
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
}

/// A conversation's calls.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Activity {
    /// The newest [`MAX_DONE`], oldest first.
    pub done: Vec<Done>,
    /// How many calls of each tool were let go past the cap.
    pub before: BTreeMap<String, u64>,
}

impl Activity {
    pub fn is_empty(&self) -> bool {
        self.done.is_empty() && self.before.is_empty()
    }

    /// Keep a call, letting the oldest go past the cap.
    pub fn push(&mut self, at: Stamp, tool: &str, subject: &str) {
        self.done.push(Done {
            at,
            tool: tool.to_string(),
            subject: subject.chars().take(MAX_SUBJECT).collect(),
        });
        while self.done.len() > MAX_DONE {
            let gone = self.done.remove(0);
            *self.before.entry(gone.tool).or_insert(0) += 1;
        }
    }

    /// The newest `n` calls, newest first.
    pub fn newest(&self, n: usize) -> impl Iterator<Item = &Done> {
        self.done.iter().rev().take(n)
    }

    /// Its fields of a `journal.yaml`.
    pub(crate) fn render(&self, out: &mut String) {
        if !self.done.is_empty() {
            out.push_str("activity:\n");
            for d in &self.done {
                out.push_str(&format!(
                    "  - {{at: {}, tool: {}, subject: {}}}\n",
                    d.at,
                    scalar(&d.tool),
                    scalar(&d.subject)
                ));
            }
        }
        if !self.before.is_empty() {
            out.push_str("activity_before:\n");
            for (tool, n) in &self.before {
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
            });
            df.done()?;
        }
        if done.len() > MAX_DONE {
            return Err(f.problem(
                "activity",
                format!("{} calls kept: at most {MAX_DONE}", done.len()),
            ));
        }
        let mut before = BTreeMap::new();
        if let Some(Value::Mapping(m)) = f.opt_map("activity_before")? {
            for (k, v) in m {
                let (Some(tool), Some(n)) = (k.as_str(), v.as_u64()) else {
                    return Err(f.problem(
                        "activity_before",
                        "`activity_before` counts each tool's calls: `tool: n`",
                    ));
                };
                before.insert(tool.to_string(), n);
            }
        }
        Ok(Activity { done, before })
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
        assert_eq!(a.before, BTreeMap::from([("log_grep".to_string(), 1)]));
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
        a.push(at("2026-11-08T03:27:00Z"), "log_grep", "^2026-11-08T0: wave");
        a.before.insert("service_status".into(), 3);
        a.render(&mut out);
        assert_eq!(
            out,
            "activity:\n  - {at: 2026-11-08T03:27:00Z, tool: log_grep, subject: \"^2026-11-08T0: wave\"}\n\
             activity_before:\n  service_status: 3\n"
        );
    }
}
