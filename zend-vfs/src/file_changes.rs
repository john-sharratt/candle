//! One conversation's changes to one repository: for each path it changed,
//! the chain of [`FileDelta`]s that took the file from the repository's copy
//! to where the conversation has it now.
//!
//! This is what moves between the pieces of a tool run: materialising a
//! conversation onto a checkout replays every chain onto the files there, and
//! capturing what a tool did afterwards yields one delta per changed path,
//! appended here with [`FileChanges::extend`].
//!
//! A chain keeps only what still matters. A replace or a delete settles the
//! file on its own ([`FileDelta::supersedes`]), so appending one drops every
//! delta before it; edits after it stack in order.
//!
//! Every delta carries the moment it was made ([`TimedDelta`]), so each
//! changed file has times of its own ([`FileChanges::times`]) without any
//! filesystem being asked.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

use super::file_delta::{self, Diverged, FileDelta, FileTimes, TimedDelta};

/// Per path, the chain of deltas from the repository's copy to the
/// conversation's. Paths are repository-relative, `/`-separated.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(transparent)]
pub struct FileChanges {
    chains: BTreeMap<String, Vec<TimedDelta>>,
}

impl FileChanges {
    pub fn new() -> Self {
        Self::default()
    }

    /// Append `delta`, made now, to `path`'s chain. A delta that settles the
    /// file on its own replaces the chain instead.
    pub fn push(&mut self, path: impl Into<String>, delta: FileDelta) {
        self.push_timed(path, TimedDelta::now(delta));
    }

    /// Append `delta` with the moment it was made, as [`Self::push`] does.
    pub fn push_timed(&mut self, path: impl Into<String>, delta: TimedDelta) {
        let chain = self.chains.entry(path.into()).or_default();
        if delta.delta.supersedes() {
            chain.clear();
        }
        chain.push(delta);
    }

    /// Append each `(path, delta)` in order, as [`Self::push_timed`] does.
    pub fn extend(&mut self, deltas: impl IntoIterator<Item = (String, TimedDelta)>) {
        for (path, delta) in deltas {
            self.push_timed(path, delta);
        }
    }

    /// `path`'s chain, oldest first, or `None` when the conversation has not
    /// changed it.
    pub fn chain(&self, path: &str) -> Option<&[TimedDelta]> {
        self.chains.get(path).map(Vec::as_slice)
    }

    /// When `path`'s current chain began and when its latest delta was made,
    /// or `None` when the conversation has not changed it.
    pub fn times(&self, path: &str) -> Option<FileTimes> {
        FileTimes::of(self.chains.get(path)?)
    }

    /// Every changed path with its chain, in path order.
    pub fn iter(&self) -> impl Iterator<Item = (&str, &[TimedDelta])> {
        self.chains.iter().map(|(p, c)| (p.as_str(), c.as_slice()))
    }

    /// Every changed path, in path order.
    pub fn paths(&self) -> impl Iterator<Item = &str> {
        self.chains.keys().map(String::as_str)
    }

    pub fn len(&self) -> usize {
        self.chains.len()
    }

    pub fn is_empty(&self) -> bool {
        self.chains.is_empty()
    }

    /// The bytes every delta holds, together.
    pub fn bytes(&self) -> usize {
        self.chains
            .values()
            .flatten()
            .map(|t| t.delta.bytes())
            .sum()
    }

    /// What `path` holds once its chain is replayed onto `base` — the
    /// repository's copy, `None` when it has none. A path with no chain is
    /// `base` itself.
    pub fn replay(&self, path: &str, base: Option<Vec<u8>>) -> Result<Option<Vec<u8>>, Diverged> {
        match self.chains.get(path) {
            Some(chain) => file_delta::replay_bytes(base, chain.iter().map(|t| &t.delta)),
            None => Ok(base),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::file_delta::Splice;

    fn edit(at: usize, removed: &str, inserted: &str) -> FileDelta {
        FileDelta::Edit {
            splices: vec![Splice {
                at,
                removed: removed.into(),
                inserted: inserted.into(),
            }],
        }
    }

    fn replace(content: &str) -> FileDelta {
        FileDelta::Replace {
            content: content.into(),
        }
    }

    fn at(at_ns: i64, delta: FileDelta) -> TimedDelta {
        TimedDelta { at_ns, delta }
    }

    /// **Edits stack; a replace or a delete starts the chain again** — and
    /// the file's times follow: its chain began at the settling delta, and its
    /// latest change is the last one.
    #[test]
    fn a_settling_delta_replaces_the_chain() {
        let mut c = FileChanges::new();
        c.push_timed("a.rs", at(1, edit(0, "x\n", "y\n")));
        c.push_timed("a.rs", at(2, edit(0, "y\n", "z\n")));
        assert_eq!(c.chain("a.rs").unwrap().len(), 2);
        assert_eq!(
            c.times("a.rs"),
            Some(FileTimes {
                ctime_ns: 1,
                mtime_ns: 2
            })
        );
        c.push_timed("a.rs", at(3, replace("fresh\n")));
        assert_eq!(c.chain("a.rs"), Some(&[at(3, replace("fresh\n"))][..]));
        c.push_timed("a.rs", at(4, edit(0, "fresh\n", "fresher\n")));
        assert_eq!(
            c.times("a.rs"),
            Some(FileTimes {
                ctime_ns: 3,
                mtime_ns: 4
            })
        );
        c.push_timed("a.rs", at(5, FileDelta::Delete));
        assert_eq!(c.chain("a.rs"), Some(&[at(5, FileDelta::Delete)][..]));
        c.push_timed(
            "a.rs",
            at(
                6,
                FileDelta::ReplaceBinary {
                    content: vec![0xff],
                },
            ),
        );
        assert_eq!(c.chain("a.rs").unwrap().len(), 1);
        assert_eq!(c.times("b.rs"), None);
    }

    /// `push` stamps the delta with the moment it is recorded.
    #[test]
    fn push_records_the_moment() {
        let mut c = FileChanges::new();
        let before = file_delta::now_ns();
        c.push("a.rs", replace("x"));
        let times = c.times("a.rs").unwrap();
        assert!(times.mtime_ns >= before && times.mtime_ns <= file_delta::now_ns());
        assert_eq!(times.ctime_ns, times.mtime_ns);
    }

    /// Appending captured deltas replays to the same file as having made the
    /// changes one by one.
    #[test]
    fn extend_appends_in_order_and_replays() {
        let mut c = FileChanges::new();
        c.push("a.txt", edit(0, "one\n", "1\n"));
        c.extend([
            ("a.txt".to_string(), at(10, edit(2, "two\n", "2\n"))),
            ("b.txt".to_string(), at(11, replace("bee\n"))),
        ]);
        assert_eq!(
            c.replay("a.txt", Some(b"one\ntwo\n".to_vec())),
            Ok(Some(b"1\n2\n".to_vec()))
        );
        assert_eq!(c.replay("b.txt", None), Ok(Some(b"bee\n".to_vec())));
        assert_eq!(
            c.replay("untouched.txt", Some(b"same".to_vec())),
            Ok(Some(b"same".to_vec()))
        );
        assert_eq!(
            c.replay("a.txt", Some(b"other\n".to_vec())),
            Err(Diverged { at: 0 })
        );
        assert_eq!(c.paths().collect::<Vec<_>>(), ["a.txt", "b.txt"]);
        assert_eq!(c.len(), 2);
        assert_eq!(c.bytes(), 4 + 2 + 4 + 2 + 4);
    }

    /// The wire form is a plain map of path to chain — exactly these bytes.
    #[test]
    fn the_wire_form_is_a_map_of_chains() {
        let mut c = FileChanges::new();
        c.push_timed("gone.txt", at(7, FileDelta::Delete));
        c.push_timed("new.txt", at(8, replace("x")));
        let wire = serde_json::to_string(&c).unwrap();
        assert_eq!(
            wire,
            r#"{"gone.txt":[{"at_ns":7,"kind":"delete"}],"new.txt":[{"at_ns":8,"kind":"replace","content":"x"}]}"#
        );
        assert_eq!(serde_json::from_str::<FileChanges>(&wire).unwrap(), c);
        assert!(FileChanges::new().is_empty());
    }
}
