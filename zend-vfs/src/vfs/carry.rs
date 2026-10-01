//! Moving a store onto another base, and what becomes of its changes.
//!
//! A store's chains are made against its base's tree; a move puts them
//! against another. A path the two trees hold identically needs nothing — its
//! chain reads the same over either. Every other path the session changed,
//! and any the caller names besides, is handed to a resolver with the
//! session's content and both trees' copies, and the answer is recorded
//! against the new tree: a chain from the new tree's copy to it, or no chain
//! at all when the answer is that copy.
//!
//! What the resolver answers is the move's policy, not this module's: a
//! commit of the session's own keeps what the session holds; a merge merges
//! the three versions and marks where they overlap.

use std::collections::HashMap;

use super::view::View;
use super::{Chain, VfsError};
use crate::file_delta::{self, FileDelta, TimedDelta};

/// One side of a path a move reconsiders.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Side {
    /// No file there.
    Absent,
    Text(String),
    /// A file that is not text — or too large to read as it.
    Binary,
}

impl Side {
    /// `norm` as `view` holds it.
    fn of(view: &View<'_>, norm: &str) -> Result<Self, VfsError> {
        match view.read_text(norm) {
            Ok(Some(text)) => Ok(Side::Text(text)),
            Ok(None) => Ok(Side::Absent),
            Err(_) if view.is_file(norm) => Ok(Side::Binary),
            Err(e) => Err(e),
        }
    }
}

/// One path a move reconsiders: what the session holds there, and the old
/// and new trees' copies.
#[derive(Debug)]
pub struct Carried<'a> {
    pub path: &'a str,
    /// The session's content; `None` where it has no file.
    pub ours: Option<&'a str>,
    pub was: &'a Side,
    pub now: &'a Side,
}

/// What a path holds after a move.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Resolved {
    /// `None` for no file.
    pub content: Option<String>,
    /// Whether the path is left in conflict, for the session to settle.
    pub conflict: bool,
}

/// The resolver a move asks about each path it reconsiders.
pub type Resolver<'r> = dyn FnMut(&Carried<'_>) -> Result<Resolved, VfsError> + 'r;

/// Carry `chains`, made against `old`, onto `new`, reconsidering every
/// changed path the two hold differently and each of `extra`. Everything is
/// resolved before anything changes, so a failure leaves `chains` as it was.
pub(super) fn carry(
    chains: &mut HashMap<String, Chain>,
    old: &View<'_>,
    new: &View<'_>,
    extra: &[String],
    resolve: &mut Resolver<'_>,
) -> Result<(), VfsError> {
    let mut paths: Vec<String> = chains
        .keys()
        .filter(|path| !old.same_file(new, path))
        .cloned()
        .collect();
    for path in extra {
        if !paths.contains(path) {
            paths.push(path.clone());
        }
    }
    paths.sort();

    let mut settled = Vec::with_capacity(paths.len());
    for path in paths {
        let was = Side::of(old, &path)?;
        let ours = match chains.get(&path) {
            Some(chain) => match content(chain, &was, &path) {
                Ok(content) => content,
                // A change that no longer fits the copy it was made to cannot
                // be carried: it is kept exactly as it is, in conflict, for
                // the conversation to write out whole — never dropped, and
                // never in the way of the rest of the move.
                Err(_) => {
                    let mut kept = chain.clone();
                    kept.conflict = true;
                    settled.push((path, Some(kept)));
                    continue;
                }
            },
            None => match &was {
                Side::Text(text) => Some(text.clone()),
                Side::Absent => None,
                Side::Binary => {
                    return Err(VfsError::Unreadable(format!(
                        "{path} is not text, so it cannot be carried"
                    )))
                }
            },
        };
        let now = Side::of(new, &path)?;
        let resolved = resolve(&Carried {
            path: &path,
            ours: ours.as_deref(),
            was: &was,
            now: &now,
        })?;
        let chain = chains.get(&path);
        let at_ns = chain.and_then(|c| c.deltas.last()).map(|t| t.at_ns);
        // A move never settles a conflict: only a change of the
        // conversation's own does.
        let was_in_conflict = chain.is_some_and(|c| c.conflict);
        let resolved = Resolved {
            conflict: resolved.conflict || was_in_conflict,
            ..resolved
        };
        settled.push((path, chain_for(resolved, &now, at_ns)));
    }
    for (path, chain) in settled {
        match chain {
            Some(chain) => chains.insert(path, chain),
            None => chains.remove(&path),
        };
    }
    Ok(())
}

/// What `chain` holds, replayed onto `was` when it opens with an edit.
fn content(chain: &Chain, was: &Side, path: &str) -> Result<Option<String>, VfsError> {
    let base = match (chain.deltas.first().map(|t| &t.delta), was) {
        (Some(FileDelta::Edit { .. }), Side::Text(text)) => Some(text.clone()),
        (Some(FileDelta::Edit { .. }), _) => {
            return Err(VfsError::Diverged(format!(
                "{path}: the change was made to a file its base no longer holds as text"
            )))
        }
        _ => None,
    };
    file_delta::replay(base, chain.deltas.iter().map(|t| &t.delta)).map_err(|_| {
        VfsError::Diverged(format!(
            "{path}: the change no longer fits the file it was made to"
        ))
    })
}

/// The chain that makes `now` read as `resolved`, dated `at_ns` when given;
/// `None` when `now` already does and nothing is in conflict. A conflict
/// always keeps a chain — reading as `now` still, but flagged: the
/// conversation's own change there (a delete meeting an edit, say) is not
/// dropped without it deciding.
fn chain_for(resolved: Resolved, now: &Side, at_ns: Option<i64>) -> Option<Chain> {
    let size = resolved.content.as_ref().map(String::len);
    let settled = !resolved.conflict;
    let delta = match (resolved.content, now) {
        (Some(content), Side::Text(now)) if content == *now && settled => return None,
        (Some(content), Side::Text(now)) if content == *now => FileDelta::Replace { content },
        (None, Side::Absent) => return None,
        (None, _) => FileDelta::Delete,
        (Some(content), Side::Text(now)) => file_delta::delta(now, &content),
        (Some(content), _) => FileDelta::Replace { content },
    };
    let timed = match at_ns {
        Some(at_ns) => TimedDelta { at_ns, delta },
        None => TimedDelta::now(delta),
    };
    Some(Chain {
        deltas: vec![timed],
        size,
        conflict: resolved.conflict,
    })
}

#[cfg(test)]
mod tests {
    use std::path::Path;

    use super::*;

    fn put(root: &Path, rel: &str, body: &str) {
        let p = root.join(rel);
        std::fs::create_dir_all(p.parent().unwrap()).unwrap();
        std::fs::write(p, body).unwrap();
    }

    fn chain(deltas: Vec<FileDelta>) -> Chain {
        Chain {
            deltas: deltas
                .into_iter()
                .enumerate()
                .map(|(n, delta)| TimedDelta {
                    at_ns: 100 + n as i64,
                    delta,
                })
                .collect(),
            size: None,
            conflict: false,
        }
    }

    fn read(chains: &HashMap<String, Chain>, new: &View<'_>, path: &str) -> Option<String> {
        let now = Side::of(new, path).unwrap();
        content(&chains[path], &now, path).unwrap()
    }

    /// **Two folders never vouch that a path is the same file**, so every
    /// changed path is reconsidered between them — which lets these tests
    /// drive the resolver with folders standing in for trees. Keeping the
    /// session's content: each
    /// path reads as it did, as a chain against the new copy, dated by the
    /// chain's latest change; one the new copy already holds leaves nothing.
    #[test]
    fn keeping_the_sessions_content_reads_as_before() {
        let (old_dir, new_dir) = (tempfile::tempdir().unwrap(), tempfile::tempdir().unwrap());
        let (old_root, new_root) = (old_dir.path(), new_dir.path());
        put(old_root, "moved.txt", "a\nb\nc\n");
        put(new_root, "moved.txt", "A\nb\nc\n");
        put(old_root, "landed.txt", "x\n");
        put(new_root, "landed.txt", "y\n");
        let mut chains = HashMap::from([
            (
                "moved.txt".to_string(),
                chain(vec![file_delta::delta("a\nb\nc\n", "a\nb\nC\n")]),
            ),
            (
                "landed.txt".to_string(),
                chain(vec![file_delta::delta("x\n", "y\n")]),
            ),
            ("gone.txt".to_string(), chain(vec![FileDelta::Delete])),
        ]);
        let (old, new) = (View::Folder(old_root), View::Folder(new_root));
        let mut asked = Vec::new();
        carry(&mut chains, &old, &new, &[], &mut |c: &Carried<'_>| {
            asked.push(c.path.to_string());
            Ok(Resolved {
                content: c.ours.map(str::to_string),
                conflict: false,
            })
        })
        .unwrap();
        assert_eq!(asked, ["gone.txt", "landed.txt", "moved.txt"]);
        let mut left: Vec<&str> = chains.keys().map(String::as_str).collect();
        left.sort_unstable();
        assert_eq!(left, ["moved.txt"], "the new copies hold the rest");
        assert_eq!(read(&chains, &new, "moved.txt").unwrap(), "a\nb\nC\n");
        assert_eq!(chains["moved.txt"].deltas[0].at_ns, 100);
        assert_eq!(chains["moved.txt"].size, Some(6));
    }

    /// **A resolver's answer is recorded against the new copy, flag and
    /// all**, and a path named besides the changed ones is reconsidered too.
    #[test]
    fn a_resolvers_answer_and_conflict_are_recorded() {
        let (old_dir, new_dir) = (tempfile::tempdir().unwrap(), tempfile::tempdir().unwrap());
        put(old_dir.path(), "a.txt", "base\n");
        put(new_dir.path(), "a.txt", "theirs\n");
        put(old_dir.path(), "b.txt", "b\n");
        put(new_dir.path(), "b.txt", "b theirs\n");
        let mut chains = HashMap::from([(
            "a.txt".to_string(),
            chain(vec![file_delta::delta("base\n", "mine\n")]),
        )]);
        let new = View::Folder(new_dir.path());
        carry(
            &mut chains,
            &View::Folder(old_dir.path()),
            &new,
            &["b.txt".to_string()],
            &mut |c: &Carried<'_>| {
                Ok(match c.path {
                    "a.txt" => {
                        assert_eq!(
                            (c.ours, c.was, c.now),
                            (
                                Some("mine\n"),
                                &Side::Text("base\n".into()),
                                &Side::Text("theirs\n".into())
                            )
                        );
                        Resolved {
                            content: Some(
                                "<<<<<<< yours\nmine\n=======\ntheirs\n>>>>>>> t\n".into(),
                            ),
                            conflict: true,
                        }
                    }
                    _ => Resolved {
                        content: Some("b\n".into()),
                        conflict: false,
                    },
                })
            },
        )
        .unwrap();
        assert!(chains["a.txt"].conflict);
        assert!(!chains["b.txt"].conflict);
        assert_eq!(read(&chains, &new, "b.txt").unwrap(), "b\n");
    }

    /// **A move never settles a conflict**: a path in conflict stays flagged
    /// whatever the resolver answers, until a change of the conversation's
    /// own settles it.
    #[test]
    fn a_move_never_settles_a_conflict() {
        let (old_dir, new_dir) = (tempfile::tempdir().unwrap(), tempfile::tempdir().unwrap());
        put(old_dir.path(), "a.txt", "1\n2\n3\n");
        put(new_dir.path(), "a.txt", "1\n2\nthree\n");
        let mut marked = chain(vec![file_delta::delta(
            "1\n2\n3\n",
            "<<<<<<< yours\none\n=======\nuno\n>>>>>>> t\n2\n3\n",
        )]);
        marked.conflict = true;
        let mut chains = HashMap::from([("a.txt".to_string(), marked)]);
        carry(
            &mut chains,
            &View::Folder(old_dir.path()),
            &View::Folder(new_dir.path()),
            &[],
            &mut |c: &Carried<'_>| {
                Ok(Resolved {
                    content: c.ours.map(str::to_string),
                    conflict: false,
                })
            },
        )
        .unwrap();
        assert!(chains["a.txt"].conflict, "still in conflict");
    }

    /// **A change that no longer fits its copy is kept exactly, in conflict,
    /// and the rest of the move goes ahead.**
    #[test]
    fn a_change_that_no_longer_fits_is_kept_and_the_move_goes_on() {
        let (old_dir, new_dir) = (tempfile::tempdir().unwrap(), tempfile::tempdir().unwrap());
        put(old_dir.path(), "stale.txt", "something else\n");
        put(new_dir.path(), "stale.txt", "and again\n");
        put(old_dir.path(), "fine.txt", "a\n");
        put(new_dir.path(), "fine.txt", "b\n");
        let stale = chain(vec![file_delta::delta("one\ntwo\n", "one\n2\n")]);
        let mut chains = HashMap::from([
            ("stale.txt".to_string(), stale.clone()),
            (
                "fine.txt".to_string(),
                chain(vec![FileDelta::Replace {
                    content: "mine\n".into(),
                }]),
            ),
        ]);
        carry(
            &mut chains,
            &View::Folder(old_dir.path()),
            &View::Folder(new_dir.path()),
            &[],
            &mut |c: &Carried<'_>| {
                Ok(Resolved {
                    content: c.ours.map(str::to_string),
                    conflict: false,
                })
            },
        )
        .unwrap();
        assert_eq!(chains["stale.txt"].deltas, stale.deltas);
        assert!(chains["stale.txt"].conflict);
        let new = View::Folder(new_dir.path());
        assert_eq!(read(&chains, &new, "fine.txt").unwrap(), "mine\n");
    }

    /// **A failing resolver changes nothing.**
    #[test]
    fn a_failure_leaves_the_chains_as_they_were() {
        let (old_dir, new_dir) = (tempfile::tempdir().unwrap(), tempfile::tempdir().unwrap());
        put(old_dir.path(), "a.txt", "a\n");
        put(new_dir.path(), "a.txt", "b\n");
        let before = chain(vec![FileDelta::Replace {
            content: "mine\n".into(),
        }]);
        let mut chains = HashMap::from([
            ("a.txt".to_string(), before.clone()),
            ("z.txt".to_string(), before.clone()),
        ]);
        let mut calls = 0;
        let failed = carry(
            &mut chains,
            &View::Folder(old_dir.path()),
            &View::Folder(new_dir.path()),
            &[],
            &mut |_: &Carried<'_>| {
                calls += 1;
                if calls == 2 {
                    return Err(VfsError::Full);
                }
                Ok(Resolved {
                    content: None,
                    conflict: false,
                })
            },
        );
        assert!(failed.is_err());
        assert_eq!(chains["a.txt"].deltas, before.deltas);
        assert_eq!(chains["z.txt"].deltas, before.deltas);
    }
}
