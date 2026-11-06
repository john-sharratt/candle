//! One `file_edit_and_commit`: its changes applied to the branch's tip as one
//! commit by the repository's author, and published — all or nothing
//! (`docs/mcp_file_tools.md` §3.2).
//!
//! The changes are applied to what the branch holds on origin, never to a
//! working tree. The commit is published under a lease through the
//! repository's pusher, which is where a gated repository's gate is asked
//! and the push announced; a gate's refusal is the call's refusal. When the
//! branch moved between reading it and pushing, the changes are applied again
//! to its new tip, up to [`ATTEMPTS`] times: an edit whose `old` the other
//! commit changed then fails, naming the edit.

use std::collections::HashMap;

use zend_vfs::{
    BranchName, ChangeSet, FileMode, GitError, Oid, Published, Rejection, Repo, RepoPath, Rev,
    Signature,
};

use super::change::Change;
use super::edits::apply;

/// How many times a commit is made again onto a branch that moved under it.
pub const ATTEMPTS: usize = 3;

/// What the call did.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Outcome {
    /// The branch holds the commit.
    Live { commit: Oid, attempts: usize },
    /// Nothing moved, and why.
    Refused(String),
}

/// `changes` committed on `branch` as `author` with `message`, and published.
pub fn commit(
    repo: &Repo,
    branch: &BranchName,
    changes: &[Change],
    message: &str,
    author: &Signature,
) -> Result<Outcome, GitError> {
    if changes.is_empty() {
        return Ok(Outcome::Refused("there are no changes to commit".into()));
    }
    if message.trim().is_empty() {
        return Ok(Outcome::Refused("a commit needs a `message`".into()));
    }
    for attempt in 1..=ATTEMPTS {
        let pulled = repo.pull_branch(branch)?;
        let Some(onto) = pulled.record().or(pulled.tip.as_ref()).cloned() else {
            return Ok(Outcome::Refused(format!("there is no branch {branch}")));
        };
        let set = match changed(repo, &onto, changes)? {
            Ok(set) => set,
            Err(why) => return Ok(Outcome::Refused(why)),
        };
        let commit = repo.commit_changes(&onto, &set, message, author, author)?;
        match repo.publish_commit(&pulled, &commit)? {
            p if p.landed() => {
                return Ok(Outcome::Live {
                    commit,
                    attempts: attempt,
                })
            }
            Published::Behind | Published::Refused(Rejection::Stale) => continue,
            Published::Refused(why) => return Ok(Outcome::Refused(refused(&why))),
            other => {
                return Ok(Outcome::Refused(format!(
                    "the commit did not land: {other:?}"
                )))
            }
        }
    }
    Ok(Outcome::Refused(format!(
        "{branch} moved under the commit {ATTEMPTS} times, so nothing was committed; try again"
    )))
}

/// Why origin refused, in words.
fn refused(why: &Rejection) -> String {
    match why {
        Rejection::Remote(detail) | Rejection::Other(detail) => detail.clone(),
        Rejection::AtomicAborted => "the push was refused as a whole".into(),
        Rejection::Stale => "the branch moved while the commit was pushed".into(),
    }
}

fn path(p: &str) -> Result<RepoPath, String> {
    RepoPath::parse(p).map_err(|e| format!("{p}: {e}"))
}

/// The files as the changes leave them, read through `onto` and the changes
/// before: `None` for a file that is not there.
struct Files<'r> {
    repo: &'r Repo,
    onto: Rev,
    /// What the changes so far left at each path they touched: its bytes and
    /// the mode it takes — `None` to keep the mode the path already has —
    /// or `None` for a path they removed.
    pending: HashMap<String, Option<Content>>,
}

/// A file's new bytes, and the mode it takes — `None` to keep its own.
type Content = (Vec<u8>, Option<FileMode>);

impl Files<'_> {
    fn read(&self, p: &str) -> Result<Option<Vec<u8>>, String> {
        if let Some(held) = self.pending.get(p) {
            return Ok(held.as_ref().map(|(bytes, _)| bytes.clone()));
        }
        self.repo
            .blobs()
            .read_at(&self.onto, &path(p)?)
            .map_err(|e| format!("{p}: {e}"))
    }

    /// The mode of the file at `p` — an executable bit, a symlink — so a
    /// moved file keeps it.
    fn mode(&self, p: &str) -> Result<Option<FileMode>, String> {
        if let Some(Some((_, Some(mode)))) = self.pending.get(p) {
            return Ok(Some(*mode));
        }
        let at = path(p)?;
        let entries = self
            .repo
            .tree_entries(&self.onto, &[&at])
            .map_err(|e| format!("{p}: {e}"))?;
        Ok(entries.into_iter().find(|e| e.path == at).map(|e| e.mode))
    }

    fn text(&self, p: &str) -> Result<String, String> {
        let bytes = self
            .read(p)?
            .ok_or_else(|| format!("{p} does not exist; to make it, `write` it"))?;
        if bytes.contains(&0) {
            return Err(format!(
                "{p} is not a text file, and only text files are edited here"
            ));
        }
        String::from_utf8(bytes).map_err(|_| format!("{p} is not UTF-8 text"))
    }
}

/// The change set `changes` make on `onto`, or why one of them cannot.
fn changed(
    repo: &Repo,
    onto: &Oid,
    changes: &[Change],
) -> Result<Result<ChangeSet, String>, GitError> {
    let mut files = Files {
        repo,
        onto: Rev::Oid(onto.clone()),
        pending: HashMap::new(),
    };
    for change in changes {
        let step = match change {
            Change::Edit { path: p, edits } => files.text(p).and_then(|text| {
                apply(&text, &Change::edits(edits))
                    .map(|t| vec![(p.clone(), Some((t.into_bytes(), None)))])
                    .map_err(|m| m.describe(p))
            }),
            Change::Write { path: p, content } => {
                if content.contains('\0') {
                    Err(format!("{p}: the content is not text"))
                } else {
                    Ok(vec![(
                        p.clone(),
                        Some((content.clone().into_bytes(), None)),
                    )])
                }
            }
            Change::Delete { path: p } => match files.read(p) {
                Ok(Some(_)) => Ok(vec![(p.clone(), None)]),
                Ok(None) => Err(format!("{p} does not exist, so there is nothing to delete")),
                Err(e) => Err(e),
            },
            Change::Move { from, to } => match (files.read(from), files.read(to)) {
                (Ok(Some(bytes)), Ok(None)) => files
                    .mode(from)
                    .map(|mode| vec![(from.clone(), None), (to.clone(), Some((bytes, mode)))]),
                (Ok(None), _) => Err(format!("{from} does not exist, so it cannot be moved")),
                (_, Ok(Some(_))) => Err(format!(
                    "{to} already exists; delete it first or move elsewhere"
                )),
                (Err(e), _) | (_, Err(e)) => Err(e),
            },
        };
        match step {
            Ok(writes) => files.pending.extend(writes),
            Err(why) => return Ok(Err(why)),
        }
    }
    let mut set = ChangeSet::new();
    for (p, content) in files.pending {
        let at = match path(&p) {
            Ok(at) => at,
            Err(why) => return Ok(Err(why)),
        };
        match content {
            Some((bytes, mode)) => set.write(at, bytes, mode)?,
            None => set.delete(at)?,
        }
    }
    Ok(Ok(set))
}

#[cfg(test)]
mod tests {
    use std::path::Path;
    use std::process::Command;

    use zend_vfs::GitTime;

    use super::*;
    use crate::file_commit::change::EditArg;

    fn git(dir: &Path, args: &[&str]) -> String {
        let out = Command::new("git")
            .args([
                "-c",
                "user.name=t",
                "-c",
                "user.email=t@t",
                "-c",
                "commit.gpgsign=false",
            ])
            .args(args)
            .current_dir(dir)
            .output()
            .unwrap();
        assert!(
            out.status.success(),
            "git {args:?}: {}",
            String::from_utf8_lossy(&out.stderr)
        );
        String::from_utf8(out.stdout).unwrap()
    }

    /// A bare origin and a clone of it holding `files` on `main`.
    fn repo(files: &[(&str, &str)]) -> (tempfile::TempDir, tempfile::TempDir) {
        let origin = tempfile::tempdir().unwrap();
        git(origin.path(), &["init", "-q", "--bare", "-b", "main"]);
        let work = tempfile::tempdir().unwrap();
        git(work.path(), &["init", "-q", "-b", "main"]);
        for (p, text) in files {
            let at = work.path().join(p);
            std::fs::create_dir_all(at.parent().unwrap()).unwrap();
            std::fs::write(at, text).unwrap();
        }
        git(work.path(), &["add", "-A"]);
        git(work.path(), &["commit", "-q", "-m", "first"]);
        let url = origin.path().to_string_lossy().to_string();
        git(work.path(), &["remote", "add", "origin", &url]);
        git(work.path(), &["push", "-q", "origin", "main"]);
        (origin, work)
    }

    fn yen() -> Signature {
        Signature::new(
            "Yen",
            "yen@example.com",
            GitTime {
                seconds: 1_793_900_000,
                offset_minutes: 0,
            },
        )
        .unwrap()
    }

    fn main_branch() -> BranchName {
        BranchName::parse("main").unwrap()
    }

    /// What origin's `main` holds at `path`.
    fn on_origin(origin: &Path, path: &str) -> String {
        git(origin, &["show", &format!("main:{path}")])
    }

    /// **Edits, a write, a delete and a move land as one commit by the
    /// repository's author**, exactly as given.
    #[test]
    fn the_changes_land_as_one_commit_by_the_author() {
        let (origin, work) = repo(&[
            ("apps/a/style.css", "h1 { colour: red; }\n"),
            ("apps/a/old.js", "export const x = 1;\n"),
            ("apps/a/gone.txt", "bye\n"),
        ]);
        let r = Repo::open(work.path()).unwrap();
        let outcome = commit(
            &r,
            &main_branch(),
            &[
                Change::Edit {
                    path: "apps/a/style.css".into(),
                    edits: vec![EditArg {
                        old: "red".into(),
                        new: "var(--accent)".into(),
                    }],
                },
                Change::Write {
                    path: "apps/a/pages/home.js".into(),
                    content: "// Trang chủ\n".into(),
                },
                Change::Delete {
                    path: "apps/a/gone.txt".into(),
                },
                Change::Move {
                    from: "apps/a/old.js".into(),
                    to: "apps/a/new.js".into(),
                },
            ],
            "recipes: accent colour, home page",
            &yen(),
        )
        .unwrap();
        let Outcome::Live { commit, attempts } = outcome else {
            panic!("{outcome:?}")
        };
        assert_eq!(attempts, 1);
        assert_eq!(
            on_origin(origin.path(), "apps/a/style.css"),
            "h1 { colour: var(--accent); }\n"
        );
        assert_eq!(
            on_origin(origin.path(), "apps/a/pages/home.js").as_bytes(),
            "// Trang chủ\n".as_bytes()
        );
        assert_eq!(
            on_origin(origin.path(), "apps/a/new.js"),
            "export const x = 1;\n"
        );
        let listed = git(origin.path(), &["ls-tree", "-r", "--name-only", "main"]);
        assert_eq!(
            listed,
            "apps/a/new.js\napps/a/pages/home.js\napps/a/style.css\n"
        );
        assert_eq!(
            git(
                origin.path(),
                &[
                    "log",
                    "-1",
                    "--format=%an <%ae>|%cn <%ce>|%s",
                    commit.as_str()
                ]
            ),
            "Yen <yen@example.com>|Yen <yen@example.com>|recipes: accent colour, home page\n"
        );
    }

    /// **One change that cannot apply moves nothing**, and says which.
    #[test]
    fn a_change_that_cannot_apply_moves_nothing() {
        let (origin, work) = repo(&[("a.txt", "one\n")]);
        let before = git(origin.path(), &["rev-parse", "main"]);
        let r = Repo::open(work.path()).unwrap();
        let outcome = commit(
            &r,
            &main_branch(),
            &[
                Change::Write {
                    path: "b.txt".into(),
                    content: "b\n".into(),
                },
                Change::Edit {
                    path: "a.txt".into(),
                    edits: vec![EditArg {
                        old: "two".into(),
                        new: "three".into(),
                    }],
                },
            ],
            "m",
            &yen(),
        )
        .unwrap();
        assert_eq!(
            outcome,
            Outcome::Refused(
                "a.txt: edit 1's `old` is not in the file as it stands after the edits before \
                 it — read the file again and copy the text exactly"
                    .into()
            )
        );
        assert_eq!(git(origin.path(), &["rev-parse", "main"]), before);
    }

    /// **A branch that moved since it was read is committed onto again** —
    /// another writer's commit is kept, and the edit lands on top of it.
    #[test]
    fn a_branch_that_moved_is_committed_onto_again() {
        let (origin, work) = repo(&[("a.txt", "one\n"), ("b.txt", "bee\n")]);
        // Someone else pushes from another clone; this clone has not seen it.
        let other = tempfile::tempdir().unwrap();
        let url = origin.path().to_string_lossy().to_string();
        git(other.path(), &["clone", "-q", &url, "."]);
        std::fs::write(other.path().join("b.txt"), "bee two\n").unwrap();
        git(other.path(), &["commit", "-q", "-am", "theirs"]);
        git(other.path(), &["push", "-q", "origin", "main"]);

        let r = Repo::open(work.path()).unwrap();
        let outcome = commit(
            &r,
            &main_branch(),
            &[Change::Edit {
                path: "a.txt".into(),
                edits: vec![EditArg {
                    old: "one".into(),
                    new: "uno".into(),
                }],
            }],
            "mine",
            &yen(),
        )
        .unwrap();
        assert!(matches!(outcome, Outcome::Live { .. }), "{outcome:?}");
        assert_eq!(on_origin(origin.path(), "a.txt"), "uno\n");
        assert_eq!(
            on_origin(origin.path(), "b.txt"),
            "bee two\n",
            "theirs is kept"
        );
    }

    /// **A moved file keeps its mode**: an executable script is still one at
    /// its new path.
    #[test]
    fn a_moved_file_keeps_its_mode() {
        let (origin, work) = repo(&[("scripts/run.sh", "#!/bin/sh\necho hi\n")]);
        git(
            work.path(),
            &["update-index", "--chmod=+x", "scripts/run.sh"],
        );
        git(work.path(), &["commit", "-q", "-m", "executable"]);
        git(work.path(), &["push", "-q", "origin", "main"]);
        let r = Repo::open(work.path()).unwrap();
        let outcome = commit(
            &r,
            &main_branch(),
            &[Change::Move {
                from: "scripts/run.sh".into(),
                to: "bin/run.sh".into(),
            }],
            "move the script",
            &yen(),
        )
        .unwrap();
        assert!(matches!(outcome, Outcome::Live { .. }), "{outcome:?}");
        assert_eq!(
            git(origin.path(), &["ls-tree", "main", "bin/run.sh"])
                .split_whitespace()
                .next(),
            Some("100755")
        );
    }

    #[test]
    fn only_text_files_are_edited() {
        let (_origin, work) = repo(&[("a.txt", "one\n")]);
        let r = Repo::open(work.path()).unwrap();
        let refused =
            |changes: &[Change]| commit(&r, &main_branch(), changes, "m", &yen()).unwrap();
        assert_eq!(
            refused(&[Change::Write {
                path: "x.bin".into(),
                content: "a\0b".into()
            }]),
            Outcome::Refused("x.bin: the content is not text".into())
        );
        assert_eq!(
            refused(&[Change::Delete {
                path: "nope.txt".into()
            }]),
            Outcome::Refused("nope.txt does not exist, so there is nothing to delete".into())
        );
        assert_eq!(
            refused(&[]),
            Outcome::Refused("there are no changes to commit".into())
        );
    }
}
