//! Differences between commits, or between a commit and the working tree,
//! from git's raw `-z` diff format.

use crate::error::GitError;
use crate::runner::utf8;
use crate::types::{FileMode, Oid, RepoPath, Rev};
use crate::Repo;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DiffStatus {
    Added,
    Modified,
    Deleted,
    TypeChanged,
    Unmerged,
    /// With git's similarity score (0–100).
    Renamed(u8),
    Copied(u8),
}

/// One side of a change.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DiffSide {
    pub mode: FileMode,
    /// `None` for a working-tree file git has not hashed.
    pub oid: Option<Oid>,
    pub path: RepoPath,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DiffEntry {
    pub status: DiffStatus,
    /// Absent for an added path.
    pub old: Option<DiffSide>,
    /// Absent for a deleted path.
    pub new: Option<DiffSide>,
}

fn parse_status(s: &str) -> Result<DiffStatus, GitError> {
    let bad = || GitError::malformed("diff", format!("status {s:?}"));
    let score = || s[1..].parse::<u8>().map_err(|_| bad());
    Ok(match s.as_bytes().first() {
        Some(b'A') => DiffStatus::Added,
        Some(b'M') => DiffStatus::Modified,
        Some(b'D') => DiffStatus::Deleted,
        Some(b'T') => DiffStatus::TypeChanged,
        Some(b'U') => DiffStatus::Unmerged,
        Some(b'R') => DiffStatus::Renamed(score()?),
        Some(b'C') => DiffStatus::Copied(score()?),
        _ => return Err(bad()),
    })
}

/// Parse `--raw -z` output: `:<m1> <m2> <o1> <o2> <status>\0<path>\0`, with a
/// second path after a rename or copy.
pub(crate) fn parse_raw(out: &[u8]) -> Result<Vec<DiffEntry>, GitError> {
    let text = std::str::from_utf8(out).map_err(|e| GitError::malformed("diff", e.to_string()))?;
    let mut fields = text.split('\0');
    let mut entries = Vec::new();
    while let Some(header) = fields.next() {
        if header.is_empty() {
            continue;
        }
        let header = header
            .strip_prefix(':')
            .ok_or_else(|| GitError::malformed("diff", header.to_string()))?;
        let parts: Vec<&str> = header.split(' ').collect();
        let [m1, m2, o1, o2, status] = parts[..] else {
            return Err(GitError::malformed("diff", header.to_string()));
        };
        let status = parse_status(status)?;
        let mut path = || -> Result<RepoPath, GitError> {
            RepoPath::parse(
                fields
                    .next()
                    .ok_or_else(|| GitError::malformed("diff", "entry without a path"))?,
            )
        };
        let first = path()?;
        let second = match status {
            DiffStatus::Renamed(_) | DiffStatus::Copied(_) => Some(path()?),
            _ => None,
        };
        let (old_path, new_path) = match second {
            Some(to) => (first, to),
            None => (first.clone(), first),
        };
        let side = |mode: &str, oid: &str, path: RepoPath| -> Result<Option<DiffSide>, GitError> {
            Ok(FileMode::parse_present(mode)?.map(|mode| DiffSide {
                mode,
                oid: Oid::parse_nonzero(oid).ok().flatten(),
                path,
            }))
        };
        entries.push(DiffEntry {
            status,
            old: side(m1, o1, old_path)?,
            new: side(m2, o2, new_path)?,
        });
    }
    Ok(entries)
}

impl Repo {
    /// What changed from commit `from` to commit `to`, with renames, limited
    /// to `paths` when any are given.
    pub fn diff(
        &self,
        from: &Rev,
        to: &Rev,
        paths: &[&RepoPath],
    ) -> Result<Vec<DiffEntry>, GitError> {
        let mut inv = self
            .git("diff-tree")
            .args(["-r", "-z", "--raw", "--no-abbrev", "-M", "--end-of-options"])
            .args([from.spec(), to.spec()]);
        if !paths.is_empty() {
            inv = inv.arg("--").args(paths.iter().map(|p| p.as_str()));
        }
        let out = inv
            .read_only()
            .about_rev(format!("{}..{}", from.spec(), to.spec()))
            .run_ok()?;
        parse_raw(&out)
    }

    /// What the working tree changes relative to commit `from` — staged and
    /// unstaged together, untracked files excluded.
    ///
    /// Never writes the user's index. `git diff` would refresh stale stat
    /// information and write the refresh back to the index, whatever
    /// `GIT_OPTIONAL_LOCKS` says, so this runs `diff-index`, which writes
    /// nothing — and which, against an index nobody has refreshed, reports a
    /// file whose timestamps changed as modified with an unhashed new side.
    /// Those candidates are hashed here through the repository's filters,
    /// the same comparison a refresh makes, and dropped when their content
    /// matches `from`. A symlink or a submodule is never a candidate: neither
    /// is a file `hash-object` can read, and a submodule's unhashed side
    /// means its checkout moved or is dirty, which is a real change.
    pub fn diff_worktree(&self, from: &Rev) -> Result<Vec<DiffEntry>, GitError> {
        let out = self
            .git("diff-index")
            .args(["-z", "--raw", "--no-abbrev", "-M", "--end-of-options"])
            .arg(from.spec())
            .read_only()
            .about_rev(from.spec())
            .run_ok()?;
        let mut entries = parse_raw(&out)?;

        let unhashed: Vec<usize> = entries
            .iter()
            .enumerate()
            .filter(|(_, e)| {
                e.status == DiffStatus::Modified
                    && matches!(&e.new, Some(side) if side.oid.is_none()
                        && !matches!(side.mode, FileMode::Symlink | FileMode::Submodule))
            })
            .map(|(i, _)| i)
            .collect();
        if unhashed.is_empty() {
            return Ok(entries);
        }
        let mut input = String::new();
        for &i in &unhashed {
            let path = &entries[i].new.as_ref().expect("filtered on new").path;
            input.push_str(path.as_str());
            input.push('\n');
        }
        let hashed = self
            .git("hash-object")
            .arg("--stdin-paths")
            .stdin(input.into_bytes())
            .read_only()
            .run_ok()?;
        let hashed = utf8("hash-object", hashed)?;
        let oids: Vec<&str> = hashed.lines().collect();
        if oids.len() != unhashed.len() {
            return Err(GitError::malformed(
                "hash-object",
                format!("{} ids for {} paths", oids.len(), unhashed.len()),
            ));
        }
        let mut unchanged = Vec::new();
        for (&i, oid) in unhashed.iter().zip(oids) {
            let oid = Oid::parse(oid)?;
            let entry = &mut entries[i];
            let old = entry.old.as_ref().and_then(|s| s.oid.as_ref());
            let same_mode =
                entry.old.as_ref().map(|s| s.mode) == entry.new.as_ref().map(|s| s.mode);
            if same_mode && old == Some(&oid) {
                unchanged.push(i);
            } else if let Some(side) = entry.new.as_mut() {
                side.oid = Some(oid);
            }
        }
        let mut index = 0;
        entries.retain(|_| {
            let keep = !unchanged.contains(&index);
            index += 1;
            keep
        });
        Ok(entries)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testing::{git_in, TestRepo};

    const A: &str = "ce013625030ba8dba906f756967f9e9ca394464a";
    const B: &str = "7898192261b8b8d7ab18ee7faa5b2d26fd8b35cc";

    #[test]
    fn raw_records_parse_from_bytes() {
        let z = "0".repeat(40);
        let raw = format!(
            ":000000 100644 {z} {A} A\0added.rs\0\
             :100644 100755 {A} {B} M\0mode and content.sh\0\
             :100644 000000 {A} {z} D\0deleted.rs\0\
             :100644 100644 {A} {A} R095\0old name.rs\0new name.rs\0\
             :100644 100644 {A} {z} M\0unhashed.rs\0"
        );
        let e = parse_raw(raw.as_bytes()).unwrap();
        assert_eq!(e.len(), 5);
        assert_eq!(e[0].status, DiffStatus::Added);
        assert_eq!(e[0].old, None);
        assert_eq!(e[0].new.as_ref().unwrap().oid, Some(Oid::parse(A).unwrap()));
        assert_eq!(e[1].old.as_ref().unwrap().mode, FileMode::Regular);
        assert_eq!(e[1].new.as_ref().unwrap().mode, FileMode::Executable);
        assert_eq!(
            e[1].new.as_ref().unwrap().path.as_str(),
            "mode and content.sh"
        );
        assert_eq!(e[2].status, DiffStatus::Deleted);
        assert_eq!(e[2].new, None);
        assert_eq!(e[3].status, DiffStatus::Renamed(95));
        assert_eq!(e[3].old.as_ref().unwrap().path.as_str(), "old name.rs");
        assert_eq!(e[3].new.as_ref().unwrap().path.as_str(), "new name.rs");
        assert_eq!(e[4].new.as_ref().unwrap().oid, None);
    }

    #[test]
    fn a_header_without_a_colon_is_malformed() {
        assert!(parse_raw(b"100644 100644 x y M\0p\0").is_err());
    }

    #[test]
    fn diff_between_commits_finds_edits_adds_deletes_and_renames() {
        let t = TestRepo::init();
        let long: String = (0..40).map(|i| format!("line {i}\n")).collect();
        t.write("moved.rs", long.as_bytes());
        t.write("edit.rs", b"before\n");
        t.write("gone.rs", b"gone\n");
        let base = t.commit_all("base");
        t.git(&["mv", "moved.rs", "renamed.rs"]);
        t.write("edit.rs", b"after\n");
        std::fs::remove_file(t.path.join("gone.rs")).unwrap();
        t.write("new.rs", b"new\n");
        let next = t.commit_all("next");

        let mut got: Vec<(DiffStatus, String)> = t
            .repo()
            .diff(&Rev::Oid(base.clone()), &Rev::Oid(next.clone()), &[])
            .unwrap()
            .into_iter()
            .map(|e| {
                let path = e.new.as_ref().or(e.old.as_ref()).unwrap().path.to_string();
                (e.status, path)
            })
            .collect();
        got.sort_by(|a, b| a.1.cmp(&b.1));
        assert_eq!(
            got,
            vec![
                (DiffStatus::Modified, "edit.rs".into()),
                (DiffStatus::Deleted, "gone.rs".into()),
                (DiffStatus::Added, "new.rs".into()),
                (DiffStatus::Renamed(100), "renamed.rs".into()),
            ]
        );

        // A pathspec narrows the listing to what it names.
        let edit = RepoPath::parse("edit.rs").unwrap();
        let narrowed = t
            .repo()
            .diff(&Rev::Oid(base), &Rev::Oid(next), &[&edit])
            .unwrap();
        assert_eq!(narrowed.len(), 1, "{narrowed:?}");
        assert_eq!(narrowed[0].new.as_ref().unwrap().path, edit);
    }

    /// **A worktree diff reports only real changes.** An index nobody has
    /// refreshed — every file touched after it was written — must not make
    /// untouched files look modified.
    #[test]
    fn a_worktree_diff_ignores_stat_only_changes() {
        let t = TestRepo::init();
        t.write("same.txt", b"same\n");
        t.write("edit.txt", b"before\n");
        let base = t.commit_all("base");
        // Rewrite identical bytes: the stat data changes, the content does not.
        std::thread::sleep(std::time::Duration::from_millis(1100));
        t.write("same.txt", b"same\n");
        t.write("edit.txt", b"after\n");
        let index_before = std::fs::read(t.path.join(".git/index")).unwrap();

        let changes = t.repo().diff_worktree(&Rev::Oid(base)).unwrap();
        assert_eq!(changes.len(), 1, "{changes:?}");
        let new = changes[0].new.as_ref().unwrap();
        assert_eq!(new.path.as_str(), "edit.txt");
        // Hashed here, since the index never held the new content.
        let expected = t.git(&["hash-object", "edit.txt"]);
        assert_eq!(new.oid.as_ref().map(Oid::as_str), Some(expected.trim()));
        assert_eq!(
            std::fs::read(t.path.join(".git/index")).unwrap(),
            index_before,
            "the refresh must not be written back to the user's index"
        );
    }

    /// **A dirty submodule is a change, not a failure.** Its worktree side
    /// comes back unhashed, and `hash-object` cannot read a folder.
    #[test]
    fn a_dirty_submodule_is_reported_not_hashed() {
        let t = TestRepo::init();
        let sub = t.path.join("sub");
        std::fs::create_dir_all(&sub).unwrap();
        git_in(&sub, &["init", "-q"]);
        std::fs::write(sub.join("a.txt"), b"one\n").unwrap();
        git_in(&sub, &["add", "a.txt"]);
        git_in(&sub, &["commit", "-q", "-m", "one"]);
        t.write("f.txt", b"f\n");
        let base = t.commit_all("base");
        std::fs::write(sub.join("a.txt"), b"two\n").unwrap();

        let changes = t.repo().diff_worktree(&Rev::Oid(base)).unwrap();
        assert_eq!(changes.len(), 1, "{changes:?}");
        let new = changes[0].new.as_ref().unwrap();
        assert_eq!((new.path.as_str(), new.mode), ("sub", FileMode::Submodule));
    }
}
