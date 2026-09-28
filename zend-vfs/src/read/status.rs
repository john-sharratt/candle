//! Working-tree status, from `status --porcelain=v2 -z`.

use crate::error::GitError;
use crate::read::head::Head;
use crate::types::{BranchName, FileMode, Oid, RepoPath};
use crate::Repo;

/// Porcelain v2's `# key value` header lines.
type Headers = Vec<(String, String)>;

/// One side of a status code: what changed between `HEAD` and the index, or
/// between the index and the working tree.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StatusCode {
    Unmodified,
    Modified,
    TypeChanged,
    Added,
    Deleted,
    Renamed,
    Copied,
    Unmerged,
}

impl StatusCode {
    fn parse(c: u8) -> Result<Self, GitError> {
        Ok(match c {
            b'.' => Self::Unmodified,
            b'M' => Self::Modified,
            b'T' => Self::TypeChanged,
            b'A' => Self::Added,
            b'D' => Self::Deleted,
            b'R' => Self::Renamed,
            b'C' => Self::Copied,
            b'U' => Self::Unmerged,
            other => {
                return Err(GitError::malformed(
                    "status",
                    format!("status code {:?}", other as char),
                ))
            }
        })
    }
}

/// The index-side and working-tree-side codes of one entry.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Xy {
    pub index: StatusCode,
    pub worktree: StatusCode,
}

impl Xy {
    fn parse(xy: &str) -> Result<Self, GitError> {
        let b = xy.as_bytes();
        if b.len() != 2 {
            return Err(GitError::malformed("status", format!("XY {xy:?}")));
        }
        Ok(Self {
            index: StatusCode::parse(b[0])?,
            worktree: StatusCode::parse(b[1])?,
        })
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum StatusEntry {
    /// A tracked path changed in the index, the working tree, or both.
    Changed {
        xy: Xy,
        path: RepoPath,
        head_mode: Option<FileMode>,
        worktree_mode: Option<FileMode>,
        head_oid: Option<Oid>,
        index_oid: Option<Oid>,
    },
    /// A tracked path renamed or copied from `from`, with git's similarity
    /// score (0–100).
    Renamed {
        xy: Xy,
        path: RepoPath,
        from: RepoPath,
        score: u8,
    },
    /// A path with an unresolved merge conflict.
    Unmerged { xy: Xy, path: RepoPath },
    /// A path git does not track.
    Untracked { path: RepoPath },
}

impl StatusEntry {
    pub fn path(&self) -> &RepoPath {
        match self {
            Self::Changed { path, .. }
            | Self::Renamed { path, .. }
            | Self::Unmerged { path, .. }
            | Self::Untracked { path } => path,
        }
    }
}

fn fields(record: &str, n: usize) -> Result<Vec<&str>, GitError> {
    let parts: Vec<&str> = record.splitn(n, ' ').collect();
    if parts.len() != n {
        return Err(GitError::malformed("status", record.to_string()));
    }
    Ok(parts)
}

/// Parse `status --porcelain=v2 -z` output.
pub(crate) fn parse_status(out: &[u8]) -> Result<Vec<StatusEntry>, GitError> {
    parse_status_with_headers(out).map(|(_, entries)| entries)
}

/// Parse `status --porcelain=v2 --branch -z` output: `HEAD`, from the
/// `# branch.oid` and `# branch.head` headers, and the entries.
pub(crate) fn parse_status_with_head(out: &[u8]) -> Result<(Head, Vec<StatusEntry>), GitError> {
    let (headers, entries) = parse_status_with_headers(out)?;
    let header = |key: &str| {
        headers
            .iter()
            .find_map(|(k, v)| (k == key).then_some(v.as_str()))
            .ok_or_else(|| GitError::malformed("status", format!("no {key} header")))
    };
    let (oid, name) = (header("branch.oid")?, header("branch.head")?);
    let head = match (oid, name) {
        ("(initial)", name) => Head::Unborn(BranchName::parse(name)?),
        (oid, "(detached)") => Head::Detached(Oid::parse(oid)?),
        (oid, name) => Head::Branch {
            branch: BranchName::parse(name)?,
            oid: Oid::parse(oid)?,
        },
    };
    Ok((head, entries))
}

/// The `# key value` headers and the entries of porcelain v2 output.
fn parse_status_with_headers(out: &[u8]) -> Result<(Headers, Vec<StatusEntry>), GitError> {
    let text =
        std::str::from_utf8(out).map_err(|e| GitError::malformed("status", e.to_string()))?;
    let mut records = text.split('\0').filter(|r| !r.is_empty());
    let mut headers = Vec::new();
    let mut entries = Vec::new();
    while let Some(record) = records.next() {
        let kind = record.as_bytes()[0];
        match kind {
            b'#' => {
                let (key, value) = record[1..]
                    .trim_start()
                    .split_once(' ')
                    .ok_or_else(|| GitError::malformed("status", record.to_string()))?;
                headers.push((key.to_string(), value.to_string()));
            }
            b'1' => {
                // 1 XY sub mH mI mW hH hI path
                let f = fields(record, 9)?;
                entries.push(StatusEntry::Changed {
                    xy: Xy::parse(f[1])?,
                    head_mode: FileMode::parse_present(f[3])?,
                    worktree_mode: FileMode::parse_present(f[5])?,
                    head_oid: Oid::parse_nonzero(f[6])?,
                    index_oid: Oid::parse_nonzero(f[7])?,
                    path: RepoPath::parse(f[8])?,
                });
            }
            b'2' => {
                // 2 XY sub mH mI mW hH hI Xscore path NUL origPath
                let f = fields(record, 10)?;
                let score = f[8]
                    .get(1..)
                    .and_then(|s| s.parse::<u8>().ok())
                    .ok_or_else(|| GitError::malformed("status", record.to_string()))?;
                let from = records
                    .next()
                    .ok_or_else(|| GitError::malformed("status", "rename without its source"))?;
                entries.push(StatusEntry::Renamed {
                    xy: Xy::parse(f[1])?,
                    path: RepoPath::parse(f[9])?,
                    from: RepoPath::parse(from)?,
                    score,
                });
            }
            b'u' => {
                // u XY sub m1 m2 m3 mW h1 h2 h3 path
                let f = fields(record, 11)?;
                entries.push(StatusEntry::Unmerged {
                    xy: Xy::parse(f[1])?,
                    path: RepoPath::parse(f[10])?,
                });
            }
            b'?' => {
                // An untracked repository nested in the tree is listed as a
                // folder, `sub/`, even under `--untracked-files=all`.
                let f = fields(record, 2)?;
                entries.push(StatusEntry::Untracked {
                    path: RepoPath::parse(f[1].strip_suffix('/').unwrap_or(f[1]))?,
                });
            }
            // Ignored entries (`!`) are not requested.
            _ => return Err(GitError::malformed("status", record.to_string())),
        }
    }
    Ok((headers, entries))
}

impl Repo {
    /// The working tree's status against `HEAD`, untracked files included.
    ///
    /// Takes no optional locks: it never refreshes or rewrites the user's
    /// index, and it succeeds while another process holds `index.lock`.
    pub fn status(&self) -> Result<Vec<StatusEntry>, GitError> {
        let out = self
            .git("status")
            .args([
                "--porcelain=v2",
                "-z",
                "--untracked-files=all",
                "--ignored=no",
            ])
            .read_only()
            .run_ok()?;
        parse_status(&out)
    }

    /// [`Self::status`] and [`Self::head`] from one process — what a checkout
    /// pass asks for together. `--no-ahead-behind` spares the comparison with
    /// the upstream that `--branch` would otherwise make.
    pub fn status_with_head(&self) -> Result<(Head, Vec<StatusEntry>), GitError> {
        let out = self
            .git("status")
            .args([
                "--porcelain=v2",
                "--branch",
                "--no-ahead-behind",
                "-z",
                "--untracked-files=all",
                "--ignored=no",
            ])
            .read_only()
            .run_ok()?;
        parse_status_with_head(&out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testing::{git_in, TestRepo};

    const A: &str = "ce013625030ba8dba906f756967f9e9ca394464a";
    const B: &str = "7898192261b8b8d7ab18ee7faa5b2d26fd8b35cc";

    /// **`HEAD` is read from the branch headers** — on a branch, detached,
    /// and on a branch with no commits — with the entries after them intact.
    #[test]
    fn head_is_read_from_the_branch_headers() {
        let branch = |n: &str| BranchName::parse(n).unwrap();
        let raw = format!("# branch.oid {A}\0# branch.head main\0? new.txt\0");
        let (head, entries) = parse_status_with_head(raw.as_bytes()).unwrap();
        assert_eq!(
            head,
            Head::Branch {
                branch: branch("main"),
                oid: Oid::parse(A).unwrap()
            }
        );
        assert_eq!(entries.len(), 1);
        let raw = format!("# branch.oid {A}\0# branch.head (detached)\0");
        assert_eq!(
            parse_status_with_head(raw.as_bytes()).unwrap().0,
            Head::Detached(Oid::parse(A).unwrap())
        );
        let raw = "# branch.oid (initial)\0# branch.head main\0";
        assert_eq!(
            parse_status_with_head(raw.as_bytes()).unwrap().0,
            Head::Unborn(branch("main"))
        );
        assert!(parse_status_with_head(b"? x\0").is_err(), "no headers");
        // Plain status passes headers over.
        assert_eq!(parse_status(raw.as_bytes()).unwrap(), vec![]);
    }

    /// **One process gives what `head` and `status` give apart**, in every
    /// state `HEAD` can be in.
    #[test]
    fn status_with_head_agrees_with_head_and_status() {
        let t = TestRepo::init();
        let repo = t.repo();
        let unborn = repo.status_with_head().unwrap();
        assert_eq!(unborn.0, repo.head().unwrap());
        t.write("a.txt", b"a\n");
        t.commit_all("first");
        t.write("b.txt", b"b\n");
        let on_branch = repo.status_with_head().unwrap();
        assert_eq!(on_branch, (repo.head().unwrap(), repo.status().unwrap()));
        t.git(&["checkout", "-q", "--detach"]);
        let detached = repo.status_with_head().unwrap();
        assert_eq!(detached, (repo.head().unwrap(), repo.status().unwrap()));
        assert!(matches!(detached.0, Head::Detached(_)));
    }

    #[test]
    fn every_record_kind_parses_from_raw_bytes() {
        let raw = format!(
            "1 .M N... 100644 100644 100644 {A} {A} src/lib.rs\0\
             1 A. N... 000000 100644 100644 {z} {B} new file.txt\0\
             2 R. N... 100644 100644 100644 {A} {A} R100 moved.rs\0old.rs\0\
             u UU N... 100644 100644 100644 100644 {A} {B} {A} conflict.rs\0\
             ? ünïcode.txt\0",
            z = "0".repeat(40)
        );
        let entries = parse_status(raw.as_bytes()).unwrap();
        let p = |s| RepoPath::parse(s).unwrap();
        assert_eq!(
            entries,
            vec![
                StatusEntry::Changed {
                    xy: Xy {
                        index: StatusCode::Unmodified,
                        worktree: StatusCode::Modified
                    },
                    path: p("src/lib.rs"),
                    head_mode: Some(FileMode::Regular),
                    worktree_mode: Some(FileMode::Regular),
                    head_oid: Some(Oid::parse(A).unwrap()),
                    index_oid: Some(Oid::parse(A).unwrap()),
                },
                StatusEntry::Changed {
                    xy: Xy {
                        index: StatusCode::Added,
                        worktree: StatusCode::Unmodified
                    },
                    path: p("new file.txt"),
                    head_mode: None,
                    worktree_mode: Some(FileMode::Regular),
                    head_oid: None,
                    index_oid: Some(Oid::parse(B).unwrap()),
                },
                StatusEntry::Renamed {
                    xy: Xy {
                        index: StatusCode::Renamed,
                        worktree: StatusCode::Unmodified
                    },
                    path: p("moved.rs"),
                    from: p("old.rs"),
                    score: 100,
                },
                StatusEntry::Unmerged {
                    xy: Xy {
                        index: StatusCode::Unmerged,
                        worktree: StatusCode::Unmerged
                    },
                    path: p("conflict.rs"),
                },
                StatusEntry::Untracked {
                    path: p("ünïcode.txt")
                },
            ]
        );
    }

    #[test]
    fn an_unknown_record_is_malformed() {
        assert!(parse_status(b"! ignored.txt\0").is_err());
        assert!(parse_status(b"x something\0").is_err());
        assert!(parse_status(b"#nospace\0").is_err());
        assert!(parse_status(b"1 .M\0").is_err());
    }

    #[test]
    fn status_reports_modified_added_deleted_and_untracked() {
        let t = TestRepo::init();
        t.write("keep.txt", b"keep\n");
        t.write("edit.txt", b"before\n");
        t.write("gone.txt", b"gone\n");
        t.commit_all("base");
        t.write("edit.txt", b"after\n");
        std::fs::remove_file(t.path.join("gone.txt")).unwrap();
        t.write("staged.txt", b"staged\n");
        t.git(&["add", "staged.txt"]);
        t.write("dir with space/untracked é.txt", b"u\n");

        let mut got: Vec<(String, String)> = t
            .repo()
            .status()
            .unwrap()
            .into_iter()
            .map(|e| {
                let kind = match &e {
                    StatusEntry::Changed { xy, .. } => format!("{:?}/{:?}", xy.index, xy.worktree),
                    StatusEntry::Untracked { .. } => "untracked".into(),
                    other => format!("{other:?}"),
                };
                (e.path().to_string(), kind)
            })
            .collect();
        got.sort();
        assert_eq!(
            got,
            vec![
                ("dir with space/untracked é.txt".into(), "untracked".into()),
                ("edit.txt".into(), "Unmodified/Modified".into()),
                ("gone.txt".into(), "Unmodified/Deleted".into()),
                ("staged.txt".into(), "Added/Unmodified".into()),
            ]
        );
    }

    /// **Status neither needs nor takes the index lock.** A user's `git add`
    /// in progress holds `index.lock`; zend's status must not fail on it, and
    /// must leave it exactly as it found it.
    #[test]
    fn status_succeeds_while_the_index_is_locked_and_leaves_the_lock() {
        let t = TestRepo::init();
        t.write("a.txt", b"a\n");
        t.commit_all("base");
        t.write("a.txt", b"changed\n");
        let lock = t.path.join(".git/index.lock");
        std::fs::write(&lock, b"held by someone else").unwrap();
        let index_before = std::fs::read(t.path.join(".git/index")).unwrap();

        let entries = t.repo().status().unwrap();
        assert_eq!(entries.len(), 1);
        assert_eq!(std::fs::read(&lock).unwrap(), b"held by someone else");
        assert_eq!(
            std::fs::read(t.path.join(".git/index")).unwrap(),
            index_before
        );
    }

    /// **An untracked repository inside the tree is listed, not an error.**
    /// Git names it as a folder with a trailing `/`.
    #[test]
    fn an_untracked_nested_repository_is_listed_by_its_folder() {
        let t = TestRepo::init();
        t.write("a.txt", b"a\n");
        t.commit_all("base");
        let nested = t.path.join("vendor/dep");
        std::fs::create_dir_all(&nested).unwrap();
        git_in(&nested, &["init", "-q"]);
        std::fs::write(nested.join("x.txt"), b"x\n").unwrap();

        let entries = t.repo().status().unwrap();
        assert_eq!(
            entries,
            vec![StatusEntry::Untracked {
                path: RepoPath::parse("vendor/dep").unwrap()
            }]
        );
    }
}
