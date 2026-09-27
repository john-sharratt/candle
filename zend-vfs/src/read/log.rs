//! Commit history, from `log -z` with NUL-delimited fields.

use crate::error::GitError;
use crate::runner::Invocation;
use crate::types::{GitTime, Oid, RepoPath, Rev, Signature};
use crate::Repo;

/// Commits reachable from `to` and not from `exclude`, limited to those that
/// touch `paths` when any are given.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LogRange {
    pub to: Rev,
    pub exclude: Option<Rev>,
    pub paths: Vec<RepoPath>,
}

impl LogRange {
    /// Every commit reachable from `to`.
    pub fn of(to: Rev) -> Self {
        Self {
            to,
            exclude: None,
            paths: Vec::new(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CommitInfo {
    pub oid: Oid,
    pub parents: Vec<Oid>,
    pub author: Signature,
    pub committer: Signature,
    /// The full message, exactly as stored.
    pub message: String,
}

impl CommitInfo {
    /// The message's first line.
    pub fn subject(&self) -> &str {
        self.message.lines().next().unwrap_or("")
    }
}

/// Fields per commit, in [`FORMAT`]'s order.
const FIELDS: usize = 9;
const FORMAT: &str = "--format=%H%x00%P%x00%an%x00%ae%x00%ad%x00%cn%x00%ce%x00%cd%x00%B";

pub(crate) fn parse_log(out: &[u8]) -> Result<Vec<CommitInfo>, GitError> {
    let text = std::str::from_utf8(out).map_err(|e| GitError::malformed("log", e.to_string()))?;
    if text.is_empty() {
        return Ok(Vec::new());
    }
    // `-z` puts one NUL between commits, and the format one between fields,
    // so the stream is a flat run of fields, `FIELDS` per commit.
    let text = text.strip_suffix('\0').unwrap_or(text);
    let fields: Vec<&str> = text.split('\0').collect();
    if !fields.len().is_multiple_of(FIELDS) {
        return Err(GitError::malformed(
            "log",
            format!("{} fields is not a whole number of commits", fields.len()),
        ));
    }
    fields
        .chunks(FIELDS)
        .map(|f| {
            let parents = f[1]
                .split(' ')
                .filter(|p| !p.is_empty())
                .map(Oid::parse)
                .collect::<Result<_, _>>()?;
            Ok(CommitInfo {
                oid: Oid::parse(f[0])?,
                parents,
                author: Signature::recorded(f[2], f[3], GitTime::parse_raw(f[4])?),
                committer: Signature::recorded(f[5], f[6], GitTime::parse_raw(f[7])?),
                message: f[8].to_string(),
            })
        })
        .collect()
}

impl Repo {
    fn log_invocation(&self, limit: usize) -> Invocation {
        self.git("log")
            .args([
                "-z",
                FORMAT,
                "--date=raw",
                "--no-color",
                "--no-show-signature",
            ])
            .arg(format!("--max-count={limit}"))
    }

    /// Up to `limit` commits in `range`, newest first.
    pub fn log(&self, range: &LogRange, limit: usize) -> Result<Vec<CommitInfo>, GitError> {
        let mut inv = self
            .log_invocation(limit)
            .arg("--end-of-options")
            .arg(range.to.spec());
        if let Some(exclude) = &range.exclude {
            inv = inv.arg(format!("^{}", exclude.spec()));
        }
        if !range.paths.is_empty() {
            inv = inv.arg("--").args(range.paths.iter().map(|p| p.as_str()));
        }
        let out = inv.read_only().about_rev(range.to.spec()).run_ok()?;
        parse_log(&out)
    }

    /// Up to `limit` commits that changed `path`, newest first, following
    /// the file back through renames.
    pub fn file_history(
        &self,
        rev: &Rev,
        path: &RepoPath,
        limit: usize,
    ) -> Result<Vec<CommitInfo>, GitError> {
        let out = self
            .log_invocation(limit)
            .arg("--follow")
            .arg("--end-of-options")
            .arg(rev.spec())
            .arg("--")
            .arg(path.as_str())
            .read_only()
            .about_rev(rev.spec())
            .run_ok()?;
        parse_log(&out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testing::{TestRepo, SETUP_DATE};

    #[test]
    fn log_fields_parse_from_raw_bytes() {
        let a = "ce013625030ba8dba906f756967f9e9ca394464a";
        let b = "4b825dc642cb6eb9a060e54bf8d69288fbee4904";
        let raw = format!(
            "{a}\0{b}\0Ada\0ada@x\01700000000 +0100\0Bob\0bob@x\01700000060 -0500\0subject\n\nbody line\n\0\
             {b}\0\0Ada\0ada@x\01600000000 +0000\0Ada\0ada@x\01600000000 +0000\0root\n"
        );
        let log = parse_log(raw.as_bytes()).unwrap();
        assert_eq!(log.len(), 2);
        assert_eq!(log[0].oid.as_str(), a);
        assert_eq!(log[0].parents, vec![Oid::parse(b).unwrap()]);
        assert_eq!(log[0].author.name(), "Ada");
        assert_eq!(log[0].author.when.offset_minutes, 60);
        assert_eq!(log[0].committer.email(), "bob@x");
        assert_eq!(log[0].committer.when.offset_minutes, -300);
        assert_eq!(log[0].message, "subject\n\nbody line\n");
        assert_eq!(log[0].subject(), "subject");
        assert!(log[1].parents.is_empty());
    }

    /// **An imported commit with no email, or no name, still reads.** SVN
    /// and CVS conversions record `Name <>`; refusing it would fail every
    /// history read that crosses one.
    #[test]
    fn a_commit_with_an_empty_identity_still_parses() {
        let a = "ce013625030ba8dba906f756967f9e9ca394464a";
        let raw = format!("{a}\0\0svnuser\0\01700000000 +0000\0\0\01700000000 +0000\0import\n");
        let log = parse_log(raw.as_bytes()).unwrap();
        assert_eq!(log[0].author.name(), "svnuser");
        assert_eq!(log[0].author.email(), "");
        assert_eq!(log[0].committer.name(), "");
    }

    #[test]
    fn a_truncated_stream_is_malformed() {
        assert!(parse_log(b"abc\0def\0").is_err());
    }

    #[test]
    fn log_walks_a_range_newest_first() {
        let t = TestRepo::init();
        t.write("a", b"1\n");
        let first = t.commit_all("first");
        t.write("a", b"2\n");
        let second = t.commit_all("second\n\nwith a body");
        t.write("a", b"3\n");
        let third = t.commit_all("third");
        let repo = t.repo();

        let all = repo.log(&LogRange::of(Rev::Head), 10).unwrap();
        let oids: Vec<&Oid> = all.iter().map(|c| &c.oid).collect();
        assert_eq!(oids, vec![&third, &second, &first]);
        assert_eq!(all[1].message, "second\n\nwith a body\n");
        assert_eq!(all[1].parents, vec![first.clone()]);
        assert_eq!(all[1].author.name(), "Setup");
        assert_eq!(all[1].author.when, GitTime::parse_raw(SETUP_DATE).unwrap());

        let since_first = repo
            .log(
                &LogRange {
                    to: Rev::Head,
                    exclude: Some(Rev::Oid(first)),
                    paths: Vec::new(),
                },
                10,
            )
            .unwrap();
        assert_eq!(since_first.len(), 2);

        let limited = repo.log(&LogRange::of(Rev::Head), 1).unwrap();
        assert_eq!(limited.len(), 1);
    }

    /// A path filter keeps only the commits that touched those paths.
    #[test]
    fn log_limited_to_a_path() {
        let t = TestRepo::init();
        t.write("a.txt", b"1\n");
        let first = t.commit_all("a");
        t.write("b.txt", b"1\n");
        t.commit_all("b");
        t.write("a.txt", b"2\n");
        let third = t.commit_all("a again");
        let range = LogRange {
            paths: vec![RepoPath::parse("a.txt").unwrap()],
            ..LogRange::of(Rev::Head)
        };
        let oids: Vec<Oid> = t
            .repo()
            .log(&range, 10)
            .unwrap()
            .into_iter()
            .map(|c| c.oid)
            .collect();
        assert_eq!(oids, vec![third, first]);
    }

    /// A file's history follows it back through a rename; a plain path
    /// filter stops at the rename.
    #[test]
    fn file_history_follows_renames() {
        let t = TestRepo::init();
        let body: String = (0..20).map(|i| format!("line {i}\n")).collect();
        t.write("old.rs", body.as_bytes());
        let created = t.commit_all("create");
        t.git(&["mv", "old.rs", "new.rs"]);
        let renamed = t.commit_all("rename");
        t.write("new.rs", format!("{body}more\n").as_bytes());
        let edited = t.commit_all("edit");
        t.write("unrelated", b"x\n");
        t.commit_all("unrelated");
        let repo = t.repo();
        let path = RepoPath::parse("new.rs").unwrap();

        let followed: Vec<Oid> = repo
            .file_history(&Rev::Head, &path, 10)
            .unwrap()
            .into_iter()
            .map(|c| c.oid)
            .collect();
        assert_eq!(followed, vec![edited.clone(), renamed.clone(), created]);

        let plain = LogRange {
            paths: vec![path],
            ..LogRange::of(Rev::Head)
        };
        let unfollowed: Vec<Oid> = repo
            .log(&plain, 10)
            .unwrap()
            .into_iter()
            .map(|c| c.oid)
            .collect();
        assert_eq!(unfollowed, vec![edited, renamed]);
    }
}
