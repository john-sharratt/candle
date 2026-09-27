//! The shapes the `git_*` tools answer in.
//!
//! Three rules hold throughout:
//!
//! - **Object ids are always full**, never abbreviated, because the id a tool
//!   prints is the id a later call hands back.
//! - **Times are ISO 8601 with the recorded offset** — the model reasons about
//!   "last Tuesday" far better than about 1700000000.
//! - **Binary is reported, not mangled.** A blob that is not valid UTF-8 comes
//!   back flagged with its byte count rather than as replacement characters
//!   the model would reason about as if they were text.

use serde::Serialize;

use crate::tools::file::grep::truncate;
use zend_vfs::{
    BlameLine, Branch, CommitInfo, DiffEntry, DiffStatus, FileMode, GitTime, GrepHit, ObjectKind,
    Remote, Signature, StatusCode, StatusEntry, Tag, TreeEntry, Upstream, Xy,
};

#[derive(Serialize)]
pub struct WireSignature {
    pub name: String,
    pub email: String,
    /// ISO 8601 with the offset git recorded.
    pub date: String,
}

impl From<&Signature> for WireSignature {
    fn from(sig: &Signature) -> Self {
        Self {
            name: sig.name().to_string(),
            email: sig.email().to_string(),
            date: iso8601(sig.when),
        }
    }
}

/// Git's raw time as ISO 8601 in its own offset. An offset git could not
/// represent falls back to the raw form rather than inventing one.
pub fn iso8601(when: GitTime) -> String {
    use chrono::{DateTime, FixedOffset};
    let offset = FixedOffset::east_opt(when.offset_minutes * 60);
    match offset
        .and_then(|o| DateTime::from_timestamp(when.seconds, 0).map(|t| t.with_timezone(&o)))
    {
        Some(t) => t.to_rfc3339(),
        None => when.to_raw(),
    }
}

/// One commit in a history listing.
///
/// **The subject, never the body.** A listing is for choosing a commit, and a
/// body is unbounded: in a repository whose messages run to several paragraphs,
/// twenty-five full commits measured 11,761 tokens, all of it prefilled before
/// the model could read a count it only needed one number from. The body is
/// `git_show`'s to give, for the one commit worth reading.
///
/// **Parents only for a merge.** An ordinary commit's parent is the next entry
/// down the listing, and a revision one back is `{"kind":"parent"}` rather
/// than a copied id, so a single parent id was a fifth of every entry and
/// told the reader nothing the order did not. A merge's parents are the one
/// case they say something, so a merge — and only a merge — lists them.
#[derive(Serialize)]
pub struct WireCommit {
    /// The full object id — what a later call hands back.
    pub id: String,
    /// Present only on a merge: every parent, first-parent first.
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub merge_of: Vec<String>,
    pub author: WireSignature,
    /// The message's first line.
    pub subject: String,
}

impl From<&CommitInfo> for WireCommit {
    fn from(c: &CommitInfo) -> Self {
        let merge_of = if c.parents.len() > 1 {
            c.parents.iter().map(|p| p.as_str().to_string()).collect()
        } else {
            Vec::new()
        };
        Self {
            id: c.oid.as_str().to_string(),
            merge_of,
            author: (&c.author).into(),
            subject: c.subject().to_string(),
        }
    }
}

pub fn diff_status(status: DiffStatus) -> &'static str {
    match status {
        DiffStatus::Added => "added",
        DiffStatus::Modified => "modified",
        DiffStatus::Deleted => "deleted",
        DiffStatus::TypeChanged => "type_changed",
        DiffStatus::Unmerged => "unmerged",
        DiffStatus::Renamed(_) => "renamed",
        DiffStatus::Copied(_) => "copied",
    }
}

#[derive(Serialize)]
pub struct WireChange {
    pub status: &'static str,
    /// Where the file is after the change — its old path for a deletion.
    pub path: String,
    /// Where it was before; present only for a rename or a copy.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub from_path: Option<String>,
}

impl From<&DiffEntry> for WireChange {
    fn from(e: &DiffEntry) -> Self {
        let path = e
            .new
            .as_ref()
            .or(e.old.as_ref())
            .map(|s| s.path.as_str().to_string())
            .unwrap_or_default();
        let from_path = match e.status {
            DiffStatus::Renamed(_) | DiffStatus::Copied(_) => {
                e.old.as_ref().map(|s| s.path.as_str().to_string())
            }
            _ => None,
        };
        Self {
            status: diff_status(e.status),
            path,
            from_path,
        }
    }
}

fn status_code(c: StatusCode) -> &'static str {
    match c {
        StatusCode::Unmodified => "unmodified",
        StatusCode::Modified => "modified",
        StatusCode::TypeChanged => "type_changed",
        StatusCode::Added => "added",
        StatusCode::Deleted => "deleted",
        StatusCode::Renamed => "renamed",
        StatusCode::Copied => "copied",
        StatusCode::Unmerged => "unmerged",
    }
}

/// One path's working-tree state.
///
/// `staged` and `unstaged` are git's two sides — what differs between the last
/// commit and the index, and between the index and the file on disk. A path
/// can be both at once. **A side with nothing on it is left out** rather than
/// spelled `"unmodified"`: nearly every path in a working tree changes on one
/// side only, so the word was a third of every entry and said nothing.
#[derive(Serialize)]
pub struct WireStatus {
    pub path: String,
    /// `untracked`, `unmerged`, `renamed` or `changed`.
    pub state: &'static str,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub staged: Option<&'static str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub unstaged: Option<&'static str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub from_path: Option<String>,
}

impl From<&StatusEntry> for WireStatus {
    fn from(e: &StatusEntry) -> Self {
        let path = e.path().as_str().to_string();
        let side =
            |code: StatusCode| (!matches!(code, StatusCode::Unmodified)).then(|| status_code(code));
        let two = |xy: &Xy| (side(xy.index), side(xy.worktree));
        match e {
            StatusEntry::Untracked { .. } => Self {
                path,
                state: "untracked",
                staged: None,
                unstaged: None,
                from_path: None,
            },
            StatusEntry::Changed { xy, .. } => {
                let (staged, unstaged) = two(xy);
                Self {
                    path,
                    state: "changed",
                    staged,
                    unstaged,
                    from_path: None,
                }
            }
            StatusEntry::Renamed { xy, from, .. } => {
                let (staged, unstaged) = two(xy);
                Self {
                    path,
                    state: "renamed",
                    staged,
                    unstaged,
                    from_path: Some(from.as_str().to_string()),
                }
            }
            StatusEntry::Unmerged { xy, .. } => {
                let (staged, unstaged) = two(xy);
                Self {
                    path,
                    state: "unmerged",
                    staged,
                    unstaged,
                    from_path: None,
                }
            }
        }
    }
}

#[derive(Serialize)]
pub struct WireUpstream {
    /// Absent when the branch tracks another local branch.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub remote: Option<String>,
    pub branch: String,
    /// Commits this branch has that its upstream does not.
    pub ahead: u32,
    /// Commits the upstream has that this branch does not.
    pub behind: u32,
    /// The remote-tracking branch no longer exists.
    pub gone: bool,
}

impl From<&Upstream> for WireUpstream {
    fn from(u: &Upstream) -> Self {
        Self {
            remote: u.remote.as_ref().map(|r| r.as_str().to_string()),
            branch: u.branch.as_str().to_string(),
            ahead: u.ahead,
            behind: u.behind,
            gone: u.gone,
        }
    }
}

#[derive(Serialize)]
pub struct WireBranch {
    pub name: String,
    pub id: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub upstream: Option<WireUpstream>,
    /// Whether this is the branch the developer has checked out.
    pub head: bool,
}

impl WireBranch {
    pub fn new(b: &Branch, head: bool) -> Self {
        Self {
            name: b.name.as_str().to_string(),
            id: b.oid.as_str().to_string(),
            upstream: b.upstream.as_ref().map(Into::into),
            head,
        }
    }
}

#[derive(Serialize)]
pub struct WireTag {
    pub name: String,
    /// What the tag finally points at — the commit, for an ordinary tag.
    pub target: String,
    /// The tag object's own id, which differs from `target` when annotated.
    pub id: String,
    pub annotated: bool,
}

impl From<&Tag> for WireTag {
    fn from(t: &Tag) -> Self {
        Self {
            name: t.name.as_str().to_string(),
            target: t.target.as_str().to_string(),
            id: t.oid.as_str().to_string(),
            annotated: t.annotated,
        }
    }
}

/// A configured remote. Any credential embedded in a URL was redacted by the
/// git layer before it reached here.
#[derive(Serialize)]
pub struct WireRemote {
    pub name: String,
    pub fetch_url: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub push_url: Option<String>,
}

impl From<&Remote> for WireRemote {
    fn from(r: &Remote) -> Self {
        Self {
            name: r.name.as_str().to_string(),
            fetch_url: r.fetch_url.clone(),
            push_url: r.push_url.clone(),
        }
    }
}

#[derive(Serialize)]
pub struct WireGrepHit {
    pub path: String,
    /// 1-based.
    pub line: u32,
    /// The matching line, clipped past file_grep's
    /// [`MAX_LINE_CHARS`](crate::tools::file::grep::MAX_LINE_CHARS) characters
    /// with a marker saying so — one hit in a minified file is otherwise a whole
    /// bundle, larger than every other match on the page together.
    pub text: String,
}

impl From<&GrepHit> for WireGrepHit {
    fn from(h: &GrepHit) -> Self {
        Self {
            path: h.path.as_str().to_string(),
            line: h.line,
            text: truncate(&String::from_utf8_lossy(&h.text)),
        }
    }
}

#[derive(Serialize)]
pub struct WireBlameLine {
    /// 1-based, in the revision blamed.
    pub line: u32,
    pub text: String,
    /// The commit that last changed this line.
    pub commit: String,
    pub author: String,
    pub date: String,
    pub summary: String,
    /// The file's path in that commit, present only when it differs — the
    /// line came across a rename.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub from_path: Option<String>,
}

impl WireBlameLine {
    pub fn new(l: &BlameLine, path: &str) -> Self {
        Self {
            line: l.final_line,
            text: String::from_utf8_lossy(&l.content).into_owned(),
            commit: l.commit.as_str().to_string(),
            author: l.author.name().to_string(),
            date: iso8601(l.author.when),
            summary: l.summary.clone(),
            from_path: (l.orig_path.as_str() != path).then(|| l.orig_path.as_str().to_string()),
        }
    }
}

fn object_kind(k: ObjectKind) -> &'static str {
    match k {
        ObjectKind::Blob => "file",
        ObjectKind::Tree => "directory",
        ObjectKind::Commit => "submodule",
    }
}

#[derive(Serialize)]
pub struct WireTreeEntry {
    pub path: String,
    pub kind: &'static str,
    pub mode: String,
    pub id: String,
}

impl From<&TreeEntry> for WireTreeEntry {
    fn from(e: &TreeEntry) -> Self {
        Self {
            path: e.path.as_str().to_string(),
            kind: object_kind(e.kind),
            mode: mode(e.mode),
            id: e.oid.as_str().to_string(),
        }
    }
}

pub fn mode(m: FileMode) -> String {
    m.as_str().to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_time_renders_in_the_offset_git_recorded() {
        assert_eq!(
            iso8601(GitTime {
                seconds: 1_700_000_000,
                offset_minutes: 0
            }),
            "2023-11-14T22:13:20+00:00"
        );
        // +05:30 is the case a whole-hour-only conversion gets wrong.
        assert_eq!(
            iso8601(GitTime {
                seconds: 1_700_000_000,
                offset_minutes: 330
            }),
            "2023-11-15T03:43:20+05:30"
        );
    }

    /// An offset outside what a fixed offset can hold falls back to git's raw
    /// form rather than inventing a time.
    #[test]
    fn an_impossible_offset_falls_back_to_the_raw_form() {
        let absurd = GitTime {
            seconds: 1_700_000_000,
            offset_minutes: 10_000,
        };
        assert_eq!(iso8601(absurd), absurd.to_raw());
    }
}
