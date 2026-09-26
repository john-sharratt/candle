//! Which commits last changed a span of a file's lines — `git_log`'s `lines`.
//!
//! "Who last changed these lines" was asked of `git_log`, measured live, with
//! the right tool (`git_show`'s `blame`) projected and ranked first: the model
//! reads it as a history question, and answered with the file's most recent
//! commit — which need not have touched those lines at all. So the history
//! tool answers it too, the way `git log -L` does: the lines are blamed, and
//! what comes back is the commits behind them, newest first, with the runs of
//! lines each one owns.

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use zend_git::{CommitInfo, GitError, LineRange, LogRange, Oid, Repo, RepoPath, Rev};

use super::GitToolError;

/// The most lines one call attributes — a `file_read` page.
pub const MAX_LINES: u32 = 200;
/// How far back the file's own history is walked to order its owners.
const MAX_FILE_WALK: usize = 2000;

/// A span of a file's lines, 1-based and inclusive.
#[derive(Debug, Clone, Copy, Deserialize, Serialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct LineSpan {
    /// The first line, counting from 1.
    pub start: u32,
    /// The last line, inclusive. Past the end of the file reads as the end.
    pub end: u32,
}

/// A run of consecutive lines last changed by the same commit.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct LineRun {
    pub from: u32,
    pub to: u32,
    pub commit: String,
}

/// The commits behind a span of lines.
#[derive(Debug)]
pub struct LineHistory {
    /// Each commit that last changed at least one of the lines, in the order
    /// the file's history reads — newest first.
    pub commits: Vec<CommitInfo>,
    /// Which commit owns which lines, in line order.
    pub runs: Vec<LineRun>,
}

/// Blame `span` of `path` at `rev` and gather the commits behind it.
///
/// A span that starts past the end of the file is refused with the file's
/// length, so the caller can ask again; one that runs past the end is
/// clamped to it, since the lines that do exist are the answer.
pub fn line_history(
    repo: &Repo,
    rev: &Rev,
    path: &RepoPath,
    span: LineSpan,
) -> Result<LineHistory, GitToolError> {
    if span.start == 0 || span.end < span.start {
        return Err(GitError::invalid(
            "`lines` counts from 1, and its `end` is at or after its `start`",
        )
        .into());
    }
    if span.end - span.start >= MAX_LINES {
        return Err(GitError::invalid(format!(
            "`lines` covers at most {MAX_LINES} lines in one call; narrow it"
        ))
        .into());
    }
    let commit = repo.resolve(rev)?;
    let missing = || -> GitToolError {
        GitError::UnknownRevision {
            rev: format!("{}:{}", commit.as_str(), path.as_str()),
        }
        .into()
    };
    let bytes = repo.blobs().read_at(rev, path)?.ok_or_else(missing)?;
    let total = line_count(&bytes);
    if span.start > total {
        return Err(GitError::invalid(format!(
            "{} has only {total} line(s) at that revision",
            path.as_str()
        ))
        .into());
    }
    let range = LineRange {
        start: span.start,
        end: span.end.min(total),
    };
    let blamed = repo.blame(rev, path, Some(range))?.ok_or_else(missing)?;

    let mut runs: Vec<LineRun> = Vec::new();
    for line in &blamed {
        let commit = line.commit.as_str();
        match runs.last_mut() {
            Some(run) if run.commit == commit && run.to + 1 == line.final_line => {
                run.to = line.final_line;
            }
            _ => runs.push(LineRun {
                from: line.final_line,
                to: line.final_line,
                commit: commit.to_string(),
            }),
        }
    }

    // In the order the file's own history reads, not by date: two commits can
    // carry the same timestamp (a rebase, a scripted import), and a date sort
    // then puts them in whatever order the tie-break falls. The walk covers
    // the file's commits only, so it is short however long the repository's
    // history is.
    let mut owners: Vec<Oid> = Vec::new();
    for line in &blamed {
        if !owners.contains(&line.commit) {
            owners.push(line.commit.clone());
        }
    }
    let walk = repo.log(
        &LogRange {
            to: rev.clone(),
            exclude: None,
            paths: vec![path.clone()],
        },
        MAX_FILE_WALK,
    )?;
    let mut commits: Vec<CommitInfo> = walk
        .into_iter()
        .filter(|c| owners.contains(&c.oid))
        .collect();
    // A line older than the walk reaches — or carried across a rename the
    // path filter does not follow — is still owned by its commit, so it is
    // looked up on its own rather than dropped.
    for oid in &owners {
        if !commits.iter().any(|c| &c.oid == oid) {
            commits.extend(repo.log(&LogRange::of(Rev::Oid(oid.clone())), 1)?);
        }
    }

    Ok(LineHistory { commits, runs })
}

/// Lines in a blob, counting a final line with no newline.
pub(super) fn line_count(bytes: &[u8]) -> u32 {
    let newlines = bytes.iter().filter(|&&b| b == b'\n').count();
    let unterminated = usize::from(!bytes.is_empty() && !bytes.ends_with(b"\n"));
    (newlines + unterminated) as u32
}

#[cfg(test)]
mod tests {
    use super::line_count;

    #[test]
    fn lines_are_counted_with_and_without_a_final_newline() {
        assert_eq!(line_count(b""), 0);
        assert_eq!(line_count(b"one\n"), 1);
        assert_eq!(line_count(b"one\ntwo"), 2);
        assert_eq!(line_count(b"one\ntwo\n"), 2);
        assert_eq!(line_count(b"\n\n"), 2, "empty lines are lines");
    }
}
