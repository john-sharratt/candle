//! git_show tool — repository content at, or between, revisions.
//!
//! Five modes behind one tool. Each was a separate tool until the choice
//! between them was measured as the family's most common routing mistake:
//! asked for a file's contents at a commit, the model reached for the patch
//! tool twice, looped, and on one turn claimed content it had never fetched.
//! A tool choice is made in the projection's top-k, where nothing enforces it;
//! `what` is an enum the constrained decoder enforces, so the same distinction
//! can no longer be got wrong.

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use validator::Validate;
use zend_git::{FilePatch, GitError, LineKind, LineRange, LogRange, ObjectFormat, Oid, Rev};

use super::line_history::line_count;
use super::wire::{WireBlameLine, WireChange, WireTreeEntry};
use super::{is_protected, open, path_arg, path_args, rev_or_head, GitToolError, RevArg};
use crate::tools::file::Paging;
use crate::{RegisteredTool, Replay, Tool, ToolContext};

/// Git's empty tree — the `from` side of a root commit's diff.
fn empty_tree(format: ObjectFormat) -> Oid {
    let hex = match format {
        ObjectFormat::Sha1 => "4b825dc642cb6eb9a060e54bf8d69288fbee4904",
        ObjectFormat::Sha256 => "6ef19b41225c5369f1c104d45d8d85efa9b057b53b14b4b9b939dd74decc5321",
    };
    Oid::parse(hex).expect("the empty tree id is valid for its format")
}

/// What to show.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, Serialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum ShowWhat {
    /// Which files differ — paths and how each changed, no line detail.
    Changes,
    /// The changed lines themselves, hunk by hunk.
    Patch,
    /// A file's full contents as the revision holds them.
    File,
    /// A directory's entries at the revision.
    Tree,
    /// Each line of a file attributed to the commit that last changed it.
    Blame,
}

#[derive(Deserialize, JsonSchema, Validate)]
#[serde(deny_unknown_fields)]
pub struct ShowRequest {
    /// The repository to read. Required.
    #[validate(length(min = 1))]
    pub repo: String,
    /// What to show. Required.
    pub what: ShowWhat,
    /// The revision to read, or the newer side of a comparison. Defaults to
    /// the checked-out `HEAD` — the last commit, not the files on disk.
    /// Optional, never `null` — see [`RevArg`].
    #[serde(default)]
    #[schemars(with = "RevArg")]
    pub rev: Option<RevArg>,
    /// The older side, for `changes` and `patch`. Defaults to `rev`'s first
    /// parent, so those modes show what one commit did.
    #[serde(default)]
    #[schemars(with = "RevArg")]
    pub from: Option<RevArg>,
    /// **A path inside the repository**, never a revision: the file to read
    /// for `file` and `blame`, or the directory to list for `tree` (empty
    /// string for the repository root). A commit id belongs in `rev`, not
    /// here. Ignored by `changes` and `patch`, which take `paths`.
    #[serde(default)]
    #[schemars(with = "String")]
    pub path: Option<String>,
    /// Limit `changes` and `patch` to these paths.
    #[serde(default)]
    #[schemars(with = "Vec<String>")]
    pub paths: Option<Vec<String>>,
    /// Unchanged lines around each change, for `patch` (0–25). Defaults to 3.
    #[validate(range(max = 25))]
    #[serde(default)]
    #[schemars(with = "u32")]
    pub context: Option<u32>,
    /// Zero-based page to return. Required — pass 0 to start. Every mode
    /// pages; the reply's `paging.next_page` is the page to ask for next, and
    /// an over-shot page clamps to the last one.
    pub page: u32,
}

// Each mode's page is sized to cost about what a `file_read` page does, a
// couple of thousand tokens, because the item each counts varies by an order
// of magnitude: a changed path is ~20 tokens, a tree entry ~40 (it carries a
// full object id), a blame line ~70 (text, commit id, author, date, summary).
// Counting all of them in the same units put a 200-entry tree at ~8k tokens and
// a 150-line blame at ~10k, all of it prefilled before the model read a line.
const CHANGES_PER_PAGE: usize = 60;
const PATCH_LINES_PER_PAGE: usize = 200;
const FILE_LINES_PER_PAGE: usize = 200;
const TREE_PER_PAGE: usize = 50;
const BLAME_PER_PAGE: usize = 40;

#[derive(Serialize)]
pub struct WirePatchLine {
    /// `context`, `added` or `removed`.
    pub kind: &'static str,
    pub text: String,
}

#[derive(Serialize)]
pub struct WireHunk {
    pub old_start: u32,
    pub new_start: u32,
    #[serde(skip_serializing_if = "String::is_empty")]
    pub context: String,
    pub lines: Vec<WirePatchLine>,
}

#[derive(Serialize)]
pub struct WireFilePatch {
    #[serde(flatten)]
    pub change: WireChange,
    pub binary: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub added: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub removed: Option<u32>,
    pub hunks: Vec<WireHunk>,
}

#[derive(Serialize)]
pub struct ShowResponse {
    pub repo: String,
    /// The commit `rev` resolved to.
    pub commit: String,
    /// Its subject, when a single commit was shown.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub subject: Option<String>,
    /// Its full message, on the first page of `changes` and `patch` only —
    /// the one place a commit's body is read, since git_log lists subjects.
    /// Later pages leave it out rather than paying for it again.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub message: Option<String>,
    pub paging: Paging,
    /// `changes`: the files that differ.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub changes: Option<Vec<WireChange>>,
    /// `patch`: the changed lines.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub files: Option<Vec<WireFilePatch>>,
    /// `file`: the text of this page of the file.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<String>,
    /// `file`: set when the blob is not valid UTF-8; `content` is then absent.
    #[serde(skip_serializing_if = "std::ops::Not::not")]
    pub binary: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub bytes: Option<usize>,
    /// `tree`: the directory's entries.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub entries: Option<Vec<WireTreeEntry>>,
    /// `blame`: one record per line.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub lines: Option<Vec<WireBlameLine>>,
}

impl ShowResponse {
    fn empty(repo: String, commit: String, paging: Paging) -> Self {
        Self {
            repo,
            commit,
            subject: None,
            message: None,
            paging,
            changes: None,
            files: None,
            content: None,
            binary: false,
            bytes: None,
            entries: None,
            lines: None,
        }
    }
}

fn line_kind(k: LineKind) -> &'static str {
    match k {
        LineKind::Context => "context",
        LineKind::Added => "added",
        LineKind::Removed => "removed",
    }
}

/// Render patches, taking exactly the page's slice of the total line budget,
/// so pages are bounded and never repeat a line. A hunk that crosses a page
/// boundary is cut there, and each piece starts at the old and new line
/// numbers of its own first line. A file whose hunks fall outside the page
/// still appears with its counts, so the reply never hides that a file
/// changed.
fn render(patches: &[FilePatch], paging: &Paging) -> Vec<WireFilePatch> {
    let mut seen = 0usize;
    let start = paging.skipped();
    let end = start + paging.per_page;
    let mut out = Vec::with_capacity(patches.len());
    for p in patches {
        let mut hunks = Vec::new();
        for h in &p.hunks {
            let n = h.lines.len();
            if seen + n > start && seen < end {
                let lo = start.saturating_sub(seen);
                let hi = (end - seen).min(n);
                let (mut old_start, mut new_start) = (h.old_start, h.new_start);
                for l in &h.lines[..lo] {
                    match l.kind {
                        LineKind::Context => {
                            old_start += 1;
                            new_start += 1;
                        }
                        LineKind::Removed => old_start += 1,
                        LineKind::Added => new_start += 1,
                    }
                }
                hunks.push(WireHunk {
                    old_start,
                    new_start,
                    context: h.context.clone(),
                    lines: h.lines[lo..hi]
                        .iter()
                        .map(|l| WirePatchLine {
                            kind: line_kind(l.kind),
                            text: String::from_utf8_lossy(&l.text).into_owned(),
                        })
                        .collect(),
                });
            }
            seen += n;
        }
        out.push(WireFilePatch {
            change: (&p.entry).into(),
            binary: p.binary,
            added: p.added(),
            removed: p.removed(),
            hunks,
        });
    }
    out
}

pub struct GitShow;

impl Tool for GitShow {
    const NAME: &'static str = "git_show";
    const DESCRIPTION: &'static str =
        "Show repository content at, or between, revisions — five views, picked by \
         `what`: what a commit changed (`changes`, `patch`), what a file or directory held \
         (`file`, `tree`), and who last changed each line of a file (`blame`). In detail: \
         `changes` lists the files that differ; `patch` gives the changed lines hunk by \
         hunk; `file` returns a file's full contents as that revision holds them; `tree` \
         lists a directory; `blame` attributes each line to the commit that last changed \
         it. **`rev` is which revision, `path` is which file** — a commit id goes in \
         `rev`, never in `path`. `rev` defaults to HEAD — the last commit, not the files \
         on disk — and for `changes` and `patch` `from` defaults to its first parent, so \
         those show what one commit did, and their first page carries the commit's full \
         message. Use for \"show me that commit\", \"what did this change\", \"show me \
         this file as of that tag\", \"who wrote this line\", \"what is in that directory\". \
         `page` is required in every mode — pass 0 to start, and `paging.next_page` for \
         the next. For the file as it stands on disk right now, uncommitted edits \
         included, use file_read. Reads only.";

    type Request = ShowRequest;
    type Response = ShowResponse;
    type Error = GitToolError;

    /// Reads objects; writes nothing.
    fn replay(_req: &Self::Request) -> Replay {
        Replay::Safe
    }

    fn run(ctx: &ToolContext, req: ShowRequest) -> Result<ShowResponse, GitToolError> {
        let repo = open(ctx, &req.repo)?;
        let to = rev_or_head(&req.rev, &repo)?;
        let commit = repo.resolve(&to)?.as_str().to_string();
        let page = req.page;

        match req.what {
            ShowWhat::Changes | ShowWhat::Patch => {
                let info = repo.log(&LogRange::of(to.clone()), 1)?;
                let head = info.first();
                let from = match &req.from {
                    Some(r) => r.resolve(&repo)?,
                    // A root commit has no parent: everything in it is an
                    // addition, which is a diff against the empty tree.
                    None => match head.and_then(|c| c.parents.first()) {
                        Some(parent) => Rev::Oid(parent.clone()),
                        None => Rev::Oid(empty_tree(repo.format())),
                    },
                };
                let subject = head.map(|c| c.subject().to_string());
                let message_on =
                    |paging: &Paging| head.filter(|_| paging.page == 0).map(|c| c.message.clone());
                let paths = path_args(req.paths.as_deref().unwrap_or(&[]))?;
                let path_refs: Vec<_> = paths.iter().collect();

                if req.what == ShowWhat::Changes {
                    let all: Vec<_> = repo
                        .diff(&from, &to, &path_refs)?
                        .into_iter()
                        .filter(|e| !touches_protected(e))
                        .collect();
                    let paging = Paging::of(all.len(), page, CHANGES_PER_PAGE);
                    let changes = all
                        .iter()
                        .skip(paging.skipped())
                        .take(paging.per_page)
                        .map(Into::into)
                        .collect();
                    return Ok(ShowResponse {
                        subject,
                        message: message_on(&paging),
                        changes: Some(changes),
                        ..ShowResponse::empty(req.repo, commit, paging)
                    });
                }

                // A patch carries file contents, so a protected path must not
                // appear — the call never had to name it.
                let patches: Vec<_> = repo
                    .patches(&from, &to, req.context.unwrap_or(3), &path_refs)?
                    .into_iter()
                    .filter(|p| !touches_protected(&p.entry))
                    .collect();
                let total: usize = patches
                    .iter()
                    .flat_map(|p| &p.hunks)
                    .map(|h| h.lines.len())
                    .sum();
                let paging = Paging::of(total, page, PATCH_LINES_PER_PAGE);
                Ok(ShowResponse {
                    subject,
                    message: message_on(&paging),
                    files: Some(render(&patches, &paging)),
                    ..ShowResponse::empty(req.repo, commit, paging)
                })
            }

            ShowWhat::Tree => {
                let dir = req.path.as_deref().unwrap_or("");
                let dir_path = (!dir.is_empty()).then(|| path_arg(dir)).transpose()?;
                let all: Vec<_> = repo
                    .ls_tree(&to, dir_path.as_ref())?
                    .into_iter()
                    .filter(|e| !is_protected(&e.path))
                    .collect();
                let paging = Paging::of(all.len(), page, TREE_PER_PAGE);
                let entries = all
                    .iter()
                    .skip(paging.skipped())
                    .take(paging.per_page)
                    .map(Into::into)
                    .collect();
                Ok(ShowResponse {
                    entries: Some(entries),
                    ..ShowResponse::empty(req.repo, commit, paging)
                })
            }

            ShowWhat::File => {
                let path = require_path(&req.path, "file")?;
                let parsed = path_arg(&path)?;
                let Some(bytes) = repo.blobs().read_at(&to, &parsed)? else {
                    return Err(missing(&commit, &path));
                };
                let Ok(text) = String::from_utf8(bytes.clone()) else {
                    let paging = Paging::of(0, page, FILE_LINES_PER_PAGE);
                    return Ok(ShowResponse {
                        binary: true,
                        bytes: Some(bytes.len()),
                        ..ShowResponse::empty(req.repo, commit, paging)
                    });
                };
                let all: Vec<&str> = text.lines().collect();
                let paging = Paging::of(all.len(), page, FILE_LINES_PER_PAGE);
                let content = all
                    .iter()
                    .skip(paging.skipped())
                    .take(paging.per_page)
                    .copied()
                    .collect::<Vec<_>>()
                    .join("\n");
                Ok(ShowResponse {
                    content: Some(content),
                    bytes: Some(bytes.len()),
                    ..ShowResponse::empty(req.repo, commit, paging)
                })
            }

            ShowWhat::Blame => {
                let path = require_path(&req.path, "blame")?;
                let parsed = path_arg(&path)?;
                // Blame the page's window rather than the whole file: this is
                // the one read that costs a record per line, and an unbounded
                // whole-file blame measured at seventeen minutes live. The
                // file's length, read from its blob, sets the paging.
                let Some(bytes) = repo.blobs().read_at(&to, &parsed)? else {
                    return Err(missing(&commit, &path));
                };
                let total = line_count(&bytes) as usize;
                let paging = Paging::of(total, page, BLAME_PER_PAGE);
                let first = paging.skipped() + 1;
                let last = (paging.skipped() + paging.per_page).min(total);
                let lines = if first > last {
                    Vec::new()
                } else {
                    let window = LineRange {
                        start: first as u32,
                        end: last as u32,
                    };
                    repo.blame(&to, &parsed, Some(window))?
                        .ok_or_else(|| missing(&commit, &path))?
                        .iter()
                        .map(|l| WireBlameLine::new(l, &path))
                        .collect()
                };
                Ok(ShowResponse {
                    lines: Some(lines),
                    ..ShowResponse::empty(req.repo, commit, paging)
                })
            }
        }
    }
}

fn require_path(path: &Option<String>, what: &str) -> Result<String, GitToolError> {
    path.as_deref()
        .filter(|p| !p.is_empty())
        .map(str::to_string)
        .ok_or_else(|| GitError::invalid(format!("`{what}` needs a `path`")).into())
}

fn missing(commit: &str, path: &str) -> GitToolError {
    GitError::UnknownRevision {
        rev: format!("{commit}:{path}"),
    }
    .into()
}

fn touches_protected(e: &zend_git::DiffEntry) -> bool {
    [&e.old, &e.new]
        .into_iter()
        .flatten()
        .any(|s| is_protected(&s.path))
}

pub const GIT_SHOW: RegisteredTool = RegisteredTool::new::<GitShow>();
