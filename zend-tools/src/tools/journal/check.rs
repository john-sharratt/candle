//! journal_check tool: `journals check` over this session's copy of the
//! memory repository, so a job's restructure is checked before it is
//! committed — and, with `against_base`, held to the rules of a change
//! (`journals::tree::compare`): no line lost but those it forgets by name,
//! and the tree's shape no further than its budget.

use std::collections::{BTreeMap, BTreeSet};

use journals::line::LineId;
use journals::tree::compare::{lost, shape, Budget, Shape};
use journals::tree::load::load;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use validator::Validate;

use crate::tools::file::FileError;
use crate::{RegisteredTool, Replay, Tool, ToolContext};

/// The most a change may move each part of the tree's shape; a part left
/// out may not move at all.
#[derive(Deserialize, JsonSchema, Default)]
pub struct BudgetRequest {
    #[serde(default)]
    pub created: usize,
    #[serde(default)]
    pub removed: usize,
    #[serde(default)]
    pub lines_moved: usize,
    #[serde(default)]
    pub summaries_rewritten: usize,
}

#[derive(Deserialize, JsonSchema, Validate)]
pub struct CheckRequest {
    /// The repository holding the journals. Required.
    #[validate(length(min = 1))]
    pub repo: String,
    /// Hold this session's changes to the rules of a change, against the
    /// tree the session began on.
    #[serde(default)]
    pub against_base: bool,
    /// Line ids the change forgets on purpose (`#2026-10-30-0912-7e3a`).
    #[serde(default)]
    pub forgets: Vec<String>,
    /// How far the change may move the tree's shape.
    #[serde(default)]
    pub budget: Option<BudgetRequest>,
}

#[derive(Serialize)]
pub struct ShapeMoved {
    pub created: usize,
    pub removed: usize,
    pub lines_moved: usize,
    pub summaries_rewritten: usize,
}

#[derive(Serialize)]
pub struct CheckResponse {
    pub repo: String,
    pub journals: usize,
    pub lines: usize,
    /// How far the change moved the tree, with `against_base`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub moved: Option<ShapeMoved>,
    /// Everything wrong with the tree as this session holds it, and with the
    /// change; empty when nothing is.
    pub problems: Vec<String>,
}

pub struct JournalCheck;

impl Tool for JournalCheck {
    const NAME: &'static str = "journal_check";
    const DESCRIPTION: &'static str =
        "Check the memory's journal repository as this session holds it, uncommitted changes \
         included, reading it as the memory service would. Use before committing a change to \
         the journals: moved lines, edited entries, a new journal. With `against_base`, also hold \
         the change to its rules: every line the session began with is still somewhere unless \
         `forgets` names it, and the tree's shape moved no further than `budget` (journals \
         created and removed, lines moved, summaries rewritten; a part left out may not move). \
         Triggered by \"check the journals\", \"validate the memory\", \"is the journal tree \
         valid\". Returns how many journals and lines it holds, how far it moved, and every \
         problem found; none means it is safe to commit.";

    type Request = CheckRequest;
    type Response = CheckResponse;
    type Error = FileError;

    fn replay(_req: &Self::Request) -> Replay {
        Replay::Safe
    }

    fn run(ctx: &ToolContext, req: CheckRequest) -> Result<CheckResponse, FileError> {
        let store = ctx.files.repo(&req.repo)?;
        let mut files = BTreeMap::new();
        for path in store.paths("") {
            if let Some(bytes) = store.read_bytes(&path)? {
                files.insert(path, bytes);
            }
        }
        let tree = load(&files);
        let journals = tree.journals();
        let lines = journals.iter().map(|(_, j)| j.all_lines().count()).sum();
        let mut problems = tree.all_problems();
        let mut moved = None;
        if req.against_base {
            let mut base_files = BTreeMap::new();
            if let Some(base) = store.peek_base()? {
                if let Some(base_tree) = store.tree_at(&base)? {
                    for (path, blob, _) in base_tree.files() {
                        if let Some(bytes) = store.blob_at(blob)? {
                            base_files.insert(path.to_string(), bytes);
                        }
                    }
                }
            }
            let old = load(&base_files);
            let mut forgets = BTreeSet::new();
            for f in &req.forgets {
                match LineId::parse(f.trim().trim_start_matches('#')) {
                    Ok(id) => {
                        forgets.insert(id);
                    }
                    Err(e) => problems.push(format!("forgets: {e}")),
                }
            }
            let b = req.budget.unwrap_or_default();
            let budget = Budget(Shape {
                created: b.created,
                removed: b.removed,
                lines_moved: b.lines_moved,
                summaries_rewritten: b.summaries_rewritten,
            });
            let s = shape(&old, &tree);
            problems.extend(lost(&old, &tree, &forgets));
            problems.extend(budget.over(s));
            moved = Some(ShapeMoved {
                created: s.created,
                removed: s.removed,
                lines_moved: s.lines_moved,
                summaries_rewritten: s.summaries_rewritten,
            });
        }
        Ok(CheckResponse {
            repo: req.repo,
            journals: journals.len(),
            lines,
            moved,
            problems,
        })
    }
}

pub const JOURNAL_CHECK: RegisteredTool = RegisteredTool::new::<JournalCheck>();
