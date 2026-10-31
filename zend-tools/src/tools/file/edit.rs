//! file_edit tool — one anchored edit (`docs/npcd_anchored_edits.md` §6).

use doc_units::render::around;
use doc_units::{apply, Doc, Edit, EditError, Op, Place};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use validator::Validate;

use super::FileError;
use crate::{RegisteredTool, Replay, Tool, ToolContext};

/// Schema-only mirror of the operations `do` accepts, so the generated schema
/// carries a real `"enum"` of them.
#[derive(JsonSchema, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum EditAction {
    Replace,
    InsertBefore,
    InsertAfter,
    Delete,
    Move,
}

/// Schema-only mirror of where a move puts its part.
#[derive(JsonSchema, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum MovePlace {
    Before,
    After,
}

#[derive(Deserialize, JsonSchema, Validate)]
pub struct EditRequest {
    /// The repository the file belongs to. Required.
    #[validate(length(min = 1))]
    pub repo: String,
    /// Path of the file to edit, relative to the repository — a project file, or one this session created (e.g. `src/main.rs`). Required.
    #[validate(length(min = 1))]
    pub path: String,
    /// The anchor of the part to change: the four characters before the `|` that `file_read` shows it under (e.g. `r7Lm`). Required.
    #[validate(length(min = 1))]
    pub at: String,
    /// What to do there: `replace` it, `insert_before` or `insert_after` it, `delete` it, or `move` it. Required.
    #[serde(rename = "do")]
    #[schemars(with = "EditAction")]
    pub action: String,
    /// For replace only: words inside the part to change, which must occur in it exactly once. With `old`, only those words are replaced by `text`; without it the whole part is. Comes before `text`: the grammar offers the arguments in this order, and `text` declared first left a call that wrote `old` no way to say what replaces it.
    pub old: Option<String>,
    /// For replace, insert_before and insert_after: the new text — the words replacing `old`, or the whole part as it is to stand, its indentation included. Not given for delete or move.
    pub text: Option<String>,
    /// For replace, delete and move: the anchor of the last part of a run of consecutive parts, so the edit covers `at` through `to`.
    pub to: Option<String>,
    /// For move: the anchor of the part to move it beside.
    pub dest: Option<String>,
    /// For move: `before` or `after` `dest`.
    #[serde(rename = "where")]
    #[schemars(with = "Option<MovePlace>")]
    pub place: Option<String>,
}

#[derive(Serialize)]
pub struct EditResponse {
    pub repo: String,
    pub path: String,
    /// What was done, and the parts it wrote under their new anchors between
    /// the parts either side of them, as `file_read` shows them.
    pub result: String,
    /// Size of the file after the edit.
    pub bytes: usize,
}

pub struct FileEdit;

impl Tool for FileEdit {
    const NAME: &'static str = "file_edit";
    const DESCRIPTION: &'static str =
        "Change one part of an existing file by its anchor, leaving the rest exactly as it is. \
         `file_read` shows every part of a file — a function, a type, a block of imports, a \
         paragraph — under a four-character anchor (`r7Lm| impl Cache > fn seal`); name the part \
         as `at` and say what to `do`: `replace` it with `text` (or, with `old`, replace just \
         those words inside it), `insert_before` / `insert_after` it a new part `text`, `delete` \
         it, or `move` it `before`/`after` the part `dest`. `to` extends replace, delete and move \
         over a run of consecutive parts, `at` through `to`. Write `text` as the part is to \
         stand, indentation included — never with anchors in it. Use for: changing a function \
         body, adding a method after another, fixing a value in a config, removing dead code, \
         reordering items. An anchor that no longer names a part — the file changed since it was \
         read — is refused; read it again. Returns what was done and the parts it wrote under \
         their new anchors, between their neighbours, so the next edit needs no read. Triggered \
         by \"change X to Y in the file\", \"update this function\", \"add this after\", \"remove \
         that\". For a new file, or rewriting most of one, use write.";

    type Request = EditRequest;
    type Response = EditResponse;
    type Error = FileError;

    /// A replace or delete re-sent after it landed names an anchor the file no
    /// longer has and is refused, and a move re-sent puts its part where it
    /// already stands — so those may be re-issued. **An insert may not**: it
    /// leaves the part it is anchored beside as it was, so the same insert sent
    /// again would land a second copy.
    fn replay(req: &Self::Request) -> Replay {
        match req.action.as_str() {
            "insert_before" | "insert_after" => Replay::Unsafe,
            _ => Replay::Safe,
        }
    }

    fn run(ctx: &ToolContext, req: EditRequest) -> Result<EditResponse, FileError> {
        // Read through the overlay, so a file that lives only in the workspace is
        // editable. Nothing is recorded unless the edit is made.
        let store = ctx.files.repo(&req.repo)?;
        let content = store
            .read(&req.path)?
            .ok_or_else(|| FileError::NothingToEdit(req.path.clone()))?;
        let edit = edit_of(&req)?;
        let doc = Doc::new(&req.path, &content);
        let done = apply(&doc, &edit)?;
        let text = done.doc.text();
        let bytes = text.len();
        if let (None, "replace", Some(new)) = (&req.old, req.action.as_str(), &req.text) {
            let (written, after) = (new.lines().count(), text.lines().count());
            if let Some(why) = clobbers(content.lines().count(), after, written) {
                return Err(FileError::InvalidArguments(why));
            }
        }
        // Recorded as an edit — the changed lines only, not the file they
        // landed in.
        store.edit(&req.path, text)?;
        let mut result = format!(
            "{}\n{}",
            done.summary,
            around(&done.doc, &done.touched, done.at_line)
        );
        if done.broke {
            result.push_str(&format!(
                "{} no longer parses as {} — mend it with the next edit.\n",
                req.path,
                done.doc.language().name()
            ));
        }
        Ok(EditResponse {
            repo: req.repo,
            path: req.path,
            result,
            bytes,
        })
    }
}

/// The fewest lines a whole-part replace may cut a part down from unasked.
const CLOBBER_LINES: usize = 8;

/// Why a whole-part replace taking a file from `before` lines to `after`, with
/// `written` lines of new text, looks like a slip, if it does: a call meant to
/// change a few words that forgot `old` replaces the whole part with them. A
/// worker wiped a function and a test that way (2026-10-31). What it means to
/// shrink that far, it can say: `delete` the part and `insert` the new one.
fn clobbers(before: usize, after: usize, written: usize) -> Option<String> {
    let replaced = (before + written).saturating_sub(after);
    (replaced >= CLOBBER_LINES && written * 4 <= replaced).then(|| {
        format!(
            "without `old` this replaces the whole {replaced}-line part with {written} line(s). \
             To change words inside it, send them as `old`; to give the whole part up for this \
             text, `delete` it and `insert_before` a neighbour"
        )
    })
}

/// The request as the edit it names, or why it names none.
fn edit_of(req: &EditRequest) -> Result<Edit, FileError> {
    let invalid = |why: &str| FileError::InvalidArguments(why.to_string());
    if req.at.chars().count() != 4 {
        return Err(invalid(&format!(
            "`at` is the four characters before the `|`; {:?} is not — copy the anchor whole \
             from `file_read`",
            req.at
        )));
    }
    let text = || {
        req.text.clone().ok_or_else(|| {
            invalid(&match (&req.old, req.action.as_str()) {
                (Some(_), "replace") => {
                    "`replace` with `old` needs `text`, the words that replace them".to_string()
                }
                (None, "replace") => {
                    "`replace` needs `text`, the whole part as it is to stand".to_string()
                }
                (_, action) => format!("`{action}` needs `text`, the part to insert"),
            })
        })
    };
    let op = match req.action.as_str() {
        "replace" => Op::Replace {
            text: text()?,
            old: req.old.clone(),
            to: req.to.clone(),
        },
        "insert_before" => Op::Insert {
            place: Place::Before,
            text: text()?,
        },
        "insert_after" => Op::Insert {
            place: Place::After,
            text: text()?,
        },
        "delete" => Op::Delete { to: req.to.clone() },
        "move" => Op::Move {
            to: req.to.clone(),
            dest: req
                .dest
                .clone()
                .ok_or_else(|| invalid("`move` needs `dest`, the part to move it beside"))?,
            place: match req.place.as_deref() {
                Some("before") => Place::Before,
                Some("after") => Place::After,
                _ => return Err(invalid("`move` needs `where`: `before` or `after`")),
            },
        },
        other => {
            return Err(invalid(&format!(
                "`do` is {other:?}; it is one of replace, insert_before, insert_after, delete, move"
            )))
        }
    };
    Ok(Edit {
        at: req.at.clone(),
        op,
    })
}

impl From<EditError> for FileError {
    fn from(e: EditError) -> Self {
        match e {
            EditError::Stale(_) => FileError::Stale(e.to_string()),
            EditError::Invalid(why) => FileError::InvalidArguments(why),
            EditError::Ambiguous(why) => FileError::Ambiguous(why),
            EditError::Unmatched(why) => FileError::TextUnmatched(why),
        }
    }
}

pub const FILE_EDIT: RegisteredTool = RegisteredTool::new::<FileEdit>();

#[cfg(test)]
mod tests {
    use super::*;

    /// **A few lines over a large part is refused.** The two lines that
    /// replaced a forty-line function; a rewrite of about its size, and a short
    /// part, are not slips.
    #[test]
    fn a_whole_part_cut_to_a_few_lines_is_refused() {
        // 100 lines, a 40-line part replaced by 2: 62 left.
        assert_eq!(
            clobbers(100, 62, 2).as_deref(),
            Some(
                "without `old` this replaces the whole 40-line part with 2 line(s). To change \
                 words inside it, send them as `old`; to give the whole part up for this text, \
                 `delete` it and `insert_before` a neighbour"
            )
        );
        assert_eq!(clobbers(100, 95, 35), None, "a rewrite of about its size");
        assert_eq!(clobbers(100, 95, 1), None, "a six-line part");
        assert_eq!(clobbers(100, 105, 45), None, "it grew");
    }

    /// **`old` is offered before `text`.** The tool grammar emits optional
    /// arguments in declared order, so with `text` first a call that wrote
    /// `old` could not go back: every word-level replace was refused for
    /// wanting `text`.
    #[test]
    fn old_is_offered_before_text() {
        let schema = serde_json::to_string(&schemars::schema_for!(EditRequest)).unwrap();
        let old = schema.find("\"old\"").unwrap();
        let text = schema.find("\"text\"").unwrap();
        assert!(old < text, "{schema}");
    }
}
