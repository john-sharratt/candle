//! file_edit tool — one anchored edit (`docs/npcd_anchored_edits.md` §6).

use doc_units::render::{around, entry, Source};
use doc_units::{apply, Doc, Edit, EditError, Op, Place};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use validator::Validate;

use super::render::fence_tag_for_path;
use super::{edit_changes, FileError};
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
    /// The anchor of the part to change: the `part=` of its block in `file_read` (e.g. `r7Lm`). Required.
    #[validate(length(min = 1))]
    pub part: String,
    /// What to do there: `replace` it, `insert_before` or `insert_after` it, `delete` it, or `move` it. Required.
    #[serde(rename = "do")]
    #[schemars(with = "EditAction")]
    pub action: String,
    /// For replace, insert and delete: the first line of the part the edit covers, `+N` as a read numbers it. Without it, the edit is to the part whole.
    #[serde(default)]
    #[schemars(with = "String")]
    pub from: Option<String>,
    /// For replace and delete with `from`: the last line covered, `+N`; left out, `from` alone.
    #[serde(default)]
    #[schemars(with = "String")]
    pub to: Option<String>,
    /// For replace only: words inside the part to change, which must occur in it exactly once. With `old`, only those words are replaced by `text`. Comes before `text`: the grammar offers the arguments in this order, and `text` declared first left a call that wrote `old` no way to say what replaces it.
    pub old: Option<String>,
    /// For replace, insert_before and insert_after: the new text — the lines `from` through `to` as they are to stand, the words replacing `old`, or the whole part — its indentation included. Not given for delete or move.
    pub text: Option<String>,
    /// For a whole-part replace, delete or move: the anchor of the last part of a run of consecutive parts, so the edit covers `part` through `through`.
    pub through: Option<String>,
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
        "Change an existing file by one of its parts, leaving the rest exactly as it is. \
         `file_read` shows every part of a file — a function, a type, a block of imports, a \
         paragraph — as a block whose fence names its four-character anchor (`part=r7Lm`) and \
         numbers each of its lines `+1`, `+2`, …; name the part as `part` and say what to `do`. \
         With `from` (and `to`), the edit covers those lines of the part, by those numbers: \
         `replace` them with `text`, `insert_before` / `insert_after` line `from` the lines \
         `text`, or `delete` them. Without `from` it covers the part whole: `replace` it with \
         `text` (or, with `old`, replace just those words inside it), `insert_before` / \
         `insert_after` it a new part `text`, `delete` it, or `move` it `before`/`after` the part \
         `dest`; `through` extends replace, delete and move over a run of consecutive parts. \
         Write `text` as the lines are to stand, indentation included — never with the `+N` \
         numbers a read shows beside them. Use for: changing a function \
         body, adding a method after another, fixing a value in a config, removing dead code, \
         reordering items. An anchor that no longer names a part — the file changed since it was \
         read — is refused; read it again. Returns what was done and the parts it wrote under \
         their new anchors, between their neighbours, so the next edit needs no read. It also \
         lists the lines that changed, `-` for what was removed and `+` for what was written: a \
         line listed there you did not mean to change is a slip to fix with the next edit. Triggered \
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
        if let Some(why) = req.text.as_deref().and_then(numbered) {
            return Err(FileError::InvalidArguments(why));
        }
        let doc = Doc::new(&req.path, &content);
        let done = apply(&doc, &edit).map_err(|e| refused(&doc, e))?;
        let text = done.doc.text();
        let bytes = text.len();
        if let (None, None, "replace", Some(new)) =
            (&req.old, &req.from, req.action.as_str(), &req.text)
        {
            let (written, after) = (new.lines().count(), text.lines().count());
            if let Some(why) = clobbers(content.lines().count(), after, written) {
                return Err(FileError::InvalidArguments(why));
            }
        }
        // The lines that changed, in file order. A whole-part replace is
        // retyped by the caller and can drop a character from a line nobody
        // meant to touch, and the parts around it read the same either way —
        // so the lines themselves are named here. A move carries no line's
        // text into new text, so it has nothing to report. Taken before the
        // store moves `text` out of reach.
        let changes = if req.action == "move" {
            String::new()
        } else {
            edit_changes::changed(&content, &text)
        };
        // Recorded as an edit — the changed lines only, not the file they
        // landed in.
        store.edit(&req.path, text)?;
        let file = format!("{}/{}", req.repo, req.path);
        let source = Source {
            file: &file,
            tag: fence_tag_for_path(&req.path),
        };
        let mut result = format!(
            "{}\n{}",
            done.summary,
            around(&done.doc, &done.touched, done.at_line, source)
        );
        if !changes.is_empty() {
            result.push_str(&format!("Changed lines:\n{changes}"));
        }
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

/// Why `text` reads as lines copied out of a read with their numbers, if it
/// does: every line that has anything on it starts with a `+N` gutter. Written
/// into the file, each would gain the number as content.
fn numbered(text: &str) -> Option<String> {
    let gutter = |line: &str| {
        let digits = line.trim_start().strip_prefix('+')?;
        let n = digits.bytes().take_while(u8::is_ascii_digit).count();
        (n > 0 && matches!(digits.as_bytes().get(n), None | Some(b' '))).then_some(())
    };
    let mut lines = text.lines().filter(|l| !l.trim().is_empty()).peekable();
    lines.peek()?;
    lines.all(|l| gutter(l).is_some()).then(|| {
        "every line of `text` starts with a `+N` number, as a read shows it beside the \
         line; write the lines without their numbers — they are not part of the file"
            .to_string()
    })
}

/// The most parts a stale edit's refusal lists.
const STALE_PARTS: usize = 60;

/// `e` as the call's error. A stale anchor carries the parts the file has
/// now, one line each, so the next edit can name one without a read in
/// between (`docs/zen_coding_challenge_fixes.md` Cause 8).
fn refused(doc: &Doc, e: EditError) -> FileError {
    let EditError::Stale(_) = &e else {
        return e.into();
    };
    let n = doc.units().len();
    let parts: String = (0..n.min(STALE_PARTS)).map(|i| entry(doc, i)).collect();
    let more = match n > STALE_PARTS {
        true => format!(
            "… and {} more — `file_read` with `list` names them all\n",
            n - STALE_PARTS
        ),
        false => String::new(),
    };
    FileError::Stale(format!("{e}. Its parts now:\n{parts}{more}"))
}

/// `+N`, a line of a part as a read numbers it, as `N`.
fn line_of(name: &str, value: &str) -> Result<usize, FileError> {
    value
        .strip_prefix('+')
        .filter(|n| !n.is_empty() && n.bytes().all(|b| b.is_ascii_digit()))
        .and_then(|n| n.parse::<usize>().ok())
        .filter(|&n| n >= 1)
        .ok_or_else(|| {
            FileError::InvalidArguments(format!(
                "`{name}` is a line of the part as a read numbers it, `+1` upwards — {value:?} \
                 is not; a line of the file is not what it takes"
            ))
        })
}

/// The lines `from` through `to` the request names, if it names any.
fn line_range(req: &EditRequest) -> Result<Option<(usize, usize)>, FileError> {
    match (&req.from, &req.to) {
        (None, None) => Ok(None),
        (None, Some(_)) => Err(FileError::InvalidArguments(
            "`to` ends a range that `from` starts; give `from` too".into(),
        )),
        (Some(from), to) => {
            let from = line_of("from", from)?;
            let to = to.as_deref().map(|t| line_of("to", t)).transpose()?;
            Ok(Some((from, to.unwrap_or(from))))
        }
    }
}

/// The request as the edit it names, or why it names none.
fn edit_of(req: &EditRequest) -> Result<Edit, FileError> {
    let invalid = |why: &str| FileError::InvalidArguments(why.to_string());
    if req.part.chars().count() != 4 {
        return Err(invalid(&format!(
            "`part` is a part's four-character anchor, the `part=` of its block; {:?} is not — \
             copy the anchor whole from `file_read`",
            req.part
        )));
    }
    let lines = line_range(req)?;
    if lines.is_some() && (req.old.is_some() || req.through.is_some()) {
        return Err(invalid(
            "`from` names lines of one part: it takes no `old`, which changes words of the \
             part, and no `through`, which runs on over whole parts",
        ));
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
    let insert = |place: Place| -> Result<Op, FileError> {
        Ok(match lines {
            Some((line, last)) if line != last => {
                return Err(invalid(
                    "an insert goes before or after one line, `from`; it takes no `to`",
                ))
            }
            Some((line, _)) => Op::InsertLines {
                place,
                line,
                text: text()?,
            },
            None => Op::Insert {
                place,
                text: text()?,
            },
        })
    };
    let op = match (req.action.as_str(), lines) {
        ("replace", Some((from, to))) => Op::ReplaceLines {
            from,
            to,
            text: text()?,
        },
        ("replace", None) => Op::Replace {
            text: text()?,
            old: req.old.clone(),
            to: req.through.clone(),
        },
        ("insert_before", _) => insert(Place::Before)?,
        ("insert_after", _) => insert(Place::After)?,
        ("delete", Some((from, to))) => Op::DeleteLines { from, to },
        ("delete", None) => Op::Delete {
            to: req.through.clone(),
        },
        ("move", Some(_)) => {
            return Err(invalid(
                "`move` moves whole parts; it takes no `from` — to move lines, delete them and \
                 insert them where they go",
            ))
        }
        ("move", None) => Op::Move {
            to: req.through.clone(),
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
        (other, _) => {
            return Err(invalid(&format!(
                "`do` is {other:?}; it is one of replace, insert_before, insert_after, delete, move"
            )))
        }
    };
    Ok(Edit {
        at: req.part.clone(),
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

    /// **Text copied with its gutter is refused**: every line numbered as a read
    /// numbers it. A line that only starts with `+` is content.
    #[test]
    fn text_copied_with_its_numbers_is_refused() {
        assert!(numbered(" +1 fn go() {\n +2     1\n +3 }\n").is_some());
        assert!(numbered("+1 x\n\n+2\n").is_some());
        assert_eq!(numbered("fn go() {}\n"), None);
        assert_eq!(numbered("+1 x\nplain\n"), None, "one line without");
        assert_eq!(numbered("+x\n"), None);
        assert_eq!(numbered("+10px\n"), None);
        assert_eq!(numbered(""), None);
    }

    fn request(action: &str, from: Option<&str>, to: Option<&str>) -> EditRequest {
        EditRequest {
            repo: "r".into(),
            path: "a.rs".into(),
            part: "KehT".into(),
            action: action.into(),
            from: from.map(String::from),
            to: to.map(String::from),
            old: None,
            text: Some("x".into()),
            through: None,
            dest: None,
            place: None,
        }
    }

    /// **`from` and `to` name lines of the part as `+N`**, and an edit with
    /// them is the line operation; without them, the part whole.
    #[test]
    fn from_and_to_name_lines_of_the_part() {
        let op = |action, from, to| edit_of(&request(action, from, to)).map(|e| e.op);
        assert_eq!(
            op("replace", Some("+2"), Some("+5")).unwrap(),
            Op::ReplaceLines {
                from: 2,
                to: 5,
                text: "x".into()
            }
        );
        assert_eq!(
            op("delete", Some("+3"), None).unwrap(),
            Op::DeleteLines { from: 3, to: 3 }
        );
        assert_eq!(
            op("insert_after", Some("+1"), None).unwrap(),
            Op::InsertLines {
                place: Place::After,
                line: 1,
                text: "x".into()
            }
        );
        assert_eq!(
            op("replace", None, None).unwrap(),
            Op::Replace {
                text: "x".into(),
                old: None,
                to: None
            }
        );
        for (action, from, to) in [
            ("replace", Some("3"), None),
            ("replace", Some("+0"), None),
            ("replace", Some("+3x"), None),
            ("replace", None, Some("+3")),
            ("insert_after", Some("+1"), Some("+2")),
            ("move", Some("+1"), None),
        ] {
            assert!(
                matches!(op(action, from, to), Err(FileError::InvalidArguments(_))),
                "{action} {from:?} {to:?}"
            );
        }
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
