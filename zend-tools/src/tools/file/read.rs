//! file_read tool.

use doc_units::Doc;
use schemars::JsonSchema;
use serde::Deserialize;
use validator::Validate;

use super::render::{list_read, lines_read, part_read};
use super::FileError;
use crate::{RegisteredTool, Replay, Tool, ToolContext};

/// A read names its file and exactly one of what to read of it: a `part`, a run
/// of `lines`, or a page of the `list` of its parts. The grammar offers exactly
/// one (the definition's `oneOf`); this type takes all three as optional and
/// [`What::of`] refuses any other count.
///
/// **Every read is bounded, because an unbounded read is not a slower read, it
/// is a different failure.** One `file_read` of a 2,499-line module put 144 KB
/// into a live conversation; three such reads made the next turn a
/// 53,288-token prefill, which the scheduler delivered in 8,192-token chunks
/// while the KV pool ratcheted 6 GB against a card already at 99% — three
/// minutes and fifty seconds of wall clock for one turn. Guidance in the tool
/// description did not prevent it: nine of nine reads in that conversation
/// asked for whole files. So a part is at most 80 lines, a run of `lines` at
/// most [`super::render::PAGE_LINES`] lines of whole parts whatever range is
/// asked for, and a list one line a part.
#[derive(Deserialize, JsonSchema, Validate)]
pub struct ReadRequest {
    /// The repository the file belongs to. Required.
    #[validate(length(min = 1))]
    pub repo: String,
    /// Path of the file to read, relative to the repository — a project file, or one this session created (e.g. `src/main.rs`). Required.
    #[validate(length(min = 1))]
    pub path: String,
    /// One part, by its anchor (`KehT`). One of `part`, `lines`, `list`.
    pub part: Option<String>,
    /// Lines of the file, from 1: `84`, or `120-180`. One of `part`, `lines`, `list`.
    pub lines: Option<String>,
    /// A page of the list of the file's parts, from 1. One of `part`, `lines`, `list`.
    #[validate(range(min = 1))]
    #[schemars(range(min = 1))]
    pub list: Option<u32>,
}

/// What a read asks for.
#[derive(Debug, PartialEq, Eq)]
enum What<'a> {
    Part(&'a str),
    Lines(usize, usize),
    List(usize),
}

impl<'a> What<'a> {
    /// The one thing `req` names, or why it does not name exactly one.
    fn of(req: &'a ReadRequest) -> Result<What<'a>, FileError> {
        match (&req.part, &req.lines, req.list) {
            (Some(part), None, None) => Ok(What::Part(part.trim())),
            (None, Some(lines), None) => {
                let (from, to) = range(lines)?;
                Ok(What::Lines(from, to))
            }
            (None, None, Some(page)) => Ok(What::List(page as usize)),
            _ => Err(FileError::InvalidArguments(
                "name exactly one of `part` (an anchor), `lines` (`84` or `120-180`) or \
                 `list` (a page of the file's parts, from 1)"
                    .to_string(),
            )),
        }
    }
}

/// `84` or `120-180` as the lines it names, from 1, inclusive.
fn range(text: &str) -> Result<(usize, usize), FileError> {
    let bad = || {
        FileError::InvalidArguments(format!(
            "`lines` is a line number or a range of them, from 1 — `84` or `120-180` — not {text:?}"
        ))
    };
    // Digits only: `+3` is a line of a part, which a read of the file's lines
    // does not take, and Rust's parse would read it as 3.
    let number = |s: &str| {
        let s = s.trim();
        s.bytes()
            .all(|b| b.is_ascii_digit())
            .then(|| s.parse::<usize>().ok())
            .flatten()
            .filter(|&n| n >= 1)
    };
    let (from, to) = match text.split_once(['-', '–']) {
        Some((a, b)) => (number(a).ok_or_else(bad)?, number(b).ok_or_else(bad)?),
        None => {
            let n = number(text).ok_or_else(bad)?;
            (n, n)
        }
    };
    match from <= to {
        true => Ok((from, to)),
        false => Err(bad()),
    }
}

pub struct FileRead;

impl Tool for FileRead {
    const NAME: &'static str = "file_read";
    const DESCRIPTION: &'static str =
        "Read a file: one part, a run of lines, or the list of its parts. Resolves \
         against this session's edits first, then falls through to the repository on \
         disk, so real project files can be read directly. Name the repo, the path \
         within it, and exactly one of `part` (an anchor, `KehT`), `lines` (`84`, or \
         `120-180`) or `list` (a page of the list of the file's parts, from 1 — a short \
         file comes back whole). Each part — a function, a type, a block of imports, a \
         paragraph — is a fenced block whose fence names the file, the part's anchor, \
         its lines and its scope (`rust file=candle/src/lib.rs part=KehT lines=55-101 \
         scope=\"impl Act > fn summary\"`), and each of its lines follows behind its \
         number in the part, from `+1`; `file_edit` names a part by its anchor. A read \
         of lines holds whole parts, up to 200 lines, and ends naming the next range \
         to read, `end of candle/src/lib.rs lines 1-198 of 940 — next: lines=199-391`, \
         or that it was the end of the file. To read where an error or a stack trace \
         points, pass its line as `lines`: the read says which part and line of it \
         that is. To find code worth reading, use file_grep, whose hits name their \
         part. For remote filesystems use remote_fs_session_get to download first, then \
         file_read.";

    type Request = ReadRequest;
    /// A rendered read, not a JSON object: the runner places a string result
    /// into the `<tool_response>` verbatim, so a live read is byte-identical to
    /// the `code_reading` ingest's prefilled responses.
    type Response = String;
    type Error = FileError;

    /// Reads a file; writes nothing.
    fn replay(_req: &Self::Request) -> Replay {
        Replay::Safe
    }

    fn run(ctx: &ToolContext, req: ReadRequest) -> Result<String, FileError> {
        let lower = req.path.trim_start().to_ascii_lowercase();
        if lower.starts_with("http://") || lower.starts_with("https://") {
            return Err(FileError::IsUrl(req.path));
        }
        let what = What::of(&req)?;
        let content = ctx
            .files
            .repo(&req.repo)?
            .read(&req.path)?
            .ok_or_else(|| FileError::NotFound(req.path.clone()))?;
        let doc = Doc::new(&req.path, &content);
        Ok(match what {
            What::Part(anchor) => match doc.find(anchor) {
                Some(i) => part_read(&req.repo, &req.path, &doc, i),
                None => {
                    return Err(FileError::Stale(format!(
                        "{anchor} is not a part of {}/{} as it stands — it changed since it \
                         was read, or was never one of its anchors; read the file's `list`, or \
                         the `lines` it was at, for the parts it has now",
                        req.repo, req.path
                    )))
                }
            },
            What::Lines(from, to) => lines_read(&req.repo, &req.path, &doc, from, to),
            What::List(page) => list_read(&req.repo, &req.path, &doc, page),
        })
    }
}

pub const FILE_READ: RegisteredTool = RegisteredTool::new::<FileRead>();

#[cfg(test)]
mod tests {
    use super::*;

    fn request(part: Option<&str>, lines: Option<&str>, list: Option<u32>) -> ReadRequest {
        ReadRequest {
            repo: "r".into(),
            path: "a.rs".into(),
            part: part.map(String::from),
            lines: lines.map(String::from),
            list,
        }
    }

    /// **Exactly one of `part`, `lines` and `list`.**
    #[test]
    fn a_read_names_exactly_one_thing() {
        assert_eq!(
            What::of(&request(Some("KehT"), None, None)).unwrap(),
            What::Part("KehT")
        );
        assert_eq!(
            What::of(&request(None, Some("120-180"), None)).unwrap(),
            What::Lines(120, 180)
        );
        assert_eq!(
            What::of(&request(None, Some("84"), None)).unwrap(),
            What::Lines(84, 84)
        );
        assert_eq!(What::of(&request(None, None, Some(2))).unwrap(), What::List(2));
        for req in [
            request(None, None, None),
            request(Some("KehT"), Some("1"), None),
            request(Some("KehT"), None, Some(1)),
        ] {
            assert!(matches!(What::of(&req), Err(FileError::InvalidArguments(_))));
        }
    }

    #[test]
    fn lines_are_a_number_or_a_range_from_one() {
        assert_eq!(range("7").unwrap(), (7, 7));
        assert_eq!(range(" 3 - 9 ").unwrap(), (3, 9));
        assert_eq!(range("3–9").unwrap(), (3, 9));
        for bad in ["0", "9-3", "x", "", "1-", "-4", "+3"] {
            assert!(range(bad).is_err(), "{bad:?}");
        }
    }
}
