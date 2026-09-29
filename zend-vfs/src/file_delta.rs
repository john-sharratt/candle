//! A change to one file, recorded as what changed rather than what resulted.
//!
//! The session layer of a [`VfsStore`](super::VfsStore) overlay keeps, per
//! path, the deltas that took the file from where it started — the
//! workspace's copy, or nothing — to where it stands now:
//!
//! - **a whole-file write** is one [`FileDelta::Replace`], carrying the new
//!   content;
//! - **an edit** is one [`FileDelta::Edit`], carrying only the lines that
//!   changed, as [`Splice`]s against the text the edit was made on — unless it
//!   changes more than [`REPLACE_ABOVE_PERCENT`] of the file's lines, when it is
//!   recorded as the whole file, a [`FileDelta::Replace`] ([`delta`]);
//! - **a deletion** is [`FileDelta::Delete`];
//! - **a file that is not text** — anything that is not valid UTF-8, which a
//!   tool run can produce — is one [`FileDelta::ReplaceBinary`], carrying its
//!   bytes. Only text is ever described by its changed lines.
//!
//! Replay is exact and checked. Every splice names the text it removes, so an
//! edit replayed onto a text other than the one it was made against is refused
//! ([`Diverged`]) rather than landing in the wrong place. [`replay_bytes`] is
//! the replay every other one is built on: it works on a file's bytes, so a
//! chain holding a binary replacement replays as exactly as one of text.

use base64::engine::general_purpose::STANDARD as BASE64;
use base64::Engine as _;
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use similar::{ChangeTag, TextDiff};

/// One change to one file.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum FileDelta {
    /// The whole file is now `content`, whatever it held before.
    Replace { content: String },
    /// The whole file is now `content`, which is not text. Carried as base64
    /// on the wire.
    ReplaceBinary {
        #[serde(serialize_with = "to_base64", deserialize_with = "from_base64")]
        content: Vec<u8>,
    },
    /// The file as it stood, changed by `splices` — in ascending position,
    /// non-overlapping, each positioned in the text before the edit.
    Edit { splices: Vec<Splice> },
    /// The file is gone.
    Delete,
}

impl FileDelta {
    /// The bytes this delta holds — what the session layer's size cap counts.
    pub fn bytes(&self) -> usize {
        match self {
            FileDelta::Replace { content } => content.len(),
            FileDelta::ReplaceBinary { content } => content.len(),
            FileDelta::Edit { splices } => splices
                .iter()
                .map(|s| s.removed.len() + s.inserted.len())
                .sum(),
            FileDelta::Delete => 0,
        }
    }

    /// Whether this delta settles the file on its own — a replace or a
    /// delete — so that nothing before it in a chain still matters.
    pub fn supersedes(&self) -> bool {
        !matches!(self, FileDelta::Edit { .. })
    }
}

/// A delta and the moment it was made — when the edit was computed or the
/// write or delete executed — in nanoseconds since the Unix epoch. The times
/// are the conversation's own record of when it changed a file; nothing reads
/// them off a filesystem.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TimedDelta {
    pub at_ns: i64,
    #[serde(flatten)]
    pub delta: FileDelta,
}

impl TimedDelta {
    /// `delta`, made now.
    pub fn now(delta: FileDelta) -> Self {
        Self {
            at_ns: now_ns(),
            delta,
        }
    }
}

/// A changed file's times, from its chain of [`TimedDelta`]s.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct FileTimes {
    /// When the file's current chain began: the write that created or replaced
    /// it, or the first edit to the repository's copy.
    pub ctime_ns: i64,
    /// When its latest delta was made.
    pub mtime_ns: i64,
}

impl FileTimes {
    /// The times `chain` records, or `None` for an empty chain.
    pub fn of(chain: &[TimedDelta]) -> Option<FileTimes> {
        Some(FileTimes {
            ctime_ns: chain.first()?.at_ns,
            mtime_ns: chain.last()?.at_ns,
        })
    }
}

/// The current time, in nanoseconds since the Unix epoch — the clock every
/// delta's time and every checkout stamp is read from.
pub fn now_ns() -> i64 {
    let since = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default();
    i64::try_from(since.as_nanos()).unwrap_or(i64::MAX)
}

fn to_base64<S: Serializer>(bytes: &[u8], s: S) -> Result<S::Ok, S::Error> {
    s.serialize_str(&BASE64.encode(bytes))
}

fn from_base64<'de, D: Deserializer<'de>>(d: D) -> Result<Vec<u8>, D::Error> {
    let text = String::deserialize(d)?;
    BASE64.decode(text).map_err(serde::de::Error::custom)
}

/// One run of changed lines: `removed` at byte `at` of the text before the
/// edit becomes `inserted`.
///
/// `removed` is kept, not just its length, for two reasons: replay checks it,
/// which is what makes applying an edit to the wrong text an error rather
/// than a silently wrong file; and it makes the delta readable — and
/// reversible — on its own.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Splice {
    pub at: usize,
    pub removed: String,
    pub inserted: String,
}

/// An edit that does not fit the text it is replayed onto: the splice at
/// byte `at` does not find the text it removes there.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Diverged {
    pub at: usize,
}

/// The splices that turn `old` into `new`, whole lines at a time. Each run of
/// changed lines is one splice; the unchanged lines between runs are not
/// stored. Line terminators are part of the lines, so a CRLF file, and a last
/// line with no newline, come back byte for byte.
///
/// Every splice removes something, so replay always has text to check: a run
/// that only inserts lines is anchored to the unchanged line after it —
/// removing that line and inserting it back after the new ones — or, at the
/// end of the file, to the line before it. An insertion checked against
/// nothing would fit anywhere, and a file changed underneath it would take
/// the new lines at the wrong place instead of refusing them. Only in an empty
/// file is there nothing to anchor to.
pub fn splices(old: &str, new: &str) -> Vec<Splice> {
    anchored(old, changed_runs(old, new))
}

/// The runs of changed lines between `old` and `new`, unanchored.
///
/// The changes are walked in order and the position in `old` is counted
/// here, from the lines each change consumes. The diff's own per-op old
/// positions are not used: an insertion that follows an unchanged line can be
/// reported at that line's index rather than after it, which placed the
/// inserted lines on the wrong side of it.
fn changed_runs(old: &str, new: &str) -> Vec<Splice> {
    let diff = TextDiff::from_lines(old, new);
    let mut out: Vec<Splice> = Vec::new();
    // Byte offset in `old` of the next line a change will consume.
    let mut at = 0;
    // The run of changed lines being gathered, if one is open. A delete next to
    // an insert is one change, not two.
    let mut run: Option<Splice> = None;
    for change in diff.iter_all_changes() {
        let text = change.value();
        match change.tag() {
            ChangeTag::Equal => {
                out.extend(run.take());
                at += text.len();
            }
            ChangeTag::Delete => {
                open(&mut run, at).removed.push_str(text);
                at += text.len();
            }
            ChangeTag::Insert => open(&mut run, at).inserted.push_str(text),
        }
    }
    out.extend(run);
    out
}

/// `runs` with each pure insertion anchored to a neighbouring unchanged line
/// of `old` — see [`splices`]. The neighbour is always unchanged: a changed
/// line next to an insertion would have joined its run.
///
/// Lines inserted just before the last line and just after it anchor to that
/// same line; the two share it as one splice — the lines before, the line,
/// the lines after.
fn anchored(old: &str, runs: Vec<Splice>) -> Vec<Splice> {
    let mut out: Vec<Splice> = Vec::with_capacity(runs.len());
    for s in anchor_each(old, runs) {
        match out.last_mut() {
            Some(prev) if s.at < prev.at + prev.removed.len() => {
                debug_assert_eq!(
                    (prev.at, &prev.removed),
                    (s.at, &s.removed),
                    "only two insertions share an anchor line"
                );
                prev.inserted.push_str(&s.inserted[s.removed.len()..]);
            }
            _ => out.push(s),
        }
    }
    out
}

/// Each of `runs` that only inserts, anchored on its own to the line after
/// it, or at the end of the file the line before.
fn anchor_each(old: &str, runs: Vec<Splice>) -> Vec<Splice> {
    runs.into_iter()
        .map(|mut s| {
            if !s.removed.is_empty() || old.is_empty() {
                return s;
            }
            if s.at < old.len() {
                // The line after: removed, and put back after the new lines.
                let rest = &old[s.at..];
                let len = rest.find('\n').map_or(rest.len(), |n| n + 1);
                let line = &rest[..len];
                s.removed.push_str(line);
                s.inserted.push_str(line);
            } else {
                // At the end: the line before, with the new lines after it.
                let before = &old[..s.at];
                let start = before[..before.len() - 1].rfind('\n').map_or(0, |n| n + 1);
                let line = &before[start..];
                s.at = start;
                s.removed = line.to_string();
                s.inserted = format!("{line}{}", s.inserted);
            }
            s
        })
        .collect()
}

/// The run being gathered, opened at `at` when there is none.
fn open(run: &mut Option<Splice>, at: usize) -> &mut Splice {
    run.get_or_insert_with(|| Splice {
        at,
        removed: String::new(),
        inserted: String::new(),
    })
}

/// An edit changing more than this share of a file's lines, in percent, is
/// recorded as the whole new file rather than as its changed lines.
///
/// Past this point the splices cost about as much as the file — each carries
/// the lines it removes as well as those it inserts — and they buy nothing
/// back: a chain of edits keeps depending on the text they were made against,
/// where a replacement stands on its own.
pub const REPLACE_ABOVE_PERCENT: usize = 50;

/// How to record changing `old` into `new`: the changed lines as a
/// [`FileDelta::Edit`], or, when they are more than
/// [`REPLACE_ABOVE_PERCENT`] of the file's lines, the whole new file as a
/// [`FileDelta::Replace`].
///
/// A run's lines are counted on whichever side of it is longer, and the file's
/// on whichever version is longer, so an edit that rewrites half of a file
/// counts as half whether it grew the file or shrank it.
pub fn delta(old: &str, new: &str) -> FileDelta {
    // Counted before anchoring: an anchor line is unchanged.
    let runs = changed_runs(old, new);
    let changed: usize = runs
        .iter()
        .map(|s| line_count(&s.removed).max(line_count(&s.inserted)))
        .sum();
    let total = line_count(old).max(line_count(new));
    if changed * 100 > total * REPLACE_ABOVE_PERCENT {
        FileDelta::Replace {
            content: new.to_string(),
        }
    } else {
        FileDelta::Edit {
            splices: anchored(old, runs),
        }
    }
}

/// Lines in `text`, a last line without a newline included.
fn line_count(text: &str) -> usize {
    text.lines().count()
}

/// How to record changing a file from `old` to `new`, each `None` when the
/// file does not exist — or `None` when nothing changed.
///
/// A file that ends deleted is a [`FileDelta::Delete`]. Text changed into text
/// is [`delta`] — its changed lines, or the whole file past the threshold. Text
/// that did not exist before is a [`FileDelta::Replace`]. Anything that is not
/// valid UTF-8 afterwards is a [`FileDelta::ReplaceBinary`]; one that was
/// binary before and is text now is a [`FileDelta::Replace`], since there are
/// no lines of the old file to describe the change against.
pub fn between(old: Option<&[u8]>, new: Option<&[u8]>) -> Option<FileDelta> {
    if old == new {
        return None;
    }
    let Some(new) = new else {
        return Some(FileDelta::Delete);
    };
    let Ok(new_text) = std::str::from_utf8(new) else {
        return Some(FileDelta::ReplaceBinary {
            content: new.to_vec(),
        });
    };
    match old.map(std::str::from_utf8) {
        Some(Ok(old_text)) => Some(delta(old_text, new_text)),
        _ => Some(FileDelta::Replace {
            content: new_text.to_string(),
        }),
    }
}

/// Why a chain did not replay.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReplayError {
    /// An edit did not fit the file it was replayed onto — see [`Diverged`].
    Diverged(Diverged),
    /// The chain replayed, but to bytes that are not text, where text was
    /// asked for.
    NotText,
}

impl From<Diverged> for ReplayError {
    fn from(d: Diverged) -> Self {
        ReplayError::Diverged(d)
    }
}

/// `base` changed by `deltas`, in order, as text: what the file holds after
/// them, or `None` when they leave it deleted. `base` is `None` for a file that
/// does not exist. A chain that does not fit is [`ReplayError::Diverged`]; one
/// that ends in bytes that are not UTF-8 is [`ReplayError::NotText`].
pub fn replay<'a>(
    base: Option<String>,
    deltas: impl IntoIterator<Item = &'a FileDelta>,
) -> Result<Option<String>, ReplayError> {
    match replay_bytes(base.map(String::into_bytes), deltas)? {
        None => Ok(None),
        Some(bytes) => String::from_utf8(bytes)
            .map(Some)
            .map_err(|_| ReplayError::NotText),
    }
}

/// `base` changed by `deltas`, in order, as bytes: what the file holds after
/// them, or `None` when they leave it deleted. `base` is `None` for a file that
/// does not exist. An edit over no file, or over bytes it was not made against,
/// is [`Diverged`].
pub fn replay_bytes<'a>(
    base: Option<Vec<u8>>,
    deltas: impl IntoIterator<Item = &'a FileDelta>,
) -> Result<Option<Vec<u8>>, Diverged> {
    let mut bytes = base;
    for delta in deltas {
        bytes = match delta {
            FileDelta::Replace { content } => Some(content.clone().into_bytes()),
            FileDelta::ReplaceBinary { content } => Some(content.clone()),
            FileDelta::Delete => None,
            FileDelta::Edit { splices } => {
                let at = splices.first().map_or(0, |s| s.at);
                Some(apply_bytes(
                    bytes.as_deref().ok_or(Diverged { at })?,
                    splices,
                )?)
            }
        };
    }
    Ok(bytes)
}

/// Apply `splices` to `text`, the text they were made against.
pub fn apply(text: &str, splices: &[Splice]) -> Result<String, Diverged> {
    // A splice positioned inside a character would split it. Refused here, so
    // every splice that remains removes whole text at a character boundary and
    // inserts whole text — and the result is text.
    if let Some(s) = splices.iter().find(|s| !text.is_char_boundary(s.at)) {
        return Err(Diverged { at: s.at });
    }
    let bytes = apply_bytes(text.as_bytes(), splices)?;
    String::from_utf8(bytes).map_err(|e| Diverged {
        at: e.utf8_error().valid_up_to(),
    })
}

/// Apply `splices` to `bytes`, the file they were made against. Each splice
/// must find exactly the bytes of its `removed` text at its position, in
/// ascending, non-overlapping order. A splice that removes nothing has nothing
/// to be checked against, so it fits an empty file only — see [`splices`].
pub fn apply_bytes(bytes: &[u8], splices: &[Splice]) -> Result<Vec<u8>, Diverged> {
    let mut out = Vec::with_capacity(bytes.len());
    let mut cursor = 0;
    for s in splices {
        let diverged = Diverged { at: s.at };
        if s.removed.is_empty() && !bytes.is_empty() {
            return Err(diverged);
        }
        let end = s.at.checked_add(s.removed.len()).ok_or(diverged)?;
        if s.at < cursor || bytes.get(s.at..end) != Some(s.removed.as_bytes()) {
            return Err(diverged);
        }
        out.extend_from_slice(&bytes[cursor..s.at]);
        out.extend_from_slice(s.inserted.as_bytes());
        cursor = end;
    }
    out.extend_from_slice(&bytes[cursor..]);
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn splice(at: usize, removed: &str, inserted: &str) -> Splice {
        Splice {
            at,
            removed: removed.to_string(),
            inserted: inserted.to_string(),
        }
    }

    /// **An insertion is anchored to an unchanged neighbour**, so it is checked
    /// on replay: lines inserted in the middle carry the line after them, lines
    /// added at the end carry the line before, and only an empty file takes a
    /// bare insertion.
    #[test]
    fn an_insertion_is_anchored_to_a_neighbouring_line() {
        let old = "one\ntwo\nthree\n";
        assert_eq!(
            splices(old, "one\nnew\ntwo\nthree\n"),
            vec![splice(4, "two\n", "new\ntwo\n")]
        );
        assert_eq!(
            splices(old, "zero\none\ntwo\nthree\n"),
            vec![splice(0, "one\n", "zero\none\n")]
        );
        assert_eq!(
            splices(old, "one\ntwo\nthree\nfour\n"),
            vec![splice(8, "three\n", "three\nfour\n")]
        );
        assert_eq!(splices("", "new\n"), vec![splice(0, "", "new\n")]);
        // Before and after the last line: one splice sharing it.
        assert_eq!(
            splices(old, "one\ntwo\nbefore\nthree\nafter\n"),
            vec![splice(8, "three\n", "before\nthree\nafter\n")]
        );
        for new in [
            "one\nnew\ntwo\nthree\n",
            "zero\none\ntwo\nthree\n",
            "one\ntwo\nthree\nfour\n",
            "one\ntwo\nbefore\nthree\nafter\n",
        ] {
            assert_eq!(apply(old, &splices(old, new)).unwrap(), new);
        }
    }

    /// **An insertion made against one version of a file does not land in
    /// another**: lines added above it on disk shift everything, and the
    /// replay refuses rather than inserting at the old offset.
    #[test]
    fn an_insertion_into_a_changed_file_diverges() {
        let old = "a\nb\nc\n";
        let edit = splices(old, "a\nb\nnew\nc\n");
        let changed_underneath = "top 1\ntop 2\na\nb\nc\n";
        assert!(apply(changed_underneath, &edit).is_err());
        // A bare insertion fits only an empty file.
        let bare = [splice(0, "", "x\n")];
        assert_eq!(apply("", &bare).unwrap(), "x\n");
        assert!(apply("something\n", &bare).is_err());
    }

    /// **An edit stores only its changed lines**, positioned in the text it
    /// was made against, and replays to the edited text exactly.
    #[test]
    fn an_edit_is_its_changed_lines_only() {
        let old = "one\ntwo\nthree\nfour\nfive\n";
        let new = "one\nTWO\nthree\nfour\nfive\nsix\n";
        let got = splices(old, new);
        assert_eq!(
            got,
            vec![
                splice(4, "two\n", "TWO\n"),
                splice(19, "five\n", "five\nsix\n")
            ]
        );
        assert_eq!(apply(old, &got).unwrap(), new);
    }

    /// A removal next to an insertion is one splice.
    #[test]
    fn a_replaced_run_is_one_splice() {
        let got = splices("a\nb\nc\nd\n", "a\nX\nY\nZ\nd\n");
        assert_eq!(got, vec![splice(2, "b\nc\n", "X\nY\nZ\n")]);
    }

    /// Terminators travel with their lines: CRLF, a missing final newline, and
    /// text past ASCII all come back byte for byte.
    #[test]
    fn replay_is_exact_for_every_kind_of_text() {
        for (old, new) in [
            ("a\r\nb\r\nc\r\n", "a\r\nB\r\nc\r\n"),
            ("alpha\nbeta", "alpha\ngamma"),
            ("alpha", "alpha\nbeta"),
            ("", "fresh\n"),
            ("gone\n", ""),
            ("café\nnaïve\n", "café\nnaïveté\n"),
            ("same\n", "same\n"),
        ] {
            let got = splices(old, new);
            assert_eq!(apply(old, &got).unwrap(), new, "{old:?} -> {new:?}");
        }
        assert!(
            splices("same\n", "same\n").is_empty(),
            "no change, no splice"
        );
    }

    /// **An edit replayed onto a different text is refused**, naming where it
    /// stopped fitting, rather than splicing into the wrong place.
    #[test]
    fn replay_onto_another_text_diverges() {
        let edit = splices("one\ntwo\n", "one\n2\n");
        assert_eq!(apply("one\nTWO\n", &edit), Err(Diverged { at: 4 }));
        assert_eq!(apply("one\n", &edit), Err(Diverged { at: 4 }));
        // Past a char boundary is a divergence, never a panic.
        assert_eq!(apply("é", &[splice(1, "x", "y")]), Err(Diverged { at: 1 }));
    }

    /// **An edit up to the threshold is its changed lines; past it, the whole
    /// file.** Exactly half is still an edit — the threshold is "more than".
    #[test]
    fn an_edit_past_the_threshold_is_recorded_whole() {
        let ten: String = (1..=10).map(|i| format!("line {i}\n")).collect();
        let changed = |n: usize| -> String {
            (1..=10)
                .map(|i| {
                    if i <= n {
                        format!("LINE {i}\n")
                    } else {
                        format!("line {i}\n")
                    }
                })
                .collect()
        };
        assert!(matches!(delta(&ten, &changed(1)), FileDelta::Edit { .. }));
        assert!(matches!(delta(&ten, &changed(5)), FileDelta::Edit { .. }));
        assert_eq!(
            delta(&ten, &changed(6)),
            FileDelta::Replace {
                content: changed(6)
            }
        );
        // Growth counts too: ten lines appended to ten is half the result.
        let grown = format!("{ten}{}", ten.replace("line", "more"));
        assert!(matches!(delta(&ten, &grown), FileDelta::Edit { .. }));
        let tripled = format!("{grown}{}", ten.replace("line", "most"));
        assert!(matches!(delta(&ten, &tripled), FileDelta::Replace { .. }));
        // A file written from nothing is all change.
        assert!(matches!(delta("", "fresh\n"), FileDelta::Replace { .. }));
    }

    /// **An insertion after an unchanged line lands after it.** The line diff
    /// reports this insertion's old position as the unchanged `use` line's own
    /// index, and a splice built from that position put `L54` before `use`
    /// instead of after — found by the generated sweep in
    /// `tests/file_deltas.rs`.
    #[test]
    fn an_insertion_after_an_unchanged_line_lands_after_it() {
        let old = "A\nfn\nL40\nuse\nX\n";
        let new = "A\nfn\nuse\nL54\nuse\nX\n";
        assert_eq!(
            splices(old, new),
            vec![splice(5, "L40\n", ""), splice(13, "X\n", "L54\nuse\nX\n")]
        );
        assert_eq!(apply(old, &splices(old, new)).unwrap(), new);
    }

    #[test]
    fn a_delta_counts_the_bytes_it_holds() {
        assert_eq!(
            FileDelta::Replace {
                content: "12345".into()
            }
            .bytes(),
            5
        );
        assert_eq!(
            FileDelta::Edit {
                splices: vec![splice(0, "ab", "xyz")]
            }
            .bytes(),
            5
        );
        assert_eq!(FileDelta::Delete.bytes(), 0);
    }

    /// The wire form is exactly these bytes — what a record of the delta
    /// will carry.
    #[test]
    fn the_wire_form_is_exactly_these_bytes() {
        let edit = FileDelta::Edit {
            splices: vec![splice(4, "two\n", "2\n")],
        };
        assert_eq!(
            serde_json::to_string(&edit).unwrap(),
            r#"{"kind":"edit","splices":[{"at":4,"removed":"two\n","inserted":"2\n"}]}"#
        );
        assert_eq!(
            serde_json::to_string(&FileDelta::Replace {
                content: "x".into()
            })
            .unwrap(),
            r#"{"kind":"replace","content":"x"}"#
        );
        assert_eq!(
            serde_json::to_string(&FileDelta::Delete).unwrap(),
            r#"{"kind":"delete"}"#
        );
        let binary = FileDelta::ReplaceBinary {
            content: vec![0, 159, 146, 150, 255],
        };
        let wire = serde_json::to_string(&binary).unwrap();
        assert_eq!(wire, r#"{"kind":"replace_binary","content":"AJ+Slv8="}"#);
        assert_eq!(serde_json::from_str::<FileDelta>(&wire).unwrap(), binary);
        assert!(serde_json::from_str::<FileDelta>(
            r#"{"kind":"replace_binary","content":"not base64!"}"#
        )
        .is_err());
    }

    /// **Every kind of change is classified the way it can be replayed**:
    /// nothing for no change, a delete, text edits by their lines, new text
    /// whole, and anything that is not UTF-8 as bytes.
    #[test]
    fn between_classifies_every_kind_of_change() {
        let bin: &[u8] = &[0xff, 0xfe, 0x00];
        assert_eq!(between(None, None), None);
        assert_eq!(between(Some(b"a\n"), Some(b"a\n")), None);
        assert_eq!(between(Some(bin), Some(bin)), None);
        assert_eq!(between(Some(b"a\n"), None), Some(FileDelta::Delete));
        assert_eq!(between(Some(bin), None), Some(FileDelta::Delete));
        assert_eq!(
            between(None, Some(b"new\n")),
            Some(FileDelta::Replace {
                content: "new\n".into()
            })
        );
        assert_eq!(
            between(Some(b"a\nb\nc\nd\n"), Some(b"a\nB\nc\nd\n")),
            Some(FileDelta::Edit {
                splices: vec![splice(2, "b\n", "B\n")]
            })
        );
        assert_eq!(
            between(Some(b"text\n"), Some(bin)),
            Some(FileDelta::ReplaceBinary {
                content: bin.to_vec()
            })
        );
        assert_eq!(
            between(None, Some(bin)),
            Some(FileDelta::ReplaceBinary {
                content: bin.to_vec()
            })
        );
        assert_eq!(
            between(Some(bin), Some(b"text now\n")),
            Some(FileDelta::Replace {
                content: "text now\n".into()
            })
        );
    }

    /// **Bytes replay exactly, text or not**, and the text replay refuses a
    /// chain that ends in bytes that are not text.
    #[test]
    fn a_chain_replays_as_bytes_and_as_text() {
        let bin = vec![0u8, 1, 2, 0xff];
        let chain = [
            FileDelta::ReplaceBinary {
                content: bin.clone(),
            },
            FileDelta::Replace {
                content: "a\nb\n".into(),
            },
            FileDelta::Edit {
                splices: vec![splice(2, "b\n", "B\n")],
            },
        ];
        assert_eq!(replay_bytes(None, &chain), Ok(Some(b"a\nB\n".to_vec())));
        assert_eq!(replay(None, &chain), Ok(Some("a\nB\n".to_string())));
        assert_eq!(replay_bytes(None, &chain[..1]), Ok(Some(bin.clone())));
        assert_eq!(replay(None, &chain[..1]), Err(ReplayError::NotText));
        assert_eq!(
            replay(Some("x".into()), &chain[2..]),
            Err(ReplayError::Diverged(Diverged { at: 2 }))
        );
        assert_eq!(
            apply_bytes(b"abc", &[splice(1, "b", "B")]),
            Ok(b"aBc".to_vec())
        );
        assert!(FileDelta::ReplaceBinary { content: bin }.supersedes());
        assert!(!chain[2].supersedes());
    }

    /// A timed delta carries its moment beside the delta on the wire, and a
    /// chain's times are its first and last deltas'.
    #[test]
    fn a_delta_carries_the_moment_it_was_made() {
        let timed = TimedDelta {
            at_ns: 1_700_000_000_000_000_000,
            delta: FileDelta::Delete,
        };
        let wire = serde_json::to_string(&timed).unwrap();
        assert_eq!(wire, r#"{"at_ns":1700000000000000000,"kind":"delete"}"#);
        assert_eq!(serde_json::from_str::<TimedDelta>(&wire).unwrap(), timed);

        let chain = [
            TimedDelta {
                at_ns: 10,
                delta: FileDelta::Replace {
                    content: "a\n".into(),
                },
            },
            TimedDelta {
                at_ns: 25,
                delta: FileDelta::Edit {
                    splices: vec![splice(0, "a\n", "b\n")],
                },
            },
        ];
        assert_eq!(
            FileTimes::of(&chain),
            Some(FileTimes {
                ctime_ns: 10,
                mtime_ns: 25
            })
        );
        assert_eq!(FileTimes::of(&[]), None);
        assert_eq!(
            replay(None, chain.iter().map(|t| &t.delta)),
            Ok(Some("b\n".to_string()))
        );

        let before = now_ns();
        let made = TimedDelta::now(FileDelta::Delete);
        assert!(made.at_ns >= before && made.at_ns <= now_ns());
    }
}
