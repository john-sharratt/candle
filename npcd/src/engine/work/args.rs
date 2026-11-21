//! Reads the arguments of a station act, which name the subject differently for
//! each act and must be taken exactly as written when they are document contents.
//!
//! Each act names its subject by a different preposition, so the subject is found
//! once here, and a file path always takes precedence over the text beside it.
//! Document contents are not trimmed: a trailing newline is part of the file, and
//! an edit has to match leading whitespace exactly. A mission's own document is
//! refused when an edit would replace most of it, and an edit's arguments are
//! turned into the anchored edit they name, or into the reason they name none.

use super::*;

pub(super) fn text(args: &Map<String, Value>, key: &str) -> Option<String> {
    let s = args.get(key)?.as_str()?.trim();
    (!s.is_empty()).then(|| s.to_string())
}

/// An argument taken exactly as it was written.
///
/// **The contents of a document are not trimmed.** [`text`] trims, which is
/// right for a name or an intent and wrong here twice over: a trailing newline
/// is part of the file, and the leading whitespace of a line is part of what an
/// edit has to match. Trimming either writes something other than what was
/// asked for, and does it silently — the act reports success and the document
/// is subtly not what the character said.
pub(super) fn verbatim(args: &Map<String, Value>, key: &str) -> Option<String> {
    let s = args.get(key)?.as_str()?;
    (!s.trim().is_empty()).then(|| s.to_string())
}

/// The one required argument naming what an act is about, whatever it is called.
///
/// Station acts name their subject differently because the sentence differs —
/// `of` a portrait, `to` an era, `for` a gap, `from` a change. Which word it is
/// carries meaning for the model and none at all for the store, so the store is
/// asked once here rather than in forty match arms.
pub(super) fn subject(args: &Map<String, Value>) -> Option<String> {
    // `path` first: only the file acts take one, and it is always their
    // subject — an edit's `to` is the anchor a span runs to, not an addressee.
    // `why` and `called` are here because two acts carry no subject of their
    // own — a commit acts on whatever you have open, and a new place is named
    // rather than found — and reaching the dispatch at all is what lets them
    // say so themselves rather than being turned away as subjectless.
    for key in [
        "path", "what", "of", "in", "to", "for", "from", "on", "about", "between", "under", "why",
        "called",
    ] {
        if let Some(v) = text(args, key) {
            return Some(v);
        }
    }
    None
}

/// Why an edit to the body's mission piece is refused: what it replaces takes
/// most of the piece, which is writing it again by hand — see [`passage`].
/// `None` for any other document, a mission with no prose floor, an edit that
/// replaces nothing, or a passage.
pub(super) fn rewritten_by_hand(
    s: &Sim,
    body: &str,
    what: &str,
    edit: &Edit,
    shelf: &dyn Shelf,
) -> Option<String> {
    let work = s.missions.active(body)?.work.as_ref()?;
    if work.min_words == 0 || !what.eq_ignore_ascii_case(&work.writes) {
        return None;
    }
    let current = s.bench.read(body, &work.writes, shelf).ok()?;
    let taken = match &edit.op {
        Op::Replace { old: Some(old), .. } => old.clone(),
        Op::Replace { old: None, to, .. } => {
            let doc = Doc::new(&work.writes, &current);
            let a = doc.find(&edit.at)?;
            let b = to.as_deref().and_then(|t| doc.find(t)).unwrap_or(a);
            (a..=b.max(a))
                .map(|i| doc.unit_text(i))
                .collect::<Vec<_>>()
                .join("\n\n")
        }
        _ => return None,
    };
    passage::rewrites_the_piece(&work.writes, &current, &taken)
}

/// A `file_edit` call's arguments — or one edit of a mending sitting's
/// `mend` — as the anchored edit they name, or why they name none.
pub(crate) fn edit_of(args: &Map<String, Value>) -> Result<Edit, String> {
    let Some(at) = text(args, "at") else {
        return Err(
            "An edit names the paragraph it changes by its anchor, as your read of the document \
             shows it. Say `at`."
                .into(),
        );
    };
    let action = text(args, "do").unwrap_or_default();
    let words =
        || verbatim(args, "text").ok_or_else(|| format!("To {action} you have to say the `text`."));
    let to = text(args, "to");
    let op = match action.as_str() {
        "replace" => Op::Replace {
            text: words()?,
            old: verbatim(args, "old").filter(|o| !o.is_empty()),
            to,
        },
        "insert_before" => Op::Insert {
            place: Place::Before,
            text: words()?,
        },
        "insert_after" => Op::Insert {
            place: Place::After,
            text: words()?,
        },
        "delete" => Op::Delete { to },
        "move" => Op::Move {
            to,
            dest: text(args, "dest")
                .ok_or("A move says which paragraph to put it beside, as `dest`.")?,
            place: match text(args, "where").as_deref() {
                Some("before") => Place::Before,
                Some("after") => Place::After,
                _ => return Err("A move says whether it goes `before` or `after` `dest`.".into()),
            },
        },
        _ => {
            return Err(
                "An edit says what to `do`: replace, insert_before, insert_after, delete or move."
                    .into(),
            )
        }
    };
    Ok(Edit {
        at: at.trim().to_string(),
        op,
    })
}
