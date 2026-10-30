//! `journal_focus {speaker?, about?}`: what this conversation has worked out —
//! who is speaking, and the journals it is about (`docs/memory_backbone.md`
//! §2.4). Every later frame starts from it, and every note written in the
//! conversation records its speaker.

use journals::ops::Op;
use journals::path::JournalPath;
use journals::scope::Section;
use journals::slug::Slug;
use journals::time::Stamp;
use journals::view::index::index_line;
use serde_json::{json, Value};

use super::mem::{Mem, PATHS};
use super::{mem, Answer, Ctx};

pub const NAME: &str = "journal_focus";

pub fn definition() -> Value {
    json!({
        "name": NAME,
        "description": format!("Say what this conversation has worked out: `speaker`, who is talking — a \
            person's name as `people` lists them, worked out from what they say, never assumed to be the \
            account's owner (ask when nothing says) — and `about`, the one to three journals it is about, \
            such as `topics/house-extension/permits`. Set each as it becomes clear and change it when the \
            conversation moves; a part left out keeps what it was. Every later frame opens with the speaker's \
            journal and those it is about, and every note is stamped with the speaker. When the speaker has \
            no journal under `people` yet, make one first with journal_create. {PATHS}"),
        "inputSchema": {
            "type": "object",
            "properties": {
                "speaker": {"type": "string", "description": "Who is speaking: `yen`, or `people/yen`."},
                "about": {"type": "array", "items": {"type": "string"}, "maxItems": 3, "description": "The journals the conversation is about, such as `topics/candle`."}
            },
            "additionalProperties": false
        }
    })
}

/// The person `text` names: `yen`, `people/yen` or `@zen/people/yen`.
fn speaker_of(m: &Mem, text: &str) -> Result<Slug, String> {
    let text = text.trim();
    if !text.contains('/') {
        return Slug::parse(text.trim_start_matches('@'));
    }
    let path = m.path(text)?;
    match path.names() {
        [section, person] if section.as_str() == Section::People.name() => Ok(person.clone()),
        _ => Err(format!(
            "{path} is no person's journal: a speaker is `people/<name>`"
        )),
    }
}

/// The speaker and journals a call names, each part only when given.
type Parts = (Option<Slug>, Option<Vec<JournalPath>>);

/// The focus a call asks for.
fn plan(m: &Mem, args: &Value) -> Result<Parts, String> {
    let speaker = args
        .get("speaker")
        .map(|s| {
            s.as_str()
                .ok_or_else(|| "`speaker` is a name".to_string())
                .and_then(|s| speaker_of(m, s))
        })
        .transpose()?;
    let about = args
        .get("about")
        .map(|a| {
            a.as_array()
                .ok_or_else(|| "`about` is a list of journals".to_string())?
                .iter()
                .map(|p| {
                    p.as_str()
                        .ok_or_else(|| "each of `about` is a path".to_string())
                        .and_then(|p| m.path(p))
                })
                .collect::<Result<Vec<_>, _>>()
        })
        .transpose()?;
    Ok((speaker, about))
}

pub fn call(ctx: &Ctx, args: &Value) -> Answer {
    let m = match mem::open(ctx) {
        Ok(m) => m,
        Err(e) => return Answer::error(e),
    };
    if let Err(e) = m.need_person() {
        return Answer::error(e);
    }
    let Some(here) = m.here.clone() else {
        return Answer::error(format!(
            "this conversation has no journal: {}",
            m.why_no_here()
        ));
    };
    let (speaker, about) = match plan(&m, args) {
        Ok(parts) => parts,
        Err(e) => return Answer::error(e),
    };
    // What the focus now opens, by cue: where the next notes belong.
    let mut paths: Vec<JournalPath> = speaker
        .iter()
        .filter_map(|s| JournalPath::person(here.group(), s).ok())
        .collect();
    paths.extend(about.iter().flatten().cloned());
    let op = Op::Focus {
        path: here,
        speaker,
        about,
    };
    let applied = match m.memory.write(&m.actor(), &op) {
        Ok(a) => a,
        Err(e) => return Answer::error(e),
    };
    let tree = m.memory.tree().unwrap_or_else(|_| m.tree.clone());
    let mut out = vec![format!("{}.", applied.subject)];
    out.extend(
        paths
            .iter()
            .filter_map(|p| tree.get(p).map(|j| index_line(p, j, Stamp::now()))),
    );
    Answer::ok(out.join("\n"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mcp::tools::mem::tests::mem;

    #[test]
    fn a_speaker_is_named_or_pathed_to_their_journal() {
        let m = mem("john");
        assert_eq!(speaker_of(&m, " yen ").unwrap().as_str(), "yen");
        assert_eq!(speaker_of(&m, "@yen").unwrap().as_str(), "yen");
        assert_eq!(
            speaker_of(&m, "@zen/topics").unwrap_err(),
            "@zen/topics is no person's journal: a speaker is `people/<name>`"
        );
    }

    #[test]
    fn a_focus_is_planned_a_part_at_a_time() {
        let m = mem("john");
        assert_eq!(
            plan(&m, &json!({"speaker": "yen", "about": ["topics"]})).unwrap(),
            (
                Some(Slug::parse("yen").unwrap()),
                Some(vec![m.path("topics").unwrap()])
            )
        );
        assert_eq!(
            plan(&m, &json!({"about": ["people"]})).unwrap(),
            (None, Some(vec![m.path("people").unwrap()]))
        );
        assert_eq!(
            plan(&m, &json!({"about": "topics"})).unwrap_err(),
            "`about` is a list of journals"
        );
    }
}
