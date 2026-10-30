//! The operations `POST /v1/journals` takes. Each is a JSON object naming its
//! journals by memory path, as the tools write them — `@group/…`, a bare path
//! in the caller's group, `here` — read for the caller into the journals
//! crate's [`Op`] and the time it is recorded at.
//!
//! | `op` | Fields |
//! |---|---|
//! | `note` | `journal`, `text`, `at`?, `speaker`? |
//! | `edit` | `line` (`<path>#<line id>`), `text` |
//! | `forget` | `line`; or `journal` and `says` |
//! | `create` | `under`, `name`, `cue`, `at`? |
//! | `link` | `under`, `name`, `to`, `cue` |
//! | `accept` | `line` |
//! | `create_group` | `group`, `name`, `about`? |
//! | `describe_group` | `about`, `group`? |
//! | `add_member` | `person`, `name`?, `emails`?, `role`?, `group`? |
//! | `add_bot` | `bot`, `reads`, `group`? |
//! | `remove_member` | `person` or `bot`, `group`? |
//! | `choose_default` | `group` |
//!
//! A group a call leaves unnamed is the caller's own. A note's `speaker` is
//! who it is from: its own, else the batch's. There is no `focus`: a focus
//! is a conversation's own, set by its model through `journal_focus` on
//! `here` (§7.2), and a request is no conversation.
//!
//! `at`, `2026-10-30T09:10:02Z`, is when a note or a journal is recorded as
//! made — the day a remembered fact was learned, for one written down later.
//! It is the line's time and its commit's author time; the commit itself is
//! dated when the service makes it. Never before 1970, nor ahead of this
//! member's clock by more than [`AHEAD_SECS`]. The other operations change a
//! line or a journal as it stands, now, and take no `at`. A field an operation
//! does not take is refused, so a misspelt one is never silently dropped.

use journals::ops::{OnTaken, Op, Which};
use journals::path::Place;
use journals::scope::Section;
use journals::slug::Slug;
use journals::time::Stamp;
use serde_json::{Map, Value};

use crate::mcp::tools::group_add::{self, group_of};
use crate::mcp::tools::mem::{Mem, Miss};
use crate::mcp::tools::string;
use crate::mcp::tools::{group_create, group_default, group_describe};

/// How far ahead of this member's clock an `at` may be: members' clocks
/// differ by a little.
pub const AHEAD_SECS: i64 = 60;

/// One operation of a request, and when it is recorded as happening.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Planned {
    pub op: Op,
    pub at: Stamp,
}

/// The operations, by the name a request gives them.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Kind {
    Note,
    Edit,
    Forget,
    Create,
    Link,
    Accept,
    CreateGroup,
    DescribeGroup,
    AddMember,
    AddBot,
    RemoveMember,
    ChooseDefault,
}

impl Kind {
    fn parse(name: &str) -> Option<Kind> {
        Some(match name {
            "note" => Kind::Note,
            "edit" => Kind::Edit,
            "forget" => Kind::Forget,
            "create" => Kind::Create,
            "link" => Kind::Link,
            "accept" => Kind::Accept,
            "create_group" => Kind::CreateGroup,
            "describe_group" => Kind::DescribeGroup,
            "add_member" => Kind::AddMember,
            "add_bot" => Kind::AddBot,
            "remove_member" => Kind::RemoveMember,
            "choose_default" => Kind::ChooseDefault,
            _ => return None,
        })
    }

    /// The fields it takes beside `op`.
    fn fields(self) -> &'static [&'static str] {
        match self {
            Kind::Note => &["journal", "text", "at", "speaker"],
            Kind::Edit => &["line", "text"],
            Kind::Forget => &["line", "journal", "says"],
            Kind::Create => &["under", "name", "cue", "at"],
            Kind::Link => &["under", "name", "to", "cue"],
            Kind::Accept => &["line"],
            Kind::CreateGroup => &["group", "name", "about"],
            Kind::DescribeGroup => &["about", "group"],
            Kind::AddMember => &["person", "name", "emails", "role", "group"],
            Kind::AddBot => &["bot", "reads", "group"],
            Kind::RemoveMember => &["person", "bot", "group"],
            Kind::ChooseDefault => &["group"],
        }
    }
}

/// A person's slug, as a `speaker` field gives it.
fn speaker_of(v: &Value) -> Result<Slug, String> {
    Slug::parse(
        v.as_str()
            .ok_or("`speaker` is a person's slug")?
            .trim()
            .trim_start_matches('@'),
    )
}

/// `v` read for `m`'s caller at `now`, its note from `speaker` when it names
/// no speaker of its own; or why not, a journal it names that is not there
/// told from an operation written wrong.
pub fn plan(m: &Mem, v: &Value, now: Stamp, speaker: Option<&Slug>) -> Result<Planned, Miss> {
    let obj = v.as_object().ok_or("an operation is a JSON object")?;
    let name = string(v, "op")?;
    let kind = Kind::parse(name).ok_or_else(|| format!("`{name}` is not an operation"))?;
    if let Some(k) = obj
        .keys()
        .find(|k| k.as_str() != "op" && !kind.fields().contains(&k.as_str()))
    {
        return Err(format!("`{k}` is not a field of `{name}`").into());
    }
    let at = at_of(obj, now)?;
    let path = |k: &str| string(v, k).map_err(Miss::from).and_then(|p| m.find(p));
    let address = || {
        string(v, "line")
            .map_err(Miss::from)
            .and_then(|a| m.find_address(a))
    };
    let text = || string(v, "text").map(str::to_string);
    let slug = |k: &str| string(v, k).and_then(Slug::parse);
    let op = match kind {
        Kind::Note => Op::Note {
            path: path("journal")?,
            text: text()?,
            speaker: match obj.get("speaker") {
                Some(s) => Some(speaker_of(s)?),
                None => speaker.cloned().or_else(|| m.speaker()),
            },
        },
        Kind::Edit => Op::Edit {
            address: address()?,
            text: text()?,
        },
        Kind::Forget => match (
            obj.contains_key("line"),
            obj.contains_key("journal"),
            obj.contains_key("says"),
        ) {
            (true, false, false) => {
                let a = address()?;
                Op::Forget {
                    path: a.path,
                    which: Which::Id(a.id),
                }
            }
            (false, true, true) => Op::Forget {
                path: path("journal")?,
                which: Which::Says(string(v, "says")?.to_string()),
            },
            _ => return Err("`forget` names a `line`, or a `journal` and what it `says`".into()),
        },
        Kind::Create => {
            let under = path("under")?;
            // A conversation's name recurs as conversations do, numbered; a
            // topic's or a person's is refused the second time.
            let on_taken = match under.place().map_err(|e| Miss::from(e.to_string()))? {
                Place::Section(Section::Conversations) => OnTaken::Number,
                _ => OnTaken::Refuse,
            };
            Op::Create {
                under,
                name: slug("name")?,
                cue: string(v, "cue")?.to_string(),
                on_taken,
            }
        }
        Kind::Link => Op::Link {
            under: path("under")?,
            name: slug("name")?,
            to: path("to")?,
            cue: string(v, "cue")?.to_string(),
        },
        Kind::Accept => Op::Accept {
            address: address()?,
        },
        Kind::CreateGroup => {
            let (group, name, about) = group_create::plan(v)?;
            Op::CreateGroup { group, name, about }
        }
        Kind::DescribeGroup => group_describe::plan(m, v)?,
        Kind::AddMember | Kind::AddBot => group_add::plan(v, group_of(m, v)?)?,
        Kind::RemoveMember => Op::RemoveMember {
            group: group_of(m, v)?,
            who: group_add::whom(v)?,
        },
        Kind::ChooseDefault => group_default::plan(v)?,
    };
    Ok(Planned { op, at })
}

/// When the operation is recorded as happening: its `at`, or `now`.
fn at_of(obj: &Map<String, Value>, now: Stamp) -> Result<Stamp, String> {
    let Some(v) = obj.get("at") else {
        return Ok(now);
    };
    let text = v
        .as_str()
        .ok_or("`at` is a time written 2026-10-30T09:10:02Z")?;
    let at = Stamp::parse(text)?;
    if at.secs() < 0 {
        return Err(format!("`at` is {text}, before 1970"));
    }
    if at.secs() > now.secs() + AHEAD_SECS {
        return Err(format!("`at` is {text}, which has not happened yet"));
    }
    Ok(at)
}

#[cfg(test)]
mod tests {
    use journals::files::Who;
    use journals::line::LineId;
    use journals::path::{Address, Below, JournalPath};
    use serde_json::json;

    use super::*;
    use crate::mcp::tools::mem::tests::mem;

    fn now() -> Stamp {
        Stamp::parse("2026-10-30T12:00:00Z").unwrap()
    }

    fn planned(v: Value) -> Result<Planned, String> {
        plan(&mem("john"), &v, now(), None).map_err(String::from)
    }

    fn op(v: Value) -> Op {
        planned(v).unwrap().op
    }

    fn path(p: &str) -> JournalPath {
        mem("john").path(p).unwrap()
    }

    fn s(x: &str) -> Slug {
        Slug::parse(x).unwrap()
    }

    fn id() -> LineId {
        LineId::parse("2026-10-30-0912-7e3a").unwrap()
    }

    /// **A note is read for the caller at now**, from the speaker it names,
    /// else the batch's.
    #[test]
    fn a_note_is_read_for_the_caller_at_now_from_its_speaker() {
        let p =
            planned(json!({"op": "note", "journal": "topics", "text": "Keep trials."})).unwrap();
        assert_eq!(
            p,
            Planned {
                op: Op::Note {
                    path: path("@zen/topics"),
                    text: "Keep trials.".into(),
                    speaker: None,
                },
                at: now()
            }
        );
        let from = |v: Value, batch: Option<&str>| {
            let b = batch.map(s);
            match plan(&mem("john"), &v, now(), b.as_ref()).unwrap().op {
                Op::Note { speaker, .. } => speaker,
                other => panic!("{other:?}"),
            }
        };
        let note = json!({"op": "note", "journal": "topics", "text": "x"});
        assert_eq!(from(note.clone(), Some("yen")), Some(s("yen")));
        let own = json!({"op": "note", "journal": "topics", "text": "x", "speaker": "@thanhlong"});
        assert_eq!(from(own, Some("yen")), Some(s("thanhlong")));
        assert_eq!(
            planned(json!({"op": "note", "journal": "topics", "text": "x", "speaker": 3}))
                .unwrap_err(),
            "`speaker` is a person's slug"
        );
    }

    /// **A remembered fact keeps when it was learned**; a time not yet come,
    /// beyond the clock skew members are allowed, or before 1970 is refused.
    #[test]
    fn at_is_when_it_happened_and_never_ahead_of_the_clock() {
        let at =
            |t: &str| planned(json!({"op": "note", "journal": "topics", "text": "x", "at": t}));
        assert_eq!(
            at("2026-05-13T09:00:00Z").unwrap().at,
            Stamp::parse("2026-05-13T09:00:00Z").unwrap()
        );
        assert_eq!(
            at("2026-10-30T12:01:00Z").unwrap().at.secs(),
            now().secs() + AHEAD_SECS
        );
        assert_eq!(
            at("2026-10-30T12:01:01Z").unwrap_err(),
            "`at` is 2026-10-30T12:01:01Z, which has not happened yet"
        );
        assert_eq!(
            at("1969-12-31T23:59:59Z").unwrap_err(),
            "`at` is 1969-12-31T23:59:59Z, before 1970"
        );
        assert_eq!(at("1970-01-01T00:00:00Z").unwrap().at.secs(), 0);
        assert_eq!(
            at("yesterday").unwrap_err(),
            Stamp::parse("yesterday").unwrap_err()
        );
        assert_eq!(
            planned(json!({"op": "note", "journal": "topics", "text": "x", "at": 5})).unwrap_err(),
            "`at` is a time written 2026-10-30T09:10:02Z"
        );
        let made = planned(json!({"op": "create", "under": "topics", "name": "trip",
            "cue": "c", "at": "2026-01-02T03:04:05Z"}))
        .unwrap();
        assert_eq!(made.at, Stamp::parse("2026-01-02T03:04:05Z").unwrap());
    }

    /// **Only a new line or journal takes a time**: an edit, a forget or a
    /// link happens to things as they stand.
    #[test]
    fn an_operation_on_what_stands_takes_no_at() {
        for (name, v) in [
            (
                "edit",
                json!({"op": "edit", "line": "topics#2026-10-30-0912-7e3a", "text": "x", "at": "2026-01-01T00:00:00Z"}),
            ),
            (
                "forget",
                json!({"op": "forget", "line": "topics#2026-10-30-0912-7e3a", "at": "2026-01-01T00:00:00Z"}),
            ),
            (
                "link",
                json!({"op": "link", "under": "people", "name": "t", "to": "topics", "cue": "c", "at": "2026-01-01T00:00:00Z"}),
            ),
            (
                "accept",
                json!({"op": "accept", "line": "topics#2026-10-30-0912-7e3a", "at": "2026-01-01T00:00:00Z"}),
            ),
        ] {
            assert_eq!(
                planned(v).unwrap_err(),
                format!("`at` is not a field of `{name}`")
            );
        }
    }

    #[test]
    fn every_operation_reads_as_its_tool_would() {
        let line = || Address {
            path: path("topics"),
            id: id(),
        };
        assert_eq!(
            op(json!({"op": "edit", "line": "topics#2026-10-30-0912-7e3a", "text": "now"})),
            Op::Edit {
                address: line(),
                text: "now".into()
            }
        );
        assert_eq!(
            op(json!({"op": "forget", "line": "topics#2026-10-30-0912-7e3a"})),
            Op::Forget {
                path: path("topics"),
                which: Which::Id(id())
            }
        );
        assert_eq!(
            op(json!({"op": "forget", "journal": "topics", "says": "Miso"})),
            Op::Forget {
                path: path("topics"),
                which: Which::Says("Miso".into())
            }
        );
        assert_eq!(
            op(
                json!({"op": "create", "under": "topics", "name": "trip", "cue": "open for the trip"})
            ),
            Op::Create {
                under: path("topics"),
                name: s("trip"),
                cue: "open for the trip".into(),
                on_taken: OnTaken::Refuse
            }
        );
        assert_eq!(
            op(
                json!({"op": "link", "under": "people", "name": "work", "to": "topics", "cue": "c"})
            ),
            Op::Link {
                under: path("people"),
                name: s("work"),
                to: path("topics"),
                cue: "c".into(),
            }
        );
        assert_eq!(
            op(json!({"op": "accept", "line": "topics#2026-10-30-0912-7e3a"})),
            Op::Accept { address: line() }
        );
    }

    /// **A request sets no focus**: that is a conversation's own, and a
    /// request could otherwise name another member's.
    #[test]
    fn a_focus_is_no_operation_of_a_request() {
        assert_eq!(
            planned(json!({"op": "focus", "journal": "conversations", "speaker": "yen"}))
                .unwrap_err(),
            "`focus` is not an operation"
        );
    }

    /// **A conversation recorded by hand is numbered when its name recurs**;
    /// anything else is refused the second time.
    #[test]
    fn a_conversation_is_created_numbered() {
        assert_eq!(
            op(
                json!({"op": "create", "under": "conversations", "name": "2026-10-30-0910-claude-code", "cue": "c"})
            ),
            Op::Create {
                under: path("conversations"),
                name: s("2026-10-30-0910-claude-code"),
                cue: "c".into(),
                on_taken: OnTaken::Number,
            }
        );
    }

    /// **Groups are managed as the tools manage them**; a group unnamed is
    /// the caller's own.
    #[test]
    fn group_operations_read_as_their_tools_do() {
        assert_eq!(
            op(
                json!({"op": "create_group", "group": "@battle-cities", "name": "Battle Cities", "about": "The game."})
            ),
            Op::CreateGroup {
                group: s("battle-cities"),
                name: "Battle Cities".into(),
                about: Some("The game.".into()),
            }
        );
        assert_eq!(
            op(json!({"op": "add_bot", "bot": "c3f1a07e", "reads": ["topics"]})),
            Op::AddBot {
                group: s("zen"),
                bot: s("c3f1a07e"),
                reads: vec![Below::parse("topics").unwrap()],
            }
        );
        assert_eq!(
            op(json!({"op": "remove_member", "bot": "c3f1a07e", "group": "@zen"})),
            Op::RemoveMember {
                group: s("zen"),
                who: Who::Bot(s("c3f1a07e")),
            }
        );
        assert_eq!(
            op(json!({"op": "choose_default", "group": "@battle-cities"})),
            Op::ChooseDefault {
                group: s("battle-cities"),
            }
        );
        assert!(matches!(
            op(json!({"op": "add_member", "person": "friend", "name": "Friend"})),
            Op::AddMember { .. }
        ));
        assert_eq!(
            planned(json!({"op": "add_bot", "bot": "b1", "role": "owner"})).unwrap_err(),
            "`role` is not a field of `add_bot`"
        );
        assert_eq!(
            planned(json!({"op": "add_member", "bot": "b1"})).unwrap_err(),
            "`bot` is not a field of `add_member`"
        );
        assert_eq!(
            planned(json!({"op": "create_group", "group": "g", "name": "G", "kind": "clan"}))
                .unwrap_err(),
            "`kind` is not a field of `create_group`"
        );
    }

    #[test]
    fn a_malformed_operation_says_what_is_wrong() {
        let m = mem("john");
        let err = |v: Value| planned(v).unwrap_err();
        assert_eq!(err(json!([1])), "an operation is a JSON object");
        assert_eq!(
            err(json!({"op": "teleport"})),
            "`teleport` is not an operation"
        );
        assert_eq!(
            err(json!({"op": "fold", "journal": "topics"})),
            "`fold` is not an operation",
            "the structure is coding's"
        );
        assert_eq!(
            err(json!({"op": "note", "journal": "topics", "txt": "x"})),
            "`txt` is not a field of `note`"
        );
        assert_eq!(
            err(json!({"op": "note", "journal": "topics"})),
            "`text` is missing: it is a non-empty string"
        );
        let either = "`forget` names a `line`, or a `journal` and what it `says`";
        for v in [
            json!({"op": "forget", "journal": "topics"}),
            json!({"op": "forget", "says": "x"}),
            json!({"op": "forget", "line": "topics#2026-10-30-0912-7e3a", "says": "x"}),
            json!({"op": "forget", "line": "topics#2026-10-30-0912-7e3a", "journal": "topics"}),
        ] {
            assert_eq!(err(v), either);
        }
        assert_eq!(
            err(json!({"op": "edit", "line": "topics", "text": "x"})),
            m.address("topics").unwrap_err()
        );
        assert_eq!(
            err(json!({"op": "note", "journal": "nowhere", "text": "x"})),
            m.path("nowhere").unwrap_err()
        );
        assert_eq!(
            err(json!({"op": "create", "under": "topics", "name": "Not A Slug", "cue": "c"})),
            Slug::parse("Not A Slug").unwrap_err()
        );
        let solo = mem("solo");
        assert_eq!(
            plan(
                &solo,
                &json!({"op": "note", "journal": "topics", "text": "x"}),
                now(),
                None
            )
            .unwrap_err(),
            solo.find("topics").unwrap_err()
        );
        assert!(matches!(
            plan(
                &m,
                &json!({"op": "note", "journal": "topics/none", "text": "x"}),
                now(),
                None
            ),
            Err(Miss::Absent(_))
        ));
    }
}
