//! The operations `POST /v1/journals` takes. Each is a JSON object naming its
//! journals by memory path, as the tools write them — `~/…`, `@family/…`, a
//! bare section — read for the caller into the journals crate's [`Op`] and
//! the time it is recorded at.
//!
//! | `op` | Fields |
//! |---|---|
//! | `note` | `journal`, `text`, `at`? |
//! | `edit` | `line` (`<path>#<line id>`), `text` |
//! | `forget` | `line`; or `journal` and `says` |
//! | `create` | `under`, `name`, `cue`, `at`? |
//! | `fold`, `unfold` | `journal` |
//! | `merge` | `journal`, `into` |
//! | `share` | `journal`, `into` (`topics` when absent), `name`? |
//! | `accept` | `line` |
//!
//! `at`, `2026-10-30T09:10:02Z`, is when a note or a journal is recorded as
//! made — the day a remembered fact was learned, for one written down later.
//! It is the line's time and its commit's author time; the commit itself is
//! dated when the service makes it. Never before 1970, nor ahead of this
//! member's clock by more than [`AHEAD_SECS`]. The other operations change a
//! line or a journal as it stands, now, and take no `at`. A field an operation
//! does not take is refused, so a misspelt one is never silently dropped.

use journals::ops::{OnTaken, Op, Which};
use journals::slug::Slug;
use journals::time::Stamp;
use serde_json::{Map, Value};

use crate::mcp::tools::mem::Mem;
use crate::mcp::tools::string;

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
    Fold,
    Unfold,
    Merge,
    Share,
    Accept,
}

impl Kind {
    fn parse(name: &str) -> Option<Kind> {
        Some(match name {
            "note" => Kind::Note,
            "edit" => Kind::Edit,
            "forget" => Kind::Forget,
            "create" => Kind::Create,
            "fold" => Kind::Fold,
            "unfold" => Kind::Unfold,
            "merge" => Kind::Merge,
            "share" => Kind::Share,
            "accept" => Kind::Accept,
            _ => return None,
        })
    }

    /// The fields it takes beside `op`.
    fn fields(self) -> &'static [&'static str] {
        match self {
            Kind::Note => &["journal", "text", "at"],
            Kind::Edit => &["line", "text"],
            Kind::Forget => &["line", "journal", "says"],
            Kind::Create => &["under", "name", "cue", "at"],
            Kind::Fold | Kind::Unfold => &["journal"],
            Kind::Merge => &["journal", "into"],
            Kind::Share => &["journal", "into", "name"],
            Kind::Accept => &["line"],
        }
    }
}

/// `v` read for `m`'s caller at `now`.
pub fn plan(m: &Mem, v: &Value, now: Stamp) -> Result<Planned, String> {
    let obj = v.as_object().ok_or("an operation is a JSON object")?;
    let name = string(v, "op")?;
    let kind = Kind::parse(name).ok_or_else(|| format!("`{name}` is not an operation"))?;
    if let Some(k) = obj
        .keys()
        .find(|k| k.as_str() != "op" && !kind.fields().contains(&k.as_str()))
    {
        return Err(format!("`{k}` is not a field of `{name}`"));
    }
    let at = at_of(obj, now)?;
    let path = |k: &str| string(v, k).and_then(|p| m.path(p));
    let address = || string(v, "line").and_then(|a| m.address(a));
    let text = || string(v, "text").map(str::to_string);
    let op = match kind {
        Kind::Note => Op::Note {
            path: path("journal")?,
            text: text()?,
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
        Kind::Create => Op::Create {
            under: path("under")?,
            name: Slug::parse(string(v, "name")?)?,
            cue: string(v, "cue")?.to_string(),
            on_taken: OnTaken::Refuse,
        },
        Kind::Fold => Op::Fold {
            path: path("journal")?,
        },
        Kind::Unfold => Op::Unfold {
            path: path("journal")?,
        },
        Kind::Merge => Op::Merge {
            from: path("journal")?,
            into: path("into")?,
        },
        Kind::Share => Op::Share {
            from: path("journal")?,
            into: match obj.get("into") {
                None => m.path("topics")?,
                Some(_) => path("into")?,
            },
            name: match obj.get("name") {
                None => None,
                Some(_) => Some(Slug::parse(string(v, "name")?)?),
            },
        },
        Kind::Accept => Op::Accept {
            address: address()?,
        },
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
    use journals::line::LineId;
    use journals::path::{Address, JournalPath};
    use serde_json::json;

    use super::*;
    use crate::mcp::tools::mem::tests::mem;

    fn now() -> Stamp {
        Stamp::parse("2026-10-30T12:00:00Z").unwrap()
    }

    fn planned(v: Value) -> Result<Planned, String> {
        plan(&mem("john"), &v, now())
    }

    fn op(v: Value) -> Op {
        planned(v).unwrap().op
    }

    fn path(p: &str) -> JournalPath {
        mem("john").path(p).unwrap()
    }

    fn id() -> LineId {
        LineId::parse("20261030-0912-7e3a").unwrap()
    }

    #[test]
    fn a_note_is_read_for_the_caller_at_now() {
        let p =
            planned(json!({"op": "note", "journal": "projects/candle", "text": "Keep trials."}))
                .unwrap();
        assert_eq!(
            p,
            Planned {
                op: Op::Note {
                    path: path("@sharratt/projects/candle"),
                    text: "Keep trials.".into()
                },
                at: now()
            }
        );
    }

    /// **A remembered fact keeps when it was learned**; a time not yet come,
    /// beyond the clock skew members are allowed, or before 1970 is refused.
    #[test]
    fn at_is_when_it_happened_and_never_ahead_of_the_clock() {
        let at = |t: &str| planned(json!({"op": "note", "journal": "~", "text": "x", "at": t}));
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
            planned(json!({"op": "note", "journal": "~", "text": "x", "at": 5})).unwrap_err(),
            "`at` is a time written 2026-10-30T09:10:02Z"
        );
        let made = planned(json!({"op": "create", "under": "topics", "name": "trip",
            "cue": "c", "at": "2026-01-02T03:04:05Z"}))
        .unwrap();
        assert_eq!(made.at, Stamp::parse("2026-01-02T03:04:05Z").unwrap());
    }

    /// **Only a new line or journal takes a time**: an edit, a forget or a
    /// fold happens to things as they stand.
    #[test]
    fn an_operation_on_what_stands_takes_no_at() {
        for (name, v) in [
            (
                "edit",
                json!({"op": "edit", "line": "~#20261030-0912-7e3a", "text": "x", "at": "2026-01-01T00:00:00Z"}),
            ),
            (
                "forget",
                json!({"op": "forget", "line": "~#20261030-0912-7e3a", "at": "2026-01-01T00:00:00Z"}),
            ),
            (
                "fold",
                json!({"op": "fold", "journal": "topics/trip", "at": "2026-01-01T00:00:00Z"}),
            ),
            (
                "merge",
                json!({"op": "merge", "journal": "topics/a", "into": "topics/b", "at": "2026-01-01T00:00:00Z"}),
            ),
            (
                "accept",
                json!({"op": "accept", "line": "~#20261030-0912-7e3a", "at": "2026-01-01T00:00:00Z"}),
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
            path: path("~"),
            id: id(),
        };
        assert_eq!(
            op(json!({"op": "edit", "line": "~#20261030-0912-7e3a", "text": "now"})),
            Op::Edit {
                address: line(),
                text: "now".into()
            }
        );
        assert_eq!(
            op(json!({"op": "forget", "line": "~#20261030-0912-7e3a"})),
            Op::Forget {
                path: path("~"),
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
                name: Slug::parse("trip").unwrap(),
                cue: "open for the trip".into(),
                on_taken: OnTaken::Refuse
            }
        );
        assert_eq!(
            op(json!({"op": "fold", "journal": "topics/trip"})),
            Op::Fold {
                path: path("topics/trip")
            }
        );
        assert_eq!(
            op(json!({"op": "unfold", "journal": "topics/trip"})),
            Op::Unfold {
                path: path("topics/trip")
            }
        );
        assert_eq!(
            op(json!({"op": "merge", "journal": "topics/a", "into": "topics/b"})),
            Op::Merge {
                from: path("topics/a"),
                into: path("topics/b")
            }
        );
        assert_eq!(
            op(json!({"op": "share", "journal": "~/people/builder"})),
            Op::Share {
                from: path("~/people/builder"),
                into: path("topics"),
                name: None
            }
        );
        assert_eq!(
            op(
                json!({"op": "share", "journal": "~/people/builder", "into": "projects", "name": "build"})
            ),
            Op::Share {
                from: path("~/people/builder"),
                into: path("projects"),
                name: Some(Slug::parse("build").unwrap())
            }
        );
        assert_eq!(
            op(json!({"op": "accept", "line": "~#20261030-0912-7e3a"})),
            Op::Accept { address: line() }
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
            err(json!({"op": "note", "journal": "~", "txt": "x"})),
            "`txt` is not a field of `note`"
        );
        assert_eq!(
            err(json!({"op": "note", "journal": "~"})),
            "`text` is missing: it is a non-empty string"
        );
        let either = "`forget` names a `line`, or a `journal` and what it `says`";
        for v in [
            json!({"op": "forget", "journal": "~"}),
            json!({"op": "forget", "says": "x"}),
            json!({"op": "forget", "line": "~#20261030-0912-7e3a", "says": "x"}),
            json!({"op": "forget", "line": "~#20261030-0912-7e3a", "journal": "~"}),
        ] {
            assert_eq!(err(v), either);
        }
        assert_eq!(
            err(json!({"op": "edit", "line": "~", "text": "x"})),
            m.address("~").unwrap_err()
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
                now()
            )
            .unwrap_err(),
            solo.path("topics").unwrap_err()
        );
    }
}
