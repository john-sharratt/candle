//! An operation and its actor as JSON, carried in the commit that made them
//! (`Journal-Op`, `Journal-Actor`). A commit not yet on origin is replayed
//! from them onto what origin holds (§6.3), and history reads as what each
//! change meant, not only what it did to the files.
//!
//! Paths are written as folders — `groups/zen/topics/permits` — which name a
//! journal whoever reads them.

use serde_json::{json, Map, Value};

use crate::actor::Actor;
use crate::error::{Error, Result};
use crate::files::summary::{Cite, Fingerprint, Sentence};
use crate::files::{AccountKey, Email, Role, Who};
use crate::line::LineId;
use crate::ops::{OnTaken, Op, Which};
use crate::path::{Address, Below, JournalPath};
use crate::period::Period;
use crate::scope::Channel;
use crate::slug::Slug;

pub fn path_to_wire(p: &JournalPath) -> String {
    p.folder()
}

pub fn path_from_wire(s: &str) -> Result<JournalPath> {
    let mut parts = s.split('/');
    if parts.next() != Some("groups") {
        return Err(Error::invalid(format!("`{s}` is not a journal's folder")));
    }
    let group = Slug::parse(parts.next().unwrap_or_default()).map_err(Error::Invalid)?;
    let names = parts
        .map(|n| Slug::parse(n).map_err(Error::Invalid))
        .collect::<Result<Vec<_>>>()?;
    JournalPath::new(group, names)
}

pub fn actor_to_wire(a: &Actor) -> Value {
    match a {
        Actor::Person {
            person,
            channel,
            admin,
        } => {
            json!({"kind": "person", "person": person.as_str(), "channel": channel.name(), "admin": admin})
        }
        Actor::Server { person, channel } => {
            json!({"kind": "server", "person": person.as_str(), "channel": channel.name()})
        }
        Actor::Zend => json!({"kind": "zend"}),
        Actor::Worker { person } => json!({"kind": "worker", "person": person.as_str()}),
        Actor::Npcd { person } => json!({"kind": "npcd", "person": person.as_str()}),
        Actor::Bot { bot } => json!({"kind": "bot", "bot": bot.as_str()}),
        Actor::Guest { channel } => json!({"kind": "guest", "channel": channel.name()}),
        Actor::System => json!({"kind": "system"}),
    }
}

struct Obj<'a>(&'a Map<String, Value>);

impl<'a> Obj<'a> {
    fn of(v: &'a Value) -> Result<Obj<'a>> {
        v.as_object()
            .map(Obj)
            .ok_or_else(|| Error::invalid("expected a JSON object"))
    }

    fn str(&self, k: &str) -> Result<&'a str> {
        self.0
            .get(k)
            .and_then(Value::as_str)
            .ok_or_else(|| Error::invalid(format!("`{k}` is missing or not text")))
    }

    fn bool(&self, k: &str) -> Result<bool> {
        self.0
            .get(k)
            .and_then(Value::as_bool)
            .ok_or_else(|| Error::invalid(format!("`{k}` is missing or not true or false")))
    }

    fn opt_str(&self, k: &str) -> Result<Option<&'a str>> {
        match self.0.get(k) {
            None | Some(Value::Null) => Ok(None),
            Some(Value::String(s)) => Ok(Some(s)),
            Some(_) => Err(Error::invalid(format!("`{k}` is not text"))),
        }
    }

    fn strs(&self, k: &str) -> Result<Vec<&'a str>> {
        self.0
            .get(k)
            .and_then(Value::as_array)
            .ok_or_else(|| Error::invalid(format!("`{k}` is missing or not a list")))?
            .iter()
            .map(|v| {
                v.as_str()
                    .ok_or_else(|| Error::invalid(format!("`{k}` lists something not text")))
            })
            .collect()
    }

    fn opt_strs(&self, k: &str) -> Result<Option<Vec<&'a str>>> {
        match self.0.get(k) {
            None | Some(Value::Null) => Ok(None),
            Some(_) => self.strs(k).map(Some),
        }
    }

    fn path(&self, k: &str) -> Result<JournalPath> {
        path_from_wire(self.str(k)?)
    }

    fn slug(&self, k: &str) -> Result<Slug> {
        Slug::parse(self.str(k)?).map_err(Error::Invalid)
    }

    fn opt_slug(&self, k: &str) -> Result<Option<Slug>> {
        self.opt_str(k)?
            .map(|s| Slug::parse(s).map_err(Error::Invalid))
            .transpose()
    }

    fn id(&self, k: &str) -> Result<LineId> {
        LineId::parse(self.str(k)?).map_err(Error::Invalid)
    }

    fn channel(&self) -> Result<Channel> {
        let c = self.str("channel")?;
        Channel::parse(c).ok_or_else(|| Error::invalid(format!("`{c}` is not a channel")))
    }
}

pub fn actor_from_wire(v: &Value) -> Result<Actor> {
    let o = Obj::of(v)?;
    Ok(match o.str("kind")? {
        "person" => Actor::Person {
            person: o.slug("person")?,
            channel: o.channel()?,
            admin: o.bool("admin")?,
        },
        "server" => Actor::Server {
            person: o.slug("person")?,
            channel: o.channel()?,
        },
        "zend" => Actor::Zend,
        "worker" => Actor::Worker {
            person: o.slug("person")?,
        },
        "npcd" => Actor::Npcd {
            person: o.slug("person")?,
        },
        "bot" => Actor::Bot {
            bot: o.slug("bot")?,
        },
        "guest" => Actor::Guest {
            channel: o.channel()?,
        },
        "system" => Actor::System,
        k => return Err(Error::invalid(format!("`{k}` is not an actor"))),
    })
}

pub fn op_to_wire(op: &Op) -> Value {
    let p = path_to_wire;
    let slug = |s: &Option<Slug>| s.as_ref().map(Slug::as_str).map(str::to_string);
    match op {
        Op::Note {
            path,
            text,
            speaker,
        } => json!({"op": "note", "path": p(path), "text": text, "speaker": slug(speaker)}),
        Op::Episode {
            path,
            text,
            speaker,
        } => json!({"op": "episode", "path": p(path), "text": text, "speaker": slug(speaker)}),
        Op::Edit { address, text } => {
            json!({"op": "edit", "path": p(&address.path), "id": address.id.as_str(), "text": text})
        }
        Op::Forget {
            path,
            which: Which::Id(id),
        } => json!({"op": "forget", "path": p(path), "id": id.as_str()}),
        Op::Forget {
            path,
            which: Which::Says(s),
        } => json!({"op": "forget", "path": p(path), "says": s}),
        Op::Accept { address } => {
            json!({"op": "accept", "path": p(&address.path), "id": address.id.as_str()})
        }
        Op::Create {
            under,
            name,
            cue,
            on_taken,
        } => json!({
            "op": "create", "under": p(under), "name": name.as_str(), "cue": cue,
            "on_taken": match on_taken { OnTaken::Number => "number", OnTaken::Refuse => "refuse" },
        }),
        Op::Link {
            under,
            name,
            to,
            cue,
        } => {
            json!({"op": "link", "under": p(under), "name": name.as_str(), "to": p(to), "cue": cue})
        }
        Op::Focus {
            path,
            speaker,
            about,
        } => json!({
            "op": "focus", "path": p(path), "speaker": slug(speaker),
            "about": about.as_ref().map(|a| a.iter().map(p).collect::<Vec<_>>()),
        }),
        Op::Close { path } => json!({"op": "close", "path": p(path)}),
        Op::Summarize {
            path,
            over,
            sentences,
            digest,
            cue,
        } => json!({
            "op": "summarize", "path": p(path), "over": over.as_str(),
            "sentences": sentences.iter().map(|s| json!({
                "text": s.text,
                "cites": s.cites.iter().map(Cite::to_string).collect::<Vec<_>>(),
            })).collect::<Vec<_>>(),
            "digest": digest, "cue": cue,
        }),
        Op::MarkStale { path, why } => json!({"op": "mark_stale", "path": p(path), "why": why}),
        Op::FoldPeriod {
            path,
            period,
            ids,
            over,
            text,
        } => json!({
            "op": "fold_period", "path": p(path), "period": period.to_string(),
            "ids": ids.iter().map(LineId::as_str).collect::<Vec<_>>(), "over": over.as_str(),
            "text": text,
        }),
        Op::CreateGroup { group, name, about } => {
            json!({"op": "create_group", "group": group.as_str(), "name": name, "about": about})
        }
        Op::DescribeGroup { group, about } => {
            json!({"op": "describe_group", "group": group.as_str(), "about": about})
        }
        Op::AddMember {
            group,
            person,
            name,
            emails,
            role,
        } => json!({
            "op": "add_member", "group": group.as_str(), "person": person.as_str(), "name": name,
            "emails": emails.iter().map(Email::as_str).collect::<Vec<_>>(), "role": role.name(),
        }),
        Op::ChooseDefault { group } => json!({"op": "choose_default", "group": group.as_str()}),
        Op::AddBot { group, bot, reads } => json!({
            "op": "add_bot", "group": group.as_str(), "bot": bot.as_str(),
            "reads": reads.iter().map(Below::to_string).collect::<Vec<_>>(),
        }),
        Op::RemoveMember {
            group,
            who: Who::Person(person),
        } => {
            json!({"op": "remove_member", "group": group.as_str(), "person": person.as_str()})
        }
        Op::RemoveMember {
            group,
            who: Who::Bot(bot),
        } => json!({"op": "remove_member", "group": group.as_str(), "bot": bot.as_str()}),
        Op::Bind { person, email, key } => {
            json!({"op": "bind", "person": person.as_str(), "email": email.as_str(), "key": key.as_str()})
        }
    }
}

pub fn op_from_wire(v: &Value) -> Result<Op> {
    let o = Obj::of(v)?;
    let address = || -> Result<Address> {
        Ok(Address {
            path: o.path("path")?,
            id: o.id("id")?,
        })
    };
    let text = || o.str("text").map(str::to_string);
    Ok(match o.str("op")? {
        "note" => Op::Note {
            path: o.path("path")?,
            text: text()?,
            speaker: o.opt_slug("speaker")?,
        },
        "episode" => Op::Episode {
            path: o.path("path")?,
            text: text()?,
            speaker: o.opt_slug("speaker")?,
        },
        "edit" => Op::Edit {
            address: address()?,
            text: text()?,
        },
        "forget" => Op::Forget {
            path: o.path("path")?,
            which: match o.opt_str("says")? {
                Some(s) => Which::Says(s.into()),
                None => Which::Id(o.id("id")?),
            },
        },
        "accept" => Op::Accept {
            address: address()?,
        },
        "create" => Op::Create {
            under: o.path("under")?,
            name: o.slug("name")?,
            cue: o.str("cue")?.into(),
            on_taken: match o.str("on_taken")? {
                "number" => OnTaken::Number,
                "refuse" => OnTaken::Refuse,
                t => return Err(Error::invalid(format!("`{t}` is not number or refuse"))),
            },
        },
        "link" => Op::Link {
            under: o.path("under")?,
            name: o.slug("name")?,
            to: o.path("to")?,
            cue: o.str("cue")?.into(),
        },
        "focus" => Op::Focus {
            path: o.path("path")?,
            speaker: o.opt_slug("speaker")?,
            about: o
                .opt_strs("about")?
                .map(|a| {
                    a.into_iter()
                        .map(path_from_wire)
                        .collect::<Result<Vec<_>>>()
                })
                .transpose()?,
        },
        "close" => Op::Close {
            path: o.path("path")?,
        },
        "summarize" => {
            let sentences =
                o.0.get("sentences")
                    .and_then(Value::as_array)
                    .ok_or_else(|| Error::invalid("`sentences` is missing"))?
                    .iter()
                    .map(|s| {
                        let so = Obj::of(s)?;
                        Ok(Sentence {
                            text: so.str("text")?.into(),
                            cites: so
                                .strs("cites")?
                                .into_iter()
                                .map(|c| Cite::parse(c).map_err(Error::Invalid))
                                .collect::<Result<_>>()?,
                        })
                    })
                    .collect::<Result<Vec<_>>>()?;
            Op::Summarize {
                path: o.path("path")?,
                over: Fingerprint::parse(o.str("over")?).map_err(Error::Invalid)?,
                sentences,
                digest: o.opt_str("digest")?.map(Into::into),
                cue: o.opt_str("cue")?.map(Into::into),
            }
        }
        "mark_stale" => Op::MarkStale {
            path: o.path("path")?,
            why: o.str("why")?.into(),
        },
        "fold_period" => Op::FoldPeriod {
            path: o.path("path")?,
            period: Period::parse(o.str("period")?).map_err(Error::Invalid)?,
            ids: o
                .strs("ids")?
                .into_iter()
                .map(|i| LineId::parse(i).map_err(Error::Invalid))
                .collect::<Result<_>>()?,
            over: Fingerprint::parse(o.str("over")?).map_err(Error::Invalid)?,
            text: text()?,
        },
        "create_group" => Op::CreateGroup {
            group: o.slug("group")?,
            name: o.str("name")?.into(),
            about: o.opt_str("about")?.map(Into::into),
        },
        "describe_group" => Op::DescribeGroup {
            group: o.slug("group")?,
            about: o.str("about")?.into(),
        },
        "add_member" => Op::AddMember {
            group: o.slug("group")?,
            person: o.slug("person")?,
            name: o.opt_str("name")?.map(Into::into),
            emails: o
                .strs("emails")?
                .into_iter()
                .map(|e| Email::parse(e).map_err(Error::Invalid))
                .collect::<Result<_>>()?,
            role: Role::parse(o.str("role")?).map_err(Error::Invalid)?,
        },
        "choose_default" => Op::ChooseDefault {
            group: o.slug("group")?,
        },
        "add_bot" => Op::AddBot {
            group: o.slug("group")?,
            bot: o.slug("bot")?,
            reads: o
                .strs("reads")?
                .into_iter()
                .map(|r| Below::parse(r).map_err(Error::Invalid))
                .collect::<Result<_>>()?,
        },
        "remove_member" => Op::RemoveMember {
            group: o.slug("group")?,
            who: match (o.opt_str("bot")?, o.opt_str("person")?) {
                (Some(_), None) => Who::Bot(o.slug("bot")?),
                (None, Some(_)) => Who::Person(o.slug("person")?),
                _ => {
                    return Err(Error::invalid(
                        "`remove_member` names a `person` or a `bot`, one of them",
                    ))
                }
            },
        },
        "bind" => Op::Bind {
            person: o.slug("person")?,
            email: Email::parse(o.str("email")?).map_err(Error::Invalid)?,
            key: AccountKey::parse(o.str("key")?).map_err(Error::Invalid)?,
        },
        k => return Err(Error::invalid(format!("`{k}` is not an operation"))),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ops::fixture::p;
    use crate::testing::slug;

    #[test]
    fn every_operation_round_trips() {
        let id = LineId::parse("2026-10-30-0912-7e3a").unwrap();
        let ops = vec![
            Op::Note {
                path: p("people/yen"),
                text: "a \"quoted\"\nline".into(),
                speaker: Some(slug("yen")),
            },
            Op::Note {
                path: p("@zen"),
                text: "x".into(),
                speaker: None,
            },
            Op::Episode {
                path: p("conversations/2026-10-31-0712-grok"),
                text: "x".into(),
                speaker: None,
            },
            Op::Edit {
                address: Address {
                    path: p("topics"),
                    id: id.clone(),
                },
                text: "y".into(),
            },
            Op::Forget {
                path: p("topics"),
                which: Which::Id(id.clone()),
            },
            Op::Forget {
                path: p("topics"),
                which: Which::Says("tea".into()),
            },
            Op::Accept {
                address: Address {
                    path: p("topics"),
                    id: id.clone(),
                },
            },
            Op::Create {
                under: p("topics"),
                name: slug("trip"),
                cue: "c".into(),
                on_taken: OnTaken::Refuse,
            },
            Op::Create {
                under: p("conversations"),
                name: slug("2026-10-31-0712-grok.2"),
                cue: "c".into(),
                on_taken: OnTaken::Number,
            },
            Op::Link {
                under: p("topics/candle"),
                name: slug("wasmer"),
                to: p("topics/wasmer"),
                cue: "c".into(),
            },
            Op::Focus {
                path: p("conversations/c"),
                speaker: Some(slug("yen")),
                about: None,
            },
            Op::Focus {
                path: p("conversations/c"),
                speaker: None,
                about: Some(vec![p("topics/candle"), p("people/yen")]),
            },
            Op::Close {
                path: p("threads/t"),
            },
            Op::Summarize {
                path: p("@zen"),
                over: Fingerprint::of_bytes(b"x"),
                sentences: vec![Sentence {
                    text: "S.".into(),
                    cites: vec![Cite::Line(id.clone()), Cite::Child(slug("topics"))],
                }],
                digest: Some("d".into()),
                cue: None,
            },
            Op::MarkStale {
                path: p("@zen"),
                why: "over its cap: \"x\"\nthen".into(),
            },
            Op::FoldPeriod {
                path: p("@zen"),
                period: Period::parse("2026-W44").unwrap(),
                ids: vec![id.clone()],
                over: Fingerprint::of_bytes(b"w"),
                text: "w".into(),
            },
            Op::CreateGroup {
                group: slug("battle-cities"),
                name: "Battle Cities".into(),
                about: Some("The game's world.".into()),
            },
            Op::CreateGroup {
                group: slug("x"),
                name: "X".into(),
                about: None,
            },
            Op::DescribeGroup {
                group: slug("battle-cities"),
                about: "The game's world.".into(),
            },
            Op::AddMember {
                group: slug("zen"),
                person: slug("yen"),
                name: Some("Yen".into()),
                emails: vec![Email::parse("yen@example.com").unwrap()],
                role: Role::Owner,
            },
            Op::AddBot {
                group: slug("battle-cities"),
                bot: slug("c3f1a07e"),
                reads: vec![Below::parse("topics/canon").unwrap()],
            },
            Op::ChooseDefault {
                group: slug("battle-cities"),
            },
            Op::RemoveMember {
                group: slug("zen"),
                who: Who::Person(slug("yen")),
            },
            Op::RemoveMember {
                group: slug("battle-cities"),
                who: Who::Bot(slug("c3f1a07e")),
            },
            Op::Bind {
                person: slug("yen"),
                email: Email::parse("yen@example.com").unwrap(),
                key: AccountKey::parse("3b1f09c2d47e8a60").unwrap(),
            },
        ];
        for op in ops {
            let wire = op_to_wire(&op);
            let text = serde_json::to_string(&wire).unwrap();
            assert!(!text.contains('\n'), "a trailer is one line: {text}");
            let back: Value = serde_json::from_str(&text).unwrap();
            assert_eq!(op_from_wire(&back).unwrap(), op, "{text}");
        }
    }

    #[test]
    fn every_actor_round_trips() {
        for a in [
            Actor::Person {
                person: slug("yen"),
                channel: Channel::Grok,
                admin: false,
            },
            Actor::Person {
                person: slug("john"),
                channel: Channel::ClaudeCode,
                admin: true,
            },
            Actor::Server {
                person: slug("john"),
                channel: Channel::Other,
            },
            Actor::Zend,
            Actor::Worker {
                person: slug("john"),
            },
            Actor::Npcd {
                person: slug("thanhlong"),
            },
            Actor::Bot {
                bot: slug("c3f1a07e"),
            },
            Actor::Guest {
                channel: Channel::Other,
            },
            Actor::System,
        ] {
            assert_eq!(actor_from_wire(&actor_to_wire(&a)).unwrap(), a);
        }
    }

    #[test]
    fn malformed_wire_is_an_error() {
        for bad in [
            json!(1),
            json!({"op": "teleport"}),
            json!({"op": "note", "path": "elsewhere/x", "text": "t"}),
            json!({"op": "note", "path": "persons/john", "text": "t"}),
            json!({"op": "note", "path": "groups/zen/projects", "text": "t"}),
            json!({"op": "note", "path": "groups/zen"}),
            json!({"op": "create", "under": "groups/zen/topics", "name": "x", "cue": "c", "on_taken": "maybe"}),
            json!({"op": "fold", "path": "groups/zen/topics/x"}),
            json!({"op": "merge", "from": "groups/zen/topics/a", "into": "groups/zen/topics/b"}),
            json!({"op": "share", "from": "groups/zen/topics/a", "into": "groups/zen/topics/b"}),
            json!({"op": "choose_groups", "framed": []}),
            json!({"op": "mark_stale", "path": "groups/zen"}),
            json!({"op": "add_bot", "group": "g", "bot": "b1"}),
            json!({"op": "focus", "path": "groups/zen/conversations/c", "about": ["topics/x"]}),
            json!({"op": "remove_member", "group": "g", "person": "yen", "bot": "b1"}),
            json!({"op": "remove_member", "group": "g"}),
        ] {
            assert!(op_from_wire(&bad).is_err(), "{bad}");
        }
        assert!(actor_from_wire(
            &json!({"kind": "person", "person": "john", "channel": "tesla", "admin": false})
        )
        .is_err());
        assert_eq!(
            actor_from_wire(&json!({"kind": "person", "person": "john", "channel": "grok"}))
                .unwrap_err()
                .to_string(),
            "`admin` is missing or not true or false"
        );
        assert_eq!(path_from_wire("groups/zen").unwrap().show(), "@zen");
    }
}
