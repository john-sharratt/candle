//! A conversation's focus (§2.4): who it has worked out is speaking, and the
//! journals it is about. Every later call's frame starts from it, and notes
//! written in the conversation record its speaker.

use crate::error::{Error, Result};
use crate::files::focus::MAX_ABOUT;
use crate::files::Focus;
use crate::ops::Applied;
use crate::path::{Below, JournalPath};
use crate::slug::Slug;
use crate::tree::Tree;

/// `path`'s focus, with `speaker` and `about` in place of what it had where
/// they are given.
pub fn focus(
    tree: &mut Tree,
    path: &JournalPath,
    speaker: Option<&Slug>,
    about: Option<&[JournalPath]>,
) -> Result<Applied> {
    if !path.place()?.is_conversation() {
        return Err(Error::invalid(format!(
            "{path} is no conversation: only a conversation has a focus"
        )));
    }
    if speaker.is_none() && about.is_none() {
        return Err(Error::invalid(
            "a focus names a speaker, what it is about, or both",
        ));
    }
    let group = path.group();
    if let Some(s) = speaker {
        if tree.get(&JournalPath::person(group, s)?).is_none() {
            return Err(Error::not_found(format!(
                "{s} has no journal under {group}'s people: make one first, with its cue"
            )));
        }
    }
    let about = match about {
        None => None,
        Some(paths) => {
            if paths.len() > MAX_ABOUT {
                return Err(Error::invalid(format!(
                    "a conversation is about at most {MAX_ABOUT} journals at once, not {}",
                    paths.len()
                )));
            }
            let mut out: Vec<Below> = Vec::new();
            for p in paths {
                if p.group() != group {
                    return Err(Error::invalid(format!(
                        "{p} is not in @{group}: a conversation is about its own group's journals"
                    )));
                }
                if p.is_root() {
                    return Err(Error::invalid(format!(
                        "{p} opens every frame already: a conversation is about journals below it"
                    )));
                }
                tree.need(p)?;
                let below = Below::from(p);
                if !out.contains(&below) {
                    out.push(below);
                }
            }
            Some(out)
        }
    };
    let j = tree.need_mut(path)?;
    j.writable(&path.show())?;
    let meta = j
        .meta
        .as_mut()
        .ok_or_else(|| Error::not_found(format!("{path} has no journal.yaml")))?;
    let mut f = meta.focus.clone().unwrap_or_default();
    if let Some(s) = speaker {
        f.speaker = Some(s.clone());
    }
    if let Some(a) = about {
        f.about = a;
    }
    let said = describe(&f);
    meta.focus = Some(f);
    Ok(Applied::touching(
        format!("Focus of {path}: {said}"),
        vec![path.clone()],
    ))
}

/// A focus as a commit subject says it: "yen; about topics/candle".
fn describe(f: &Focus) -> String {
    let mut parts = Vec::new();
    if let Some(s) = &f.speaker {
        parts.push(format!("{s} speaking"));
    }
    if !f.about.is_empty() {
        let about: Vec<String> = f.about.iter().map(Below::to_string).collect();
        parts.push(format!("about {}", about.join(", ")));
    }
    parts.join("; ")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::actor::Actor;
    use crate::ops::fixture::{john, now, p, zen};
    use crate::ops::{apply, OnTaken, Op};
    use crate::scope::Channel;
    use crate::testing::slug;

    fn with_conversation() -> (Tree, JournalPath) {
        let mut t = zen();
        let server = Actor::Server {
            person: slug("john"),
            channel: Channel::Grok,
        };
        let conv = apply(
            &mut t,
            &server,
            &Op::Create {
                under: p("conversations"),
                name: slug("2026-10-31-0712-grok"),
                cue: "open for a drive".into(),
                on_taken: OnTaken::Number,
            },
            now(),
        )
        .unwrap()
        .created
        .unwrap();
        for (under, name) in [
            ("people", "yen"),
            ("topics", "house"),
            ("topics/house", "permits"),
        ] {
            apply(
                &mut t,
                &john(),
                &Op::Create {
                    under: p(under),
                    name: slug(name),
                    cue: format!("open for {name}"),
                    on_taken: OnTaken::Refuse,
                },
                now(),
            )
            .unwrap();
        }
        (t, conv)
    }

    /// **A conversation narrows down**: who is speaking, then what about,
    /// each kept as the other is set.
    #[test]
    fn a_focus_is_set_a_part_at_a_time() {
        let (mut t, conv) = with_conversation();
        let op = |speaker: Option<&str>, about: Option<Vec<&str>>| Op::Focus {
            path: conv.clone(),
            speaker: speaker.map(slug),
            about: about.map(|a| a.into_iter().map(p).collect()),
        };
        let a = apply(&mut t, &john(), &op(Some("yen"), None), now()).unwrap();
        assert_eq!(
            a.subject,
            "Focus of @zen/conversations/2026-10-31-0712-grok: yen speaking"
        );
        let a = apply(
            &mut t,
            &john(),
            &op(None, Some(vec!["topics/house"])),
            now(),
        )
        .unwrap();
        assert_eq!(
            a.subject,
            "Focus of @zen/conversations/2026-10-31-0712-grok: yen speaking; about topics/house"
        );
        apply(
            &mut t,
            &john(),
            &op(
                None,
                Some(vec!["topics/house/permits", "topics/house/permits"]),
            ),
            now(),
        )
        .unwrap();
        let f = t
            .get(&conv)
            .unwrap()
            .meta
            .as_ref()
            .unwrap()
            .focus
            .clone()
            .unwrap();
        assert_eq!(f.speaker, Some(slug("yen")));
        assert_eq!(f.about, vec![Below::parse("topics/house/permits").unwrap()]);
    }

    #[test]
    fn a_focus_names_what_is_there_on_a_conversation() {
        let (mut t, conv) = with_conversation();
        for (path, speaker, about, says) in [
            (
                conv.clone(),
                Some("nobody"),
                None,
                "nobody has no journal under zen's people",
            ),
            (
                conv.clone(),
                None,
                Some(vec!["topics/nothing"]),
                "there is no journal @zen/topics/nothing",
            ),
            (
                conv.clone(),
                None,
                Some(vec!["@guest/topics"]),
                "is not in @zen",
            ),
            (
                conv.clone(),
                None,
                Some(vec!["topics", "people", "threads", "bots"]),
                "at most 3 journals at once, not 4",
            ),
            (
                conv.clone(),
                None,
                Some(vec!["@zen"]),
                "@zen opens every frame already",
            ),
            (
                conv.clone(),
                None,
                None,
                "names a speaker, what it is about, or both",
            ),
            (p("topics/house"), Some("yen"), None, "is no conversation"),
        ] {
            let op = Op::Focus {
                path,
                speaker: speaker.map(slug),
                about: about.map(|a| a.into_iter().map(p).collect()),
            };
            let err = apply(&mut t, &john(), &op, now()).unwrap_err().to_string();
            assert!(err.contains(says), "{err}");
        }
    }
}
