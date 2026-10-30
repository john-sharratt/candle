//! A new line: a note a model kept, or an episode the server recorded. Each
//! records the account the call came from and, when the conversation's
//! focus has one, who was speaking (§2.4, §3.2).

use crate::actor::Actor;
use crate::error::{Error, Result};
use crate::line::{Kind, Line, LineId, MAX_TEXT};
use crate::ops::{mint_keeping, Applied};
use crate::path::JournalPath;
use crate::scope::Section;
use crate::slug::Slug;
use crate::time::Stamp;
use crate::tree::Tree;

/// What a new line is, besides its text.
struct Said<'a> {
    kind: Kind,
    /// Who the focus says is talking, when it says.
    speaker: Option<&'a Slug>,
    /// A worker's note outside its thread: kept once a member accepts it.
    proposed: bool,
}

pub fn note(
    tree: &mut Tree,
    actor: &Actor,
    path: &JournalPath,
    speaker: Option<&Slug>,
    text: &str,
    now: Stamp,
    keep: Option<&LineId>,
) -> Result<Applied> {
    let said = Said {
        kind: Kind::Note,
        speaker,
        proposed: matches!(actor, Actor::Worker { .. }) && path.section() != Some(Section::Threads),
    };
    append(tree, actor, path, said, text, now, keep)
}

/// `text` as a line's text: trimmed, not empty, within the cap.
pub(crate) fn clean(text: &str) -> Result<String> {
    let t = text.trim();
    if t.is_empty() {
        return Err(Error::invalid("the text is empty"));
    }
    let n = t.chars().count();
    if n > MAX_TEXT {
        return Err(Error::invalid(format!(
            "the text is {n} characters; a line holds at most {MAX_TEXT} — split it, or keep the gist"
        )));
    }
    Ok(t.to_string())
}

fn append(
    tree: &mut Tree,
    actor: &Actor,
    path: &JournalPath,
    said: Said<'_>,
    text: &str,
    now: Stamp,
    keep: Option<&LineId>,
) -> Result<Applied> {
    let text = clean(text)?;
    // A line is from someone the group knows, as a focus is (§2.4).
    if let Some(s) = said.speaker {
        if tree.get(&JournalPath::person(path.group(), s)?).is_none() {
            return Err(Error::not_found(format!(
                "{s} has no journal under {}'s people: a line is from someone the group knows",
                path.group()
            )));
        }
    }
    let j = tree.need_mut(path)?;
    j.writable(&path.show())?;
    let by = actor.writer();
    let line = Line {
        id: mint_keeping(j, keep, now, by, &text),
        at: now,
        kind: said.kind,
        by,
        account: actor.person().cloned(),
        speaker: said.speaker.cloned(),
        channel: match actor {
            // The server writes for the channel it serves; a channel writing
            // for itself is already its writer.
            Actor::Server { channel, .. } => Some(*channel),
            _ => None,
        },
        folds: None,
        proposed: said.proposed,
        text,
    };
    line.check().map_err(Error::Invalid)?;
    let id = line.id.clone();
    // A window runs oldest to newest, so its tail is what is newest: a line
    // recorded as made in the past takes its place in time — by time, then
    // id within a second, as a journal's files are read back.
    let at = j
        .window
        .partition_point(|l| (l.at, &l.id) < (line.at, &line.id));
    j.window.insert(at, line);
    let what = match (said.kind, said.proposed) {
        (Kind::Note, true) => "Proposal on",
        (Kind::Note, false) => "Note on",
        _ => "Episode on",
    };
    Ok(Applied {
        line: Some(id),
        ..Applied::touching(format!("{what} {path}"), vec![path.clone()])
    })
}

pub fn episode(
    tree: &mut Tree,
    actor: &Actor,
    path: &JournalPath,
    speaker: Option<&Slug>,
    text: &str,
    now: Stamp,
    keep: Option<&LineId>,
) -> Result<Applied> {
    let said = Said {
        kind: Kind::Episode,
        speaker,
        proposed: false,
    };
    append(tree, actor, path, said, text, now, keep)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::error::Problem;
    use crate::line::Writer;
    use crate::ops::fixture::{john, now, p, yen, zen};
    use crate::ops::{apply, OnTaken, Op};
    use crate::scope::Channel;
    use crate::testing::slug;

    fn note_op(path: &str, text: &str) -> Op {
        Op::Note {
            path: p(path),
            text: text.into(),
            speaker: None,
        }
    }

    /// `people/<person>`, made as any people journal is.
    fn known(t: &mut Tree, person: &str) {
        apply(
            t,
            &john(),
            &Op::Create {
                under: p("people"),
                name: slug(person),
                cue: format!("open for {person}"),
                on_taken: OnTaken::Refuse,
            },
            now(),
        )
        .unwrap();
    }

    #[test]
    fn a_note_is_appended_as_its_writer_wrote_it() {
        let mut t = zen();
        known(&mut t, "john");
        let path = p("topics");
        let a = apply(
            &mut t,
            &yen(),
            &Op::Note {
                path: path.clone(),
                text: "  The builder starts Monday. ".into(),
                speaker: Some(slug("john")),
            },
            now(),
        )
        .unwrap();
        assert_eq!(a.subject, "Note on @zen/topics");
        assert_eq!(a.touched, vec![path.clone()]);
        let line = &t.get(&path).unwrap().window[0];
        assert_eq!(line.text, "The builder starts Monday.");
        assert_eq!(line.by, Writer::Channel(Channel::Grok));
        assert_eq!(
            line.account,
            Some(slug("yen")),
            "Yen's account made the call"
        );
        assert_eq!(line.speaker, Some(slug("john")), "John was speaking on it");
        assert_eq!(line.channel, None);
        assert_eq!(line.at, now());
        assert_eq!(Some(&line.id), a.line.as_ref());
        // A speaker the group has no journal for is no one it knows.
        assert_eq!(
            apply(
                &mut t,
                &yen(),
                &Op::Note {
                    path,
                    text: "x".into(),
                    speaker: Some(slug("nobody")),
                },
                now(),
            )
            .unwrap_err()
            .to_string(),
            "nobody has no journal under zen's people: a line is from someone the group knows"
        );
    }

    #[test]
    fn the_server_names_the_channel_it_wrote_for() {
        let mut t = zen();
        let path = p("conversations");
        let server = Actor::Server {
            person: slug("john"),
            channel: Channel::Grok,
        };
        apply(
            &mut t,
            &server,
            &Op::Episode {
                path: path.clone(),
                text: "image_create — a cat".into(),
                speaker: None,
            },
            now(),
        )
        .unwrap();
        let line = &t.get(&path).unwrap().window[0];
        assert_eq!(
            (line.kind, line.by, line.channel),
            (Kind::Episode, Writer::Server, Some(Channel::Grok))
        );
    }

    #[test]
    fn the_same_line_twice_in_a_minute_gets_two_ids() {
        let mut t = zen();
        let op = note_op("people", "same");
        let a = apply(&mut t, &john(), &op, now()).unwrap().line.unwrap();
        let b = apply(&mut t, &john(), &op, now()).unwrap().line.unwrap();
        assert_ne!(a, b);
        assert_eq!(t.get(&p("people")).unwrap().window.len(), 2);
    }

    /// **A line recorded in the past takes its place in time**, so the
    /// window's tail is still its newest lines.
    #[test]
    fn a_backdated_line_takes_its_place_in_time() {
        let mut t = zen();
        for (at, text) in [
            (0, "first"),
            (120, "third"),
            (60, "second"),
            (120, "fourth"),
        ] {
            apply(&mut t, &john(), &note_op("@zen", text), now().plus(at)).unwrap();
        }
        let texts: Vec<&str> = t
            .get(&p("@zen"))
            .unwrap()
            .window
            .iter()
            .map(|l| l.text.as_str())
            .collect();
        assert_eq!(texts, ["first", "second", "third", "fourth"]);
    }

    #[test]
    fn a_workers_note_is_a_proposal_outside_its_thread() {
        let mut t = zen();
        let w = Actor::Worker {
            person: slug("john"),
        };
        let a = apply(&mut t, &w, &note_op("topics", "Use keep trials."), now()).unwrap();
        assert_eq!(a.subject, "Proposal on @zen/topics");
        assert!(t.get(&p("topics")).unwrap().window[0].proposed);
        apply(&mut t, &w, &note_op("threads", "Branch pushed."), now()).unwrap();
        assert!(!t.get(&p("threads")).unwrap().window[0].proposed);
    }

    #[test]
    fn empty_and_overlong_text_and_missing_journals_are_refused() {
        let mut t = zen();
        assert!(apply(&mut t, &john(), &note_op("@zen", " \n "), now()).is_err());
        let long = "x".repeat(MAX_TEXT + 1);
        assert!(apply(&mut t, &john(), &note_op("@zen", &long), now())
            .unwrap_err()
            .to_string()
            .contains("at most 2000"));
        assert_eq!(
            apply(&mut t, &john(), &note_op("topics/nothing-here", "x"), now())
                .unwrap_err()
                .to_string(),
            "there is no journal @zen/topics/nothing-here"
        );
        assert!(
            t.get(&p("@zen")).unwrap().window.is_empty(),
            "a refused write changes nothing"
        );
    }

    #[test]
    fn a_broken_journal_takes_no_notes() {
        let mut t = zen();
        t.get_mut(&p("@zen"))
            .unwrap()
            .problems
            .push(Problem::new("f", None, "bad"));
        assert!(matches!(
            apply(&mut t, &john(), &note_op("@zen", "x"), now()),
            Err(Error::Refused(_))
        ));
    }
}
