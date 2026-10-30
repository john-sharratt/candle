//! A new line: a note a model kept, or an episode the server recorded.

use crate::actor::Actor;
use crate::error::{Error, Result};
use crate::line::{Kind, Line, LineId, MAX_TEXT};
use crate::ops::{mint_keeping, Applied};
use crate::path::JournalPath;
use crate::scope::Section;
use crate::time::Stamp;
use crate::tree::Tree;

pub fn note(
    tree: &mut Tree,
    actor: &Actor,
    path: &JournalPath,
    text: &str,
    now: Stamp,
    keep: Option<&LineId>,
) -> Result<Applied> {
    // A worker's note outside its thread waits for its person to accept it.
    let proposed =
        matches!(actor, Actor::Worker { .. }) && path.section() != Some(Section::Threads);
    append(tree, actor, path, (Kind::Note, proposed), text, now, keep)
}

pub fn episode(
    tree: &mut Tree,
    actor: &Actor,
    path: &JournalPath,
    text: &str,
    now: Stamp,
    keep: Option<&LineId>,
) -> Result<Applied> {
    append(tree, actor, path, (Kind::Episode, false), text, now, keep)
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

/// `kind` is the line's kind and whether it is a proposal.
fn append(
    tree: &mut Tree,
    actor: &Actor,
    path: &JournalPath,
    (kind, proposed): (Kind, bool),
    text: &str,
    now: Stamp,
    keep: Option<&LineId>,
) -> Result<Applied> {
    let text = clean(text)?;
    let j = tree.need_mut(path)?;
    j.writable(&path.show())?;
    let by = actor.writer();
    let line = Line {
        id: mint_keeping(j, keep, now, by, &text),
        at: now,
        kind,
        by,
        person: actor.person().cloned(),
        client: match actor {
            // The server writes for the client it serves; a client writing
            // for itself is already its writer.
            Actor::Server { client, .. } => Some(*client),
            _ => None,
        },
        folds: None,
        proposed,
        text,
    };
    line.check().map_err(Error::Invalid)?;
    let id = line.id.clone();
    // A window runs oldest to newest, so its tail is what is newest: a line
    // recorded as made in the past takes its place in time, after any line
    // of the same moment.
    let at = j.window.partition_point(|l| l.at <= line.at);
    j.window.insert(at, line);
    let what = match (kind, proposed) {
        (Kind::Note, true) => "Proposal on",
        (Kind::Note, false) => "Note on",
        _ => "Episode on",
    };
    Ok(Applied {
        line: Some(id),
        ..Applied::touching(format!("{what} {path}"), vec![path.clone()])
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::error::Problem;
    use crate::line::Writer;
    use crate::ops::fixture::{john, now, p, sharratt, yen};
    use crate::ops::{apply, Op};
    use crate::scope::Client;
    use crate::testing::slug;

    #[test]
    fn a_note_is_appended_as_its_writer_wrote_it() {
        let mut t = sharratt();
        let path = p("topics");
        let a = apply(
            &mut t,
            &yen(),
            &Op::Note {
                path: path.clone(),
                text: "  The builder starts Monday. ".into(),
            },
            now(),
        )
        .unwrap();
        assert_eq!(a.subject, "Note on @sharratt/topics");
        assert_eq!(a.touched, vec![path.clone()]);
        let line = &t.get(&path).unwrap().window[0];
        assert_eq!(line.text, "The builder starts Monday.");
        assert_eq!(line.by, Writer::Client(Client::Grok));
        assert_eq!(line.person, Some(slug("yen")));
        assert_eq!(line.client, None);
        assert_eq!(line.at, now());
        assert_eq!(Some(&line.id), a.line.as_ref());
    }

    #[test]
    fn the_server_names_the_client_it_wrote_for() {
        let mut t = sharratt();
        let path = p("~/clients/grok");
        let server = Actor::Server {
            person: slug("john"),
            client: Client::Grok,
        };
        apply(
            &mut t,
            &server,
            &Op::Episode {
                path: path.clone(),
                text: "image_create — a cat".into(),
            },
            now(),
        )
        .unwrap();
        let line = &t.get(&path).unwrap().window[0];
        assert_eq!(
            (line.kind, line.by, line.client),
            (Kind::Episode, Writer::Server, Some(Client::Grok))
        );
    }

    #[test]
    fn the_same_line_twice_in_a_minute_gets_two_ids() {
        let mut t = sharratt();
        let path = p("~");
        let op = Op::Note {
            path: path.clone(),
            text: "same".into(),
        };
        let a = apply(&mut t, &john(), &op, now()).unwrap().line.unwrap();
        let b = apply(&mut t, &john(), &op, now()).unwrap().line.unwrap();
        assert_ne!(a, b);
        assert_eq!(t.get(&path).unwrap().window.len(), 2);
    }

    /// **A line recorded in the past takes its place in time**, so the
    /// window's tail is still its newest lines.
    #[test]
    fn a_backdated_line_takes_its_place_in_time() {
        let mut t = sharratt();
        let path = p("~");
        for (at, text) in [
            (0, "first"),
            (120, "third"),
            (60, "second"),
            (120, "fourth"),
        ] {
            apply(
                &mut t,
                &john(),
                &Op::Note {
                    path: path.clone(),
                    text: text.into(),
                },
                now().plus(at),
            )
            .unwrap();
        }
        let texts: Vec<&str> = t
            .get(&path)
            .unwrap()
            .window
            .iter()
            .map(|l| l.text.as_str())
            .collect();
        assert_eq!(texts, ["first", "second", "third", "fourth"]);
    }

    #[test]
    fn a_workers_note_is_a_proposal_outside_its_thread() {
        let mut t = sharratt();
        let w = Actor::Worker {
            person: slug("john"),
        };
        let a = apply(
            &mut t,
            &w,
            &Op::Note {
                path: p("projects"),
                text: "Use keep trials.".into(),
            },
            now(),
        )
        .unwrap();
        assert_eq!(a.subject, "Proposal on @sharratt/projects");
        assert!(t.get(&p("projects")).unwrap().window[0].proposed);
        apply(
            &mut t,
            &w,
            &Op::Note {
                path: p("threads"),
                text: "Branch pushed.".into(),
            },
            now(),
        )
        .unwrap();
        assert!(!t.get(&p("threads")).unwrap().window[0].proposed);
    }

    #[test]
    fn empty_and_overlong_text_and_missing_journals_are_refused() {
        let mut t = sharratt();
        let path = p("~");
        assert!(apply(
            &mut t,
            &john(),
            &Op::Note {
                path: path.clone(),
                text: " \n ".into()
            },
            now()
        )
        .is_err());
        let long = "x".repeat(MAX_TEXT + 1);
        assert!(apply(
            &mut t,
            &john(),
            &Op::Note {
                path: path.clone(),
                text: long
            },
            now()
        )
        .unwrap_err()
        .to_string()
        .contains("at most 2000"));
        let missing = p("topics/nothing-here");
        assert_eq!(
            apply(
                &mut t,
                &john(),
                &Op::Note {
                    path: missing,
                    text: "x".into()
                },
                now()
            )
            .unwrap_err()
            .to_string(),
            "there is no journal @sharratt/topics/nothing-here"
        );
        assert!(
            t.get(&path).unwrap().window.is_empty(),
            "a refused write changes nothing"
        );
    }

    #[test]
    fn a_broken_journal_takes_no_notes() {
        let mut t = sharratt();
        let path = p("~");
        t.get_mut(&path)
            .unwrap()
            .problems
            .push(Problem::new("f", None, "bad"));
        assert!(matches!(
            apply(
                &mut t,
                &john(),
                &Op::Note {
                    path,
                    text: "x".into()
                },
                now()
            ),
            Err(Error::Refused(_))
        ));
    }
}
