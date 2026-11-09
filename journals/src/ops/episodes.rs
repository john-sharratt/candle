//! A bot's experience since npcd last wrote (`docs/npc_conversations.md`
//! §5.2): many episodes in one commit, each at the time it arose rather than
//! when the batch was written, each with what it was to the bot.
//!
//! One commit per bot per flush, on the model of a conversation's activity:
//! a bot that heard, said and did twenty things in five seconds is one write
//! to the repository, not twenty.

use crate::actor::Actor;
use crate::error::{Error, Result};
use crate::experience::Experience;
use crate::line::{Kind, Line, LineId};
use crate::ops::append::clean;
use crate::ops::{mint_keeping, Applied};
use crate::path::JournalPath;
use crate::terms::Terms;
use crate::time::Stamp;
use crate::tree::Tree;

/// The most lines one batch writes: a bot's buffer is capped at this many,
/// newest kept, so a flush is always one op.
pub const MAX_EPISODES: usize = 200;

/// One episode of a batch: when it arose, what was said of it, and what it
/// was to the bot.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Lived {
    pub at: Stamp,
    pub text: String,
    pub experience: Experience,
}

/// Each of `lines` as an episode of `path`, the `i`th taking `keep[i]` when
/// that id is free there — as a batch replayed onto a newer origin keeps the
/// ids it was first written with.
pub fn episodes(
    tree: &mut Tree,
    actor: &Actor,
    path: &JournalPath,
    lines: &[Lived],
    keep: &[LineId],
) -> Result<Applied> {
    if lines.is_empty() {
        return Err(Error::invalid("a batch of episodes holds at least one"));
    }
    if lines.len() > MAX_EPISODES {
        return Err(Error::invalid(format!(
            "a batch of episodes holds at most {MAX_EPISODES}, not {}",
            lines.len()
        )));
    }
    let j = tree.need_mut(path)?;
    j.writable(&path.show())?;
    let by = actor.writer();
    let mut written = Vec::with_capacity(lines.len());
    for (i, lived) in lines.iter().enumerate() {
        let text = clean(&lived.text)?;
        let line = Line {
            id: mint_keeping(j, keep.get(i), lived.at, by, &text),
            at: lived.at,
            kind: Kind::Episode,
            by,
            account: None,
            speaker: None,
            channel: None,
            folds: None,
            proposed: false,
            terms: Terms::default(),
            experience: Some(lived.experience.clone()),
            text,
        };
        line.check().map_err(Error::Invalid)?;
        written.push(line.id.clone());
        // By time, then id within a second, as a journal's files are read
        // back: a batch's lines take their places among what is there.
        let at = j
            .window
            .partition_point(|l| (l.at, &l.id) < (line.at, &line.id));
        j.window.insert(at, line);
    }
    Ok(Applied {
        lines: written,
        ..Applied::touching(
            format!("{} episodes on {path}", lines.len()),
            vec![path.clone()],
        )
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::experience::ExperienceKind;
    use crate::ops::fixture::{john, now, p, zen};
    use crate::ops::{apply, apply_keeping, OnTaken, Op};
    use crate::testing::slug;
    use crate::tree::render::own_files;

    fn npcd() -> Actor {
        Actor::Npcd { group: slug("zen") }
    }

    /// `bots/zen`, as npcd makes it before its first batch.
    fn with_bot() -> Tree {
        let mut t = zen();
        apply(
            &mut t,
            &npcd(),
            &Op::Create {
                under: p("bots"),
                name: slug("zen"),
                cue: "open for what Zen heard, said and did".into(),
                on_taken: OnTaken::Refuse,
            },
            now(),
        )
        .unwrap();
        t
    }

    fn batch() -> Vec<Lived> {
        vec![
            Lived {
                at: now().plus(-30),
                text: "John, on the phone: is the parcel in?".into(),
                experience: Experience::about(ExperienceKind::Heard, p("people/john")),
            },
            Lived {
                at: now().plus(-25),
                text: "Told John it is at the door.".into(),
                experience: Experience::about(ExperienceKind::Said, p("people/john")),
            },
            Lived {
                at: now().plus(-40),
                text: "Turned the lounge TV off.".into(),
                experience: Experience::of(ExperienceKind::Did),
            },
        ]
    }

    /// **A batch is one commit of many lines, each at its own time and with
    /// its own experience**, written into the window in time order.
    #[test]
    fn a_batch_writes_each_line_at_its_own_time() {
        let mut t = with_bot();
        let path = p("bots/zen");
        let op = Op::Episodes {
            path: path.clone(),
            lines: batch(),
        };
        let a = apply(&mut t, &npcd(), &op, now()).unwrap();
        assert_eq!(a.subject, "3 episodes on @zen/bots/zen");
        assert_eq!(a.touched, vec![path.clone()]);
        assert_eq!(a.lines.len(), 3);
        let j = t.get(&path).unwrap();
        let seen: Vec<(Stamp, &str, ExperienceKind)> = j
            .window
            .iter()
            .map(|l| {
                (
                    l.at,
                    l.text.as_str(),
                    l.experience.as_ref().unwrap().kind,
                )
            })
            .collect();
        assert_eq!(
            seen,
            [
                (now().plus(-40), "Turned the lounge TV off.", ExperienceKind::Did),
                (
                    now().plus(-30),
                    "John, on the phone: is the parcel in?",
                    ExperienceKind::Heard
                ),
                (now().plus(-25), "Told John it is at the door.", ExperienceKind::Said),
            ]
        );
        for l in &j.window {
            assert_eq!((l.kind, l.account.as_ref()), (Kind::Episode, None));
        }
        let files = own_files(&path, j);
        let heard = &files[&format!("groups/zen/bots/zen/{}.yaml", a.lines[0])];
        assert_eq!(
            String::from_utf8(heard.clone()).unwrap(),
            "at: 2026-10-30T11:59:30Z\nkind: episode\nby: npcd\nexperience: heard\n\
             about: \"@zen/people/john\"\ntext: \"John, on the phone: is the parcel in?\"\n"
        );
    }

    /// **A replayed batch keeps every id that is still free**, and takes a
    /// fresh one only for an id another line has taken meanwhile.
    #[test]
    fn a_replay_keeps_each_free_id() {
        let mut first = with_bot();
        let op = Op::Episodes {
            path: p("bots/zen"),
            lines: batch(),
        };
        let ids = apply(&mut first, &npcd(), &op, now()).unwrap().lines;
        let mut onto = with_bot();
        let again = apply_keeping(&mut onto, &npcd(), &op, now(), &ids).unwrap();
        assert_eq!(again.lines, ids);
        let taken = apply_keeping(&mut onto, &npcd(), &op, now(), &ids).unwrap();
        assert!(taken.lines.iter().all(|id| !ids.contains(id)));
    }

    #[test]
    fn an_empty_or_overlong_batch_and_a_bad_line_are_refused_whole() {
        let mut t = with_bot();
        let empty = Op::Episodes {
            path: p("bots/zen"),
            lines: vec![],
        };
        assert!(apply(&mut t, &npcd(), &empty, now())
            .unwrap_err()
            .to_string()
            .contains("at least one"));
        let mut lines = batch();
        lines[1].text = "  ".into();
        let bad = Op::Episodes {
            path: p("bots/zen"),
            lines,
        };
        assert!(apply(&mut t, &npcd(), &bad, now()).is_err());
        assert!(
            t.get(&p("bots/zen")).unwrap().window.is_empty(),
            "a refused batch writes nothing"
        );
        let many = Op::Episodes {
            path: p("bots/zen"),
            lines: vec![batch()[0].clone(); MAX_EPISODES + 1],
        };
        assert!(apply(&mut t, &npcd(), &many, now())
            .unwrap_err()
            .to_string()
            .contains("at most 200"));
    }

    /// **Episodes are npcd's, under its own group's bots**: no person, and not
    /// npcd anywhere else.
    #[test]
    fn only_npcd_writes_a_batch_and_only_under_bots() {
        let mut t = with_bot();
        let op = |path: &str| Op::Episodes {
            path: p(path),
            lines: batch(),
        };
        assert!(apply(&mut t, &john(), &op("bots/zen"), now()).is_err());
        assert!(apply(&mut t, &npcd(), &op("people"), now()).is_err());
        assert!(apply(&mut t, &Actor::Zend, &op("bots/zen"), now()).is_err());
        let other = Actor::Npcd {
            group: slug("guest"),
        };
        assert!(apply(&mut t, &other, &op("bots/zen"), now()).is_err());
        assert!(apply(&mut t, &npcd(), &op("bots/zen"), now()).is_ok());
    }
}
