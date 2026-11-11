//! A journal commit's message (§7.1): its subject, then trailers naming who
//! wrote, for whom, through what, when — and the operation itself, so a
//! commit not yet on origin can be replayed onto what origin holds.

use crate::actor::Actor;
use crate::error::{Error, Result};
use crate::line::LineId;
use crate::ops::wire::{actor_from_wire, actor_to_wire, op_from_wire, op_to_wire};
use crate::ops::Op;
use crate::time::Stamp;

const AT: &str = "Journal-At";
const LINE: &str = "Journal-Line";
const ACTOR: &str = "Journal-Actor";
const OP: &str = "Journal-Op";

/// The message of `op`'s commit; `lines` are the lines it wrote, in order —
/// one, or a batch's — so a replay keeps their ids.
pub fn message(subject: &str, actor: &Actor, op: &Op, now: Stamp, lines: &[LineId]) -> String {
    let mut out = format!("{subject}\n\n");
    for (k, v) in actor.trailers() {
        out.push_str(&format!("{k}: {v}\n"));
    }
    out.push_str(&format!("{AT}: {now}\n"));
    for id in lines {
        out.push_str(&format!("{LINE}: {id}\n"));
    }
    out.push_str(&format!("{ACTOR}: {}\n", actor_to_wire(actor)));
    out.push_str(&format!("{OP}: {}\n", op_to_wire(op)));
    out
}

/// What a journal commit did, read back from its message.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Recorded {
    pub actor: Actor,
    pub op: Op,
    pub at: Stamp,
    /// The lines it wrote, in order: none, one, or a batch's.
    pub lines: Vec<LineId>,
}

fn trailer<'a>(message: &'a str, k: &str) -> Option<&'a str> {
    let prefix = format!("{k}: ");
    message.lines().rev().find_map(|l| l.strip_prefix(&prefix))
}

/// Every `k` trailer, in the order written.
fn trailers<'a>(message: &'a str, k: &str) -> Vec<&'a str> {
    let prefix = format!("{k}: ");
    message
        .lines()
        .filter_map(|l| l.strip_prefix(&prefix))
        .collect()
}

pub fn read(message: &str) -> Result<Recorded> {
    let need = |k: &str| {
        trailer(message, k).ok_or_else(|| Error::invalid(format!("the commit has no {k} trailer")))
    };
    let json = |s: &str| {
        serde_json::from_str(s).map_err(|e| Error::invalid(format!("a trailer is not JSON: {e}")))
    };
    Ok(Recorded {
        actor: actor_from_wire(&json(need(ACTOR)?)?)?,
        op: op_from_wire(&json(need(OP)?)?)?,
        at: Stamp::parse(need(AT)?).map_err(Error::Invalid)?,
        lines: trailers(message, LINE)
            .into_iter()
            .map(LineId::parse)
            .collect::<std::result::Result<_, _>>()
            .map_err(Error::Invalid)?,
    })
}

#[cfg(test)]
mod tests {
    use std::slice;

    use super::*;
    use crate::experience::{Experience, ExperienceKind};
    use crate::ops::fixture::{now, p, yen};
    use crate::ops::Lived;
    use crate::terms::Terms;
    use crate::testing::slug;

    #[test]
    fn a_message_reads_back_as_what_it_recorded() {
        let op = Op::Note {
            path: p("topics"),
            text: "Builder starts Monday.\nBring keys.".into(),
            speaker: Some(slug("john")),
            terms: Terms::default(),
        };
        let id = LineId::parse("2026-10-30-1200-7e3a").unwrap();
        let m = message(
            "Note on @zen/topics",
            &yen(),
            &op,
            now(),
            slice::from_ref(&id),
        );
        assert!(m.starts_with(
            "Note on @zen/topics\n\nJournal-Writer: grok\nJournal-Account: yen\nJournal-Channel: grok\nJournal-At: 2026-10-30T12:00:00Z\nJournal-Line: 2026-10-30-1200-7e3a\nJournal-Actor: "
        ), "{m}");
        assert_eq!(
            read(&m).unwrap(),
            Recorded {
                actor: yen(),
                op: op.clone(),
                at: now(),
                lines: vec![id]
            }
        );
        let no_line = message("x", &yen(), &op, now(), &[]);
        assert_eq!(read(&no_line).unwrap().lines, vec![]);
    }

    /// **A batch's commit names each of its lines, in order**, so a replay
    /// keeps every id.
    #[test]
    fn a_batch_names_every_line_in_order() {
        let op = Op::Episodes {
            path: p("bots/zen"),
            lines: vec![Lived {
                at: now(),
                text: "x".into(),
                experience: Experience::of(ExperienceKind::Did),
            }],
        };
        let ids = [
            LineId::parse("2026-10-30-1200-aaaa").unwrap(),
            LineId::parse("2026-10-30-1159-bbbb").unwrap(),
        ];
        let npcd = Actor::Npcd { group: slug("zen") };
        let m = message("2 episodes on @zen/bots/zen", &npcd, &op, now(), &ids);
        assert!(
            m.contains("Journal-Line: 2026-10-30-1200-aaaa\nJournal-Line: 2026-10-30-1159-bbbb\n")
        );
        assert_eq!(read(&m).unwrap().lines, ids);
    }

    #[test]
    fn a_commit_made_by_hand_is_no_recorded_operation() {
        assert!(read("Fix a typo in the permits journal\n").is_err());
        assert!(read("x\n\nJournal-At: 2026-10-30T12:00:00Z\nJournal-Actor: {\"kind\":\"zend\"}\nJournal-Op: {not json}\n").is_err());
    }
}
