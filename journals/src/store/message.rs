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

/// The message of `op`'s commit; `line` is the line it wrote, when it wrote
/// one, so a replay keeps that line's id.
pub fn message(subject: &str, actor: &Actor, op: &Op, now: Stamp, line: Option<&LineId>) -> String {
    let mut out = format!("{subject}\n\n");
    for (k, v) in actor.trailers() {
        out.push_str(&format!("{k}: {v}\n"));
    }
    out.push_str(&format!("{AT}: {now}\n"));
    if let Some(id) = line {
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
    pub line: Option<LineId>,
}

fn trailer<'a>(message: &'a str, k: &str) -> Option<&'a str> {
    let prefix = format!("{k}: ");
    message.lines().rev().find_map(|l| l.strip_prefix(&prefix))
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
        line: trailer(message, LINE)
            .map(LineId::parse)
            .transpose()
            .map_err(Error::Invalid)?,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ops::fixture::{now, p, yen};

    #[test]
    fn a_message_reads_back_as_what_it_recorded() {
        let op = Op::Note {
            path: p("topics"),
            text: "Builder starts Monday.\nBring keys.".into(),
        };
        let id = LineId::parse("20261030-1200-7e3a").unwrap();
        let m = message("Note on @sharratt/topics", &yen(), &op, now(), Some(&id));
        assert!(m.starts_with(
            "Note on @sharratt/topics\n\nJournal-Writer: grok\nJournal-Person: yen\nJournal-Client: grok\nJournal-At: 2026-10-30T12:00:00Z\nJournal-Line: 20261030-1200-7e3a\nJournal-Actor: "
        ), "{m}");
        assert_eq!(
            read(&m).unwrap(),
            Recorded {
                actor: yen(),
                op: op.clone(),
                at: now(),
                line: Some(id)
            }
        );
        let no_line = message("x", &yen(), &op, now(), None);
        assert_eq!(read(&no_line).unwrap().line, None);
    }

    #[test]
    fn a_commit_made_by_hand_is_no_recorded_operation() {
        assert!(read("Fix a typo in the permits journal\n").is_err());
        assert!(read("x\n\nJournal-At: 2026-10-30T12:00:00Z\nJournal-Actor: {\"kind\":\"zend\"}\nJournal-Op: {not json}\n").is_err());
    }
}
