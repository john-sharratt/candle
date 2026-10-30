//! A person's notes from before the memory backbone — the flat `U:` journal
//! zend kept in its substrate — moved onto their tree the first time they
//! call: what they asked to be kept onto `~`, what was done for them onto
//! their client's journal, each at the time it was written. A moved note
//! leaves the substrate; one the tree refuses stays there and is logged.

use std::sync::Arc;
use std::thread;

use journals::actor::Actor;
use journals::ops::Op;
use journals::path::{JournalPath, Owner};
use journals::scope::Client;
use journals::slug::Slug;
use journals::time::Stamp;

use super::conversations::client_journal;
use super::{held, Memory};
use crate::mcp::records::Records;
use crate::mcp::threads::{self, Note};

/// The key the substrate kept a person's old notes under.
const OLD_JOURNAL: &str = "U:";
/// How the old journal labelled what a person asked to be kept.
const KEPT: &str = "Noted:";

/// The write that moves `note` onto `person`'s tree.
pub fn move_of(person: &Slug, note: &Note) -> (Actor, Op, Stamp) {
    let client = note
        .client
        .as_deref()
        .and_then(Client::parse)
        .unwrap_or(Client::Other);
    let at = Stamp::from_ms(note.at_ms);
    if note.label == KEPT {
        (
            Actor::Person {
                person: person.clone(),
                client,
            },
            Op::Note {
                path: JournalPath::root(Owner::Person(person.clone())),
                text: note.text.clone(),
            },
            at,
        )
    } else {
        (
            Actor::Server {
                person: person.clone(),
                client,
            },
            Op::Episode {
                path: client_journal(person, client),
                text: note.text.clone(),
            },
            at,
        )
    }
}

/// Move `owner`'s old notes onto `person`'s tree, once per process, on a
/// thread of its own: a person with hundreds of old notes is not kept waiting
/// for the commits on their first call.
pub fn once<R: Records + 'static>(memory: &Arc<Memory>, records: R, owner: &str, person: &Slug) {
    if !held(&memory.moved).insert(owner.to_string()) {
        return;
    }
    let memory = Arc::clone(memory);
    let owner = owner.to_string();
    let person = person.clone();
    let spawned = thread::Builder::new()
        .name("journal-move".into())
        .spawn(move || move_all(&memory, &records, &owner, &person));
    if let Err(e) = spawned {
        tracing::warn!("journals: the old notes' move did not start: {e}");
    }
}

/// Every one of `owner`'s old notes onto `person`'s tree, each ended in the
/// substrate once it is on the tree.
pub fn move_all(memory: &Memory, records: &dyn Records, owner: &str, person: &Slug) {
    let old = threads::notes_keyed(records, owner, OLD_JOURNAL);
    if old.is_empty() {
        return;
    }
    let mut moved = 0usize;
    for (key, note) in &old {
        let (actor, op, at) = move_of(person, note);
        match memory.write_at(&actor, &op, at) {
            Ok(_) => match records.cease(key) {
                Ok(()) => moved += 1,
                Err(e) => {
                    tracing::warn!(person = %person, "journals: moved a note but could not end it: {e}")
                }
            },
            Err(e) => {
                tracing::warn!(person = %person, "journals: an old note stays where it was: {e}: {}", note.text)
            }
        }
    }
    tracing::info!(person = %person, moved, of = old.len(), "journals: moved old notes onto the tree");
}

#[cfg(test)]
mod tests {
    use super::*;

    fn note(label: &str, client: Option<&str>) -> Note {
        Note {
            at_ms: 1_793_351_402_000,
            label: label.into(),
            text: "the cat is Miso".into(),
            worker: None,
            client: client.map(Into::into),
        }
    }

    #[test]
    fn kept_notes_go_to_the_root_and_episodes_to_their_client() {
        let john = Slug::parse("john").unwrap();
        let (actor, op, at) = move_of(&john, &note("Noted:", Some("grok")));
        assert_eq!(
            actor,
            Actor::Person {
                person: john.clone(),
                client: Client::Grok
            }
        );
        assert!(matches!(&op, Op::Note { path, .. } if path.show() == "~"));
        assert_eq!(at.to_string(), "2026-10-30T09:10:02Z");
        let (actor, op, _) = move_of(&john, &note("Done:", None));
        assert_eq!(
            actor,
            Actor::Server {
                person: john,
                client: Client::Other
            }
        );
        assert!(matches!(&op, Op::Episode { path, .. } if path.show() == "~/clients/other"));
    }
}
