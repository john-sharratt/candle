//! A person's notes from before the memory backbone — the flat `U:` journal
//! zend kept in its substrate — moved onto their own journal in their home
//! group, `people/<person>`, the first time they call: what they asked to be
//! kept as notes, what was done for them as episodes, each at the time it was
//! written. A moved note leaves the substrate; one the tree refuses stays
//! there and is logged.

use std::sync::Arc;
use std::thread;

use journals::actor::Actor;
use journals::ops::Op;
use journals::path::JournalPath;
use journals::scope::Channel;
use journals::slug::Slug;
use journals::time::Stamp;

use super::{held, Memory};
use crate::mcp::records::Records;
use crate::mcp::threads::{self, Note};

/// The key the substrate kept a person's old notes under.
const OLD_JOURNAL: &str = "U:";
/// How the old journal labelled what a person asked to be kept.
const KEPT: &str = "Noted:";

/// `person`'s own journal in `group`, when their slug can name one.
pub fn own_journal(group: &Slug, person: &Slug) -> Result<JournalPath, String> {
    JournalPath::person(group, person).map_err(|e| e.to_string())
}

/// The write that moves `note` onto `person`'s journal at `path`.
pub fn move_of(path: JournalPath, person: &Slug, note: &Note) -> (Actor, Op, Stamp) {
    let channel = note
        .client
        .as_deref()
        .and_then(Channel::parse)
        .unwrap_or(Channel::Other);
    let at = Stamp::from_ms(note.at_ms);
    if note.label == KEPT {
        (
            Actor::Person {
                person: person.clone(),
                channel,
                admin: false,
            },
            Op::Note {
                path,
                text: note.text.clone(),
                speaker: Some(person.clone()),
            },
            at,
        )
    } else {
        (
            Actor::Server {
                person: person.clone(),
                channel,
            },
            Op::Episode {
                path,
                text: note.text.clone(),
                speaker: None,
            },
            at,
        )
    }
}

/// Move `owner`'s old notes onto `person`'s journal in `group`, once per
/// process, on a thread of its own: a person with hundreds of old notes is
/// not kept waiting for the commits on their first call. Until that journal
/// is there the notes have nowhere to go, and a later call moves them.
pub fn once<R: Records + 'static>(
    memory: &Arc<Memory>,
    records: R,
    owner: &str,
    group: &Slug,
    person: &Slug,
) {
    let there =
        own_journal(group, person).is_ok_and(|p| memory.tree().is_ok_and(|t| t.get(&p).is_some()));
    if !there || !held(&memory.moved).insert(owner.to_string()) {
        return;
    }
    let memory = Arc::clone(memory);
    let owner = owner.to_string();
    let group = group.clone();
    let person = person.clone();
    let spawned = thread::Builder::new()
        .name("journal-move".into())
        .spawn(move || move_all(&memory, &records, &owner, &group, &person));
    if let Err(e) = spawned {
        tracing::warn!("journals: the old notes' move did not start: {e}");
    }
}

/// Every one of `owner`'s old notes onto `person`'s journal in `group`, each
/// ended in the substrate once it is on the tree.
pub fn move_all(memory: &Memory, records: &dyn Records, owner: &str, group: &Slug, person: &Slug) {
    let old = threads::notes_keyed(records, owner, OLD_JOURNAL);
    if old.is_empty() {
        return;
    }
    let path = match own_journal(group, person) {
        Ok(p) => p,
        Err(e) => {
            tracing::warn!(person = %person, "journals: old notes stay where they were: {e}");
            return;
        }
    };
    let mut moved = 0usize;
    for (key, note) in &old {
        let (actor, op, at) = move_of(path.clone(), person, note);
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

    fn note(label: &str, channel: Option<&str>) -> Note {
        Note {
            at_ms: 1_793_351_402_000,
            label: label.into(),
            text: "the cat is Miso".into(),
            worker: None,
            client: channel.map(Into::into),
        }
    }

    #[test]
    fn kept_notes_and_episodes_go_to_their_own_journal() {
        let john = Slug::parse("john").unwrap();
        let zen = Slug::parse("zen").unwrap();
        let own = own_journal(&zen, &john).unwrap();
        let (actor, op, at) = move_of(own.clone(), &john, &note("Noted:", Some("grok")));
        assert_eq!(
            actor,
            Actor::Person {
                person: john.clone(),
                channel: Channel::Grok,
                admin: false,
            }
        );
        assert!(
            matches!(&op, Op::Note { path, speaker: Some(s), .. } if path.show() == "@zen/people/john" && s == &john)
        );
        assert_eq!(at.to_string(), "2026-10-30T09:10:02Z");
        let (actor, op, _) = move_of(own, &john, &note("Done:", None));
        assert_eq!(
            actor,
            Actor::Server {
                person: john,
                channel: Channel::Other
            }
        );
        assert!(matches!(&op, Op::Episode { path, .. } if path.show() == "@zen/people/john"));
        assert!(own_journal(&zen, &Slug::parse("archive").unwrap()).is_err());
    }
}
