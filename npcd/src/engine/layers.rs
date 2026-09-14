//! What one character can read from each layer of its projection.
//!
//! **The layers are the mind's.** The console lists every layer the mind's
//! `projection.yaml` declares, and what a layer holds for a character is read
//! here from the substrate — never a list the page keeps, so a layer added to
//! the schema shows up without anybody touching the console.
//!
//! **Read the way the character reads it.** A layer's groups are taken from the
//! character's own schema — the one its conversations open with, its dreams
//! scoped to it (see [`crate::engine::dreams::scoped`]) — so what this lists and
//! what the character's projection could gather cannot disagree about whose a
//! conversation is. [`readable`] is the rule.

use std::cmp::Reverse;
use std::collections::BTreeMap;

use candle_conversation::projection::{Builder, GroupId, TimelineId};
use candle_conversation::ConversationEngine;
use serde::Serialize;

/// How much of one layer a character can read.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct LayerCount {
    pub layer: String,
    pub conversations: usize,
    pub turns: u64,
}

/// One conversation in a layer, as the console shows it.
#[derive(Debug, Clone, Serialize)]
pub struct Held {
    pub timeline: u64,
    /// The name it was written under — a day's conversation, a dream, an
    /// ingested file. `None` for one that was never named.
    pub name: Option<String>,
    /// What it was written with: a dream's assumption, an ingest's hash.
    pub metadata: BTreeMap<String, String>,
    pub turns: Vec<HeldTurn>,
}

/// One turn, both halves, verbatim as stored.
#[derive(Debug, Clone, Serialize)]
pub struct HeldTurn {
    pub user: String,
    pub assistant: String,
}

/// The newest conversations of one layer, and whether there are older ones.
#[derive(Debug, Clone, Serialize)]
pub struct Page {
    pub conversations: Vec<Held>,
    pub more: bool,
}

/// What every conversation a character writes for itself is named with first.
///
/// The dash is part of it: without it `npc-12` would claim `npc-123`'s days.
pub fn own_prefix(npc_id: u64) -> String {
    format!("npc-{npc_id}-")
}

/// Whether a character reads a conversation written to a group.
///
/// Three cases, and the order is the rule:
///
/// - **The live group** — the one every character's own conversations are
///   written to — holds only this character's, by name. The group is shared by
///   the whole cast; a character's projection reads only its own conversation
///   in it, so that is all it shows.
/// - **A tag-scoped group** holds only the conversations carrying its tags. The
///   character's schema sets those to its own, which is what makes its dreams
///   its own.
/// - **Anything else** is shared: the world, what was ingested. Every
///   conversation in it is one the character can gather.
///
/// `carries` is asked only for a tag-scoped group, because answering it reads
/// every turn's tags.
pub fn readable(
    live: bool,
    scope: &[String],
    npc_id: u64,
    name: Option<&str>,
    carries: impl FnOnce(&[String]) -> bool,
) -> bool {
    if live {
        return name.is_some_and(|n| n.starts_with(&own_prefix(npc_id)));
    }
    if !scope.is_empty() {
        return carries(scope);
    }
    true
}

/// Every conversation of `layer` this character can read, newest first.
/// `None` for a layer the schema does not declare.
fn readable_in(
    engine: &ConversationEngine,
    builder: &Builder,
    live_group: GroupId,
    npc_id: u64,
    layer: &str,
) -> Option<Vec<TimelineId>> {
    let layer = builder.schema().layers.iter().find(|l| l.name == layer)?;
    let mut out: Vec<TimelineId> = Vec::new();
    for group in &layer.groups {
        let live = group.id == live_group;
        for tl in engine.group_conversations(group.id) {
            let name = engine.conversation_conv_id(tl);
            if readable(live, &group.policy.tags, npc_id, name.as_deref(), |tags| {
                engine.conversation_carries(tl, tags)
            }) {
                out.push(tl);
            }
        }
    }
    // Timeline ids are minted in order, so the highest is the newest.
    out.sort_by_key(|tl| Reverse(tl.raw()));
    Some(out)
}

/// Every layer the schema declares, in its order, with how much of each this
/// character can read.
pub fn counts(
    engine: &ConversationEngine,
    builder: &Builder,
    live_group: GroupId,
    npc_id: u64,
) -> Vec<LayerCount> {
    builder
        .schema()
        .layers
        .iter()
        .map(|l| {
            let held =
                readable_in(engine, builder, live_group, npc_id, &l.name).unwrap_or_default();
            LayerCount {
                layer: l.name.clone(),
                conversations: held.len(),
                turns: held.iter().map(|&tl| engine.timeline_turn_count(tl)).sum(),
            }
        })
        .collect()
}

/// The newest `limit` conversations of `layer` this character can read, with
/// their turns. `None` for a layer the schema does not declare.
pub fn page(
    engine: &ConversationEngine,
    builder: &Builder,
    live_group: GroupId,
    npc_id: u64,
    layer: &str,
    limit: usize,
) -> Option<Page> {
    let held = readable_in(engine, builder, live_group, npc_id, layer)?;
    let more = held.len() > limit;
    let conversations = held
        .into_iter()
        .take(limit)
        .map(|tl| Held {
            timeline: tl.raw(),
            name: engine.conversation_conv_id(tl),
            metadata: engine.conversation_metadata(tl).unwrap_or_default(),
            turns: engine
                .conversation_texts(tl)
                .into_iter()
                .map(|(user, assistant)| HeldTurn { user, assistant })
                .collect(),
        })
        .collect();
    Some(Page {
        conversations,
        more,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tags(t: &[&str]) -> Vec<String> {
        t.iter().map(|s| s.to_string()).collect()
    }

    /// The live group is the whole cast's, and a character reads only its own
    /// conversations in it — by name, with the dash that keeps `npc-12` out
    /// of `npc-123`'s.
    #[test]
    fn in_the_live_group_a_character_reads_only_its_own_conversations() {
        let never = |_: &[String]| panic!("the live group is not tag-scoped");
        assert!(readable(true, &[], 12, Some("npc-12-day-3"), never));
        assert!(!readable(true, &[], 12, Some("npc-123-day-3"), never));
        assert!(!readable(true, &[], 12, Some("npc-7-day-3"), never));
        assert!(!readable(true, &[], 12, None, never));
    }

    /// A tag-scoped group — the dreams — is read through its tags, and only
    /// its tags: the name says nothing about whose a dream is.
    #[test]
    fn a_tag_scoped_group_holds_what_carries_its_tags() {
        let scope = tags(&["dreams:12"]);
        assert!(readable(false, &scope, 12, Some("npc-12-dream-9"), |t| t
            == scope.as_slice()));
        assert!(!readable(false, &scope, 12, Some("npc-12-dream-9"), |_| {
            false
        }));
    }

    /// Everything else is shared: the world is everybody's.
    #[test]
    fn an_unscoped_group_is_shared() {
        let never = |_: &[String]| panic!("an unscoped group reads no tags");
        assert!(readable(false, &[], 12, Some("world/the-waste.md"), never));
        assert!(readable(false, &[], 12, None, never));
    }
}
