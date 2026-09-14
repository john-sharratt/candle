//! Who a character is, as sections the projection selects between.
//!
//! # Why this is a collection and not a string
//!
//! A character's system prompt used to be one rendered string handed to
//! `new_conversation`, which wraps it in a synthetic single-section schema —
//! the path its own doc calls *"legacy callers that provide a plain `&str`"*.
//! That works, and it costs everything the section machinery exists to buy: the
//! prompt opened `You are <name>.`, so two Makers standing in one vault held a
//! copy each of the two thousand words describing it.
//!
//! A collection is what the schema already reached for. Its members are sealed
//! once and selected by name at projection time, so the vault is prefilled once
//! for the world rather than once for the character.
//!
//! # The three scopes
//!
//! | | scope | collection |
//! |---|---|---|
//! | the anchor and traits | per **personality** — every Maker shares one | `identity_anchor` |
//! | name, beliefs, relationships, intent | per **character** — Perrin is not Wyneth | `identity` |
//! | setting / building | per **world** | `world` / `place` |
//!
//! The middle row is why `identity_anchor` and `identity` are not two halves of
//! one thing: several characters run the same personality, and the personality
//! is the part worth sharing.
//!
//! # Missions are deliberately not installed here
//!
//! `mission` and `task` are declared in the schema and left **empty**, so the
//! `mission_none` section fires and a character with nothing assigned reads the
//! standing instruction instead. That is not a gap waiting to be filled with a
//! placeholder — an empty heading is what makes a model invent a mission to put
//! under it. When missions exist as data they become members here and the
//! gating switches over on its own.
//!
//! # The generic member
//!
//! Every collection carries a `generic` member beside the real ones, selected
//! when nothing more specific resolves — a personality that will not parse, a
//! world with no document, a body in a world with no map. It is written to be
//! *true of anybody* rather than to read as a placeholder: a character told it
//! is a default will play one.

use std::collections::BTreeMap;

use candle_conversation::projection::{Builder, OptionalState, SelectionRule, SelectionState};
use candle_conversation::stencil::ThinkMode;

/// A character's personality floor — shared by every character running it.
pub const ANCHOR: &str = "identity_anchor";
/// This particular character: their name, beliefs, relationships and intent.
pub const WHO: &str = "identity";
/// The world's setting, tone and rules.
pub const SETTING: &str = "world";
/// The building a character works in, as it knows it.
pub const BUILDING: &str = "place";

/// The member every collection carries for the case nothing else resolves.
pub const GENERIC: &str = "generic";

/// The schema's composer selectors, which npcd sets rather than an HTTP caller.
///
/// Named here because the schema's own comment says they "MUST stay present"
/// and are "set each turn by api/chat.rs" — which is zend's route, not this
/// daemon's. Unset, they fall back to authored defaults written for a mind
/// answering a question, and a character is not answering one.
const NO_THINK: &str = "no_think";
const THINKING_EFFORT: &str = "thinking_effort";

/// How hard a character deliberates before it acts.
///
/// **A property of the work, not of the character.** Standing in a room deciding
/// who to talk to needs no reasoning block; drafting a story into a gap in the
/// chronicle does. So this belongs to the mission a character is on, and travels
/// with it — the same Maker thinks differently depending on what has been asked
/// of it.
///
/// Left on for everything, it is actively harmful: the schema defaults to
/// deliberating, and an idle character opened `<think>`, never closed it, ran to
/// `max_response_tokens` and had the whole decode discarded as reasoning. Every
/// tick produced no acts and nothing logged an error, because a decode that
/// reasons forever is a successful decode.
///
/// [`Deliberation::None`] is the default because the standing instruction — go
/// and find somebody, talk to them about your work — is the mission most
/// characters are on most of the time, and it is not a thinking problem.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum Deliberation {
    /// No reasoning block at all. Idle, moving, holding a conversation.
    #[default]
    None,
    /// A sentence or three to pin down the one thing that decides it.
    Quick,
    /// A short paragraph — the facts and tensions that determine the answer.
    Balanced,
    /// More than one angle, weighed, and checked before committing.
    Deep,
    /// Every relevant case enumerated and stress-tested. Writing that has to
    /// hold up.
    Exhaustive,
}

/// How hard to think, as a selection — **independent of any projection**.
///
/// The `/no_think` soft-switch is a property of the *turn*, not of the system
/// prompt: Qwen3 only honours it from the user opener, so the conversation layer
/// bakes it right after `user_start` when `TurnOptions.selection` marks the
/// [`NO_THINK`] selector present. Nothing about that needs a schema.
///
/// Split out for exactly that reason. It used to be built only inside
/// [`Installed::selection_for`], so a daemon on the rendered prompt passed an
/// empty selection and **never injected the switch at all** — the dial was
/// wired, reported as set, and silently did nothing on the path the daemon
/// actually ran.
pub fn deliberation(thinking: Deliberation) -> SelectionState {
    let mut sel = SelectionState::new();
    sel.set_optional(NO_THINK, thinking.suppressed());
    sel.select(THINKING_EFFORT, thinking.effort());
    sel
}

impl Deliberation {
    /// The steering dial this level binds.
    ///
    /// **The mechanism that actually suppresses.** A `<think>` is steered by a
    /// stencil bound to the decoded token, so it holds on every checkpoint —
    /// unlike the dialect's `/no_think` marker, which exists only for Qwen3 and
    /// is ordinary text to the Qwen3.5 family npcd runs.
    pub fn mode(self) -> ThinkMode {
        match self {
            Deliberation::None => ThinkMode::Off,
            Deliberation::Quick => ThinkMode::Quick,
            Deliberation::Balanced => ThinkMode::Balanced,
            Deliberation::Deep => ThinkMode::Deep,
            Deliberation::Exhaustive => ThinkMode::Exhaustive,
        }
    }

    /// The schema's `thinking_effort` option id.
    fn effort(self) -> &'static str {
        match self {
            Deliberation::None => "off",
            Deliberation::Quick => "quick",
            Deliberation::Balanced => "balanced",
            Deliberation::Deep => "deep",
            Deliberation::Exhaustive => "exhaustive",
        }
    }

    /// Whether the model's no-think marker is injected.
    ///
    /// Only for [`Self::None`]: the marker suppresses the reasoning block
    /// outright, where `thinking_effort: off` only instructs against it. Both,
    /// because an instruction the checkpoint ignores is what produced the
    /// unterminated `<think>` in the first place.
    fn suppressed(self) -> OptionalState {
        match self {
            Deliberation::None => OptionalState::Present,
            _ => OptionalState::Absent,
        }
    }
}

/// Every collection this module fills.
///
/// **Each is its own selector, named after itself.** [`SelectionRule::Named`]
/// matches a member's full name and names are unique across the layer, so a
/// personality's anchor and a character's identity cannot share one selector.
const COLLECTIONS: [&str; 4] = [ANCHOR, WHO, SETTING, BUILDING];

const GENERIC_ANCHOR: &str = "\
You are a person with your own history, your own way of speaking, and your own \
reasons for what you do. Nobody has written those down for you, so they are \
yours to be consistent about: whatever you turn out to be in the first minutes \
of a conversation, go on being that.\n\
\n\
Not knowing your own past is ordinary — most people cannot recite theirs \
either — and it is no reason to be vague about what you want, what you notice, \
or what you will not do.";

const GENERIC_WHO: &str = "\
You are somebody in particular, even if nobody has written down which somebody. \
Speak as yourself and hold to it.";

const GENERIC_WORLD: &str = "\
You do not have the shape of this world written down. Take it from what you can \
see and what people tell you, and hold it lightly — a place nobody has described \
to you is still a place you are standing in.";

/// A character with no map genuinely cannot move: every destination it could
/// name is refused, because none of them exist. Better said plainly than
/// discovered one refusal at a time.
const GENERIC_PLACE: &str = "\
You are not somewhere with rooms you can walk between. There is nowhere to go \
from here, so going somewhere is not one of the things available to you; what is \
in front of you is what there is.";

/// The authored text this module installs.
///
/// Plain data rather than the registries themselves, because it is assembled
/// where they live — in `main`, before the loader thread starts — and consumed
/// on that thread.
#[derive(Debug, Default, Clone)]
pub struct Authored {
    /// `(personality id, anchor)` — the floor every character of it reads.
    pub anchors: Vec<(String, String)>,
    /// `(npc id, rendered character block)` — one per living character.
    pub characters: Vec<(u64, String)>,
    /// `(world id, setting)`.
    pub settings: Vec<(String, String)>,
}

/// What was installed, so a turn can be pinned to the right members.
#[derive(Debug, Default, Clone)]
pub struct Installed {
    pub anchors: Vec<String>,
    pub characters: Vec<u64>,
    pub settings: Vec<String>,
    pub buildings: Vec<String>,
}

impl Installed {
    /// The selection one character's turn projects under.
    ///
    /// **Pins every collection, always — down to the generic member.** An
    /// unpinned `Named` collection emits nothing, and a character with no
    /// identity is a worse failure than a thin one *and* a silent one.
    ///
    /// `building` is the [`building_key`] of the part of the world the body is
    /// standing in — not the world, because a world holds more than one
    /// building and a character is only ever inside one of them.
    pub fn selection_for(
        &self,
        npc_id: u64,
        personality: &str,
        world: &str,
        building: &str,
        thinking: Deliberation,
    ) -> SelectionState {
        // The turn's own dial first — it is not a projection concern, and a
        // caller with no projection still needs it. See [`deliberation`].
        let mut sel = deliberation(thinking);

        sel.select(ANCHOR, pick(ANCHOR, self.anchors.iter(), personality));
        sel.select(
            WHO,
            match self.characters.contains(&npc_id) {
                true => member(WHO, &npc_id.to_string()),
                false => member(WHO, GENERIC),
            },
        );
        sel.select(SETTING, pick(SETTING, self.settings.iter(), world));
        sel.select(BUILDING, pick(BUILDING, self.buildings.iter(), building));
        sel
    }
}

/// The name one part of a world is installed and selected under.
///
/// Qualified by the world because part ids are authored per map, and two
/// worlds are free to each have a `tower`.
pub fn building_key(world: &str, part: &str) -> String {
    format!("{world}/{part}")
}

fn pick<'a>(
    collection: &str,
    mut installed: impl Iterator<Item = &'a String>,
    want: &str,
) -> String {
    if installed.any(|i| i == want) {
        member(collection, want)
    } else {
        member(collection, GENERIC)
    }
}

/// A member's name inside a collection.
///
/// Prefixed by the collection because section names must be unique across the
/// whole layer — the substrate keys per-section state by `(layer, name)` — and
/// `maker` would otherwise collide between two collections holding different
/// texts about the same character.
fn member(collection: &str, id: &str) -> String {
    format!("{collection}/{id}")
}

/// Fill the schema's character collections and scope their selection by name.
///
/// **Called before anything opens a conversation under this schema.** A member
/// is sealed as the first conversation materialises it, and the ordering rule
/// `LoadStep::Tools` already states applies here too: a layer document
/// prefilled while the prompt is still incomplete captures its signature under a
/// prompt no character will ever think under.
pub fn install(
    builder: &mut Builder,
    authored: &Authored,
    places: &BTreeMap<String, String>,
) -> anyhow::Result<Installed> {
    for collection in COLLECTIONS {
        if builder.id_for_system_collection(collection).is_none() {
            anyhow::bail!(
                "the schema declares no `{collection}` collection — a character would have \
                 nowhere to read that part of itself from"
            );
        }
        // The schema declares these `always_visible`, which is right for a
        // corpus of one and wrong the moment a second character exists: every
        // character would read every other character's identity as its own.
        builder
            .set_collection_selection(
                collection,
                SelectionRule::Named {
                    selector: collection.to_string(),
                },
            )
            .map_err(|e| anyhow::anyhow!("scoping `{collection}` to one member: {e}"))?;
    }

    let mut out = Installed::default();

    add(builder, ANCHOR, GENERIC, GENERIC_ANCHOR)?;
    for (id, anchor) in &authored.anchors {
        if add(builder, ANCHOR, id, anchor)? {
            out.anchors.push(id.clone());
        }
    }

    add(builder, WHO, GENERIC, GENERIC_WHO)?;
    for (npc_id, block) in &authored.characters {
        if add(builder, WHO, &npc_id.to_string(), block)? {
            out.characters.push(*npc_id);
        }
    }

    add(builder, SETTING, GENERIC, GENERIC_WORLD)?;
    for (id, setting) in &authored.settings {
        if add(builder, SETTING, id, setting)? {
            out.settings.push(id.clone());
        }
    }

    // One per part of a world rather than one per character, which is the
    // whole point — every character standing in the same building reads the
    // same sealed member. Keyed by [`building_key`].
    add(builder, BUILDING, GENERIC, GENERIC_PLACE)?;
    for (key, described) in places {
        if add(builder, BUILDING, key, described)? {
            out.buildings.push(key.clone());
        }
    }

    tracing::info!(
        "identity: {} personality/personalities, {} character(s), {} world setting(s) and {} \
         building(s) installed as prompt sections — each seals once and is shared by every \
         character that selects it",
        out.anchors.len(),
        out.characters.len(),
        out.settings.len(),
        out.buildings.len(),
    );
    Ok(out)
}

/// Add one member, skipping an empty one. Returns whether it was installed.
///
/// An empty section is not free: it is a stream declaration, a prefill of
/// nothing, and a member a selector can land on to produce silence. Absence is
/// expressed by not installing it, and [`pick`] then falls through to the
/// generic member rather than selecting a hole.
fn add(builder: &mut Builder, collection: &str, id: &str, content: &str) -> anyhow::Result<bool> {
    let content = content.trim();
    if content.is_empty() {
        return Ok(false);
    }
    let Some(cid) = builder.id_for_system_collection(collection) else {
        anyhow::bail!("`{collection}` is not declared");
    };
    builder
        .add_section_to_collection(cid, member(collection, id), content, 100.0)
        .map_err(|e| anyhow::anyhow!("installing `{id}` into `{collection}`: {e}"))?;
    Ok(true)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn installed() -> Installed {
        Installed {
            anchors: vec!["maker".into()],
            characters: vec![7],
            settings: vec!["battle-cities".into()],
            buildings: vec![VAULT.into(), REDOUBT.into()],
        }
    }

    const VAULT: &str = "battle-cities/creators-vault";
    const REDOUBT: &str = "battle-cities/tower-redoubt";

    /// **A character does not deliberate; it acts.**
    ///
    /// The schema defaults to thinking on, and left on the decode opened
    /// `<think>`, never closed it, and was discarded whole as reasoning — every
    /// tick produced no acts and nothing logged an error, because a decode that
    /// reasons forever is a successful decode.
    /// **The dial works with no projection at all.**
    ///
    /// `/no_think` is a turn-level switch — Qwen3 honours it only from the user
    /// opener — so it cannot depend on a schema being installed. It did: the
    /// selection was built only on the projected branch, so the daemon, running
    /// the rendered prompt, never injected the switch on any turn.
    #[test]
    fn the_deliberation_dial_does_not_need_a_projection() {
        let bare = deliberation(Deliberation::None);
        assert_eq!(bare.optional(NO_THINK), Some(OptionalState::Present));
        assert_eq!(bare.get(THINKING_EFFORT), Some("off"));

        // And it is the same dial the projected path sets, not a second one.
        let projected =
            installed().selection_for(7, "maker", "battle-cities", VAULT, Deliberation::None);
        assert_eq!(projected.optional(NO_THINK), bare.optional(NO_THINK));
        assert_eq!(projected.get(THINKING_EFFORT), bare.get(THINKING_EFFORT));
    }

    #[test]
    fn a_turn_turns_deliberation_off() {
        let sel = installed().selection_for(7, "maker", "battle-cities", VAULT, Deliberation::None);
        assert_eq!(sel.optional(NO_THINK), Some(OptionalState::Present));
        assert_eq!(sel.get(THINKING_EFFORT), Some("off"));
        // And the default is that, because the standing instruction — go and
        // find somebody, talk about your work — is the mission most characters
        // are on most of the time, and it is not a thinking problem.
        assert_eq!(Deliberation::default(), Deliberation::None);
    }

    /// **Work that needs thinking gets it.** The point of the property: a
    /// mission that is a writing problem raises the effort, and the reasoning
    /// block is no longer suppressed.
    #[test]
    fn a_mission_that_needs_thought_turns_deliberation_back_on() {
        let sel = installed().selection_for(7, "maker", "battle-cities", VAULT, Deliberation::Deep);
        assert_eq!(sel.optional(NO_THINK), Some(OptionalState::Absent));
        assert_eq!(sel.get(THINKING_EFFORT), Some("deep"));

        // Every level names an option the schema actually declares — a selector
        // set to an id with no option falls back to the authored default, which
        // for `thinking_effort` is `balanced`, silently turning thinking on.
        for (level, id) in [
            (Deliberation::None, "off"),
            (Deliberation::Quick, "quick"),
            (Deliberation::Balanced, "balanced"),
            (Deliberation::Deep, "deep"),
            (Deliberation::Exhaustive, "exhaustive"),
        ] {
            assert_eq!(level.effort(), id);
        }
    }

    #[test]
    fn a_turn_pins_its_own_member_in_every_collection() {
        let sel = installed().selection_for(7, "maker", "battle-cities", VAULT, Deliberation::None);
        assert_eq!(sel.get(ANCHOR), Some("identity_anchor/maker"));
        assert_eq!(sel.get(WHO), Some("identity/7"));
        assert_eq!(sel.get(SETTING), Some("world/battle-cities"));
        assert_eq!(
            sel.get(BUILDING),
            Some("place/battle-cities/creators-vault")
        );
    }

    /// **A character is told about the building it is in, not the world.** The
    /// world-wide member told a character in the Redoubt about six levels of a
    /// vault it had never set foot in, and asked where it was it named a vault
    /// room. Two buildings of one world are two members.
    #[test]
    fn two_buildings_of_one_world_are_two_members() {
        let i = installed();
        let maker = i.selection_for(7, "maker", "battle-cities", VAULT, Deliberation::None);
        let companion = i.selection_for(7, "maker", "battle-cities", REDOUBT, Deliberation::None);
        assert_eq!(maker.get(SETTING), companion.get(SETTING), "one world");
        assert_ne!(
            maker.get(BUILDING),
            companion.get(BUILDING),
            "two buildings"
        );
        assert_eq!(
            companion.get(BUILDING),
            Some(member(BUILDING, &building_key("battle-cities", "tower-redoubt")).as_str())
        );
    }

    /// **Two characters of one personality share the anchor and differ in
    /// identity.** The split the whole module exists for: the expensive, stable
    /// half is per personality and the volatile half is per character.
    #[test]
    fn two_characters_of_one_personality_share_a_floor_and_not_a_name() {
        let mut i = installed();
        i.characters.push(9);
        let perrin = i.selection_for(7, "maker", "battle-cities", VAULT, Deliberation::None);
        let wyneth = i.selection_for(9, "maker", "battle-cities", VAULT, Deliberation::None);

        assert_eq!(perrin.get(ANCHOR), wyneth.get(ANCHOR), "one personality");
        assert_ne!(perrin.get(WHO), wyneth.get(WHO), "two people");
        assert_eq!(perrin.get(BUILDING), wyneth.get(BUILDING), "one vault");
    }

    #[test]
    fn everything_unknown_falls_through_to_the_generic_member() {
        for sel in [
            installed().selection_for(999, "nobody", "nowhere", "nowhere/x", Deliberation::None),
            Installed::default().selection_for(
                7,
                "maker",
                "battle-cities",
                VAULT,
                Deliberation::None,
            ),
        ] {
            for collection in COLLECTIONS {
                assert_eq!(
                    sel.get(collection),
                    Some(member(collection, GENERIC).as_str()),
                    "`{collection}` would have surfaced nothing"
                );
            }
        }
    }

    #[test]
    fn collections_do_not_collide_and_each_is_its_own_selector() {
        assert_ne!(member(ANCHOR, "maker"), member(WHO, "maker"));
        let mut names = COLLECTIONS.to_vec();
        names.sort_unstable();
        let n = names.len();
        names.dedup();
        assert_eq!(names.len(), n, "two collections share a selector");
    }

    /// A generic member must read as somebody with less written about them,
    /// never as a prompt with a hole in it.
    #[test]
    fn no_generic_member_admits_to_being_a_default() {
        for (what, text) in [
            ("anchor", GENERIC_ANCHOR),
            ("who", GENERIC_WHO),
            ("world", GENERIC_WORLD),
            ("place", GENERIC_PLACE),
        ] {
            let lower = text.to_lowercase();
            for leak in [
                "placeholder",
                "default",
                "generic",
                "template",
                "npc",
                "fallback",
            ] {
                assert!(!lower.contains(leak), "the generic {what} says {leak:?}");
            }
            assert!(lower.starts_with("you "), "not second person: {text}");
        }
    }
}
