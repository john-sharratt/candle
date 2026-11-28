//! Chooses which acts a turn's prompt describes, so the prompt matches the grammar
//! the turn is decoded under.
//!
//! Every act is installed once in the mind's tools collection, and each turn shows
//! only what its room, companions, body and mission make possible, ranked by how
//! well each fits what the character is doing. An act offered in one place only is
//! always shown, since its address means nothing to a character never told what it
//! does. A turn of plain words is shown no act at all, so the character is not
//! offered a call it would then write out as text. A body with no hands or feet is
//! never offered walking.

use super::*;

/// The tools offered in a mode, whatever the character's situation.
///
/// **What the rendered prompt lists** — the prompt a daemon with no mind uses,
/// written once when a conversation opens, so only tools that are always there
/// belong in it. Under the mind's projection every tool is installed and each
/// turn shows the ones it can take; see [`show_within`].
pub fn for_mode(mode: Mode) -> Vec<&'static Tool> {
    for_body(mode, true)
}

/// [`for_mode`], for a mind that may not have a body.
///
/// `embodied` is false for a tower consciousness or anything else that acts only
/// through systems. Every act that needs hands, feet or a place to stand goes
/// with it — not refused, absent, because a mind offered a way to walk when it
/// has never had legs will try to use it.
pub fn for_body(mode: Mode, embodied: bool) -> Vec<&'static Tool> {
    CATALOG
        .iter()
        .filter(|t| match t.availability {
            Availability::Always => true,
            Availability::MessagingOnly => mode.remote(),
            Availability::Pictorial => mode.carries_pictures(),
            Availability::PhysicalOnly => mode == Mode::Physical && embodied,
            Availability::Embodied => embodied,
            // Depends on the room, which the prompt cannot know. Arrives with
            // the situation, like everything else conditional.
            Availability::AtPart => false,
            // Depends on who is standing next to you, which the prompt cannot
            // know and the situation can. See `nearby`.
            Availability::Nearby
            | Availability::PersonNearby
            | Availability::PersonAmongOthers
            | Availability::PersonInEarshot => false,
            // Depends on where the body is standing, which changes every time
            // it walks. The prompt is written once, so this is the situation's
            // to offer — the same reason `Nearby` is absent here.
            Availability::AwayFromHome => false,
            // What a body holds changes whenever it claims or gives back, so it
            // is the situation's to offer, not the prompt's.
            Availability::Holding => false,
            // Whether you are at the lift, and whether the car is there, are both
            // facts about where the body is standing this moment — the
            // situation's to offer, not the prompt's.
            Availability::AtLift | Availability::AtLanding => false,
            // A grant is the carried mission's, in its room — the situation's.
            Availability::Granted => false,
            // What a personality reaches is the character's, read per turn.
            Availability::Reached => false,
            // A conversation's own act is never the main loop's.
            Availability::OnThread => false,
        })
        .collect()
}

/// The tools a character has *because somebody else is here*.
///
/// Offered with the situation and computed from the world — the same placement
/// the percept gets, and for the same reason: who is in the room with you
/// changes every tick.
///
/// Empty when alone, and that emptiness is the point. A character with nobody
/// to address is never shown a way to address somebody, so it never invents one
/// to address.
///
/// `company` is how many others are here, because not every one of these arrives
/// with the first: a whisper needs somebody besides its listener to keep it from
/// (see [`Availability::PersonAmongOthers`]), and this line naming it to a
/// character with one person beside it undid the grammar's rule in words.
/// `people` is how many of them are people, and `heard_by_person` whether a
/// shout from here reaches one — room speech is for people (§12 of
/// `docs/npc_conversations.md`).
pub fn nearby(company: usize, people: usize, heard_by_person: bool) -> Vec<&'static Tool> {
    CATALOG
        .iter()
        .filter(|t| match t.availability {
            Availability::Nearby => company >= 1,
            Availability::PersonNearby => people >= 1,
            Availability::PersonAmongOthers => people >= 1 && company >= 2,
            Availability::PersonInEarshot => heard_by_person,
            _ => false,
        })
        .collect()
}

pub fn by_name(name: &str) -> Option<&'static Tool> {
    CATALOG.iter().find(|t| t.name == name)
}

/// Whether an act has **migrated to the effector device** — reached only through
/// `invoke` on its resource, and therefore no longer offered in the
/// compiled grammar (effector design Step 6, "deleting each corresponding `Tool`
/// … as it moves").
///
/// **This is the seam between the compiled catalogue and the device.** The `Tool`
/// stays in [`CATALOG`] — the device route reads its `params` to build the
/// resource schema, and [`crate::engine::body::perform`] still dispatches it when
/// a route invokes it — but [`specs_within`] filters it out, so a character can
/// no longer emit it as a plain call. It has to find the resource on its
/// effector device and act there, which is the whole point of the migration: the
/// tool surface stays fixed (`invoke`) while the world's affordances move
/// behind it.
///
/// The migration is incremental, one namespace at a time so the pulse can prove
/// each cut before the next. The command table is first: taking up and reporting
/// a mission is done by invoking `http://local/command/<id>/…`, not by a compiled
/// act, so those three leave the grammar here. [`mission_acts::MISSION_ACTS`] is
/// the one list of which three that is — read from there rather than repeated,
/// so a future namespace's migration cannot drift out of step with its own
/// registry the way a second hand-written name list would.
pub fn routed(tool: &str) -> bool {
    mission_acts::is_mine(tool)
}

/// Whether this act's outcome is *narrated* for the character to read back,
/// rather than handed the bare confirmation the world gives it ("You tell
/// Pax.").
///
/// The intent-carrying speech and expression acts: what a character *means* by
/// one is the whole of it, and the world's reply is a bare acknowledgement that
/// drops the intent — so the character reads back a rendering of what it did
/// instead. Acts with a real, fixed reply are absent on purpose: a `move_to`
/// reports the distance left, a world act reports what changed, and that reply
/// *is* the information, not a placeholder for it.
pub fn narrates(tool: &str) -> bool {
    matches!(
        tool,
        "tell" | "ask" | "whisper" | "shout" | "gesture" | "act"
    )
}

/// The system-prompt collection every tool is installed into — see [`install`].
pub const COLLECTION: &str = "tools";

/// A tool's member name in [`COLLECTION`].
pub fn member(tool: &str) -> String {
    format!("{COLLECTION}/{tool}")
}

/// `say(intent, manner?) — Say something aloud…` — a tool as one line: the
/// call, with optional parameters marked, and what it is for.
pub fn one_line(t: &Tool) -> String {
    let params: Vec<String> = t
        .params
        .iter()
        .map(|p| match p.required {
            true => p.name.to_string(),
            false => format!("{}?", p.name),
        })
        .collect();
    format!("{}({}) — {}", t.name, params.join(", "), t.description)
}

/// A tool as the character reads it in the prompt: [`one_line`], then each
/// parameter and what it takes, one per line.
///
/// **The parameters are the half a grammar cannot teach.** The mask forces
/// `intent` to be present; only its description says that it takes what you
/// *mean* rather than the sentence, which is the whole of this module's first
/// rule. No trailing newline — the collection's glue separates entries.
pub fn entry(t: &Tool) -> String {
    let mut s = one_line(t);
    if routed(t.name) {
        s.push_str(&format!(
            "\n  reached with `invoke`: use the address offered for `{}`, and put the arguments \
             below in its body",
            t.name
        ));
    }
    for p in t.params {
        let optional = if p.required { "" } else { " (optional)" };
        s.push_str(&format!("\n  {}{optional}: {}", p.name, p.description));
    }
    s
}

/// How many of the acts a room offers one turn's prompt describes, ranked by
/// provenance. An act [`shown_wherever_offered`] is not counted.
pub const SHOWN_ACTS: usize = 10;

/// Whether `t` is described on every turn that offers it, however it ranks.
///
/// **An act offered in one place only is the point of standing there.** A
/// routed act is one: its address in the mask is useless to a character never
/// told what the verb means. So is the lift's — a Drone sent to another level
/// reached the shaft twenty times, was told each time to ride with `lift_use`,
/// and walked away again because the ten acts its prompt described that turn
/// did not include it: "that tool does not exist in my available toolset". And
/// so is an act that is granted — a mission's own, a terminal's verbs while it
/// is connected — and a house device's, which are offered only for as long as
/// they are what the character is there to do.
pub fn shown_wherever_offered(t: &Tool) -> bool {
    routed(t.name)
        || BRIDGE_ACTS.iter().any(|b| b.name == t.name)
        || matches!(
            t.availability,
            Availability::Granted | Availability::AtLift | Availability::AtLanding
        )
}

/// Install `tools` into the schema's [`COLLECTION`], one member each, and make
/// it chosen by both the room and provenance — so nothing shows until a turn
/// offers it, and of what it offers the most relevant are shown.
///
/// **Every tool, whatever it needs.** What a body can do changes every tick
/// with who is here, what it carries and where it stands, and the system prompt
/// is sealed once; so the prompt holds all of them and each turn chooses. The
/// room decides which acts are *possible* ([`show_within`] offers exactly the
/// ones the grammar allows), and provenance scoring then ranks those by how
/// well each fits what the character is meant to be doing — so a collection
/// scored alone can no longer show an act the room refuses, and a room can no
/// longer bury the relevant act under every other one it allows. An act
/// offered in one place only is mandatory — see [`shown_wherever_offered`].
///
/// Forced to [`SelectionRule::Offered`] whatever the schema declared, because
/// the selector is this module's to name: a collection authored
/// `always_visible` would offer a character alone in a corridor someone to
/// `tell` something to.
///
/// `installed` is called after each one, with how many are in so far.
pub fn install<'a>(
    builder: &mut Builder,
    tools: impl IntoIterator<Item = &'a Tool>,
    mut installed: impl FnMut(usize, &Tool),
) -> anyhow::Result<usize> {
    let Some(cid) = builder.id_for_system_collection(COLLECTION) else {
        anyhow::bail!(
            "the schema declares no `{COLLECTION}` collection — a character would be held to acts \
             it has never read a word about"
        );
    };
    builder
        .set_collection_selection(
            COLLECTION,
            SelectionRule::Offered {
                selector: COLLECTION.to_string(),
                k: SHOWN_ACTS,
            },
        )
        .map_err(|e| anyhow::anyhow!("selecting `{COLLECTION}` by room and provenance: {e}"))?;
    let mut n = 0;
    for t in tools {
        let id = builder
            .add_section_to_collection(cid, member_of(t), entry(t), 100.0)
            .map_err(|e| anyhow::anyhow!("installing `{}` into `{COLLECTION}`: {e}", t.name))?;
        if shown_wherever_offered(t) {
            builder
                .set_collection_member_mandatory(cid, id)
                .map_err(|e| anyhow::anyhow!("act `{}` mandatory: {e}", t.name))?;
        }
        n += 1;
        installed(n, t);
    }
    Ok(n)
}

/// Show exactly these tools on a turn, replacing whatever it showed before,
/// whatever they score. A reflection or a journal entry is forced into one
/// answer, so the prompt must hold precisely that answer's definition.
pub fn show<'a>(selection: &mut SelectionState, tools: impl IntoIterator<Item = &'a str>) {
    let members: Vec<String> = tools.into_iter().map(member).collect();
    selection.select(
        FORCE_TOOL_SELECTOR,
        members.join(&FORCE_TOOL_SEPARATOR.to_string()),
    );
    selection.select_all(COLLECTION, members);
}

/// Offer the tools a character standing *here* can take — exactly the acts
/// [`specs_within`] builds the turn's grammar from, so the prompt can never
/// describe an act the mask refuses; provenance then ranks them, and shows
/// the most relevant (see [`install`]).
///
/// **Plus the acts only `invoke` reaches.** A routed act is not in the grammar
/// ([`routed`]), so [`specs_within`] never names it — and a character at the
/// command table then read `invoke` and nothing about `report_done`, with an
/// address in the mask for a verb it had never been told the meaning of. Every
/// act behind an address the turn offers is shown with the rest.
pub fn show_within(selection: &mut SelectionState, mode: Mode, within: &Within) {
    let specs = specs_within(mode, within);
    let mut names: Vec<&str> = specs.iter().map(|s| s.name.as_str()).collect();
    for act in invoke_body::acts(within) {
        if !names.contains(&act) {
            names.push(act);
        }
    }
    // Offered, replacing whatever the turn offered before; the collection
    // shows the most relevant of them. A name that was never installed offers
    // nothing.
    let members: Vec<String> = names
        .into_iter()
        .map(|name| match by_name(name) {
            Some(t) => member_of(variant(t, within)),
            None => member(name),
        })
        .collect();
    selection.select_all(COLLECTION, members);
}

/// Offer no act at all: the selection a turn of plain words reads.
///
/// **A turn that is to say words must not be shown what it could call.** The
/// words of a conversation's turn are decoded with `<tool_call>` banned, but
/// under a selection still listing the thread's acts; Zen then wrote its
/// decision out as text — `reply` and the call's JSON, sent to the person as
/// its answer.
pub fn show_none(selection: &mut SelectionState) {
    selection.select_all(COLLECTION, Vec::<String>::new());
}

/// The act `t` is for a character standing *here*: an ordered body's
/// mission acts in place of the agenda's of the same names
/// ([`agenda::ordered`]), and every other act as it is.
pub fn variant(t: &'static Tool, within: &Within) -> &'static Tool {
    match within.orders && agenda::is_mine(t.name) {
        true => ordered::counterpart(t.name).unwrap_or(t),
        false => t,
    }
}

/// The prompt member `t` is installed under: its name, or for an ordered
/// mission act, a member of its own beside the agenda's.
pub fn member_of(t: &Tool) -> String {
    match ordered::is_ordered(t) {
        true => format!("{COLLECTION}/{}", ordered::member(t.name)),
        false => member(t.name),
    }
}
