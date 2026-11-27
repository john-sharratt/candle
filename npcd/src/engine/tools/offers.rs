//! Says which fields a chosen value brings with it, and builds the live sets the
//! world offers for each parameter.
//!
//! Some arguments carry fields of their own: an edit's operation brings the text
//! or destination it takes, and a message's way brings its own addressee. Those
//! fields are written after their value and are not multiplied in beside it, so
//! each operation adds its own paths. The live sets are built in one function that
//! both the grammar and the emptiness check call, because two sources of truth
//! could keep a parameter in one place and build it empty in the other. Duplicate
//! names are dropped in first-mention order, since two identical arms make the
//! whole grammar fail to compile.

use super::*;

use serde_json::{Map, Value};

use crate::sim::missions::Missions;

/// What each value of a choosing parameter brings with it, written after that
/// value and nowhere else. `(tool, param) → [(value, fields)]`.
///
/// **A sibling is multiplied in; a brought field is added.** An edit's fields
/// belong to its operation — a replace takes `text`, a move takes `dest` and
/// `where`, a delete takes nothing — and left as optional siblings of `do` they
/// were offered after every operation, each one doubling the decodes the act
/// admits (`estimated_paths`): with `file_edit` at every bench a crowded room
/// went to 4 × 10⁷ decodes over a turn. Brought by the value, each field is a
/// path of that value's alone, and the operations sum instead of multiply.
///
/// A value with no entry takes nothing. Closed by
/// [`tests::every_brought_field_is_a_real_optional_field`].
pub(super) const BRINGS: &[(&str, &str, &[Needs])] = &[
    (
        "file_edit",
        "do",
        &[
            ("replace", &["text"]),
            ("insert_before", &["text"]),
            ("insert_after", &["text"]),
            ("move", &["dest", "where"]),
        ],
    ),
    (
        "message",
        "via",
        &[(PRIVATE, &["to"]), (CHANNEL_WAY, &["to"])],
    ),
    (
        "message_someone_else",
        "via",
        &[(PRIVATE, &["to"]), (CHANNEL_WAY, &["to"])],
    ),
];

/// A `message` sent to somebody alone — the value of its `via`.
pub const PRIVATE: &str = "private";

/// A `message` sent on a channel or a group — the value of its `via`.
pub const CHANNEL_WAY: &str = "channel";

/// What a field a value brings is chosen from, when that is the world's to
/// say and differs by the value. `(tool, value, field) → Choices`.
///
/// `message`'s `to` is a person for `private` and a channel for `channel`:
/// one name offered from the set the way chose, never from both.
pub(super) const BROUGHT_CHOICES: &[(&str, &str, &str, Choices)] = &[
    ("message", PRIVATE, "to", Choices::Private),
    ("message", CHANNEL_WAY, "to", Choices::Channels),
    ("message_someone_else", PRIVATE, "to", Choices::Private),
    ("message_someone_else", CHANNEL_WAY, "to", Choices::Channels),
];

/// Every field some value of `tool`'s choices brings. See [`BRINGS`].
pub(crate) fn brought(tool: &str) -> Vec<&'static str> {
    BRINGS
        .iter()
        .filter(|(t, _, _)| *t == tool)
        .flat_map(|(_, _, rules)| rules.iter())
        .flat_map(|(_, fields)| fields.iter().copied())
        .collect()
}

/// What `tool`'s `field` is chosen from once its chooser said `value`. See
/// [`BROUGHT_CHOICES`].
pub(super) fn brought_choice(tool: &str, value: &str, field: &str) -> Option<Choices> {
    BROUGHT_CHOICES
        .iter()
        .find(|(t, v, f, _)| *t == tool && *v == value && *f == field)
        .map(|(_, _, _, c)| *c)
}

/// The choices whose every value brings the parameters the world declares for
/// it, rather than fields of the act's own. `(tool, param)`.
///
/// **A standing job's arguments are its own fields.** They were one free
/// `args` string of `name: value` lines, which the grammar let close empty and
/// nothing told the model how to fill: asked to post `coding-task`, Zen wrote
/// `"args": ""` on nine trials of twelve. Chosen as the job, `coding-task`
/// brings `what` and `repository`, each required, and the call cannot close
/// without them.
pub(super) const DECLARED: &[(&str, &str)] = &[("order_post", "job")];

/// The fields a call of `tool` writing `fields` carries beyond the act's own
/// params: for each [`DECLARED`] choice it makes, the parameters the world
/// declares for the value it chose.
///
/// **What a value brought is the act's to read.** Reached through the order
/// table's device, a post kept only `order_post`'s own params, so a
/// `coding-task` written with its `what` and `repository` arrived without them
/// and was refused for want of `what`.
pub fn declared_fields(
    tool: &str,
    fields: &Map<String, Value>,
    missions: &Missions,
) -> Vec<String> {
    DECLARED
        .iter()
        .filter(|(t, _)| *t == tool)
        .filter_map(|(_, param)| fields.get(*param)?.as_str())
        .filter_map(|job| missions.routine(job))
        .flat_map(|r| r.params.clone())
        .collect()
}

/// The fields each value of `t.param` brings, as the grammar's own: those
/// [`BRINGS`] names, and for a [`DECLARED`] choice the parameters the world
/// declares for each value.
pub(super) fn value_shapes(
    t: &Tool,
    param: &str,
    within: &Within,
) -> Vec<(String, Vec<StencilParam>)> {
    let mut shapes = brought_shapes(t, param, within);
    shapes.extend(declared_shapes(t, param, within));
    shapes
}

/// The parameters each standing job declares, as the fields choosing it as
/// `t.param` brings — required, each one of the values the world names for
/// it, or else a free string bounded as an act's is. A parameter named as one
/// of the act's own fields is that field, already offered, and is not brought
/// again.
fn declared_shapes(t: &Tool, param: &str, within: &Within) -> Vec<(String, Vec<StencilParam>)> {
    if !DECLARED.contains(&(t.name, param)) {
        return Vec::new();
    }
    within
        .job_params
        .iter()
        .map(|(job, params)| {
            let fields = params
                .iter()
                .filter(|p| !t.params.iter().any(|own| own.name == p.name))
                .map(|p| StencilParam {
                    name: p.name.clone(),
                    ty: ParamType::String,
                    required: true,
                    enum_values: (!p.choices.is_empty()).then(|| p.choices.clone()),
                    items: None,
                    min_items: 0,
                    max_items: None,
                    properties: None,
                    nullable: false,
                    minimum: None,
                    maximum: None,
                    max_tokens: p.choices.is_empty().then_some(ACT_VALUE_TOKENS),
                    requires: Vec::new(),
                    shapes: Vec::new(),
                    pick: Pick::None,
                    pattern: None,
                })
                .collect();
            (job.clone(), fields)
        })
        .collect()
}

/// The fields [`BRINGS`] names for each value of `t.param`.
fn brought_shapes(t: &Tool, param: &str, within: &Within) -> Vec<(String, Vec<StencilParam>)> {
    BRINGS
        .iter()
        .filter(|(tool, p, _)| *tool == t.name && *p == param)
        .flat_map(|(_, _, rules)| rules.iter())
        .filter_map(|(value, fields)| {
            // Required, as the stencil writes every field a value brings: after
            // the value it is part of the call, not a choice beside it. A
            // field the world enumerates for this value takes that set, and a
            // value whose set is empty brings nothing — its chooser does not
            // offer it, and a branch with no arms would fail the grammar.
            let params = fields
                .iter()
                .filter_map(|f| t.params.iter().find(|p| p.name == *f))
                .map(|p| {
                    let enum_values = match brought_choice(t.name, value, p.name) {
                        Some(c) => Some(live_set(c, within)).filter(|v| !v.is_empty()),
                        None => stencil_param(t, p, within).enum_values,
                    };
                    let offered =
                        brought_choice(t.name, value, p.name).is_none() || enum_values.is_some();
                    offered.then(|| StencilParam {
                        required: true,
                        enum_values,
                        ..stencil_param(t, p, within)
                    })
                })
                .collect::<Option<Vec<_>>>()?;
            Some(((*value).to_string(), params))
        })
        .collect()
}

/// What the world offers for one live parameter, here.
///
/// **One function, called from both places.** The set the grammar is built from
/// and the set the emptiness check consults have to be the same set, or a
/// parameter is kept because the check thinks it has values and then built with
/// an empty branch — which does not fail loudly. It fails by never arming the
/// grammar again, and every turn afterwards free-decodes prose where a call
/// belongs.
pub(super) fn live_set(choice: Choices, within: &Within) -> Vec<String> {
    // **A live set is a set, and nothing upstream enforces that.**
    //
    // Two arms with the same text tokenize to one common prefix with nothing
    // left over, which the stencil refuses as `EmptyArm` — and that refusal is
    // not local. It fails the *whole* grammar, so the turn falls through to a
    // free decode, the model emits something that is not a call, and the
    // character is told "that did not come out as a call" instead of acting.
    // Every turn, for as long as the duplicate exists.
    //
    // This was live: a character opened two groups it had both called the same
    // thing, and its grammar never compiled again. The thread names are the
    // obvious source — a group's name is whatever the character typed — but any
    // set naming things a character chose can collide, so the guard is here, at
    // the one place every world-enumerated branch is built, rather than at each
    // of the fifteen that feed it.
    //
    // Order is preserved: first mention wins, because these lists are already
    // ordered by what the room or the ledger considers most relevant and
    // sorting to deduplicate would throw that away.
    let mut seen: std::collections::BTreeSet<String> = std::collections::BTreeSet::new();
    let mut out = live_values(choice, within);
    out.retain(|v| seen.insert(v.clone()));
    out
}

/// What the world offers, before it is made a set.
pub(super) fn live_values(choice: Choices, within: &Within) -> Vec<String> {
    match choice {
        Choices::Company => within.company.clone(),
        Choices::CompanyOrSelf => {
            // Yourself last: the offered order is the order the model reads, and
            // acting on somebody else is the ordinary case.
            let mut who = within.company.clone();
            who.push(SELF.to_string());
            who
        }
        Choices::Reachable => within.places.clone(),
        Choices::Floors => within.floors.clone(),
        Choices::Years => within.years.clone(),
        Choices::Rooms => within.rooms.clone(),
        Choices::Jobs => within.jobs.clone(),
        Choices::Takers => within.takers.clone(),
        Choices::Links => within.links.clone(),
        Choices::Sonos => within.sonos.clone(),
        Choices::Receiver => within.receiver.clone(),
        Choices::Cast => within.cast.clone(),
        Choices::Lights => within.lights.clone(),
        Choices::Pairable => within.pairable.clone(),
        Choices::Pairing => within.pairing.clone(),
        Choices::Terminals => within.terminals.clone(),
        Choices::Checks => within.checks.clone(),
        Choices::Devices => within.devices.clone(),
        Choices::HouseModes => within.house_modes.clone(),
        Choices::Findings => within.findings.clone(),
        Choices::Unlinked => within.unlinked.clone(),
        Choices::Onboarded => within.onboarded.clone(),
        Choices::Doors => within.doors.clone(),
        Choices::Feelings => within.feelings.clone(),
        Choices::Carried => within.carried.clone(),
        Choices::Equippable => within.equippable.clone(),
        Choices::Usable => within.usable.clone(),
        Choices::Operable => within.operable.clone(),
        Choices::DeviceModes => within.device_modes.clone(),
        Choices::Extractable => within.extractable.clone(),
        Choices::Hostiles => within.hostiles.clone(),
        Choices::Makeable => within.makeable.clone(),
        Choices::Queues => within.queues.clone(),
        Choices::TowerActions => within.tower_actions.clone(),
        Choices::Owed => within.owed.clone(),
        Choices::Claimable => within.claimable.clone(),
        Choices::Readable => within.readable.clone(),
        Choices::Postable => within.postable.clone(),
        Choices::Presentable => within.presentable.clone(),
        Choices::Postures => within.postures.clone(),
        Choices::Threads => within.threads.clone(),
        Choices::Ways => [(PRIVATE, &within.private), (CHANNEL_WAY, &within.channels)]
            .into_iter()
            .filter(|(_, to)| !to.is_empty())
            .map(|(way, _)| way.to_string())
            .collect(),
        Choices::Private => within.private.clone(),
        Choices::Channels => within.channels.clone(),
        Choices::Awaiting => within.awaiting.clone(),
        Choices::Agenda => within.agenda.clone(),
        Choices::Priorities => within.priorities.clone(),
        Choices::Leavable => within.leavable.clone(),
        Choices::Invitable => within.invitable.clone(),
        Choices::Invitees => within.invitees.clone(),
        Choices::Contacts => within.contacts.clone(),
        Choices::InvokeUrl => invoke_body::urls(within),
    }
}

/// Whether this parameter is world-enumerated and the world has nothing to
/// offer for it here.
pub(super) fn live_empty(tool: &str, param: &str, within: &Within) -> bool {
    // **Asked, not assumed, and asked of the same function the spec is built
    // from.** An earlier version answered some of these from a constant on the
    // reasoning that a closed set is never empty. That reasoning was wrong for
    // any set filtered by where the character is standing — which most of them
    // are — and it was wrong in the direction that does not announce itself.
    //
    // Were it to stop being true, the parameter would be *kept* by a check that
    // believed it had values and then *built* with an empty enum: a branch with
    // no arms, the whole catalog failing to compile over it, and the failure is
    // the silent one — no turn is constrained again and every character
    // free-decodes prose where a call belongs. Two sources of truth for one
    // question is the whole bug; there is now one.
    match live_choice(tool, param) {
        // A vocabulary is never *empty* in the sense this question is asking —
        // see [`Choices::steers`]. It falls back to free text, so the parameter
        // stays and its act stays with it.
        Some(c) if !c.steers() => false,
        Some(c) => live_set(c, within).is_empty(),
        // A fixed set is written down in this file and cannot go empty without
        // somebody deleting it, but the same argument applies, so it is checked
        // rather than assumed.
        None => fixed_values(tool, param).is_some_and(|v| v.is_empty()),
    }
}

/// Whether this parameter's values come from the world. See [`LIVE`].
pub fn live_choice(tool: &str, param: &str) -> Option<Choices> {
    LIVE.iter()
        .find(|(t, p, _)| *t == tool && *p == param)
        .map(|(_, _, c)| *c)
}
