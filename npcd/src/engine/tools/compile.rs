//! Compiles the act catalogue into the constrained grammar a turn is decoded
//! under, and bounds how many decodes that grammar admits.
//!
//! Each act becomes a spec the decoder is forced through, so a required parameter
//! is unskippable rather than merely requested. The number of acts a turn may
//! contain sets the size of the grammar exponentially, so `estimated_paths` counts
//! the decodes a turn admits, and a catalogue change that would exhaust memory
//! fails a test instead. String arguments are bounded in tokens, so a runaway value
//! closes inside the turn with its call. Parameter types the grammar cannot bound
//! are refused by the tests rather than compiled.

use super::*;

/// How many acts one turn may contain.
///
/// **A bound, not a target.** The grammar offers the closing arm after every
/// call, so a character with one thing to do does one thing. What the bound
/// removes is the unbounded loop a model can sit in, which is the same runaway
/// the reasoning block had in a different costume.
///
/// Two, because the catalog's own guidance is that acts may be combined when
/// they "genuinely go together" — turning as you speak, moving as you signal —
/// and both of those examples are pairs. Beyond a pair a character is no longer
/// acting in a moment, it is narrating a plan; the turn comes round again in
/// half a second.
///
/// **It is also the exponent on the decodes a turn admits, which is what fixed
/// the number.** A turn admitting `p` paths per act admits `p^ACTS_PER_TURN`.
/// Measured on the shipped catalog in a busy room: `p` is about 360, so four acts
/// is 1.7 × 10¹⁰ paths — a compile that allocated ~390 MB a second without bound
/// and took the machine down twice. Two acts is about 130,000, which compiles in
/// milliseconds.
///
/// [`tests::a_full_room_stays_inside_the_path_budget`] holds the whole product
/// under [`MAX_TURN_PATHS`], so a future act with two enumerated arguments fails
/// a test rather than a machine.
pub const ACTS_PER_TURN: usize = 2;

/// How many acts a conversation's decision may contain
/// (`docs/npc_conversations.md` §6.2): a promise or a mission change, and the
/// reply beside it.
///
/// **Two, for the same reason as the main loop's.** The thread's catalog is
/// small, but `reflect` binds the whole mood library: with the shipped mind's
/// hundred and sixteen moods and an agenda to change, one act is about three
/// hundred paths, so three acts admitted 2.6 × 10⁷ decodes — over
/// [`MAX_TURN_PATHS`] twelve times — and two admit under 10⁵.
pub const TALK_ACTS: usize = 2;

/// The most tokens one argument of an act may run to before the grammar closes
/// it where it stands.
///
/// **A runaway value must end inside the turn, with its call.** Unbounded, a
/// `reflect` whose `situation` fell into "I am standing in the catalogue." over
/// and over ran to the turn's 16k-token cap, the call never closed, and the
/// character was told only that it had not written a call — every turn, for
/// twelve hours, each one sixteen thousand tokens long. An act's argument is a
/// sentence or a passage; a document is written whole through `compose`, which
/// has its own budget. Bounded here, the value closes, the call lands, and the
/// character reads back what it actually did.
pub const ACT_VALUE_TOKENS: u32 = 1024;

/// The whole catalog as constrained-decoding specs.
///
/// **This is what makes a required parameter actually required.** Compiled into
/// a stencil ([`candle_conversation::ConversationEngine::compile_tool_stencil`])
/// and armed on every thinking step, a required param becomes a `Static` node
/// the decoder is forced through — the grammar emits `"to": "` itself and the
/// model only chooses the string inside it. Omitting one stops being unlikely
/// and becomes unreachable, and the JSON frame is emitted by the grammar rather
/// than hoped for.
///
/// Without it the catalog is documentation: a character read the tool list in
/// its prompt and free-decoded whatever shape it inferred. One of them spent an
/// entire evening calling `tell` with no `to` — rejected every single time,
/// never once managing to speak — while its companion, hearing nothing, looped
/// on the same sentence for want of anything else to react to.
///
/// **The whole catalog, in the shape a grammar with no world to consult can
/// hold it.** Every tool, every value free. [`specs_within`] is what a live
/// character gets; this is the fallback for a caller that has no room to look
/// at — the probe, and the tests.
pub fn specs() -> Vec<ToolSpec> {
    specs_within(Mode::Physical, &Within::nowhere())
}

/// [`specs`] with an ordered body's mission acts in place of the agenda's
/// ([`agenda::ordered`]): the catalog a conversation's call for a character
/// that takes orders is read against where it lands, which has no `Within`
/// of the turn that decided it.
pub fn specs_ordered() -> Vec<ToolSpec> {
    specs_within(
        Mode::Physical,
        &Within {
            orders: true,
            ..Within::nowhere()
        },
    )
}

/// The catalog as a grammar, for a character standing *here*.
///
/// # Two things the world decides, not the character
///
/// **Which acts are reachable at all.** A stencil that offers `ask` to a
/// character alone in a corridor is offering a field, and a model handed a
/// field fills it in — it invents somebody to address, and the world refuses,
/// and nothing about being refused teaches it not to. Advertising the act only
/// in the prompt was never enough: the prompt is a request and the mask is the
/// guarantee, and they disagreed. Alone, the acts that need company are not in
/// the branch, so they are unreachable rather than discouraged.
///
/// **Which names an addressee may be.** `to` used to be free text, so a
/// character could write any string and did: it asked for "Perrin" when the
/// world had written down "Perrin Vastwood", was refused, and asked the same
/// way on every tick for as long as it stood there. Constrained to the names
/// actually present, the wrong name is not a mistake it can make — the same
/// reason an invented *tool* name is not one.
///
/// Rebuilt per turn because who is here changes per turn; see
/// `mind::Minds::grammar_for`, which caches on exactly this value.
pub fn specs_within(mode: Mode, within: &Within) -> Vec<ToolSpec> {
    CATALOG
        .iter()
        .map(|t| variant(t, within))
        // **An act that has migrated to the effector device is not in the
        // grammar** — it is reached by `invoke` on its resource, not
        // offered as a compiled call (effector design Step 6). The `Tool` stays
        // in the catalogue because the device route builds its schema from it;
        // this is the one place it leaves the *grammar*. See [`routed`].
        .filter(|t| !routed(t.name))
        .filter(|t| match t.availability {
            Availability::Always => true,
            Availability::MessagingOnly => mode.remote(),
            Availability::Pictorial => mode.carries_pictures(),
            Availability::PhysicalOnly => mode == Mode::Physical,
            // A situation is composed for a body standing somewhere, so
            // anything reaching this far has one. A disembodied mind is gated
            // in [`for_body`], which is what builds its prompt.
            Availability::Embodied => true,
            Availability::Nearby => !within.alone(),
            Availability::PersonNearby => !within.people.is_empty(),
            Availability::PersonAmongOthers => {
                !within.people.is_empty() && within.company.len() >= 2
            }
            Availability::PersonInEarshot => within.heard_by_person || within.in_field,
            // A journey home is only a journey from somewhere else.
            Availability::AwayFromHome => within.away_from_home,
            // Only a body with something in its hands can give it back.
            Availability::Holding => !within.held.is_empty(),
            // The map decides. A station in the room is what puts its acts in
            // reach, and walking out takes them with you.
            Availability::AtPart => within.station.iter().any(|s| s == t.name),
            // On a landing, calling the car is offered only while it is away;
            // riding it only while it is open here. The two never overlap, so a
            // body at the lift is offered exactly one of them.
            Availability::AtLift => within.at_lift && !within.lift_here,
            Availability::AtLanding => within.at_lift,
            Availability::Granted => within.grants.iter().any(|g| g == t.name),
            Availability::Reached => within.reaches.iter().any(|r| r == t.name),
            Availability::OnThread => within.allow.is_some(),
        })
        .filter(|t| performable(t, within))
        .map(|t| ToolSpec {
            name: t.name.to_string(),
            params: stencil_params(t, within),
        })
        .collect()
}

/// Whether a body standing *here* could do this act right now, apart from
/// whether the act is offered by name: it is not cooling, and every required
/// argument has something to choose from.
///
/// The one test both ways of reaching an act ask — the compiled call in
/// [`specs_within`] and the `invoke` address in [`invoke_body`] — so a station
/// act is offered at its address exactly when it would be offered as a call.
///
/// **A body that has just done this cannot do it again yet.** Absent rather than
/// refused, for the reason every other absence here is: a refusal is the most
/// recent thing in the character's window, and a live cast read its own refusals
/// back and emitted the same act again.
///
/// **An act whose required argument has nothing to choose from cannot be
/// performed**, so it goes. An optional parameter with an empty set drops the
/// parameter; a required one drops the act, because a branch with no arms is not
/// a constraint but an unrepresentable node, and the whole grammar fails to
/// compile over it — silently, since the only symptom is that no turn is ever
/// constrained again. A character in a room with nowhere to walk to genuinely
/// cannot `move_to`.
pub(in super::super) fn performable(t: &Tool, within: &Within) -> bool {
    within
        .allow
        .as_ref()
        .is_none_or(|allow| allow.iter().any(|a| a == t.name))
        && !within.cooling.iter().any(|c| c == t.name)
        && !t
            .params
            .iter()
            .any(|p| p.required && live_empty(t.name, p.name, within))
}

/// One act's parameters as the grammar's own, for a character standing *here*.
///
/// The one mapping every grammar is built through: the act's own call, and the
/// `body` object each `invoke` address carries ([`invoke_body`]) — so a station
/// act is decoded against the same enums whether it is named as a call or
/// reached through the device.
pub(in super::super) fn stencil_params(t: &Tool, within: &Within) -> Vec<StencilParam> {
    t.params
        .iter()
        // **An empty set removes the parameter; it never empties its branch.** A
        // branch with no arms is not a constraint, it is an unrepresentable node,
        // and the whole catalog fails to compile over it — which fails *quietly*:
        // the grammar simply does not arm, every turn free-decodes, and what
        // comes back is a model writing an essay where a call should be.
        //
        // Only optional parameters reach this. A required one with nothing to
        // choose from means the act itself is impossible, and those are gated a
        // level up by `Availability::Nearby` — `ask` and `tell` are absent when
        // alone rather than present with nobody to name. `gesture` is the case
        // this is for: aimed at nobody it is still a thing you can do, so the aim
        // drops and the act stays.
        .filter(|p| !(live_empty(t.name, p.name, within) && !p.required))
        // A field some value brings ([`BRINGS`]) is written only after that
        // value, never beside the choice as a sibling.
        .filter(|p| !brought(t.name).contains(&p.name))
        .map(|p| stencil_param(t, p, within))
        .collect()
}

/// One parameter of `t` as the grammar's own, for a character standing *here*.
pub(super) fn stencil_param(t: &Tool, p: &Param, within: &Within) -> StencilParam {
    StencilParam {
        name: p.name.to_string(),
        ty: param_type(p.ty),
        required: p.required,
        // Free unless the world enumerates it. A closed set over an argument
        // the character *means* would be the machinery writing its lines; a
        // closed set over one that merely names something present is the
        // machinery declining to let it name what is not.
        enum_values: match live_choice(t.name, p.name) {
            // **A vocabulary with nothing in it steers nothing.** The empty
            // set here would be a branch with no arms, so it falls back to
            // free text — see [`Choices::steers`] for why that is right for a
            // vocabulary and wrong for everything else.
            Some(c) => match live_set(c, within) {
                v if v.is_empty() => None,
                v => Some(v),
            },
            // Not world-enumerated. A fixed set is still a closed branch —
            // the difference is only who computed it.
            None => {
                fixed_values(t.name, p.name).map(|v| v.iter().map(|s| (*s).to_string()).collect())
            }
        },
        // An act's arguments are scalars, or a list of lines ([`LIST`]) —
        // at least one, at most [`LIST_MOST`], each a line opening on
        // something written. None may be null.
        items: (p.ty == LIST).then(|| Box::new(line_item())),
        min_items: usize::from(p.ty == LIST),
        max_items: (p.ty == LIST).then_some(LIST_MOST),
        properties: None,
        nullable: false,
        minimum: minimum_of(t.name, p.name),
        maximum: None,
        max_tokens: (p.ty == "string").then_some(ACT_VALUE_TOKENS),
        requires: value_requirements(t.name, p.name),
        // **What each value carries is decided by which value it is.**
        // `invoke`'s `url` brings the typed body its endpoint accepts, in
        // place of a free `body` beside it; a choice in [`BRINGS`] brings
        // the fields that value takes.
        shapes: match live_choice(t.name, p.name) {
            Some(Choices::InvokeUrl) => invoke_body::shapes(within),
            _ => value_shapes(t, p.name, within),
        },
        pick: pick_of(t.name, p.name),
        pattern: None,
    }
}

/// The acts whose optional fields make a choice the call must make — at least
/// one of these written — so that an empty call does not parse. `(tool, the
/// members)`.
///
/// **A call of all-optional fields closes at `{}` unless it must choose.**
/// `order_post` takes a standing `job`, or a `brief` and its `steps`: each is
/// optional because the other may stand in its place, and with nothing
/// requiring either, the empty post would be the cheapest path through the
/// grammar.
///
/// **The choice is the work, not its description.** Chosen between `job` and
/// `brief`, a brief alone closed the call — a post with no step, which the
/// table refuses. Between `job` and `steps`, a post is either a standing job or
/// a list of at least one step ([`LIST`]); a brief written without its steps
/// cannot close.
pub(super) const PICKS: &[(&str, &[&str])] = &[("order_post", &["job", "steps"])];

/// What `param` of `tool` owes a choice among the act's fields.
pub(super) fn pick_of(tool: &str, param: &str) -> Pick {
    match PICKS
        .iter()
        .any(|(t, members)| *t == tool && members.contains(&param))
    {
        true => Pick::AtLeastOne,
        false => Pick::None,
    }
}

/// How many distinct decodes a turn's grammar admits, near enough to budget it.
///
/// **A budget on what the grammar means, not on what the compile costs.** The
/// compile memoises on `(spec node, left context)`, and a choice's successors
/// are shared, so the lowered tree grows with the number of *distinct* nodes —
/// which [`invoke_body`]'s tests hold separately. The path count is the larger
/// number: every combination of arguments the grammar admits, which is what a
/// change to the catalog multiplies. A catalog that keeps it inside
/// [`MAX_TURN_PATHS`] keeps the tree small whatever the sharing does, and an act
/// with three enumerated arguments is still the thing this catches first.
///
/// A parameter contributes its enum's width, or 1 when it is free text (one
/// path, whatever the model writes). An *optional* parameter contributes
/// `1 + width`, because absent is a path of its own. A parameter whose values
/// bring fields of their own ([`StencilParam::shapes`]) contributes the *sum*
/// over its values of those fields' paths — each value is one shape, not a
/// factor.
///
/// This is not theoretical. Nine all-string acts gave about eight paths an act —
/// four thousand over a turn, and instant. Adding one act with three enumerated
/// arguments took it to six hundred an act, which is 10¹¹ over a turn: the
/// compile allocated about 390 MB a second, without bound, and took the machine
/// down twice before the cause was understood.
///
/// `calls` is how many acts the turn's loop admits: [`ACTS_PER_TURN`] for the
/// main loop, [`TALK_ACTS`] for a conversation's decision.
pub fn estimated_paths(specs: &[ToolSpec], calls: usize) -> u128 {
    let per_act: u128 = specs.iter().map(|t| paths_of(&t.params)).sum();
    per_act.saturating_pow(calls as u32)
}

/// The paths through a list of sibling parameters: the product of each one's.
pub(super) fn paths_of(params: &[StencilParam]) -> u128 {
    params
        .iter()
        .fold(1u128, |acc, p| acc.saturating_mul(param_paths(p)))
}

/// The paths through one parameter, absent included when it is optional.
pub(super) fn param_paths(p: &StencilParam) -> u128 {
    let choices = if !p.shapes.is_empty() {
        p.shapes
            .iter()
            .map(|(_, fields)| paths_of(fields))
            .fold(0u128, u128::saturating_add)
    } else if let Some(fields) = &p.properties {
        paths_of(fields)
    } else {
        p.enum_values.as_ref().map_or(1, |v| v.len() as u128)
    };
    if p.required {
        choices
    } else {
        choices.saturating_add(1)
    }
}

/// The most paths a turn's grammar may admit before it is refused.
///
/// Sized from what is known to work rather than from a theory: the shipped
/// catalog in a busy room compiles in milliseconds well under this, and the
/// runaway was eight orders of magnitude above it. A tree at this bound is a few
/// hundred thousand nodes — large, and finite.
pub const MAX_TURN_PATHS: u128 = 2_000_000;

/// This catalog's type names, as the stencil's types.
///
/// Exhaustive rather than defaulting: an unrecognised type would compile to
/// "any JSON value", which is a silently weaker constraint than the one the
/// catalog asked for, on the one path whose whole job is to be exact.
pub(super) fn param_type(ty: &str) -> ParamType {
    match ty {
        "string" => ParamType::String,
        "integer" => ParamType::Integer,
        "number" => ParamType::Number,
        "boolean" => ParamType::Boolean,
        "array" => ParamType::Array,
        other => unreachable!("tool parameter type `{other}` has no stencil type"),
    }
}
