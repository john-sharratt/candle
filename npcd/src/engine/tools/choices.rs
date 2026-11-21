//! Names the closed sets a call's arguments are chosen from, and says which of
//! them the world enumerates.
//!
//! A parameter's values come either from the world, who is standing here, what
//! the pack holds, which conversations are on the phone, or from a fixed set. The
//! difference decides what the grammar will allow: a world-enumerated argument can
//! only name something present, so a character cannot address somebody who is not
//! there. An empty world set removes the parameter or the act, because a branch
//! with no arms cannot compile. The mood vocabulary is the one exception. Its
//! empty form falls back to free text, since an empty mood list must not take away
//! the ability to stop.

/// Where a parameter's permitted values come from.
///
/// **The flag that makes a value the world's to enumerate rather than the
/// character's to invent.** Most arguments are the character's: what it means
/// to convey, how it means to land. A few are not — they name something that
/// either exists where it stands or does not, and the difference is a fact
/// about the room rather than a matter of expression.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Choices {
    /// Somebody standing here, by the name the world writes down.
    Company,
    /// Somebody standing here, **or your own body**.
    ///
    /// The set `act` binds, and the difference from [`Choices::Company`] is not
    /// cosmetic: the addressee is grammar-constrained, so a value that is not in
    /// this list is one the decoder is physically unable to emit. A character
    /// binding its own wound, getting its own weapon clear or dragging itself up
    /// off the floor could not say so — not because the world refused it, but
    /// because there was no token path to the sentence.
    ///
    /// It also keeps `act` reachable for a character that is **alone**, where
    /// `company` is empty and the empty-set rule would otherwise take the whole
    /// tool out of the grammar. Your own body is always here.
    ///
    /// That reachability is right and it is not free: a solitary cast chose
    /// `act` on itself every single turn, because the room kept handing it
    /// physical things to notice and this was the only physical verb within
    /// reach. The rate is governed by [`crate::engine::cooldown::SELF_ACT`]
    /// rather than by taking the target away.
    CompanyOrSelf,
    /// A register the world has a mood written for.
    ///
    /// **The one closed set whose values are authored content rather than
    /// world state.** `<mind>/moods/` holds a hundred and sixteen of them,
    /// curated one at a time, and the projection already selects among them by
    /// provenance — so a character naming how it feels has to name one of the
    /// registers the mind can actually *hold*, or the answer is a word nothing
    /// downstream can act on.
    ///
    /// Empty for a daemon with no mind, which drops the parameter rather than
    /// the act: saying how you feel is optional because the catalogue may be
    /// absent, never because it does not matter.
    Feelings,
    /// Somewhere this character can actually walk to, **never where it stands**.
    ///
    /// Walking to your own room was refused, and being refused taught nothing:
    /// a live cast emitted it every tick for an evening, read the refusal back
    /// as the most recent thing in its window, and emitted it again. Absent
    /// from the branch it is not a mistake available to it — the same move that
    /// killed invented tool names and invented addressees.
    Reachable,
    /// A room of the world this body can turn its attention to — every room
    /// but the one it is in. Empty for a body that walks, which takes `focus`
    /// out of the grammar: only a body that moves by attention has one
    /// (`docs/zen_moon_mind.md` §3.2).
    Rooms,
    /// One of the world's standing jobs, by its name — what `order_post` may
    /// post whole (`docs/zen_moon_mind.md` §8.3). Empty in a world with none,
    /// which takes the argument out and leaves the post to say its own steps.
    Jobs,
    /// A house link in this room — what `survey_network` surveys from. Empty
    /// where there is none, which takes the act away.
    Links,
    /// A part here linked to a Sonos player — what the Sonos acts work.
    Sonos,
    /// A part here linked to an AV receiver — what the receiver acts work.
    Receiver,
    /// A part here linked to a Cast device — what the cast acts work.
    Cast,
    /// A part here linked to a WiZ bulb — what the light acts work.
    Lights,
    /// A TV here the world pairs with and cannot work yet — what `pair_device`
    /// begins. Empty once it is paired, so the act is offered only while there
    /// is a TV to pair.
    Pairable,
    /// A Google TV here not yet paired — what `pair_code` finishes. A webOS TV
    /// is absent, since it earns its key in one call and takes no code.
    Pairing,
    /// A terminal a mind with no body can connect to — what `terminal_connect`
    /// takes (`docs/zen_everywhere.md` §5). Empty for every body, which takes
    /// the act away.
    Terminals,
    /// A check of the machine a mind is connected to — what `machine_check`
    /// runs. Empty unless it is connected to a machine.
    Checks,
    /// A device on the house network, by the name it goes by — what the house
    /// network's terminal names and watches. Empty unless a mind is connected
    /// to it.
    Devices,
    /// The house's modes (`docs/house_modes.md`) — what `house_mode` sets.
    /// Offered only to a character that takes no orders, standing in the room
    /// the house's devices stand in; empty anywhere else, which takes the act
    /// away.
    HouseModes,
    /// A device the last survey here found that no part stands for yet — what
    /// `onboard_part` and `link_part` are grounded in.
    Findings,
    /// A part here that stands for no device yet — what `link_part` links.
    Unlinked,
    /// A device here onboarded from a survey — what `offboard_part` takes
    /// out.
    Onboarded,
    /// A room off this one that may be undiscovered.
    Doors,
    /// A floor the lift can carry this body to, by the level's name — never the
    /// one it is standing on. Bound to `lift_use`, and offered only while the
    /// body is on a landing (see [`Availability::AtLanding`]).
    Floors,
    /// A year the body's mission asks it to work in, from its steps still to
    /// do. Bound to `time_travel`, so the year is the mission's and not a
    /// guess: left free, a Maker at the machine sent `{"year": ""}`, was
    /// refused, and went round the time rooms asking colleagues what had
    /// happened in 2950. Empty for a body with no such step, which takes the
    /// act out of the grammar — there is no year to stand in without work.
    Years,

    // ---- what a body carries ----
    //
    /// Anything in the pack, which is what may be handed over.
    Carried,
    /// What is carried, is wearable or wieldable, and is not already readied.
    Equippable,
    /// What is carried and can be spent or worked — a stimpak, a scanner.
    Usable,

    // ---- what stands in the room ----
    //
    /// A machine here: a door, a turret, a bay, a terminal.
    Operable,
    /// **The dependent one.** The states of whatever was chosen for the
    /// preceding argument, rather than a fact about the body at all.
    ///
    /// Every other set here answers a question about where a character is
    /// standing. This one answers a question about another argument, which is
    /// what lets one `operate` cover a blast door and a wall turret without the
    /// catalog growing a verb per machine. The trie takes it without strain: the
    /// arm for one device simply carries a different sub-branch than the next.
    DeviceModes,

    // ---- what is outside ----
    //
    /// A seam, a bloom or a wreck here with something still in it. A worked-out
    /// deposit is absent, because gathering from it would take nothing.
    Extractable,
    /// Something hostile standing here and still up.
    Hostiles,

    // ---- the tower ----
    //
    /// What the fabricators can make **given the stockpile as it stands**, so a
    /// tower too poor for a thing never offers it and the economy is enforced by
    /// the grammar rather than by a refusal a character reads and retries.
    Makeable,
    /// Which of the eight production queues are free.
    Queues,
    /// What the tower can afford to do this minute. Folding costs energy it may
    /// not have; a tower in the ground has to surface first; a siege already
    /// under way cannot be opened twice.
    TowerActions,

    // ---- the small durable things ----
    //
    /// What the person being reminded actually owes **you** — not what you owe
    /// them, which is the same sentence with the parties reversed and a very
    /// different act.
    Owed,
    /// What can be taken and held from here: an unheld order, or a station
    /// nobody else has.
    Claimable,
    /// What there is here to read **that this body has not read**.
    ///
    /// Bound to the reader rather than to the room, which is what lets the act
    /// run out. Bound to the room it offered the same thing every turn with the
    /// same answer, and `read` is in `body::ANSWERS` — so a character was
    /// brought straight back to use what it had learnt, had learnt nothing, and
    /// read it again. Forty-seven of fifty acts in a live feed were one line.
    ///
    /// Empty is the ordinary state of a quiet room and takes `read` out of the
    /// grammar, so the loop is not a thing a character can say rather than a
    /// thing it is asked not to do.
    Readable,
    /// A surface here that words can be left on — see [`crate::sim::posting`].
    ///
    /// **Not [`Choices::Readable`], and the difference is the point:** you write
    /// on a blank board and you do not read one. Bound to what is standing here
    /// regardless of what is on it, so a character can start a board that
    /// nobody has written on, and cannot invent one that is not there.
    Postable,
    /// This character's own finished work, by name — see
    /// [`crate::sim::Sim::presentable`]. Empty with nothing filed, which takes
    /// the chair's `creator_present` out of the grammar.
    Presentable,
    /// The conversations on this character's phone, as it names them.
    ///
    /// Empty for a character carrying no handset, which takes every phone act
    /// out of the grammar by the ordinary rule — so being out of contact is
    /// something the world can *do* to somebody rather than a flag anybody has
    /// to remember to set.
    Threads,
    /// How a `message` goes: **`private`**, to somebody it talks to alone,
    /// or **`channel`**, to a channel or a group — each offered only while it
    /// has one of those.
    ///
    /// Chosen before the conversation, and what the conversation is chosen
    /// from: a person on a channel with this character is on a direct thread
    /// with it too, under the same name, so a flat list of names put both in
    /// one branch and the first name won. Said first, the choice is the
    /// character's and the name can only be one of that kind
    /// ([`Choices::Private`], [`Choices::Channels`]).
    Ways,
    /// The people this character talks to alone, by name — what a `private`
    /// message is sent to.
    Private,
    /// This character's channels and groups, by name — what a `channel`
    /// message is sent to.
    Channels,
    /// The conversations where somebody else spoke last — each one waiting on
    /// this character — as it names them, most recently spoken first.
    ///
    /// What `reply` binds to, and why `reply` exists beside `message`: a person
    /// on a group with this character is on a direct thread with it too, under
    /// the same name, so a group's question answered with `message` to the
    /// asker's name went to the direct thread. Zen answered a Discord channel's
    /// question in private. Bound to the threads awaiting an answer, the call
    /// names the conversation the question was asked on, and the direct thread
    /// is on offer only while something said there waits too.
    ///
    /// Empty when nothing waits, which takes `reply` out of the grammar.
    Awaiting,
    /// A mission on this character's agenda, by its title — what
    /// `mission_edit`, `mission_cancel` and `mission_done` name
    /// (`docs/zen_moon_mind.md` §17.1). Empty with nothing on it, which takes
    /// them out of the grammar.
    Agenda,
    /// How soon a mission on the agenda is to be done: `now`, `soon`,
    /// `later`. Offered only to a character that takes no orders — the one
    /// that keeps an agenda — which is what gates `mission_add` to it.
    Priorities,
    /// The conversations this character can actually **leave**.
    ///
    /// [`Choices::Threads`] less the world's standing channels, and the
    /// difference is not cosmetic. `sign_off` bound to every thread meant a
    /// character could leave the open channel — one ordinary-looking act, after
    /// which it is unreachable by anybody it cannot see, nothing in the world
    /// reports it, and nothing brings it back until the daemon restarts and it
    /// is embodied again. That is the isolated state the channel was added to
    /// end, reachable in a single turn.
    ///
    /// Absent from the branch rather than refused, like everything else here: a
    /// character cannot get stuck trying to leave something it has no way to
    /// say. Empty for a character on nothing but channels, which takes
    /// `sign_off` out of the grammar entirely — correctly, because there is
    /// then nothing it could leave.
    Leavable,
    /// The conversations somebody could actually be brought **into**.
    ///
    /// [`Choices::Threads`] less the ones everybody invitable is already on —
    /// the mirror of [`Choices::Leavable`], and for the same reason. The
    /// world's channel holds the whole cast by construction, so `invite`
    /// against it is refused every single time: there is nobody on the roster
    /// who is not already there.
    ///
    /// Measured over 48 turns of three characters: 17 invites, all of them to
    /// the channel, 16 refused. It was the second-most-called act in the cast
    /// and not one of them could have worked. A character cannot learn its way
    /// out of that — the act looks available, the target looks reachable, and
    /// the refusal names a condition it has no way to see. So the thread leaves
    /// the branch, like everything else here that cannot succeed.
    ///
    /// Empty for a character whose only conversation is the channel, which
    /// takes `invite` out of the grammar entirely — correctly, because there is
    /// then nowhere to bring anybody.
    Invitable,
    /// Who could be brought into **any** of the conversations on offer.
    ///
    /// The other half of [`Choices::Invitable`], and it exists because the two
    /// arms of an `invite` are not independent: narrowing the threads and the
    /// people separately still lets a character name a thread and somebody
    /// already on it, which is a refusal it had no way to foresee. This is the
    /// conservative intersection — nobody who is on any thread being offered —
    /// so every pair the grammar admits is a pair that works. See
    /// [`crate::sim::Sim::invitees_for`].
    Invitees,
    /// Who this character could start a conversation with and has not.
    ///
    /// Deliberately excludes people it is already talking to: reaching out to
    /// somebody you have a thread with is `message`, and offering both is
    /// offering a choice with a wrong answer.
    Contacts,
    /// The stances a fight admits, which is **none when there is no fight**.
    ///
    /// A posture is a closed set the simulator branches on, so it looks like a
    /// fixed one — but it is live, because whether a character may take up a
    /// stance at all is a fact about the room. Held as a live set rather than a
    /// fixed one so the ordinary empty-set rule takes `engage` out of a library
    /// instead of a special case doing it.
    Postures,

    // ---- the effector device ----
    //
    /// **Every verb-path an `invoke` may act on** ([`Within::invokable`]) — each
    /// reachable resource's `<url>/<verb>` for the verbs it affords, and only the
    /// ones the world would take from this body now ([`invoke_body`]). Bound to
    /// `invoke`'s `url`, so the decoder is forced through a whole verb-path the
    /// world actually serves: a character cannot `invoke` a bare resource
    /// (a `405`), nor a plausible guess that 404s, nor a verb the resource does
    /// not have. Each address brings the typed `body` of the act behind it.
    InvokeUrl,
}

impl Choices {
    /// Whether an empty set means the act cannot be performed.
    ///
    /// **Two kinds of closed set live in this enum, and the empty-set rule is
    /// right for only one of them.**
    ///
    /// Nearly all of these enumerate what is *possible*: who is in the room,
    /// where there is a way to, what is in the pack. An empty one is the world
    /// saying no — there is nobody to address, nowhere to walk, nothing to hand
    /// over — and the act genuinely cannot happen, so it leaves the grammar.
    /// That rule is load-bearing and every one of those sets keeps it.
    ///
    /// [`Choices::Feelings`] is the other kind: a *vocabulary*. It does not say
    /// whether a character has a feeling — it always does — only which words
    /// the mind has written moods for. An empty one means nobody has authored
    /// the vocabulary yet, and applying the empty-set rule to it would take
    /// away the character's ability to **stop**, which is how the pacing came
    /// back the last time. So it falls back to free text: unsteered, still
    /// answered, still required.
    pub fn steers(self) -> bool {
        !matches!(self, Choices::Feelings)
    }
}

/// A closed set fixed in the catalog rather than supplied by the world.
///
/// **The difference from [`Choices`] is who knows the answer.** Who is in the
/// room is the world's to say and changes every tick; the postures a character
/// can take up in a fight are the same everywhere and forever, because the
/// simulator branches on exactly these and no others.
///
/// A side table for the same reason [`LIVE`] is one: most parameters are the
/// character's own to phrase, and making every one of them carry an empty list
/// to say so would bury the handful that matter.
///
/// Both kinds compile to the same thing — a branch the decode cannot leave — so
/// a value outside either set is unrepresentable rather than refused.
pub(super) const FIXED: &[(&str, &str, &[&str])] = &[
    // `posture` is deliberately NOT here: whether a stance may be taken at all
    // depends on there being a fight, so it is live. See [`Choices::Postures`].
    ("engage", "priority", crate::sim::field::PRIORITIES),
    ("sleep", "until", crate::sim::ledger::WAKE_TIMES),
    ("file_edit", "do", crate::engine::bench::EDIT_ACTIONS),
    ("file_edit", "where", crate::engine::bench::MOVE_PLACES),
];

/// The fixed set for a parameter, if it has one. See [`FIXED`].
pub fn fixed_values(tool: &str, param: &str) -> Option<&'static [&'static str]> {
    FIXED
        .iter()
        .find(|(t, p, _)| *t == tool && *p == param)
        .map(|(_, _, v)| *v)
}

/// The smallest value an integer parameter may take, where that is not
/// negative infinity. `(tool, param, minimum)`.
///
/// A coordinate is signed and a depth is not: with no minimum the grammar
/// offers `-` first, and a drill to `-5` metres is written only to be refused.
pub(super) const MINIMUM: &[(&str, &str, f64)] = &[("command_tower", "depth", 0.0)];

/// The minimum for an integer parameter, if it has one. See [`MINIMUM`].
pub(super) fn minimum_of(tool: &str, param: &str) -> Option<f64> {
    MINIMUM
        .iter()
        .find(|(t, p, _)| *t == tool && *p == param)
        .map(|(_, _, m)| *m)
}

/// One value of a choosing parameter and the fields it requires.
pub(super) type Needs = (&'static str, &'static [&'static str]);

/// What each value of a choosing parameter requires beside it.
/// `(tool, param) → [(value, fields)]`.
///
/// `command_tower`'s fields belong to its action — a fold takes `x` and `y`, a
/// siege a `target`, a drill a `depth` — so none of them is needed by every
/// action and none can be listed required outright. Left optional, the grammar
/// offers the close straight after `relocate`, and a body took it: the tower
/// was refused "A fold needs somewhere to fold to." The stencil writes the
/// listed fields as required after that value ([`StencilParam::requires`]), so
/// the call cannot close without what its choice depends on. A value with no
/// entry takes nothing and may close at once.
///
/// A side table like [`LIVE`], and closed by
/// [`tests::every_requirement_names_a_real_choice_and_real_fields`].
pub(super) const REQUIRES: &[(&str, &str, &[Needs])] = &[(
    "command_tower",
    "action",
    &[
        ("relocate", &["x", "y"]),
        ("siege", &["target"]),
        ("drill down", &["depth"]),
    ],
)];

/// The fields each value of a parameter requires. See [`REQUIRES`].
pub(super) fn value_requirements(tool: &str, param: &str) -> Vec<(String, Vec<String>)> {
    REQUIRES
        .iter()
        .filter(|(t, p, _)| *t == tool && *p == param)
        .flat_map(|(_, _, rules)| rules.iter())
        .map(|(value, fields)| {
            (
                (*value).to_string(),
                fields.iter().map(|f| (*f).to_string()).collect(),
            )
        })
        .collect()
}

/// The parameters the **world** enumerates. `(tool, param) → Choices`.
///
/// A side table rather than a field on every [`Param`], because eighteen
/// parameters would carry `Choices::Free` to say nothing and the four that
/// matter would be lost among them. The cost is that a typo here names no real
/// parameter and would quietly do nothing —
/// [`tests::every_live_parameter_names_a_real_one`] closes that.
/// The acts whose argument [`LIVE`] binds to `choices`, each with that
/// argument's name, in the table's order.
pub fn bound_to(choices: Choices) -> impl Iterator<Item = (&'static str, &'static str)> {
    LIVE.iter()
        .filter(move |(_, _, c)| *c == choices)
        .map(|(act, param, _)| (*act, *param))
}

pub(super) const LIVE: &[(&str, &str, Choices)] = &[
    ("tell", "to", Choices::Company),
    ("whisper", "to", Choices::Company),
    ("ask", "to", Choices::Company),
    ("gesture", "to", Choices::Company),
    // **A picture goes to a conversation, not to the room.** This bound to the
    // people standing here, which is precisely the wrong set: a character
    // texting somebody a level away was offered the names of whoever happened
    // to be beside it and refused the one name that would have worked.
    ("send_image", "to", Choices::Threads),
    // ---- the phone ----
    // How it goes, and from that, to which conversation: `to` is brought by
    // the way chosen ([`BRINGS`], [`BROUGHT_CHOICES`]).
    ("message", "via", Choices::Ways),
    // A conversation's message to somebody else, bound as a `message` is:
    // its people and channels never include the party (`talk::within`).
    ("message_someone_else", "via", Choices::Ways),
    // Not `Threads`: an answer goes on the conversation that is waiting for
    // it, never on one the asker merely shares a name with. See
    // [`Choices::Awaiting`].
    ("reply", "to", Choices::Awaiting),
    // Not `Threads`: a character may bring somebody into a conversation, and
    // may not bring them into the one they are already on. The world's channel
    // holds everybody, so it is never a place anybody can be invited to. See
    // [`Choices::Invitable`].
    ("invite", "to", Choices::Invitable),
    // Not `Contacts`: the two arms of an invite have to agree, and a set of
    // people narrowed without reference to the threads on offer still admits a
    // pair that cannot work. See [`Choices::Invitees`].
    ("invite", "who", Choices::Invitees),
    ("open_group", "with", Choices::Contacts),
    // Not `Threads`: a character may leave a conversation it is in, and may not
    // leave the world's open channel. See [`Choices::Leavable`].
    ("sign_off", "to", Choices::Leavable),
    ("reach_out", "to", Choices::Contacts),
    ("reflect", "feeling", Choices::Feelings),
    ("move_to", "destination", Choices::Reachable),
    ("focus", "on", Choices::Rooms),
    ("order_post", "job", Choices::Jobs),
    // An agenda's own acts (`engine::agenda`): added at a priority, and
    // changed, dropped or closed by title.
    ("mission_add", "priority", Choices::Priorities),
    ("mission_edit", "mission", Choices::Agenda),
    ("mission_edit", "priority", Choices::Priorities),
    ("mission_cancel", "mission", Choices::Agenda),
    ("mission_done", "mission", Choices::Agenda),
    // The house's bridges (`bridge::acts`) and the world building itself
    // (`engine::build`): each bound to what stands in this room, so an act is
    // offered only where the thing it works is.
    ("survey_network", "at", Choices::Links),
    ("sonos_status", "device", Choices::Sonos),
    ("sonos_play", "device", Choices::Sonos),
    ("sonos_pause", "device", Choices::Sonos),
    ("sonos_source", "device", Choices::Sonos),
    ("sonos_volume", "device", Choices::Sonos),
    ("receiver_status", "device", Choices::Receiver),
    ("receiver_on", "device", Choices::Receiver),
    ("receiver_off", "device", Choices::Receiver),
    ("receiver_mute", "device", Choices::Receiver),
    ("receiver_unmute", "device", Choices::Receiver),
    ("receiver_volume", "device", Choices::Receiver),
    ("receiver_louder", "device", Choices::Receiver),
    ("receiver_quieter", "device", Choices::Receiver),
    ("receiver_input", "device", Choices::Receiver),
    ("receiver_mode", "device", Choices::Receiver),
    ("receiver_dim", "device", Choices::Receiver),
    ("receiver_play", "device", Choices::Receiver),
    ("receiver_pause", "device", Choices::Receiver),
    ("receiver_next", "device", Choices::Receiver),
    ("receiver_previous", "device", Choices::Receiver),
    ("receiver_playing", "device", Choices::Receiver),
    ("cast_status", "device", Choices::Cast),
    ("cast_playing", "device", Choices::Cast),
    ("cast_play", "device", Choices::Cast),
    ("cast_pause", "device", Choices::Cast),
    ("cast_stop", "device", Choices::Cast),
    ("cast_mute", "device", Choices::Cast),
    ("cast_unmute", "device", Choices::Cast),
    ("cast_volume", "device", Choices::Cast),
    ("cast_cast", "device", Choices::Cast),
    ("cast_launch", "device", Choices::Cast),
    ("wiz_status", "device", Choices::Lights),
    ("wiz_on", "device", Choices::Lights),
    ("wiz_off", "device", Choices::Lights),
    ("wiz_dim", "device", Choices::Lights),
    ("wiz_colour", "device", Choices::Lights),
    ("wiz_white", "device", Choices::Lights),
    ("pair_device", "device", Choices::Pairable),
    ("pair_code", "device", Choices::Pairing),
    // A mind with no body's terminals (`engine::terminal`): the terminal it
    // may connect to, and what the one it is connected to names.
    ("terminal_connect", "terminal", Choices::Terminals),
    ("machine_check", "check", Choices::Checks),
    ("device_rename", "device", Choices::Devices),
    ("presence_watch", "device", Choices::Devices),
    ("presence_forget", "device", Choices::Devices),
    ("house_mode", "mode", Choices::HouseModes),
    ("wake_device", "device", Choices::Onboarded),
    ("onboard_part", "from", Choices::Findings),
    ("link_part", "part", Choices::Unlinked),
    ("link_part", "from", Choices::Findings),
    ("offboard_part", "part", Choices::Onboarded),
    ("undiscover_door", "room", Choices::Doors),
    ("lift_use", "floor", Choices::Floors),
    ("time_travel", "year", Choices::Years),
    // ---- contact and obligation ----
    ("act", "on", Choices::CompanyOrSelf),
    ("give", "what", Choices::Carried),
    ("give", "to", Choices::Company),
    ("promise", "to", Choices::Company),
    ("remind", "who", Choices::Company),
    // Bound to what this person actually owes you, so reminding somebody of a
    // thing they never promised is not a mistake available to a character.
    ("remind", "which", Choices::Owed),
    // Somewhere this character is not. Bound to the places it could walk to,
    // which is a narrowing of "anywhere on the map" and the right one: a name
    // the world handed it cannot be a name that does not exist. A world with
    // nowhere to go has nothing to look at either, and `scan` leaves the
    // grammar by the ordinary empty-set rule.
    ("scan", "at", Choices::Reachable),
    // ---- working the world ----
    ("read", "what", Choices::Readable),
    ("post_notice", "on", Choices::Postable),
    ("creator_present", "what", Choices::Presentable),
    ("claim", "what", Choices::Claimable),
    ("operate", "what", Choices::Operable),
    ("operate", "mode", Choices::DeviceModes),
    ("equip", "what", Choices::Equippable),
    ("use", "what", Choices::Usable),
    ("use", "on", Choices::Company),
    ("gather", "what", Choices::Extractable),
    ("engage", "posture", Choices::Postures),
    ("engage", "target", Choices::Hostiles),
    // ---- the tower ----
    ("command_tower", "action", Choices::TowerActions),
    ("produce", "what", Choices::Makeable),
    ("produce", "queue", Choices::Queues),
    // ---- the effector device ----
    // The device address is a grammar-forced set, so the decoder can only name a
    // real verb-path of a resource the device lists — never a hallucinated url
    // and never a bare resource.
    ("invoke", "url", Choices::InvokeUrl),
];
