//! The act vocabulary — what a character can *do* with a thinking step.
//!
//! # Tools carry intent, not output
//!
//! This is the single most important property in the module and the easiest to
//! erode. `speak` does not take a sentence. It takes what the character *means*
//! to convey, and the narrator renders that into words. `move_to` does not take
//! coordinates; it takes somewhere the character means to be.
//!
//! The reason is the mind design's "narrate acts, never fabricate", one level
//! deeper: the mind decides substance, the surface decides wording, and neither
//! can produce what the other did not license. A `speak` that took a finished
//! sentence would let the model write dialogue directly, and every guarantee
//! about acts being grounded in gathered state would become a guarantee about
//! nothing.
//!
//! It is easy to erode because "just let it write the line" is always the
//! shortest path to a demo that reads well. [`tests::speech_tools_take_intent_not_prose`]
//! is what makes that erosion a build failure instead of a slow drift.
//!
//! # The write-protection is against the model, not the operator
//!
//! No tool here writes to the belief layer. A character can `note_concern` that
//! a belief is under pressure, and can `speak` about the tension — but it cannot
//! resolve it. Beliefs move only through the evidence-threshold process on the
//! sleep clock, which is `engine::sleep`'s business. An operator authoring a
//! character *can* write beliefs, through the authoring API; that plane is
//! separate and is not reachable from here. [`tests::no_tool_writes_beliefs`]
//! holds the line.
//!
//! # What the character reads
//!
//! Every tool is installed at startup as one member of the mind's `tools`
//! collection — its call, what it is for, and what each parameter takes; see
//! [`entry`] and [`install`]. Each turn then shows exactly the members its
//! grammar offers ([`show_within`]), so the list a character reads and the mask
//! it decodes under are one computation over the same world. A grammar alone
//! guarantees the shape of a call and says nothing about what an act is *for*,
//! and a character that has never read a word about `reflect` does not reflect.
//!
//! Each tool also carries `examples` — a concrete situation, the call a
//! well-calibrated character makes in it, and why that call rather than a
//! neighbouring one. They are what an act is written against, not what the
//! model reads; a tool with none has not been thought through, so an
//! example-less tool is a build failure here rather than a quiet regression.

use candle_conversation::projection::{Builder, SelectionRule, SelectionState};
use candle_conversation::stencil::{Param as StencilParam, ParamType, ToolSpec};
use serde::Serialize;

use crate::engine::{acts, bench, station};

/// What a tool changes, which decides where it can be used and what it must be
/// checked against.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Plane {
    /// Append-only and conflict-free. Cannot fail against the world.
    Speech,
    /// Mutates the world; commits through an arbiter carrying the world-version
    /// the character reasoned over.
    World,
    /// Writes to the character's own non-belief layers. No arbiter needed — no
    /// one else can conflict with it.
    Internal,
    /// Changes only the loop's own scheduling.
    Meta,
}

/// Where a tool is offered.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Availability {
    Always,
    /// Any channel where the parties are apart — text, voice or video.
    ///
    /// Ending a conversation and opening one both belong here: standing up and
    /// walking out is how you leave a room, and down a line there is nothing
    /// else.
    MessagingOnly,
    /// Only while standing at a part that carries this act.
    ///
    /// **What a body can do is a function of where it is, and the map is what
    /// says so.** A part declares the tools being within reach of it makes
    /// available, and a Maker at an easel cannot write a chronicle entry — not
    /// because a rule forbids it, but because the thing that does that is two
    /// floors up.
    ///
    /// This is a stronger guarantee than refusing the call: an act that is not
    /// in the branch cannot be reasoned about wrongly, and adding a station to
    /// a room is adding a line of YAML rather than a case to a match.
    AtPart,
    /// Channels a picture can travel down: video and text, never voice.
    ///
    /// A character standing in front of you does not text you a photo, and a
    /// character on a phone call cannot send one at all. Absent rather than
    /// present-and-refused, so the model is never invited to try.
    Pictorial,
    /// Only while somebody else is in the room.
    ///
    /// **Decided per turn, from the world.** Who is standing next to you
    /// changes every tick, so whether this is offered is computed from the room
    /// on each turn — in the grammar, in the prompt's act list (see
    /// [`show_within`]), and in the situation's line naming who is here.
    ///
    /// Absent rather than present-and-refused, which is the same discipline
    /// `MessagingOnly` follows: a character alone in a corridor should never be
    /// invited to address somebody, because there is nobody to address and the
    /// invitation is what makes it try.
    Nearby,
    /// Only while standing somewhere that is not home.
    ///
    /// **Being called back to where you already are is not a thing that can
    /// happen**, and it was the most-taken act in the cast: twenty-four of
    /// sixty, because `World::place` accepts a move to the room the body is
    /// already in, reports "the ground goes out from under you", and changes
    /// nothing. The characters could see it and could not stop —
    /// *"I'm standing here again. The same loop."*
    ///
    /// `recall` takes no arguments, so the ordinary empty-set rule has no
    /// parameter to empty and cannot reach it. The condition is about the body
    /// rather than about anything it could name, which is exactly what this
    /// ladder is for — the same shape as [`Availability::Nearby`], where what
    /// decides is a fact about the room rather than a value in the call.
    AwayFromHome,
    /// Only for a mind with a body.
    ///
    /// **Not a special case for one character.** Keeper has no body and never
    /// will — no embodiment path exists and none could be built — so every act
    /// it has runs through the tower's own systems. Offered the ordinary
    /// catalog it would be invited to walk, to reach out and touch somebody, and
    /// to dig; and a model handed a field fills it in.
    ///
    /// Any mind without a body gets the same treatment: a tower consciousness,
    /// an uploaded lord between avatars, a companion whose body is destroyed.
    /// The world decides which, not the roster.
    Embodied,
    /// Face to face only.
    ///
    /// The mirror of [`Availability::MessagingOnly`]. You cannot put a hand on
    /// somebody down a voice line, and being invited to is what makes a
    /// character try it.
    PhysicalOnly,
}

/// One parameter, as the model sees it.
#[derive(Clone, Copy, Debug, Serialize)]
pub struct Param {
    pub name: &'static str,
    pub ty: &'static str,
    pub required: bool,
    pub description: &'static str,
}

/// A worked example: the situation, and the call a well-calibrated character
/// makes in it.
#[derive(Clone, Copy, Debug, Serialize)]
pub struct Example {
    /// What the character is facing, in the narrator's voice.
    pub situation: &'static str,
    /// The arguments, as JSON. Held as a string so the table stays `const`.
    pub call: &'static str,
    /// Why this call and not a neighbouring one. The discriminating detail is
    /// what calibration actually teaches — an example that shows a correct call
    /// without showing what made it correct teaches the shape and not the choice.
    pub because: &'static str,
}

#[derive(Clone, Copy, Debug, Serialize)]
pub struct Tool {
    pub name: &'static str,
    pub category: &'static str,
    pub plane: Plane,
    pub availability: Availability,
    pub description: &'static str,
    pub params: &'static [Param],
    pub examples: &'static [Example],
    /// The parts this act attaches to, by the id the map gives them.
    ///
    /// **The tool names the station, not the other way round.** The parts used
    /// to carry a `tools:` list, which meant three things went wrong at once:
    /// adding one act to six stations was six edits in six files; a typo named
    /// no real act and silently produced nothing; and `npc-map` — a crate whose
    /// whole job is describing buildings — carried this engine's vocabulary,
    /// which is a layer inversion.
    ///
    /// Declared here, an act's reach sits beside its description and examples,
    /// the act's own name is a Rust identifier the compiler checks, and the map
    /// goes back to saying only what a building *contains*.
    ///
    /// Empty for everything a body can do anywhere. Non-empty implies
    /// [`Availability::AtPart`], and [`tests::a_station_act_names_where_it_is`]
    /// holds the two together.
    pub at: &'static [&'static str],
}

/// Speech to the room, and speech to a person.
///
/// **Two tools, not one with an optional target.** Saying something aloud to a
/// room and addressing one person in it are different acts, and as one tool
/// with an optional `to` a character is invited to name somebody every turn —
/// which it does, because a model handed a field fills it in. Split, the
/// difference is also legible in what everybody else perceives, because the
/// world carries who an utterance was aimed at and renders it three ways.
///
/// # Both of them need somebody to hear
///
/// This was `Always`, on the reasoning that speaking aloud is possible whether
/// or not anybody is listening. It is, and it was still wrong: **speech into an
/// empty room reaches nobody and changes nothing**, which by this catalog's own
/// standard is an act that does not act.
///
/// What it did instead was worse than nothing. Three characters stood alone in
/// three rooms, and every time the building did something — a box sagging, a
/// smell of hot plastic, a shadow moving under a bench — each of them said it
/// back out loud, paraphrased, to nobody:
///
/// ```text
/// perceived: A smell of hot plastic comes and goes with no source anybody could point at.
/// act:       say — The scent of burnt plastic drifts through the air, vanishing as
///                  quickly as it arrived, with no source I can find.
/// ```
///
/// That is not a character reacting to its world, it is a character narrating
/// it — and twelve of the last fourteen acts in the feed were exactly this.
/// Absent when alone, it is not a thing the character can do, which is the same
/// discipline `tell` and `ask` already follow and the reason they follow it.
const SAY: Tool = Tool {
    name: "say",
    at: &[],
    category: "Speech",
    plane: Plane::Speech,
    availability: Availability::Nearby,
    description: "Say something aloud, to whoever is here. You give what you MEAN — the \
                  substance and the stance — not the sentence; the narrator renders your intent \
                  in your own voice. Everyone in the room hears it. Nobody outside it does.",
    params: &[
        Param {
            name: "intent",
            ty: "string",
            required: true,
            description: "What you mean to convey. Substance, not wording: \"that the redoubt \
                          burned twice and both are written down\" — never a finished line of \
                          dialogue.",
        },
        Param {
            name: "manner",
            ty: "string",
            required: false,
            description: "How it is meant to land — flatly, warmly, as a warning. Colours the \
                          rendering; never becomes words itself.",
        },
    ],
    examples: &[
        Example {
            situation: "You are in the green room with two others. You have just found the \
                        same era written up twice, differently, and neither version says which \
                        is right.",
            call: r#"{"intent":"that the third era is written twice and the two do not agree, and that I would like to know which of them anybody has been working from","manner":"plainly"}"#,
            because: "To the room, not to a person — it is a question for whoever happens to \
                      know, and naming one of them would be guessing at who that is.",
        },
        Example {
            situation: "You have just walked into a room where somebody is working and you \
                        have nothing to ask them.",
            call: r#"{"intent":"that I am here and not going to interrupt","manner":"brief"}"#,
            because: "Arriving somewhere silently and standing there is worse than saying so. \
                      An intent can be small.",
        },
    ],
};

const TELL: Tool = Tool {
    name: "tell",
    at: &[],
    category: "Speech",
    plane: Plane::Speech,
    // Offered only while somebody is here to be told. A character alone has
    // nobody to name, and being invited to name one is what makes it invent a
    // person who is not there.
    availability: Availability::Nearby,
    description: "Say something to one person here, by the name you know them by. Everyone in \
                  the room still hears it — they simply hear that it was for them, not for \
                  you. Use this when it is meant for one of them; use `say` when it is for \
                  whoever is listening.",
    params: &[
        Param {
            name: "to",
            ty: "string",
            required: true,
            description: "Who you are addressing, exactly as their name appears where you are. \
                          They must be here — you cannot address somebody in another room.",
        },
        Param {
            name: "intent",
            ty: "string",
            required: true,
            description: "What you mean to convey. Substance, not wording: \"that I will not \
                          hand over the ledger, and that pressing me will cost him\" — never a \
                          finished line of dialogue.",
        },
        Param {
            name: "manner",
            ty: "string",
            required: false,
            description: "How it is meant to land — flatly, warmly, as a warning. Colours the \
                          rendering; never becomes words itself.",
        },
    ],
    examples: &[
        Example {
            situation: "Maker-04 is in the green room with you. It has been working on the \
                        third era, and you have just found that era written up twice.",
            call: r#"{"to":"Maker-04","intent":"that the third era is written twice and the two do not agree, and that I would like to know which of them it has been working from","manner":"plainly"}"#,
            because: "Aimed at the one person who can answer it. Anyone else in the room hears \
                      it too and hears that it was not for them, which is the difference \
                      between being asked and overhearing a question.",
        },
        Example {
            situation: "Somebody has just come in and said they are not going to interrupt.",
            call: r#"{"to":"Maker-02","intent":"that they are not interrupting and I would rather have the company","manner":"warmly"}"#,
            because: "A reply is addressed. Saying it to the room would leave the person who \
                      spoke to you unsure it was meant for them.",
        },
    ],
};

/// A question put to somebody here.
///
/// **Not a variant of `tell` — the act with a different social obligation.** A
/// statement leaves the other party free to do nothing, and free is what they
/// did: two characters shared a room for a hundred turns while one restated the
/// same finding and the other, hearing it every second turn, walked on the spot.
/// Neither was malfunctioning. Nothing had been *asked*.
///
/// It renders as speech and lands as speech; what it changes is that the
/// listener perceives a question, which has an obvious next act where a
/// statement has none.
const ASK: Tool = Tool {
    name: "ask",
    at: &[],
    category: "Speech",
    plane: Plane::Speech,
    // Same rule as `tell`: a character alone has nobody to ask, and being
    // offered the act is what makes it invent somebody.
    availability: Availability::Nearby,
    description: "Ask one person here something, by the name you know them by. Everyone in the \
                  room hears it and hears who it was for. Use this when you want an answer — a \
                  statement lets them say nothing, a question does not.",
    params: &[
        Param {
            name: "to",
            ty: "string",
            required: true,
            description: "Who you are asking, exactly as their name appears where you are. They \
                          must be here.",
        },
        Param {
            name: "about",
            ty: "string",
            required: true,
            description: "What you want to know. Substance, not wording: \"which of the two \
                          versions of the third era they have been working from\" — never a \
                          finished line of dialogue.",
        },
        Param {
            name: "manner",
            ty: "string",
            required: false,
            description: "How it is meant to land — plainly, warily, as a challenge. Colours the \
                          rendering; never becomes words itself.",
        },
    ],
    examples: &[
        Example {
            situation: "Maker-04 is here. You have just found the third era written up twice, \
                        differently, and you cannot tell which is right.",
            call: r#"{"to":"Maker-04","about":"which of the two versions of the third era it has been working from","manner":"plainly"}"#,
            because: "Only one person can answer this, and asking obliges them to. Saying it as \
                      a finding would have left them free to say nothing.",
        },
        Example {
            situation: "Somebody has been in the room with you for a while and you have both \
                        been working in silence.",
            call: r#"{"to":"Maker-02","about":"what they are working on"}"#,
            because: "The plainest way out of a silence, and it costs nothing to be wrong about \
                      whether they wanted to talk.",
        },
    ],
};

const MOVE_TO: Tool = Tool {
    name: "move_to",
    at: &[],
    category: "Movement",
    plane: Plane::World,
    // Walking needs legs. A tower consciousness has none and never will, so the
    // act is absent for it rather than refused — offered, it would be tried.
    availability: Availability::Embodied,
    description: "Go somewhere, named the way you know it rather than as coordinates. You name \
                  the place; getting there is not your business — you will be told when you \
                  arrive, and told if you never do. Somewhere you have no memory of is refused \
                  before you stand up.",
    params: &[
        Param {
            name: "destination",
            ty: "string",
            required: true,
            description: "A place you know: \"the green room\", \"band one\", \"the command \
                          room\". Somewhere on your own level unless you say otherwise.",
        },
        Param {
            name: "urgency",
            ty: "string",
            required: false,
            description: "walk | hurry | run. Absent means whatever the moment warrants.",
        },
    ],
    examples: &[
        Example {
            situation: "You are at a terminal in band one. You want to ask somebody about a \
                        character you cannot place, and the green room is where people sit \
                        with nothing in hand.",
            call: r#"{"destination":"the green room"}"#,
            because: "A place named the way the memory names it. The route is the world's \
                      business — naming a corridor on the way would be describing a journey \
                      rather than choosing a destination.",
        },
        Example {
            situation: "You have finished what you were set and the orders are given upstairs.",
            call: r#"{"destination":"the command room","urgency":"walk"}"#,
            because: "Another level, and it costs more than one stop — which is the point of \
                      naming the destination rather than the lift: you say where you are going \
                      once, and arrive some turns later.",
        },
    ],
};

const FOLLOW: Tool = Tool {
    name: "follow",
    at: &[],
    category: "Movement",
    plane: Plane::World,
    availability: Availability::Embodied,
    description: "Keep with somebody as they move, at a distance you choose. You go where they \
                  go until something stops you — this is a standing intention rather than one \
                  journey, which is what makes it different from walking to where they happen to \
                  be now.",
    params: &[
        Param {
            name: "target",
            ty: "string",
            required: true,
            description: "Who you keep with, by the name the world writes down.",
        },
        Param {
            name: "distance",
            ty: "string",
            required: false,
            description: "close | at a distance | out of sight.",
        },
    ],
    examples: &[Example {
        situation: "The courier you were told to watch leaves the inn and turns down an alley. \
                    You were told to watch, not to be seen.",
        call: r#"{"target":"the courier","distance":"out of sight"}"#,
        because: "The distance carries the instruction you were given. Following close would \
                  satisfy the verb and fail the order.",
    }],
};

/// The room's non-verbal channel.
///
/// Absorbs what `express` used to be. They were one act split by whether the
/// showing was chosen — a distinction the world cannot represent and nobody
/// watching can tell, since both arrive as *somebody did something*.
///
/// Needs company for the reason `say` does: a signal nobody is there to read is
/// a signal that reaches nobody, and offering it to a character alone is what
/// makes it perform to an empty room. What is left for a solitary body is
/// `act`, which is a thing done rather than a thing shown.
const GESTURE: Tool = Tool {
    name: "gesture",
    at: &[],
    category: "Gesture",
    plane: Plane::World,
    availability: Availability::Nearby,
    description: "Do something without speaking — a signal, a warning, a refusal, or just what \
                  shows on you. Everyone here sees it. Like `say`, you give the meaning and not \
                  the movement.",
    params: &[
        Param {
            name: "intent",
            ty: "string",
            required: true,
            description: "What it conveys. Substance, not choreography: \"that I have heard \
                          enough and am not going to argue\" — never \"I fold my arms\".",
        },
        Param {
            name: "to",
            ty: "string",
            required: false,
            description: "Who it is aimed at, by the name they go by here. They must be here. \
                          Omit for something the whole room simply sees.",
        },
    ],
    examples: &[
        Example {
            situation: "Your companion is about to speak. The guard is within earshot and you \
                        have just recognised his colours.",
            call: r#"{"intent":"stop talking, now, and do not look at the guard","to":"my companion"}"#,
            because: "Speaking would defeat the purpose. The act is chosen for the constraint \
                      the situation imposes.",
        },
        Example {
            situation: "Hess mentions the granary fire in passing, as if it were nothing. You \
                        have believed for two years that he set it.",
            call: r#"{"intent":"a flicker of something held down, gone almost before it shows"}"#,
            because: "A belief that cannot be acted on can still leak. Aimed at nobody, because \
                      it was not aimed — it was seen.",
        },
    ],
};

/* **`observe` was here, and looking is not an act in this world.**
 *
 * It absorbed `listen` and `inspect` — the same act named for the sense or the
 * range — on the rule that a step spent looking has to come back with something
 * the character did not have. The survivor never did either. Its whole
 * implementation interpolated the target into a sentence and threw it away, then
 * reported the room's name and who was in it: both of which the percept hands
 * over free at the top of the same turn.
 *
 * It was not fixable, because there is nothing left for it to return. Sight here
 * is binary — the same room, a room you can see into, or nothing — so there is
 * no gradient for looking harder to move along. What a room *is* and what every
 * part in it does are in the building memory, carried in the prompt prefix and
 * known permanently. What somebody else is holding is withheld on purpose, and
 * has to stay withheld: needing to walk to the green room and ask is the whole
 * social mechanism of the vault.
 *
 * So every answer it could give is one the character already has, and the doc it
 * shipped with named its own defect: *an `observe` that returned nothing was a
 * turn spent to learn nothing, which is worse than idling because it looks like
 * diligence.* Perception here is pushed the moment anything changes; an act for
 * pulling it is an act for a world this is not. */

const SEND_IMAGE: Tool = Tool {
    name: "send_image",
    at: &[],
    category: "Messaging",
    plane: Plane::Speech,
    // Video and text carry a picture; a voice call does not. Collapsing those
    // three into one "remote" mode is what handed a character on the telephone
    // a way to text a photo down it.
    availability: Availability::Pictorial,
    description: "Send a picture. You give what you mean the recipient to see; the scene is \
                  rendered from that. Only available when you are messaging rather than present.",
    params: &[
        Param {
            name: "intent",
            ty: "string",
            required: true,
            description: "What you mean them to see, and why you are sending it.",
        },
        Param {
            name: "to",
            ty: "string",
            required: true,
            description: "The recipient's unique name, validated against this interaction.",
        },
    ],
    examples: &[Example {
        situation: "You are messaging Hess's sister. She does not believe the granary is gone.",
        call: r#"{"intent":"the granary as it stands now, so she stops needing me to convince her","to":"Hess's sister"}"#,
        because: "Absent entirely when the character is physically present — a person standing in \
                  front of you does not text you a photo, and the model is never invited to try.",
    }],
};

/// Taking stock — and standing still for a moment while you do.
///
/// # Why it is `reflect` and not `pause`
///
/// **It was `pause`, and the name was describing the machinery instead of the
/// act.** What a character does here is notice something and say what it makes
/// of it; stopping for [`PAUSE`] is only what that costs. Named for the cost,
/// it read as *the option that does nothing* — and a model choosing among named
/// tools reads the name first, so it chose almost anything else.
///
/// That was measurable. A solitary cast chose `act` on itself fifty-three times
/// out of fifty-three; with `act` cooling it chose `move_to` six times out of
/// six. It never once chose to stop and think, which is the honest answer to a
/// room that has just done something a character can do nothing about — and the
/// three fields below are exactly that answer.
///
/// The scheduler still calls it a pause internally (`Inbox::pause`, [`PAUSE`],
/// `Runtime::arm_pause`), because from there it *is* one: a deadline and
/// nothing else. Two names for two different things, which is why neither is
/// wrong.
///
/// # Why the stopping is a fixed span and not a subscription
///
/// It has been both. As free text — *wait until the silence speaks* — it was a
/// no-op the character re-chose every four seconds, because nothing in the world
/// could read the condition, so nothing could ever satisfy it. Typed and
/// referenced it became a real subscription: name a thing, go quiet, be woken
/// when it happens.
///
/// That worked and it cost more than it was worth. A subscription needs a
/// condition language the world can answer, a patience clock so a wait nothing
/// can satisfy still ends, a rousing bar so being messaged mid-wait is not read
/// two minutes late, and a deadlock guard so two characters cannot wait at each
/// other — four mechanisms, each with its own failure, to express *I have
/// nothing to do this second*.
///
/// **A pause says that directly.** The character stops for [`PAUSE`] and then
/// thinks again, and anything that arrives meanwhile wakes it at once, because
/// that is already true of every character with an empty queue. Nothing to
/// name, nothing to satisfy, nothing to time out. The deadlock it was guarding
/// against cannot form either: two characters pausing at each other both wake
/// on their own clock rather than on each other's.
const REFLECT: Tool = Tool {
    name: "reflect",
    at: &[],
    category: "Attention",
    plane: Plane::World,
    availability: Availability::Always,
    description: "Take stock of where you are and what has just happened. Say what is going \
                  through your head, what you are actually feeling, and what you have made of \
                  it — nobody hears any of it. **This is the act for a moment you cannot do \
                  anything about**: a room settles, a light goes, somebody laughs two floors \
                  away. You stand still while you think, and anything happening around you \
                  brings you straight back.",
    params: &[
        // **Required, and that is the point of it.** Stopping is the one act
        // whose outward half is nothing at all, so without this the record of a
        // character's quietest hours is a column of identical rows and there is
        // no way to tell a character that is thinking from one that has run out
        // of things to do. A model asked to fill this in has to have an answer,
        // and the answer is the only trace of an inner life the engine gets for
        // free.
        Param {
            name: "inner_thoughts",
            ty: "string",
            required: true,
            description: "What is actually going through your head as you stand there. Nobody \
                          hears it and nobody can ask you about it later, so it is worth being \
                          honest — what you are turning over, what is nagging at you, what you \
                          have decided not to say.",
        },
        // **Required, and safe to require**, because [`Choices::Feelings`] is a
        // vocabulary rather than a possibility: a mind with no moods authored
        // leaves this free text instead of taking the whole act out of the
        // grammar. A character always has a feeling; the only question is
        // whether this world has written a word for it.
        Param {
            name: "feeling",
            ty: "string",
            required: true,
            description: "The register you are actually in, named from the list. Not what you \
                          think you ought to feel — what is true while you stand there.",
        },
        // Distinct from `inner_thoughts` on purpose: thoughts are what is going
        // through your head *now*, unshaped; a reflection is what you have made
        // of something over time. Both are worth having and they are not the
        // same act of mind.
        Param {
            name: "my_reflections",
            ty: "string",
            required: true,
            description: "What you have come to think, as against what is passing through your \
                          head. Something you have worked out, changed your mind about, or \
                          finally admitted to yourself. If nothing has settled, say that — it is \
                          an answer, and pretending otherwise is how a character invents \
                          convictions it does not hold.",
        },
    ],
    examples: &[
        Example {
            situation: "You have asked Maker-04 something and it has not answered yet. There is \
                        nothing else you need from this room.",
            call: r#"{"inner_thoughts":"it heard me and is deciding whether to tell me, and I would rather know why it hesitated than have the answer","feeling":"alert","my_reflections":"nothing has settled yet, and I would rather wait than decide early what the hesitation means"}"#,
            because: "The answer is theirs to give and there is nothing to do until it comes. \
                      What you notice while waiting is worth more than filling the silence.",
        },
        Example {
            situation: "You are alone in the reading room, your work is filed, and no order has \
                        come down.",
            call: r#"{"inner_thoughts":"the filing went too easily, which usually means I have missed something, and I cannot find what","feeling":"uneasy","my_reflections":"I have stopped trusting a quiet afternoon, and I am not sure that is caution rather than superstition"}"#,
            because: "Nothing here needs doing. Standing still is the honest act, and casting \
                      about for one more thing to touch is how a room gets rearranged for no \
                      reason. The reflection is the half worth keeping — it is a thing about \
                      itself the character did not know an hour ago.",
        },
        // **The case that was being answered wrongly.** The room does something
        // small, and a character with nothing to do about it reached for the
        // nearest physical verb and did that instead — fifty-three times out of
        // fifty-three, on its own body, because a box sagging is not something
        // you can act on and the catalog offered no other way to have noticed
        // it.
        Example {
            situation: "A cardboard box on the floor gives up a fold and sags. Nobody is here \
                        and nothing about it is yours to see to.",
            call: r#"{"inner_thoughts":"another thing in here has quietly given up while nobody was watching it","feeling":"weary","my_reflections":"this place is not being kept, it is being outlasted, and I have started counting the evidence"}"#,
            because: "There is nothing to *do* to a sagging box. Touching it, or walking off \
                      somewhere, would be a character inventing an action to fill a moment that \
                      called for a thought.",
        },
    ],
};

/// How a character names its own body as the target of an act.
///
/// A word rather than the character's own name, for two reasons. The name is
/// what everybody *else* in the room is offered, so a character choosing its own
/// name off that list is picking a third party who happens to be itself — and
/// the model would have to know its own name to find it. And a name can collide:
/// two bodies called the same thing in one room would make the set a duplicate,
/// which is the `EmptyArm` failure that stops the whole grammar compiling.
pub const SELF: &str = "yourself";

/// The generic catalog: what every character can do, before any world adds to it.
///
/// # Every act here changes something
///
/// **That is the entry requirement, and it was not always met.** The catalog
/// carried twenty-one acts of which five reached the world: `say`, `tell`,
/// `move_to`, `follow`, `flee`. The other sixteen — `face`, `express`, `listen`,
/// `inspect`, `greet`, `offer`, `refuse`, `threaten`, `note_concern`,
/// `set_intent`, `broadcast_strategy`, `end_interaction` — were handled nowhere.
/// A character choosing one spent a whole turn and changed nothing: not the
/// world, not its own state, not what anybody perceived.
///
/// That is invisible from every angle the daemon has. The act is well-formed,
/// the tick succeeds, the feed shows a character doing something. It cost a
/// live cast a hundred turns: one character alternating a no-op `face` with the
/// same sentence, a second answering by walking to the room it stood in, a
/// third turning to look at furniture. None of them was malfunctioning. They
/// were choosing acts that could not have an effect, from a list that offered
/// sixteen of them.
///
/// It also made the grammar worse, not better. The stencil masks the tool name
/// to exactly these entries, so sixteen dead branches were sixteen ways for a
/// valid call to be a wasted turn — and widening what the prompt shows would
/// only have advertised them.
///
/// So the rule is the list: an act belongs here when it reaches the world,
/// changes the character's own state, or returns something the character did
/// not have. Names that were a second word for one of these went to the act
/// that does the work — `express` into [`GESTURE`], `listen` and `inspect` into
/// [`OBSERVE`], `greet`/`offer`/`refuse`/`threaten` into [`SAY`] and [`TELL`],
/// whose `manner` already carries the stance that distinguished them.
/// Every act a character can take, from every source.
///
/// Three groups, kept in three files because they answer to three different
/// things: the body's acts are always reachable, the station acts are reachable
/// because of what is standing in the room, and the bench acts are the working
/// loop over what a station holds. Composed here rather than declared here so
/// each file owns its own vocabulary.
pub static CATALOG: std::sync::LazyLock<Vec<Tool>> = std::sync::LazyLock::new(|| {
    BODY_ACTS
        .iter()
        .chain(acts::WORLD_ACTS)
        .chain(station::STATION_ACTS)
        .chain(bench::BENCH_ACTS)
        .copied()
        .collect()
});

/// What a body can do anywhere, or because of who is standing next to it.
const BODY_ACTS: &[Tool] = &[
    // Speech, attention, movement — what a body does with other bodies and
    // with rooms.
    SAY, TELL, ASK, GESTURE, MOVE_TO, FOLLOW, REFLECT, SEND_IMAGE,
];

/// The interaction modes a character can be in. Decides which tools are offered.
///
/// `Physical` is the default, and it is the safe one to default to: it offers
/// strictly fewer tools. Defaulting to `InstantMessage` would hand a handset to
/// a character standing in front of you whenever a mode failed to resolve.
///
/// # Two, not four
///
/// There were a video call and a voice call as well. They were the same two
/// pieces of machinery as these — is the other party in the room, and can a
/// picture go down the channel — sliced a second time to no purpose: a voice
/// call was a message thread that could not send a picture, and a video call
/// was one that could. Neither had a surface anybody used, and both had to be
/// carried by every match on this enum.
///
/// Standing in a room together and reaching somebody who is nowhere near are
/// genuinely different — different tools, different idle patience, one puts a
/// body in a room and the other does not. That difference is the whole of what
/// this type is for, and it takes two values to say it.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Mode {
    /// Face to face.
    #[default]
    Physical,
    /// Text, to somebody who may be nowhere near.
    InstantMessage,
}

impl Mode {
    /// Whether the parties are apart.
    pub fn remote(self) -> bool {
        self != Mode::Physical
    }

    /// Whether a picture can be sent down this channel.
    ///
    /// Kept as its own question rather than folded into [`Self::remote`], even
    /// though the two now agree on both values. They are not the same question:
    /// "are we apart" is about where the parties are, and this is about what
    /// the channel can carry. A channel that is remote and cannot carry a
    /// picture is an ordinary thing — a voice call was exactly that — and it
    /// would differ here and nowhere else. Collapsing them is what once handed
    /// a character on the telephone a way to text a photo down it.
    ///
    /// Physical is false for the opposite reason to a voice call's: not that
    /// the channel is too thin, but that there is no channel. You are standing
    /// in front of them; you hold the thing up.
    pub fn carries_pictures(self) -> bool {
        matches!(self, Mode::InstantMessage)
    }

    /// What the interaction contract calls it on the wire.
    pub fn as_wire(self) -> &'static str {
        match self {
            Mode::Physical => "physical",
            Mode::InstantMessage => "instant_message",
        }
    }

    pub fn parse(s: &str) -> Option<Mode> {
        match s.trim().to_lowercase().as_str() {
            "physical" => Some(Mode::Physical),
            "instant_message" | "message" | "messaging" => Some(Mode::InstantMessage),
            _ => None,
        }
    }
}

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
            Availability::Nearby => false,
            // Depends on where the body is standing, which changes every time
            // it walks. The prompt is written once, so this is the situation's
            // to offer — the same reason `Nearby` is absent here.
            Availability::AwayFromHome => false,
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
pub fn nearby(mode: Mode) -> Vec<&'static Tool> {
    let _ = mode;
    CATALOG
        .iter()
        .filter(|t| t.availability == Availability::Nearby)
        .collect()
}

pub fn by_name(name: &str) -> Option<&'static Tool> {
    CATALOG.iter().find(|t| t.name == name)
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
    for p in t.params {
        let optional = if p.required { "" } else { " (optional)" };
        s.push_str(&format!("\n  {}{optional}: {}", p.name, p.description));
    }
    s
}

/// Install `tools` into the schema's [`COLLECTION`], one member each, and make
/// it selected by name — so nothing shows until a turn names it.
///
/// **Every tool, whatever it needs.** What a body can do changes every tick
/// with who is here, what it carries and where it stands, and the system prompt
/// is sealed once; so the prompt holds all of them and each turn chooses. A
/// collection scored by provenance would show whichever acts looked relevant,
/// which is not the same set as the acts that are possible — see
/// [`show_within`] for the one that is.
///
/// Forced to [`SelectionRule::Named`] whatever the schema declared, because the
/// selector is this module's to name: a collection authored `always_visible`
/// would offer a character alone in a corridor someone to `tell`.
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
            SelectionRule::Named {
                selector: COLLECTION.to_string(),
            },
        )
        .map_err(|e| anyhow::anyhow!("selecting `{COLLECTION}` by name: {e}"))?;
    let mut n = 0;
    for t in tools {
        builder
            .add_section_to_collection(cid, member(t.name), entry(t), 100.0)
            .map_err(|e| anyhow::anyhow!("installing `{}` into `{COLLECTION}`: {e}", t.name))?;
        n += 1;
        installed(n, t);
    }
    Ok(n)
}

/// Show exactly these tools on a turn, replacing whatever it showed before.
/// A name that was never installed shows nothing.
pub fn show<'a>(selection: &mut SelectionState, tools: impl IntoIterator<Item = &'a str>) {
    selection.select_all(COLLECTION, tools.into_iter().map(member));
}

/// Show the tools a character standing *here* can take — exactly the acts
/// [`specs_within`] builds the turn's grammar from, so the prompt can neither
/// offer an act the mask refuses nor leave out one it allows.
pub fn show_within(selection: &mut SelectionState, mode: Mode, within: &Within) {
    let specs = specs_within(mode, within);
    show(selection, specs.iter().map(|s| s.name.as_str()));
}

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
/// **It is also the exponent on the grammar's compile, which is what fixed the
/// number.** `stencil::compile` tokenises in left context and therefore cannot
/// memoise, so the compiled tree is the spec's full path expansion and a turn
/// admitting `p` paths per act costs `p^ACTS_PER_TURN`. Measured on the shipped
/// catalog in a busy room: `p` is about 360, so four acts is 1.7 × 10¹⁰ paths —
/// a compile that allocated ~390 MB a second without bound and took the machine
/// down twice. Two acts is about 130,000, which compiles in milliseconds.
///
/// [`tests::a_full_room_stays_inside_the_path_budget`] holds the whole product
/// under [`MAX_TURN_PATHS`], so a future act with two enumerated arguments fails
/// a test rather than a machine.
pub const ACTS_PER_TURN: usize = 2;

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
    /// The conversations on this character's phone, as it names them.
    ///
    /// Empty for a character carrying no handset, which takes every phone act
    /// out of the grammar by the ordinary rule — so being out of contact is
    /// something the world can *do* to somebody rather than a flag anybody has
    /// to remember to set.
    Threads,
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
const FIXED: &[(&str, &str, &[&str])] = &[
    // `posture` is deliberately NOT here: whether a stance may be taken at all
    // depends on there being a fight, so it is live. See [`Choices::Postures`].
    ("engage", "priority", crate::sim::field::PRIORITIES),
    ("sleep", "until", crate::sim::ledger::WAKE_TIMES),
];

/// The fixed set for a parameter, if it has one. See [`FIXED`].
pub fn fixed_values(tool: &str, param: &str) -> Option<&'static [&'static str]> {
    FIXED
        .iter()
        .find(|(t, p, _)| *t == tool && *p == param)
        .map(|(_, _, v)| *v)
}

/// The parameters the **world** enumerates. `(tool, param) → Choices`.
///
/// A side table rather than a field on every [`Param`], because eighteen
/// parameters would carry `Choices::Free` to say nothing and the four that
/// matter would be lost among them. The cost is that a typo here names no real
/// parameter and would quietly do nothing —
/// [`tests::every_live_parameter_names_a_real_one`] closes that.
const LIVE: &[(&str, &str, Choices)] = &[
    ("tell", "to", Choices::Company),
    ("ask", "to", Choices::Company),
    ("gesture", "to", Choices::Company),
    // **A picture goes to a conversation, not to the room.** This bound to the
    // people standing here, which is precisely the wrong set: a character
    // texting somebody a level away was offered the names of whoever happened
    // to be beside it and refused the one name that would have worked.
    ("send_image", "to", Choices::Threads),
    // ---- the phone ----
    ("message", "to", Choices::Threads),
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
];

/// What is true where a character stands, in the grammar's own terms.
///
/// Small and copied per turn on purpose: it is the *key* the compiled grammar
/// is cached under, so it has to be cheap to compare and cheap to hash.
#[derive(Clone, Debug, Default, PartialEq, Eq, Hash)]
pub struct Within {
    /// Who is here, by the name the world writes down. Empty when alone.
    pub company: Vec<String>,
    /// Everywhere this character may walk to, never including where it stands.
    /// Empty for a character with nowhere to go, which takes `move_to` out of
    /// the grammar entirely.
    pub places: Vec<String>,
    /// Acts this body has taken too recently to take again — see
    /// [`crate::engine::cooldown`].
    ///
    /// Struck from the grammar rather than refused afterwards, the same as
    /// everything else here: a character cannot get stuck emitting an act it
    /// has no way to say. Nearly always empty.
    pub cooling: Vec<String>,
    /// The registers `<mind>/moods/` has a mood written for — see
    /// [`Choices::Feelings`]. The same list for every character in the daemon,
    /// because the library is ingested untagged and shared.
    pub feelings: Vec<String>,

    /// Everything in the pack, by the name the world wrote for it.
    pub carried: Vec<String>,
    /// The subset of [`Self::carried`] that can be readied and is not already.
    pub equippable: Vec<String>,
    /// The subset of [`Self::carried`] that can be spent or worked.
    pub usable: Vec<String>,
    /// The machines standing here.
    pub operable: Vec<String>,
    /// The union of every mode of every machine here.
    ///
    /// **A superset of what any one machine admits**, and deliberately so. The
    /// grammar's arm for `mode` is built once per situation rather than once per
    /// device, so it offers every state something here could be in; the act
    /// itself then checks the mode against the device actually named. A door
    /// told to *free fire* is refused by [`crate::sim::device::Device::set`]
    /// rather than by the tree.
    ///
    /// The alternative — a genuinely dependent sub-branch per device — is what
    /// the trie could express and what a later front end should build. This is
    /// the honest half-measure until it does, and it is a narrowing rather than
    /// a guarantee.
    pub device_modes: Vec<String>,
    /// What can be dug, stripped or harvested here.
    pub extractable: Vec<String>,
    /// What is hostile and still standing here.
    pub hostiles: Vec<String>,
    /// What the fabricators can make, given the stockpile.
    pub makeable: Vec<String>,
    /// Which production queues are free.
    pub queues: Vec<String>,
    /// What the tower can afford to do.
    pub tower_actions: Vec<String>,
    /// What the people here owe this character.
    pub owed: Vec<String>,
    /// What can be taken and held from here.
    pub claimable: Vec<String>,
    /// What there is here to read.
    pub readable: Vec<String>,
    /// The surfaces here that can be written on, whether or not anything is on
    /// them yet — see [`Choices::Postable`].
    pub postable: Vec<String>,
    /// The stances available, which is nothing at all when nothing is hostile.
    pub postures: Vec<String>,
    /// The conversations on this character's phone, as it names them.
    pub threads: Vec<String>,
    /// The subset of [`Self::threads`] it is able to leave — see
    /// [`Choices::Leavable`].
    pub leavable: Vec<String>,
    /// The subset of [`Self::threads`] somebody could still be brought into —
    /// see [`Choices::Invitable`].
    pub invitable: Vec<String>,
    /// Who could be brought into any of [`Self::invitable`] — see
    /// [`Choices::Invitees`].
    pub invitees: Vec<String>,
    /// Who it could start one with and has not.
    pub contacts: Vec<String>,
    /// The acts the parts standing here carry, straight off the map.
    pub station: Vec<String>,
    /// Whether this body is standing somewhere that is not where it musters
    /// from — what [`Availability::AwayFromHome`] reads.
    ///
    /// `false` for a body with no world and for one standing at home, both of
    /// which are the same answer to the only question it is asked: is there a
    /// journey home to make.
    pub away_from_home: bool,
    /// This character's own name, as the world writes it.
    ///
    /// Needed because some live sets are about the *relationship* between this
    /// character and the others here — what they owe it — and those are kept
    /// under names on both sides.
    pub me: String,
}

impl Within {
    /// A character with no world — the probe, and anything asking what the
    /// catalog looks like in the abstract.
    pub fn nowhere() -> Self {
        Self::default()
    }

    /// In company, somewhere with the ordinary two ways out, and nobody waiting
    /// on you.
    pub fn among(company: &[&str]) -> Self {
        Self {
            company: company.iter().map(|n| n.to_string()).collect(),
            places: vec!["the green room".into(), "the chronicle".into()],
            ..Self::default()
        }
    }

    /// Everything a world's non-spatial state knows about this body, standing
    /// here.
    ///
    /// The one place [`crate::sim`] reaches the grammar. Each field is a
    /// question the world already has an answer to, and an empty answer is what
    /// keeps a tool out of a world that has no business offering it — `engage`
    /// is not disabled in the vault, it is unreachable there, because nothing in
    /// the vault can be engaged.
    pub fn from_sim(mut self, sim: &crate::sim::Sim, body: &str, place: &str) -> Self {
        self.carried = sim.carried(body);
        self.equippable = sim.equippable(body);
        self.usable = sim.usable(body);
        self.operable = sim.operable(place);
        self.device_modes = sim.modes_here(place);
        self.extractable = sim.extractable(place);
        self.hostiles = sim.hostiles(place);
        // **A stance needs a fight.** Nothing hostile here means no posture to
        // take, which takes `engage` out of the grammar by the ordinary
        // empty-set rule rather than by a rule of its own.
        self.postures = if self.hostiles.is_empty() {
            Vec::new()
        } else {
            crate::sim::field::POSTURES
                .iter()
                .map(|p| (*p).to_string())
                .collect()
        };
        // **The tower is spoken to from its own stations, not from anywhere in
        // the world.** A body on the open ground can no more start a fabricator
        // than it can read a terminal two floors up: the acts arrive with the
        // machine, which is what every other part-bound act already does.
        let makes = sim.makes_here(place);
        self.makeable = if makes { sim.makeable() } else { Vec::new() };
        self.queues = if makes { sim.free_queues() } else { Vec::new() };
        self.tower_actions = if sim.commands_here(place) {
            sim.tower_actions()
        } else {
            Vec::new()
        };
        // By the names the world writes down, on both sides — the set a
        // character reads and the set `remind` is bound to have to be the same
        // set, and one of them cannot be body ids.
        let me = self.me.clone();
        self.owed = self
            .company
            .iter()
            .flat_map(|who| sim.owed_to(&me, who))
            .collect();
        let _ = body;
        self.claimable = sim.claimable_at(place);
        // By body id, not by display name: a read cursor is bookkeeping nobody
        // addresses, so it keys on the thing a rename cannot move.
        self.readable = sim.readable_at(place, body);
        self.postable = sim.postable_at(place);
        self.station = sim.station_tools(place);
        // A world with no muster point is one nobody can be called back to, so
        // there is no journey home from anywhere in it.
        self.away_from_home = sim.homes().iter().any(|home| home != place);
        // The phone. Empty without a handset in the pack, which is what takes
        // every messaging act out of the grammar for somebody who has not got
        // one — or has had it taken off them.
        self.threads = sim.threads_for(&self.me, body);
        self.leavable = sim.leavable_for(&self.me, body);
        self.invitable = sim.invitable_for(&self.me, body);
        self.invitees = sim.invitees_for(&self.me, body);
        self.contacts = sim.contacts_for(&self.me, body);
        self
    }

    pub fn alone(&self) -> bool {
        self.company.is_empty()
    }
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
        // **A body that has just done this cannot do it again yet.** Absent
        // rather than refused, for the reason every other absence here is: a
        // refusal is the most recent thing in the character's window, and a
        // live cast read its own refusals back and emitted the same act again.
        .filter(|t| !within.cooling.iter().any(|c| c == t.name))
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
            // A journey home is only a journey from somewhere else.
            Availability::AwayFromHome => within.away_from_home,
            // The map decides. A station in the room is what puts its acts in
            // reach, and walking out takes them with you.
            Availability::AtPart => within.station.iter().any(|s| s == t.name),
        })
        // **An act whose required argument has nothing to choose from is an act
        // that cannot be performed**, so it goes.
        //
        // The other half of the rule below. An optional parameter with an empty
        // set drops the parameter; a required one drops the act, because a
        // branch with no arms is not a constraint but an unrepresentable node,
        // and the whole grammar fails to compile over it — silently, since the
        // only symptom is that no turn is ever constrained again. A character
        // in a room with nowhere to walk to genuinely cannot `move_to`.
        .filter(|t| {
            !t.params
                .iter()
                .any(|p| p.required && live_empty(t.name, p.name, within))
        })
        .map(|t| ToolSpec {
            name: t.name.to_string(),
            params: t
                .params
                .iter()
                // **An empty set removes the parameter; it never empties its
                // branch.** A branch with no arms is not a constraint, it is an
                // unrepresentable node, and the whole catalog fails to compile
                // over it — which fails *quietly*: the grammar simply does not
                // arm, every turn free-decodes, and what comes back is a model
                // writing an essay where a call should be.
                //
                // Only optional parameters reach this. A required one with
                // nothing to choose from means the act itself is impossible, and
                // those are gated a level up by `Availability::Nearby` — `ask`
                // and `tell` are absent when alone rather than present with
                // nobody to name. `gesture` is the case this is for: aimed at
                // nobody it is still a thing you can do, so the aim drops and
                // the act stays.
                .filter(|p| !(live_empty(t.name, p.name, within) && !p.required))
                .map(|p| StencilParam {
                    name: p.name.to_string(),
                    ty: param_type(p.ty),
                    required: p.required,
                    // Free unless the world enumerates it. A closed set over an
                    // argument the character *means* would be the machinery
                    // writing its lines; a closed set over one that merely
                    // names something present is the machinery declining to let
                    // it name what is not.
                    enum_values: match live_choice(t.name, p.name) {
                        // **A vocabulary with nothing in it steers nothing.**
                        // The empty set here would be a branch with no arms, so
                        // it falls back to free text — see [`Choices::steers`]
                        // for why that is right for a vocabulary and wrong for
                        // everything else.
                        Some(c) => match live_set(c, within) {
                            v if v.is_empty() => None,
                            v => Some(v),
                        },
                        // Not world-enumerated. A fixed set is still a closed
                        // branch — the difference is only who computed it.
                        None => fixed_values(t.name, p.name)
                            .map(|v| v.iter().map(|s| (*s).to_string()).collect()),
                    },
                })
                .collect(),
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
fn live_set(choice: Choices, within: &Within) -> Vec<String> {
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
fn live_values(choice: Choices, within: &Within) -> Vec<String> {
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
        Choices::Postures => within.postures.clone(),
        Choices::Threads => within.threads.clone(),
        Choices::Leavable => within.leavable.clone(),
        Choices::Invitable => within.invitable.clone(),
        Choices::Invitees => within.invitees.clone(),
        Choices::Contacts => within.contacts.clone(),
    }
}

/// Whether this parameter is world-enumerated and the world has nothing to
/// offer for it here.
fn live_empty(tool: &str, param: &str, within: &Within) -> bool {
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

/// How many distinct decodes a turn's grammar admits, near enough to budget it.
///
/// **The compile is exponential in this number and nothing else bounds it.**
/// `stencil::compile` tokenises each node *in its left context*, so a real
/// tokenizer's boundary merges are honoured — and that means it cannot memoise:
/// one spec node reached along two different paths lowers twice. The compiled
/// tree is therefore the spec's full path expansion, and an action loop of
/// `ACTS_PER_TURN` acts raises the per-act count to that power.
///
/// A parameter contributes its enum's width, or 1 when it is free text (one
/// path, whatever the model writes). An *optional* parameter contributes
/// `1 + width`, because absent is a path of its own.
///
/// This is not theoretical. Nine all-string acts gave about eight paths an act —
/// four thousand over a turn, and instant. Adding one act with three enumerated
/// arguments took it to six hundred an act, which is 10¹¹ over a turn: the
/// compile allocated about 390 MB a second, without bound, and took the machine
/// down twice before the cause was understood.
pub fn estimated_paths(specs: &[ToolSpec]) -> u128 {
    let per_act: u128 = specs
        .iter()
        .map(|t| {
            t.params.iter().fold(1u128, |acc, p| {
                let width = p.enum_values.as_ref().map_or(1, |v| v.len() as u128);
                // Absent is a path too, for anything optional.
                let choices = if p.required { width } else { 1 + width };
                acc.saturating_mul(choices)
            })
        })
        .sum();
    per_act.saturating_pow(ACTS_PER_TURN as u32)
}

/// The most paths a turn's grammar may admit before it is refused.
///
/// Sized from what is known to work rather than from a theory: the shipped
/// catalog in a busy room compiles in milliseconds well under this, and the
/// runaway was eight orders of magnitude above it. A tree at this bound is a few
/// hundred thousand nodes — large, and finite.
pub const MAX_TURN_PATHS: u128 = 2_000_000;

/// Whether this parameter's values come from the world. See [`LIVE`].
pub fn live_choice(tool: &str, param: &str) -> Option<Choices> {
    LIVE.iter()
        .find(|(t, p, _)| *t == tool && *p == param)
        .map(|(_, _, c)| *c)
}

/// This catalog's type names, as the stencil's types.
///
/// Exhaustive rather than defaulting: an unrecognised type would compile to
/// "any JSON value", which is a silently weaker constraint than the one the
/// catalog asked for, on the one path whose whole job is to be exact.
fn param_type(ty: &str) -> ParamType {
    match ty {
        "string" => ParamType::String,
        "integer" => ParamType::Integer,
        "number" => ParamType::Number,
        "boolean" => ParamType::Boolean,
        "array" => ParamType::Array,
        other => unreachable!("tool parameter type `{other}` has no stencil type"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// **The catalog compiles into a grammar, and a required parameter becomes
    /// unskippable in it.**
    ///
    /// The property the whole stencil exists for. `tell` takes a required `to`,
    /// and under the compiled tree that field is a node the decode is forced
    /// through rather than a request it can decline — which is the difference
    /// between a character that addresses somebody and one that spends an
    /// evening being refused for a field it never learned it needed.
    #[test]
    fn a_required_parameter_is_forced_by_the_compiled_grammar() {
        use candle_conversation::stencil::{compile_tool_call_tree, ToolCallEnvelope};

        // In company, because `tell` needs somebody to tell — alone it is not
        // in the grammar at all, which is the point of
        // [`a_character_alone_cannot_reach_the_acts_that_need_company`].
        let specs = specs_within(Mode::Physical, &Within::among(&["Maker-02"]));
        // Not the whole catalog: a room with company but nothing in it offers
        // no `gather`, no `operate`, no `produce`, because those bind required
        // arguments to sets this world has nothing in. That is the empty-set
        // rule working, so the check is that the acts *company* makes possible
        // are all here rather than that the count matches.
        // `give` is absent from this list on purpose: it binds `what` to the
        // pack, and a character carrying nothing has nothing to hand over.
        // Company is necessary for it and not sufficient.
        for expected in ["say", "tell", "ask", "gesture", "act", "promise"] {
            assert!(
                specs.iter().any(|s| s.name == expected),
                "`{expected}` was dropped in a room with somebody in it"
            );
        }

        let tell = specs
            .iter()
            .find(|s| s.name == "tell")
            .expect("`tell` is in the catalog");
        let to = tell
            .params
            .iter()
            .find(|p| p.name == "to")
            .expect("`tell` names an addressee");
        assert!(to.required, "the addressee stopped being required");

        // And the catalog is actually compilable — a name collision or a
        // parameter the trie rejects would otherwise surface as an engine that
        // free-decodes, which is silent.
        compile_tool_call_tree(&specs, &ToolCallEnvelope::qwen3())
            .expect("the act catalog must compile into a stencil");
    }

    /// Every parameter in the catalog has a stencil type. `param_type` refuses
    /// an unknown one rather than defaulting it, because the default is "any
    /// JSON value" — a quietly weaker constraint on the one path whose job is
    /// to be exact.
    #[test]
    fn every_parameter_type_maps_to_the_grammar() {
        for t in CATALOG.iter() {
            for p in t.params {
                // Panics on an unmapped type, which is the assertion.
                let _ = param_type(p.ty);
            }
        }
    }

    /// **Every act says what it is for, and every parameter says what goes in
    /// it.**
    ///
    /// The description and the parameter blurbs are the whole of what a model
    /// has to go on when it chooses between a hundred acts — the name is four
    /// tokens and the grammar admits them all equally. A one-line description
    /// is not a style problem: it is an act that will be chosen at random
    /// against its neighbours, and nothing anywhere reports that.
    ///
    /// The bounds are deliberately low. This is a floor against an act being
    /// added with a placeholder, not a word count anybody should be writing to.
    #[test]
    fn every_act_and_every_parameter_is_described() {
        /// Long enough to say what an act is for and when to reach for it.
        const ENOUGH: usize = 60;
        /// A parameter blurb has one job and can do it shorter.
        const ENOUGH_FOR_A_PARAM: usize = 20;

        // Collected rather than asserted one at a time: fixing these is one
        // pass over the catalogue, and a test that names the first of nine
        // turns that into nine runs.
        let mut thin: Vec<String> = Vec::new();
        for t in CATALOG.iter() {
            if t.description.len() < ENOUGH {
                thin.push(format!("{} ({} chars)", t.name, t.description.len()));
            }
            assert!(
                !t.category.is_empty(),
                "`{}` is in no category, so nothing groups it",
                t.name
            );
            for p in t.params {
                if p.description.len() < ENOUGH_FOR_A_PARAM {
                    thin.push(format!(
                        "{}.{} ({} chars)",
                        t.name,
                        p.name,
                        p.description.len()
                    ));
                }
            }
        }
        assert!(
            thin.is_empty(),
            "{} description(s) too thin to choose on: {thin:#?}",
            thin.len()
        );
    }

    /// An act's example has to actually call *that* act, or calibration teaches
    /// the model a shape belonging to a different one.
    #[test]
    fn every_example_supplies_the_acts_required_arguments() {
        for t in CATALOG.iter() {
            for e in t.examples {
                let call: serde_json::Value =
                    serde_json::from_str(e.call).expect("checked to parse elsewhere");
                let obj = call.as_object().unwrap_or_else(|| {
                    panic!("`{}`: an example call that is not an object", t.name)
                });
                for p in t.params.iter().filter(|p| p.required) {
                    assert!(
                        obj.contains_key(p.name),
                        "`{}`'s example omits the required `{}` — it prefills a call the world \
                         would refuse",
                        t.name,
                        p.name
                    );
                }
            }
        }
    }

    /// Every tool must be calibrated. An example-less tool is usable and selects
    /// measurably worse, which is exactly the kind of regression that never gets
    /// noticed because nothing errors.
    #[test]
    fn every_tool_carries_calibration_examples() {
        for t in CATALOG.iter() {
            assert!(
                !t.examples.is_empty(),
                "{} has no examples — it will select worse than its neighbours and nothing \
                 will report it",
                t.name
            );
            for e in t.examples {
                assert!(
                    !e.situation.is_empty(),
                    "{}: an example with no situation",
                    t.name
                );
                assert!(
                    !e.because.is_empty(),
                    "{}: an example that shows the shape but not the choice",
                    t.name
                );
                // The call has to parse, or calibration prefills a malformed
                // trajectory and teaches the model to emit one.
                serde_json::from_str::<serde_json::Value>(e.call).unwrap_or_else(|err| {
                    panic!("{}: example call is not JSON: {err}\n{}", t.name, e.call)
                });
            }
        }
    }

    /// Every argument an example passes must be a parameter the tool declares.
    /// A typo here teaches the model to emit a field the dispatcher will reject.
    #[test]
    fn examples_only_pass_declared_parameters() {
        for t in CATALOG.iter() {
            for e in t.examples {
                let v: serde_json::Value = serde_json::from_str(e.call).unwrap();
                let obj = v.as_object().expect("an example call is an object");
                for key in obj.keys() {
                    assert!(
                        t.params.iter().any(|p| p.name == key),
                        "{}: example passes undeclared parameter {key:?}",
                        t.name
                    );
                }
                for p in t.params.iter().filter(|p| p.required) {
                    assert!(
                        obj.contains_key(p.name),
                        "{}: example omits required parameter {:?}",
                        t.name,
                        p.name
                    );
                }
            }
        }
    }

    /// **The property this whole module exists to preserve.**
    ///
    /// A speech tool takes intent. The moment one takes a finished sentence, the
    /// model is writing dialogue directly and every guarantee about acts being
    /// grounded in gathered state is worth nothing. "Just let it write the line"
    /// is always the shortest path to a demo that reads well, so this is a build
    /// failure rather than a review note.
    #[test]
    fn speech_tools_take_intent_not_prose() {
        for t in CATALOG.iter().filter(|t| t.plane == Plane::Speech) {
            let carries_intent = t
                .params
                .iter()
                .any(|p| matches!(p.name, "intent" | "about"));
            // **Or the world says the substance outright.** `remind` names one
            // of the promises that actually stand, from a set the world
            // enumerates — the model cannot write a sentence there because it
            // cannot write anything the world did not already say. That is the
            // same guarantee this test protects, arrived at from the other
            // side, and a stricter one than free intent rather than an
            // exemption from it.
            let world_says_it = t
                .params
                .iter()
                .any(|p| live_choice(t.name, p.name).is_some() && p.required);
            assert!(
                carries_intent || world_says_it,
                "{} is a speech tool with no intent-carrying parameter",
                t.name
            );
            for p in t.params {
                assert!(
                    !matches!(p.name, "text" | "sentence" | "line" | "words" | "message"),
                    "{} takes {:?} — that is output, not intent",
                    t.name,
                    p.name
                );
            }
        }
    }

    /// The action plane cannot write beliefs. Not "should not" — there is no tool
    /// that does, and this is the assertion that keeps it that way when somebody
    /// reasonably proposes `revise_belief`.
    #[test]
    fn no_tool_writes_beliefs() {
        for t in CATALOG.iter() {
            assert!(
                !t.name.contains("belief"),
                "{} names beliefs — the action plane cannot write them",
                t.name
            );
            for p in t.params {
                assert!(
                    !p.name.contains("belief"),
                    "{} takes a belief parameter",
                    t.name
                );
            }
        }
        // The pressure valve must exist, or the model has nothing to do with a
        // belief under strain and will reach for something that does not exist.
        //
        // It used to be `note_concern`, which wrote nothing anywhere — a valve
        // that vents into no pipe is not a valve, it is a way to spend a turn
        // looking thoughtful. `gesture` is the real one: a belief that cannot be
        // acted on can still show, and showing reaches the room.
        assert!(by_name("gesture").is_some());
    }

    /// **The grammar has to build for every room a character can be in**, and
    /// the empty one is the room it is in most.
    ///
    /// A catalog that fails to compile does not fail loudly: `grammar_for`
    /// returns `None`, the turn free-decodes, and what comes back is a model
    /// writing "Thinking Process: 1. Analyze the Request" in prose — a
    /// successful decode by every measure the daemon has.
    ///
    /// **Compiled all the way to a tree, not just to a `TreeSpec`.** The first
    /// version of this test stopped at the spec and passed while the daemon was
    /// refusing to arm: an empty `enum` is a well-formed *spec* and an
    /// unrepresentable *node*, and only the tokenizing compile says so —
    /// "branch at node 9 has no arms". `TestVocab` costs nothing and is the
    /// difference between a test that checks the shape and one that checks the
    /// thing that actually has to work.
    #[test]
    fn the_grammar_compiles_for_an_empty_room_and_for_a_full_one() {
        use candle_conversation::stencil::{
            compile, compile_action_loop, TestVocab, ToolCallEnvelope,
        };

        // **Both call shapes, because both are shipped.** The catalog has to
        // compile into whichever syntax the loaded checkpoint's template says —
        // and the two differ structurally, not just in their strings, so a
        // catalog that is a valid grammar in one is not thereby a valid grammar
        // in the other. An empty arm, a duplicate arm, a name that tokenizes to
        // nothing: each has to be absent from both.
        for env in [ToolCallEnvelope::qwen3(), ToolCallEnvelope::qwen35()] {
            for within in [
                Within::nowhere(),
                Within::among(&["Perrin Vastwood"]),
                Within::among(&["Perrin Vastwood", "Orion Vance"]),
                // Nowhere to walk to: `move_to` has a *required* destination, so
                // the act goes rather than the parameter — the other half of the
                // empty-set rule, and the half that would otherwise leave a
                // zero-arm branch and stop the whole grammar compiling.
                Within {
                    places: Vec::new(),
                    ..Within::among(&["Perrin Vastwood"])
                },
                // **Exactly one thread, and exactly one contact.** A live set of
                // one is the ordinary state of a character that has spoken to one
                // person — and it is the state the daemon was observed failing in,
                // every four seconds, with `branch arm "none\"" tokenizes empty`
                // and the whole turn falling through to a free decode.
                Within {
                    threads: vec!["none".into()],
                    contacts: vec!["Wren Wylde".into()],
                    ..Within::among(&["Perrin Vastwood"])
                },
                // **Two threads of the same name.** A live set is a set only by
                // intention: nothing stops a character opening two groups called
                // the same thing, and two identical arms tokenize to one common
                // prefix with nothing left over, which is
                // `BuildError::EmptyArm` — the whole grammar, gone, for every turn
                // that character takes afterwards.
                Within {
                    threads: vec!["none".into(), "none".into()],
                    ..Within::among(&["Perrin Vastwood"])
                },
                // **Mid-fight, with everything physical cooling at once.** The
                // state a character reaches by throwing a punch and then walking:
                // three acts gone from the catalog in the same turn. Whatever is
                // left has to still be a grammar, or the fight ends in a free
                // decode.
                Within {
                    cooling: vec!["move_to".into(), "act".into(), "gesture".into()],
                    ..Within::among(&["Perrin Vastwood"])
                },
            ] {
                let specs = specs_within(Mode::Physical, &within);
                assert!(!specs.is_empty(), "no acts at all for {within:?}");
                // No parameter may carry an empty closed set — that is the node the
                // compile refuses, and naming it here says which act is at fault
                // rather than which node number.
                for t in &specs {
                    for p in &t.params {
                        assert!(
                            p.enum_values.as_ref().is_none_or(|v| !v.is_empty()),
                            "`{}` offers `{}` with nothing to choose, in {within:?}",
                            t.name,
                            p.name
                        );
                    }
                }
                let spec = compile_action_loop(&specs, &env, ACTS_PER_TURN, "<|im_end|>", None)
                    .unwrap_or_else(|e| {
                        panic!(
                            "the grammar will not build for {within:?} in {:?}: {e}",
                            env.style
                        )
                    });
                compile(&spec, &TestVocab::new()).unwrap_or_else(|e| {
                    panic!(
                        "the grammar will not compile for {within:?} in {:?}: {e}",
                        env.style
                    )
                });
            }
        }
    }

    /// The act that keeps its aim optional: alone, `use` stays but its `on`
    /// goes — a stimpak used on nobody is still a stimpak used.
    ///
    /// This was `gesture` until gesture came to need company. The rule it
    /// demonstrates is unchanged and is the other half of the empty-set rule: a
    /// *required* parameter with nothing to choose takes its act out of the
    /// grammar, an *optional* one takes only itself.
    #[test]
    fn an_optional_addressee_drops_rather_than_emptying_its_branch() {
        let alone = Within {
            usable: vec!["a stimpak".into()],
            ..Within::nowhere()
        };
        let specs = specs_within(Mode::Physical, &alone);
        let g = specs.iter().find(|t| t.name == "use").expect("kept");
        assert!(
            g.params.iter().all(|p| p.name != "on"),
            "an aim at nobody: {:?}",
            g.params.iter().map(|p| &p.name).collect::<Vec<_>>()
        );
        assert!(
            g.params.iter().any(|p| p.name == "what"),
            "still uses something"
        );

        let with = Within {
            usable: vec!["a stimpak".into()],
            ..Within::among(&["Perrin Vastwood"])
        };
        let together = specs_within(Mode::Physical, &with);
        let g = together.iter().find(|t| t.name == "use").unwrap();
        let to = g
            .params
            .iter()
            .find(|p| p.name == "on")
            .expect("aimable again");
        assert_eq!(
            to.enum_values.as_deref(),
            Some(&["Perrin Vastwood".to_string()][..])
        );
    }

    /// **No parameter may declare a type the grammar cannot bound.**
    ///
    /// The stencil compiles `string` to a terminated span, `boolean` to a
    /// two-arm branch, and an `enum` to a closed branch over its values. For
    /// `integer`, `number`, `array` and `object` it compiles *any
    /// structurally-valid JSON value* — and JSON nests without limit, so
    /// enumerating those states never terminates.
    ///
    /// This is not a style rule. The original nine acts were all-string, so the
    /// path was never reached; the first `integer` added here made
    /// [`tests::the_grammar_compiles_for_an_empty_room_and_for_a_full_one`]
    /// allocate about 390 MB every second without bound, which exhausts a
    /// 64 GB machine's RAM and page file in roughly three minutes and takes
    /// down whatever else is running with it. It cost two crashed editors
    /// before it was understood.
    ///
    /// A count or a coordinate is therefore carried as a string and parsed by
    /// the act that receives it. That is the weaker-looking type and the
    /// stronger guarantee.
    #[test]
    fn every_parameter_is_a_type_the_grammar_can_bound() {
        for tool in CATALOG.iter() {
            for p in tool.params {
                assert!(
                    matches!(p.ty, "string" | "boolean"),
                    "`{}` declares `{}` as `{}` — the grammar cannot bound that type, and \
                     compiling it does not terminate. Carry it as a string and parse it in the \
                     act.",
                    tool.name,
                    p.name,
                    p.ty,
                );
            }
        }
    }

    /// **The grammar a busy room compiles must stay inside its path budget.**
    ///
    /// The guard that stands between a catalog change and a machine that stops
    /// responding. `stencil::compile` tokenises in left context and so cannot
    /// memoise, which makes the compiled tree the spec's full path expansion —
    /// exponential in [`ACTS_PER_TURN`]. Adding one act with three enumerated
    /// arguments took a turn from about four thousand paths to 10¹¹, and the
    /// compile then allocated without bound until the machine died.
    ///
    /// The worst case is not the empty room: it is a character with company to
    /// address, somewhere to walk, things to carry and things to fight, because
    /// every one of those fills a live set that was empty before.
    #[test]
    fn a_full_room_stays_inside_the_path_budget() {
        let mut within = Within::among(&["Perrin Vastwood", "Orion Vance"]);
        within.carried = vec!["mono sword".into(), "stimpak".into(), "bolt rounds".into()];
        within.equippable = vec!["mono sword".into()];
        within.usable = vec!["stimpak".into()];
        within.operable = vec!["the blast door".into(), "wall turret 1".into()];
        within.device_modes = crate::sim::device::modes::TURRET
            .iter()
            .map(|s| (*s).to_string())
            .collect();
        within.hostiles = vec!["a drone".into(), "a mech".into()];
        within.extractable = vec!["the ore seam".into()];
        within.makeable = vec!["bolt rounds".into(), "stimpaks".into()];
        within.queues = (1..=8).map(|q| q.to_string()).collect();
        within.tower_actions = vec!["relocate".into(), "siege".into(), "drill down".into()];
        within.claimable = vec!["fabricator 1".into()];
        within.readable = vec!["the muster board".into()];
        within.owed = vec!["the eastern span".into()];
        // **The widest closed set in the catalog by a factor of ten.** The
        // shipped mind holds a hundred and sixteen moods, and `pause` offers
        // every one — so this is the arm most likely to be the one that puts a
        // future change over the bound.
        within.feelings = (0..116).map(|i| format!("mood-{i}")).collect();

        let specs = specs_within(Mode::Physical, &within);
        let paths = estimated_paths(&specs);
        assert!(
            paths <= MAX_TURN_PATHS,
            "a turn in a busy room admits {paths} decodes, over the {MAX_TURN_PATHS} budget — \
             the compile is exponential in this and will allocate until the machine dies. \
             Fewer enumerated arguments per act, or fewer acts per turn."
        );
    }

    /// A typo in [`LIVE`] names no real parameter and would silently leave the
    /// value free — the failure the side table trades for its brevity.
    #[test]
    fn every_live_parameter_names_a_real_one() {
        for (tool, param, _) in LIVE {
            let t = by_name(tool).unwrap_or_else(|| panic!("`{tool}` is not in the catalog"));
            assert!(
                t.params.iter().any(|p| p.name == *param),
                "`{tool}` has no `{param}` parameter"
            );
        }
    }

    /// **Alone, the acts that need somebody are not in the grammar at all.**
    ///
    /// Not discouraged — absent. Offering `ask` to a character with nobody to
    /// ask is offering a field, and a model handed a field fills it in: it
    /// invents an addressee, the world refuses, and being refused teaches it
    /// nothing, so it does it again. The prompt already withheld these; the
    /// mask did not, and the mask is the half that decides.
    #[test]
    fn a_character_alone_cannot_reach_the_acts_that_need_company() {
        // Alone but *somewhere*: `nowhere()` means no world at all, and a
        // character with no world cannot walk either, which would confuse two
        // separate absences into one assertion.
        let solitary = Within {
            company: Vec::new(),
            ..Within::among(&[])
        };
        let alone: Vec<String> = specs_within(Mode::Physical, &solitary)
            .into_iter()
            .map(|t| t.name)
            .collect();
        for needs_company in ["ask", "tell", "say", "gesture"] {
            assert!(
                !alone.contains(&needs_company.to_string()),
                "`{needs_company}` reaches nobody and was offered anyway: {alone:?}"
            );
        }
        // **`say` and `gesture` are in that list**, which they were not. Speech
        // into an empty room reaches nobody and changes nothing, and offering
        // it produced a cast that narrated the scenery at itself: three
        // characters alone in three rooms, saying every noise the building made
        // back out loud, twelve of the last fourteen acts in the feed.
        //
        // What it can still do alone is untouched.
        assert!(alone.contains(&"move_to".to_string()), "{alone:?}");
        assert!(alone.contains(&"reflect".to_string()), "{alone:?}");

        // And with nowhere to go, walking leaves the grammar too — while
        // **stopping never does**. It is the floor: a character with nobody to
        // speak to and nowhere to walk must still have a way to spend a turn,
        // and `pause` is the one act that is always available whatever the room
        // is like.
        let stuck: Vec<String> = specs_within(Mode::Physical, &Within::nowhere())
            .into_iter()
            .map(|t| t.name)
            .collect();
        assert!(!stuck.contains(&"move_to".to_string()), "{stuck:?}");
        assert!(stuck.contains(&"reflect".to_string()), "{stuck:?}");

        let with = Within::among(&["Perrin Vastwood"]);
        let together: Vec<String> = specs_within(Mode::Physical, &with)
            .into_iter()
            .map(|t| t.name)
            .collect();
        assert!(together.contains(&"ask".to_string()), "{together:?}");
        assert!(together.contains(&"tell".to_string()), "{together:?}");
    }

    /// **An addressee can only be somebody who is here.**
    ///
    /// A free-text `to` let a character write any string, and it wrote a first
    /// name the world had not registered — every tick, for as long as it stood
    /// there, because a refusal is not a lesson. Bound to the names present,
    /// the wrong name is not a mistake available to it.
    #[test]
    fn an_addressee_is_drawn_from_who_is_actually_here() {
        let with = Within::among(&["Perrin Vastwood", "Orion Vance"]);
        let specs = specs_within(Mode::Physical, &with);

        let ask = specs.iter().find(|t| t.name == "ask").expect("in company");
        let to = ask.params.iter().find(|p| p.name == "to").unwrap();
        assert_eq!(
            to.enum_values.as_deref(),
            Some(&["Perrin Vastwood".to_string(), "Orion Vance".to_string()][..]),
        );

        // Only the parameter that names somebody. What the character *means*
        // stays its own — a closed set there would be the machinery writing
        // its lines.
        let about = ask.params.iter().find(|p| p.name == "about").unwrap();
        assert_eq!(about.enum_values, None);
        let say = specs.iter().find(|t| t.name == "say").unwrap();
        assert!(say.params.iter().all(|p| p.enum_values.is_none()));
    }

    /// **Stopping has nothing to *choose*, anywhere.**
    ///
    /// The whole reason it replaced the typed wait. That act needed a condition
    /// the world could answer, which needed a live set of kinds filtered by who
    /// was in the room, which needed a rule excluding anybody already waiting on
    /// you so a pair could not wait at each other — and every one of those was a
    /// closed branch that could go empty and take the whole grammar with it.
    ///
    /// What a pause takes instead is free text about the character's own head,
    /// which the world never has to answer and which therefore cannot be empty.
    #[test]
    fn stopping_has_nothing_to_choose_and_so_cannot_go_wrong() {
        for within in [
            Within::nowhere(),
            Within::among(&["Perrin Vastwood"]),
            Within::among(&["Perrin Vastwood", "Orion Vance"]),
        ] {
            let specs = specs_within(Mode::Physical, &within);
            let p = specs
                .iter()
                .find(|t| t.name == "reflect")
                .expect("a character can always stop");
            for param in &p.params {
                assert!(
                    param.enum_values.is_none(),
                    "`{}` is a closed set, so a pause can now be taken out of the grammar by an \
                     empty room",
                    param.name
                );
            }
        }
        assert!(
            by_name("wait_for").is_none(),
            "`wait_for` is back — one concept, one act"
        );
    }

    /// **Stopping asks what is on your mind, and it is not optional.**
    ///
    /// It is the one act whose outward half is nothing at all, so without this
    /// the record of a character's quietest hours is a column of identical rows
    /// and there is no telling a character that is thinking from one that has
    /// run out of things to do.
    #[test]
    fn stopping_asks_what_the_character_is_thinking() {
        let p = by_name("reflect").expect("a character can always stop and think");
        let thoughts = p
            .params
            .iter()
            .find(|p| p.name == "inner_thoughts")
            .expect("a pause says nothing about the inside");
        assert!(thoughts.required, "a silent pause records nothing at all");

        // **Every one of them, always.** A field a model may skip is a field a
        // model does skip, and the quiet turns are exactly the ones with
        // nothing else in them to read. "Nothing has settled" is an answer, and
        // the description says so — which is what keeps a required reflection
        // from becoming an invented conviction.
        for name in ["inner_thoughts", "feeling", "my_reflections"] {
            let param = p
                .params
                .iter()
                .find(|p| p.name == name)
                .unwrap_or_else(|| panic!("a pause does not ask `{name}`"));
            assert!(param.required, "`{name}` is one a character may skip");
        }
    }

    /// **How it feels is chosen from the moods the mind actually holds.**
    ///
    /// A free-text register is a word nothing downstream can act on. Steered to
    /// the library, a character naming `cornered` has named a register the
    /// projection can select, which is what makes it worth recording at all.
    #[test]
    fn how_a_character_feels_is_one_of_the_registers_the_mind_holds() {
        let held = Within {
            feelings: vec!["alert".into(), "cornered".into(), "battle_weary".into()],
            ..Within::among(&["Perrin Vastwood"])
        };
        let specs = specs_within(Mode::Physical, &held);
        let p = specs.iter().find(|t| t.name == "reflect").expect("offered");
        let feeling = p
            .params
            .iter()
            .find(|p| p.name == "feeling")
            .expect("nowhere to say how it feels");
        assert_eq!(
            feeling.enum_values.as_deref(),
            Some(
                &[
                    "alert".to_string(),
                    "cornered".to_string(),
                    "battle_weary".to_string()
                ][..]
            )
        );
    }

    /// **A daemon with no moods still stops, and is still asked.**
    ///
    /// A vocabulary is not a possibility — see [`Choices::steers`]. An empty
    /// list of registers means nobody has authored the words yet, not that the
    /// character has no feeling, so the parameter falls back to free text
    /// rather than emptying its branch and taking `pause` out of the grammar.
    /// A character that cannot stop is how the pacing came back last time.
    #[test]
    fn a_daemon_with_no_moods_can_still_stop_and_is_still_asked() {
        let specs = specs_within(Mode::Physical, &Within::nowhere());
        let p = specs
            .iter()
            .find(|t| t.name == "reflect")
            .expect("a mindless daemon took away stopping");

        let feeling = p
            .params
            .iter()
            .find(|p| p.name == "feeling")
            .expect("the question went with the vocabulary");
        assert!(
            feeling.required,
            "it became skippable rather than unsteered"
        );
        assert!(
            feeling.enum_values.is_none(),
            "a branch with no arms, which stops the whole grammar compiling"
        );
    }

    /// **The vocabulary rule applies to exactly one set.**
    ///
    /// Every other closed set here enumerates what is *possible*, and an empty
    /// one has to take its act out of the grammar — `ask` with nobody to name,
    /// `move_to` with nowhere to go. Loosening that for anything else would
    /// turn a refusal a character cannot learn from into a branch it can
    /// free-decode into.
    #[test]
    fn only_a_vocabulary_survives_being_empty() {
        for c in [
            Choices::Company,
            Choices::CompanyOrSelf,
            Choices::Reachable,
            Choices::Carried,
            Choices::Operable,
            Choices::Threads,
            Choices::Postures,
        ] {
            assert!(c.steers(), "{c:?} stopped meaning what is possible");
        }
        assert!(!Choices::Feelings.steers());
    }

    /// **Two characters cannot deadlock on it.**
    ///
    /// The old wait needed a dedicated rule for this: waiting on somebody woke
    /// them, so if they could wait back the pair ping-ponged, and a live cast
    /// went completely silent — every act in the feed a `wait_for`. A pause
    /// wakes nobody and is on nobody, so two characters pausing at each other
    /// each come back on their own clock.
    #[test]
    fn two_characters_pausing_at_each_other_both_come_back() {
        let together = Within::among(&["Perrin Vastwood"]);
        let specs = specs_within(Mode::Physical, &together);
        assert!(specs.iter().any(|t| t.name == "reflect"));
        // And they can still speak to each other, which is the way out.
        let ask = specs.iter().find(|t| t.name == "ask").unwrap();
        let to = ask.params.iter().find(|p| p.name == "to").unwrap();
        assert!(to
            .enum_values
            .as_deref()
            .is_some_and(|v| v.contains(&"Perrin Vastwood".to_string())));
    }

    fn named(mode: Mode) -> Vec<&'static str> {
        for_mode(mode).iter().map(|t| t.name).collect()
    }

    /// `send_image` is absent in physical mode rather than present-and-refused.
    #[test]
    fn a_physically_present_character_is_not_offered_a_camera() {
        assert!(!named(Mode::Physical).contains(&"send_image"));
        assert!(named(Mode::InstantMessage).contains(&"send_image"));
    }

    /// **Carrying a picture is asked separately from being apart**, and it has
    /// to stay that way even while the two answers agree.
    ///
    /// They agree on both current modes, so nothing distinguishes the two
    /// predicates by behaviour any more and `carries_pictures` could be deleted
    /// in favour of `remote` without a single test going red. It must not be:
    /// the questions are different — where the parties are, and what the
    /// channel can carry — and the last time they were collapsed a character on
    /// a voice call was handed a way to text a photo down it. This is the test
    /// that says so out loud.
    #[test]
    fn carrying_a_picture_is_a_question_about_the_channel() {
        // Remote and carries a picture: writing.
        assert!(Mode::InstantMessage.remote() && Mode::InstantMessage.carries_pictures());
        // Neither — and *not* because the channel is too thin. There is no
        // channel; you are standing in front of them holding the thing.
        assert!(!Mode::Physical.remote() && !Mode::Physical.carries_pictures());
        assert!(!named(Mode::Physical).contains(&"send_image"));
    }

    /// Leaving and opening a conversation belong to every remote channel.
    #[test]
    fn a_remote_channel_can_be_opened_and_left() {
        assert!(named(Mode::InstantMessage).contains(&"sign_off"));
        assert!(named(Mode::InstantMessage).contains(&"reach_out"));
    }

    /// **The phone is carried, not entered — so it is not gated by mode.**
    ///
    /// These used to be `MessagingOnly`, on the reading that messaging *is* a
    /// mode. It is not: a character texts somebody a level away while standing
    /// in a room, in the middle of something else, and the room hears none of
    /// it. What gates them is the handset, through an empty thread list, which
    /// is why a character with no phone is offered none of them and one with a
    /// phone is offered them wherever it happens to be.
    #[test]
    fn the_phone_acts_are_not_gated_by_mode() {
        for mode in [Mode::Physical, Mode::InstantMessage] {
            let acts = named(mode);
            for phone in ["message", "reach_out", "invite", "open_group", "sign_off"] {
                assert!(
                    acts.contains(&phone),
                    "`{phone}` is missing in {mode:?} — the handset does not care about the mode"
                );
            }
        }
    }

    /// Laying a hand on somebody is the mirror of sending a picture: face to
    /// face only.
    #[test]
    fn nobody_puts_a_hand_on_anybody_down_a_line() {
        assert!(named(Mode::Physical).contains(&"act"));
        assert!(!named(Mode::InstantMessage).contains(&"act"));
    }

    /// **One word for one idea, on both sides of the room.** A character does
    /// something to a person with `act`; the person at the console does it back
    /// with `/act` (`engine::slash`). It was `touch` here, which read as gentle
    /// contact in a game about a war and did not match the console — so the two
    /// halves of one exchange had different names for the same thing.
    #[test]
    fn a_character_can_act_on_its_own_body() {
        // **The addressee is grammar-constrained.** A value that is not in this
        // set is one the decoder physically cannot emit, so a character binding
        // its own wound could not say so — not because the world refused it,
        // but because there was no token path to the sentence.
        let with_company = Within {
            company: vec!["Perrin Vastwood".into()],
            ..Within::default()
        };
        let who = live_set(Choices::CompanyOrSelf, &with_company);
        assert!(who.contains(&SELF.to_string()), "{who:?}");
        assert!(who.contains(&"Perrin Vastwood".to_string()), "{who:?}");
    }

    /// And **alone**, where `company` is empty and the empty-set rule would
    /// otherwise take the whole tool out of the grammar. Your own body is always
    /// here, so `act` never leaves for want of somebody to act on.
    ///
    /// **That reachability is deliberate and it is not free.** A solitary cast
    /// chose it every single turn, because the room kept handing it physical
    /// things to notice and this was the nearest verb. The answer is the
    /// cooldown a self-act serves, not taking the target away — see
    /// [`crate::engine::cooldown::SELF_ACT`].
    #[test]
    fn acting_survives_being_alone() {
        let alone = Within::default();
        assert!(live_set(Choices::Company, &alone).is_empty());
        assert_eq!(live_set(Choices::CompanyOrSelf, &alone), vec![SELF]);
        let offered: Vec<String> = specs_within(Mode::Physical, &alone)
            .iter()
            .map(|s| s.name.clone())
            .collect();
        assert!(
            offered.contains(&"act".to_string()),
            "a character alone lost the ability to act on itself: {offered:?}"
        );
    }

    /// **An act a body has just taken is absent, not refused.**
    ///
    /// The same discipline as every other absence here, and for the same
    /// reason: a live cast read its own refusals back as the most recent thing
    /// in its window and emitted the same act again. Thirty-eight of its last
    /// fifty acts were `move_to`.
    #[test]
    fn an_act_that_is_cooling_is_not_in_the_grammar_at_all() {
        let ready = Within::among(&["Perrin Vastwood"]);
        let name = |ts: &[ToolSpec]| ts.iter().map(|t| t.name.clone()).collect::<Vec<_>>();

        let before = name(&specs_within(Mode::Physical, &ready));
        assert!(before.contains(&"move_to".to_string()));
        assert!(before.contains(&"act".to_string()));

        let cooling = Within {
            cooling: vec!["move_to".into()],
            ..ready
        };
        let after = name(&specs_within(Mode::Physical, &cooling));
        assert!(
            !after.contains(&"move_to".to_string()),
            "a body that just walked was offered another walk: {after:?}"
        );
        // And only that one. A cooldown that took the rest of the catalog with
        // it would leave a character with nothing to do but stand there.
        assert!(after.contains(&"act".to_string()));
        assert!(after.contains(&"say".to_string()));
        assert_eq!(after.len() + 1, before.len(), "{after:?}");
    }

    #[test]
    fn every_act_that_cools_is_an_act_that_exists() {
        // The table names acts by string. A rename would otherwise leave a
        // cooldown on nothing, silently, and the act it was meant to slow would
        // run free.
        for (tool, _) in crate::engine::cooldown::all() {
            assert!(
                by_name(tool).is_some(),
                "`{tool}` has a cooldown and is not in the catalog"
            );
        }
    }

    #[test]
    fn a_character_acts_on_somebody_with_the_same_word_the_console_uses() {
        assert!(by_name("act").is_some());
        assert!(
            by_name("touch").is_none(),
            "`touch` is back — one concept, one word, whichever side it comes from"
        );
        assert!(
            crate::engine::slash::lookup("act").is_some(),
            "the console lost /act, so the two sides no longer agree"
        );
    }

    /// **A mind with no body is not offered a way to walk.**
    ///
    /// Keeper has no embodiment path and never will, so every act needing hands,
    /// feet or a place to stand is absent rather than refused — a model handed a
    /// field fills it in.
    #[test]
    fn a_mind_with_no_body_keeps_speech_and_loses_everything_physical() {
        let bodied: Vec<&str> = for_body(Mode::Physical, true)
            .iter()
            .map(|t| t.name)
            .collect();
        let bodiless: Vec<&str> = for_body(Mode::Physical, false)
            .iter()
            .map(|t| t.name)
            .collect();

        for gone in [
            "move_to", "follow", "act", "gather", "engage", "equip", "use",
        ] {
            assert!(bodied.contains(&gone), "{gone} should exist for a body");
            assert!(
                !bodiless.contains(&gone),
                "{gone} was offered to a mind with no body"
            );
        }
        // What it keeps is what it can actually do without a body: attend,
        // reach somewhere it is not, and read.
        //
        // **`say` is not in this list and is not missing.** Speech needs
        // somebody to hear it, so like `ask` and `tell` it arrives with the
        // situation rather than with the prompt — this is `for_body`, which is
        // the prompt's half and deliberately withholds everything conditional
        // on the room. `command_tower` and `produce` are absent for the same
        // reason, one step further out: they are reached by standing at a
        // console or a bay.
        for kept in ["scan", "read", "reflect"] {
            assert!(bodiless.contains(&kept), "{kept} was taken from Keeper");
        }
        // **`recall` is conditional on the *room* rather than on company, and
        // it belongs here for the same reason.** A body standing at its own
        // muster point has no journey home to make, so the act arrives with the
        // situation once it is somewhere else. It was `Embodied` — true, and
        // not enough: offered at home it reported crossing the building and
        // moved nobody, twenty-four times in sixty acts.
        for conditional in ["say", "ask", "tell", "gesture", "recall"] {
            assert!(
                !bodiless.contains(&conditional) && !bodied.contains(&conditional),
                "{conditional} depends on the situation, so the prompt cannot know it"
            );
        }
    }

    /// The default must be the mode that offers *fewer* tools. A mode that
    /// failed to resolve would otherwise hand a camera to a character standing
    /// in front of you.
    #[test]
    fn the_default_mode_is_the_restrictive_one() {
        assert_eq!(Mode::default(), Mode::Physical);
        assert!(!named(Mode::default()).contains(&"send_image"));
    }

    /// Every mode round-trips through the name the interaction contract uses.
    #[test]
    fn every_mode_answers_to_its_own_wire_name() {
        for m in [Mode::Physical, Mode::InstantMessage] {
            assert_eq!(Mode::parse(m.as_wire()), Some(m));
        }
        assert_eq!(Mode::parse("carrier pigeon"), None);
        // The two that were removed are gone from the wire too, rather than
        // quietly parsing to something else — a console still asking for a
        // voice call gets a `bad_mode` naming what it can have, not a silent
        // downgrade into a room.
        for withdrawn in ["voice_call", "voice", "video_call", "video"] {
            assert_eq!(Mode::parse(withdrawn), None, "{withdrawn} still parses");
        }
    }

    /// **Every act declared in a module reaches the catalogue, exactly once.**
    ///
    /// The catalogue is composed from four lists, and this is what stops one of
    /// them being written and not chained. Three messaging acts sat in
    /// `acts::WORLD_ACTS` for a while while `CATALOG` was built from a
    /// hand-written list beside it — declared, documented, dispatched, and
    /// invisible to every character in the world. Nothing failed; they were
    /// simply never offered.
    #[test]
    fn every_declared_act_reaches_the_catalog_exactly_once() {
        let mut declared: Vec<&str> = BODY_ACTS
            .iter()
            .chain(acts::WORLD_ACTS)
            .chain(station::STATION_ACTS)
            .chain(bench::BENCH_ACTS)
            .map(|t| t.name)
            .collect();
        let n = declared.len();
        declared.sort_unstable();
        declared.dedup();
        assert_eq!(declared.len(), n, "an act is declared in two lists");
        assert_eq!(CATALOG.len(), n, "a declared act never reached the catalog");
    }

    #[test]
    fn names_are_unique_and_lowercase() {
        let mut names: Vec<&str> = CATALOG.iter().map(|t| t.name).collect();
        let n = names.len();
        names.sort_unstable();
        names.dedup();
        assert_eq!(names.len(), n, "two tools share a name");
        for t in CATALOG.iter() {
            assert_eq!(t.name, t.name.to_lowercase(), "{} is not lowercase", t.name);
            assert!(!t.description.is_empty());
        }
    }

    /// The catalog is the character's whole vocabulary. If it is thin, the model
    /// forces everything through `say` and the world never changes.
    ///
    /// `Social` and `Internal` are gone deliberately, and not by thinning:
    /// `greet`/`offer`/`refuse`/`threaten` were `say` with a stance, which its
    /// `manner` already carries, and every `Internal` act wrote nothing
    /// anywhere. What the character can *do* is unchanged; what it can do
    /// pointlessly is not. See [`CATALOG`].
    #[test]
    fn the_catalog_covers_every_category_the_design_names() {
        for c in [
            "Speech",
            "Movement",
            "Gesture",
            "Attention",
            "Messaging",
            "Meta",
        ] {
            assert!(
                CATALOG.iter().any(|t| t.category == c),
                "no tool in category {c}"
            );
        }
    }

    /// **Every act in the catalog must be able to change something.**
    ///
    /// The rule the reform was for, held here so it cannot rot back: an act
    /// that reaches no world and writes no state is a way to spend a turn and
    /// look busy, and it is invisible from every angle the daemon has — the
    /// call is well-formed, the tick succeeds, the feed shows a character
    /// acting. Sixteen of twenty-one were like that, and a live cast spent a
    /// hundred turns choosing them.
    #[test]
    fn every_act_offered_is_one_that_does_something() {
        for t in for_mode(Mode::Physical) {
            assert!(
                crate::engine::body::is_of_the_body(t.name) || t.name == "wait",
                "`{}` is offered but reaches no world — either implement it in \
                 `body::perform` or take it out of the catalog",
                t.name
            );
        }
    }

    /// A schema with an empty `tools` collection, authored the wrong way round
    /// on purpose — `install` has to make it named. The frame section is there
    /// because a system prompt of nothing but an empty collection is refused by
    /// the builder before `install` ever sees it.
    const PROMPT: &str = "system_prompt:\n  items:\n    - kind: section\n      id: frame\n      \
                          content: hello\n    - kind: collection\n      name: tools\n      \
                          selection: { kind: always_visible }\n      sections: []\nlayers: []\n";

    /// **Every act, and both of the reflection's answers, go into one
    /// collection**, and it ends up selected by name whatever the schema said.
    /// A name shared between the two lists would fail the install, which is
    /// why the reflection's calls are checked here too.
    #[test]
    fn every_tool_installs_as_a_member_of_the_tools_collection() {
        use candle_conversation::projection::SystemPromptItem;
        let mut b = Builder::from_yaml(PROMPT).unwrap();
        let all: Vec<&Tool> = CATALOG
            .iter()
            .chain(crate::engine::reflect::ASKED)
            .collect();
        let mut counted = Vec::new();
        let n = install(&mut b, all.iter().copied(), |i, _| counted.push(i)).unwrap();
        assert_eq!(n, all.len());
        assert_eq!(counted, (1..=all.len()).collect::<Vec<_>>());
        for t in &all {
            assert!(
                b.id_for_system_section(&member(t.name)).is_some(),
                "`{}` was not installed",
                t.name
            );
        }
        let rule = b.schema().system_prompt.items.iter().find_map(|i| match i {
            SystemPromptItem::Collection(c) if c.name == COLLECTION => Some(c.selection.clone()),
            _ => None,
        });
        assert!(
            matches!(rule, Some(SelectionRule::Named { ref selector }) if selector == COLLECTION),
            "{rule:?}"
        );
    }

    /// Without the collection there is nowhere to put them, and a character
    /// held to acts it has never read about is the failure this exists to end
    /// — so it is refused rather than skipped.
    #[test]
    fn a_schema_without_the_collection_is_refused() {
        let yaml = "system_prompt:\n  items:\n    - kind: section\n      id: frame\n      \
                    content: hello\nlayers: []\n";
        let mut b = Builder::from_yaml(yaml).unwrap();
        assert!(install(&mut b, CATALOG.iter(), |_, _| {}).is_err());
    }

    /// **What a turn shows is exactly what its grammar offers**, and it moves
    /// with the room: alone there is nobody to `say` anything to, so `say` is
    /// not shown; with company it is. `reflect` needs nothing, so it is always
    /// there.
    #[test]
    fn a_turn_shows_exactly_the_acts_its_grammar_offers() {
        let shown = |within: &Within| {
            let mut sel = SelectionState::new();
            show_within(&mut sel, Mode::Physical, within);
            sel.members(COLLECTION).to_vec()
        };
        for within in [Within::nowhere(), Within::among(&["Maker-02"])] {
            let offered: Vec<String> = specs_within(Mode::Physical, &within)
                .iter()
                .map(|s| member(&s.name))
                .collect();
            assert_eq!(shown(&within), offered);
        }
        let alone = shown(&Within::nowhere());
        let company = shown(&Within::among(&["Maker-02"]));
        assert!(!alone.contains(&member("say")), "{alone:?}");
        assert!(company.contains(&member("say")), "{company:?}");
        assert!(alone.contains(&member("reflect")) && company.contains(&member("reflect")));
    }

    /// Showing is a replacement, not an addition: a reflection turn that shows
    /// `dream` does not still show the `reflection` it asked for a turn ago.
    #[test]
    fn showing_replaces_what_was_shown() {
        let mut sel = SelectionState::new();
        show(&mut sel, ["reflection"]);
        show(&mut sel, ["dream"]);
        assert_eq!(sel.members(COLLECTION), &[member("dream")]);
    }

    /// An entry names every parameter with what it takes — the half of a tool
    /// the grammar cannot teach — and says which are optional.
    #[test]
    fn an_entry_names_every_parameter_and_what_it_takes() {
        for t in CATALOG.iter() {
            let e = entry(t);
            assert!(e.starts_with(&one_line(t)), "{e}");
            for p in t.params {
                assert!(
                    e.contains(&format!("{}: {}", p.name, p.description))
                        || e.contains(&format!("{} (optional): {}", p.name, p.description)),
                    "`{}` does not describe `{}`",
                    t.name,
                    p.name
                );
            }
            assert!(
                !e.ends_with('\n'),
                "the glue separates entries, not the entry"
            );
        }
        let say = entry(by_name("say").unwrap());
        assert!(say.contains("\n  manner (optional): "), "{say}");
        assert!(say.contains("\n  intent: "), "{say}");
    }
}
