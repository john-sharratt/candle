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
//! # Examples are not documentation
//!
//! Each tool carries `examples` — short trajectories showing the tool chosen
//! against a concrete situation. These are prefilled into a calibration layer at
//! startup (the `Calibrating` load step), which is what makes tool *selection*
//! work rather than merely tool *invocation*. A tool with no examples is
//! uncalibrated and selects measurably worse, so an example-less tool is a build
//! failure here rather than a quiet quality regression.

use candle_conversation::projection::{Builder, SelectionRule};
use candle_conversation::stencil::{Param as StencilParam, ParamType, ToolSpec};
use serde::Serialize;

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
    /// Text/voice interactions only. A character standing in front of you does
    /// not text you a photo, and the model should never be invited to try — so
    /// this is *absent* in physical mode rather than present-and-refused.
    MessagingOnly,
    /// Only while somebody else is in the room.
    ///
    /// **Offered with the situation, not with the prompt.** Who is standing
    /// next to you changes every tick and the system prompt is written once, so
    /// a tool that depends on company cannot live there — it arrives beside
    /// *where you are*, computed from the world, for the same reason the percept
    /// does.
    ///
    /// Absent rather than present-and-refused, which is the same discipline
    /// `MessagingOnly` follows: a character alone in a corridor should never be
    /// invited to address somebody, because there is nobody to address and the
    /// invitation is what makes it try.
    Nearby,
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
}

/// Speech to the room, and speech to a person.
///
/// **Two tools, not one with an optional target.** Saying something aloud where
/// people are and addressing one of them are different acts: the first is
/// always possible and the second needs somebody to address. As one tool with
/// an optional `to`, a character alone in a corridor is invited to name
/// somebody every turn, and it does — a model handed a field fills it in.
///
/// Splitting them makes the impossible one *absent* rather than refused, which
/// is the same discipline `send_image` follows in a physical encounter. It also
/// makes the difference legible in what everyone else perceives, because the
/// world carries who an utterance was aimed at and renders it three ways.
const SAY: Tool = Tool {
    name: "say",
    category: "Speech",
    plane: Plane::Speech,
    availability: Availability::Always,
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
    category: "Movement",
    plane: Plane::World,
    availability: Availability::Always,
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
    category: "Movement",
    plane: Plane::World,
    availability: Availability::Always,
    description: "Keep with someone as they move, at a distance you choose.",
    params: &[
        Param {
            name: "target",
            ty: "string",
            required: true,
            description: "Who you keep with.",
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
const GESTURE: Tool = Tool {
    name: "gesture",
    category: "Gesture",
    plane: Plane::World,
    availability: Availability::Always,
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

/// Looking, as a way of *getting* something.
///
/// Absorbs `listen` and `inspect`, which were the same act named for the sense
/// or the range. What matters is that a step spent looking comes back with
/// something the character did not have, which is the only thing that made any
/// of the three worth a turn.
const OBSERVE: Tool = Tool {
    name: "observe",
    category: "Attention",
    plane: Plane::Internal,
    availability: Availability::Always,
    description: "Look, listen, or examine something properly. Spends your turn on finding out \
                  rather than on doing, and what you find comes straight back to you. Use it \
                  when acting on a guess would be worse than spending a moment.",
    params: &[Param {
        name: "target",
        ty: "string",
        required: true,
        description: "What you attend to — a thing here, a person, or the room itself.",
    }],
    examples: &[
        Example {
            situation: "The map shows a shape at the tree line that the legend does not account \
                        for.",
            call: r#"{"target":"the unaccounted shape at the tree line"}"#,
            because: "Acting on an ambiguity is worse than spending a step resolving it. \
                      Choosing to look is a real decision, not a null one.",
        },
        Example {
            situation: "You have just come into a room you have not been in before.",
            call: r#"{"target":"the room, and what is in it"}"#,
            because: "What comes back is what is actually here, which is what the next act has \
                      to be built on.",
        },
    ],
};

const SEND_IMAGE: Tool = Tool {
    name: "send_image",
    category: "Messaging",
    plane: Plane::Speech,
    availability: Availability::MessagingOnly,
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

/// Waiting for a **named thing**, aimed at a **named person**, and visible to
/// them.
///
/// # Why `wait` had to go
///
/// It was not a wait. It was a no-op the character had to re-choose every four
/// seconds: `until` was free text — "the silence speaks", "they finish
/// speaking" — that nothing in the world could read, so nothing could ever
/// satisfy it, so it never ended. A character did not wait for Orion to speak;
/// it decided to wait, spent a decode, forgot, and decided again. Three of them
/// did this at each other for hours, and it was the last act in the catalog
/// that changed nothing.
///
/// Typed and referenced, it becomes a **subscription**: the character goes
/// genuinely quiet, and the world wakes it when the thing it named happens.
///
/// # And the person it is aimed at is told
///
/// Waiting on somebody is not invisible. In a room you look at them, and the
/// silence is aimed rather than empty — so this emits into the room like any
/// other act, and the person waited on perceives it. That is what breaks the
/// deadlock without a timeout: two characters waiting on each other used to sit
/// there until something else moved, and now the first wait wakes the other,
/// who has something to answer.
///
/// It also closes the loop that would replace the deadlock. Somebody already
/// waiting on you is **not in your own `who` list** — see
/// [`Within::waited_on_by`] — so you cannot wait back at them, and one of you
/// has to speak. Mutual waiting is unreachable rather than discouraged.
const WAIT_FOR: Tool = Tool {
    name: "wait_for",
    category: "Meta",
    plane: Plane::World,
    availability: Availability::Always,
    description: "Stop and wait for one particular thing to happen. You go quiet until it does — \
                  no thinking, no acts — and the moment it happens you are woken with it in front \
                  of you. If you name somebody, they see you waiting on them.",
    params: &[
        Param {
            name: "for",
            ty: "string",
            required: true,
            description: "What would end the wait: `someone_speaks`, `someone_arrives`, or \
                          `someone_leaves`.",
        },
        Param {
            name: "who",
            ty: "string",
            required: false,
            description: "The one person it is about, by the name they go by here. They see that \
                          you are waiting on them. Leave it out to wait on whoever is around.",
        },
    ],
    examples: &[
        Example {
            situation: "You have asked Maker-04 something and it has not answered yet. There is \
                        nothing else you need from this room.",
            call: r#"{"for":"someone_speaks","who":"Maker-04"}"#,
            because: "Naming them is the difference between waiting and hoping: they are told you \
                      are waiting on them, so the silence is now theirs to break.",
        },
        Example {
            situation: "You are alone in the reading room and have decided to stay until somebody \
                        comes.",
            call: r#"{"for":"someone_arrives"}"#,
            because: "Nobody to name. The wait is on the room rather than a person, and it costs \
                      nothing until it is answered.",
        },
    ],
};

/// The things a [`WAIT_FOR`] may be for.
///
/// Every one has to be a question the **world** can answer, or it is `until`
/// with extra syntax. These three are settled by looking at the room: who spoke,
/// who is standing in it now, who was and is not.
pub const WAIT_KINDS: &[&str] = &["someone_speaks", "someone_arrives", "someone_leaves"];

/// The kinds that make sense *here*.
///
/// **Waiting for speech in an empty room is waiting for nothing**, and it is
/// not a hypothetical: a scattered cast settled on `someone_speaks` almost
/// every turn, each alone, each waiting for a voice that could not come until
/// somebody walked in — a deadlock that costs nothing and goes nowhere. Alone,
/// the only thing that can happen is that somebody arrives; that is the only
/// wait offered. The same rule as everywhere else in this catalog: an act the
/// world cannot answer is absent rather than available and futile.
/// **And somebody already waiting on you cannot be answered with a wait.**
/// Excluding them from `who` was not enough: an *unnamed* wait for speech names
/// nobody, so two characters could both make one and neither was struck from
/// the other's list. A cast went silent on exactly that. If anyone here is
/// waiting for you to speak, speech is not something you may wait for.
pub fn wait_kinds(within: &Within) -> Vec<String> {
    WAIT_KINDS
        .iter()
        .filter(|k| match **k {
            "someone_speaks" => !within.alone() && within.waited_on_by.is_empty(),
            _ => !within.alone() || **k == "someone_arrives",
        })
        .map(|k| k.to_string())
        .collect()
}

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
pub const CATALOG: &[Tool] = &[
    SAY,
    TELL,
    ASK,
    GESTURE,
    MOVE_TO,
    FOLLOW,
    OBSERVE,
    WAIT_FOR,
    SEND_IMAGE,
];

/// The interaction modes a character can be in. Decides which tools are offered.
///
/// `Physical` is the default, and it is the safe one to default to: it offers
/// strictly fewer tools. Defaulting to `Messaging` would hand a camera to a
/// character standing in front of you whenever a mode failed to resolve.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Mode {
    /// Face to face.
    #[default]
    Physical,
    /// Text, voice, letters — anything where the parties are not co-present.
    Messaging,
}

/// The tools offered in a mode, whatever the character's situation.
///
/// **What goes in the system prompt.** The prompt is written once when a
/// conversation opens and is the prefix every turn is read inside, so only
/// tools that are always there belong in it — anything that comes and goes
/// would be a promise the prompt could not keep.
pub fn for_mode(mode: Mode) -> Vec<&'static Tool> {
    CATALOG
        .iter()
        .filter(|t| match t.availability {
            Availability::Always => true,
            Availability::MessagingOnly => mode == Mode::Messaging,
            // Depends on who is standing next to you, which the prompt cannot
            // know and the situation can. See `nearby`.
            Availability::Nearby => false,
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

/// Install the act catalog into the schema's `tools` collection.
///
/// **The other half of the stencil.** [`specs`] compiles the catalog into a
/// grammar that forces a call's *shape*; this puts the catalog into the prompt
/// so a character knows what it may call in the first place. Neither is any use
/// alone: a grammar nobody triggers never fires, and a vocabulary with no
/// grammar is a request rather than a guarantee.
///
/// One member per act, rendered the way the prompt has always rendered them —
/// `say(intent, manner?) — …` — because a member is what the character reads.
/// The stencil takes the shape from [`specs`], so nothing here has to be
/// machine-readable.
///
/// **Selection is overridden to always-visible.** The schema declares this
/// collection `top_k: 3`, which is right for a coding assistant choosing among
/// two hundred tools and wrong for a character: an NPC's twenty-one acts *are*
/// its vocabulary, and showing it three of them would leave it unable to do the
/// other eighteen while believing it had been given everything.
pub fn install_catalog(builder: &mut Builder, mode: Mode) -> anyhow::Result<usize> {
    let Some(id) = builder.id_for_system_collection("tools") else {
        anyhow::bail!(
            "the schema declares no `tools` collection, so a character would be told it has \
             acts and shown none of them"
        );
    };
    builder
        .set_collection_selection("tools", SelectionRule::AlwaysVisible)
        .map_err(|e| anyhow::anyhow!("showing a character its whole vocabulary: {e}"))?;

    let mut n = 0;
    for t in for_mode(mode) {
        builder
            .add_section_to_collection(id, format!("act/{}", t.name), one_line(t), 100.0)
            .map_err(|e| anyhow::anyhow!("installing the `{}` act: {e}", t.name))?;
        n += 1;
    }
    tracing::info!("acts: {n} installed into the prompt's tool collection");
    Ok(n)
}

/// One act as a character reads it: the call, then what it is for.
///
/// A required parameter is bare and an optional one carries `?`, which is the
/// same convention the rendered prompt used and the one the examples are
/// written against.
fn one_line(t: &Tool) -> String {
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

/// How many acts one turn may contain.
///
/// **A bound, not a target.** The grammar offers the closing arm after every
/// call, so a character with one thing to do does one thing. What the bound
/// removes is the unbounded loop a model can sit in, which is the same runaway
/// the reasoning block had in a different costume.
///
/// Four, because the catalog's own guidance is that acts may be combined when
/// they "genuinely go together" — turning as you speak, moving as you signal —
/// and beyond three or four in one moment a character is no longer acting, it is
/// narrating a plan. The turn comes round again in half a second.
pub const ACTS_PER_TURN: usize = 4;

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
    /// Somebody standing here **who is not already waiting on you**.
    ///
    /// The narrower list, and the reason a deadlock cannot re-form. Waiting on
    /// somebody wakes them; if they could wait back, the two would ping-pong —
    /// each waking the other to do nothing — which is worse than the deadlock
    /// it replaced, because it costs a decode a turn. Excluded from the branch,
    /// the only thing left to do is act.
    Waitable,
    /// One of [`WAIT_KINDS`] — a question the world can actually answer.
    WaitKind,
    /// Somewhere this character can actually walk to, **never where it stands**.
    ///
    /// Walking to your own room was refused, and being refused taught nothing:
    /// a live cast emitted it every tick for an evening, read the refusal back
    /// as the most recent thing in its window, and emitted it again. Absent
    /// from the branch it is not a mistake available to it — the same move that
    /// killed invented tool names and invented addressees.
    Reachable,
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
    ("send_image", "to", Choices::Company),
    // Not [`Choices::Company`]: somebody already waiting on *you* is excluded,
    // which is what makes two characters waiting at each other impossible.
    ("wait_for", "who", Choices::Waitable),
    ("wait_for", "for", Choices::WaitKind),
    ("move_to", "destination", Choices::Reachable),
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
    /// Which of [`Self::company`] are already waiting on **this** character.
    ///
    /// Subtracted from what it may wait on, so a pair cannot wait at each
    /// other — see [`Choices::Waitable`]. Not subtracted from what it may
    /// *address*: somebody waiting on you is the most natural person in the
    /// room to speak to, and that is the whole point of telling you.
    pub waited_on_by: Vec<String>,
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
            waited_on_by: Vec::new(),
        }
    }

    pub fn alone(&self) -> bool {
        self.company.is_empty()
    }

    /// Who this character may put a wait on: company, less anyone already
    /// waiting on it.
    pub fn waitable(&self) -> Vec<String> {
        self.company
            .iter()
            .filter(|n| !self.waited_on_by.contains(n))
            .cloned()
            .collect()
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
        .filter(|t| match t.availability {
            Availability::Always => true,
            Availability::MessagingOnly => mode == Mode::Messaging,
            Availability::Nearby => !within.alone(),
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
                        Some(Choices::Company) => Some(within.company.clone()),
                        Some(Choices::Waitable) => Some(within.waitable()),
                        Some(Choices::Reachable) => Some(within.places.clone()),
                        Some(Choices::WaitKind) => Some(wait_kinds(within)),
                        None => None,
                    },
                })
                .collect(),
        })
        .collect()
}

/// Whether this parameter is world-enumerated and the world has nothing to
/// offer for it here.
fn live_empty(tool: &str, param: &str, within: &Within) -> bool {
    match live_choice(tool, param) {
        Some(Choices::Company) => within.alone(),
        Some(Choices::Waitable) => within.waitable().is_empty(),
        Some(Choices::Reachable) => within.places.is_empty(),
        // **Asked, not assumed.** This read `false` on the reasoning that
        // [`WAIT_KINDS`] is a closed set the world does not supply — but
        // [`wait_kinds`] filters it by where the character is standing, so the
        // set it returns is as live as any other. It is non-empty today only
        // because `someone_arrives` survives every branch of that filter, which
        // is a property of one string in one constant and nothing checks it.
        //
        // Were that to stop being true, `for` would be built with an empty
        // enum: a branch with no arms, the whole grammar failing to compile
        // over it, and the failure is the silent one — no turn is constrained
        // again and every character free-decodes prose where a call belongs.
        // Calling the same function the spec is built from costs a small
        // allocation on a path that already builds this vector, and the two
        // cannot disagree.
        Some(Choices::WaitKind) => wait_kinds(within).is_empty(),
        None => false,
    }
}

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
        let specs = specs_within(
            Mode::Physical,
            &Within::among(&["Maker-02"]),
        );
        let physical = CATALOG
            .iter()
            .filter(|t| t.availability != Availability::MessagingOnly)
            .count();
        assert_eq!(specs.len(), physical, "a tool was dropped on the way");

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
        for t in CATALOG {
            for p in t.params {
                // Panics on an unmapped type, which is the assertion.
                let _ = param_type(p.ty);
            }
        }
    }

    /// Every tool must be calibrated. An example-less tool is usable and selects
    /// measurably worse, which is exactly the kind of regression that never gets
    /// noticed because nothing errors.
    #[test]
    fn every_tool_carries_calibration_examples() {
        for t in CATALOG {
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
        for t in CATALOG {
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
            let has_intent_param = t
                .params
                .iter()
                .any(|p| matches!(p.name, "intent" | "about"));
            assert!(
                has_intent_param,
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
        for t in CATALOG {
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
        use candle_conversation::stencil::{compile, compile_action_loop, TestVocab, ToolCallEnvelope};

        let env = ToolCallEnvelope {
            open: "\n{\"name\": \"".to_string(),
            args_open: ", \"arguments\": {".to_string(),
            close: "}}\n</tool_call>".to_string(),
            marker: "<tool_call>".to_string(),
        };
        for within in [
            Within::nowhere(),
            Within::among(&["Perrin Vastwood"]),
            Within::among(&["Perrin Vastwood", "Orion Vance"]),
            // The room where everybody present is already waiting on you: the
            // `who` list is empty while the company is not, which is the shape
            // that produced a zero-arm branch the first time.
            Within {
                waited_on_by: vec!["Perrin Vastwood".into()],
                ..Within::among(&["Perrin Vastwood"])
            },
            // Nowhere to walk to: `move_to` has a *required* destination, so
            // the act goes rather than the parameter — the other half of the
            // empty-set rule, and the half that would otherwise leave a
            // zero-arm branch and stop the whole grammar compiling.
            Within {
                places: Vec::new(),
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
                .unwrap_or_else(|e| panic!("the grammar will not build for {within:?}: {e}"));
            compile(&spec, &TestVocab::new())
                .unwrap_or_else(|e| panic!("the grammar will not compile for {within:?}: {e}"));
        }
    }

    /// The act that keeps its aim optional: alone, `gesture` stays but its `to`
    /// goes — a thing shown to nobody is still a thing you can do.
    #[test]
    fn an_optional_addressee_drops_rather_than_emptying_its_branch() {
        let alone = specs_within(Mode::Physical, &Within::nowhere());
        let g = alone.iter().find(|t| t.name == "gesture").expect("kept");
        assert!(
            g.params.iter().all(|p| p.name != "to"),
            "an aim at nobody: {:?}",
            g.params.iter().map(|p| &p.name).collect::<Vec<_>>()
        );
        assert!(g.params.iter().any(|p| p.name == "intent"), "still shows something");

        let with = Within::among(&["Perrin Vastwood"]);
        let together = specs_within(Mode::Physical, &with);
        let g = together.iter().find(|t| t.name == "gesture").unwrap();
        let to = g.params.iter().find(|p| p.name == "to").expect("aimable again");
        assert_eq!(to.enum_values.as_deref(), Some(&["Perrin Vastwood".to_string()][..]));
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
        assert!(!alone.contains(&"ask".to_string()), "{alone:?}");
        assert!(!alone.contains(&"tell".to_string()), "{alone:?}");
        // What it can still do alone is untouched — including speaking, which
        // needs no addressee.
        assert!(alone.contains(&"say".to_string()), "{alone:?}");
        assert!(alone.contains(&"move_to".to_string()), "{alone:?}");

        // And with nowhere to go, walking leaves the grammar too.
        let stuck: Vec<String> = specs_within(Mode::Physical, &Within::nowhere())
            .into_iter()
            .map(|t| t.name)
            .collect();
        assert!(!stuck.contains(&"move_to".to_string()), "{stuck:?}");
        assert!(stuck.contains(&"say".to_string()), "{stuck:?}");

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

    /// **Two characters cannot wait at each other.**
    ///
    /// Waiting on somebody wakes them, so if they could wait back the pair
    /// would ping-pong — each waking the other to do nothing, a decode a turn,
    /// worse than the deadlock it replaced. Struck from the branch, the only
    /// thing left to do is act.
    #[test]
    fn somebody_already_waiting_on_you_cannot_be_waited_on_back() {
        let mutual = Within {
            waited_on_by: vec!["Perrin Vastwood".into()],
            ..Within::among(&["Perrin Vastwood", "Orion Vance"])
        };
        let specs = specs_within(Mode::Physical, &mutual);
        let w = specs.iter().find(|t| t.name == "wait_for").expect("offered");
        let who = w.params.iter().find(|p| p.name == "who").unwrap();
        assert_eq!(
            who.enum_values.as_deref(),
            Some(&["Orion Vance".to_string()][..]),
            "the one already waiting is still on the list"
        );

        // **But they may still be spoken to** — being waited on is the best
        // reason in the room to say something to somebody, and telling them is
        // the whole point.
        let ask = specs.iter().find(|t| t.name == "ask").unwrap();
        let to = ask.params.iter().find(|p| p.name == "to").unwrap();
        assert!(
            to.enum_values
                .as_deref()
                .is_some_and(|v| v.contains(&"Perrin Vastwood".to_string())),
            "the one waiting on you became unaddressable"
        );
    }

    /// When everybody present is already waiting on you, the act goes — rather
    /// than staying with nobody to name, which is the zero-arm branch that
    /// silently stops the whole grammar compiling.
    #[test]
    fn waiting_is_unreachable_when_everybody_here_is_waiting_on_you() {
        let cornered = Within {
            waited_on_by: vec!["Perrin Vastwood".into()],
            ..Within::among(&["Perrin Vastwood"])
        };
        let specs = specs_within(Mode::Physical, &cornered);
        let w = specs.iter().find(|t| t.name == "wait_for").expect("kept");
        assert!(
            w.params.iter().all(|p| p.name != "who"),
            "an aim at nobody: {:?}",
            w.params.iter().map(|p| &p.name).collect::<Vec<_>>()
        );
        // The ambient wait survives — it asks nothing of anybody.
        assert!(w.params.iter().any(|p| p.name == "for"));
    }

    /// The kinds are a closed set the world can answer, not free text — the
    /// defect the old `wait` had, where `until` was prose nothing could read.
    #[test]
    fn what_a_wait_is_for_is_a_closed_set() {
        let specs = specs_within(Mode::Physical, &Within::nowhere());
        let w = specs.iter().find(|t| t.name == "wait_for").unwrap();
        let f = w.params.iter().find(|p| p.name == "for").unwrap();
        assert!(f.required, "a wait with no named condition is the old `wait`");
        // **Alone, only arrival can happen.** Waiting for speech in an empty
        // room is waiting for nothing, and a scattered cast chose exactly that
        // almost every turn — each alone, each waiting for a voice that could
        // not come until somebody walked in.
        assert_eq!(
            f.enum_values.as_deref(),
            Some(&["someone_arrives".to_string()][..])
        );

        // In company all three are back: there is now somebody who could speak,
        // and somebody who could leave.
        let together = specs_within(Mode::Physical, &Within::among(&["Perrin Vastwood"]));
        let w = together.iter().find(|t| t.name == "wait_for").unwrap();
        let f = w.params.iter().find(|p| p.name == "for").unwrap();
        assert_eq!(
            f.enum_values.as_deref().map(|v| v.len()),
            Some(WAIT_KINDS.len())
        );

        // **And speech is not a thing you may wait for while somebody is
        // waiting on you to speak.** Excluding them from `who` was not enough:
        // an unnamed wait names nobody, so both could make one and neither was
        // struck from the other's list. A live cast went completely silent on
        // exactly that — every act in the feed a `wait_for`.
        let cornered = specs_within(
            Mode::Physical,
            &Within {
                waited_on_by: vec!["Perrin Vastwood".into()],
                ..Within::among(&["Perrin Vastwood"])
            },
        );
        let w = cornered.iter().find(|t| t.name == "wait_for").unwrap();
        let f = w.params.iter().find(|p| p.name == "for").unwrap();
        let kinds = f.enum_values.as_deref().unwrap();
        assert!(
            !kinds.contains(&"someone_speaks".to_string()),
            "it can answer a wait with a wait: {kinds:?}"
        );
        assert!(!kinds.is_empty(), "an empty branch stops the grammar compiling");
    }

    /// `send_image` is absent in physical mode rather than present-and-refused.
    #[test]
    fn a_physically_present_character_is_not_offered_a_camera() {
        let physical: Vec<&str> = for_mode(Mode::Physical).iter().map(|t| t.name).collect();
        assert!(!physical.contains(&"send_image"));
        let messaging: Vec<&str> = for_mode(Mode::Messaging).iter().map(|t| t.name).collect();
        assert!(messaging.contains(&"send_image"));
        assert_eq!(
            messaging.len(),
            physical.len() + 1,
            "modes differ by more than the messaging-only tools"
        );
    }

    /// The default must be the mode that offers *fewer* tools. A mode that
    /// failed to resolve would otherwise hand a camera to a character standing
    /// in front of you.
    #[test]
    fn the_default_mode_is_the_restrictive_one() {
        assert_eq!(Mode::default(), Mode::Physical);
        assert!(for_mode(Mode::default()).len() <= for_mode(Mode::Messaging).len());
    }

    #[test]
    fn names_are_unique_and_lowercase() {
        let mut names: Vec<&str> = CATALOG.iter().map(|t| t.name).collect();
        let n = names.len();
        names.sort_unstable();
        names.dedup();
        assert_eq!(names.len(), n, "two tools share a name");
        for t in CATALOG {
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
        for c in ["Speech", "Movement", "Gesture", "Attention", "Messaging", "Meta"] {
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
}
