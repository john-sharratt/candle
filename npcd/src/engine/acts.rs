//! The acts that touch the world's state rather than its shape.
//!
//! [`super::tools`] holds the acts of speech, attention and movement — what a
//! body does with other bodies and with rooms. These are what it does with
//! *things*: what it carries, what it works, what it digs out of the ground,
//! what it fights, and what a tower does when somebody tells it to.
//!
//! # Two consumers, two kinds of argument
//!
//! The rule every parameter below is checked against, and the one that is
//! invisible when broken:
//!
//! > **A free-text argument is only ever read by a model.** If formula code
//! > consumes it, the argument is an enumerated value or a bound entity —
//! > because code cannot read a sentence, and a string nothing reads is written
//! > every turn and costs a decode to produce.
//!
//! So `act` takes an intent, because a narrator renders it into prose. And
//! `engage` takes a posture, a priority and a filter, because a simulator
//! branches on them. A `guidance` string on `engage` would look exactly like
//! control and do nothing at all, which is worse than a missing parameter:
//! nothing fails to reveal it.
//!
//! # Every parameter here is a string or a boolean, and that is a hard rule
//!
//! The stencil compiles a closed branch for an enum, a bounded span for a
//! string, and a `true`/`false` branch for a boolean. For `integer`, `number`,
//! `array` and `object` it compiles *any structurally-valid JSON value*, which
//! is unbounded — and a JSON value nests without limit, so materialising those
//! states does not terminate. Measured, not theorised: one `integer` parameter
//! in this catalog made the grammar compile allocate ~390 MB/second forever and
//! take the machine down with it.
//!
//! So a count is a string and a coordinate is a string, parsed by the act that
//! receives it. That reads like a weaker type and is a stronger guarantee: the
//! grammar can hold a string to a shape and cannot hold a JSON value to
//! anything. [`super::tools::tests::every_parameter_is_a_type_the_grammar_can_bound`]
//! is what stops this coming back.
//!
//! The tactic that will not enumerate is **spoken** — *"the drones first, the
//! mech is somebody else's problem"* is a `tell` to the squad. The narrator
//! renders it, the others perceive it, and it lands where language works.
//!
//! # Examples are calibration, not documentation
//!
//! Each tool carries worked examples, prefilled at startup. A tool with no
//! examples selects measurably worse, and the discriminating detail — *why this
//! call and not the neighbouring one* — is what calibration actually teaches.

use super::tools::{Availability, Example, Param, Plane, Tool};

// ---------------------------------------------------------------------------
// The body, continued
// ---------------------------------------------------------------------------

/// A physical act that lands on a person.
///
/// **Not `gesture`.** A gesture is shown to a room and lands on nobody; this is
/// done *to* somebody, who feels it, may refuse it, and answers it. The world
/// can represent the difference — a target who experiences the act against an
/// audience who observes one — and everything that hangs on it is material:
/// consent, resistance, and what the other party does next.
///
/// # Why it is `act` and not `touch`
///
/// It was `touch`, and the name was doing damage in two directions.
///
/// It read as *gentle contact* — steadying an elbow, a hand on a shoulder — in
/// a game about a war. This is the act a character reaches for to break a jaw,
/// take a weapon off somebody or hold them against a wall, and a model choosing
/// between named tools reads the name first. A character with no way to name
/// what it was doing did not do it.
///
/// And it did not match the console. The person standing in the room does the
/// same thing with `/act` (`engine::slash`), so the two halves of one exchange —
/// what you do to a character and what it does back — had different names for
/// one idea. One concept, one word, whichever side of the room it comes from.
pub const ACT: Tool = Tool {
    name: "act",
    at: &[],
    category: "Contact",
    plane: Plane::World,
    // You cannot put a hand on somebody down a line, and being invited to is
    // what makes a character try.
    //
    // **Reachable while alone, on purpose.** `yourself` is a real target — a
    // character binding its own wound, getting its own weapon clear or dragging
    // itself up off the floor — so the tool stays even with nobody to touch.
    // What that cost is *frequency*: a solitary cast chose it fifty-three times
    // out of fifty-three, and the answer to that is the cooldown a self-act
    // serves, not taking the act away. See `cooldown::SELF_ACT`.
    availability: Availability::PhysicalOnly,
    description: "Do something physical to one person here. Anything your body can do to \
                  theirs: steady them, block their way, take something out of their hand, put \
                  yourself between them and something else — or put them on the floor, break \
                  their grip, hurt them. Like `say`, you give what you MEAN by it and not the \
                  choreography; the narrator renders the movement. They feel it, they may \
                  refuse it, and what they do next is theirs.",
    params: &[
        Param {
            name: "on",
            ty: "string",
            required: true,
            description: "Who you are doing it to, exactly as their name appears where you are. \
                          They must be here. `yourself` to do it to your own body.",
        },
        Param {
            name: "intent",
            ty: "string",
            required: true,
            description: "What you mean by it. Substance, not movement: \"steady her before she \
                          goes over\" or \"put him down before he reaches the door\" — never \
                          \"I put my hand under her elbow\".",
        },
    ],
    examples: &[
        Example {
            situation: "Someone you have been arguing with turns for the door, and the thing you \
                        actually needed to say is still unsaid.",
            call: r#"{"on":"Perrin Vastwood","intent":"stop him leaving, without making it a hold"}"#,
            because: "The act is aimed at one person and they can refuse it, which `gesture` \
                      cannot express — a gesture would only have been seen.",
        },
        Example {
            situation: "The one who came back from the ridge is on their feet and should not be.",
            call: r#"{"on":"Wren","intent":"take the weight off her before she stands on it again"}"#,
            because: "Contact, not signal. Telling her to sit down is `tell`; this is doing it.",
        },
        Example {
            situation: "He has a knife out and is between you and the only way down.",
            call: r#"{"on":"Hess","intent":"take the knife off him, and put him down if he keeps hold of it"}"#,
            because: "The same act as steadying somebody, and the intent is what makes it \
                      violence. There is no separate tool for a blow — a body does one kind of \
                      thing to another body, and what it means is the argument.",
        },
        Example {
            situation: "The bleeding has not stopped and there is nobody here to do it for you.",
            call: r#"{"on":"yourself","intent":"get the wound closed before it costs me the arm"}"#,
            because: "A body is a thing you can act on, including your own. Everybody in the \
                      room sees it and nobody else feels it, which is exactly what `yourself` \
                      means here.",
        },
    ],
};

/// Stop, for a stretch of the world's time.
///
/// **Not `pause`.** A pause is a moment and stays responsive to everything —
/// anything at all brings the character straight back. Sleeping is a stretch of
/// the day and is unresponsive by design, which is exactly why something has to
/// be able to break through it. Being woken is a perception, not a failure —
/// the duration is an intention the world may end early.
///
/// It is also what makes a day mean anything. Two clusters of the repertoire
/// open and close one, and a character with nothing to do that sleeps until
/// dawn has *decided* something, where one nudged awake after ninety seconds
/// has merely been interrupted.
pub const SLEEP: Tool = Tool {
    name: "sleep",
    at: &[],
    // **Meta, which is what its own plane has always said.** Filed under
    // `Attention` while declaring `Plane::Meta`, which is a disagreement about
    // what the act *is*: attention is turning towards something, and this is
    // the act of turning away from everything until a named hour. `reflect` is
    // the one that attends.
    category: "Meta",
    plane: Plane::Meta,
    availability: Availability::Embodied,
    description: "Stop, and stay stopped until a time you name. You will not answer what happens \
                  around you while you are down — that is the difference between this and \
                  waiting. Anything urgent enough will wake you anyway, and being woken is not a \
                  failure.",
    params: &[Param {
        name: "until",
        ty: "string",
        required: true,
        description: "When you mean to be up again.",
    }],
    examples: &[
        Example {
            situation: "You have filed the thing you were holding, there is nothing on the board \
                        you could start tonight, and the level is empty.",
            call: r#"{"until":"dawn"}"#,
            because: "Choosing to stop is a real act. Standing in an empty room until something \
                      happens to you is not.",
        },
        Example {
            situation: "You are to go out with the others when the light comes up, and you have \
                        been awake since the last sortie.",
            call: r#"{"until":"the next bell"}"#,
            because: "Short and deliberate. Going out tired is a decision too, and this is the \
                      other one.",
        },
    ],
};

/// Hand a thing over.
pub const GIVE: Tool = Tool {
    name: "give",
    at: &[],
    category: "Contact",
    plane: Plane::World,
    availability: Availability::Nearby,
    description: "Hand something you are carrying to somebody here. It leaves you and it arrives \
                  with them; you cannot give what you do not have, and you cannot give more of a \
                  thing than you are carrying.",
    params: &[
        Param {
            name: "what",
            ty: "string",
            required: true,
            description: "What you are handing over, by the name it goes by.",
        },
        Param {
            name: "to",
            ty: "string",
            required: true,
            description: "Who you are handing it to. They must be here.",
        },
        Param {
            name: "count",
            ty: "string",
            required: false,
            description: "How many, when it is the kind of thing there can be several of. One by \
                          default.",
        },
    ],
    examples: &[
        Example {
            situation: "The one holding the line beside you has been firing single shots for a \
                        while and you are carrying sixty rounds.",
            call: r#"{"what":"bolt rounds","to":"Wren","count":30}"#,
            because: "Half, not all. What is given is gone, so the count is the decision.",
        },
        Example {
            situation: "You are handing on a thing you took charge of, to whoever takes it next.",
            call: r#"{"what":"the eastern survey","to":"Perrin Vastwood"}"#,
            because: "One of a thing needs no count. Saying what you did to it is a separate act \
                      and belongs in `tell`.",
        },
    ],
};

// ---------------------------------------------------------------------------
// Working the world
// ---------------------------------------------------------------------------

/// Read what a station holds.
///
/// **One tool, not twenty-nine.** Reading the era you are holding and reading
/// the dispatch board are not two acts; they are one act against two things, and
/// which things is a fact about where a body is standing. That is what `Choices`
/// is for, and a named tool per readable thing would make the model choose among
/// names it must remember rather than among things that are actually there.
///
/// **This is the only act for attending to anything**, now that `observe` is
/// gone. The two were never the same: looking around a room returned what the
/// percept had already handed over, where this returns *recorded content* —
/// what is actually written on the board, which nothing else in the engine
/// delivers and which the character cannot know until it reads it.
pub const READ: Tool = Tool {
    name: "read",
    at: &[],
    category: "Attention",
    plane: Plane::Internal,
    availability: Availability::Always,
    description: "Read something the place you are standing holds — a board, a panel, a page, a \
                  terminal's subject. What comes back is its contents, not a description of it. \
                  You are already told what is in the room and who is in it; this is for what is \
                  written down, which you cannot know until you read it.",
    params: &[Param {
        name: "what",
        ty: "string",
        required: true,
        description: "What you are reading, from what is actually here to read.",
    }],
    examples: &[Example {
        situation: "You have just come to the board and do not know what is outstanding.",
        call: r#"{"what":"the muster board"}"#,
        because: "Spending a step finding out beats acting on what you assume is still true.",
    }],
};

/// Leave words on something, for whoever comes by.
///
/// # The other half of `read`
///
/// A world where things can be read and not written is one where everything
/// readable had to be authored before anybody arrived. This is what lets the
/// people living in a place leave something in it — and what makes a board a
/// board rather than a decorated wall.
///
/// # Why it is not `say` written down
///
/// Speech reaches whoever is standing there, now, and is gone. This reaches
/// whoever comes to this spot afterwards, and keeps. That is a different act
/// with a different audience — the character writing it is addressing people
/// who are not in the room and may not be born yet, which is the whole
/// difference between telling somebody and posting a notice.
///
/// # Why the surface is enumerated
///
/// `on` binds to [`crate::engine::tools::Choices::Postable`], which is the
/// fixtures actually standing here. A character cannot invent a board, and
/// where there is nothing to write on the act leaves the grammar — the same
/// empty-set rule the phone acts run on, rather than an `AtPart` list that
/// would have to name every board id in every map for ever.
pub const POST_NOTICE: Tool = Tool {
    name: "post_notice",
    at: &[],
    category: "Attention",
    plane: Plane::World,
    availability: Availability::Always,
    description: "Write something on a board, a panel or a page here, for whoever comes to it \
                  next. It stays until it is pushed off the bottom by newer things, and everybody \
                  who reads it is told you wrote it. Use this when what you have to say outlives \
                  the people currently in the room — `say` reaches whoever is standing here now \
                  and is gone.",
    params: &[
        Param {
            name: "on",
            ty: "string",
            required: true,
            description: "What you are writing on, named exactly as it stands here.",
        },
        Param {
            name: "what",
            ty: "string",
            required: true,
            description: "What you are leaving, in one line somebody arriving cold can act on. \
                          Substance, not a finished sentence — the wording follows.",
        },
    ],
    examples: &[
        Example {
            situation: "You have just found that the lift on five is not answering, and you are \
                        the only one who knows.",
            call: r#"{"on":"the muster board","what":"that the lift on five is not answering and the stairwell is the only way up until somebody looks at it"}"#,
            because: "It matters to people who are not here, and will still matter in an hour. \
                      Saying it to an empty room reaches nobody and keeps nothing.",
        },
        Example {
            situation: "You are leaving a subject half-finished and somebody else will pick it \
                        up before you are back.",
            call: r#"{"on":"the accession desk","what":"that the third era is written up twice and I have not settled which is right — do not file either yet"}"#,
            because:
                "A handover is exactly the thing that has to outlast the handshake. The next \
                      person to stand here reads it whether or not anybody remembered to tell them.",
        },
    ],
};

/// Take something on, and hold it.
///
/// The six `take_*` acts an earlier catalog had were one operation with six
/// names. Taking a station and taking an order off a board are both *taking
/// something on*, and a character does not distinguish them.
pub const CLAIM: Tool = Tool {
    name: "claim",
    at: &[],
    category: "Work",
    plane: Plane::World,
    availability: Availability::Always,
    description: "Take something on: a station to work at, an order off a board, a subject nobody \
                  else has. It is yours until you give it back or walk away, and while you hold \
                  it nobody else can. Taking one thing does not release another.",
    params: &[Param {
        name: "what",
        ty: "string",
        required: true,
        description: "What you are taking on, from what is actually free here.",
    }],
    examples: &[
        Example {
            situation: "You have decided which of the unwritten spans you would most regret \
                        leaving, and nobody is holding it.",
            call: r#"{"what":"close the longest silence in the record"}"#,
            because: "Saying out loud what you are taking is what stops two people doing it.",
        },
        Example {
            situation: "You mean to work, and the bay you want is free.",
            call: r#"{"what":"fabricator 3"}"#,
            because: "A station is claimed while it is worked, so two bodies cannot queue over \
                      each other.",
        },
    ],
};

/// Give back what you are holding, without walking away.
pub const RELEASE: Tool = Tool {
    name: "release",
    at: &[],
    category: "Work",
    plane: Plane::World,
    availability: Availability::Always,
    description: "Give back what you are holding so somebody else can take it. Walking away does \
                  this too; use this when you mean to stay where you are.",
    params: &[],
    examples: &[Example {
        situation: "You could finish the thing you are holding, and somebody who needs it more \
                    has just said so.",
        call: r#"{}"#,
        because: "Giving a thing back is an act with a cost, and it is not the same as leaving.",
    }],
};

/// Ready a piece of kit.
pub const EQUIP: Tool = Tool {
    name: "equip",
    at: &[],
    category: "Kit",
    plane: Plane::World,
    availability: Availability::Embodied,
    description: "Ready something you are carrying — put it on, or take it in hand. What is \
                  readied is what you fight and work with; the rest is weight.",
    params: &[Param {
        name: "what",
        ty: "string",
        required: true,
        description: "What you are readying, from what you carry that can be.",
    }],
    examples: &[Example {
        situation: "You are going out through the gate and everything you have is in the pack.",
        call: r#"{"what":"combat armour"}"#,
        because: "Readying is a separate act from carrying, and going out with it stowed is a \
                  mistake that is only obvious afterwards.",
    }],
};

/// Spend or work a thing you carry.
pub const USE: Tool = Tool {
    name: "use",
    at: &[],
    category: "Kit",
    plane: Plane::World,
    availability: Availability::Embodied,
    description: "Use something you carry, on yourself or on somebody here. A thing that is spent \
                  is gone; a thing that is merely worked is not.",
    params: &[
        Param {
            name: "what",
            ty: "string",
            required: true,
            description: "What you are using, from what you carry that can be used.",
        },
        Param {
            name: "on",
            ty: "string",
            required: false,
            description: "Who you are using it on, if not yourself. They must be here.",
        },
    ],
    examples: &[
        Example {
            situation: "The one who came back on a stretcher is not going to last the hour, and \
                        you have three stimpaks.",
            call: r#"{"what":"stimpak","on":"Wren"}"#,
            because: "One tool covers healing, repairing and reviving, because they are all \
                      spending a thing on somebody.",
        },
        Example {
            situation: "There is something on the ridge and it is too far to make out.",
            call: r#"{"what":"advanced scanner"}"#,
            because: "Worked rather than spent, and aimed at nothing in particular — the scanner \
                      is still there afterwards.",
        },
    ],
};

/// Take something out of the ground, a wreck or a cache.
pub const GATHER: Tool = Tool {
    name: "gather",
    at: &[],
    category: "Field",
    plane: Plane::World,
    availability: Availability::Embodied,
    description: "Work something here for what is in it — a seam, a bloom, a wreck, a cache. What \
                  comes out goes into your pack, and what you take is gone from the ground. A \
                  worked-out deposit is not offered.",
    params: &[Param {
        name: "what",
        ty: "string",
        required: true,
        description: "What you are working, from what is actually here with something in it.",
    }],
    examples: &[Example {
        situation: "You were sent for ore and the seam on the east ridge is the near one.",
        call: r#"{"what":"the ore seam"}"#,
        because: "Mining, stripping a wreck and picking over a cache are one act against \
                  different things.",
    }],
};

/// **The one combat tool.**
///
/// The simulator fights; this says how. Every argument is an enumerated value or
/// a bound entity, because the consumer is formula code and a decode is far too
/// slow to aim a weapon. A character's part is to set a stance and change its
/// mind about it, which is the same division of labour `say` keeps with the
/// narrator: the model supplies substance, something faster supplies mechanism.
pub const ENGAGE: Tool = Tool {
    name: "engage",
    at: &[],
    category: "Field",
    plane: Plane::World,
    availability: Availability::Embodied,
    description: "Say how you are fighting. You do not aim and you do not fire — that happens far \
                  faster than you can think about it. What you decide is the posture you hold, \
                  who you single out, what to do about everything else, and what to honour while \
                  you do it. The stance stands until you change it. Anything that will not fit \
                  these words is something you should be SAYING to the people beside you.",
    params: &[
        Param {
            name: "posture",
            ty: "string",
            required: true,
            description: "How you are fighting. `break off` is how you stop.",
        },
        Param {
            name: "target",
            ty: "string",
            required: false,
            description: "One thing you are singling out, if you are. Everything else is decided \
                          by `priority`.",
        },
        Param {
            name: "priority",
            ty: "string",
            required: false,
            description: "How to choose among everything you have not singled out.",
        },
        // **There is no third enumerated argument, and that is a budget rather
        // than an oversight.** The grammar's compile is exponential in the
        // number of distinct decodes a turn admits (see
        // [`super::tools::estimated_paths`]), and each enumerated argument
        // multiplies that count. Two is what a four-act turn affords.
        //
        // Little is lost. Every constraint worth honouring was already sayable
        // as one of the other two — `cover` *is* staying in cover, `hold` is
        // holding ground, `conserve ammunition` is what `hold fire` and
        // `suppress` differ over — and the one genuinely new axis, whether to
        // work around the player, is what `priority` carries. Anything that
        // still will not fit belongs in a `tell` to the squad, where language
        // works and a grammar is not being asked to enumerate it.
    ],
    examples: &[
        Example {
            situation: "Two drones are on the ridge, you are carrying the wounded, and the way \
                        back is behind you.",
            call: r#"{"posture":"fall back","priority":"whatever is firing on us"}"#,
            because: "Attacking while retreating is a posture and a set of constraints, not a \
                      sentence. Nothing here needs a weapon's cycle time to be understood.",
        },
        Example {
            situation: "The mech is the thing that matters and the player is somewhere in the \
                        middle of it.",
            call: r#"{"posture":"press","target":"a mech","priority":"greatest threat"}"#,
            because: "Singling one thing out and constraining how, in one act. Withholding the \
                      heavy weapons is a filter because the simulator can act on it.",
        },
        Example {
            situation: "It is over, and standing here costs something.",
            call: r#"{"posture":"break off"}"#,
            because: "There is no separate act for disengaging. Breaking off is a posture.",
        },
    ],
};

/// Put a machine into a state.
///
/// Doors, turrets, bays, carriers, terminals. One tool because they are one kind
/// of object: something standing in a room, in a state, that a body can put into
/// a different one. **A turret's targeting policy is a mode** — *air only*,
/// *nearest first*, *conserve* are the enumerated form of what a sentence would
/// have tried to say, and they are values the firing code branches on.
pub const OPERATE: Tool = Tool {
    name: "operate",
    at: &[],
    category: "Work",
    plane: Plane::World,
    availability: Availability::Always,
    description: "Put something here into a different state — open or lock a door, set what a \
                  turret is allowed to shoot at, start or stop a bay, open a working copy at a \
                  terminal. Each thing has its own states and takes no others.",
    params: &[
        Param {
            name: "what",
            ty: "string",
            required: true,
            description: "The thing you are working, from what stands here.",
        },
        Param {
            name: "mode",
            ty: "string",
            required: true,
            description: "The state you are putting it into. It must be one of that thing's own \
                          states — a door does not take a turret's.",
        },
    ],
    examples: &[
        Example {
            situation: "Something is coming across the flats and the gate is standing open.",
            call: r#"{"what":"the blast door","mode":"locked"}"#,
            because: "Locked is not closed. A closed door is one somebody can open.",
        },
        Example {
            situation: "Drones are inbound and there are people still outside the wall.",
            call: r#"{"what":"wall turret 1","mode":"air only"}"#,
            because: "You are not aiming it. You are deciding what it is allowed to do without \
                      being asked, which is the only thing about it that is yours.",
        },
    ],
};

/// Port home.
pub const RECALL: Tool = Tool {
    name: "recall",
    at: &[],
    category: "Movement",
    plane: Plane::World,
    // **Absent while you are already home, not refused there.**
    //
    // It was `Embodied`, which is true and not enough: a body standing at its
    // own muster point was still offered the journey back to it, `World::place`
    // accepted the move to the room it was already in, and the act reported
    // "the ground goes out from under you" having done nothing. Twenty-four of
    // sixty acts across a live cast, and the characters could see the loop
    // without being able to leave it.
    //
    // Being embodied is implied — a body is what has somewhere to be called
    // back from — so nothing is lost by naming the stricter condition.
    availability: Availability::AwayFromHome,
    description: "Go straight back to where you muster from. It crosses nothing on the way, it \
                  always goes to the same place, and it can fail — there may not be the power for \
                  it, or something may be in the way of it.",
    params: &[],
    examples: &[Example {
        situation: "You have what you were sent for and the ground between here and the gate has \
                    something on it.",
        call: r#"{}"#,
        because: "Walking is `move_to`. This is the journey that is not walked, and it is the \
                  reason going out is survivable.",
    }],
};

/// Look somewhere you are not.
pub const SCAN: Tool = Tool {
    name: "scan",
    // Not station-bound: a body carrying a scanner reaches somewhere it is not
    // without standing at anything. The bridge and the sensor panel are the
    // long-range version of the same act, and a mind with no body has only
    // those — which is `Availability` rather than attachment.
    at: &[],
    category: "Attention",
    plane: Plane::Internal,
    availability: Availability::Always,
    description: "Look at somewhere you are not, through instruments. Name the place you want \
                  looked at. It tells you nothing about the room you are standing in — you are \
                  told that already.",
    // **`at` is required, and it was the whole bug that it was not.**
    //
    // This act takes a place *or* a pair of coordinates, and every one of the
    // three was optional — a disjunction the flat grammar has no way to state,
    // so what it actually said was "all three may be absent". A character
    // emitted `scan` with nothing in it and `enact::scan` refused it, every
    // time, unavoidably: measured live, twelve of one character's sixteen acts
    // were that refusal.
    //
    // So the grammar offers the shape that cannot be wrong — a place, from the
    // live set of places there are — and the coordinate form stays reachable
    // through the API and the harness, which are not grammar-constrained. That
    // is the same split `operate` already makes for a mode a device does not
    // admit.
    params: &[Param {
        name: "at",
        ty: "string",
        required: true,
        description: "The place you want looked at, named exactly as it is written.",
    }],
    examples: &[
        Example {
            situation: "You want to know whether the ridge is clear before anybody walks it.",
            call: r#"{"at":"the east ridge"}"#,
            because: "A named place, because a name is harder to get wrong than a pair of \
                      numbers — and because the name is one the world handed you.",
        },
        Example {
            situation: "The watch reported movement somewhere nobody has eyes on.",
            call: r#"{"at":"the gatehouse"}"#,
            because: "Looking somewhere you are not is the whole act. Where you are standing \
                      is already in front of you and is never worth a scan.",
        },
    ],
};

// ---------------------------------------------------------------------------
// The tower
// ---------------------------------------------------------------------------

/// Tell the tower to do one of the enormous things it does.
///
/// One tool with a bound action rather than a verb per capability, and the
/// action list is **what the tower can afford this minute**: folding costs
/// energy it may not have, a tower in the ground has to surface first, and a
/// siege already under way cannot be opened twice. A tower too poor to fold does
/// not offer the option, so the economy is enforced where a character cannot
/// argue with it.
pub const COMMAND_TOWER: Tool = Tool {
    name: "command_tower",
    // The tower is spoken to as a whole from the one place built for it. A body
    // on the open ground can no more fold the tower than it can read a terminal
    // two floors up.
    at: &["bridge-console"],
    category: "Tower",
    plane: Plane::World,
    availability: Availability::AtPart,
    description: "Tell the tower to do one of the things only it can do — fold to somewhere else, \
                  open or lift a siege, drill down, come back up, raise or drop the shields. What \
                  you are offered is what it can actually afford right now; if a thing is not \
                  there, it cannot currently be done.",
    params: &[
        Param {
            name: "action",
            ty: "string",
            required: true,
            description: "What the tower is to do, from what it can currently do.",
        },
        Param {
            name: "target",
            ty: "string",
            required: false,
            description: "What is being besieged, when you are opening a siege.",
        },
        Param {
            name: "x",
            ty: "string",
            required: false,
            description: "Where to fold to, east-west. Required when relocating.",
        },
        Param {
            name: "y",
            ty: "string",
            required: false,
            description: "Where to fold to, north-south. Required when relocating.",
        },
        Param {
            name: "depth",
            ty: "string",
            required: false,
            description: "How far down, in metres, when drilling.",
        },
    ],
    examples: &[
        Example {
            situation: "A horde is coming across the flats in numbers the wall will not hold, and \
                        the reserves are full.",
            call: r#"{"action":"relocate","x":-300,"y":180}"#,
            because: "Folding takes a coordinate because a destination is a number here. The \
                      option is only there at all because the energy is.",
        },
        Example {
            situation: "There is something buried under this position worth more than the \
                        position is.",
            call: r#"{"action":"drill down","depth":60}"#,
            because: "Different actions take different arguments. A depth means nothing to a \
                      fold and a coordinate means nothing to a drill.",
        },
    ],
};

/// Put a batch on one of the eight queues.
pub const PRODUCE: Tool = Tool {
    name: "produce",
    at: &["fabricator"],
    category: "Tower",
    plane: Plane::World,
    availability: Availability::AtPart,
    description: "Have something made. What you are offered is what the stockpile will actually \
                  cover — a thing you cannot pay for is not on the list rather than refused after \
                  the fact. Simple ammunition is minutes; a companion is weeks.",
    params: &[
        Param {
            name: "what",
            ty: "string",
            required: true,
            description: "What is to be made, from what can currently be paid for.",
        },
        Param {
            name: "count",
            ty: "string",
            required: false,
            description: "How many batches. One by default, and every batch costs again.",
        },
        Param {
            name: "queue",
            ty: "string",
            required: false,
            description: "Which queue to put it on, from the ones standing free.",
        },
    ],
    examples: &[Example {
        situation: "The rampart has been firing all night and the racks are down to nothing.",
        call: r#"{"what":"bolt rounds","count":4}"#,
        because: "Four batches, each paid for separately. What is not on the list is not a thing \
                  to argue about — it is a thing the stockpile will not cover.",
    }],
};

// ---------------------------------------------------------------------------
// Obligations
// ---------------------------------------------------------------------------

/// Commit to a thing by a time.
///
/// The only act that creates an obligation outliving the conversation it was
/// made in, which is what lets a mission span more than one meeting. `tell` can
/// already carry the words; what it cannot do is make the commitment survive.
pub const PROMISE: Tool = Tool {
    name: "promise",
    at: &[],
    category: "Speech",
    plane: Plane::World,
    availability: Availability::Nearby,
    description: "Commit to somebody that you will do a thing, by a time. Unlike saying you will, \
                  this is written down: they can hold you to it, and you will be reminded of it \
                  whether or not you want to be.",
    params: &[
        Param {
            name: "to",
            ty: "string",
            required: true,
            description: "Who you are promising. They must be here.",
        },
        Param {
            name: "what",
            ty: "string",
            required: true,
            description: "What you are committing to, in your own words.",
        },
        Param {
            name: "by",
            ty: "string",
            required: true,
            description: "When it is due, said the way you would say it out loud — the point at \
                          which somebody is entitled to ask you where it is.",
        },
    ],
    examples: &[Example {
        situation: "Somebody is blocked on a thing you are holding and has asked when it will be \
                    done.",
        call: r#"{"to":"Perrin Vastwood","what":"the eastern span read through and marked","by":"nightfall"}"#,
        because: "Saying \"soon\" is `tell`. This is the one that can be held against you.",
    }],
};

/// Put somebody in mind of a thing they promised you.
pub const REMIND: Tool = Tool {
    name: "remind",
    at: &[],
    category: "Speech",
    plane: Plane::Speech,
    availability: Availability::Nearby,
    description: "Put somebody in mind of a thing they promised you and have not done. You can \
                  only name something they actually promised — which is what keeps this from \
                  being an accusation about a thing that was never said.",
    params: &[
        Param {
            name: "who",
            ty: "string",
            required: true,
            description: "Who you are reminding. They must be here.",
        },
        Param {
            name: "which",
            ty: "string",
            required: true,
            description: "Which outstanding promise, from what they actually owe you.",
        },
        Param {
            name: "manner",
            ty: "string",
            required: false,
            description: "How it is meant to land. Reminding somebody without making it an \
                          accusation is most of the skill in it.",
        },
    ],
    examples: &[Example {
        situation: "The thing you were promised by nightfall has not arrived and it is well past.",
        call: r#"{"who":"Perrin Vastwood","which":"the eastern span read through and marked","manner":"lightly, as if it had only just occurred to me"}"#,
        because: "The promise is named from what stands, so this cannot become a reminder of a \
                  thing they never agreed to.",
    }],
};

/// Judge a made thing, durably.
///
/// `verdict` is a currency three clusters consume and filing has the signature
/// `accord, verdict → filed`. A judgement that is only something somebody said
/// cannot be spent by anything, so it attaches to the thing and persists.
pub const RECORD_VERDICT: Tool = Tool {
    name: "record_verdict",
    at: &[],
    category: "Work",
    plane: Plane::World,
    availability: Availability::Always,
    description: "Say whether a made thing passes, and have it stand against the thing rather \
                  than merely being said. A refusal that does not say what would change it is not \
                  a verdict, it is a mood.",
    params: &[
        Param {
            name: "on",
            ty: "string",
            required: true,
            description: "What you are judging.",
        },
        Param {
            name: "judgement",
            ty: "string",
            required: true,
            description: "What you actually think of it, in your own words.",
        },
        Param {
            name: "what_would_change_it",
            ty: "string",
            required: false,
            description: "What would have to be different. Leave it out when you are passing the \
                          thing; a refusal without it is not actionable.",
        },
    ],
    examples: &[Example {
        situation: "You have read a thing somebody is asking to file and it contradicts the span \
                    either side of it.",
        call: r#"{"on":"the third era","judgement":"it cannot go in as it stands — it disagrees with both neighbours on the same date","what_would_change_it":"the dates reconciled with the spans either side, or a line saying which is wrong"}"#,
        because: "Refusing and saying exactly what would change it are one act. Refusing without \
                  it leaves the other person nothing to do.",
    }],
};

// ---------------------------------------------------------------------------
// Reaching people who are not here
// ---------------------------------------------------------------------------

/// End a conversation you are in.
pub const SIGN_OFF: Tool = Tool {
    name: "sign_off",
    at: &[],
    category: "Messaging",
    plane: Plane::Speech,
    availability: Availability::Always,
    description: "Leave one of your conversations, saying why you are going. Standing up and \
                  walking out is how you do this in a room; on a thread there is nothing else, \
                  and staying in a conversation you should have left is its own kind of mistake. \
                  A thread you leave carries on without you unless there is nobody left on it.",
    params: &[
        Param {
            name: "to",
            ty: "string",
            required: true,
            description: "Which conversation you are leaving.",
        },
        Param {
            name: "intent",
            ty: "string",
            required: true,
            description: "What you mean by going, and why now. The narrator writes the parting.",
        },
    ],
    examples: &[Example {
        situation: "This has been going in circles for a while and the next thing you say is \
                    going to be one you cannot take back.",
        call: r#"{"to":"Soren","intent":"that I am going before I say something worse, and it is not finished"}"#,
        because: "Leaving before the damage is a decision. Being unable to leave at all is not a \
                  character trait, it is a missing act.",
    }],
};

/// Start a conversation with somebody who is not here.
pub const REACH_OUT: Tool = Tool {
    name: "reach_out",
    at: &[],
    category: "Messaging",
    plane: Plane::Speech,
    availability: Availability::Always,
    description: "Get hold of somebody you have no conversation with yet, and start one. You can \
                  do this from anywhere and while doing something else — that is what carrying a \
                  handset is for. Somebody you already have a thread with is `message`, not this.",
    params: &[
        Param {
            name: "to",
            ty: "string",
            required: true,
            description: "Who you are reaching.",
        },
        Param {
            name: "intent",
            ty: "string",
            required: true,
            description: "What you are opening with, and what you want. Substance, not the \
                          message itself.",
        },
    ],
    examples: &[Example {
        situation: "You have not properly spoken to somebody in a long time and the reason has \
                    stopped being a reason.",
        call: r#"{"to":"Wren","intent":"that I have been the one not writing, and would like to fix it rather than explain it"}"#,
        because: "Speaking first is an act somebody has to take. A character that can only ever \
                  answer is a presence, not a person.",
    }],
};

// ---------------------------------------------------------------------------
// The phone
// ---------------------------------------------------------------------------
//
// **Not the room, and not a mode.** These are the acts of a handset a character
// carries: several conversations at once, to people who are not here, at a time
// that is not now. A character texts while standing in a room and the room
// hears none of it — which is why these are separate acts rather than `say` and
// `tell` pointed somewhere else.
//
// None of them declares an availability. They bind to the threads on the phone,
// and a character without a phone has none, so the ordinary empty-set rule takes
// every one of them out of the grammar. Being out of contact is therefore a
// thing the world can *do* to somebody — take the handset — rather than a flag
// anybody has to remember to set.

/// Say something on one of your threads.
///
/// # The one act that can reach somebody you cannot see
///
/// Every world here gives its cast a standing channel — `sim::phone::CHANNEL` —
/// which every character joins on arrival without choosing to. This is the act
/// that speaks on it, and it needed no changes to become that: a channel is a
/// thread with everybody on it, so it arrives in `Choices::Threads` beside the
/// private ones and binds the same way.
///
/// That matters more than it sounds. A character alone could previously reach
/// nobody it could not already see: the percept names who is in this room, and
/// `move_to` offers rooms with no indication of who is in any of them, so the
/// building was a list of identical doors. A character that says where it is
/// has handed everybody else a room name with a person in it, which is the
/// fact that was missing — and it is why the channel is left general rather
/// than made into a find-people feature. Finding people is downstream of
/// ordinary talk, not a separate mechanism.
///
/// # Why the description insists on speaking first
///
/// A handset that reaches everybody, offered on every turn, is an invitation to
/// stop walking anywhere — and this catalogue has the scars to prove that a
/// model takes the cheapest act that looks social. Two characters standing in
/// one room texting each other is worse than either of them saying nothing.
///
/// So the rule is in the description, in the examples, and in
/// `prompt::frame`: if they are here, speak. The grammar cannot enforce it —
/// `say` and `tell` appear only in company, but `message` is `Always`, and it
/// has to be, because the whole point is reaching people who are not here.
pub const MESSAGE: Tool = Tool {
    name: "message",
    at: &[],
    category: "Messaging",
    plane: Plane::Speech,
    availability: Availability::Always,
    description: "Send something on one of your conversations. It reaches everybody on that \
                  thread and nobody else — not the room you are standing in, and not people on \
                  your other threads. They will see it whenever they next look, which may not be \
                  now. As with speaking, you give what you MEAN and the wording follows.\n\
                  \n\
                  **For people who are not here.** If the person you want is standing in front of \
                  you, `say` or `ask` — texting somebody in the same room is a worse version of \
                  talking to them. The channel reaches everyone at once, so it is for anything \
                  that concerns whoever happens to be listening: a question you need an answer \
                  to, something the others need to know before they act, a decision that is not \
                  yours alone.\n\
                  \n\
                  **Ask rather than announce.** A message that reports your own status — where \
                  you are, that you are ready, that you are standing by — gives nobody a reason \
                  to reply and nothing to do about it, and a channel full of those is a room of \
                  people talking past each other. A question obliges an answer. If you want to \
                  be with somebody, ask where they are and go there; do not announce that you \
                  are ready and wait.",
    params: &[
        Param {
            name: "to",
            ty: "string",
            required: true,
            description: "Which conversation — a person's name for one of yours with them, or a \
                          group's name.",
        },
        Param {
            name: "intent",
            ty: "string",
            required: true,
            description: "What you mean to get across. Substance, not the message itself.",
        },
    ],
    examples: &[
        Example {
            situation: "You are two rooms away from somebody who is waiting on a thing you have \
                        just finished.",
            call: r#"{"to":"Soren","intent":"that it is done and I am not walking it over tonight"}"#,
            because: "Reaching somebody who is not here is what the handset is for. Walking over \
                      is `move_to` and costs the evening.",
        },
        Example {
            situation: "Three of you are working the same span and one of them has just found \
                        something that changes what the other two are doing.",
            call: r#"{"to":"the eastern sweep","intent":"stop dating anything until we have settled the boundary"}"#,
            because: "One send, everybody on the thread. A group is a thread with more people on \
                      it, not a different kind of act.",
        },
        // **The case a solitary character had no answer to.** Nobody here, and
        // no way to learn that anybody exists anywhere — the percept names who
        // is in this room, and `move_to` offers rooms with no sign of who is in
        // any of them, so the building is a list of identical doors. The
        // question is the act that was missing.
        Example {
            situation: "You have been alone in the chronicle for an hour and you do not know \
                        where anybody else is.",
            call: r#"{"to":"the channel","intent":"where each of you is working, because I want to bring the third era to whoever is nearest"}"#,
            because: "**Asking is what finds people.** Nothing else tells you which of seventy \
                      rooms has somebody in it, and an answer names one you can walk to. \
                      Reporting your own position instead leaves everybody informed and \
                      stationary.",
        },
        // **The failure this example is here to prevent**, taken from a live
        // feed: two characters spent an afternoon agreeing to meet and never
        // met. Seven of twenty-seven messages were the words "I'm standing by",
        // and the pair oscillated between the same two rooms, each walking to
        // where it guessed the other was. Nothing they sent each other named a
        // place, because none of them ever asked.
        Example {
            situation: "Somebody has said they want to work through something with you, and you \
                        do not know which room they are in.",
            call: r#"{"to":"the channel","intent":"which room you are in, so I can come to you rather than both of us moving"}"#,
            because: "One of you has to name a place or you will cross. \"I am ready\" from both \
                      sides is two people waiting; a question has an answer, and the answer is \
                      somewhere to walk.",
        },
    ],
};

/// Bring somebody onto one of your threads.
///
/// **This is the whole of group messaging.** A thread with two people on it is a
/// direct message and one with three is a group; the difference is the
/// membership, not a mode, a second act, or a branch in the catalogue.
pub const INVITE: Tool = Tool {
    name: "invite",
    at: &[],
    category: "Messaging",
    plane: Plane::Speech,
    availability: Availability::Always,
    description: "Bring somebody onto one of your conversations. From then on they see what is \
                  said on it and can answer; what was said before they arrived is yours to \
                  repeat or not. Two people is a message and three is a group — this is how one \
                  becomes the other.",
    params: &[
        Param {
            name: "to",
            ty: "string",
            required: true,
            description: "Which conversation you are bringing them onto.",
        },
        Param {
            name: "who",
            ty: "string",
            required: true,
            description: "Who you are bringing in.",
        },
        Param {
            name: "intent",
            ty: "string",
            required: false,
            description: "Why, if it is not obvious. Everybody on the thread sees it.",
        },
    ],
    examples: &[Example {
        situation: "You and one other have been working out something that turns on a third \
                    person's part of the record, and relaying it back and forth is losing detail.",
        call: r#"{"to":"Soren","who":"Perrin Vastwood","intent":"that the dates are theirs and we have been guessing"}"#,
        because: "Bringing them on beats relaying. It also means the next thing said reaches all \
                  three without anybody having to remember to pass it along.",
    }],
};

/// Start a group outright.
pub const OPEN_GROUP: Tool = Tool {
    name: "open_group",
    at: &[],
    category: "Messaging",
    plane: Plane::Speech,
    availability: Availability::Always,
    description: "Start a named conversation with several people at once, rather than growing one \
                  from a pair. Use it when the thing concerns a set of people from the start — \
                  everybody on it sees everything said on it.",
    params: &[
        Param {
            name: "called",
            ty: "string",
            required: true,
            description: "What the group is called. Everybody on it sees this name, so it should \
                          say what the group is for.",
        },
        Param {
            name: "with",
            ty: "string",
            required: true,
            description: "Who is on it, by name. Several, separated by commas.",
        },
        Param {
            name: "intent",
            ty: "string",
            required: false,
            description: "What to open with — the thing that makes it clear why everybody has \
                          been put on one conversation. Leave it out to open the group silently.",
        },
    ],
    examples: &[Example {
        situation: "Four people are each about to date the same span differently and none of them \
                    knows the others are doing it.",
        call: r#"{"called":"the boundary","with":"Soren, Perrin Vastwood, Orion Vance","intent":"that nobody should date anything until we agree where it falls"}"#,
        because: "A standing thread is a reason for four people to stay in step that no single \
                  message between two of them can be.",
    }],
};

/// Everything in this module, in the order it is offered.
pub const WORLD_ACTS: &[Tool] = &[
    MESSAGE,
    INVITE,
    OPEN_GROUP,
    ACT,
    SLEEP,
    GIVE,
    READ,
    POST_NOTICE,
    CLAIM,
    RELEASE,
    EQUIP,
    USE,
    GATHER,
    ENGAGE,
    OPERATE,
    RECALL,
    SCAN,
    COMMAND_TOWER,
    PRODUCE,
    PROMISE,
    REMIND,
    RECORD_VERDICT,
    SIGN_OFF,
    REACH_OUT,
];
