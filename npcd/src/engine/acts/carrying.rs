//! Defines the acts by which a body carries, works, reads and gathers things, each with the grammar it is offered under.
//!
//! These acts operate on objects a body holds or stands beside, and each is a single named tool so the model chooses among names it can see rather than among many similar ones. `read` is one act for everything readable, because which things are readable is a fact about where the body is standing, and it returns what is actually recorded rather than what was perceived. `claim` replaces six earlier take-acts, since taking a station and taking an order are the same operation to a character. `sleep` is deliberately unresponsive, because a day has meaning only if something can break through it, while `pause` stays responsive to anything. Each act is offered only when its object is present, so an absent object removes the act from the grammar rather than producing a refusal.

use super::*;

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

// Working the world

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
                  terminal's subject — or a conversation on your phone, by the name you call it, \
                  or `messages` for everything unread there. What comes back is its contents, not \
                  a description of it. You are already told what is in the room and who is in \
                  it; this is for what is written down, which you cannot know until you read it.",
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
/// # Why it is not `tell` written down
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
                  the people currently in the room — speak to the people here, message the \
                  others; both are gone once said.",
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
            call: r#"{"on":"the muster board","what":"that the lift on five is not answering and nobody can get up or down until somebody looks at it"}"#,
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
    availability: Availability::Holding,
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
