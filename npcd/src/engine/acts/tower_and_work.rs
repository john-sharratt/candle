//! Defines the acts that spend the tower's capabilities, queue production, make promises and record verdicts.
//!
//! The tower is one tool with a bound action list, and the list is what the tower can afford at that moment, so an option it cannot pay for is not offered and the economy is enforced by the grammar. Only a body at the tower can address it. `promise` is the only act that creates an obligation outliving the conversation it was made in, which is what lets a mission span more than one meeting, and it is announced aloud to a person who hears it. A verdict on a made thing is recorded durably so the other clusters that consume verdicts can spend it. Ending a conversation is the last act in this group.

use super::*;

// The tower

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
            description: "What is being besieged. Required when opening a siege.",
        },
        Param {
            name: "x",
            ty: "integer",
            required: false,
            description: "Where to fold to, east-west. Required when relocating.",
        },
        Param {
            name: "y",
            ty: "integer",
            required: false,
            description: "Where to fold to, north-south. Required when relocating.",
        },
        Param {
            name: "depth",
            ty: "integer",
            required: false,
            description: "How far down, in metres. Required when drilling.",
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

// Obligations

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
    // A person here to hear it, or the party of a conversation: room speech is
    // for people, and a promise is announced aloud.
    availability: Availability::PersonNearby,
    description: "Commit to somebody that you will do a thing, by a time. Unlike saying you will, \
                  this is written down: they can hold you to it, and you will be reminded of it \
                  whether or not you want to be.",
    params: &[
        Param {
            name: "to",
            ty: "string",
            required: true,
            description: "Who you are promising: somebody here, or who you are talking with.",
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
        situation: "Perrin Vastwood, on the phone to you, is blocked on a thing you are holding \
                    and has asked when it will be done.",
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
    availability: Availability::PersonNearby,
    description: "Put somebody in mind of a thing they promised you and have not done. You can \
                  only name something they actually promised — which is what keeps this from \
                  being an accusation about a thing that was never said.",
    params: &[
        Param {
            name: "who",
            ty: "string",
            required: true,
            description: "Who you are reminding: somebody here, or who you are talking with.",
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

// Reaching people who are not here

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
