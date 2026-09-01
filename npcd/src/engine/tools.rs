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

const SPEAK: Tool = Tool {
    name: "speak",
    category: "Speech",
    plane: Plane::Speech,
    availability: Availability::Always,
    description: "Say something. You give what you MEAN — the substance, the stance, who it is \
                  for. You do not write the sentence; the narrator renders your intent into your \
                  own voice, using how you speak and what you are feeling.",
    params: &[
        Param {
            name: "intent",
            ty: "string",
            required: true,
            description: "What you mean to convey. Substance, not wording: \"that I will not \
                          hand over the ledger, and that pressing me will cost him\" — never a \
                          finished line of dialogue.",
        },
        Param {
            name: "to",
            ty: "string",
            required: false,
            description: "Who you are addressing, by the name you know them by. Omit to speak \
                          to whoever is present.",
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
            situation: "Hess demands the ledger. You believe he burned the east granary, and \
                        you owe his sister your life.",
            call: concat!(
                r#"{"intent":"that he will not get the ledger from me, and that I have not "#,
                r#"forgotten what I owe his sister","to":"Hess","manner":"level, without heat"}"#
            ),
            because: "Two gathered things are in tension — a belief about Hess and a debt to \
                      his family — and the intent carries both rather than resolving one away. \
                      A finished sentence here would pick a tone the character has not decided on.",
        },
        Example {
            situation: "A stranger asks your name in a crowded market. You have no reason to \
                        lie and no reason to linger.",
            call: r#"{"intent":"my name, and that I am in the middle of something","manner":"brief"}"#,
            because: "No `to` — the stranger has no name you know yet. Brevity is an intent, \
                      not a word count.",
        },
    ],
};

const MOVE_TO: Tool = Tool {
    name: "move_to",
    category: "Movement",
    plane: Plane::World,
    availability: Availability::Always,
    description: "Go somewhere, described the way you would think of it rather than as \
                  coordinates. Commits against the world you reasoned over; if the world has \
                  moved under you, the act is refused and you will perceive why.",
    params: &[
        Param {
            name: "destination",
            ty: "string",
            required: true,
            description: "Where you mean to be: \"the far side of the gate\", \"behind the \
                          cart\", \"back to Hess\".",
        },
        Param {
            name: "urgency",
            ty: "string",
            required: false,
            description: "walk | hurry | run. Absent means whatever the moment warrants.",
        },
    ],
    examples: &[Example {
        situation: "The tactical map shows crossbowmen on the wall to your north. You are in \
                    the open.",
        call: r#"{"destination":"behind the overturned cart, out of the wall's line","urgency":"run"}"#,
        because: "The destination is reasoned from the map that was gathered, and says WHY that \
                  spot — cover from a specific threat — so a world that has moved the cart can \
                  refuse it intelligibly.",
    }],
};

const FACE: Tool = Tool {
    name: "face",
    category: "Movement",
    plane: Plane::World,
    availability: Availability::Always,
    description:
        "Turn your attention and your body toward something without leaving where you are.",
    params: &[Param {
        name: "target",
        ty: "string",
        required: true,
        description: "What you turn toward.",
    }],
    examples: &[Example {
        situation: "You hear a bowstring draw somewhere behind and to your left.",
        call: r#"{"target":"the sound behind me, to the left"}"#,
        because: "You cannot name what you have not seen. Facing a *sound* is honest; facing \
                  \"the archer\" would be acting on knowledge you do not have.",
    }],
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

const FLEE: Tool = Tool {
    name: "flee",
    category: "Movement",
    plane: Plane::World,
    availability: Availability::Always,
    description: "Break away from danger. Distinct from move_to: you are leaving something, and \
                  where you end up matters less than not being here.",
    params: &[Param {
        name: "from",
        ty: "string",
        required: true,
        description: "What you are getting away from.",
    }],
    examples: &[Example {
        situation: "The roof beam above you cracks and begins to give. You are carrying a child.",
        call: r#"{"from":"under the beam, toward the doorway"}"#,
        because: "Fleeing names the threat, not a destination — the character has no time to \
                  reason about where, only about away-from-what.",
    }],
};

const GESTURE: Tool = Tool {
    name: "gesture",
    category: "Gesture",
    plane: Plane::World,
    availability: Availability::Always,
    description: "Convey something with your body instead of your voice — a signal, a warning, \
                  a refusal. Like speak, you give the meaning and not the movement.",
    params: &[
        Param {
            name: "intent",
            ty: "string",
            required: true,
            description: "What the gesture is meant to convey.",
        },
        Param {
            name: "to",
            ty: "string",
            required: false,
            description: "Who it is for.",
        },
    ],
    examples: &[Example {
        situation: "Your companion is about to speak. The guard is within earshot and you have \
                    just recognised his colours.",
        call: r#"{"intent":"stop talking, now, and do not look at the guard","to":"my companion"}"#,
        because: "Speaking would defeat the purpose. The tool is chosen for the constraint the \
                  situation imposes, which is exactly what calibration has to teach.",
    }],
};

const EXPRESS: Tool = Tool {
    name: "express",
    category: "Gesture",
    plane: Plane::World,
    availability: Availability::Always,
    description: "Let something show on your face or in your bearing. Involuntary or nearly so — \
                  what you are feeling becomes visible whether or not you chose it.",
    params: &[Param {
        name: "feeling",
        ty: "string",
        required: true,
        description: "What shows.",
    }],
    examples: &[Example {
        situation: "Hess mentions the granary fire in passing, as if it were nothing. You have \
                    believed for two years that he set it.",
        call: r#"{"feeling":"a flicker of something held down, gone almost before it shows"}"#,
        because: "The belief cannot be acted on and cannot be revised — but it can leak. This \
                  is the tool for the pressure a belief exerts when nothing else is available.",
    }],
};

const OBSERVE: Tool = Tool {
    name: "observe",
    category: "Attention",
    plane: Plane::Internal,
    availability: Availability::Always,
    description: "Look at something properly. Spends your thinking step on perceiving rather \
                  than acting, and what you find arrives as a new event.",
    params: &[Param {
        name: "target",
        ty: "string",
        required: true,
        description: "What you look at.",
    }],
    examples: &[Example {
        situation: "The map shows a shape at the tree line that the legend does not account for.",
        call: r#"{"target":"the unaccounted shape at the tree line"}"#,
        because: "Acting on an ambiguity is worse than spending a step resolving it. Choosing to \
                  look is a real decision, not a null one.",
    }],
};

const LISTEN: Tool = Tool {
    name: "listen",
    category: "Attention",
    plane: Plane::Internal,
    availability: Availability::Always,
    description: "Attend to sound — a conversation you are not part of, something you half heard.",
    params: &[Param {
        name: "target",
        ty: "string",
        required: false,
        description: "What you listen to. Omit to listen to everything.",
    }],
    examples: &[Example {
        situation: "Two of the garrison are talking quietly by the well and stop when you pass.",
        call: r#"{"target":"the two by the well, once I am past"}"#,
        because: "The target carries the timing that makes it possible. Listening now would be \
                  visibly listening.",
    }],
};

const INSPECT: Tool = Tool {
    name: "inspect",
    category: "Attention",
    plane: Plane::Internal,
    availability: Availability::Always,
    description: "Examine a specific thing closely — handle it, read it, take it apart.",
    params: &[Param {
        name: "target",
        ty: "string",
        required: true,
        description: "What you examine.",
    }],
    examples: &[Example {
        situation: "You have the ledger. The last three entries are in a different hand.",
        call: r#"{"target":"the last three entries, and whose hand they are in"}"#,
        because: "Inspect is narrower than observe: a specific question of a specific object, \
                  which is what makes the answer worth a step.",
    }],
};

const GREET: Tool = Tool {
    name: "greet",
    category: "Social",
    plane: Plane::Speech,
    availability: Availability::Always,
    description: "Acknowledge someone. Distinct from speak because how you greet a person \
                  encodes your standing with them, and that comes from your relationship rather \
                  than from anything you decide now.",
    params: &[Param {
        name: "who",
        ty: "string",
        required: true,
        description: "Who you acknowledge.",
    }],
    examples: &[Example {
        situation: "You pass Hess's sister in the corridor. You owe her your life and have been \
                    avoiding her for a month.",
        call: r#"{"who":"Hess's sister"}"#,
        because: "No manner and no intent — the relationship supplies both. Adding them here \
                  would override the calibration trajectory with a guess.",
    }],
};

const OFFER: Tool = Tool {
    name: "offer",
    category: "Social",
    plane: Plane::Speech,
    availability: Availability::Always,
    description: "Put something forward — help, a trade, a way out. What you offer is substance; \
                  the wording is the narrator's.",
    params: &[
        Param {
            name: "what",
            ty: "string",
            required: true,
            description: "What you put forward.",
        },
        Param {
            name: "to",
            ty: "string",
            required: false,
            description: "Who it is for.",
        },
    ],
    examples: &[Example {
        situation: "The courier is cornered and frightened. You want what she is carrying and \
                    you do not want to take it by force.",
        call: r#"{"what":"safe passage out of the district, in exchange for what she carries","to":"the courier"}"#,
        because: "The offer names both sides. An offer that names only what you want is a demand, \
                  and `threaten` is the tool for that.",
    }],
};

const REFUSE: Tool = Tool {
    name: "refuse",
    category: "Social",
    plane: Plane::Speech,
    availability: Availability::Always,
    description: "Decline. Carries whether the refusal is final and whether you are willing to \
                  say why.",
    params: &[
        Param {
            name: "what",
            ty: "string",
            required: true,
            description: "What you decline.",
        },
        Param {
            name: "final",
            ty: "boolean",
            required: false,
            description: "Whether there is anything left to discuss.",
        },
    ],
    examples: &[Example {
        situation: "Hess asks a third time, and has begun implying what refusing will cost.",
        call: r#"{"what":"handing over the ledger","final":true}"#,
        because: "The third asking is what makes it final. A first refusal with `final: true` \
                  would close a conversation the character has not decided to close.",
    }],
};

const THREATEN: Tool = Tool {
    name: "threaten",
    category: "Social",
    plane: Plane::Speech,
    availability: Availability::Always,
    description: "Make a cost explicit. You supply the cost you mean to imply, not the words.",
    params: &[
        Param {
            name: "cost",
            ty: "string",
            required: true,
            description: "What the other party stands to lose.",
        },
        Param {
            name: "to",
            ty: "string",
            required: false,
            description: "Who you are addressing.",
        },
    ],
    examples: &[Example {
        situation: "Hess has implied a cost of his own. You know something about the granary he \
                    does not know you know.",
        call: r#"{"cost":"that what I know about the granary does not stay with me","to":"Hess"}"#,
        because: "The threat is drawn from a gathered belief, and stops short of asserting it as \
                  fact. The character cannot revise the belief — but it can spend it.",
    }],
};

const NOTE_CONCERN: Tool = Tool {
    name: "note_concern",
    category: "Internal",
    plane: Plane::Internal,
    availability: Availability::Always,
    description: "Mark something as bothering you, so it stays available to think with. This \
                  does NOT change what you believe — it records that something is pressing on a \
                  belief. Beliefs move only slowly, and never because you decided they should.",
    params: &[Param {
        name: "concern",
        ty: "string",
        required: true,
        description: "What is bothering you, and what it presses against.",
    }],
    examples: &[Example {
        situation: "Everything you have seen today contradicts what you know about Hess. You \
                    cannot make it fit.",
        call: r#"{"concern":"nothing I saw today fits what I know about Hess, and I cannot make it fit"}"#,
        because: "This is the ONLY thing available when a belief is under pressure. The character \
                  notices the tension and cannot resolve it — resolution is evidence over time, \
                  not a decision. An agent reaching for a belief-write here has misunderstood \
                  the architecture.",
    }],
};

const SET_INTENT: Tool = Tool {
    name: "set_intent",
    category: "Internal",
    plane: Plane::Internal,
    availability: Availability::Always,
    description: "Commit to something that outlasts this moment — what you are trying to achieve \
                  across the next stretch of time. Persists across thinking steps and shapes \
                  what you gather.",
    params: &[
        Param {
            name: "intent",
            ty: "string",
            required: true,
            description: "What you are set on.",
        },
        Param {
            name: "until",
            ty: "string",
            required: false,
            description: "What would end it — achieved, or abandoned.",
        },
    ],
    examples: &[Example {
        situation: "You have the ledger and Hess knows it. Staying in the district is no longer \
                    survivable.",
        call: r#"{"intent":"get the ledger out of the district tonight","until":"it is out, or I am taken"}"#,
        because: "An intent is what makes a character coherent across ticks rather than reactive \
                  within them. The `until` is what lets it end — an intent with no end condition \
                  never releases the character.",
    }],
};

const BROADCAST_STRATEGY: Tool = Tool {
    name: "broadcast_strategy",
    category: "Internal",
    plane: Plane::World,
    availability: Availability::Always,
    description: "Make your intent legible to allies so they can act with you rather than \
                  around you.",
    params: &[Param {
        name: "strategy",
        ty: "string",
        required: true,
        description: "What you want your side to understand about your plan.",
    }],
    examples: &[Example {
        situation: "You are about to draw the guards toward the north gate. Two allies are in \
                    the yard and do not know it.",
        call: r#"{"strategy":"I am pulling the guards north; the yard will be clear for about a minute"}"#,
        because: "Broadcasting the consequence, not the action — allies need what it means for \
                  them, which is the part they can act on.",
    }],
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

const WAIT: Tool = Tool {
    name: "wait",
    category: "Meta",
    plane: Plane::Meta,
    availability: Availability::Always,
    description: "Do nothing, on purpose, and say how long you are willing to. A real choice: \
                  it decides how soon you think again.",
    params: &[Param {
        name: "until",
        ty: "string",
        required: false,
        description: "What you are waiting for.",
    }],
    examples: &[Example {
        situation: "The guard has not moved. Nothing has changed in some minutes. You are not in \
                    danger and you are not finished here.",
        call: r#"{"until":"the guard moves, or something else does"}"#,
        because: "Waiting is what an idle character SHOULD do — the alternative is inventing \
                  activity to fill a step, which is how an NPC starts pacing for no reason.",
    }],
};

const END_INTERACTION: Tool = Tool {
    name: "end_interaction",
    category: "Meta",
    plane: Plane::Meta,
    availability: Availability::Always,
    description: "This exchange is over as far as you are concerned. You stop attending to it.",
    params: &[Param {
        name: "because",
        ty: "string",
        required: false,
        description: "Why you are done.",
    }],
    examples: &[Example {
        situation: "You refused, finally, and Hess has said the same thing twice more since.",
        call: r#"{"because":"there is nothing further to say and staying invites him to keep asking"}"#,
        because: "Ending is an act with a reason, not a timeout. A character that never ends an \
                  interaction is one that can be held in conversation indefinitely by anyone.",
    }],
};

/// The generic catalog: what every character can do, before any world adds to it.
pub const CATALOG: &[Tool] = &[
    SPEAK,
    MOVE_TO,
    FACE,
    FOLLOW,
    FLEE,
    GESTURE,
    EXPRESS,
    OBSERVE,
    LISTEN,
    INSPECT,
    GREET,
    OFFER,
    REFUSE,
    THREATEN,
    NOTE_CONCERN,
    SET_INTENT,
    BROADCAST_STRATEGY,
    SEND_IMAGE,
    WAIT,
    END_INTERACTION,
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

/// The tools offered in a mode.
pub fn for_mode(mode: Mode) -> Vec<&'static Tool> {
    CATALOG
        .iter()
        .filter(|t| match t.availability {
            Availability::Always => true,
            Availability::MessagingOnly => mode == Mode::Messaging,
        })
        .collect()
}

pub fn by_name(name: &str) -> Option<&'static Tool> {
    CATALOG.iter().find(|t| t.name == name)
}

#[cfg(test)]
mod tests {
    use super::*;

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
                .any(|p| matches!(p.name, "intent" | "what" | "cost" | "who"));
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
        assert!(by_name("note_concern").is_some());
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
    /// forces everything through `speak` and the world never changes.
    #[test]
    fn the_catalog_covers_every_category_the_design_names() {
        for c in [
            "Speech",
            "Movement",
            "Gesture",
            "Attention",
            "Social",
            "Internal",
            "Messaging",
            "Meta",
        ] {
            assert!(
                CATALOG.iter().any(|t| t.category == c),
                "no tool in category {c}"
            );
        }
    }
}
