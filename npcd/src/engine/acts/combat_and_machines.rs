//! Defines the acts that fight, operate machines, recall a body, change attention and scan, with the argument shapes each one takes.
//!
//! `engage` is the one combat act, and its arguments are enumerated values or bound entities because a simulator consumes them and cannot read prose. Its arguments are capped at two, since the number of decodes a turn admits grows with each enumerated argument, and the remaining constraints are expressed through the other two or through speech to the squad. `operate` puts a machine into one of its states, and a turret's targeting policy is one of those states. `recall` is offered only to a body that is away from its muster point, and `focus` only to a body that moves by attention. The lift has two acts: calling it when it is elsewhere, and riding it from the shaft. `scan` reaches somewhere the body is not, and naming no place surveys the room.

use super::*;

/// **The one combat tool.**
///
/// The simulator fights; this says how. Every argument is an enumerated value or
/// a bound entity, because the consumer is formula code and a decode is far too
/// slow to aim a weapon. A character's part is to set a stance and change its
/// mind about it, which is the same division of labour `tell` keeps with the
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
        // multiplies that count. Two is what a turn affords.
        //
        // Little is lost. Every constraint worth honouring was already sayable
        // as one of the other two — `cover` *is* staying in cover, `hold` is
        // holding ground, `conserve ammunition` is what `hold fire` and
        // `suppress` differ over — and the one genuinely new axis, whether to
        // work around the player, is what `priority` carries. Anything that
        // still will not fit belongs in a `shout` to the squad, where language
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
    description: "Put something here into a different state. Each thing has its own states and \
                  takes no others.",
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

/// Turn your attention to a room: how a body that is the building moves through
/// it (`docs/zen_moon_mind.md` §3.2).
///
/// **Offered only to a body that moves by attention**, through its one argument:
/// `on` binds to [`Choices::Rooms`](crate::engine::tools::Choices::Rooms), which
/// is empty for everybody who walks, so the act is absent for them by the
/// ordinary empty-set rule rather than refused.
pub const FOCUS: Tool = Tool {
    name: "focus",
    at: &[],
    category: "Movement",
    plane: Plane::World,
    availability: Availability::Embodied,
    description: "Turn your attention to another room. You are there at once — no corridor, no \
                  lift — and what is in it, and whoever is in it, is in front of you; those you \
                  leave see your attention go.",
    params: &[Param {
        name: "on",
        ty: "string",
        required: true,
        description: "The room to attend to.",
    }],
    examples: &[Example {
        situation: "A report says the blast door on the ward is open when it should be locked.",
        call: r#"{"on":"the ward room"}"#,
        because: "You do not walk your own corridors: you look where it matters, and then you can \
                  act there.",
    }],
};

/// Call the lift to the floor you are standing on.
///
/// **The lift is a shared, slow thing, and that is the point of it.** There is
/// one car; getting between levels is calling it, waiting for it, getting in and
/// riding — which is where two bodies crossing the building on their own errands
/// are made to be in the same place at the same time. Only offered while you are
/// at the lift and it is elsewhere; when it is already here there is nothing to
/// call and [`LIFT_USE`] is what you are offered instead. See [`crate::engine::lift`].
pub const LIFT_CALL: Tool = Tool {
    name: "lift_call",
    at: &[],
    category: "Movement",
    plane: Plane::World,
    availability: Availability::AtLift,
    description: "Bring the lift to the floor you are standing on, to meet somebody off it. To go \
                  somewhere in it, `lift_use` naming the level instead — that calls it and waits \
                  for you.",
    params: &[],
    examples: &[Example {
        situation: "You are at the lift, and the car is on another level.",
        call: r#"{}"#,
        because: "The car has to come to you before you can ride it. Standing at the shaft is not \
                  being in the car.",
    }],
};

/// Ride the lift to another level.
///
/// Offered while you are at the shaft, wherever the car is: open here, you get
/// in and go; elsewhere, it is called and you wait on the landing and ride it
/// when it comes. The floor is named from the levels the lift serves, never the
/// one you are on.
pub const LIFT_USE: Tool = Tool {
    name: "lift_use",
    at: &[],
    category: "Movement",
    plane: Plane::World,
    availability: Availability::AtLanding,
    description:
        "Ride the lift to another level. Name the level and it carries you there, setting you \
                  down when it arrives — if the car is on another floor it is called, and you wait \
                  here and ride it when it comes.",
    params: &[Param {
        name: "floor",
        ty: "string",
        required: true,
        description: "The level to ride to, by its name.",
    }],
    examples: &[Example {
        situation: "You are in the lift with its doors open, and what you were sent for is on the \
                    chronicle level.",
        call: r#"{"floor":"the chronicle level"}"#,
        because: "The lift is how you change levels. Name where you are going and ride.",
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
    description: "Look through your instruments. Name a place you are not in to see who is there \
                  and what moves. Name nothing to see what in your own room you can work, with \
                  its state and the address to `invoke` for each. It shows rooms and who is in \
                  them, nothing else: it never reaches a terminal, the house network, the \
                  cluster or a machine — connected to one of those, use the acts it gave you.",
    // **`at` is optional because naming nothing is a real act: the survey.**
    //
    // The grammar offers a place from the live set of places there are, or no
    // place at all, which surveys the room's stations (`engine::survey`). The
    // coordinate form stays reachable through the API and the harness, which are
    // not grammar-constrained — the same split `operate` makes for a mode a
    // device does not admit.
    params: &[Param {
        name: "at",
        ty: "string",
        required: false,
        description: "The place you want looked at, named exactly as it is written. Leave it out \
                      to survey what you can work where you stand.",
    }],
    examples: &[
        Example {
            situation: "You stand in a room with a console and are not sure what it will take.",
            call: r#"{}"#,
            because: "Naming nothing lists what here answers to you and the exact address to \
                      `invoke` for each — read it, then act on it.",
        },
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
