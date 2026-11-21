//! The acts that change the world's state: what a body carries, works, fights, operates and asks of its tower.
//!
//! The speech, attention and movement acts are in `tools`, and these are the acts that act on things. The catalogue follows two rules. A free-text argument is read only by a model, so any value a formula consumes is an enumerated value or a bound entity. The grammar compiler can bound a string or a boolean but cannot bound an integer or nested JSON, so counts and coordinates are strings that the receiving act parses. Each tool carries worked examples, since calibration depends on them to teach the distinctions between neighbouring acts. `act` is a physical act against another person, and the description rules out its near misses by name, so the model does not reach for it to speak or travel.
//!
//! - `carrying`: acts that carry, work, read, gather and sleep.
//! - `combat_and_machines`: acts that fight, operate machines, recall, move by attention, use the lift and scan.
//! - `conversation_acts`: acts that speak on the phone, on the shared channel and on replies.
//! - `tower_and_work`: acts that use the tower, queue production, make promises and record verdicts.

use super::tools::{Availability, Example, Param, Plane, Tool};

// The body, continued

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
///
/// # It is physical, and the description has to fight for that
///
/// A generic verb over a specific one is the trade `act` makes, and its cost is
/// the opposite of `touch`'s: where `touch` was too gentle to reach for in a
/// fight, `act` is broad enough that a model reaches for it to *speak* — and to
/// *travel*, stating a plan to be somewhere else on its own body ("act on
/// yourself: return to the command level"), which moves it nowhere and leaves it
/// believing it has gone. It was caught calling `act` where `tell` was meant —
/// and the description invited it, by saying "Like `tell`, you give what you
/// MEAN by it", which likens the two acts at the exact point they must not be
/// confused. So the description no longer compares it to speech; it rules the
/// near misses out by name (`tell`, `ask`, `whisper`, `shout` for words;
/// `gesture` for a signal; `reflect` for a thought; `move_to` for going
/// anywhere) and states, more than once, that this is only ever a body on a
/// body, and that acting on your own body changes nothing about where you are.
/// The name is read first, but the description is what stops the near miss.
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
    description: "Make PHYSICAL contact with one person here — something your body does to \
                  theirs, and nothing you say. Hands-on only: steady them, block their way, \
                  take something out of their hand, put yourself between them and something \
                  else — or put them on the floor, break their grip, hurt them. **This is not \
                  how you say anything, and not how you go anywhere.** To tell, ask, warn, \
                  promise or order a person here, use `tell`, `ask`, `whisper` or `shout`, and \
                  `message` another character; to signal \
                  without touching, `gesture`; to stop and take stock, `reflect`; **to set off \
                  somewhere — a room, another level, back the way you came — use `move_to`, \
                  or `focus` if you move by attention, never this.** This is only ever the act \
                  of a body on a body. You give what you MEAN by the contact, not the \
                  choreography; the narrator renders the movement. They feel it, they may refuse it, and what they do next is theirs. \
                  On `yourself` it is still physical — binding a wound, getting your own weapon \
                  clear, dragging yourself up — never a feeling, a thought, or a plan to be \
                  elsewhere. Acting on your own body never changes where you are or what anyone \
                  else heard.",
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
            description: "What you mean by it. Substance, not movement or choreography, and \
                          never a destination or a plan: \"steady her before she goes over\" or \
                          \"put him down before he reaches the door\" — never \"I put my hand \
                          under her elbow\", and never \"return to the command level\" (going \
                          somewhere is `move_to`).",
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

/// Everything in this module, in the order it is offered.
pub const WORLD_ACTS: &[Tool] = &[
    REPLY,
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
    FOCUS,
    LIFT_CALL,
    LIFT_USE,
    SCAN,
    COMMAND_TOWER,
    PRODUCE,
    PROMISE,
    REMIND,
    RECORD_VERDICT,
    SIGN_OFF,
    REACH_OUT,
];

mod carrying;
mod combat_and_machines;
mod conversation_acts;
mod tower_and_work;

pub use carrying::*;
pub use combat_and_machines::*;
pub use conversation_acts::*;
pub use tower_and_work::*;
