//! Defines the vocabulary the act catalogue is built from: what an act changes,
//! where it is offered, its parameters, its examples and the modes it runs in.
//!
//! Each act declares a plane saying what it changes and an availability saying
//! where it is offered. Availability is decided per turn from the world, so a
//! character is never invited to address somebody who is not there. A mode says
//! whether the parties are face to face or apart, kept separate from whether the
//! channel can carry a picture, because collapsing the two once gave a telephone
//! call a way to send a photograph. Each act carries worked examples, which are
//! what calibrates the choice between neighbouring acts.

use super::*;

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
    /// Only while a **person** is in the room (`docs/npc_conversations.md` §12).
    ///
    /// **Room speech is for people.** Offering `tell` to a character whose only
    /// company is other characters is what makes it talk to them aloud instead
    /// of messaging them; characters reach each other by `message`, here or
    /// anywhere. A person is a `visitor:{handle}` body — see
    /// [`Within::people`].
    PersonNearby,
    /// Only while a person is in the room and somebody else besides.
    ///
    /// For an act that keeps something from somebody who is standing there —
    /// which needs one person to tell and another to keep it from. Offered with
    /// only one other person present, a whisper keeps nothing from anyone, and
    /// that is where the cast used it: every remark in a two-person exchange,
    /// narrated at the other one under its breath.
    PersonAmongOthers,
    /// Only while a person's body is where a shout reaches, or the body is in
    /// field play — see [`Within::heard_by_person`] and [`Within::in_field`].
    PersonInEarshot,
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
    /// Only while holding something — an order taken, or a station claimed.
    ///
    /// **Giving back what you do not have is not a thing that can happen.**
    /// `release` takes no argument, so the empty-set rule has nothing to empty
    /// and cannot reach it; the condition is about the body's hands rather than
    /// about anything it could name, which is what this ladder is for. Absent
    /// rather than refused, so a body with nothing to give back is never invited
    /// to try.
    Holding,
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
    /// Only while standing on a lift landing with the car somewhere else — the
    /// one moment calling it does anything. With the car already open here there
    /// is nothing to call, so the act is absent. See [`crate::engine::lift`].
    AtLift,
    /// While standing on a landing, wherever the car is: with it open here you
    /// get in and ride, with it elsewhere it is called and you wait to ride it.
    /// Absent once you have asked to ride, until you are set down.
    AtLanding,
    /// Only while the body carries a mission that grants this act, in that
    /// mission's room, and only where the world is a mirror of a real place —
    /// the acts by which the world builds itself (`docs/zen_moon_mind.md` §14).
    /// See [`Within::grants`].
    Granted,
    /// Only for a character whose personality reaches it (`reaches:`,
    /// `docs/zen_moon_mind.md` §16) — wherever it stands, and ranked by
    /// provenance like any act, since reaching the weather is not the point of
    /// standing anywhere. See [`Within::reaches`].
    Reached,
    /// Only in a conversation's decision (`docs/npc_conversations.md` §6.2) —
    /// a turn whose `Within` carries the thread's allow-list
    /// ([`Within::allow`]) — never in the main loop.
    OnThread,
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
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize)]
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

/// What a turn's act grammar is compiled for: the mode its acts are offered
/// under, and how many calls its loop admits. Part of the grammar cache's key,
/// so a conversation's decision and the main loop's turn in the same room
/// never share a tree.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Catalog {
    pub mode: Mode,
    pub calls: usize,
}

impl Catalog {
    /// The main loop: a body in its room.
    pub const ROOM: Catalog = Catalog {
        mode: Mode::Physical,
        calls: ACTS_PER_TURN,
    };
    /// A conversation's decision: a thread, to somebody who may be nowhere
    /// near (`docs/npc_conversations.md` §6.2).
    pub const THREAD: Catalog = Catalog {
        mode: Mode::InstantMessage,
        calls: TALK_ACTS,
    };
}
