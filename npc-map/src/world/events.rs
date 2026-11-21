//! Defines what the world records as having happened, and who can perceive each
//! kind of happening.
//!
//! Most happenings are somebody's doing and are seen by the people in the room.
//! Journey outcomes are private to the walker, and a collision is shared by the
//! two bodies in it. Non-verbal acts are a kind of their own, so a character
//! with nothing to say still has something real to do in a room. A happening
//! with no actor carries a sentence that names its own subject, because the
//! building itself did it. Voice records how an utterance was pitched, which
//! decides who beyond the speaker's room can make it out.

use super::*;

/// Something that happened somewhere.
///
/// Three of these — [`SetOut`](Happening::SetOut), [`GotThere`](Happening::GotThere)
/// and [`LostTheWay`](Happening::LostTheWay) — happen *to* one body and are
/// seen by nobody else. Intent is not visible and neither is arriving; what a
/// bystander sees is a person leaving a room and a person coming into one,
/// which is [`Left`](Happening::Left) and [`Arrived`](Happening::Arrived).
/// [`crate::witness`] enforces that; the record keeps all of it.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum Happening {
    Arrived,
    Left,
    /// Began a journey. Private to the walker.
    SetOut {
        toward: Where,
    },
    /// The journey finished. Private to the walker, and the answer to a
    /// question it asked several ticks ago.
    GotThere {
        toward: Where,
    },
    /// The journey ended without arriving. Private to the walker.
    LostTheWay {
        toward: Where,
        why: Lost,
    },
    /// Two bodies travelling crossed paths and stopped — a collision at a shared
    /// doorway or lift that ends both journeys where they met. `into` is the id
    /// of the other body. Unlike a journey's other outcomes it is **not**
    /// private: the one bumped into is standing right there and knows it
    /// happened, and a bystander sees two people run into each other. It is an
    /// outcome all the same — the walker is told, because a collision it did not
    /// plan is news — which is why it rides the `is_outcome` gate without the
    /// `is_private` one.
    Bumped {
        into: String,
    },
    /// A station lit up. The subject is carried but is only *shown* to
    /// somebody in the same room.
    TookStation {
        subject: Option<String>,
    },
    LeftStation {
        subject: Option<String>,
    },
    /// Somebody spoke. `to` is who they aimed it at, and is a fact about the
    /// utterance rather than about who received it — at an ordinary pitch
    /// everybody in the room hears it either way. `None` is spoken to the room
    /// at large. `voice` is how it was pitched, which is what decides who else
    /// makes it out — see [`Voice`].
    Said {
        to: Option<String>,
        words: String,
        voice: Voice,
    },
    /// Somebody did something without speaking — a look held, a hand raised, a
    /// chair pushed back.
    ///
    /// **The room's non-verbal channel, and it is not decoration.** Without it
    /// every act a character can take that anybody else perceives is speech, so
    /// a character with nothing to say has nothing to *do* that registers, and
    /// the acts it reaches for instead — turning to face a thing, looking at a
    /// wall — change nothing and are perceived by nobody. Two characters then
    /// share a room while being, to each other, motionless.
    ///
    /// `to` aims it the same way [`Happening::Said`] does: at somebody, in
    /// front of everybody, who all see who it was meant for.
    Did {
        to: Option<String>,
        what: String,
    },
    /// The building did something. Nobody did it.
    ///
    /// **The one happening with no actor**, and that is the whole of why it
    /// exists. Everything else here is somebody's doing, so a room where nobody
    /// is doing anything is a room where nothing is true — which is exactly the
    /// state a character standing alone in a corridor is in, and it is why one
    /// of them stood there gesturing at nothing every four seconds. A vent
    /// cycling, a light failing, a rat crossing the floor: those happen whether
    /// or not anybody is thinking, and a character perceiving one has something
    /// real to react to.
    ///
    /// `text` is a whole sentence naming its own subject — *"The lights in the
    /// ceiling flicker and steady."* — because there is no actor to put in
    /// front of it. `npcd`'s `engine::stir` produces them and holds that rule;
    /// `crate::witness::narrate` relies on it, standing the sentence on its own
    /// rather than trying to attribute it.
    ///
    /// `weight` rides on the event rather than being derived from it. Every
    /// other happening's weight follows from what *kind* of thing it is, but a
    /// fan changing note and a breaker going are the same kind of thing and are
    /// worth entirely different amounts, so the thing that knows says so.
    Stirred {
        text: String,
        weight: Weight,
    },
}

impl Happening {
    /// Whether this is the world answering something the actor set in motion,
    /// rather than something the actor did.
    ///
    /// The distinction only exists because movement takes time. A body is not
    /// told what it just did — it knows — but it must be told how a journey it
    /// began four ticks ago turned out, because by then the answer is news.
    pub fn is_outcome(&self) -> bool {
        matches!(
            self,
            Happening::GotThere { .. } | Happening::LostTheWay { .. } | Happening::Bumped { .. }
        )
    }

    /// Whether this happened inside one body and is visible to nobody else.
    ///
    /// The journey internals are — where a body meant to go, and its own sense of
    /// having got there or given up. A [`Happening::Bumped`] is an outcome but
    /// **not** private: it is a collision two bodies share, so the private set is
    /// listed rather than derived from [`Self::is_outcome`].
    pub fn is_private(&self) -> bool {
        matches!(
            self,
            Happening::SetOut { .. } | Happening::GotThere { .. } | Happening::LostTheWay { .. }
        )
    }
}

/// How an utterance was pitched — the one thing about it that decides who else
/// can make it out.
///
/// Carried on the event rather than inferred from who was listening, because
/// the world records what happened in full and [`crate::witness`] narrows it on
/// the way out: a whisper is a thing everybody in the room saw happen and one
/// person heard.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Voice {
    /// An ordinary voice. Everybody in the room hears it; nobody outside does.
    Said,
    /// Raised. The room hears it, and so does every room that can see into it
    /// — the one kind of speech that carries through a doorway.
    Shouted,
    /// Lowered. Only the person it is aimed at makes out the words; everybody
    /// else in the room sees it happen and hears nothing of it.
    Whispered,
}

/// Something that happened, as the log keeps it. Serialisable so a world's
/// authority can send what happened to the members following it, which take
/// it into their own logs ([`World::witness`]).
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Event {
    pub at: Tick,
    pub actor: String,
    pub place: Where,
    pub what: Happening,
}

/// Why an action did not happen.
///
/// Typed rather than a message, because every one of these is an ordinary
/// outcome an NPC has to be able to act on — a full room is not an error, it
/// is a fact about the room — and because a test asserting on prose is a test
/// that breaks when the prose improves.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Refused {
    NoSuchActor(String),
    NoSuchPlace(Where),
    /// There is no route from where the actor is to where it asked to go.
    ///
    /// A refusal rather than a journey that fails later, because a body that
    /// knows a building knows before it stands up. Walking to a door to find
    /// out it leads nowhere is what a body without a memory does.
    NoWay {
        from: Where,
        to: Where,
    },
    /// This world names nowhere to teleport to.
    NoTeleport,
    /// Somebody was addressed who is not in the room. Talking *about* an
    /// absent person is undirected speech with their name in it.
    NotHere {
        who: String,
    },
    /// A body addressed itself. Muttering is [`World::say`].
    SpeakingToYourself,
    /// Nothing here is worked at.
    NothingToWorkAt,
    /// Every station in the room is taken.
    EveryStationTaken {
        of: u32,
    },
    /// Somebody else has it, anywhere in the world.
    AlreadyHeld {
        subject: String,
        by: String,
    },
    /// This actor is already at a station.
    AlreadyAtAStation,
    /// A station that claims something was taken without saying what.
    SubjectNeeded {
        binds: String,
    },
    /// A station that claims nothing was taken with a subject anyway.
    SubjectRefused,
    NotAtAStation,
}

impl fmt::Display for Refused {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Refused::NoSuchActor(id) => write!(f, "there is no actor `{id}`"),
            Refused::NoSuchPlace(w) => write!(f, "there is no place `{w}`"),
            Refused::NoWay { from, to } => write!(f, "no way from `{from}` to `{to}`"),
            Refused::NoTeleport => write!(f, "there is nowhere to teleport to"),
            Refused::NotHere { who } => write!(f, "`{who}` is not in this room"),
            Refused::SpeakingToYourself => write!(f, "cannot address yourself"),
            Refused::NothingToWorkAt => write!(f, "nothing here is worked at"),
            Refused::EveryStationTaken { of } => {
                write!(f, "all {of} stations here are taken")
            }
            Refused::AlreadyHeld { subject, by } => {
                write!(f, "`{subject}` is already held by {by}")
            }
            Refused::AlreadyAtAStation => write!(f, "already at a station"),
            Refused::SubjectNeeded { binds } => write!(f, "a station here takes {binds}"),
            Refused::SubjectRefused => write!(f, "a station here takes hold of nothing"),
            Refused::NotAtAStation => write!(f, "not at a station"),
        }
    }
}

impl std::error::Error for Refused {}
