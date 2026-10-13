//! Frame kinds, their tags, their lanes and their flags (§4.1, §4.2).

use crate::wire::WireError;

/// `flags & CATCHUP`: an `EVENT` on the catch-up lane.
pub const FLAG_CATCHUP: u8 = 0x01;
/// `flags & ADDRESSED`: an `EVENT` answering a `FETCH`, carrying `reply_to`.
pub const FLAG_ADDRESSED: u8 = 0x02;
/// `flags & TIEBREAK`: a `PROMISE`, `NACK` or `ACCEPTED` from a gateway voting
/// as the tie-break acceptor (§5.9). A member's vote carries no flag.
pub const FLAG_TIEBREAK: u8 = 0x01;

/// Which of the writer's three lanes a frame waits in (§4.2).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Lane {
    /// Always first.
    Control,
    /// One FIFO of stream frames and live events, in the order produced.
    Ordered,
    /// Only when the ordered lane is empty and the socket is under its
    /// low-water mark.
    CatchUp,
}

macro_rules! kinds {
    ($($name:ident = $tag:literal),* $(,)?) => {
        /// A frame's kind, by its tag on the wire.
        #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
        pub enum Kind {
            $($name = $tag,)*
        }

        impl Kind {
            /// Every kind, in tag order.
            pub const ALL: &'static [Kind] = &[$(Kind::$name,)*];

            pub fn tag(self) -> u8 {
                self as u8
            }

            pub fn from_tag(tag: u8) -> Result<Kind, WireError> {
                match tag {
                    $($tag => Ok(Kind::$name),)*
                    other => Err(WireError::UnknownKind(other)),
                }
            }
        }
    };
}

kinds! {
    Hello = 1,
    Challenge = 2,
    Prove = 3,
    Admit = 4,
    Members = 10,
    Capacity = 11,
    Ping = 12,
    Pong = 13,
    Routes = 14,
    Claim = 15,
    Unclaim = 16,
    Coord = 17,
    Prepare = 20,
    Promise = 21,
    Nack = 22,
    Propose = 23,
    Accepted = 24,
    Release = 25,
    BuildOffer = 30,
    BuildHave = 31,
    BuildFetch = 32,
    BuildChunk = 33,
    Verdict = 34,
    Pulse = 40,
    Open = 50,
    Data = 51,
    Window = 52,
    Fin = 53,
    Reset = 54,
    Event = 60,
    Receipt = 61,
    Ack = 62,
    Catchup = 63,
    Headers = 64,
    Fetch = 65,
    Done = 66,
    Have = 67,
    PeerHello = 70,
    AcmeChallenge = 71,
    AcmeAck = 72,
    AcmeClear = 73,
    AcmeCert = 74,
    AcmeLedger = 75,
    AcmeInventory = 76,
    AcmeOrdering = 77,
    AcmeDone = 78,
    AcmeAccount = 79,
}

impl Kind {
    /// Whether the header's stream field names a virtual connection.
    pub fn carries_stream(self) -> bool {
        matches!(
            self,
            Kind::Open | Kind::Data | Kind::Window | Kind::Fin | Kind::Reset
        )
    }

    /// The flag bits this kind may carry.
    pub fn allowed_flags(self) -> u8 {
        match self {
            Kind::Event => FLAG_CATCHUP | FLAG_ADDRESSED,
            Kind::Promise | Kind::Nack | Kind::Accepted => FLAG_TIEBREAK,
            _ => 0,
        }
    }

    /// Part of the handshake (§3.3): nothing else crosses before `ADMIT`.
    pub fn is_handshake(self) -> bool {
        matches!(
            self,
            Kind::Hello | Kind::Challenge | Kind::Prove | Kind::Admit
        )
    }
}
