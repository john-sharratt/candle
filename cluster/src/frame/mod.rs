//! The frame codec (§4.1). Each WebSocket binary message is one frame:
//!
//! ```text
//! u8 kind | u8 flags | u32 stream (LE) | body…
//! ```
//!
//! The stream field names a virtual connection on `OPEN`, `DATA`, `WINDOW`,
//! `FIN` and `RESET` and is zero on every other kind. `EVENT` carries its lane
//! flags; a gateway's `PROMISE`, `NACK` and `ACCEPTED` carry the tie-break
//! flag (§5.9). A frame that is short, long, of an unknown kind, or carries a
//! stream or a flag its kind does not is refused, never misread.

mod builds;
mod consensus;
mod handshake;
mod kind;
mod membership;
mod packing;
mod peering;
mod pulse;
mod streams;
mod substrate;
mod types;

pub use builds::{BuildArtifacts, BuildChunk, BuildFetch, Verdict, VerdictResult, MAX_BUILD_CHUNK};
pub use consensus::{Accepted, AcceptedValue, Nack, Prepare, Promise, Propose, Release};
pub use handshake::{Admit, Challenge, Hello, Prove};
pub use kind::{Kind, Lane, FLAG_ADDRESSED, FLAG_CATCHUP, FLAG_TIEBREAK};
pub use membership::{Capacity, Claim, Complete, Coord, Members, Routes, Unclaim, PLACEMENT_WIDE};
pub use packing::{read_applied, write_applied};
pub use peering::{AcmeCert, AcmeChallenge, CertLine, PeerHello};
pub use pulse::Pulse;
pub use streams::{Data, Open, Window, MAX_DATA};
pub use substrate::{
    Ack, Catchup, Done, Event, Fetch, Gone, Have, Headers, Receipt, RecordHeader, MAX_FRAGMENT,
};
pub use types::{
    Artifact, Ballot, BuildId, ClaimInfo, Epoch, MemberInfo, MemberState, ModelId, RepoAd,
};

use crate::wire::{Reader, WireError, Writer};

/// The frame header's length.
pub const HEADER_LEN: usize = 6;

/// One frame.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Frame {
    /// Boxed: by far the largest body, and sent once per socket.
    Hello(Box<Hello>),
    Challenge(Challenge),
    Prove(Prove),
    Admit(Admit),
    Members(Members),
    Capacity(Capacity),
    Ping([u8; 8]),
    Pong([u8; 8]),
    Routes(Routes),
    Claim(Claim),
    Unclaim(Unclaim),
    Coord(Coord),
    Complete(Complete),
    Prepare(Prepare),
    Promise(Promise),
    Nack(Nack),
    Propose(Propose),
    Accepted(Accepted),
    Release(Release),
    BuildOffer(BuildArtifacts),
    BuildHave(BuildArtifacts),
    BuildFetch(BuildFetch),
    BuildChunk(BuildChunk),
    Verdict(Verdict),
    Pulse(Pulse),
    Open(Open),
    Data(Data),
    Window(Window),
    Fin {
        stream: u32,
    },
    Reset {
        stream: u32,
    },
    /// One `EVENT` fragment; `catchup` is the `CATCHUP` flag, and `ADDRESSED`
    /// is set exactly when the event carries a `reply_to`.
    Event {
        catchup: bool,
        event: Event,
    },
    Receipt(Receipt),
    Ack(Ack),
    Catchup(Catchup),
    Headers(Headers),
    Fetch(Fetch),
    Done(Done),
    Have(Have),
    PeerHello(PeerHello),
    AcmeChallenge(AcmeChallenge),
    /// A peer holds the challenge `token` and answers it.
    AcmeAck {
        token: String,
    },
    AcmeClear {
        token: String,
    },
    AcmeCert(AcmeCert),
    /// The sender's order ledger, as its JSON.
    AcmeLedger {
        ledger: Vec<u8>,
    },
    /// On connect: each site's certificate the sender holds.
    AcmeInventory {
        certs: Vec<CertLine>,
    },
    AcmeOrdering {
        site: String,
    },
    AcmeDone {
        site: String,
    },
    /// The shared ACME account's credentials, as their JSON.
    AcmeAccount {
        account: Vec<u8>,
    },
}

impl Frame {
    pub fn kind(&self) -> Kind {
        match self {
            Frame::Hello(_) => Kind::Hello,
            Frame::Challenge(_) => Kind::Challenge,
            Frame::Prove(_) => Kind::Prove,
            Frame::Admit(_) => Kind::Admit,
            Frame::Members(_) => Kind::Members,
            Frame::Capacity(_) => Kind::Capacity,
            Frame::Ping(_) => Kind::Ping,
            Frame::Pong(_) => Kind::Pong,
            Frame::Routes(_) => Kind::Routes,
            Frame::Claim(_) => Kind::Claim,
            Frame::Unclaim(_) => Kind::Unclaim,
            Frame::Coord(_) => Kind::Coord,
            Frame::Complete(_) => Kind::Complete,
            Frame::Prepare(_) => Kind::Prepare,
            Frame::Promise(_) => Kind::Promise,
            Frame::Nack(_) => Kind::Nack,
            Frame::Propose(_) => Kind::Propose,
            Frame::Accepted(_) => Kind::Accepted,
            Frame::Release(_) => Kind::Release,
            Frame::BuildOffer(_) => Kind::BuildOffer,
            Frame::BuildHave(_) => Kind::BuildHave,
            Frame::BuildFetch(_) => Kind::BuildFetch,
            Frame::BuildChunk(_) => Kind::BuildChunk,
            Frame::Verdict(_) => Kind::Verdict,
            Frame::Pulse(_) => Kind::Pulse,
            Frame::Open(_) => Kind::Open,
            Frame::Data(_) => Kind::Data,
            Frame::Window(_) => Kind::Window,
            Frame::Fin { .. } => Kind::Fin,
            Frame::Reset { .. } => Kind::Reset,
            Frame::Event { .. } => Kind::Event,
            Frame::Receipt(_) => Kind::Receipt,
            Frame::Ack(_) => Kind::Ack,
            Frame::Catchup(_) => Kind::Catchup,
            Frame::Headers(_) => Kind::Headers,
            Frame::Fetch(_) => Kind::Fetch,
            Frame::Done(_) => Kind::Done,
            Frame::Have(_) => Kind::Have,
            Frame::PeerHello(_) => Kind::PeerHello,
            Frame::AcmeChallenge(_) => Kind::AcmeChallenge,
            Frame::AcmeAck { .. } => Kind::AcmeAck,
            Frame::AcmeClear { .. } => Kind::AcmeClear,
            Frame::AcmeCert(_) => Kind::AcmeCert,
            Frame::AcmeLedger { .. } => Kind::AcmeLedger,
            Frame::AcmeInventory { .. } => Kind::AcmeInventory,
            Frame::AcmeOrdering { .. } => Kind::AcmeOrdering,
            Frame::AcmeDone { .. } => Kind::AcmeDone,
            Frame::AcmeAccount { .. } => Kind::AcmeAccount,
        }
    }

    /// The lane the writer queues this frame on (§4.1's table).
    pub fn lane(&self) -> Lane {
        match self {
            Frame::Pulse(_)
            | Frame::Open(_)
            | Frame::Data(_)
            | Frame::Fin { .. }
            | Frame::Reset { .. } => Lane::Ordered,
            Frame::Event { catchup, event } => {
                if *catchup || event.reply_to.is_some() {
                    Lane::CatchUp
                } else {
                    Lane::Ordered
                }
            }
            Frame::BuildChunk(_) | Frame::Headers(_) | Frame::Done(_) | Frame::Have(_) => {
                Lane::CatchUp
            }
            _ => Lane::Control,
        }
    }

    fn stream(&self) -> u32 {
        match self {
            Frame::Open(o) => o.stream,
            Frame::Data(d) => d.stream,
            Frame::Window(w) => w.stream,
            Frame::Fin { stream } | Frame::Reset { stream } => *stream,
            _ => 0,
        }
    }

    fn flags(&self) -> u8 {
        match self {
            Frame::Event { catchup, event } => {
                let mut flags = 0;
                if *catchup {
                    flags |= FLAG_CATCHUP;
                }
                if event.reply_to.is_some() {
                    flags |= FLAG_ADDRESSED;
                }
                flags
            }
            Frame::Promise(p) if p.tiebreak => FLAG_TIEBREAK,
            Frame::Nack(n) if n.tiebreak => FLAG_TIEBREAK,
            Frame::Accepted(a) if a.tiebreak => FLAG_TIEBREAK,
            _ => 0,
        }
    }

    /// The frame's length on the wire.
    pub fn encoded_len(&self) -> usize {
        match self {
            Frame::Data(d) => HEADER_LEN + d.bytes.len(),
            _ => self.encode().len(),
        }
    }

    pub fn encode(&self) -> Vec<u8> {
        let mut w = Writer::with_capacity(HEADER_LEN + 64);
        w.u8(self.kind().tag()).u8(self.flags()).u32(self.stream());
        match self {
            Frame::Hello(b) => b.write(&mut w),
            Frame::Challenge(b) => b.write(&mut w),
            Frame::Prove(b) => b.write(&mut w),
            Frame::Admit(b) => b.write(&mut w),
            Frame::Members(b) => b.write(&mut w),
            Frame::Capacity(b) => b.write(&mut w),
            Frame::Ping(n) | Frame::Pong(n) => {
                w.fixed(n);
            }
            Frame::Routes(b) => {
                w.blob(&b.table);
            }
            Frame::Claim(b) => b.write(&mut w),
            Frame::Unclaim(b) => b.write(&mut w),
            Frame::Coord(b) => {
                w.blob(&b.state);
            }
            Frame::Complete(b) => b.write(&mut w),
            Frame::Prepare(b) => b.write(&mut w),
            Frame::Promise(b) => b.write(&mut w),
            Frame::Nack(b) => b.write(&mut w),
            Frame::Propose(b) => b.write(&mut w),
            Frame::Accepted(b) => b.write(&mut w),
            Frame::Release(b) => b.write(&mut w),
            Frame::BuildOffer(b) | Frame::BuildHave(b) => b.write(&mut w),
            Frame::BuildFetch(b) => b.write(&mut w),
            Frame::BuildChunk(b) => b.write(&mut w),
            Frame::Verdict(b) => b.write(&mut w),
            Frame::Pulse(b) => b.write(&mut w),
            Frame::Open(b) => b.write(&mut w),
            Frame::Data(b) => {
                w.fixed(&b.bytes);
            }
            Frame::Window(b) => {
                w.u32(b.credit);
            }
            Frame::Fin { .. } | Frame::Reset { .. } => {}
            Frame::Event { event, .. } => event.write(&mut w),
            Frame::Receipt(b) => {
                w.u64(b.through);
            }
            Frame::Ack(b) => b.write(&mut w),
            Frame::Catchup(b) => b.write(&mut w),
            Frame::Headers(b) => b.write(&mut w),
            Frame::Fetch(b) => b.write(&mut w),
            Frame::Done(b) => b.write(&mut w),
            Frame::Have(b) => b.write(&mut w),
            Frame::PeerHello(b) => b.write(&mut w),
            Frame::AcmeChallenge(b) => b.write(&mut w),
            Frame::AcmeAck { token } | Frame::AcmeClear { token } => {
                w.str(token);
            }
            Frame::AcmeCert(b) => b.write(&mut w),
            Frame::AcmeLedger { ledger: bytes } | Frame::AcmeAccount { account: bytes } => {
                w.blob(bytes);
            }
            Frame::AcmeInventory { certs } => {
                w.list(certs, |w, c| c.write(w));
            }
            Frame::AcmeOrdering { site } | Frame::AcmeDone { site } => {
                w.str(site);
            }
        }
        w.into_bytes()
    }

    pub fn decode(bytes: &[u8]) -> Result<Frame, WireError> {
        if bytes.len() < HEADER_LEN {
            return Err(WireError::Truncated {
                what: "frame header",
                need: HEADER_LEN - bytes.len(),
            });
        }
        let kind = Kind::from_tag(bytes[0])?;
        let flags = bytes[1];
        let stream = u32::from_le_bytes(bytes[2..6].try_into().unwrap());
        let what = name(kind);
        if flags & !kind.allowed_flags() != 0 {
            return Err(WireError::Invalid {
                what,
                value: flags as u64,
            });
        }
        if kind.carries_stream() == (stream == 0) {
            return Err(WireError::Invalid {
                what,
                value: stream as u64,
            });
        }
        let mut r = Reader::new(&bytes[HEADER_LEN..], what);
        let frame = match kind {
            Kind::Hello => Frame::Hello(Box::new(Hello::read(&mut r)?)),
            Kind::Challenge => Frame::Challenge(Challenge::read(&mut r)?),
            Kind::Prove => Frame::Prove(Prove::read(&mut r)?),
            Kind::Admit => Frame::Admit(Admit::read(&mut r)?),
            Kind::Members => Frame::Members(Members::read(&mut r)?),
            Kind::Capacity => Frame::Capacity(Capacity::read(&mut r)?),
            Kind::Ping => Frame::Ping(r.fixed()?),
            Kind::Pong => Frame::Pong(r.fixed()?),
            Kind::Routes => Frame::Routes(Routes {
                table: r.blob_vec()?,
            }),
            Kind::Claim => Frame::Claim(Claim::read(&mut r)?),
            Kind::Unclaim => Frame::Unclaim(Unclaim::read(&mut r)?),
            Kind::Coord => Frame::Coord(Coord {
                state: r.blob_vec()?,
            }),
            Kind::Complete => Frame::Complete(Complete::read(&mut r)?),
            Kind::Prepare => Frame::Prepare(Prepare::read(&mut r)?),
            Kind::Promise => {
                let mut body = Promise::read(&mut r)?;
                body.tiebreak = flags & FLAG_TIEBREAK != 0;
                Frame::Promise(body)
            }
            Kind::Nack => {
                let mut body = Nack::read(&mut r)?;
                body.tiebreak = flags & FLAG_TIEBREAK != 0;
                Frame::Nack(body)
            }
            Kind::Propose => Frame::Propose(Propose::read(&mut r)?),
            Kind::Accepted => {
                let mut body = Accepted::read(&mut r)?;
                body.tiebreak = flags & FLAG_TIEBREAK != 0;
                Frame::Accepted(body)
            }
            Kind::Release => Frame::Release(Release::read(&mut r)?),
            Kind::BuildOffer => Frame::BuildOffer(BuildArtifacts::read(&mut r)?),
            Kind::BuildHave => Frame::BuildHave(BuildArtifacts::read(&mut r)?),
            Kind::BuildFetch => Frame::BuildFetch(BuildFetch::read(&mut r)?),
            Kind::BuildChunk => Frame::BuildChunk(BuildChunk::read(&mut r)?),
            Kind::Verdict => Frame::Verdict(Verdict::read(&mut r)?),
            Kind::Pulse => Frame::Pulse(Pulse::read(&mut r)?),
            Kind::Open => Frame::Open(Open::read(&mut r, stream)?),
            Kind::Data => Frame::Data(Data::read(&mut r, stream)?),
            Kind::Window => Frame::Window(Window::read(&mut r, stream)?),
            Kind::Fin => Frame::Fin { stream },
            Kind::Reset => Frame::Reset { stream },
            Kind::Event => Frame::Event {
                catchup: flags & FLAG_CATCHUP != 0,
                event: Event::read(&mut r, flags & FLAG_ADDRESSED != 0)?,
            },
            Kind::Receipt => Frame::Receipt(Receipt { through: r.u64()? }),
            Kind::Ack => Frame::Ack(Ack::read(&mut r)?),
            Kind::Catchup => Frame::Catchup(Catchup::read(&mut r)?),
            Kind::Headers => Frame::Headers(Headers::read(&mut r)?),
            Kind::Fetch => Frame::Fetch(Fetch::read(&mut r)?),
            Kind::Done => Frame::Done(Done::read(&mut r)?),
            Kind::Have => Frame::Have(Have::read(&mut r)?),
            Kind::PeerHello => Frame::PeerHello(PeerHello::read(&mut r)?),
            Kind::AcmeChallenge => Frame::AcmeChallenge(AcmeChallenge::read(&mut r)?),
            Kind::AcmeAck => Frame::AcmeAck { token: r.str()? },
            Kind::AcmeClear => Frame::AcmeClear { token: r.str()? },
            Kind::AcmeCert => Frame::AcmeCert(AcmeCert::read(&mut r)?),
            Kind::AcmeLedger => Frame::AcmeLedger {
                ledger: r.blob_vec()?,
            },
            Kind::AcmeInventory => Frame::AcmeInventory {
                certs: r.list(CertLine::read)?,
            },
            Kind::AcmeOrdering => Frame::AcmeOrdering { site: r.str()? },
            Kind::AcmeDone => Frame::AcmeDone { site: r.str()? },
            Kind::AcmeAccount => Frame::AcmeAccount {
                account: r.blob_vec()?,
            },
        };
        r.finish()?;
        Ok(frame)
    }
}

/// The kind's name as the design spells it, for error messages.
pub fn name(kind: Kind) -> &'static str {
    match kind {
        Kind::Hello => "HELLO",
        Kind::Challenge => "CHALLENGE",
        Kind::Prove => "PROVE",
        Kind::Admit => "ADMIT",
        Kind::Members => "MEMBERS",
        Kind::Capacity => "CAPACITY",
        Kind::Ping => "PING",
        Kind::Pong => "PONG",
        Kind::Routes => "ROUTES",
        Kind::Claim => "CLAIM",
        Kind::Unclaim => "UNCLAIM",
        Kind::Coord => "COORD",
        Kind::Complete => "COMPLETE",
        Kind::Prepare => "PREPARE",
        Kind::Promise => "PROMISE",
        Kind::Nack => "NACK",
        Kind::Propose => "PROPOSE",
        Kind::Accepted => "ACCEPTED",
        Kind::Release => "RELEASE",
        Kind::BuildOffer => "BUILD_OFFER",
        Kind::BuildHave => "BUILD_HAVE",
        Kind::BuildFetch => "BUILD_FETCH",
        Kind::BuildChunk => "BUILD_CHUNK",
        Kind::Verdict => "VERDICT",
        Kind::Pulse => "PULSE",
        Kind::Open => "OPEN",
        Kind::Data => "DATA",
        Kind::Window => "WINDOW",
        Kind::Fin => "FIN",
        Kind::Reset => "RESET",
        Kind::Event => "EVENT",
        Kind::Receipt => "RECEIPT",
        Kind::Ack => "ACK",
        Kind::Catchup => "CATCHUP",
        Kind::Headers => "HEADERS",
        Kind::Fetch => "FETCH",
        Kind::Done => "DONE",
        Kind::Have => "HAVE",
        Kind::PeerHello => "PEER_HELLO",
        Kind::AcmeChallenge => "ACME_CHALLENGE",
        Kind::AcmeAck => "ACME_ACK",
        Kind::AcmeClear => "ACME_CLEAR",
        Kind::AcmeCert => "ACME_CERT",
        Kind::AcmeLedger => "ACME_LEDGER",
        Kind::AcmeInventory => "ACME_INVENTORY",
        Kind::AcmeOrdering => "ACME_ORDERING",
        Kind::AcmeDone => "ACME_DONE",
        Kind::AcmeAccount => "ACME_ACCOUNT",
    }
}

#[cfg(test)]
mod tests;
