//! The cluster socket (`docs/replicated_substrate.md` §3–§4): what members and
//! gateways say to each other, and how one WebSocket carries all of it.
//!
//! - [`frame`]: every frame kind, encoded to exact bytes, and the applied set's
//!   packing.
//! - [`handshake`]: the mutual HMAC challenge over the group key.
//! - [`session`]: one admitted socket — the three-lane writer, virtual
//!   connections with flow control, `EVENT` reassembly and keep-alive.
//! - [`dial`] and [`backoff`]: a member's reconnect loop.
//! - [`consensus`]: Paxos among an app's members — leases, decisions, the
//!   vote journal, the quorum and rendezvous.
//! - [`delivery`]: the sender's outbox (unsent, in flight, received) and the
//!   frame counting `RECEIPT`s refer to.
//! - [`gateway`] and [`member`]: the two ends of a socket — admission, routing
//!   and relaying; dialling, claims and the member's own listener.
//! - [`replicate`]: a member's substrate replication — events out and in,
//!   catch-ups, and the rejoin after a cut-off.
//! - [`machine`]: the host as `CAPACITY` reports it.
//! - [`version`]: this build's `count`, `hash` and `local`, and
//!   `--build-info`.
//! - [`upgrade`]: a member's side of A/B upgrades — artifacts, fetching,
//!   compatibility, the swap, kept builds and the watchdog.
//! - [`launch`]: how this process was launched, to start an identical one.
//! - [`peer`]: gateways linked to each other, for the state active-active
//!   gateways share.
//!
//! No candle and no CUDA: the gateway links this crate, and a record inside an
//! `EVENT` stays opaque bytes here.

pub mod backoff;
pub mod clock;
pub mod close;
pub mod consensus;
pub mod coord;
pub mod delivery;
pub mod dial;
pub mod dispatch;
pub mod fragment;
pub mod frame;
pub mod gateway;
pub mod handshake;
pub mod launch;
pub mod machine;
pub mod member;
pub mod peer;
pub mod pipe;
pub mod replicate;
pub mod session;
pub mod upgrade;
pub mod version;
pub mod wire;
