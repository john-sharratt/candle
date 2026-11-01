//! The shared protocol libraries (`docs/zen_moon_mind.md` §15): the transports
//! a device profile is built on, so each profile is thin and no two reimplement
//! the same wire.
//!
//! - [`soap`] — UPnP SOAP over HTTP (the Sonos player, a generic media
//!   renderer).
//! - [`eiscp`] — the Integra Serial Control Protocol framed over TCP (an AV
//!   receiver).
//! - [`cast`] — CASTv2, a protobuf framed over TLS (a Google Cast device).
//! - [`webos`] — SSAP over a WebSocket (an LG webOS TV).
//! - [`udp`] — a UDP broadcast on the house's own subnets (wake-on-LAN).

pub mod androidtv;
pub mod cast;
pub mod eiscp;
pub mod soap;
pub mod udp;
pub mod webos;
