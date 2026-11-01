//! The shared protocol libraries (`docs/zen_moon_mind.md` §15): the transports
//! a device profile is built on, so each profile is thin and no two reimplement
//! the same wire.
//!
//! - [`soap`] — UPnP SOAP over HTTP (the Sonos player, a generic media
//!   renderer).
//! - [`textline`] — a line-at-a-time TCP control protocol (an AV receiver).
//! - [`udp`] — a UDP broadcast on the house's own subnets (wake-on-LAN).

pub mod soap;
pub mod textline;
pub mod udp;
