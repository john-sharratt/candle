//! Fetching what a URL serves, from the public internet only
//! (`docs/mcp_file_tools.md` §3.7): `web_fetch`, and the transfers that land a
//! URL's bytes in a repository or send a file to one.
//!
//! A fetch runs on a member's machine, inside its network, where keep's API
//! and every daemon behind the gateway believe identity headers from loopback.
//! So every address a fetch may connect to is checked, and the check is the
//! connection's: the host is resolved once, every address it resolves to must
//! be public, the connection is pinned to the address checked, every other
//! name the client looks up goes through the same check, and each redirect is
//! checked again. A body is read only up to its cap, and abandoned past it.
//!
//! | Module | |
//! |---|---|
//! | [`address`] | which addresses are public, and resolving a host to them |
//! | [`client`] | an HTTP client pinned to a checked address, its resolver and redirects guarded |
//! | [`body`] | a response's bytes, read only up to a cap |
//! | [`error`] | why a fetch did not happen |

mod address;
mod body;
mod client;
mod error;

pub use address::{is_private_ip, is_private_url};
pub use body::{read_head, read_within};
pub use client::{public_client, Pinned};
pub use error::FetchError;
