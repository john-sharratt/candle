//! Signing in as John and Yen to Spotify, and as John to TIDAL (§4.1):
//! PKCE ([`pkce`]), kept and refreshed tokens ([`token`]), and the flow
//! through a loopback redirect ([`flow`]).

pub mod flow;
pub mod pkce;
pub mod token;

pub use flow::{login, Account, Provider};
