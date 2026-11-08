//! Spotify, for taste and requests (§4.1, §4.5): each person's privacy
//! export ([`export`]), the Web API ([`api`]), and the request playlist
//! watched for what the phones add ([`requests`]).

pub mod api;
pub mod export;
pub mod requests;

pub use api::Spotify;
pub use requests::{Asked, Watch};
