//! Talking to a remote. Authentication is the user's own transport setup —
//! for SSH, their keys and agent — and nothing here handles a credential.

pub mod bundle;
pub mod fetch;
pub mod fetch_bundle;
pub mod fetcher;
pub mod ls_remote;
pub mod manage;
pub mod probe;
pub mod push;
pub mod pusher;
