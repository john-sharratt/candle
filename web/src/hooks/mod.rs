//! What other services tell the gateway happened, and what it does about it
//! (`docs/battlecities_app.md` §4).
//!
//! | Module | |
//! |---|---|
//! | [`github`] | a push, as GitHub's webhook tells it: its signature and its body |
//! | [`http`] | `POST /gateway/hooks/github`, where GitHub delivers |
//! | [`pushes`] | telling a member of a push, from the webhook and from the watch |

pub mod github;
pub mod http;
pub mod pushes;
