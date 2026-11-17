//! What other services tell the gateway happened (`docs/battlecities_app.md`
//! §4.1).
//!
//! | Module | |
//! |---|---|
//! | [`github`] | a push, as GitHub's webhook tells it: its signature and its body |

pub mod github;
