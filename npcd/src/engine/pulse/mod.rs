//! Pulse, the instrument for watching a cast think, tick by tick.
//!
//! The loop is asynchronous and invisible, and a transcript records what happened
//! but never why it happened then. Pulse shows every tick across the cast in
//! order: what the character perceived as prose, why it woke, what it did, and
//! what its metabolism did next. The same view is the authoring instrument, so it
//! is a first-class surface rather than a debug flag.
//!
//! - `broadcasts`: admin events and announcements to every character at once.
//! - `character`: a character's system prompt, prompt tokens and journal.
//! - `inputs`: console lines and direct speech put on a character's inbox.
//! - `scope`: which characters a caller may watch.
//! - `views`: the feed, census, world, window and scan.
//! - `vocabulary`: the command catalogue and the act vocabulary.
//!
//! - `tests_pulse`: tests of the routes, defaults and catalogue.

use std::collections::HashSet;
use std::sync::Arc;

use axum::extract::{Path, Query, State};
use axum::http::{HeaderMap, StatusCode};
use axum::response::{IntoResponse, Response};
use axum::Json;
use serde::Deserialize;
use serde_json::{json, Value};
use web::auth::{Identity, Role};

use crate::api::{err, owner_of, Authored};
use crate::engine::event::{Addressed, Event, EventKind, Salience};
use crate::engine::journal::state::JournalState;
use crate::engine::journal::text as journal_text;
use crate::engine::runtime::Runtime;
use crate::engine::slash;
use crate::engine::survey::Survey;
use crate::engine::{owned, owned_by};
use crate::npcs;
use crate::suspend;

/// How many ticks a feed request may ask for. The scheduler's own ring is the
/// hard bound; this stops one request rendering all of it.
const MAX_FEED: usize = 200;

const DEFAULT_FEED: usize = 50;

fn no_scheduler() -> Response {
    err(
        StatusCode::SERVICE_UNAVAILABLE,
        "no_engine",
        "the tick scheduler is not running",
    )
}

mod broadcasts;
mod character;
mod inputs;
mod scope;
#[cfg(test)]
mod tests_pulse;
mod views;
mod vocabulary;

pub use broadcasts::*;
pub use character::*;
pub use inputs::*;
pub use scope::*;
pub use views::*;
pub use vocabulary::*;
