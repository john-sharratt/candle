//! As whom a request reads and writes the mind (§10.3): an admin as their own
//! author, everyone else the published mind.

use axum::http::{HeaderMap, StatusCode};
use axum::response::Response;
use web::auth::Role;

use super::Author;
use crate::accounts::user_id_of;
use crate::api::{err, Authored};
use crate::identity::{identify, require};

/// The calling admin as an author: their view of the mind, and where their
/// writes go. Refused for a caller who is not an admin, which the route's own
/// guard has already refused.
pub fn author(s: &Authored, headers: &HeaderMap) -> Result<Author, Box<Response>> {
    let id = require(headers, &s.roles, Role::Admin)?
        .into_identity()
        .ok_or_else(|| Box::new(err(StatusCode::FORBIDDEN, "forbidden", "sign in first")))?;
    Ok(s.author(&user_id_of(&id)))
}

/// The caller as an author when they are an admin — whose reads see their own
/// drafts — and `None` for anyone else, who reads the mind as published.
pub fn admin(s: &Authored, headers: &HeaderMap) -> Option<Author> {
    let id = identify(headers).ok()?;
    s.roles
        .of(Some(&id))
        .at_least(Role::Admin)
        .then(|| s.author(&user_id_of(&id)))
}
