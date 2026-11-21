//! Names the people a character talks to, in the form the world writes them down.
//!
//! A person's name on a thread is the name the world uses for them, since threads
//! and rooms address people the same way, and one person must not be two. An
//! account handle is used only when the provider gave no display name, because a
//! handle reads as a stranger in the prose the character is handed. The creator's
//! role is derived from the same check the access guard uses, so the name a
//! character reads cannot disagree with the access decision.

use super::*;

/// How the person who made these worlds is marked, in the world.
///
/// **Part of the name rather than a field beside it**, because a name is the
/// whole of how anybody is addressed here: threads, rooms and the roster all
/// key on it, and `tell`, `ask` and `message` bind their addressee to a closed
/// set of exactly these strings. A separate "is the creator" flag would have to
/// be carried to every one of those places and rendered into the prose at each,
/// and the first one that forgot would be a character talking to a stranger.
///
/// Carried in the name, it needs no plumbing at all: it is what a character
/// reads when the message arrives, and it is what the grammar offers back when
/// the character answers.
pub(super) const CREATOR_MARK: &str = "(The Creator)";

/// What a person is called, on a thread with a character.
///
/// **The name the world writes down**, because a thread addresses people the
/// same way a room does and one person must never be two. An account handle
/// (`u_1a2b3c4d`) would read as a stranger in the prose the character is handed
/// — "u_1a2b3c4d messages you" — so the identity's own name is used when the
/// provider gave one, and the handle is the fallback that at least stays
/// stable.
///
/// # Why the role is read here
///
/// This is the one place a person outside the world acquires a name inside it,
/// so it is the only place [`Role::Creator`] can be turned into something the
/// fiction can see. The role is derived from the identity through the same
/// [`Roles::of`] the guard used, rather than passed in — two computations of
/// "is this the creator" could disagree, and the one that decided what a
/// character *reads* would be the one nothing tested.
pub(crate) fn speaking_as(
    id: &web::auth::session::Identity,
    handle: &str,
    roles: &web::auth::Roles,
) -> String {
    let name = match id.name.trim() {
        "" => handle,
        name => name,
    };
    match roles.of(Some(id)).is_creator() {
        true => format!("{name} {CREATOR_MARK}"),
        false => name.to_string(),
    }
}

/// What a message or channel body says ([`Say`]): a body with no `text` says
/// nothing, and is refused as empty in the route's own words.
pub(super) fn said(body: Result<Json<Say>, JsonRejection>) -> String {
    body.map(|Json(b)| b.text).unwrap_or_default()
}
