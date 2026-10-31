//! The bar of a bot tool that reads a bot's insides rather than calling a
//! route: its owner, or an admin. Anyone else is told what a missing
//! character's route tells them.

use std::sync::Arc;

use axum::http::HeaderMap;
use web::auth::Role;

use super::route::refusal;
use crate::api::{owner_of, Authored};
use crate::identity::identify;
use crate::npcs;

/// What a character the caller may not see is answered with, as the routes
/// word it.
pub const NOT_FOUND: &str = "npc_not_found: no such character (404)";

/// Bot `bot`'s id, if the caller owns it or is an admin.
pub async fn owned_or_admin(
    s: &Arc<Authored>,
    headers: &HeaderMap,
    bot: &str,
) -> Result<u64, String> {
    let npc = npcs::npc_id_of_wire(bot)
        .filter(|npc| s.npcs.view().is_living(*npc))
        .ok_or(NOT_FOUND)?;
    let id = identify(headers).ok();
    if s.roles.of(id.as_ref()).at_least(Role::Admin) {
        return Ok(npc);
    }
    let (_, owner) = match owner_of(s, headers).await {
        Ok(owner) => owner,
        Err(res) => return Err(refusal(*res).await),
    };
    match s.npcs.view().visible_to(npc, &owner) {
        Some(_) => Ok(npc),
        None => Err(NOT_FOUND.into()),
    }
}
