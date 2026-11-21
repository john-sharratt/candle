//! Decides which characters a caller may watch in the pulse feed.
//!
//! Ownership is the default, because a tick records the literal prose a character
//! perceived and another owner's character's perceptions belong to that owner. An
//! admin may ask for everything with `?all=1`, and receives it only if the role is
//! actually admin. The check is a bar on the ordered role ladder rather than an
//! equality, so a higher role keeps the control. `None` means no filter, which is
//! kept distinct from an empty set that would read as nothing to show.

use super::*;

/// Which characters this caller may watch, and how to describe the choice.
pub(super) struct Scope {
    /// `None` means no filter — an admin who asked for everything.
    pub(super) ids: Option<HashSet<u64>>,
    /// Whether the caller could have asked for everything, so the console knows
    /// whether to offer the toggle at all. A control that appears and then
    /// refuses is worse than one that never appears.
    pub(super) may_see_all: bool,
    /// Whether they did.
    pub(super) showing_all: bool,
}

/// Resolve what this caller may watch.
///
/// **Ownership is the default here as everywhere else on this daemon.** The tick
/// loop runs the whole cast — the quartermaster counts sacks whether or not his
/// author is signed in — but a tick record carries the literal prose a character
/// perceived, and somebody else's character's perceptions are somebody else's
/// world.
///
/// An admin may ask for everything with `?all=1`, and gets it only if they
/// actually are one: the flag is a request, never a grant.
pub(super) async fn scope_of(s: &Arc<Authored>, id: &Identity, owner: &str, all: bool) -> Scope {
    // **A bar, not an equality.** This read `matches!(.., Role::Admin)`, which
    // was the same thing while `Admin` was the top of the ladder and stopped
    // being it the moment `Creator` went above — the person who owns the estate
    // signed in and silently lost a control an admin has. An exact match on a
    // level in an *ordered* enum is a demotion waiting for the next level to be
    // added, and it does not fail, it just quietly refuses.
    let may_see_all = s.roles.of(Some(id)).at_least(Role::Admin);
    if all && may_see_all {
        return Scope {
            // `None`, not an empty set — an empty set reads as "nothing to
            // show", which is the opposite of "no filter".
            ids: None,
            may_see_all,
            showing_all: true,
        };
    }
    let ids = s.npcs.view().owned_by(owner);
    Scope {
        ids: Some(ids),
        may_see_all,
        showing_all: false,
    }
}

impl Scope {
    pub(super) fn admits(&self, npc_id: u64) -> bool {
        self.ids.as_ref().is_none_or(|set| set.contains(&npc_id))
    }
}

/// The query of the per-character read routes that let an admin look past
/// ownership: `GET /v1/npc/:nid/journal` and `GET /v1/npc/:nid/system-prompt`.
#[derive(Debug, Default, Deserialize)]
pub struct AllQuery {
    /// Admin only: read a character the caller does not own.
    #[serde(default)]
    pub(super) all: bool,
}

/// The character's id when the caller owns it, or when the caller is an admin.
///
/// An id that is not a character is 404 either way, so an admin asking about
/// nothing is told so rather than getting a 503 from an empty scheduler slot.
pub(super) async fn owned_or_admin(
    s: &Arc<Authored>,
    headers: &HeaderMap,
    nid: &str,
) -> Result<u64, Box<Response>> {
    let (id, _) = owner_of(s, headers).await?;
    if !s.roles.of(Some(&id)).at_least(Role::Admin) {
        return owned(s, headers, nid).await;
    }
    let not_found = || {
        Box::new(err(
            StatusCode::NOT_FOUND,
            "npc_not_found",
            "no such character",
        ))
    };
    let npc_id = npcs::npc_id_of_wire(nid).ok_or_else(not_found)?;
    if !s.npcs.view().is_living(npc_id) {
        return Err(not_found());
    }
    Ok(npc_id)
}

/// The character a read route may look at: the caller's own, or — for an admin
/// who asked with `?all=true` — any living one.
pub(super) async fn readable(
    s: &Arc<Authored>,
    headers: &HeaderMap,
    nid: &str,
    q: &AllQuery,
) -> Result<u64, Box<Response>> {
    if q.all {
        owned_or_admin(s, headers, nid).await
    } else {
        owned(s, headers, nid).await
    }
}
