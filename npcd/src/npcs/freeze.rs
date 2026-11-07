//! Freezing characters: setting and clearing [`SUSPENDED`], the one state that
//! stops a character taking turns of any kind.
//!
//! Two ways in, with different owners of the answer:
//!
//! - **One character** ([`Npcs::freeze`]) is its owner's Suspend / Resume:
//!   suspended, or put back to the state a freeze of the whole cast found it in
//!   — `idle` when none did. An edit like any other, with its revision.
//! - **The whole cast** ([`Npcs::freeze_all`]) suspends every character the
//!   manager may manage, keeping the state each was in as
//!   [`NpcPayload::suspended_from`], and its thaw puts back exactly those it
//!   suspended. A character suspended any other way has no `suspended_from`, so
//!   the thaw leaves it suspended. Operational rather than authored, for the
//!   reason a mood is: no revision and no `updated_ms`, so a roster sorted by
//!   "recently edited" does not jump every time somebody frees the card.
//!
//! Every write is a superseding record, so it is durable before it is
//! published, survives a restart, and replicates to every member.

use std::mem;

use candle_conversation::persistence::record::NpcPayload;
use serde_json::Value;

use super::{bumped, one, wire, NpcError, Npcs, INITIAL_STATE, SUSPENDED};

/// Who is freezing, and so which characters they may freeze.
#[derive(Clone, Copy, Debug)]
pub enum Manager<'a> {
    /// An account, which manages the characters it owns. Holds its public id.
    Owner(&'a str),
    /// An admin, who manages every living character. Holds the admin's own
    /// public id, which the wire's `access` is answered against.
    Admin(&'a str),
}

impl Manager<'_> {
    /// The account the answer is written for.
    fn caller(&self) -> &str {
        match self {
            Manager::Owner(id) | Manager::Admin(id) => id,
        }
    }

    /// Whether this manager may freeze `npc`. A tombstoned character is
    /// nobody's to freeze.
    fn manages(&self, npc: &NpcPayload) -> bool {
        if npc.is_tombstoned() {
            return false;
        }
        match self {
            Manager::Owner(id) => npc.owner_id == *id,
            Manager::Admin(_) => true,
        }
    }
}

impl Npcs {
    /// Suspend one character, or resume it, and answer with it as it now
    /// stands.
    ///
    /// Resuming puts it back to the state a freeze of the whole cast found it
    /// in, or `idle`. A character the manager may not manage reads as absent,
    /// as every other write's does; setting what is already set writes nothing.
    pub fn freeze(
        &self,
        npc_id: u64,
        by: Manager<'_>,
        frozen: bool,
        now_ms: u64,
    ) -> Result<Value, NpcError> {
        self.write(|cast| {
            let npc = cast
                .by_id
                .get(&npc_id)
                .filter(|n| by.manages(n))
                .ok_or(NpcError::NotFound)?;
            if (npc.state == SUSPENDED) == frozen {
                return Ok((Vec::new(), wire(npc, by.caller())));
            }
            let mut npc = npc.clone();
            npc.state = match frozen {
                true => SUSPENDED.to_string(),
                false => npc
                    .suspended_from
                    .take()
                    .unwrap_or_else(|| INITIAL_STATE.to_string()),
            };
            npc.suspended_from = None;
            Ok(one(bumped(npc, now_ms), by.caller()))
        })
    }

    /// Suspend every character the manager may manage, or resume every one a
    /// freeze of the whole cast suspended, in one write behind one fsync.
    /// Returns how many changed.
    ///
    /// Freezing keeps each character's state as `suspended_from` and passes
    /// over one already suspended, which keeps none; thawing restores the state
    /// kept and passes over a suspended character with none kept.
    pub fn freeze_all(&self, by: Manager<'_>, frozen: bool) -> Result<usize, NpcError> {
        self.write(|cast| {
            let changed: Vec<NpcPayload> = cast
                .by_id
                .values()
                .filter(|n| by.manages(n))
                .filter_map(|n| {
                    let mut n = n.clone();
                    match (frozen, n.state == SUSPENDED) {
                        (true, false) => {
                            n.suspended_from =
                                Some(mem::replace(&mut n.state, SUSPENDED.to_string()));
                        }
                        (false, true) => n.state = n.suspended_from.take()?,
                        _ => return None,
                    }
                    Some(n)
                })
                .collect();
            let n = changed.len();
            Ok((changed, n))
        })
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashSet;
    use std::path::PathBuf;
    use std::sync::atomic::{AtomicU64, Ordering};

    use serde_json::json;
    use web::auth::session::Identity;

    use super::*;
    use crate::npcs::npc_id_of_wire;

    const ME: &str = "u_1a2b3c4d";
    const OTHER: &str = "u_99887766";
    const ADMIN: &str = "u_0ad0ad0a";

    fn tmp() -> PathBuf {
        static N: AtomicU64 = AtomicU64::new(0);
        let p = std::env::temp_dir().join(format!(
            "npcd-freeze-{}-{}",
            std::process::id(),
            N.fetch_add(1, Ordering::Relaxed)
        ));
        let _ = std::fs::remove_dir_all(&p);
        std::fs::create_dir_all(&p).unwrap();
        p
    }

    fn ident(sub: &str) -> Identity {
        Identity {
            provider: "google".to_string(),
            sub: sub.to_string(),
            email: String::new(),
            name: String::new(),
            picture: String::new(),
            exp: 0,
        }
    }

    fn create(n: &Npcs, owner: &str, name: &str, at: u64) -> u64 {
        let body =
            json!({ "name": name, "world_id": "battle-cities", "personality_id": "commander" });
        let v = n.create(&ident("u1"), owner, &body, at).unwrap();
        npc_id_of_wire(v["npc_id"].as_str().unwrap()).unwrap()
    }

    /// `(state, suspended_from)` of one character as the cast now holds it.
    fn standing(n: &Npcs, id: u64) -> (String, Option<String>) {
        let view = n.view();
        let p = view.by_id.get(&id).unwrap();
        (p.state.clone(), p.suspended_from.clone())
    }

    fn st(state: &str, from: Option<&str>) -> (String, Option<String>) {
        (state.to_string(), from.map(str::to_string))
    }

    /// **One character's freeze is its owner's Suspend / Resume**: suspended
    /// with nothing kept, resumed to `idle`, each an edit with its revision,
    /// and both survive a restart.
    #[test]
    fn one_characters_freeze_is_suspend_and_resume() {
        let dir = tmp();
        let varek = {
            let n = Npcs::load(&dir).unwrap();
            let varek = create(&n, ME, "Varek", 1_000);
            n.patch(varek, ME, &json!({ "state": "active" }), 1_500)
                .unwrap();
            let v = n.freeze(varek, Manager::Owner(ME), true, 2_000).unwrap();
            assert_eq!(v["state"], "suspended");
            assert_eq!(v["frozen"], true);
            assert_eq!(v["revision"], 3);
            assert_eq!(v["updated_ms"], 2_000);
            assert_eq!(standing(&n, varek), st("suspended", None));
            assert_eq!(n.view().frozen(), HashSet::from([varek]));
            varek
        };

        let n = Npcs::load(&dir).unwrap();
        assert_eq!(standing(&n, varek), st("suspended", None));
        let v = n.freeze(varek, Manager::Owner(ME), false, 3_000).unwrap();
        assert_eq!(v["state"], "idle");
        assert_eq!(v["frozen"], false);
        drop(n);
        let n = Npcs::load(&dir).unwrap();
        assert_eq!(standing(&n, varek), st("idle", None));
        assert!(n.view().frozen().is_empty());
    }

    /// Setting what is already set writes no record at all.
    #[test]
    fn freezing_a_frozen_character_writes_nothing() {
        let n = Npcs::load(&tmp()).unwrap();
        let id = create(&n, ME, "Varek", 1_000);
        n.freeze(id, Manager::Owner(ME), true, 2_000).unwrap();
        let identity = n.view().read_as.get(&id).copied();
        let v = n.freeze(id, Manager::Owner(ME), true, 3_000).unwrap();
        assert_eq!(v["updated_ms"], 2_000);
        assert_eq!(n.view().read_as.get(&id).copied(), identity);
    }

    /// **An owner freezes their own, and an admin freezes anybody's.** A
    /// stranger's character reads as absent to an owner, as every write's does.
    #[test]
    fn an_owner_freezes_their_own_and_an_admin_anybodys() {
        let n = Npcs::load(&tmp()).unwrap();
        let mine = create(&n, ME, "Varek", 1_000);
        let theirs = create(&n, OTHER, "Ilse", 1_001);

        assert!(matches!(
            n.freeze(theirs, Manager::Owner(ME), true, 2_000),
            Err(NpcError::NotFound)
        ));
        assert!(n.view().frozen().is_empty());

        let v = n
            .freeze(theirs, Manager::Admin(ADMIN), true, 2_000)
            .unwrap();
        assert_eq!(v["state"], "suspended");
        assert_eq!(v["access"], "viewer");
        n.freeze(mine, Manager::Owner(ME), true, 2_000).unwrap();
        assert_eq!(n.view().frozen(), HashSet::from([mine, theirs]));
    }

    /// **A freeze of the whole cast and its thaw put each character back in
    /// the state it was in**: an active one active, an idle one idle — neither
    /// an edit — across a restart between the two.
    #[test]
    fn a_cast_freeze_and_thaw_return_each_character_to_its_state() {
        let dir = tmp();
        let (varek, ilse) = {
            let n = Npcs::load(&dir).unwrap();
            let varek = create(&n, ME, "Varek", 1_000);
            let ilse = create(&n, ME, "Ilse", 1_001);
            n.patch(varek, ME, &json!({ "state": "active" }), 1_500)
                .unwrap();

            assert_eq!(n.freeze_all(Manager::Owner(ME), true).unwrap(), 2);
            assert_eq!(standing(&n, varek), st("suspended", Some("active")));
            assert_eq!(standing(&n, ilse), st("suspended", Some("idle")));
            let view = n.view();
            let p = view.by_id.get(&varek).unwrap();
            assert_eq!((p.revision, p.updated_ms), (2, 1_500), "not an edit");
            assert_eq!(n.freeze_all(Manager::Owner(ME), true).unwrap(), 0);
            (varek, ilse)
        };

        let n = Npcs::load(&dir).unwrap();
        assert_eq!(n.view().frozen(), HashSet::from([varek, ilse]));
        assert_eq!(n.freeze_all(Manager::Owner(ME), false).unwrap(), 2);
        assert_eq!(standing(&n, varek), st("active", None));
        assert_eq!(standing(&n, ilse), st("idle", None));
        assert!(n.view().frozen().is_empty());
        assert_eq!(n.freeze_all(Manager::Owner(ME), false).unwrap(), 0);
    }

    /// **A character its owner suspended stays suspended through a thaw of the
    /// whole cast**, whether it was suspended before the freeze or during it —
    /// the owner's word on its state clears what the freeze kept.
    #[test]
    fn an_owner_suspended_character_stays_suspended_through_a_cast_thaw() {
        let n = Npcs::load(&tmp()).unwrap();
        let before = create(&n, ME, "Varek", 1_000);
        let during = create(&n, ME, "Ilse", 1_001);
        let free = create(&n, ME, "Wren", 1_002);
        n.patch(before, ME, &json!({ "state": "suspended" }), 1_500)
            .unwrap();

        assert_eq!(n.freeze_all(Manager::Owner(ME), true).unwrap(), 2);
        assert_eq!(standing(&n, before), st("suspended", None));
        assert_eq!(standing(&n, during), st("suspended", Some("idle")));
        n.patch(during, ME, &json!({ "state": "suspended" }), 2_000)
            .unwrap();
        assert_eq!(standing(&n, during), st("suspended", None));

        assert_eq!(n.freeze_all(Manager::Owner(ME), false).unwrap(), 1);
        assert_eq!(standing(&n, before), st("suspended", None));
        assert_eq!(standing(&n, during), st("suspended", None));
        assert_eq!(standing(&n, free), st("idle", None));
        assert_eq!(n.view().frozen(), HashSet::from([before, during]));
    }

    /// **A freeze of the whole cast reaches exactly the manager's**: an
    /// owner's leaves a stranger's characters running, an admin's reaches
    /// everybody's, and a deleted character is nobody's.
    #[test]
    fn a_cast_freeze_reaches_the_managers_cast() {
        let n = Npcs::load(&tmp()).unwrap();
        let a = create(&n, ME, "Varek", 1_000);
        let b = create(&n, OTHER, "Wren", 1_001);
        let gone = create(&n, OTHER, "Pax", 1_002);
        n.delete(gone, OTHER, 1_003).unwrap();

        assert_eq!(n.freeze_all(Manager::Owner(ME), true).unwrap(), 1);
        assert_eq!(n.view().frozen(), HashSet::from([a]));
        assert_eq!(n.freeze_all(Manager::Admin(ADMIN), true).unwrap(), 1);
        assert_eq!(n.view().frozen(), HashSet::from([a, b]));
        assert_eq!(standing(&n, gone), st("tombstoned", None));

        assert_eq!(n.freeze_all(Manager::Owner(OTHER), false).unwrap(), 1);
        assert_eq!(n.view().frozen(), HashSet::from([a]));
        assert_eq!(n.freeze_all(Manager::Admin(ADMIN), false).unwrap(), 1);
        assert!(n.view().frozen().is_empty());
    }
}
