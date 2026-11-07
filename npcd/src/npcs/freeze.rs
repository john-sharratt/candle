//! Freezing characters: the one write that stops a character taking turns of
//! any kind, and starts it again.
//!
//! A freeze is a superseding record like every other write to the cast, so it
//! is durable before it is published, survives a restart, and replicates to
//! every member with the record. It is operational rather than authored — it
//! moves no revision and no `updated_ms`, for the reason a mood does not: a
//! roster sorted by "recently edited" would otherwise jump every time somebody
//! freezes the cast to free the card.

use candle_conversation::persistence::record::NpcPayload;
use serde_json::Value;

use super::{one, wire, NpcError, Npcs};

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
            Manager::Owner(id) | Manager::Admin(id) => *id,
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
    /// Freeze or unfreeze one character, and answer with it as it now stands.
    ///
    /// A character the manager may not manage reads as absent, as every other
    /// write's does. Setting what is already set writes nothing.
    pub fn freeze(&self, npc_id: u64, by: Manager<'_>, frozen: bool) -> Result<Value, NpcError> {
        self.write(|cast| {
            let npc = cast
                .by_id
                .get(&npc_id)
                .filter(|n| by.manages(n))
                .ok_or(NpcError::NotFound)?;
            if npc.frozen == frozen {
                return Ok((Vec::new(), wire(npc, by.caller())));
            }
            let mut npc = npc.clone();
            npc.frozen = frozen;
            Ok(one(npc, by.caller()))
        })
    }

    /// Freeze or unfreeze every character the manager may manage, in one
    /// write behind one fsync. Returns how many changed; the ones already so
    /// are left alone and not counted.
    pub fn freeze_all(&self, by: Manager<'_>, frozen: bool) -> Result<usize, NpcError> {
        self.write(|cast| {
            let changed: Vec<NpcPayload> = cast
                .by_id
                .values()
                .filter(|n| by.manages(n) && n.frozen != frozen)
                .cloned()
                .map(|mut n| {
                    n.frozen = frozen;
                    n
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

    /// **A freeze survives a restart, and so does the thaw.** Written by one
    /// registry, read back by another over the same log: the record is the
    /// whole of where the state lives.
    #[test]
    fn a_freeze_survives_a_restart() {
        let dir = tmp();
        let (varek, ilse) = {
            let n = Npcs::load(&dir).unwrap();
            let varek = create(&n, ME, "Varek", 1_000);
            let ilse = create(&n, ME, "Ilse", 1_001);
            let v = n.freeze(varek, Manager::Owner(ME), true).unwrap();
            assert_eq!(v["frozen"], true);
            assert_eq!(n.view().frozen(), HashSet::from([varek]));
            (varek, ilse)
        };

        let n = Npcs::load(&dir).unwrap();
        assert_eq!(n.view().frozen(), HashSet::from([varek]));
        assert_eq!(n.view().get(varek, ME).unwrap()["frozen"], true);
        assert_eq!(n.view().get(ilse, ME).unwrap()["frozen"], false);

        n.freeze(varek, Manager::Owner(ME), false).unwrap();
        drop(n);
        let back = Npcs::load(&dir).unwrap().view();
        assert!(
            back.frozen().is_empty(),
            "the thaw did not survive the restart"
        );
    }

    /// Freezing is not an edit: no revision, no `updated_ms`, and setting what
    /// is already set writes no record at all.
    #[test]
    fn freezing_is_not_an_edit() {
        let n = Npcs::load(&tmp()).unwrap();
        let id = create(&n, ME, "Varek", 1_000);
        let before = n.view().payload(id).unwrap().clone();

        n.freeze(id, Manager::Owner(ME), true).unwrap();
        let view = n.view();
        let after = view.payload(id).unwrap();
        assert!(after.frozen);
        assert_eq!(after.revision, before.revision);
        assert_eq!(after.updated_ms, before.updated_ms);

        let identity = view.read_as.get(&id).cloned();
        n.freeze(id, Manager::Owner(ME), true).unwrap();
        assert_eq!(
            n.view().read_as.get(&id).cloned(),
            identity,
            "freezing a frozen character wrote a record"
        );
    }

    /// **An owner freezes their own, and an admin freezes anybody's.** A
    /// stranger's character reads as absent to an owner, as every write's does.
    #[test]
    fn an_owner_freezes_their_own_and_an_admin_anybodys() {
        let n = Npcs::load(&tmp()).unwrap();
        let mine = create(&n, ME, "Varek", 1_000);
        let theirs = create(&n, OTHER, "Ilse", 1_001);

        assert!(matches!(
            n.freeze(theirs, Manager::Owner(ME), true),
            Err(NpcError::NotFound)
        ));
        assert!(n.view().frozen().is_empty());

        let v = n.freeze(theirs, Manager::Admin(ADMIN), true).unwrap();
        assert_eq!(v["frozen"], true);
        assert_eq!(v["access"], "viewer");
        assert_eq!(n.view().frozen(), HashSet::from([theirs]));
        n.freeze(mine, Manager::Owner(ME), true).unwrap();
        assert_eq!(n.view().frozen(), HashSet::from([mine, theirs]));
    }

    /// **Freezing all of them reaches exactly the manager's cast, and counts
    /// only what changed.** An owner's sweep leaves a stranger's characters
    /// running; an admin's reaches everybody's; a deleted character is nobody's.
    #[test]
    fn freezing_all_reaches_the_managers_cast_and_counts_the_changes() {
        let n = Npcs::load(&tmp()).unwrap();
        let a = create(&n, ME, "Varek", 1_000);
        let b = create(&n, ME, "Ilse", 1_001);
        let c = create(&n, OTHER, "Wren", 1_002);
        let gone = create(&n, OTHER, "Pax", 1_003);
        n.delete(gone, OTHER, 1_004).unwrap();
        n.freeze(a, Manager::Owner(ME), true).unwrap();

        assert_eq!(n.freeze_all(Manager::Owner(ME), true).unwrap(), 1);
        assert_eq!(n.view().frozen(), HashSet::from([a, b]));
        assert_eq!(n.freeze_all(Manager::Owner(ME), true).unwrap(), 0);

        assert_eq!(n.freeze_all(Manager::Admin(ADMIN), true).unwrap(), 1);
        assert_eq!(n.view().frozen(), HashSet::from([a, b, c]));
        assert!(!n.view().by_id.get(&gone).unwrap().frozen);

        assert_eq!(n.freeze_all(Manager::Owner(OTHER), false).unwrap(), 1);
        assert_eq!(n.view().frozen(), HashSet::from([a, b]));
        assert_eq!(n.freeze_all(Manager::Admin(ADMIN), false).unwrap(), 2);
        assert!(n.view().frozen().is_empty());
    }
}
