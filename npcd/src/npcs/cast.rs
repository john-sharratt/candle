//! One published version of the cast — what every read of it is answered from
//! (`docs/npcd_cast_lock_free.md` §2).
//!
//! A `Cast` never changes once [`super::Npcs`] has published it. A reader holds
//! it as an `Arc` for as long as it likes; a write builds the next one from it
//! on the persistence owner's thread and publishes that instead.

use std::collections::{BTreeMap, HashSet};

use candle_conversation::persistence::identity::RecordIdentity;
use candle_conversation::persistence::record::NpcPayload;
use imbl::{HashMap as ImHashMap, OrdMap};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use web::auth::session::Identity;

use super::{npc_id_wire, wire, Casting, Filter, NpcError};

/// The whole cast, including tombstoned characters.
///
/// Tombstoned ones are kept because their ids must stay taken; every read path
/// filters them out.
#[derive(Clone, Default)]
pub struct Cast {
    /// Ordered by id: [`Self::cast`], [`Self::export_all`] and the roster's
    /// tie-break read in that order.
    pub(super) by_id: OrdMap<u64, NpcPayload>,
    /// The record each character was last read from, so a refresh reads only
    /// what changed since.
    pub(super) read_as: ImHashMap<u64, RecordIdentity>,
}

impl Cast {
    /// Every character this caller may see, newest first.
    ///
    /// Ownership is the filter (§8.2). Hidden characters are excluded unless
    /// asked for, and no total is ever produced — a count of everything you own
    /// is the one figure that gives a hidden character away (§8.3).
    pub fn list(&self, owner: &str, f: &Filter<'_>) -> Vec<Value> {
        let mut rows: Vec<&NpcPayload> = self
            .by_id
            .values()
            .filter(|n| n.owner_id == owner && !n.is_tombstoned())
            .filter(|n| f.include_hidden || !n.hidden)
            .filter(|n| f.matches(n))
            .collect();
        rows.sort_unstable_by(|a, b| {
            b.updated_ms
                .cmp(&a.updated_ms)
                .then(a.npc_id.cmp(&b.npc_id))
        });
        rows.into_iter().map(|n| wire(n, owner)).collect()
    }

    /// How many living characters each authored document has, keyed by slug.
    ///
    /// `pick` selects which reference to count by, so one walk serves both
    /// registries. Every owner's cast is included, because the figure this
    /// answers is a global one: publishing doctrine reaches every character of
    /// that personality, not only the publisher's.
    ///
    /// Hidden characters **are** counted, which is the opposite of what it
    /// first looks like it should be.
    ///
    /// §8.3 says a hidden character must not be deducible. *Excluding* them is
    /// what breaches it: the figure would drop the moment one was hidden, so
    /// anybody polling learns that a character was just hidden and under which
    /// personality — a sharper signal than the roster gives, because the
    /// denominator is smaller. Including them makes hiding invisible here,
    /// which is the whole point of hiding.
    ///
    /// What remains is a global aggregate over every owner. It answers "how
    /// many of these exist" and never "how many do *you* have" — the per-owner
    /// total is the one §8.3 forbids, and [`Self::list`] still refuses to
    /// produce it.
    pub fn counts_by<'a>(
        &'a self,
        pick: fn(&'a NpcPayload) -> &'a str,
    ) -> BTreeMap<&'a str, usize> {
        let mut out = BTreeMap::new();
        for n in self.by_id.values() {
            if n.is_tombstoned() {
                continue;
            }
            *out.entry(pick(n)).or_insert(0) += 1;
        }
        out
    }

    /// One character, if this caller may see it.
    ///
    /// A character owned by somebody else reads as absent rather than
    /// forbidden. A 403 would confirm the id exists, which is enough to
    /// enumerate a stranger's cast one guess at a time.
    pub fn visible_to(&self, npc_id: u64, owner: &str) -> Option<&NpcPayload> {
        self.by_id
            .get(&npc_id)
            .filter(|n| n.owner_id == owner && !n.is_tombstoned())
    }

    /// Whether this id is a living character, whoever owns it.
    pub fn is_living(&self, npc_id: u64) -> bool {
        self.by_id.get(&npc_id).is_some_and(|n| !n.is_tombstoned())
    }

    pub fn get(&self, npc_id: u64, owner: &str) -> Result<Value, NpcError> {
        self.visible_to(npc_id, owner)
            .map(|n| wire(n, owner))
            .ok_or(NpcError::NotFound)
    }

    /// Every living character, with the world it belongs to.
    ///
    /// **Deliberately not filtered by owner.** The tick scheduler runs the whole
    /// cast — the quartermaster counts sacks whether or not the person who
    /// authored him is signed in — so this is the one read that crosses
    /// ownership. Every route that reaches a character on a user's behalf still
    /// goes through [`Self::visible_to`]; this is for the engine, which serves
    /// the world rather than a caller.
    pub fn cast(&self) -> Vec<Casting> {
        self.by_id
            .values()
            .filter(|n| !n.is_tombstoned())
            .map(|n| Casting {
                npc_id: n.npc_id,
                world_id: n.world_id.clone(),
                at: n.at.clone(),
                mood: n.mood.clone(),
            })
            .collect()
    }

    /// The world a character belongs to, whoever owns it. Used by the tick
    /// driver to resolve which clock a character lives on.
    pub fn world_of(&self, npc_id: u64) -> Option<&str> {
        self.by_id
            .get(&npc_id)
            .filter(|n| !n.is_tombstoned())
            .map(|n| n.world_id.as_str())
    }

    /// The ids of every living character this account owns.
    ///
    /// A set, because the caller is testing membership per tick record and a
    /// linear scan over a large cast per row is the kind of cost that only shows
    /// up once somebody has two hundred characters.
    pub fn owned_by(&self, owner: &str) -> HashSet<u64> {
        self.by_id
            .values()
            .filter(|n| !n.is_tombstoned() && n.owner_id == owner)
            .map(|n| n.npc_id)
            .collect()
    }

    /// A living character's record, whoever owns it.
    ///
    /// Ownership-blind for the same reason [`Self::cast`] is: the engine serves
    /// the world rather than a caller. Every route that reaches a character on a
    /// user's behalf still goes through [`Self::visible_to`].
    pub fn payload(&self, npc_id: u64) -> Option<&NpcPayload> {
        self.by_id.get(&npc_id).filter(|n| !n.is_tombstoned())
    }

    /// The register a character last said it was in.
    ///
    /// What the persona reads back to it on the next turn, so a mood survives a
    /// restart.
    pub fn mood_of(&self, npc_id: u64) -> Option<&str> {
        self.by_id
            .get(&npc_id)
            .filter(|n| !n.is_tombstoned())
            .and_then(|n| n.mood.as_deref())
    }

    /// The image every character's portrait names — what a drawn picture's
    /// retirement leaves alone.
    pub fn portrait_ids(&self) -> HashSet<String> {
        self.by_id
            .values()
            .filter_map(|n| n.portrait_image_id.clone())
            .collect()
    }

    /// Every character record, live and tombstoned, for a backup.
    ///
    /// Tombstoned ones travel too, so a restore keeps the whole id-space taken —
    /// an id that was retired must not be minted again for a different character
    /// after a wipe-and-restore.
    pub fn export_all(&self) -> Vec<NpcPayload> {
        self.by_id.values().cloned().collect()
    }

    /// Every living character and the account it belongs to, by name — what an
    /// admin reassigning characters chooses from.
    pub fn owners(&self) -> Vec<Value> {
        let mut rows: Vec<&NpcPayload> =
            self.by_id.values().filter(|n| !n.is_tombstoned()).collect();
        rows.sort_by(|a, b| a.name.cmp(&b.name).then(a.npc_id.cmp(&b.npc_id)));
        rows.into_iter()
            .map(|n| {
                json!({
                    "npc_id": npc_id_wire(n.npc_id),
                    "name": n.name,
                    "world_id": n.world_id,
                    "owner_id": n.owner_id,
                })
            })
            .collect()
    }

    /// The character, if the caller owns it. Every authoring write starts here:
    /// ownership is authorization (§8.2), and a role cannot express "yours".
    pub(super) fn owned(&self, npc_id: u64, owner: &str) -> Result<NpcPayload, NpcError> {
        let npc = self.visible_to(npc_id, owner).ok_or(NpcError::NotFound)?;
        if npc.owner_id != owner {
            return Err(NpcError::NotFound);
        }
        Ok(npc.clone())
    }

    /// A fresh id, unique across the whole cast including tombstoned ones.
    ///
    /// Derived rather than sequential: a sequential id leaks how many
    /// characters exist and in what order they were made, which is the §8.3
    /// enumeration problem by another route. The hash is not a secret — the
    /// subject and name are known to their owner — it is only a spread.
    pub(super) fn mint_id(&self, id: &Identity, name: &str, now_ms: u64) -> u64 {
        for salt in 0u32.. {
            let mut h = Sha256::new();
            h.update(id.sub.as_bytes());
            h.update(name.as_bytes());
            h.update(now_ms.to_le_bytes());
            h.update(salt.to_le_bytes());
            let d = h.finalize();
            let v = u64::from_le_bytes(d[..8].try_into().expect("32-byte digest"));
            // Never zero: zero is the header's "no stream" value, and an id that
            // collides with an absence is a debugging session nobody needs.
            if v != 0 && !self.by_id.contains_key(&v) {
                return v;
            }
        }
        unreachable!("u32 salts exhausted against a u64 keyspace")
    }
}

/// A fresh child id for a `break_down` of `parent` on `npc`, `<parent>-<n>`
/// where `n` starts at how many children the parent already has and advances
/// past any id already taken by a *different* strategy — a directly-authored
/// `widget-0` with no parent, say. Without that check the guessed
/// `<parent>-<n>` could name an unrelated existing strategy, and the write's
/// create-or-replace would silently overwrite and reparent it.
pub(super) fn child_id(npc: &NpcPayload, parent: &str) -> String {
    let taken: HashSet<&str> = npc.agency.iter().map(|s| s.strategy_id.as_str()).collect();
    let mut n = npc
        .agency
        .iter()
        .filter(|s| s.parent_id.as_deref() == Some(parent))
        .count();
    let mut candidate = format!("{parent}-{n}");
    while taken.contains(candidate.as_str()) {
        n += 1;
        candidate = format!("{parent}-{n}");
    }
    candidate
}
