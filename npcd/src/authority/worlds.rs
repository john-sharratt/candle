//! What this member knows of the worlds it hosts beyond the worlds themselves
//! (`docs/replicated_substrate.md` §9): its role in each, each world's records
//! as it last wrote or applied them, how far it has pulsed each world's events,
//! the handover it waits on, and which characters it runs.
//!
//! A daemon with no cluster is the authority of every world and runs every
//! character; nothing here is consulted beyond that.

use std::collections::{BTreeMap, BTreeSet, HashSet};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, OnceLock};

use cluster::member::AppMember;
use npc_map::world::Tick;

use super::npc_lease::{held_by, leases_of, NpcLease};
use super::role::{Role, Roles};
use super::told::Told;
use crate::world_state::Records;

pub struct Standing {
    clustered: AtomicBool,
    /// Whether this member runs characters (`characters:` in its config). One
    /// that does not runs none, is never a world's authority, and hands back
    /// any lease still naming it (`docs/replicated_substrate.md` §9.5).
    characters: AtomicBool,
    roles: Roles,
    /// When each world took its role, in wall-clock milliseconds.
    since: Mutex<BTreeMap<String, u64>>,
    /// Each world's records as this member last wrote them, as its authority,
    /// or applied them, following it.
    written: Mutex<BTreeMap<String, Records>>,
    /// The substrate's custom-object version each world was last read at.
    followed: Mutex<BTreeMap<String, u64>>,
    /// The last tick of each world's events sent as a pulse.
    pulsed: Mutex<BTreeMap<String, Tick>>,
    /// The handover each world waits on: the member that released its lease,
    /// and through which of its own `oseq`s it wrote.
    waiting: Mutex<BTreeMap<String, (u64, u64)>>,
    /// The member another holds each world's lease by, as this member was
    /// last told on asking for it.
    busy: Mutex<BTreeMap<String, u64>>,
    /// Characters handed back to the authority, by world, to lend on.
    handed: Mutex<BTreeMap<String, BTreeMap<u64, NpcLease>>>,
    /// Deliveries for characters that run elsewhere, gathered over a beat.
    told: Mutex<BTreeMap<String, Vec<Told>>>,
    /// The characters this member runs.
    held: Mutex<BTreeSet<u64>>,
    /// Characters being handed back, which take no turn meanwhile.
    handing: Mutex<HashSet<u64>>,
    /// Characters part-way through a turn: a hand-back waits for its end.
    turning: Mutex<HashSet<u64>>,
    /// Writes on their way to a world's authority, each sent once until it
    /// has answered.
    sending: Mutex<HashSet<String>>,
    /// Worlds taken in this boot from the objects kept before world records:
    /// all of them, and those whose records are not written yet.
    legacy: Mutex<(BTreeSet<String>, BTreeSet<String>)>,
    member: OnceLock<Arc<AppMember>>,
}

impl Standing {
    /// A daemon hosting no world yet: the authority of every world it will
    /// host when `clustered` is false, a replica of each until its lease says
    /// otherwise when it is true.
    pub fn new(clustered: bool) -> Self {
        let standing = Self {
            clustered: AtomicBool::new(false),
            characters: AtomicBool::new(true),
            roles: Roles::new(Role::Authority { epoch: 0 }),
            since: Mutex::new(BTreeMap::new()),
            written: Mutex::new(BTreeMap::new()),
            followed: Mutex::new(BTreeMap::new()),
            pulsed: Mutex::new(BTreeMap::new()),
            waiting: Mutex::new(BTreeMap::new()),
            busy: Mutex::new(BTreeMap::new()),
            handed: Mutex::new(BTreeMap::new()),
            told: Mutex::new(BTreeMap::new()),
            held: Mutex::new(BTreeSet::new()),
            handing: Mutex::new(HashSet::new()),
            turning: Mutex::new(HashSet::new()),
            sending: Mutex::new(HashSet::new()),
            legacy: Mutex::new((BTreeSet::new(), BTreeSet::new())),
            member: OnceLock::new(),
        };
        if clustered {
            standing.set_clustered();
        }
        standing
    }

    pub fn clustered(&self) -> bool {
        self.clustered.load(Ordering::Relaxed)
    }

    /// This daemon is a member of a cluster: every world it hosts is followed
    /// until its lease says otherwise. Set before any world is hosted.
    pub fn set_clustered(&self) {
        self.clustered.store(true, Ordering::Relaxed);
        self.roles.set_default(Role::Replica);
    }

    pub fn characters(&self) -> bool {
        self.characters.load(Ordering::Relaxed)
    }

    /// Whether this member runs characters. Set before any world is hosted.
    pub fn set_characters(&self, characters: bool) {
        self.characters.store(characters, Ordering::Relaxed);
    }

    /// The cluster this member joined.
    pub fn set_member(&self, member: Arc<AppMember>) {
        let _ = self.member.set(member);
    }

    pub fn member(&self) -> Option<&Arc<AppMember>> {
        self.member.get()
    }

    /// This member's node, once it has joined.
    pub fn node(&self) -> Option<u64> {
        self.member.get().map(|m| m.node())
    }

    pub fn role(&self, world: &str) -> Role {
        self.roles.get(world)
    }

    /// Give `world` `role` as of `now_ms`; the role it had.
    pub fn set_role(&self, world: &str, role: Role, now_ms: u64) -> Role {
        let was = self.roles.set(world, role);
        if was.name() != role.name() {
            self.since.lock().unwrap().insert(world.to_string(), now_ms);
        }
        was
    }

    pub fn since(&self, world: &str) -> u64 {
        self.since.lock().unwrap().get(world).copied().unwrap_or(0)
    }

    /// Whether this member writes `world`: it runs it, as its authority or cut
    /// off from it.
    pub fn writes(&self, world: &str) -> bool {
        !matches!(self.role(world), Role::Replica)
    }

    pub fn written(&self, world: &str) -> Records {
        self.written
            .lock()
            .unwrap()
            .get(world)
            .cloned()
            .unwrap_or_default()
    }

    pub fn set_written(&self, world: &str, records: Records) {
        self.written
            .lock()
            .unwrap()
            .insert(world.to_string(), records);
    }

    /// The substrate's version `world` was last read at.
    pub fn followed(&self, world: &str) -> Option<u64> {
        self.followed.lock().unwrap().get(world).copied()
    }

    pub fn set_followed(&self, world: &str, version: u64) {
        self.followed
            .lock()
            .unwrap()
            .insert(world.to_string(), version);
    }

    pub fn pulsed(&self, world: &str) -> Tick {
        self.pulsed.lock().unwrap().get(world).copied().unwrap_or(0)
    }

    pub fn set_pulsed(&self, world: &str, tick: Tick) {
        self.pulsed.lock().unwrap().insert(world.to_string(), tick);
    }

    pub fn waiting(&self, world: &str) -> Option<(u64, u64)> {
        self.waiting.lock().unwrap().get(world).copied()
    }

    pub fn set_waiting(&self, world: &str, on: Option<(u64, u64)>) {
        let mut waiting = self.waiting.lock().unwrap();
        match on {
            Some(on) => waiting.insert(world.to_string(), on),
            None => waiting.remove(world),
        };
    }

    /// The member holding `world`'s lease, as this member was last told.
    pub fn busy(&self, world: &str) -> Option<u64> {
        self.busy.lock().unwrap().get(world).copied()
    }

    pub fn set_busy(&self, world: &str, holder: Option<u64>) {
        let mut busy = self.busy.lock().unwrap();
        match holder {
            Some(h) => busy.insert(world.to_string(), h),
            None => busy.remove(world),
        };
    }

    /// A character handed back, for the authority's next beat to lend on.
    pub fn handed_back(&self, world: &str, npc: u64, lease: NpcLease) {
        self.handed
            .lock()
            .unwrap()
            .entry(world.to_string())
            .or_default()
            .insert(npc, lease);
    }

    /// The characters handed back since the last beat.
    pub fn take_handed(&self, world: &str) -> BTreeMap<u64, NpcLease> {
        self.handed
            .lock()
            .unwrap()
            .remove(world)
            .unwrap_or_default()
    }

    /// Something for a character that runs elsewhere, to go with this beat's
    /// pulse.
    pub fn tell(&self, world: &str, told: Told) {
        self.told
            .lock()
            .unwrap()
            .entry(world.to_string())
            .or_default()
            .push(told);
    }

    pub fn take_told(&self, world: &str) -> Vec<Told> {
        self.told.lock().unwrap().remove(world).unwrap_or_default()
    }

    /// Every world's leases as last written or applied, and which name this
    /// member: the characters it runs. Recomputed whenever a world's records
    /// change. Returns whether the set changed.
    pub fn recompute_held(&self) -> bool {
        let Some(me) = self.node() else {
            return false;
        };
        let now: BTreeSet<u64> = self
            .written
            .lock()
            .unwrap()
            .values()
            .flat_map(|records| held_by(&leases_of(records), me))
            .collect();
        let mut held = self.held.lock().unwrap();
        let changed = *held != now;
        *held = now;
        changed
    }

    /// Whether this member runs character `npc`. A daemon with no cluster
    /// runs them all; one that runs no characters, none.
    pub fn runs(&self, npc: u64) -> bool {
        self.characters() && (!self.clustered() || self.held.lock().unwrap().contains(&npc))
    }

    pub fn held(&self) -> BTreeSet<u64> {
        self.held.lock().unwrap().clone()
    }

    /// Mark `npc` as being handed back; `false` if it already is.
    pub fn begin_handing(&self, npc: u64) -> bool {
        self.handing.lock().unwrap().insert(npc)
    }

    pub fn end_handing(&self, npc: u64) {
        self.handing.lock().unwrap().remove(&npc);
    }

    pub fn handing(&self, npc: u64) -> bool {
        self.handing.lock().unwrap().contains(&npc)
    }

    /// `npc`'s turn began; `false` if it is being handed back and must take
    /// none.
    pub fn begin_turn(&self, npc: u64) -> bool {
        if self.handing(npc) {
            return false;
        }
        self.turning.lock().unwrap().insert(npc);
        true
    }

    pub fn end_turn(&self, npc: u64) {
        self.turning.lock().unwrap().remove(&npc);
    }

    pub fn turning(&self, npc: u64) -> bool {
        self.turning.lock().unwrap().contains(&npc)
    }

    /// Mark the write `key` as on its way; `false` if it already is.
    pub fn begin_sending(&self, key: &str) -> bool {
        self.sending.lock().unwrap().insert(key.to_string())
    }

    pub fn end_sending(&self, key: &str) {
        self.sending.lock().unwrap().remove(key);
    }

    /// Every world given a role, with it.
    pub fn roles(&self) -> BTreeMap<String, Role> {
        self.roles.all()
    }

    /// `world` was taken in from the objects kept before world records.
    pub fn mark_legacy(&self, world: &str) {
        let mut legacy = self.legacy.lock().unwrap();
        legacy.0.insert(world.to_string());
        legacy.1.insert(world.to_string());
    }

    /// Whether `world` was taken in from the old objects this boot: its
    /// characters' places come from their own records, once.
    pub fn from_legacy(&self, world: &str) -> bool {
        self.legacy.lock().unwrap().0.contains(world)
    }

    /// `world`'s records are written: when it was taken in from the old
    /// objects and this is the first time, whether it was the last such world
    /// to be written — the command table's one key goes with the last.
    pub fn legacy_written(&self, world: &str) -> Option<bool> {
        let mut legacy = self.legacy.lock().unwrap();
        legacy.1.remove(world).then(|| legacy.1.is_empty())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A daemon with no cluster runs every world and every character; a
    /// member follows each world until its lease says otherwise.
    #[test]
    fn standalone_runs_everything_and_a_member_follows() {
        let alone = Standing::new(false);
        assert!(alone.writes("w1"));
        assert!(alone.runs(42));
        let member = Standing::new(true);
        assert!(!member.writes("w1"));
        assert!(!member.runs(42));
        member.set_role("w1", Role::Disconnected, 5);
        assert!(member.writes("w1"));
        assert_eq!(member.since("w1"), 5);
    }

    /// **A member that runs no characters runs none**, whatever it would run
    /// otherwise.
    #[test]
    fn a_member_without_characters_runs_none() {
        let w = Standing::new(false);
        assert!(w.characters() && w.runs(42));
        w.set_characters(false);
        assert!(!w.runs(42));
    }

    /// The old objects go once each converted world has written its records,
    /// the shared one with the last of them.
    #[test]
    fn a_converted_world_is_retired_once_and_the_last_takes_the_shared_key() {
        let w = Standing::new(false);
        w.mark_legacy("a");
        w.mark_legacy("b");
        assert!(w.from_legacy("a"));
        assert_eq!(w.legacy_written("a"), Some(false));
        assert_eq!(w.legacy_written("a"), None, "once");
        assert_eq!(w.legacy_written("b"), Some(true));
        assert!(w.from_legacy("a"), "still read for its characters' places");
    }

    /// A character being handed back takes no turn; one mid-turn is known
    /// to be.
    #[test]
    fn a_character_handed_back_takes_no_turn() {
        let w = Standing::new(true);
        assert!(w.begin_turn(1));
        assert!(w.turning(1));
        w.end_turn(1);
        assert!(w.begin_handing(1));
        assert!(!w.begin_handing(1), "once");
        assert!(!w.begin_turn(1));
        w.end_handing(1);
        assert!(w.begin_turn(1));
    }
}
