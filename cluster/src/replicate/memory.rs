//! A [`Store`] held in memory: keyed values under version stamps, the applied
//! set, generations and decisions — the substrate's replication contract
//! without a model, so members can replicate in one process. The harnesses
//! and the tests run members on it.
//!
//! A record is `u64 gen · u64 epoch · u64 hlc · u16 key_len · key · value`,
//! little-endian; its origin and `oseq` travel beside it, as an `EVENT`'s do.
//! A key's value is the record with the greatest stamp `(gen, epoch, hlc,
//! origin)`, the order §6.2 defines, so the same records give the same values
//! in any arrival order.

use std::collections::BTreeMap;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Mutex, MutexGuard};

use applied_set::AppliedSet;

use super::store::{Applied, Disk, Listing, Store, Tap, Tombstones, GEN_DETACHED};
use crate::consensus::{Change, MaxMembers};
use crate::frame::RecordHeader;

const MAX_MEMBERS_SET: &[u8] = b"cluster/max_members/set";
const MAX_MEMBERS_RAISE: &[u8] = b"cluster/max_members/raise/";
const DECISION: &[u8] = b"cluster/";
const GENERATION: &[u8] = b"cluster/gen/";
/// Keys a timeline's turns are written under: `turn/<timeline>/…`. A turn
/// ranked below one held asks for a reload (§6.6).
const TURN: &[u8] = b"turn/";

/// A record's version stamp: `(gen, epoch, hlc, origin)`.
pub type Stamp = (u64, u64, u64, u64);

/// One record, decoded.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MemoryRecord {
    pub gen: u64,
    pub epoch: u64,
    pub hlc: u64,
    pub key: Vec<u8>,
    pub value: Vec<u8>,
}

impl MemoryRecord {
    pub fn encode(&self) -> Vec<u8> {
        let mut b = Vec::with_capacity(26 + self.key.len() + self.value.len());
        b.extend_from_slice(&self.gen.to_le_bytes());
        b.extend_from_slice(&self.epoch.to_le_bytes());
        b.extend_from_slice(&self.hlc.to_le_bytes());
        b.extend_from_slice(&(self.key.len() as u16).to_le_bytes());
        b.extend_from_slice(&self.key);
        b.extend_from_slice(&self.value);
        b
    }

    pub fn decode(b: &[u8]) -> Option<MemoryRecord> {
        let word = |at: usize| Some(u64::from_le_bytes(b.get(at..at + 8)?.try_into().ok()?));
        let key_len = u16::from_le_bytes(b.get(24..26)?.try_into().ok()?) as usize;
        Some(MemoryRecord {
            gen: word(0)?,
            epoch: word(8)?,
            hlc: word(16)?,
            key: b.get(26..26 + key_len)?.to_vec(),
            value: b.get(26 + key_len..)?.to_vec(),
        })
    }

    fn stamp(&self, origin: u64) -> Stamp {
        (self.gen, self.epoch, self.hlc, origin)
    }
}

/// A 16-byte digest of a record's bytes: two FNV-1a passes from different
/// offsets.
pub fn digest(bytes: &[u8]) -> [u8; 16] {
    let fnv = |mut h: u64| {
        for &b in bytes {
            h ^= b as u64;
            h = h.wrapping_mul(0x0000_0100_0000_01b3);
        }
        h
    };
    let mut d = [0u8; 16];
    d[..8].copy_from_slice(&fnv(0xcbf2_9ce4_8422_2325).to_le_bytes());
    d[8..].copy_from_slice(&fnv(0x8422_2325_cbf2_9ce4).to_le_bytes());
    d
}

#[derive(Debug, Default)]
struct Inner {
    /// Every record held, as written, by `(origin, oseq)`.
    records: BTreeMap<(u64, u64), Vec<u8>>,
    /// Other origins' records applied here or known gone.
    applied: AppliedSet,
    next_oseq: u64,
    /// The first `oseq` written cut off, while no rejoin has fixed it.
    detached_from: Option<u64>,
    /// `(from, through, gen)` for what each rejoin fixed.
    rebases: Vec<(u64, u64, u64)>,
    gen: u64,
    hlc: u64,
    epoch: u64,
}

impl Inner {
    fn own_through(&self) -> u64 {
        self.detached_from.unwrap_or(self.next_oseq) - 1
    }

    fn tick(&mut self) -> u64 {
        self.hlc += 1;
        self.hlc
    }

    /// The record as this member reads it back: its own cut-off records with
    /// the generation a rejoin fixed.
    fn fixed(&self, node: u64, origin: u64, oseq: u64, bytes: &[u8]) -> Vec<u8> {
        if origin != node {
            return bytes.to_vec();
        }
        let Some(&(_, _, gen)) = self
            .rebases
            .iter()
            .find(|(from, through, _)| (*from..=*through).contains(&oseq))
        else {
            return bytes.to_vec();
        };
        let mut r = MemoryRecord::decode(bytes).expect("own records decode");
        r.gen = gen;
        r.encode()
    }

    /// Every record held, decoded, with the generation its writer fixed.
    fn decoded(&self, node: u64) -> Vec<(u64, u64, MemoryRecord)> {
        self.records
            .iter()
            .filter_map(|(&(origin, oseq), b)| {
                MemoryRecord::decode(&self.fixed(node, origin, oseq, b)).map(|r| (origin, oseq, r))
            })
            .collect()
    }

    /// The greatest-stamped record under each key matching `pick`.
    fn latest(
        &self,
        node: u64,
        pick: impl Fn(&[u8]) -> bool,
    ) -> BTreeMap<Vec<u8>, (Stamp, Vec<u8>)> {
        let mut out: BTreeMap<Vec<u8>, (Stamp, Vec<u8>)> = BTreeMap::new();
        for (origin, _, r) in self.decoded(node) {
            if !pick(&r.key) {
                continue;
            }
            let stamp = r.stamp(origin);
            match out.get(&r.key) {
                Some((held, _)) if *held >= stamp => {}
                _ => {
                    out.insert(r.key, (stamp, r.value));
                }
            }
        }
        out
    }
}

/// See the module documentation.
pub struct MemoryStore {
    node: u64,
    inner: Mutex<Inner>,
    tap: Mutex<Option<Tap>>,
    reloads: AtomicU64,
    /// The volume it reports, set by the tests.
    disk: Mutex<Disk>,
    /// Whether a test write fails: the volume is full.
    full: AtomicBool,
    /// The record types it reports holding, set by the tests.
    census: Mutex<Vec<u8>>,
    /// Origins whose records it refuses, set by the tests.
    refused_origins: Mutex<Vec<u64>>,
    /// Record types it never takes from another member, set by the tests.
    local_only: Mutex<Vec<u8>>,
    /// How many times the replicator has marked it joined.
    joined_marks: AtomicU64,
}

impl MemoryStore {
    /// How many times the replicator has marked it joined
    /// ([`Store::mark_joined`]).
    pub fn joined_marks(&self) -> u64 {
        self.joined_marks.load(Ordering::Relaxed)
    }

    pub fn new(node: u64) -> MemoryStore {
        MemoryStore {
            node,
            inner: Mutex::new(Inner {
                next_oseq: 1,
                ..Inner::default()
            }),
            tap: Mutex::new(None),
            reloads: AtomicU64::new(0),
            disk: Mutex::new(Disk {
                free: 1 << 40,
                total: 2 << 40,
                write_failed: false,
            }),
            full: AtomicBool::new(false),
            census: Mutex::new(Vec::new()),
            refused_origins: Mutex::new(Vec::new()),
            local_only: Mutex::new(Vec::new()),
            joined_marks: AtomicU64::new(0),
        }
    }

    /// Records of `record_type` never travel to this store, as a substrate's
    /// local-only types do not. Every record it lists is of type `0`.
    pub fn set_local_only(&self, record_type: u8) {
        self.local_only
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .push(record_type);
    }

    /// Refuse every record of `origin` from now on, as a store refuses one it
    /// cannot take — so it is fetched again.
    pub fn refuse_origin(&self, origin: u64) {
        self.refused_origins
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .push(origin);
    }

    /// It holds record types `tags` (§3.9).
    pub fn set_census(&self, tags: Vec<u8>) {
        *self.census.lock().unwrap_or_else(|e| e.into_inner()) = tags;
    }

    /// The volume now has `free` of `total` bytes free; `full` says whether a
    /// write fails for lack of space.
    pub fn set_disk(&self, free: u64, total: u64, full: bool) {
        *self.disk.lock().unwrap_or_else(|e| e.into_inner()) = Disk {
            free,
            total,
            write_failed: full,
        };
        self.full.store(full, Ordering::Relaxed);
    }

    fn inner(&self) -> MutexGuard<'_, Inner> {
        self.inner.lock().unwrap_or_else(|e| e.into_inner())
    }

    fn tap(&self, oseqs: impl IntoIterator<Item = u64>) {
        if let Some(tap) = self.tap.lock().unwrap_or_else(|e| e.into_inner()).as_ref() {
            for oseq in oseqs {
                tap(oseq);
            }
        }
    }

    /// Write `key = value` under the claim epoch `epoch`, returning its
    /// `oseq`.
    pub fn write_under(&self, epoch: u64, key: &[u8], value: &[u8]) -> u64 {
        let (oseq, sendable) = {
            let mut s = self.inner();
            let oseq = s.next_oseq;
            s.next_oseq += 1;
            let gen = if s.detached_from.is_some() {
                GEN_DETACHED
            } else {
                s.gen
            };
            let hlc = s.tick();
            let record = MemoryRecord {
                gen,
                epoch: epoch.max(s.epoch),
                hlc,
                key: key.to_vec(),
                value: value.to_vec(),
            };
            s.records.insert((self.node, oseq), record.encode());
            (oseq, gen != GEN_DETACHED)
        };
        if sendable {
            self.tap([oseq]);
        }
        oseq
    }

    /// Write `key = value`, returning its `oseq`.
    pub fn write(&self, key: &[u8], value: &[u8]) -> u64 {
        self.write_under(0, key, value)
    }

    /// Every record whose key starts with `prefix`, in stamp order, as `(key,
    /// value, origin)`: a timeline's turns, as every member orders them.
    pub fn ordered(&self, prefix: &[u8]) -> Vec<(Vec<u8>, Vec<u8>, u64)> {
        let s = self.inner();
        let mut v: Vec<(Stamp, Vec<u8>, Vec<u8>, u64)> = s
            .decoded(self.node)
            .into_iter()
            .filter(|(_, _, r)| r.key.starts_with(prefix))
            .map(|(origin, _, r)| (r.stamp(origin), r.key, r.value, origin))
            .collect();
        v.sort();
        v.into_iter().map(|(_, k, val, o)| (k, val, o)).collect()
    }

    /// `key`'s value: the record with the greatest stamp.
    pub fn value(&self, key: &[u8]) -> Option<Vec<u8>> {
        self.inner()
            .latest(self.node, |k| k == key)
            .remove(key)
            .map(|(_, v)| v)
    }

    /// What this store holds, as every member must end holding it: each
    /// record's bytes as its writer fixed them, by `(origin, oseq)`.
    pub fn snapshot(&self) -> BTreeMap<(u64, u64), Vec<u8>> {
        let s = self.inner();
        s.records
            .iter()
            .map(|(&(o, q), b)| ((o, q), s.fixed(self.node, o, q, b)))
            .collect()
    }

    /// Drop a record the way compaction drops one superseded: it is no longer
    /// held, and stays applied, so a catch-up lists it as gone.
    pub fn compact_away(&self, origin: u64, oseq: u64) {
        self.inner().records.remove(&(origin, oseq));
    }

    /// Records held.
    pub fn len(&self) -> usize {
        self.inner().records.len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// How many reloads were asked for.
    pub fn reloads(&self) -> u64 {
        self.reloads.load(Ordering::Relaxed)
    }

    /// The epoch every later own write carries at least — a claim taken.
    pub fn set_epoch(&self, epoch: u64) {
        let mut s = self.inner();
        s.epoch = s.epoch.max(epoch);
    }
}

impl Store for MemoryStore {
    fn node(&self) -> u64 {
        self.node
    }

    fn own_through(&self) -> u64 {
        self.inner().own_through()
    }

    fn set_tap(&self, tap: Tap) {
        *self.tap.lock().unwrap_or_else(|e| e.into_inner()) = Some(tap);
    }

    fn read(&self, origin: u64, oseq: u64) -> Option<Vec<u8>> {
        let s = self.inner();
        let bytes = s.records.get(&(origin, oseq))?;
        Some(s.fixed(self.node, origin, oseq, bytes))
    }

    fn applied(&self) -> AppliedSet {
        let s = self.inner();
        let mut set = s.applied.clone();
        let own = s.own_through();
        if own > 0 {
            set.insert_range(self.node, 1, own);
        }
        set
    }

    fn apply(&self, records: &[(u64, u64, Vec<u8>)]) -> Applied {
        let refused = self
            .refused_origins
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .clone();
        let mut s = self.inner();
        let mut out = Applied::default();
        for (origin, oseq, bytes) in records {
            let (origin, oseq) = (*origin, *oseq);
            if refused.contains(&origin) {
                out.taken.push(false);
                continue;
            }
            let Some(r) = MemoryRecord::decode(bytes) else {
                out.taken.push(false);
                continue;
            };
            if r.gen == GEN_DETACHED || origin == self.node {
                out.taken.push(false);
                continue;
            }
            if s.applied.contains(origin, oseq) {
                out.taken.push(true);
                continue;
            }
            if r.key.starts_with(TURN) {
                let stamp = r.stamp(origin);
                let timeline = timeline_of(&r.key);
                let below_held = s.decoded(self.node).iter().any(|(o, _, held)| {
                    held.key.starts_with(TURN)
                        && timeline_of(&held.key) == timeline
                        && held.stamp(*o) > stamp
                });
                out.reload |= below_held;
            }
            if let Some(k) = generation_key(&r.key) {
                s.gen = s.gen.max(k);
            }
            s.gen = s.gen.max(r.gen);
            s.hlc = s.hlc.max(r.hlc);
            s.records.insert((origin, oseq), bytes.clone());
            s.applied.insert(origin, oseq);
            out.taken.push(true);
        }
        out
    }

    fn list(&self, origin: u64, from: u64, to: u64) -> Listing {
        let s = self.inner();
        let mut listing = Listing::default();
        for oseq in from..=to {
            match s.records.get(&(origin, oseq)) {
                Some(b) => {
                    let bytes = s.fixed(self.node, origin, oseq, b);
                    listing.records.push(RecordHeader {
                        oseq,
                        record_type: 0,
                        stream_id: 0,
                        chunk_index: 0,
                        len: bytes.len() as u32,
                        digest: digest(&bytes),
                    });
                }
                None => listing.gone.push(oseq),
            }
        }
        listing
    }

    fn gone(&self, origin: u64, gone: &[u64]) {
        if origin == self.node {
            return;
        }
        let mut s = self.inner();
        for &oseq in gone {
            s.applied.insert(origin, oseq);
        }
    }

    fn travels(&self, record_type: u8) -> bool {
        !self
            .local_only
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .contains(&record_type)
    }

    fn ticket(&self) -> u64 {
        0
    }

    fn wait_written(&self, _ticket: u64) {}

    fn detach(&self) {
        let mut s = self.inner();
        if s.detached_from.is_none() {
            s.detached_from = Some(s.next_oseq);
        }
    }

    fn detached(&self) -> bool {
        let s = self.inner();
        s.detached_from.is_some_and(|from| from < s.next_oseq)
    }

    fn rejoin(&self, gen: u64) {
        let fixed = {
            let mut s = self.inner();
            let Some(from) = s.detached_from.take() else {
                return;
            };
            let through = s.next_oseq - 1;
            s.gen = s.gen.max(gen);
            if from > through {
                return;
            }
            s.rebases.push((from, through, gen));
            from..=through
        };
        self.tap(fixed);
    }

    fn generation(&self) -> u64 {
        let s = self.inner();
        let decided = s
            .latest(self.node, |k| generation_key(k).is_some())
            .keys()
            .filter_map(|k| generation_key(k))
            .max()
            .unwrap_or(0);
        s.gen.max(decided)
    }

    fn record_decision(&self, key: &[u8], value: &[u8]) {
        let mut full = DECISION.to_vec();
        full.extend_from_slice(key);
        if let Some(k) = generation_key(&full) {
            let mut s = self.inner();
            s.gen = s.gen.max(k);
        }
        self.write(&full, value);
    }

    fn decision(&self, key: &[u8]) -> Option<Vec<u8>> {
        let mut full = DECISION.to_vec();
        full.extend_from_slice(key);
        self.value(&full)
    }

    fn max_members(&self) -> u32 {
        let s = self.inner();
        let mut m = MaxMembers::<Stamp>::new();
        let keys = s.latest(self.node, |k| {
            k == MAX_MEMBERS_SET || k.starts_with(MAX_MEMBERS_RAISE)
        });
        for (key, (stamp, value)) in keys {
            let n = value
                .get(..4)
                .map_or(0, |b| u32::from_le_bytes(b.try_into().unwrap()));
            let change = if key == MAX_MEMBERS_SET {
                Change::Set(n)
            } else {
                Change::Raise(n)
            };
            m.apply(stamp, change);
        }
        m.value()
    }

    fn raise_max_members(&self, seen: u32) {
        let mut key = MAX_MEMBERS_RAISE.to_vec();
        key.extend_from_slice(format!("{:016x}", self.node).as_bytes());
        self.write(&key, &seen.to_le_bytes());
    }

    fn set_max_members(&self, n: u32) {
        self.write(MAX_MEMBERS_SET, &n.to_le_bytes());
    }

    fn reload(&self) {
        self.reloads.fetch_add(1, Ordering::Relaxed);
    }

    fn disk(&self) -> Disk {
        *self.disk.lock().unwrap_or_else(|e| e.into_inner())
    }

    fn test_write(&self) -> bool {
        let ok = !self.full.load(Ordering::Relaxed);
        if ok {
            self.disk
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .write_failed = false;
        }
        ok
    }

    fn drop_dead_segment(&self) -> bool {
        false
    }

    fn deferred(&self) -> Vec<(String, u64, usize)> {
        Vec::new()
    }

    fn census(&self) -> Vec<u8> {
        self.census
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .clone()
    }

    fn decisions(&self, prefix: &str) -> Vec<(String, Vec<u8>)> {
        let mut full = DECISION.to_vec();
        full.extend_from_slice(prefix.as_bytes());
        self.inner()
            .latest(self.node, |k| k.starts_with(&full))
            .into_iter()
            .map(|(k, (_, v))| {
                (
                    String::from_utf8_lossy(&k[DECISION.len()..]).into_owned(),
                    v,
                )
            })
            .collect()
    }

    fn tombstones(&self) -> Tombstones {
        Tombstones::default()
    }

    fn mark_joined(&self) {
        self.joined_marks.fetch_add(1, Ordering::Relaxed);
    }
}

/// `k` for a key `cluster/gen/<k>`.
fn generation_key(key: &[u8]) -> Option<u64> {
    std::str::from_utf8(key.strip_prefix(GENERATION)?)
        .ok()?
        .parse()
        .ok()
}

/// The `<timeline>` of a key `turn/<timeline>/…`.
fn timeline_of(key: &[u8]) -> &[u8] {
    let rest = &key[TURN.len()..];
    rest.split(|b| *b == b'/').next().unwrap_or(rest)
}

#[cfg(test)]
mod tests {
    use std::slice::from_ref;

    use super::*;

    #[test]
    fn a_record_encodes_to_exact_bytes() {
        let r = MemoryRecord {
            gen: 1,
            epoch: 2,
            hlc: 3,
            key: b"k".to_vec(),
            value: b"vv".to_vec(),
        };
        let mut want = vec![1, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0];
        want.extend([3, 0, 0, 0, 0, 0, 0, 0, 1, 0, b'k', b'v', b'v']);
        assert_eq!(r.encode(), want);
        assert_eq!(MemoryRecord::decode(&want), Some(r));
        assert_eq!(MemoryRecord::decode(&want[..20]), None);
    }

    /// Two members' writes to one key resolve to the greater stamp in either
    /// arrival order.
    #[test]
    fn the_greatest_stamp_wins_in_any_order() {
        let a = MemoryStore::new(1);
        let b = MemoryStore::new(2);
        a.write(b"label", b"from a");
        b.write(b"x", b"");
        b.write(b"label", b"from b");
        let from_a = (1, 1, a.read(1, 1).unwrap());
        let from_b: Vec<_> = (1..=2).map(|q| (2, q, b.read(2, q).unwrap())).collect();
        let one = MemoryStore::new(3);
        one.apply(from_ref(&from_a));
        one.apply(&from_b);
        let other = MemoryStore::new(4);
        other.apply(&from_b);
        other.apply(&[from_a]);
        assert_eq!(one.value(b"label"), other.value(b"label"));
        assert_eq!(one.snapshot(), other.snapshot());
    }

    /// A cut-off member's writes carry the provisional generation, are not
    /// tapped and are refused by anyone who receives them; a rejoin fixes and
    /// taps them, and they read back under the fixed generation.
    #[test]
    fn cut_off_writes_wait_for_a_rejoin() {
        let s = MemoryStore::new(1);
        let tapped = std::sync::Arc::new(Mutex::new(Vec::new()));
        let t = tapped.clone();
        s.set_tap(Box::new(move |o| t.lock().unwrap().push(o)));
        s.write(b"a", b"1");
        s.detach();
        s.write(b"b", b"2");
        assert!(s.detached());
        assert_eq!(s.own_through(), 1);
        assert_eq!(*tapped.lock().unwrap(), vec![1]);
        let raw = s.inner().records[&(1, 2)].clone();
        assert_eq!(MemoryRecord::decode(&raw).unwrap().gen, GEN_DETACHED);
        let peer = MemoryStore::new(2);
        assert_eq!(peer.apply(&[(1, 2, raw)]).taken, vec![false]);
        s.rejoin(7);
        assert_eq!(*tapped.lock().unwrap(), vec![1, 2]);
        assert_eq!(s.own_through(), 2);
        assert_eq!(MemoryRecord::decode(&s.read(1, 2).unwrap()).unwrap().gen, 7);
        assert_eq!(s.generation(), 7);
    }

    #[test]
    fn a_record_compacted_away_is_listed_as_gone() {
        let s = MemoryStore::new(1);
        for i in 0..3u8 {
            s.write(&[i], b"v");
        }
        s.compact_away(1, 2);
        let l = s.list(1, 1, 3);
        assert_eq!(
            l.records.iter().map(|r| r.oseq).collect::<Vec<_>>(),
            vec![1, 3]
        );
        assert_eq!(l.gone, vec![2]);
    }

    /// `max_members` is raised by sightings and lowered by a set, and a raise
    /// after the set raises it again.
    #[test]
    fn max_members_follows_raises_and_sets() {
        let s = MemoryStore::new(1);
        assert_eq!(s.max_members(), 1);
        s.raise_max_members(3);
        assert_eq!(s.max_members(), 3);
        s.set_max_members(2);
        assert_eq!(s.max_members(), 2);
        s.raise_max_members(4);
        assert_eq!(s.max_members(), 4);
    }

    #[test]
    fn a_decided_generation_is_known() {
        let s = MemoryStore::new(1);
        s.record_decision(b"gen/3", &1u64.to_le_bytes());
        assert_eq!(s.generation(), 3);
        assert_eq!(s.decision(b"gen/3"), Some(1u64.to_le_bytes().to_vec()));
    }

    /// The decisions under a prefix are listed by their keys, nothing else.
    #[test]
    fn decisions_are_listed_by_prefix() {
        let s = MemoryStore::new(1);
        s.record_decision(b"certify/zend/7-abc/1", b"one");
        s.record_decision(b"certify/zend/7-abc/2", b"two");
        s.record_decision(b"certify/npcd/7-abc/1", b"other");
        assert_eq!(
            s.decisions("certify/zend/7-abc/"),
            [
                ("certify/zend/7-abc/1".to_string(), b"one".to_vec()),
                ("certify/zend/7-abc/2".to_string(), b"two".to_vec()),
            ]
        );
        assert!(s.decisions("rollout/").is_empty());
    }

    /// A turn ranked below a turn held asks for a reload; one above does not.
    #[test]
    fn a_turn_ranked_below_one_held_asks_for_a_reload() {
        let a = MemoryStore::new(1);
        let b = MemoryStore::new(2);
        a.write(b"turn/t/1", b"early");
        b.set_epoch(5);
        b.write(b"turn/t/1", b"late");
        let early = (1, 1, a.read(1, 1).unwrap());
        let late = (2, 1, b.read(2, 1).unwrap());
        let c = MemoryStore::new(3);
        assert!(!c.apply(from_ref(&early)).reload);
        assert!(!c.apply(from_ref(&late)).reload);
        let d = MemoryStore::new(4);
        assert!(!d.apply(&[late]).reload);
        assert!(d.apply(&[early]).reload);
    }
}
