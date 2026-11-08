//! A [`MemoryStore`] whose every [`Store::own_through`] waits on a write made
//! meanwhile on another thread — the way a persistent store's answer waits on
//! its owner thread, which taps the replicator for each record it writes. A
//! caller holding the replicator's state across the call deadlocks against
//! that tap, as it did against the live store.

use std::sync::{Arc, Mutex};

use applied_set::AppliedSet;

use cluster::replicate::{Applied, Disk, Listing, MemoryStore, Store, Tombstones};

type SharedTap = Arc<dyn Fn(u64) + Send + Sync>;

pub struct TappingWhileAsked {
    pub inner: Arc<MemoryStore>,
    tap: Mutex<Option<SharedTap>>,
}

impl TappingWhileAsked {
    pub fn new(node: u64) -> Arc<TappingWhileAsked> {
        Arc::new(TappingWhileAsked {
            inner: Arc::new(MemoryStore::new(node)),
            tap: Mutex::new(None),
        })
    }
}

impl Store for TappingWhileAsked {
    fn node(&self) -> u64 {
        self.inner.node()
    }
    fn own_through(&self) -> u64 {
        let through = self.inner.own_through();
        let tap = self.tap.lock().unwrap().clone();
        if let (Some(tap), true) = (tap, through > 0) {
            std::thread::spawn(move || tap(through)).join().unwrap();
        }
        through
    }
    fn set_tap(&self, tap: Box<dyn Fn(u64) + Send + Sync>) {
        let tap: SharedTap = Arc::from(tap);
        *self.tap.lock().unwrap() = Some(tap.clone());
        self.inner.set_tap(Box::new(move |oseq| tap(oseq)))
    }
    fn read(&self, origin: u64, oseq: u64) -> Option<Vec<u8>> {
        self.inner.read(origin, oseq)
    }
    fn applied(&self) -> AppliedSet {
        self.inner.applied()
    }
    fn apply(&self, records: &[(u64, u64, Vec<u8>)]) -> Applied {
        self.inner.apply(records)
    }
    fn frees(&self, bytes: &[u8]) -> bool {
        self.inner.frees(bytes)
    }
    fn list(&self, origin: u64, from: u64, to: u64) -> Listing {
        self.inner.list(origin, from, to)
    }
    fn gone(&self, origin: u64, gone: &[u64]) {
        self.inner.gone(origin, gone)
    }
    fn travels(&self, record_type: u8) -> bool {
        self.inner.travels(record_type)
    }
    fn ticket(&self) -> u64 {
        self.inner.ticket()
    }
    fn wait_written(&self, ticket: u64) {
        self.inner.wait_written(ticket)
    }
    fn detach(&self) {
        self.inner.detach()
    }
    fn detached(&self) -> bool {
        self.inner.detached()
    }
    fn rejoin(&self, gen: u64) {
        self.inner.rejoin(gen)
    }
    fn generation(&self) -> u64 {
        self.inner.generation()
    }
    fn record_decision(&self, key: &[u8], value: &[u8]) {
        self.inner.record_decision(key, value)
    }
    fn decision(&self, key: &[u8]) -> Option<Vec<u8>> {
        self.inner.decision(key)
    }
    fn max_members(&self) -> u32 {
        self.inner.max_members()
    }
    fn raise_max_members(&self, seen: u32) {
        self.inner.raise_max_members(seen)
    }
    fn set_max_members(&self, n: u32) {
        self.inner.set_max_members(n)
    }
    fn reload(&self) {
        self.inner.reload()
    }
    fn disk(&self) -> Disk {
        self.inner.disk()
    }
    fn test_write(&self) -> bool {
        self.inner.test_write()
    }
    fn drop_dead_segment(&self) -> bool {
        self.inner.drop_dead_segment()
    }
    fn deferred(&self) -> Vec<(String, u64, usize)> {
        self.inner.deferred()
    }
    fn census(&self) -> Vec<u8> {
        self.inner.census()
    }
    fn decisions(&self, prefix: &str) -> Vec<(String, Vec<u8>)> {
        self.inner.decisions(prefix)
    }
    fn tombstones(&self) -> Tombstones {
        self.inner.tombstones()
    }
    fn mark_joined(&self) {
        self.inner.mark_joined()
    }
    fn note_busy(&self, busy: bool) {
        self.inner.note_busy(busy)
    }
}
