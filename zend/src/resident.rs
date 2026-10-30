//! The conversations a member holds in memory, by key, with no lock.
//!
//! A request for a conversation that is not resident forks it from the
//! substrate — a scheduler round-trip. Behind one map mutex, that fork held
//! every other request for any conversation until it returned. Here each key
//! owns a cell: the first request for a key inserts it and forks into it,
//! every other request for the same key awaits that cell, and a request for a
//! different key waits on nothing.
//!
//! # Releasing without a lock
//!
//! A conversation is *held* while anything beyond the map has a handle to it
//! — a turn running on it — and *being acquired* while a request is inside
//! [`Resident::get_or_try_init`] for it. The idle sweep must release only one
//! that is neither, and a request may take a handle at any moment, so the
//! sweep's "nothing holds it" and a request's "it is still resident" are the
//! two halves of one handshake:
//!
//! - a request announces itself (acquiring, or by the handle it took), then
//!   confirms the entry is still resident and not retiring, and reads the map
//!   again when it is not;
//! - the sweep marks the entry retiring, then counts handles and acquirers,
//!   and either clears the mark (something holds it) or removes the entry.
//!
//! Both halves fence (`SeqCst`) between their step and their check, so one of
//! them always sees the other: a request that kept its handle was counted, and
//! a removed entry has no request holding it. The entry stays in the map until
//! the count decides, so no second fork can start under the key while a
//! request still holds the first.
//!
//! # Letting go of what is released
//!
//! The map frees an entry it has removed only in batches, once enough have
//! been retired — so a removed entry still holding its value would keep the
//! conversation, its scheduler slot and its resident KV with it, alive long
//! after it was released. Each entry's value is therefore a slot, emptied as
//! the entry leaves the map: the released value goes when its last holder
//! lets it go, and the map's retired copy holds nothing.

use std::future::Future;
use std::sync::atomic::{fence, AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use arc_swap::ArcSwapOption;
use tokio::sync::OnceCell;

use crate::conv_sweep::{self, Sweep};

/// One key's cell: the value once forked, and when it was last used.
struct Entry<T> {
    /// The fork, once made: a slot that holds it until the entry is released.
    value: OnceCell<ArcSwapOption<T>>,
    /// Nanoseconds after the map's `base` at which a request last used it.
    used: AtomicU64,
    /// Requests inside `get_or_try_init` for this entry now.
    acquiring: AtomicUsize,
    /// The sweep is deciding whether to release it — no request may keep a
    /// handle taken now.
    retiring: AtomicBool,
}

impl<T> Entry<T> {
    fn new(stamp: u64) -> Self {
        Self {
            value: OnceCell::new(),
            used: AtomicU64::new(stamp),
            acquiring: AtomicUsize::new(0),
            retiring: AtomicBool::new(false),
        }
    }

    /// The value, while the entry holds one.
    fn loaded(&self) -> Option<Arc<T>> {
        self.value.get()?.load_full()
    }

    /// Nothing beyond the map holds it and no request is acquiring it, judged
    /// through `value`, a handle loaded from the entry's slot: the slot's own
    /// and that one are the only two.
    fn unheld(&self, value: &Arc<T>) -> bool {
        self.acquiring.load(Ordering::SeqCst) == 0 && Arc::strong_count(value) == 2
    }

    /// Empty the slot as the entry leaves the map, returning what it held.
    fn release(&self) -> Option<Arc<T>> {
        self.value.get()?.swap(None)
    }
}

/// A request inside `get_or_try_init` for an entry, for as long as it is —
/// returned, failed or cancelled.
struct Acquiring<'a>(&'a AtomicUsize);

impl<'a> Acquiring<'a> {
    fn start(count: &'a AtomicUsize) -> Self {
        count.fetch_add(1, Ordering::SeqCst);
        Self(count)
    }
}

impl Drop for Acquiring<'_> {
    fn drop(&mut self) {
        self.0.fetch_sub(1, Ordering::SeqCst);
    }
}

/// Resident values by key — see the module docs.
pub struct Resident<T> {
    entries: papaya::HashMap<String, Arc<Entry<T>>>,
    /// The clock `used` stamps count from.
    base: Instant,
}

impl<T> Default for Resident<T> {
    fn default() -> Self {
        Self {
            entries: papaya::HashMap::new(),
            base: Instant::now(),
        }
    }
}

impl<T> Resident<T> {
    fn stamp(&self, at: Instant) -> u64 {
        at.saturating_duration_since(self.base).as_nanos() as u64
    }

    fn instant(&self, stamp: u64) -> Instant {
        self.base + Duration::from_nanos(stamp)
    }

    /// Whether a handle this thread took from `entry` may be kept: `key` still
    /// names it and no sweep is retiring it.
    fn confirm(&self, key: &str, entry: &Arc<Entry<T>>) -> bool {
        fence(Ordering::SeqCst);
        !entry.retiring.load(Ordering::SeqCst)
            && self
                .entries
                .pin()
                .get(key)
                .is_some_and(|e| Arc::ptr_eq(e, entry))
    }

    /// The value resident under `key`, if one is.
    pub fn get(&self, key: &str) -> Option<Arc<T>> {
        loop {
            let entry = Arc::clone(self.entries.pin().get(key)?);
            let value = entry.loaded()?;
            if self.confirm(key, &entry) {
                return Some(value);
            }
            drop(value);
            std::thread::yield_now();
        }
    }

    /// The value under `key`, forked by `init` when none is resident, and
    /// whether this call is the one that forked it. Used now, at `now`.
    ///
    /// `init` runs at most once per resident entry, outside any lock; every
    /// other caller for the same key awaits its result, and is handed the
    /// value only once `init` returns — so whatever `init` does to the value
    /// first (lock it, say) happens before anyone else can reach it. A failed
    /// `init` leaves the key unforked, and the next caller tries again.
    pub async fn get_or_try_init<F, Fut, E>(
        &self,
        key: &str,
        now: Instant,
        mut init: F,
    ) -> Result<(Arc<T>, bool), E>
    where
        F: FnMut() -> Fut,
        Fut: Future<Output = Result<Arc<T>, E>>,
    {
        let stamp = self.stamp(now);
        loop {
            let entry = Arc::clone(
                self.entries
                    .pin()
                    .get_or_insert_with(key.to_string(), || Arc::new(Entry::new(stamp))),
            );
            let acquiring = Acquiring::start(&entry.acquiring);
            entry.used.fetch_max(stamp, Ordering::Relaxed);
            let mut minted = false;
            let slot = entry
                .value
                .get_or_try_init(|| async {
                    minted = true;
                    init().await.map(|v| ArcSwapOption::from(Some(v)))
                })
                .await?;
            // Empty: released by a `remove` since it was forked. The key forks
            // again under a fresh entry.
            let Some(value) = slot.load_full() else {
                drop(acquiring);
                tokio::task::yield_now().await;
                continue;
            };
            if self.confirm(key, &entry) {
                return Ok((value, minted));
            }
            // While this call was acquiring, no sweep could retire the entry
            // for it, so a value it forked left the map by a `remove`, or as
            // an unforked entry the sweep took just as this fork began. It
            // goes back under the key — its slot holding it again — unless
            // another entry has taken the key, which then wins and this fork
            // is dropped.
            if minted && !entry.retiring.load(Ordering::SeqCst) {
                slot.store(Some(Arc::clone(&value)));
                if self
                    .entries
                    .pin()
                    .try_insert(key.to_string(), Arc::clone(&entry))
                    .is_ok()
                {
                    return Ok((value, minted));
                }
                slot.store(None);
            }
            drop(acquiring);
            drop(value);
            tokio::task::yield_now().await;
        }
    }

    /// Stamp `key` used at `now`, if it is resident.
    pub fn touch(&self, key: &str, now: Instant) {
        if let Some(entry) = self.entries.pin().get(key) {
            entry.used.fetch_max(self.stamp(now), Ordering::Relaxed);
        }
    }

    /// Release `key` whatever holds it, returning its value. A holder keeps its
    /// own handle; the next request forks the key again.
    pub fn remove(&self, key: &str) -> Option<Arc<T>> {
        let entry = Arc::clone(self.entries.pin().remove(key)?);
        entry.release()
    }

    /// Release every value nothing has used for `ttl` and nothing holds,
    /// returning what went. A held value is in use now, so its clock restarts.
    pub fn sweep(&self, now: Instant, ttl: Duration) -> Vec<(String, Arc<T>)> {
        let mut released = Vec::new();
        let candidates: Vec<(String, Arc<Entry<T>>)> = self
            .entries
            .pin()
            .iter()
            .map(|(k, e)| (k.clone(), Arc::clone(e)))
            .collect();
        for (key, entry) in candidates {
            if entry.acquiring.load(Ordering::SeqCst) > 0 {
                continue;
            }
            let last = self.instant(entry.used.load(Ordering::Relaxed));
            let Some(slot) = entry.value.get() else {
                // A fork that failed and was never tried again.
                if conv_sweep::sweep(last, now, ttl, false) == Sweep::Release {
                    let _ = self.entries.pin().remove_if(&key, |_, e| {
                        Arc::ptr_eq(e, &entry)
                            && e.value.get().is_none()
                            && e.acquiring.load(Ordering::SeqCst) == 0
                    });
                }
                continue;
            };
            let Some(value) = slot.load_full() else {
                // Released by a `remove` while this sweep was collecting.
                continue;
            };
            match conv_sweep::sweep(last, now, ttl, !entry.unheld(&value)) {
                Sweep::Keep => {}
                Sweep::Touch => {
                    entry.used.fetch_max(self.stamp(now), Ordering::Relaxed);
                }
                Sweep::Release => {
                    if entry
                        .retiring
                        .compare_exchange(false, true, Ordering::SeqCst, Ordering::SeqCst)
                        .is_err()
                    {
                        continue;
                    }
                    fence(Ordering::SeqCst);
                    if !entry.unheld(&value) {
                        // A request took it after the first count: it is in
                        // use, so it stays resident.
                        entry.used.fetch_max(self.stamp(now), Ordering::Relaxed);
                        entry.retiring.store(false, Ordering::SeqCst);
                        continue;
                    }
                    let removed = self
                        .entries
                        .pin()
                        .remove_if(&key, |_, e| Arc::ptr_eq(e, &entry))
                        .is_ok_and(|r| r.is_some());
                    if removed {
                        entry.release();
                        released.push((key, value));
                    }
                }
            }
        }
        released
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    async fn ready(v: u32) -> Result<Arc<u32>, ()> {
        Ok(Arc::new(v))
    }

    fn runtime() -> tokio::runtime::Runtime {
        tokio::runtime::Builder::new_multi_thread()
            .worker_threads(4)
            .enable_all()
            .build()
            .unwrap()
    }

    /// **The first request forks; the next one is handed the same value.**
    #[test]
    fn a_key_is_forked_once_and_then_reused() {
        let rt = runtime();
        let map = Resident::<u32>::default();
        let now = Instant::now();
        let (a, minted_a) = rt
            .block_on(map.get_or_try_init("c", now, || ready(7)))
            .unwrap();
        let (b, minted_b) = rt
            .block_on(map.get_or_try_init("c", now, || ready(9)))
            .unwrap();
        assert_eq!((*a, minted_a), (7, true));
        assert_eq!((*b, minted_b), (7, false));
        assert!(Arc::ptr_eq(&a, &b));
        assert_eq!(map.get("c").as_deref(), Some(&7));
    }

    /// **Concurrent requests for one key fork it once**, and every one of them
    /// is handed the value that fork made.
    #[test]
    fn concurrent_requests_for_one_key_fork_it_once() {
        let rt = runtime();
        let map = Arc::new(Resident::<u32>::default());
        let forks = Arc::new(AtomicUsize::new(0));
        let tasks: Vec<_> = (0..32u32)
            .map(|i| {
                let (map, forks) = (Arc::clone(&map), Arc::clone(&forks));
                rt.spawn(async move {
                    map.get_or_try_init("c", Instant::now(), || {
                        let forks = Arc::clone(&forks);
                        async move {
                            forks.fetch_add(1, Ordering::SeqCst);
                            tokio::task::yield_now().await;
                            Ok::<_, ()>(Arc::new(i))
                        }
                    })
                    .await
                    .unwrap()
                })
            })
            .collect();
        let got: Vec<(Arc<u32>, bool)> = rt.block_on(async {
            let mut out = Vec::new();
            for t in tasks {
                out.push(t.await.unwrap());
            }
            out
        });
        assert_eq!(forks.load(Ordering::SeqCst), 1);
        assert_eq!(got.iter().filter(|(_, minted)| *minted).count(), 1);
        assert!(got.iter().all(|(v, _)| Arc::ptr_eq(v, &got[0].0)));
    }

    /// **A failed fork leaves the key unforked**, and the next request forks it.
    #[test]
    fn a_failed_fork_is_tried_again() {
        let rt = runtime();
        let map = Resident::<u32>::default();
        let now = Instant::now();
        let failed =
            rt.block_on(map.get_or_try_init("c", now, || async { Err::<Arc<u32>, &str>("no") }));
        assert_eq!(failed.err(), Some("no"));
        assert!(map.get("c").is_none());
        let (v, minted) = rt
            .block_on(map.get_or_try_init("c", now, || async { Ok::<_, &str>(Arc::new(3)) }))
            .unwrap();
        assert_eq!((*v, minted), (3, true));
    }

    /// **The sweep releases what is idle and unheld, and keeps what is held**,
    /// restarting a held one's clock.
    #[test]
    fn the_sweep_releases_only_what_nothing_holds() {
        let rt = runtime();
        let map = Resident::<u32>::default();
        let t0 = Instant::now();
        let ttl = Duration::from_secs(60);
        rt.block_on(map.get_or_try_init("idle", t0, || ready(1)))
            .unwrap();
        let (held, _) = rt
            .block_on(map.get_or_try_init("held", t0, || ready(2)))
            .unwrap();
        rt.block_on(map.get_or_try_init("fresh", t0 + ttl, || ready(3)))
            .unwrap();

        let released = map.sweep(t0 + ttl, ttl);
        assert_eq!(
            released
                .iter()
                .map(|(k, v)| (k.as_str(), **v))
                .collect::<Vec<_>>(),
            vec![("idle", 1)]
        );
        assert!(map.get("idle").is_none());
        assert_eq!(map.get("held").as_deref(), Some(&2));
        assert_eq!(map.get("fresh").as_deref(), Some(&3));

        // The held one's clock restarted at the sweep: released once dropped
        // and a whole TTL has passed since.
        drop(held);
        assert!(map.sweep(t0 + ttl + ttl / 2, ttl).is_empty());
        let released = map.sweep(t0 + 2 * ttl, ttl);
        assert_eq!(released.len(), 2, "held and fresh have both gone idle");
    }

    /// An unforked key — its fork failed and nothing asked again — goes once
    /// it has been idle a whole TTL.
    #[test]
    fn an_abandoned_unforked_key_is_swept() {
        let rt = runtime();
        let map = Resident::<u32>::default();
        let t0 = Instant::now();
        let ttl = Duration::from_secs(60);
        let _ = rt.block_on(map.get_or_try_init("c", t0, || async { Err::<Arc<u32>, ()>(()) }));
        assert!(map.sweep(t0 + ttl, ttl).is_empty());
        assert!(map.entries.pin().get("c").is_none());
    }

    /// **A request racing the sweep never keeps a released value, and two
    /// values are never live under one key**: across many sweeps, gets and
    /// forks running at once, no handle a request keeps is to a value a sweep
    /// released, and every handle a request kept is to the value the key names.
    ///
    /// A request may clone a value the sweep is retiring and drop the clone
    /// when it finds the entry retiring; what it never does is keep it.
    #[test]
    fn a_handle_a_request_keeps_is_never_to_a_released_value() {
        let rt = runtime();
        let map = Arc::new(Resident::<u32>::default());
        let forks = Arc::new(AtomicUsize::new(0));
        let stop = Arc::new(AtomicBool::new(false));
        let released = Arc::new(std::sync::Mutex::new(std::collections::HashSet::new()));
        let sweeper = {
            let (map, stop, released) =
                (Arc::clone(&map), Arc::clone(&stop), Arc::clone(&released));
            std::thread::spawn(move || {
                while !stop.load(Ordering::SeqCst) {
                    // A zero TTL: anything unheld goes.
                    for (_, v) in map.sweep(Instant::now(), Duration::ZERO) {
                        released.lock().unwrap().insert(*v);
                    }
                }
            })
        };
        let workers: Vec<_> = (0..4)
            .map(|_| {
                let (map, forks, released) =
                    (Arc::clone(&map), Arc::clone(&forks), Arc::clone(&released));
                rt.spawn(async move {
                    for _ in 0..5_000 {
                        let (v, _) = map
                            .get_or_try_init("c", Instant::now(), || {
                                let n = forks.fetch_add(1, Ordering::SeqCst) as u32;
                                async move { Ok::<_, ()>(Arc::new(n)) }
                            })
                            .await
                            .unwrap();
                        // Whatever the key names now is the value this request
                        // holds — a second fork never stood beside it.
                        if let Some(now) = map.get("c") {
                            assert!(Arc::ptr_eq(&now, &v), "two values live under one key");
                        }
                        tokio::task::yield_now().await;
                        assert!(
                            !released.lock().unwrap().contains(&*v),
                            "a value this request holds was released"
                        );
                    }
                })
            })
            .collect();
        rt.block_on(async {
            for w in workers {
                w.await.unwrap();
            }
        });
        stop.store(true, Ordering::SeqCst);
        sweeper.join().unwrap();
        assert!(
            !released.lock().unwrap().is_empty(),
            "the race was exercised"
        );
    }

    /// **What `init` does to the value happens before anyone else reaches it**:
    /// a lock the fork takes is already held when a second request is handed
    /// the value.
    #[test]
    fn a_lock_taken_in_init_is_held_before_anyone_else_sees_the_value() {
        let rt = runtime();
        let map = Arc::new(Resident::<tokio::sync::Mutex<u32>>::default());
        let (guard_tx, mut guard_rx) = tokio::sync::mpsc::unbounded_channel();
        let (first, minted) = rt
            .block_on(map.get_or_try_init("c", Instant::now(), || {
                let guard_tx = guard_tx.clone();
                async move {
                    let value = Arc::new(tokio::sync::Mutex::new(1));
                    let guard = Arc::clone(&value).try_lock_owned().unwrap();
                    guard_tx.send(guard).unwrap();
                    Ok::<_, ()>(value)
                }
            }))
            .unwrap();
        assert!(minted);
        let guard = guard_rx.try_recv().expect("the fork's guard");
        let (second, minted) = rt
            .block_on(map.get_or_try_init("c", Instant::now(), || async {
                Ok::<_, ()>(Arc::new(tokio::sync::Mutex::new(2)))
            }))
            .unwrap();
        assert!(!minted && Arc::ptr_eq(&first, &second));
        assert!(second.try_lock().is_err(), "the fork's lock is still held");
        drop(guard);
        assert_eq!(*second.try_lock().unwrap(), 1);
    }

    /// A value that counts its own drops.
    struct Counted(Arc<AtomicUsize>);

    impl Drop for Counted {
        fn drop(&mut self) {
            self.0.fetch_add(1, Ordering::SeqCst);
        }
    }

    /// **A released value goes when its last holder lets it go**, not when the
    /// map gets round to freeing the entry it retired: by `remove` and by the
    /// sweep alike, the drop is counted the moment the returned handle drops.
    #[test]
    fn a_released_value_is_dropped_with_its_last_handle() {
        let rt = runtime();
        let map = Resident::<Counted>::default();
        let dropped = Arc::new(AtomicUsize::new(0));
        let now = Instant::now();
        let fork = |dropped: &Arc<AtomicUsize>| {
            let dropped = Arc::clone(dropped);
            move || {
                let dropped = Arc::clone(&dropped);
                async move { Ok::<_, ()>(Arc::new(Counted(dropped))) }
            }
        };

        let (held, _) = rt
            .block_on(map.get_or_try_init("removed", now, fork(&dropped)))
            .unwrap();
        drop(held);
        let removed = map.remove("removed").expect("it was resident");
        assert_eq!(dropped.load(Ordering::SeqCst), 0);
        drop(removed);
        assert_eq!(dropped.load(Ordering::SeqCst), 1, "remove kept the value");

        rt.block_on(map.get_or_try_init("swept", now, fork(&dropped)))
            .unwrap();
        let ttl = Duration::from_secs(60);
        let released = map.sweep(now + ttl, ttl);
        assert_eq!(released.len(), 1);
        assert_eq!(dropped.load(Ordering::SeqCst), 1);
        drop(released);
        assert_eq!(
            dropped.load(Ordering::SeqCst),
            2,
            "the sweep kept the value"
        );
    }

    /// `remove` releases a key whatever holds it; the holder keeps its handle.
    #[test]
    fn remove_releases_a_held_key() {
        let rt = runtime();
        let map = Resident::<u32>::default();
        let (held, _) = rt
            .block_on(map.get_or_try_init("c", Instant::now(), || ready(5)))
            .unwrap();
        assert_eq!(map.remove("c").as_deref(), Some(&5));
        assert!(map.get("c").is_none());
        assert_eq!(*held, 5);
    }
}
