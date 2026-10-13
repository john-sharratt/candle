//! One reader for a unit two members find at the same time.
//!
//! Branch ingest already skips a unit whose conversation has replicated
//! (`docs/replicated_substrate.md` §10.4). That misses the startup window,
//! where every member plans the same unread tree before any of them has
//! written a content key. Each unit is leased under `ingest/<content key>`
//! before its conversation is minted. The member that holds the lease reads
//! it; the others leave it, and the conversation arrives with replication.
//!
//! A member that cannot propose — it is not joined — reads the unit itself.
//! Two such members can still both read it, and the lower version stamp
//! keeps the unit, as it does for any pair of readings that both finish.

use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, MutexGuard, OnceLock};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

use cluster::replicate::{LeaseEvent, Replicator};
use tokio::sync::broadcast::{self, Receiver};

/// How long a reading may hold its unit. A crash frees it when this runs out
/// unrenewed, and another member's next pass takes it.
const INGEST_TTL: Duration = Duration::from_secs(120);

/// How often the holder renews [`INGEST_TTL`] while the reading is still
/// running.
const RENEW_EVERY: Duration = Duration::from_secs(30);

/// How long to wait for the vote before leaving the unit for a later pass.
const CLAIM_WAIT: Duration = Duration::from_secs(5);

/// How long to wait for a previous holder's writes to arrive before deciding
/// the unit is still unread.
const APPLY_WAIT: Duration = Duration::from_secs(5);

const PREFIX: &[u8] = b"ingest/";

/// The lease key for a content key.
pub fn lease_key(content_key: &str) -> Vec<u8> {
    let mut key = Vec::with_capacity(PREFIX.len() + content_key.len());
    key.extend_from_slice(PREFIX);
    key.extend_from_slice(content_key.as_bytes());
    key
}

fn is_ingest(key: &[u8]) -> bool {
    key.starts_with(PREFIX)
}

/// What this member should do with a unit it found unread.
pub enum Share {
    /// Read it. The guard is dropped when the reading finishes or is abandoned,
    /// which hands the lease back.
    Mine(Guard),
    /// Another member holds it.
    Theirs,
}

/// Holds an ingest lease until dropped, or nothing when this member is reading
/// because it could not propose.
pub struct Guard {
    seat: Option<Arc<dyn LeaseSeat>>,
    key: Vec<u8>,
    stop: Arc<AtomicBool>,
    renew: Option<JoinHandle<()>>,
    /// A previous holder handed the unit back and its writes were waited on.
    /// The caller checks the substrate again before minting a conversation.
    recheck: bool,
}

impl Guard {
    fn local() -> Guard {
        Guard {
            seat: None,
            key: Vec::new(),
            stop: Arc::new(AtomicBool::new(true)),
            renew: None,
            recheck: false,
        }
    }

    /// Whether the substrate should be consulted again before this unit is read.
    pub fn recheck(&self) -> bool {
        self.recheck
    }
}

impl Drop for Guard {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Relaxed);
        if let Some(renew) = self.renew.take() {
            let _ = renew.join();
        }
        if let Some(seat) = self.seat.take() {
            seat.release(&self.key);
        }
    }
}

/// The member's replication, once it has joined. Absent until then, and a
/// daemon with no cluster never sets it: every unit is read here.
static SEAT: Mutex<Option<Arc<dyn LeaseSeat>>> = Mutex::new(None);

/// Who handed an `ingest/` lease back, and how far their own writes reached.
type Handback = (u64, u64);

/// Every ingest lease handed back and not yet taken up again.
type Handbacks = HashMap<Vec<u8>, Handback>;

/// `ingest/` leases handed back, so the next holder waits for those writes
/// before it decides the unit is still unread.
static RELEASED: Mutex<Option<Handbacks>> = Mutex::new(None);

static WATCH: OnceLock<()> = OnceLock::new();

/// Remember `replicator` and follow its lease events. Later calls replace the
/// seat the readings propose through; the event watch keeps the first one,
/// which is this process's replicator for the life of the load.
pub fn install(replicator: Arc<Replicator>) {
    let seat: Arc<dyn LeaseSeat> = Arc::new(ClusterSeat(replicator));
    *SEAT.lock().unwrap_or_else(|e| e.into_inner()) = Some(Arc::clone(&seat));
    if WATCH.set(()).is_ok() {
        let _ = thread::Builder::new()
            .name("ingest-lease".into())
            .spawn(move || watch(seat));
    }
}

/// Lease `content_key`. [`Share::Mine`] when this member should read the unit.
pub fn share(content_key: &str) -> Share {
    let seat = SEAT.lock().unwrap_or_else(|e| e.into_inner()).clone();
    match seat {
        Some(seat) => claim(seat, content_key),
        None => Share::Mine(Guard::local()),
    }
}

fn claim(seat: Arc<dyn LeaseSeat>, content_key: &str) -> Share {
    let key = lease_key(content_key);
    let mut events = seat.subscribe();
    if !seat.try_acquire(&key) {
        return Share::Mine(Guard::local());
    }
    let deadline = Instant::now() + CLAIM_WAIT;
    loop {
        if seat.holds(&key) {
            let recheck = wait_applied(&seat, &key);
            return Share::Mine(held(seat, key, recheck));
        }
        match events.try_recv() {
            Ok(LeaseEvent::Held { key: got }) if got == key => {
                let recheck = wait_applied(&seat, &key);
                return Share::Mine(held(seat, key, recheck));
            }
            Ok(LeaseEvent::Busy { key: got, .. }) if got == key => return Share::Theirs,
            Ok(LeaseEvent::Lost { key: got }) if got == key => return Share::Theirs,
            Ok(_) => {}
            Err(broadcast::error::TryRecvError::Empty) => {
                if Instant::now() >= deadline {
                    return if seat.holds(&key) {
                        let recheck = wait_applied(&seat, &key);
                        Share::Mine(held(seat, key, recheck))
                    } else {
                        Share::Theirs
                    };
                }
                thread::sleep(Duration::from_millis(10));
            }
            Err(broadcast::error::TryRecvError::Lagged(_)) => {}
            Err(broadcast::error::TryRecvError::Closed) => {
                return if seat.holds(&key) {
                    let recheck = wait_applied(&seat, &key);
                    Share::Mine(held(seat, key, recheck))
                } else {
                    Share::Theirs
                };
            }
        }
    }
}

fn held(seat: Arc<dyn LeaseSeat>, key: Vec<u8>, recheck: bool) -> Guard {
    let stop = Arc::new(AtomicBool::new(false));
    let renew_stop = Arc::clone(&stop);
    let renew_seat = Arc::clone(&seat);
    let renew_key = key.clone();
    let renew = thread::Builder::new()
        .name("ingest-lease-renew".into())
        .spawn(move || renew(renew_seat, renew_key, renew_stop))
        .ok();
    Guard {
        seat: Some(seat),
        key,
        stop,
        renew,
        recheck,
    }
}

fn renew(seat: Arc<dyn LeaseSeat>, key: Vec<u8>, stop: Arc<AtomicBool>) {
    let mut next = Instant::now() + RENEW_EVERY;
    while !stop.load(Ordering::Relaxed) {
        thread::sleep(Duration::from_millis(50));
        if stop.load(Ordering::Relaxed) {
            break;
        }
        if Instant::now() >= next {
            seat.try_acquire(&key);
            next = Instant::now() + RENEW_EVERY;
        }
    }
}

/// Wait out a recorded hand-back. Returns whether one was recorded.
fn wait_applied(seat: &Arc<dyn LeaseSeat>, key: &[u8]) -> bool {
    let Some((by, through)) = take_release(key) else {
        return false;
    };
    let deadline = Instant::now() + APPLY_WAIT;
    while seat.applied(by) < through && Instant::now() < deadline {
        thread::sleep(Duration::from_millis(20));
    }
    true
}

fn handbacks() -> MutexGuard<'static, Option<Handbacks>> {
    RELEASED.lock().unwrap_or_else(|e| e.into_inner())
}

fn take_release(key: &[u8]) -> Option<Handback> {
    handbacks().get_or_insert_with(HashMap::new).remove(key)
}

fn note_release(key: Vec<u8>, by: u64, through: u64) {
    if !is_ingest(&key) {
        return;
    }
    handbacks()
        .get_or_insert_with(HashMap::new)
        .insert(key, (by, through));
}

fn watch(seat: Arc<dyn LeaseSeat>) {
    let mut events = seat.subscribe();
    loop {
        match events.blocking_recv() {
            Ok(LeaseEvent::Released { key, by, through }) => note_release(key, by, through),
            Ok(_) => {}
            Err(_) => break,
        }
    }
}

trait LeaseSeat: Send + Sync {
    fn try_acquire(&self, key: &[u8]) -> bool;
    fn holds(&self, key: &[u8]) -> bool;
    fn release(&self, key: &[u8]);
    fn subscribe(&self) -> Receiver<LeaseEvent>;
    fn applied(&self, origin: u64) -> u64;
}

struct ClusterSeat(Arc<Replicator>);

impl LeaseSeat for ClusterSeat {
    fn try_acquire(&self, key: &[u8]) -> bool {
        self.0.acquire(key, INGEST_TTL)
    }

    fn holds(&self, key: &[u8]) -> bool {
        self.0.holds(key)
    }

    fn release(&self, key: &[u8]) {
        self.0.release_lease(key, self.0.own_through());
    }

    fn subscribe(&self) -> Receiver<LeaseEvent> {
        self.0.lease_events()
    }

    fn applied(&self, origin: u64) -> u64 {
        self.0.applied_through(origin)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Fake {
        id: u64,
        allow: bool,
        held: Arc<Mutex<HashMap<Vec<u8>, u64>>>,
        applied: Mutex<u64>,
        tx: broadcast::Sender<LeaseEvent>,
    }

    impl Fake {
        fn pair(id: u64, held: Arc<Mutex<HashMap<Vec<u8>, u64>>>) -> Arc<Fake> {
            let (tx, _) = broadcast::channel(32);
            Arc::new(Fake {
                id,
                allow: true,
                held,
                applied: Mutex::new(0),
                tx,
            })
        }
    }

    impl LeaseSeat for Fake {
        fn try_acquire(&self, key: &[u8]) -> bool {
            if !self.allow {
                return false;
            }
            let mut held = self.held.lock().unwrap();
            match held.get(key).copied() {
                Some(holder) if holder == self.id => {
                    let _ = self.tx.send(LeaseEvent::Held { key: key.to_vec() });
                }
                Some(holder) => {
                    let _ = self.tx.send(LeaseEvent::Busy {
                        key: key.to_vec(),
                        holder,
                    });
                }
                None => {
                    held.insert(key.to_vec(), self.id);
                    let _ = self.tx.send(LeaseEvent::Held { key: key.to_vec() });
                }
            }
            true
        }

        fn holds(&self, key: &[u8]) -> bool {
            self.held.lock().unwrap().get(key) == Some(&self.id)
        }

        fn release(&self, key: &[u8]) {
            let removed = self.held.lock().unwrap().remove(key);
            if removed == Some(self.id) {
                note_release(key.to_vec(), self.id, 7);
                let _ = self.tx.send(LeaseEvent::Released {
                    key: key.to_vec(),
                    by: self.id,
                    through: 7,
                });
            }
        }

        fn subscribe(&self) -> Receiver<LeaseEvent> {
            self.tx.subscribe()
        }

        fn applied(&self, _origin: u64) -> u64 {
            *self.applied.lock().unwrap()
        }
    }

    #[test]
    fn lease_key_names_the_unit() {
        assert_eq!(
            lease_key("candle/zend/src/lib.rs@abc"),
            b"ingest/candle/zend/src/lib.rs@abc".to_vec()
        );
    }

    #[test]
    fn one_member_reads_a_unit_both_found() {
        let held = Arc::new(Mutex::new(HashMap::new()));
        let a = Fake::pair(1, Arc::clone(&held));
        let b = Fake::pair(2, Arc::clone(&held));
        let key = "candle/src/main.rs@111";
        let mine = claim(a, key);
        let Share::Mine(guard) = mine else {
            panic!("the first member should read the unit");
        };
        assert!(!guard.recheck());
        assert!(matches!(claim(b, key), Share::Theirs));
        drop(guard);
        assert!(held.lock().unwrap().is_empty());
    }

    #[test]
    fn a_handed_back_unit_is_taken_after_its_writes_apply() {
        let held = Arc::new(Mutex::new(HashMap::new()));
        let a = Fake::pair(1, Arc::clone(&held));
        let b = Fake::pair(2, held);
        let key = "repo/dir@222";
        let Share::Mine(guard) = claim(Arc::clone(&a) as Arc<dyn LeaseSeat>, key) else {
            panic!("first holder");
        };
        drop(guard);
        *b.applied.lock().unwrap() = 7;
        let Share::Mine(next) = claim(b, key) else {
            panic!("the unit is free");
        };
        assert!(next.recheck());
    }

    #[test]
    fn a_member_that_cannot_propose_reads_locally() {
        let held = Arc::new(Mutex::new(HashMap::new()));
        let a = Fake::pair(1, held);
        let quiet = Arc::new(Fake {
            id: 1,
            allow: false,
            held: Arc::clone(&a.held),
            applied: Mutex::new(0),
            tx: a.tx.clone(),
        });
        let Share::Mine(guard) = claim(quiet, "local-only") else {
            panic!("a cut-off member still reads");
        };
        assert!(!guard.recheck());
        assert!(a.held.lock().unwrap().is_empty());
    }

    #[test]
    fn no_cluster_reads_every_unit() {
        let Share::Mine(guard) = share("unclustered") else {
            panic!("without a seat every unit is read here");
        };
        assert!(!guard.recheck());
    }
}
