//! The machine's one build slot and the queue in front of it
//! (`docs/build_agent.md` §2, §3).
//!
//! A run joins the queue once its store is ready and takes the slot when it is
//! at the front, the queue is open and no hold stands. A restart holds the
//! queue: no run takes the slot while a hold stands, and the run holding it is
//! waited for or set aside by the hold ([`super::hold`]). A queue starts
//! closed only for a process that proves itself first: it opens once the
//! process is serving.

use std::collections::VecDeque;
use std::sync::{Mutex, MutexGuard};
use std::time::Instant;

use tokio::sync::watch;

/// The run holding the slot.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Holder {
    pub id: String,
    /// It is a deploy's build: a restart waits for it.
    pub deploy: bool,
    pub since: Instant,
}

#[derive(Debug)]
struct State {
    order: VecDeque<String>,
    running: Option<Holder>,
    holds: usize,
    open: bool,
}

/// See the module documentation.
pub struct Queue {
    s: Mutex<State>,
    /// Bumped on every change, so a waiting run looks again.
    changed: watch::Sender<u64>,
}

impl Default for Queue {
    fn default() -> Self {
        Queue {
            s: Mutex::new(State {
                order: VecDeque::new(),
                running: None,
                holds: 0,
                open: true,
            }),
            changed: watch::channel(0).0,
        }
    }
}

impl Queue {
    fn state(&self) -> MutexGuard<'_, State> {
        self.s.lock().unwrap_or_else(|e| e.into_inner())
    }

    fn bump(&self) {
        self.changed.send_modify(|n| *n = n.wrapping_add(1));
    }

    /// A receiver that hears every change from now on.
    pub fn watch(&self) -> watch::Receiver<u64> {
        self.changed.subscribe()
    }

    /// Run `id` waits for the slot: at the back, or at the front when it is a
    /// run set aside by a hold, which goes first once the hold lifts.
    pub fn join(&self, id: &str, front: bool) {
        {
            let mut s = self.state();
            if s.order.iter().any(|o| o == id) {
                return;
            }
            match front {
                true => s.order.push_front(id.to_string()),
                false => s.order.push_back(id.to_string()),
            }
        }
        self.bump();
    }

    /// Wait until run `id` holds the slot: it is at the front, the queue is
    /// open, no hold stands and nothing holds the slot.
    pub async fn take(&self, id: &str, deploy: bool) {
        let mut rx = self.watch();
        loop {
            {
                let mut s = self.state();
                let free = s.open && s.holds == 0 && s.running.is_none();
                if free && s.order.front().is_some_and(|f| f == id) {
                    s.order.pop_front();
                    s.running = Some(Holder {
                        id: id.to_string(),
                        deploy,
                        since: Instant::now(),
                    });
                    drop(s);
                    self.bump();
                    return;
                }
            }
            if rx.changed().await.is_err() {
                return;
            }
        }
    }

    /// Run `id` is done with the slot, or never took it: it leaves the queue,
    /// and gives the slot back if it held it.
    pub fn leave(&self, id: &str) {
        {
            let mut s = self.state();
            s.order.retain(|o| o != id);
            if s.running.as_ref().is_some_and(|r| r.id == id) {
                s.running = None;
            }
        }
        self.bump();
    }

    /// The run holding the slot.
    pub fn running(&self) -> Option<Holder> {
        self.state().running.clone()
    }

    /// The runs waiting, in their order.
    pub fn waiting(&self) -> Vec<String> {
        self.state().order.iter().cloned().collect()
    }

    /// Nothing holds the slot, nothing waits, no hold stands and the queue is
    /// open: a run sent here starts at once.
    pub fn free(&self) -> bool {
        let s = self.state();
        s.open && s.holds == 0 && s.running.is_none() && s.order.is_empty()
    }

    /// A hold begins: no run takes the slot until every hold has ended.
    pub fn hold(&self) {
        self.state().holds += 1;
        self.bump();
    }

    /// A hold ended; whether none stands now.
    pub fn release(&self) -> bool {
        let none = {
            let mut s = self.state();
            s.holds = s.holds.saturating_sub(1);
            s.holds == 0
        };
        self.bump();
        none
    }

    /// Whether a hold stands.
    pub fn held(&self) -> bool {
        self.state().holds > 0
    }

    /// Runs wait until [`Self::open`]: for a process that is not serving yet.
    pub fn close(&self) {
        self.state().open = false;
        self.bump();
    }

    /// Runs may take the slot.
    pub fn open(&self) {
        self.state().open = true;
        self.bump();
    }

    pub fn is_open(&self) -> bool {
        self.state().open
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;
    use std::time::Duration;

    use tokio::time::timeout;

    use super::*;

    async fn takes(q: &Arc<Queue>, id: &'static str) -> bool {
        let q = Arc::clone(q);
        timeout(
            Duration::from_millis(50),
            async move { q.take(id, false).await },
        )
        .await
        .is_ok()
    }

    /// **One run holds the slot at a time, in the order they joined.**
    #[tokio::test]
    async fn runs_take_the_slot_one_at_a_time_in_order() {
        let q = Arc::new(Queue::default());
        q.join("a", false);
        q.join("b", false);
        assert!(!takes(&q, "b").await, "b waits behind a");
        assert!(takes(&q, "a").await);
        assert!(!q.free());
        assert!(!takes(&q, "b").await, "a holds the slot");
        q.leave("a");
        assert!(takes(&q, "b").await);
        q.leave("b");
        assert!(q.free());
    }

    /// **A hold stops the queue, and a run set aside by it goes first after.**
    #[tokio::test]
    async fn a_hold_stops_the_queue_and_a_set_aside_run_goes_first() {
        let q = Arc::new(Queue::default());
        q.join("a", false);
        q.hold();
        assert!(q.held());
        assert!(!takes(&q, "a").await);
        q.join("cut", true);
        assert_eq!(q.waiting(), ["cut", "a"]);
        assert!(q.release());
        assert!(!takes(&q, "a").await, "the set-aside run is first");
        assert!(takes(&q, "cut").await);
    }

    /// **Holds join**: the queue stays stopped until the last one ends.
    #[test]
    fn the_queue_moves_again_only_when_the_last_hold_ends() {
        let q = Queue::default();
        q.hold();
        q.hold();
        assert!(!q.release());
        assert!(q.held());
        assert!(q.release());
        assert!(!q.held());
    }

    /// **A closed queue starts nothing until it opens.**
    #[tokio::test]
    async fn a_closed_queue_waits_to_open() {
        let q = Arc::new(Queue::default());
        q.close();
        q.join("a", false);
        assert!(!takes(&q, "a").await);
        assert!(!q.free());
        q.open();
        assert!(takes(&q, "a").await);
    }

    /// **A run that leaves the queue before its turn no longer blocks it.**
    #[tokio::test]
    async fn a_run_that_leaves_does_not_block_the_queue() {
        let q = Arc::new(Queue::default());
        q.join("gone", false);
        q.join("b", false);
        q.leave("gone");
        assert!(takes(&q, "b").await);
        assert_eq!(q.running().map(|h| h.id), Some("b".to_string()));
    }
}
