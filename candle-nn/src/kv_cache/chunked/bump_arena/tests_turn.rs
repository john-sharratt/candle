//! Tests that a shared domain admits one holding thread at a time.
//!
//! A thread that nests on a domain it already holds proceeds at once, while
//! another thread waits until the holder has given the domain back entirely,
//! nested holds included. Without that wait, two threads' staging ranges would
//! stack on one cursor and overrun the span.

use super::*;

#[cfg(test)]
pub(super) mod turn_tests {
    use super::Turn;

    use std::sync::atomic::{AtomicBool, Ordering};

    use std::sync::{mpsc, Arc};

    use std::thread;

    use std::time::Duration;

    /// A thread nesting on a domain it already holds does not wait on itself.
    #[test]
    fn the_holder_nests_without_waiting() {
        let turn = Turn::new();
        let outer = turn.enter();
        let inner = turn.enter();
        drop(inner);
        drop(outer);
        assert!(turn.holder.lock().unwrap().is_none(), "fully released");
    }

    /// **The fault this exists for.** A second thread does not open a
    /// generation while the first holds one — it waits, then takes the domain
    /// once the first has given it back entirely, nested holds included.
    #[test]
    fn another_thread_waits_until_the_holder_has_left() {
        let turn = Turn::new();
        let outer = turn.enter();
        let inner = turn.enter();
        let entered = Arc::new(AtomicBool::new(false));
        let (ready_tx, ready_rx) = mpsc::channel();
        let waiter = {
            let (turn, entered) = (Arc::clone(&turn), Arc::clone(&entered));
            thread::spawn(move || {
                ready_tx.send(()).unwrap();
                let _held = turn.enter();
                entered.store(true, Ordering::SeqCst);
            })
        };
        ready_rx.recv().unwrap();
        thread::sleep(Duration::from_millis(50));
        assert!(!entered.load(Ordering::SeqCst), "entered while held");
        drop(inner);
        thread::sleep(Duration::from_millis(50));
        assert!(
            !entered.load(Ordering::SeqCst),
            "entered while a nested hold was still open"
        );
        drop(outer);
        waiter.join().unwrap();
        assert!(entered.load(Ordering::SeqCst));
        assert!(turn.holder.lock().unwrap().is_none());
    }
}
