//! Serialises generations on a domain that several threads stage through.
//!
//! A bump cursor rewinds only when the last open generation closes, so two
//! threads whose generations overlap would stack their ranges on one cursor and
//! overrun the span. A generation on such a domain therefore waits until no
//! other thread holds one, and the same thread may nest freely. The wait is one
//! staging batch, and nothing a holder does inside its generation waits on
//! another thread, so the wait ends when that batch does.

use super::*;

/// One thread at a time on a domain that several threads stage through.
///
/// A bump cursor rewinds only when the last open generation closes, so two
/// threads whose generations overlap stack their ranges on one cursor — and
/// while they keep overlapping it never rewinds at all. The persistence domain
/// is used by the persistence thread (hot→warm staging) and by the scheduler
/// (warm→hot elevate, seal gathering, sampled selection), each of which sizes
/// its batches to the domain's whole span; stacked, the second batch lands past
/// the end and the migrate fails. Measured: `persist-staging: transient span
/// exhausted — 127296 B at offset 67108096`, which dropped selected turns from
/// an npc's context.
///
/// So a generation on such a domain waits until no other thread holds one. The
/// same thread may nest freely. The wait is one staging batch — bounded by the
/// span and fenced on the copy stream — and nothing a holder does inside a
/// generation waits on another thread.
pub(super) struct Turn {
    pub(super) holder: Mutex<Option<(ThreadId, usize)>>,
    pub(super) freed: Condvar,
}

impl Turn {
    pub(super) fn new() -> Arc<Self> {
        Arc::new(Self {
            holder: Mutex::new(None),
            freed: Condvar::new(),
        })
    }

    /// Take the domain for this thread, waiting while another thread holds it.
    pub(super) fn enter(self: &Arc<Self>) -> TurnHeld {
        let me = current().id();
        let mut holder = self.holder.lock().unwrap_or_else(|e| e.into_inner());
        loop {
            match holder.as_mut() {
                None => {
                    *holder = Some((me, 1));
                    break;
                }
                Some((owner, depth)) if *owner == me => {
                    *depth += 1;
                    break;
                }
                Some(_) => {
                    holder = self.freed.wait(holder).unwrap_or_else(|e| e.into_inner());
                }
            }
        }
        TurnHeld(Arc::clone(self))
    }
}

/// This thread's hold on a [`Turn`], given back when the generation that took
/// it has finished closing — its reset included.
pub(super) struct TurnHeld(pub(super) Arc<Turn>);

impl Drop for TurnHeld {
    fn drop(&mut self) {
        let mut holder = self.0.holder.lock().unwrap_or_else(|e| e.into_inner());
        if let Some((_, depth)) = holder.as_mut() {
            *depth -= 1;
            if *depth == 0 {
                *holder = None;
                self.0.freed.notify_all();
            }
        }
    }
}
