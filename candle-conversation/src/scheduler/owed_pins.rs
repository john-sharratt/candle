//! The substrates whose working-set pins the wave thread owes a republish.
//!
//! A slot freed takes its working set out of its substrate's keep-set, and
//! the republish that says so needs the substrate's write lock. Waiting for it
//! held the wave thread for as long as a maintenance op held the lock: 390 ms,
//! and up to 1,035 ms, of one `free_sequence` (BlackWorld, 2026-10-27) with
//! every decode waiting behind it. A removal is not urgent — a pin standing a
//! wave longer only keeps a residence resident a wave longer — so it is taken
//! when the lock is free and owed until then (`Scheduler::republish_pins`).

use crate::projection::Conversation;

/// One entry per substrate, in the order first owed.
#[derive(Default)]
pub(super) struct OwedPins {
    owed: Vec<Conversation>,
}

impl OwedPins {
    /// Owe `conversation`'s substrate a republish; once, however often owed.
    pub(super) fn owe(&mut self, conversation: &Conversation) {
        if !self.owed.iter().any(|c| c.same_substrate(conversation)) {
            self.owed.push(conversation.clone());
        }
    }

    /// `conversation`'s substrate has been republished: it is owed nothing.
    pub(super) fn settled(&mut self, conversation: &Conversation) {
        self.owed.retain(|c| !c.same_substrate(conversation));
    }

    /// Every substrate owed, taken out — what is still owed after a retry is
    /// owed again.
    pub(super) fn take(&mut self) -> Vec<Conversation> {
        std::mem::take(&mut self.owed)
    }

    #[cfg(test)]
    pub(super) fn len(&self) -> usize {
        self.owed.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A substrate is owed once, however many of its slots are freed, and a
    /// republish settles it whichever handle names it.
    #[test]
    fn a_substrate_is_owed_once_and_settled_by_any_handle() {
        let a = Conversation::new();
        let b = Conversation::new();
        let mut owed = OwedPins::default();
        owed.owe(&a);
        owed.owe(&a.clone());
        owed.owe(&b);
        assert_eq!(owed.len(), 2);
        owed.settled(&a.clone());
        assert_eq!(owed.len(), 1);
        let left = owed.take();
        assert!(left[0].same_substrate(&b));
        assert_eq!(owed.len(), 0);
    }
}
