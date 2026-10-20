//! A conversation another member let go becomes this member's to take once it
//! holds it complete (`docs/cooperative_admission.md` §2.2,
//! `docs/replicated_substrate.md` §4.4).
//!
//! A release names the point its conversation must be held through: the
//! releaser's newest commit of it. Judged by the releaser's lane as a whole, a
//! member still receiving that member's KV and snapshots of other
//! conversations was never a candidate, though it held this one entire — so
//! every heavy-read turn after the first stayed where it started. Here the
//! conversation is judged by its own records: that commit applied, and every
//! turn up to the one it committed held. A member that holds it so says so
//! (`COMPLETE`), and the gateway may place its next turn there.
//!
//! And this member's own releases name the same point for others: its newest
//! commit of the conversation, not the whole of what it has written.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use candle_conversation::projection::{Conversation, TimelineId};
use cluster::member::AppMember;
use host::cluster::store::ConversationStore;

/// Released conversations this member does not yet hold complete: each claim
/// key with the releaser and the `oseq` of its newest commit.
///
/// **Either order.** A release may be heard before the records it names have
/// arrived, or after: the batch carrying its commit may have been applied
/// already. So a release is checked the moment it is heard as well as after
/// every batch, and a key held complete is reported whichever came last.
#[derive(Default)]
struct Awaited {
    keys: HashMap<Vec<u8>, (u64, u64)>,
}

impl Awaited {
    /// A release of `key` by `by` through its commit `through`: `Some(key)` when
    /// it is held complete already, otherwise awaited.
    fn heard(
        &mut self,
        key: &[u8],
        by: u64,
        through: u64,
        holds: impl Fn(&[u8], u64, u64) -> bool,
    ) -> Option<Vec<u8>> {
        if holds(key, by, through) {
            self.keys.remove(key);
            return Some(key.to_vec());
        }
        self.keys.insert(key.to_vec(), (by, through));
        None
    }

    /// A batch applied: the awaited keys now held complete, no longer awaited.
    fn applied(&mut self, holds: impl Fn(&[u8], u64, u64) -> bool) -> Vec<Vec<u8>> {
        let held: Vec<Vec<u8>> = self
            .keys
            .iter()
            .filter(|(key, &(by, through))| holds(key, by, through))
            .map(|(key, _)| key.clone())
            .collect();
        for key in &held {
            self.keys.remove(key);
        }
        held
    }
}

/// Wire this member's handovers: hear every release, check each after every
/// batch applied, report those held complete, and name this member's own
/// releases by their conversations' newest commits. `timeline_of` maps a
/// claim key's conversation id to the timeline that holds it.
pub fn install(
    member: Arc<AppMember>,
    store: &ConversationStore,
    conversation: Conversation,
    timeline_of: fn(&str) -> TimelineId,
) {
    let awaited: Arc<Mutex<Awaited>> = Arc::default();
    let holds = {
        let conversation = conversation.clone();
        move |key: &[u8], by: u64, through: u64| {
            std::str::from_utf8(key)
                .is_ok_and(|id| conversation.holds_complete_through(timeline_of(id), by, through))
        }
    };
    {
        let awaited = awaited.clone();
        let member = member.clone();
        let holds = holds.clone();
        member.clone().on_unclaim(Box::new(move |key, by, through| {
            let held = awaited
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .heard(key, by, through, &holds);
            if let Some(key) = held {
                member.report_complete(vec![key]);
            }
        }));
    }
    {
        let conversation = conversation.clone();
        member.key_through(Box::new(move |key| {
            let id = std::str::from_utf8(key).ok()?;
            conversation.last_commit_oseq(timeline_of(id), conversation.node())
        }));
    }
    store.on_applied(move || {
        let held = awaited
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .applied(&holds);
        if !held.is_empty() {
            member.report_complete(held);
        }
    });
}

#[cfg(test)]
mod tests {
    use super::Awaited;
    use std::cell::Cell;

    /// **Held complete whichever comes last.** A release heard after the batch
    /// carrying its commit was applied is reported at once; one heard before
    /// is reported by the batch that completes it; and one not yet held stays
    /// awaited until then.
    #[test]
    fn a_release_is_reported_held_complete_in_either_order() {
        let holding = Cell::new(false);
        let holds = |_: &[u8], _: u64, _: u64| holding.get();

        // The commit's batch first: nothing is awaited yet, so nothing to say.
        let mut a = Awaited::default();
        holding.set(true);
        assert!(a.applied(holds).is_empty());
        // Then the release: held already, reported at once.
        assert_eq!(a.heard(b"k1", 7, 30, holds), Some(b"k1".to_vec()));
        assert!(a.keys.is_empty(), "nothing left awaited");

        // The release first: not held yet, so awaited.
        let mut b = Awaited::default();
        holding.set(false);
        assert_eq!(b.heard(b"k2", 7, 40, holds), None);
        assert!(b.applied(holds).is_empty(), "still short");
        // The batch that completes it reports it, once.
        holding.set(true);
        assert_eq!(b.applied(holds), vec![b"k2".to_vec()]);
        assert!(b.applied(holds).is_empty());
    }
}
