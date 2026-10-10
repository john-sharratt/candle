//! The substrates attached to one engine core (`docs/shared_model_host.md`
//! §3.2), and what refuses one more.

use std::collections::HashSet;
use std::path::Path;
use std::sync::Arc;

use candle_nn::kv_cache::ChunkedKvBacking;

use crate::error::ConversationError;
use crate::projection::{Conversation, TimelineId};

/// The substrates attached to one engine core. Every change — an attach, a
/// detach, the stop — happens under its lock together with the scheduler
/// request that announces it, so the scheduler sees them in the order the list
/// changed: a detach is always queued ahead of the `Shutdown` the last
/// attachment's stop sends, and an attach never lands on a stopped core.
pub(crate) struct Attachments {
    /// Every attached substrate, by attachment id.
    pub(crate) list: Vec<(u64, Conversation)>,
    /// The per-layer KV backings a new attachment's persistence thread is
    /// handed, while the core runs. The last attachment's stop takes them, so
    /// the backings — and the pinned staging arena each holds — are released
    /// with the scheduler that used them, not kept for as long as some engine
    /// handle happens to live. Absent: the core has stopped, and nothing
    /// attaches after it.
    pub(crate) persist_with: Option<Arc<Vec<ChunkedKvBacking>>>,
}

impl Attachments {
    /// One substrate, attached as `id` — the engine's first — on a core whose
    /// attachments persist through `backings`.
    pub(crate) fn first(
        id: u64,
        conversation: Conversation,
        backings: Arc<Vec<ChunkedKvBacking>>,
    ) -> Self {
        Self {
            list: vec![(id, conversation)],
            persist_with: Some(backings),
        }
    }

    /// The backings `conversation` (the substrate at `dir`) persists through,
    /// or why it cannot join: the core has stopped, the substrate is attached
    /// already, or it holds a timeline an attached substrate also holds — the
    /// scheduler keys its state by timeline, so two holders of one would share
    /// it.
    pub(crate) fn admit(
        &self,
        conversation: &Conversation,
        dir: &Path,
    ) -> Result<Arc<Vec<ChunkedKvBacking>>, ConversationError> {
        let Some(backings) = &self.persist_with else {
            return Err(ConversationError::SchedulerGone);
        };
        if self
            .list
            .iter()
            .any(|(_, c)| c.same_substrate(conversation))
        {
            return Err(ConversationError::Other(format!(
                "the substrate at {} is already attached to this engine",
                dir.display()
            )));
        }
        let theirs: HashSet<TimelineId> = conversation.timeline_ids().into_iter().collect();
        for (_, c) in &self.list {
            if let Some(shared) = c.timeline_ids().into_iter().find(|t| theirs.contains(t)) {
                return Err(ConversationError::Other(format!(
                    "the substrate at {} holds timeline {shared}, which a substrate already \
                     attached to this engine also holds; the scheduler keys its state by \
                     timeline, so the two cannot share an engine",
                    dir.display()
                )));
            }
        }
        Ok(Arc::clone(backings))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::projection::{GroupId, LayerId};

    fn holding(timelines: &[u64]) -> Conversation {
        let c = Conversation::new();
        let layer = LayerId::from_raw(1).unwrap();
        let group = GroupId::from_raw(1).unwrap();
        for &t in timelines {
            c.register_timeline(TimelineId::from_raw(t).unwrap(), layer, group);
        }
        c
    }

    /// A substrate with timelines of its own joins, and is handed the core's
    /// backings; the same substrate twice, one sharing a timeline, and
    /// anything after the stop are refused — each saying why.
    #[test]
    fn an_attach_is_refused_for_each_reason_and_admitted_otherwise() {
        let dir = Path::new("ws/substrate");
        let first = holding(&[1, 2]);
        let backings = Arc::new(Vec::new());
        let mut attached = Attachments::first(1, first.clone(), Arc::clone(&backings));

        let admitted = attached.admit(&holding(&[3, 4]), dir).unwrap();
        assert!(Arc::ptr_eq(&admitted, &backings));
        drop(admitted);

        let twice = attached.admit(&first, dir).unwrap_err().to_string();
        assert!(twice.contains("already attached"), "{twice}");

        let shared = attached
            .admit(&holding(&[5, 2]), dir)
            .unwrap_err()
            .to_string();
        assert!(shared.contains("holds timeline 2"), "{shared}");

        // The stop takes the core's backings: what remains is this test's own.
        attached.list.clear();
        attached.persist_with = None;
        assert_eq!(Arc::strong_count(&backings), 1);
        assert!(matches!(
            attached.admit(&holding(&[9]), dir),
            Err(ConversationError::SchedulerGone)
        ));
    }
}
