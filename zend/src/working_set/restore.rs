//! A dialogue's locks, rebuilt from the marks on its own turns
//! (`docs/zend_working_set.md` §4.8).
//!
//! The marks decide, not a re-run of the screen: re-screening could answer
//! differently — a seed resolved to another size, the budget came out tighter
//! — and a model holding an `in_context` reply with no lock behind it is the
//! failure the working set exists to prevent. The marks are what the model was
//! told, recorded on the conversation it was told about.

use candle_conversation::projection::TimelineId;
use candle_conversation::working_set::marks::standing_locks;
use candle_conversation::ConversationEngine;

/// Put back the locks standing at the end of `target`'s history — the lock
/// marks after its last release, in order. A conversation retired since is
/// left out. Returns how many were restored.
pub fn restore_locks(engine: &ConversationEngine, target: TimelineId) -> usize {
    let tags = engine.turn_tag_lists(target);
    standing_locks(tags.iter().map(Vec::as_slice))
        .into_iter()
        .filter(|&timeline| engine.working_set_restore_lock(target, timeline))
        .count()
}
