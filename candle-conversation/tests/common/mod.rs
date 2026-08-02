//! Helpers shared across candle-conversation integration tests.
//!
//! Currently just [`open_conversation`]: reopen a redo log the way the `zend`
//! daemon does at startup (walk every record into a fresh [`Substrate`], then
//! bind it to a [`Conversation`]) so tests exercise the same recovery path as
//! production instead of `Conversation::with_persistence`'s empty-substrate
//! shortcut, which recovers nothing.

use std::path::Path;

use candle_conversation::persistence::SubstratePersistence;
use candle_conversation::projection::Conversation;
use candle_conversation::substrate::Substrate;

/// Open the redo log under `dir`, driving every record through the substrate
/// walker in one pass (exactly as the daemon does in `ConversationEngine::new`),
/// then bind the populated substrate to a fresh conversation.
///
/// This is the only correct way to reopen for recovery: `open_in` with an empty
/// `Substrate::new()` (the old `Conversation::with_persistence` path) leaves the
/// substrate unpopulated, so `reconstruct_from_log` finds no turn decls and
/// recovers nothing. For a fresh (empty) log the walker is simply a no-op.
pub fn open_conversation(dir: &Path) -> Conversation {
    let mut substrate = Substrate::new();
    let persistence = SubstratePersistence::open_in_with_substrate(dir, &mut substrate).unwrap();
    Conversation::from_parts(substrate, persistence)
}
