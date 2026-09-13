//! A throwaway conversation's timeline, retired however its request ends.
//!
//! The probe, the dream and the prose job each decode on a conversation minted
//! for that one request, and each used to tombstone it explicitly at the end.
//! Those decodes are awaited now, so the whole future can be dropped mid-turn
//! — a console that stops polling, a client that disconnects — and an explicit
//! tail never runs on that path: every abandoned request stranded a
//! registered, path-less "(untitled)" timeline. This guard is the tail every
//! path shares.

use std::sync::{Arc, Mutex};

use candle_conversation::projection::TimelineId;
use candle_conversation::ConversationEngine;

/// Tombstones its timeline on drop.
///
/// Unconditional, because these conversations are equally dead however the
/// request ended — finished, failed, or abandoned, nothing may ever surface
/// them. A tombstone under a still-winding-down decode is safe: the wind-down
/// seals onto a tombstoned timeline whose records nothing reads and the next
/// compaction drops.
pub struct Throwaway {
    engine: Arc<Mutex<ConversationEngine>>,
    timeline: TimelineId,
    /// What the conversation was for, for the warning when the tombstone
    /// itself fails.
    what: &'static str,
}

impl Throwaway {
    pub fn new(
        engine: &Arc<Mutex<ConversationEngine>>,
        timeline: TimelineId,
        what: &'static str,
    ) -> Self {
        Self {
            engine: Arc::clone(engine),
            timeline,
            what,
        }
    }
}

impl Drop for Throwaway {
    fn drop(&mut self) {
        if let Ok(engine) = self.engine.lock() {
            if let Err(e) = engine.tombstone_timeline(self.timeline) {
                tracing::warn!(
                    "{} conversation {} could not be retired: {e:?} — it stays selectable \
                     and nothing will ever read it",
                    self.what,
                    self.timeline
                );
            }
        }
    }
}
