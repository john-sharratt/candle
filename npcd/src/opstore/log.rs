//! The substrate as the operation stores' file log.

use candle_conversation::persistence::vfs::{VfsEventId, VfsEventPayload, VfsWrite};
use candle_conversation::persistence::SharedSubstrate;
use candle_conversation::projection::TimelineId;
use store_events::FileLog;

/// A shared substrate, keeping each store's events on its timeline.
pub struct Substrate<'a>(pub &'a SharedSubstrate);

impl FileLog for Substrate<'_> {
    fn write_files(
        &self,
        timeline: TimelineId,
        write: &VfsWrite,
    ) -> Result<Vec<VfsEventId>, String> {
        self.0
            .write_vfs(timeline.raw(), write)
            .map_err(|e| e.to_string())
    }

    fn read_files(&self, timeline: TimelineId) -> Result<Vec<VfsEventPayload>, String> {
        self.0.vfs_events(timeline.raw()).map_err(|e| e.to_string())
    }
}
