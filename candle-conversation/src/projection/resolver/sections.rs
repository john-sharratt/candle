//! Declares, triages and counts a workspace's system-prompt sections in the redo
//! log.
//!
//! A section is declared under its content address, and one whose chunks are
//! durable is restored from the log rather than prefilled again. Triage counts each
//! restore and each prefill, so a restart that prefills the prompt again shows up
//! as a count. A tombstoned stream reads as not persisted, so triage falls through
//! to a fresh prefill rather than restoring corrupt bytes.

use super::*;

/// Running totals behind [`Conversation::section_loads`].
#[derive(Default)]
pub(super) struct SectionLoadCounters {
    pub(super) restored: AtomicU64,
    pub(super) prefilled: AtomicU64,
}

/// How many prompt sections a workspace restored from its redo log, and how
/// many it prefilled, since the handle was opened.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct SectionLoads {
    pub restored: u64,
    pub prefilled: u64,
}

impl Conversation {
    /// Pinned-section byte snapshot.
    pub fn section_bytes(&self) -> usize {
        self.read().section_bytes()
    }

    /// Declare a section stream — appends a `StreamDecl::PromptSection`
    /// record carrying the content address and debug name.  The
    /// derived stream id matches `section_stream_id(address)`.  Called
    /// by the scheduler at section seal time; pairs with later
    /// `Tokens` / `Signatures` / `Chunks` records keyed by the same id.
    pub fn declare_section_stream(
        &self,
        address: ContentAddress,
        debug_name: &str,
    ) -> candle::Result<StreamId> {
        let decl = StreamDecl::PromptSection(SectionDecl {
            address,
            debug_name: debug_name.to_string(),
        });
        Ok(self.declare_and_mirror(decl))
    }

    /// True when the workspace's manifest already holds durable
    /// chunks for `stream_id` — i.e. a section under this content
    /// address has been persisted and can be cold-loaded back into
    /// hot without re-prefilling.  The check matches the ingest
    /// loop's skip-if-present gate.
    ///
    /// A tombstoned stream (see [`Substrate::tombstone_section`]) always
    /// reads `false` here, even though its `Chunk` records are still on
    /// disk until the next compaction pass physically drops them — the
    /// whole point of the tombstone is that those chunks must never be
    /// restored again, so the triage that reads this must see "not
    /// persisted" and fall through to a fresh prefill instead.
    pub fn section_stream_is_persisted(&self, stream_id: StreamId) -> bool {
        let view = self.read();
        if view.is_section_tombstoned(stream_id) {
            return false;
        }
        view.stream_of(stream_id)
            .map(|s| s.committed_through.is_some() && !s.chunks.is_empty())
            .unwrap_or(false)
    }

    /// Count one section-ingest triage's outcome — see [`Self::section_loads`].
    pub fn record_section_loads(&self, restored: usize, prefilled: usize) {
        self.section_loads
            .restored
            .fetch_add(restored as u64, Ordering::Relaxed);
        self.section_loads
            .prefilled
            .fetch_add(prefilled as u64, Ordering::Relaxed);
    }

    /// Prompt sections restored from the redo log, and prefilled, since this
    /// workspace was opened.
    pub fn section_loads(&self) -> SectionLoads {
        SectionLoads {
            restored: self.section_loads.restored.load(Ordering::Relaxed),
            prefilled: self.section_loads.prefilled.load(Ordering::Relaxed),
        }
    }

    /// Look up a section by the human-readable `debug_name` recorded
    /// on its `SectionDecl`.  Wrapper around
    /// [`Substrate::section_id_for_debug_name`].  Used by calibration
    /// consumers that pick scenarios out of a loaded workspace by id.
    pub fn section_id_for_debug_name(&self, debug_name: &str) -> Option<SectionId> {
        self.read().section_id_for_debug_name(debug_name)
    }
}
