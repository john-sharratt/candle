//! Holds a conversation's lifecycle state: its archived flag, branch choices, debug
//! id and sidebar record.
//!
//! Every state record carries the whole state, so replay is last-writer-wins on the
//! complete state and a compaction keeps one record per timeline. The state written
//! is read after the identity is minted, so the record with the greatest identity
//! holds every change published before its mint. Setters are idempotent and write
//! nothing when nothing changes.

use super::*;

impl Conversation {
    /// Set a conversation's `archived` lifecycle flag and persist its
    /// state. Idempotent: if the substrate already holds the requested
    /// flag, nothing is written.
    pub fn set_conversation_archived(
        &self,
        timeline: TimelineId,
        archived: bool,
    ) -> candle::Result<()> {
        self.update_conv_state(timeline, |sub| sub.set_archived(timeline, archived));
        Ok(())
    }

    /// Mark a conversation used now and persist its state, so the sidebar
    /// ranks it above every conversation used before it.
    pub fn touch_conversation(&self, timeline: TimelineId) {
        self.update_conv_state(timeline, |sub| sub.touch_conversation(timeline));
    }

    /// Set the branch `timeline` works on in each repository `branches`
    /// names — repository name to branch — leaving any other repository's as
    /// it is, and persist the state once. Idempotent: when nothing changes,
    /// nothing is written.
    pub fn set_conversation_branches(
        &self,
        timeline: TimelineId,
        branches: &BTreeMap<String, String>,
    ) {
        self.update_conv_state(timeline, |sub| {
            let mut changed = false;
            for (repo, branch) in branches {
                changed |= sub.set_branch(timeline, repo, branch);
            }
            changed
        });
    }

    /// Stage `write` — a set of `timeline`'s changes to its files — as
    /// events on the timeline, events before tombstones, for the persistence
    /// thread's group commit to make durable. Returns the names the events
    /// took. A read-only substrate refuses: the
    /// caller mirrors what was written, and nothing was.
    pub fn write_conversation_files(
        &self,
        timeline: TimelineId,
        write: &VfsWrite,
    ) -> candle::Result<Vec<VfsEventId>> {
        if self.read_only {
            return Err(candle::Error::Msg(
                "write_conversation_files: the substrate is read-only".to_string(),
            ));
        }
        self.persistence
            .call(|p| p.write_vfs(timeline.raw(), write))
            .map_err(|e| candle::Error::Msg(format!("write_conversation_files: {e}")))
    }

    /// Every live event of `timeline`'s files, read back in
    /// `(repo, key, seq)` order.
    pub fn conversation_files(&self, timeline: TimelineId) -> candle::Result<Vec<VfsEventPayload>> {
        self.persistence
            .call(|p| p.vfs_events(timeline.raw()))
            .map_err(|e| candle::Error::Msg(format!("conversation_files: {e}")))
    }

    /// `timeline`'s conversation state — archived flag and branches — or
    /// `None` for an unregistered timeline.
    pub fn conversation_state(&self, timeline: TimelineId) -> Option<ConvState> {
        self.read().conv_state(timeline)
    }

    /// Apply `change` to the substrate and, when it reports a change, persist
    /// `timeline`'s whole resulting state as a `RecordType::ConvState` record.
    ///
    /// Every record carries the complete state, so replay is last-writer-wins
    /// on the whole of it and a later compaction keeps one record per
    /// timeline.
    ///
    /// **The state written is read after the identity is minted.** Two
    /// changes made at once would otherwise each write the state as their own
    /// edit left it, and the greater identity could carry the older state — a
    /// branch set, then lost to an archive that read the state before it. So
    /// the change is published first, the identity minted after, and the state
    /// read in a second edit that stamps it: whichever writer minted last read
    /// a state holding every change published before its mint, and every
    /// change's own mint follows its publication — so the record with the
    /// greatest identity holds them all.
    pub(super) fn update_conv_state(
        &self,
        timeline: TimelineId,
        mut change: impl FnMut(&mut Substrate) -> bool,
    ) {
        if !self.update(|s| change(s)) {
            return;
        }
        // Minted only for a change that will be written: an identity minted
        // and never written is a hole in this node's sequence.
        let identity = self.minter.mint(0);
        let state = self.update(|s| {
            s.stamp_conv_state(timeline, identity);
            s.conv_state(timeline)
        });
        let Some(state) = state else {
            return;
        };
        if self.read_only {
            return;
        }
        self.persistence.enqueue(WriteJob::ConvState {
            timeline: timeline.raw(),
            payload: manifest::encode_conv_state_payload(timeline.raw(), &state),
            identity,
        });
    }

    /// Whether `timeline` is currently archived. Untouched / unknown
    /// timelines return `false`.
    pub fn is_conversation_archived(&self, timeline: TimelineId) -> bool {
        self.read().is_archived(timeline)
    }

    /// Set the substrate-side resume key (`debug_id`) for `timeline`
    /// and persist a `RecordType::DebugId` record to the redo log.
    /// Last-write-wins on replay.  Idempotent: if the substrate
    /// already holds the requested key, the record is not written and
    /// the call returns `Ok(())` without touching the log.
    pub fn set_conversation_debug_id(
        &self,
        timeline: TimelineId,
        debug_id: &str,
    ) -> candle::Result<()> {
        if debug_id.is_empty() {
            return Ok(());
        }
        let identity = self.minter.mint(0);
        self.update(|s| s.set_debug_id(timeline, debug_id, identity));
        if self.read_only {
            return Ok(());
        }
        self.persistence
            .call(|p| p.write_debug_id(timeline.raw(), debug_id, identity))
            .map_err(|e| candle::Error::Msg(format!("write_debug_id: {e}")))?;
        Ok(())
    }

    /// Look up a timeline by `debug_id`.  O(1).
    pub fn lookup_by_debug_id(&self, debug_id: &str) -> Option<TimelineId> {
        self.read().lookup_by_debug_id(debug_id)
    }

    /// Number of turns currently waiting on the summariser thread to
    /// absorb them into the summary tree (§9 backpressure metric).
    /// `0` means steady state: the background tempo is keeping up
    /// with the foreground turn rate.
    pub fn pending_summary_len(&self, timeline: TimelineId) -> usize {
        self.read().pending_summary_len(timeline)
    }

    /// Most recent score-density [`SelectionDiagnostics`] for
    /// `timeline`, or `None` if no projection has run yet (or the
    /// projection used the rule-based path).  Pure test-harness
    /// instrumentation: the substrate retains only the latest
    /// selection per timeline, written by the scheduler at projection
    /// time.  Production daemons can ignore.
    pub fn last_selection_diagnostics(&self, timeline: TimelineId) -> Option<SelectionDiagnostics> {
        self.read().last_selection_of(timeline).cloned()
    }
}
