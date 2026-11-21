//! Reads and writes a conversation's metadata: its label, conv_id, custom bag and
//! fork lineage.
//!
//! Each setter is last-write-wins and persists to the redo log, and
//! `update_conversation_metadata` rewrites one key from the value it holds, as one
//! step. A fork records which timeline a child continues and refuses a cycle. The
//! seed alias lets a conversation continue whichever priming chain the member
//! holds. `inherited_chain` returns the lineage oldest first, which projections
//! walk to treat an ancestor's turns as the child's opening.

use super::*;

impl ConversationEngine {
    /// Persist a sidebar label for `timeline` to the workspace substrate.
    /// Last-write-wins; preserves whatever `conv_id` is already known
    /// for this timeline. The daemon's titler is the typical caller.
    pub fn set_conversation_label(&self, timeline: TimelineId, label: &str) -> crate::Result<()> {
        self.conversation
            .set_conversation_label(timeline, label)
            .map_err(ConversationError::Model)
    }

    /// Persist the client-supplied `conv_id` for `timeline`. Idempotent;
    /// callers can invoke on every submit. Preserves any existing label.
    /// This is the "substrate-as-single-source-of-truth" replacement for
    /// the old daemon-side `conv_labels.json` sidecar — the conv_id ↔
    /// timeline mapping now lives in the redo log.
    pub fn set_conversation_conv_id(
        &self,
        timeline: TimelineId,
        conv_id: &str,
    ) -> crate::Result<()> {
        self.conversation
            .set_conversation_conv_id(timeline, conv_id)
            .map_err(ConversationError::Model)
    }

    /// Read the workspace substrate's sidebar label for `timeline`, or
    /// `None` if none has been recorded. Useful for "should we still run
    /// the titler?" checks at submit time.
    pub fn conversation_label_of(&self, timeline: TimelineId) -> Option<String> {
        self.conversation.label_of(timeline)
    }

    /// Merge a `(key, value)` into `timeline`'s free-form `custom`
    /// metadata bag and persist it. Used by utility ingests to tag each
    /// conversation with a content hash + descriptive fields for the
    /// restart-resume cache.
    pub fn set_conversation_metadata(
        &self,
        timeline: TimelineId,
        key: &str,
        value: &str,
    ) -> crate::Result<()> {
        self.conversation
            .set_conversation_metadata(timeline, key, value)
            .map_err(ConversationError::Model)
    }

    /// Merge several `(key, value)` pairs into `timeline`'s `custom` metadata in
    /// one persisted record, so values that describe one decision — a turn's
    /// composer dials — are never recovered half-written.
    pub fn set_conversation_metadata_many(
        &self,
        timeline: TimelineId,
        kv: &std::collections::BTreeMap<String, String>,
    ) -> crate::Result<()> {
        self.conversation
            .set_conversation_metadata_many(timeline, kv)
            .map_err(ConversationError::Model)
    }

    /// `timeline`'s `custom` metadata bag, or `None` if unregistered.
    pub fn conversation_metadata(
        &self,
        timeline: TimelineId,
    ) -> Option<std::collections::BTreeMap<String, String>> {
        self.conversation.conversation_metadata(timeline)
    }

    /// Record that `child` continues `parent` — the durable fork lineage a
    /// projection walks to treat an ancestor's turns as the child's own
    /// opening (`Substrate::inherited_chain`).
    ///
    /// This is metadata only. Nothing is copied and nothing has to stay
    /// resident: the ancestor's K/V is fetched from whatever tier it is on,
    /// by the same working-set elevation every other selected turn uses.
    ///
    /// A parent that is `child` itself or one of its descendants is refused
    /// with [`ConversationError::ForkCycle`]: the lineage would loop, and
    /// every projection of a timeline on the loop would lose the ancestors
    /// past it.
    ///
    /// The cycle check and the write are one step, so two forks recorded at
    /// once cannot each pass the check against the lineage the other is
    /// about to change.
    pub fn set_forked_from(&self, child: TimelineId, parent: TimelineId) -> crate::Result<()> {
        let mut cycle = false;
        self.conversation.update_conversation_metadata(
            child,
            Substrate::FORKED_FROM_KEY,
            |s, _| {
                cycle = s.lineage_contains(parent, child);
                (!cycle).then(|| parent.raw().to_string())
            },
        );
        if cycle {
            return Err(ConversationError::ForkCycle {
                child: child.raw(),
                parent: parent.raw(),
            });
        }
        Ok(())
    }

    /// Record that `child` continues **the seed** ([`Substrate::SEED_PARENT`]):
    /// whatever priming chain the member reading it holds, resolved where the
    /// lineage is read, so the conversation opens on the current chain on
    /// every member and after every boot.
    pub fn set_forked_from_seed(&self, child: TimelineId) -> crate::Result<()> {
        self.set_conversation_metadata(child, Substrate::FORKED_FROM_KEY, Substrate::SEED_PARENT)
    }

    /// Name `seed` — the priming chain's end this boot built — as what
    /// [`Substrate::SEED_PARENT`] resolves to on this member.
    pub fn set_seed(&self, seed: Option<TimelineId>) {
        self.conversation.update(|s| s.set_seed(seed));
    }

    /// Re-point every conversation parented on a base like `base`
    /// ([`Substrate::parented_on_a_base`]) at the seed. Returns how many were
    /// re-pointed. The rewrite is metadata, so it replicates: a member that
    /// never held the base reads the alias, and the chain it holds.
    pub fn reseed_base_children(&self, base: TimelineId) -> crate::Result<usize> {
        let children = self.conversation.read().parented_on_a_base(base);
        for child in &children {
            self.set_forked_from_seed(*child)?;
        }
        Ok(children.len())
    }

    /// Rewrite `key` of `timeline`'s `custom` metadata from the value it
    /// holds, as one step — see
    /// [`Conversation::update_conversation_metadata`]. Returns what was stored.
    pub fn update_conversation_metadata(
        &self,
        timeline: TimelineId,
        key: &str,
        next: impl FnMut(&Substrate, Option<&str>) -> Option<String>,
    ) -> Option<String> {
        self.conversation
            .update_conversation_metadata(timeline, key, next)
    }

    /// Record that `timeline` continues nothing: whatever parent it recorded
    /// before is dropped, and its projection inherits no ancestor. The empty
    /// value reads as absent ([`Substrate::parent_timeline`](crate::substrate::Substrate::parent_timeline)).
    pub fn clear_forked_from(&self, timeline: TimelineId) -> crate::Result<()> {
        self.set_conversation_metadata(timeline, Substrate::FORKED_FROM_KEY, "")
    }

    /// Every live conversation whose `custom` metadata contains `key == value`.
    /// The content-addressed lookup utility ingests use after substrate
    /// load to skip rebuilding units already present (tombstoned excluded).
    pub fn find_conversations_by_metadata(&self, key: &str, value: &str) -> Vec<TimelineId> {
        self.conversation.find_timelines_by_metadata(key, value)
    }

    /// [`Self::find_conversations_by_metadata`] for many values in one pass: a
    /// value no live conversation carries is absent from the map.
    pub fn find_conversations_by_metadata_values(
        &self,
        key: &str,
        values: &BTreeSet<String>,
    ) -> BTreeMap<String, Vec<TimelineId>> {
        self.conversation
            .find_timelines_by_metadata_values(key, values)
    }

    /// Whether a live conversation whose K/V is intact carries `key ==
    /// value` — one membership of [`Self::conversation_metadata_values`],
    /// stopping at the first match.
    pub fn has_live_intact_metadata(&self, key: &str, value: &str) -> bool {
        self.conversation.has_live_intact_metadata(key, value)
    }

    /// One value of `timeline`'s metadata, without copying the bag.
    pub fn conversation_metadata_value(&self, timeline: TimelineId, key: &str) -> Option<String> {
        self.conversation.conversation_metadata_value(timeline, key)
    }

    /// Whether `timeline`'s metadata holds `key`.
    pub fn has_conversation_metadata_key(&self, timeline: TimelineId, key: &str) -> bool {
        self.conversation
            .has_conversation_metadata_key(timeline, key)
    }

    /// `f` over `timeline`'s metadata, read in place — for a reader of
    /// several keys.
    pub fn conversation_metadata_with<T>(
        &self,
        timeline: TimelineId,
        f: impl FnOnce(&POrdMap<String, String>) -> T,
    ) -> Option<T> {
        self.conversation.conversation_metadata_with(timeline, f)
    }

    /// The fork lineage `timeline` inherits, oldest ancestor first, ending at
    /// `timeline` itself.
    pub fn inherited_chain(&self, timeline: TimelineId) -> Vec<TimelineId> {
        self.conversation.read().inherited_chain(timeline)
    }

    /// [`Self::find_conversations_by_metadata`] plus tombstoned conversations
    /// that carry a distillation mode — the provenance corpus, whose designed
    /// end state is archived + distilled + tombstoned. Ordinary tombstones stay
    /// excluded. Used by the calibration resume filter.
    pub fn find_conversations_by_metadata_including_distilled(
        &self,
        key: &str,
        value: &str,
    ) -> Vec<TimelineId> {
        self.conversation
            .find_timelines_by_metadata_including_distilled(key, value)
    }

    /// One-pass snapshot of the distinct `custom[key]` values across live
    /// conversations — for O(1) resume-cache membership probing.
    pub fn conversation_metadata_values(&self, key: &str) -> HashSet<String> {
        self.conversation.metadata_values_for_key(key)
    }

    /// Live conversations carrying `key`, paired with its value. Drives
    /// ingest reconciliation (tombstone units whose source file is gone).
    pub fn conversations_with_metadata_key(&self, key: &str) -> Vec<(TimelineId, String)> {
        self.conversation.timelines_with_metadata_key(key)
    }

    /// The name a conversation was written under (its `conv_id`), if it was
    /// given one.
    pub fn conversation_conv_id(&self, timeline: TimelineId) -> Option<String> {
        self.conversation.conv_id_of(timeline)
    }
}
