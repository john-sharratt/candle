//! Lists and tombstones the engine's conversations, and reads their file and turn
//! state.
//!
//! Listings come from the substrate's registry. `conversations_with_conv_id_prefix`
//! serves a caller that names its conversations by scheme, so it pays only for its
//! own members. Tombstoning retires a timeline or a single turn, and the lookups
//! check for a tombstone before a write. Conversation state holds the archived
//! flag, the branch per repository, and the file events on the timeline, which
//! replay in order to rebuild the conversation's files.

use super::*;

impl ConversationEngine {
    /// `timeline`'s recorded token total, when one was recorded.
    pub fn timeline_token_total(&self, timeline: TimelineId) -> usize {
        self.conversation.read().timeline_token_total(timeline)
    }

    /// Every conversation the workspace substrate knows about —
    /// `(timeline, conv_id, label, archived, order)` tuples, where `order`
    /// is the creation-order rank ([`crate::substrate::TimelineEntry::order`]).
    /// Drives the daemon's `GET /v1/conversations` sidebar listing directly.
    pub fn known_conversations(&self) -> Vec<(TimelineId, String, String, bool, u64)> {
        self.conversation.known_conversations()
    }

    /// Every live conversation, whether or not it carries a `conv_id` —
    /// tombstoned ones excluded.
    ///
    /// [`Self::known_conversations`] lists only the named ones, which is what a
    /// sidebar wants and not what a caller retiring everything wants: an
    /// ingested document is a conversation with no `conv_id` at all.
    pub fn live_conversations(&self) -> Vec<TimelineId> {
        self.conversation.live_timeline_ids()
    }

    /// Live conversations whose `conv_id` starts with `prefix`, as
    /// `(timeline, conv_id)` — tombstoned ones excluded.
    ///
    /// For a caller that names its conversations by a scheme rather than
    /// listing them, and wants one scheme's members without paying for the
    /// whole registry: [`Self::known_conversations`] materialises every
    /// conversation the workspace has ever held, and that grows for the life of
    /// the log while the live set stays bounded. `npcd` finds one character's
    /// `npc-<id>-day-*` conversations this way.
    pub fn conversations_with_conv_id_prefix(&self, prefix: &str) -> Vec<(TimelineId, String)> {
        self.conversation.conversations_with_conv_id_prefix(prefix)
    }

    /// Set the branch a conversation works on in each repository `branches`
    /// names (repository workspace name → branch), leaving any other
    /// repository's as it is. Persisted with the rest of the conversation's
    /// state as one `RecordType::ConvState` record (last-writer-wins); a call
    /// that changes nothing writes nothing.
    pub fn set_conversation_branches(
        &self,
        timeline: TimelineId,
        branches: &BTreeMap<String, String>,
    ) {
        self.conversation
            .set_conversation_branches(timeline, branches)
    }

    /// Stage a set of a conversation's changes to its files as events on its
    /// timeline, events before tombstones, for the group commit to make
    /// durable — see `docs/zend_vfs_events.md`. Returns the sequence numbers
    /// the events took, in order.
    pub fn write_conversation_files(
        &self,
        timeline: TimelineId,
        write: &VfsWrite,
    ) -> crate::Result<Vec<VfsEventId>> {
        self.conversation
            .write_conversation_files(timeline, write)
            .map_err(ConversationError::Model)
    }

    /// Every live event of a conversation's files, in `(repo, key)` order and
    /// within a key in stamp order — what building the conversation again
    /// replays.
    pub fn conversation_files(&self, timeline: TimelineId) -> crate::Result<Vec<VfsEventPayload>> {
        self.conversation
            .conversation_files(timeline)
            .map_err(ConversationError::Model)
    }

    /// A conversation's state — archived flag and the branch it works on in
    /// each repository — or `None` for an unknown timeline.
    pub fn conversation_state(&self, timeline: TimelineId) -> Option<ConvState> {
        self.conversation.conversation_state(timeline)
    }

    /// Toggle the archived lifecycle flag for a conversation. Persists
    /// to the redo log as `RecordType::ConvState` (last-writer-wins)
    /// and updates the in-RAM substrate. Drives the daemon's
    /// `POST /v1/conversations/{id}/archive` and `/unarchive`.
    pub fn set_conversation_archived(
        &self,
        timeline: TimelineId,
        archived: bool,
    ) -> crate::Result<()> {
        self.conversation
            .set_conversation_archived(timeline, archived)
            .map_err(ConversationError::Model)
    }

    /// Mark a conversation used now — the sidebar lists it first until another
    /// is used. Persists its `ConvState`.
    pub fn touch_conversation(&self, timeline: TimelineId) {
        self.conversation.touch_conversation(timeline);
    }

    /// The group `timeline` belongs to, and its creation-order rank
    /// ([`crate::substrate::TimelineEntry::order`]) — `None` for a timeline
    /// this substrate does not hold.
    pub fn timeline_group_and_order(&self, timeline: TimelineId) -> Option<(GroupId, u64)> {
        self.conversation
            .read()
            .timeline_entry(timeline)
            .map(|e| (e.group, e.order))
    }

    /// Whether `timeline` is archived. Unlike [`Self::known_conversations`]
    /// — which omits internal conversations that never set a `conv_id` — this
    /// reads the flag directly, so it works for reserved/utility timelines too.
    pub fn is_conversation_archived(&self, timeline: TimelineId) -> bool {
        self.conversation.is_conversation_archived(timeline)
    }

    /// Tombstone `timeline` — see
    /// [`crate::projection::Conversation::tombstone_timeline`].
    ///
    /// Sections the timeline owns are released with it, and the scheduler's
    /// tables keyed by them are cleared on the wave thread.
    pub fn tombstone_timeline(&self, timeline: TimelineId) -> crate::Result<()> {
        let owned = self
            .conversation
            .release_owned_sections(timeline)
            .map_err(ConversationError::Model)?;
        self.retire_scheduler_sections(owned);
        self.conversation
            .tombstone_timeline(timeline)
            .map_err(ConversationError::Model)
    }

    /// Clear the scheduler's per-section tables for `sections` — see
    /// [`SchedulerRequest::RetireSections`].
    pub(crate) fn retire_scheduler_sections(&self, sections: Vec<SectionId>) {
        if sections.is_empty() {
            return;
        }
        let _ = self
            .core
            .scheduler_tx
            .send(SchedulerRequest::RetireSections {
                conversation: self.conversation.clone(),
                sections,
            });
    }

    /// Tombstone one turn of a live timeline — see
    /// [`crate::projection::Conversation::tombstone_turn`].
    pub fn tombstone_turn(&self, timeline: TimelineId, turn_index: u32) -> crate::Result<()> {
        self.conversation
            .tombstone_turn(timeline, turn_index)
            .map_err(ConversationError::Model)
    }

    /// Whether `(timeline, turn)` was already dropped by a turn-scoped
    /// tombstone.
    pub fn is_turn_tombstoned(&self, timeline: TimelineId, turn_index: u32) -> bool {
        self.conversation.is_turn_tombstoned(timeline, turn_index)
    }

    /// Whether the whole of `timeline` has been tombstoned.
    ///
    /// The companion to [`Self::is_turn_tombstoned`], and what a caller about
    /// to retire a conversation consults so an already-dead one costs a lookup
    /// rather than another record.
    pub fn is_timeline_tombstoned(&self, timeline: TimelineId) -> bool {
        self.conversation.is_timeline_tombstoned(timeline)
    }

    /// How many turns `timeline` actually holds — the substrate's own count.
    ///
    /// **The number a retention sweep must reckon against.** `Sequence::turn_count`
    /// is a different quantity: it is a per-sequence diagnostic that counts a
    /// user and an assistant message separately (`+= 2` an exchange) and resets
    /// whenever a process opens a fresh sequence. Turn *indices* advance once
    /// per exchange, so a horizon computed from that counter runs at twice the
    /// rate of the turns it indexes — past ~65 exchanges it exceeds the last
    /// real turn and retires the entire conversation, which is `keep_turns`
    /// meaning its own opposite.
    ///
    /// This counts turns the way the index does, so a horizon derived from it
    /// names a turn that exists.
    pub fn timeline_turn_count(&self, timeline: TimelineId) -> u64 {
        self.conversation.read().turn_count(timeline) as u64
    }

    /// Every live conversation written to `group` — the set selection reads,
    /// so archived and tombstoned conversations are not in it.
    pub fn group_conversations(&self, group: GroupId) -> Vec<TimelineId> {
        self.conversation
            .read()
            .active_timelines_for_group(group)
            .collect()
    }

    /// The `(layer, group)` `timeline` was written to — fixed when it was
    /// created. `None` for a timeline this engine does not hold.
    pub fn timeline_target(&self, timeline: TimelineId) -> Option<(LayerId, GroupId)> {
        self.conversation.timeline_target(timeline)
    }

    /// The wall-clock milliseconds each declared turn of `timeline` was written
    /// at, in turn order — read from the hybrid logical clock its declaration
    /// carries, so a turn replicated from another member keeps the time it was
    /// written there.
    pub fn timeline_turn_wall_ms(&self, timeline: TimelineId) -> Vec<(TurnIndex, u64)> {
        let read = self.conversation.read();
        read.turn_streams_of(timeline.raw())
            .map(|(turn, _, rt)| {
                (
                    turn,
                    crate::persistence::identity::wall_ms(rt.ids.decl.stamp().hlc),
                )
            })
            .collect()
    }
}
