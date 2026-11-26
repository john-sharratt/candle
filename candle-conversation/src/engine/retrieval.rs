//! Retrieval state for projections: working sets, retrieval scopes, and the turn
//! reads they depend on.
//!
//! A working set is the set of member conversations a dialogue may draw on this
//! turn. It is seeded at a momentum and bounded by a share of the budget, and the
//! files read this turn are withheld from it until the turn ends. A retrieval scope
//! names the conversations a scoped group may offer a target. The diagnostics
//! return the last score-density selection for a timeline, for the test harness.

use super::*;

impl ConversationEngine {
    /// Test-harness diagnostic — the most recent score-density
    /// [`SelectionDiagnostics`] for `timeline`, or `None` if no
    /// projection has run yet (or projection used the rule-based
    /// path).  Last-write-wins across reprojections within a turn.
    pub fn last_selection_diagnostics(&self, timeline: TimelineId) -> Option<SelectionDiagnostics> {
        self.conversation.last_selection_diagnostics(timeline)
    }

    /// Whether every turn of `timeline` holds its content, so the conversation
    /// can be placed whole ([`Substrate::holds_whole`](crate::substrate::Substrate::holds_whole)).
    pub fn holds_whole(&self, timeline: TimelineId) -> bool {
        self.conversation.read().holds_whole(timeline)
    }

    /// Whether some turn of `timeline` is a gap: another member's turn whose
    /// records are still arriving — the only kind of hole catch-up fills.
    pub fn holds_gap(&self, timeline: TimelineId) -> bool {
        self.conversation.read().holds_gap(timeline)
    }

    /// Seed `target`'s pool with `candidates` at `momentum`, in list order
    /// while the pool has room. Returns the candidates that did not enter.
    pub fn working_set_seed(
        &self,
        target: TimelineId,
        candidates: &[TimelineId],
        momentum: f32,
    ) -> Vec<TimelineId> {
        self.conversation
            .update(|s| s.working_set_seed(target, candidates, momentum))
    }

    /// Keep `reads` out of `target`'s working set until
    /// [`Self::working_set_clear_withheld`]: the files they read are verbatim
    /// in its dialogue, read for real this turn. A member among them stays.
    pub fn working_set_withhold(&self, target: TimelineId, reads: &[TimelineId]) {
        self.conversation
            .update(|s| s.working_set_withhold(target, reads));
    }

    /// Keep `files`, which `target`'s turn read for real, out of its knowledge
    /// base until its next question ([`Self::working_set_clear_withheld`]).
    pub fn files_withhold(&self, target: TimelineId, files: &[FileKey]) {
        self.conversation
            .update(|s| s.files_withhold(target, files));
    }

    /// Pin what `target`'s knowledge base holds: nothing, or one file whole
    /// (`docs/provenance_system_collection.md` §12 D11). Set before the
    /// conversation's first turn, which places it.
    pub fn files_pin(&self, target: TimelineId, pin: FilesPin) {
        self.conversation.update(|s| s.files_pin(target, pin));
    }

    /// Let `target`'s withheld conversations and files be placed again: the
    /// turn that read them has ended.
    pub fn working_set_clear_withheld(&self, target: TimelineId) {
        self.conversation
            .update(|s| s.working_set_clear_withheld(target));
    }

    /// A copy of `target`'s working set, when it has one.
    pub fn working_set(&self, target: TimelineId) -> Option<WorkingSet> {
        self.conversation.read().working_set(target).cloned()
    }

    /// Forget `target`'s working set, so it is rebuilt from the conversation's
    /// own history.
    pub fn working_set_clear(&self, target: TimelineId) {
        self.conversation.update(|s| s.working_set_clear(target));
    }

    /// Every turn's tags on `timeline`, oldest first — the record the working
    /// set's locks are rebuilt from.
    pub fn turn_tag_lists(&self, timeline: TimelineId) -> Vec<Vec<String>> {
        self.indexed_turn_tags(timeline)
            .into_iter()
            .map(|(_, tags)| tags)
            .collect()
    }

    /// Every turn's index and tags on `timeline`, oldest first — what says
    /// which turn carries each working-set lock.
    pub fn indexed_turn_tags(&self, timeline: TimelineId) -> Vec<(TurnIndex, Vec<String>)> {
        let view = self.conversation.read();
        let mut indices: Vec<_> = view.turn_indices(timeline).collect();
        indices.sort_by_key(|i| i.0);
        indices
            .into_iter()
            .map(|idx| (idx, view.turn_tags(timeline, idx)))
            .collect()
    }

    /// Every turn's `(user, assistant)` text on `timeline`, oldest first.
    pub fn turn_texts(&self, timeline: TimelineId) -> Vec<(String, String)> {
        let view = self.conversation.read();
        let mut indices: Vec<_> = view.turn_indices(timeline).collect();
        indices.sort_by_key(|i| i.0);
        indices
            .into_iter()
            .map(|idx| {
                (
                    view.user_text_of(timeline, idx),
                    view.assistant_text_of(timeline, idx),
                )
            })
            .collect()
    }

    /// The request `timeline`'s next turn continues: its first turn and each of
    /// its turns' assistant text, in order. `None` when the next turn opens a
    /// request of its own.
    pub fn request_replies(&self, timeline: TimelineId) -> Option<(TurnIndex, Vec<String>)> {
        let view = self.conversation.read();
        let turns = view.request_turns(timeline);
        let first = *turns.first()?;
        Some((
            first,
            turns
                .into_iter()
                .map(|idx| view.assistant_text_of(timeline, idx))
                .collect(),
        ))
    }

    /// Offer `group`'s conversations to a projection only as its target's
    /// scope names them ([`Self::set_retrieval_scope`]) — for ingested
    /// content that is only right for a conversation whose own files hold the
    /// same. Idempotent; in-memory, so it is marked at every setup.
    pub fn mark_group_scoped(&self, group: GroupId) {
        self.conversation.update(|s| s.mark_group_scoped(group));
    }

    /// Record which share of a dialogue's working-set budget `group`'s
    /// conversations draw on — what the schema's `working_set` selection rule
    /// declares for it. A conversation in no such group is never a working-set
    /// member. Idempotent; in-memory, so it is set at every setup.
    pub fn set_working_set_share(&self, group: GroupId, share: WorkingSetShare) {
        self.conversation
            .update(|s| s.set_working_set_share(group, share));
    }

    /// Cap each of `types` at its share of a dialogue's working-set budget,
    /// a conversation's type read from the subject its metadata holds under
    /// `subject_key` (`docs/zend_working_set.md` §4.10). In-memory, so it is
    /// set at every setup.
    pub fn set_working_set_types(&self, types: TypeShares, subject_key: &str) {
        // The edit may run more than once, so each run takes its own copy.
        self.conversation
            .update(|s| s.set_working_set_types(types.clone(), subject_key));
    }

    /// Record that `drawing` places units of `from`'s conversations — what a
    /// group's `draws:` declares (`docs/guidance_layer.md` §1). Idempotent;
    /// in-memory, so it is set at every setup.
    pub fn set_group_draws(&self, drawing: GroupId, from: GroupId) {
        self.conversation
            .update(|s| s.set_group_draws(drawing, from));
    }

    /// Name the conversations the scoped `group` may offer `target`, replacing
    /// what it named before. Both the belief scan and selection honour it.
    pub fn set_retrieval_scope(
        &self,
        target: TimelineId,
        group: GroupId,
        allowed: Arc<HashSet<TimelineId>>,
    ) {
        self.conversation
            .update(|s| s.set_retrieval_scope(target, group, Arc::clone(&allowed)));
    }

    /// Forget every scope `target` was given.
    pub fn clear_retrieval_scope(&self, target: TimelineId) {
        self.conversation
            .update(|s| s.clear_retrieval_scope(target));
    }

    /// Both halves of every turn in `timeline`, in order — `(user, assistant)`,
    /// verbatim as stored.
    pub fn conversation_texts(&self, timeline: TimelineId) -> Vec<(String, String)> {
        let read = self.conversation.read();
        (0..read.turn_count(timeline))
            .map(|i| {
                (
                    read.user_text_of(timeline, TurnIndex(i)),
                    read.assistant_text_of(timeline, TurnIndex(i)),
                )
            })
            .collect()
    }

    /// Whether any turn of `timeline` carries one of `tags` — the test a
    /// tag-scoped group applies to decide whether a conversation is in scope.
    pub fn conversation_carries(&self, timeline: TimelineId, tags: &[String]) -> bool {
        let read = self.conversation.read();
        (0..read.turn_count(timeline)).any(|i| {
            read.turn_tags(timeline, TurnIndex(i))
                .iter()
                .any(|t| tags.contains(t))
        })
    }
}
