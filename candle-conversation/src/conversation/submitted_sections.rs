//! Sections submitted to one conversation after it opened.
//!
//! A schema's sections are sealed when a conversation is created and belong to
//! every conversation built from it. [`Sequence::submit_section`] adds one to a
//! collection for this conversation only: the section is tokenized and sealed
//! off the caller's thread, goes into the substrate's persisted streams under
//! an address salted with this conversation's timeline, and joins the
//! collection's projection candidates once it is sealed. It is released when
//! [`Sequence::remove_section`] is called or the timeline is tombstoned.
//!
//! The substrate's section map is shared by every conversation, so ids come from
//! the substrate's owned partition and ownership is recorded there — see
//! [`crate::projection::OwnedSections`]. This module holds the per-conversation
//! side: the handle a caller waits on, the list of submitted sections, and the
//! merge of the sealed ones into the schema the conversation projects with.

use std::sync::{Arc, Condvar, Mutex};

use flume::Sender;

use super::{alloc_scratch_slot_on, Sequence};
use crate::error::ConversationError;
use crate::persistence::content_hash::{
    hash_tokens, owned_section_prefix, section_stream_id, ContentChain,
};
use crate::persistence::streams::{ContentAddress, StreamId};
use crate::projection::{
    prefix_before_collection, Builder, CollectionId, Conversation, SectionId, TimelineId,
};
use crate::scheduler::SchedulerRequest;
use crate::token_buffer::TokenBuffer;

/// Where a submitted section is in its life.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum SectionState {
    /// Being tokenized and sealed.
    Pending,
    /// Sealed; `blocks` is how many KV blocks it occupies.
    Ready { blocks: usize },
    /// Sealing failed; the message says why.
    Failed(String),
    /// Released by the conversation.
    Removed,
}

/// The state of one submitted section, shared between the conversation and the
/// thread sealing it.
struct Settled {
    state: Mutex<SectionState>,
    changed: Condvar,
}

impl Settled {
    fn new() -> Arc<Self> {
        Arc::new(Self {
            state: Mutex::new(SectionState::Pending),
            changed: Condvar::new(),
        })
    }

    fn get(&self) -> SectionState {
        self.state.lock().unwrap().clone()
    }

    /// Move a pending section to `outcome`. A section already settled or
    /// removed keeps its state; returns whether the move happened.
    fn settle(&self, outcome: SectionState) -> bool {
        let mut state = self.state.lock().unwrap();
        if *state != SectionState::Pending {
            return false;
        }
        *state = outcome;
        self.changed.notify_all();
        true
    }

    fn remove(&self) {
        *self.state.lock().unwrap() = SectionState::Removed;
        self.changed.notify_all();
    }

    fn wait_settled(&self) -> SectionState {
        let mut state = self.state.lock().unwrap();
        while *state == SectionState::Pending {
            state = self.changed.wait(state).unwrap();
        }
        state.clone()
    }
}

/// A reference to a section submitted to a conversation.
///
/// Cheap to clone. [`Sequence::submit_section`] returns one immediately; the
/// section is sealing in the background until [`Self::wait`] returns or
/// [`Self::state`] reports it ready.
#[derive(Clone)]
pub struct SectionRef {
    section: SectionId,
    name: String,
    settled: Arc<Settled>,
}

impl SectionRef {
    /// The section's id in the substrate.
    pub fn section_id(&self) -> SectionId {
        self.section
    }

    /// The name it was submitted under, which is also what a collection's named
    /// selection pins it by.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Where the section is now.
    pub fn state(&self) -> SectionState {
        self.settled.get()
    }

    /// Block until the section is sealed, returning its block count. Fails when
    /// sealing failed or the section was removed first.
    pub fn wait(&self) -> crate::Result<usize> {
        match self.settled.wait_settled() {
            SectionState::Ready { blocks } => Ok(blocks),
            SectionState::Failed(why) => Err(ConversationError::Other(format!(
                "section {:?} failed to seal: {why}",
                self.name
            ))),
            SectionState::Removed => Err(ConversationError::Other(format!(
                "section {:?} was removed before it sealed",
                self.name
            ))),
            SectionState::Pending => unreachable!("wait_settled returns once settled"),
        }
    }
}

/// One submitted section as the conversation tracks it.
struct Submitted {
    collection: CollectionId,
    name: String,
    content: String,
    priority: f32,
    section: SectionId,
    settled: Arc<Settled>,
}

impl Submitted {
    fn handle(&self) -> SectionRef {
        SectionRef {
            section: self.section,
            name: self.name.clone(),
            settled: Arc::clone(&self.settled),
        }
    }
}

/// The sections submitted to a conversation, and the substrate releases it has
/// held back while a turn runs.
#[derive(Default)]
pub(super) struct SubmittedSections {
    entries: Vec<Submitted>,
    deferred: Vec<SectionId>,
}

impl SubmittedSections {
    fn position_of_name(&self, name: &str) -> Option<usize> {
        self.entries.iter().position(|e| e.name == name)
    }

    fn position_of_section(&self, section: SectionId) -> Option<usize> {
        self.entries.iter().position(|e| e.section == section)
    }

    /// Drop the entries that will never project — failed or removed — and return
    /// their ids so the caller can release the claim each still holds.
    fn prune(&mut self) -> Vec<SectionId> {
        let mut gone = Vec::new();
        self.entries.retain(|e| match e.settled.get() {
            SectionState::Failed(_) | SectionState::Removed => {
                gone.push(e.section);
                false
            }
            SectionState::Pending | SectionState::Ready { .. } => true,
        });
        gone
    }

    /// The schema to project with: `base`, plus every sealed submitted section in
    /// its collection. `base` itself when none are sealed yet.
    fn merged_into(&self, base: &Arc<Builder>) -> Arc<Builder> {
        let ready: Vec<&Submitted> = self
            .entries
            .iter()
            .filter(|e| matches!(e.settled.get(), SectionState::Ready { .. }))
            .collect();
        if ready.is_empty() {
            return Arc::clone(base);
        }
        let mut merged = (**base).clone();
        for e in ready {
            if let Err(err) = merged.add_section_to_collection_with_id(
                e.collection,
                e.section,
                &e.name,
                &e.content,
                e.priority,
            ) {
                tracing::warn!(
                    section = %e.name,
                    "submitted section left out of the projection: {err}"
                );
            }
        }
        Arc::new(merged)
    }
}

/// Everything the thread sealing one section needs, so it does not borrow the
/// conversation.
struct SealJob {
    scheduler_tx: Sender<SchedulerRequest>,
    substrate: Conversation,
    tokenizer: Arc<tokenizers::Tokenizer>,
    schema: Arc<Builder>,
    owner: TimelineId,
    section: SectionId,
    name: String,
    content: String,
    prefix: Vec<SectionId>,
    settled: Arc<Settled>,
}

impl SealJob {
    fn run(self) {
        match self.seal() {
            Ok((blocks, stream)) => {
                if self.substrate.owned_section_named(self.owner, &self.name) != Some(self.section)
                {
                    self.discard();
                    self.settled.remove();
                } else {
                    self.settled.settle(SectionState::Ready { blocks });
                    self.release_superseded(stream);
                }
            }
            Err(e) => {
                tracing::warn!(section = %self.name, "submitted section failed to seal: {e}");
                self.settled.settle(SectionState::Failed(e.to_string()));
            }
        }
    }

    /// End the streams this owner persisted under this name for other content:
    /// the wording this section replaced before a restart, which nothing in
    /// memory claims.
    fn release_superseded(&self, stream: StreamId) {
        if let Err(e) = self.substrate.release_persisted_owned_sections(
            self.owner,
            Some(&self.name),
            Some(stream),
        ) {
            tracing::warn!(section = %self.name, "release of a superseded stream failed: {e}");
        }
    }

    /// Seal the section into the substrate, restoring it from the persisted
    /// stream when the same content was sealed under this address before.
    /// Returns the block count and the stream it lives in.
    fn seal(&self) -> crate::Result<(usize, StreamId)> {
        let encoding = self
            .tokenizer
            .encode(self.content.as_str(), false)
            .map_err(|e| ConversationError::Tokenizer(e.to_string()))?;
        let tokens = TokenBuffer::from(encoding.get_ids());
        if tokens.is_empty() {
            return Err(ConversationError::Other(format!(
                "section {:?} has no tokens",
                self.name
            )));
        }
        let address = ContentAddress {
            prefix_hash: owned_section_prefix(self.owner.raw(), &self.name, self.prefix_hash()),
            section_hash: hash_tokens(&tokens),
        };
        let stream = section_stream_id(address);

        if self.substrate.section_stream_is_persisted(stream) {
            let (tx, rx) = flume::bounded(1);
            self.scheduler_tx
                .send(SchedulerRequest::RestoreSection {
                    conversation: self.substrate.clone(),
                    section_id: self.section,
                    stream_id: stream,
                    tokens: tokens.clone(),
                    response_tx: tx,
                })
                .map_err(|_| ConversationError::SchedulerGone)?;
            match rx.recv().map_err(|_| ConversationError::SchedulerGone)? {
                Ok(chunks) => {
                    self.substrate.record_section_loads(1, 0);
                    return Ok((chunks, stream));
                }
                Err(e) => tracing::warn!(
                    section = %self.name,
                    "section restore failed ({e}) — sealing it afresh"
                ),
            }
        }

        self.substrate.record_section_loads(0, 1);
        let slot = alloc_scratch_slot_on(&self.scheduler_tx, &self.substrate)?;
        let (tx, rx) = flume::bounded(1);
        let sent = self
            .scheduler_tx
            .send(SchedulerRequest::IngestSection {
                sequence_id: slot,
                section_id: self.section,
                prefix_section_ids: self.prefix.clone(),
                tokens,
                address,
                debug_name: format!("conv/{}/{}", self.owner.raw(), self.name),
                in_collection: true,
                response_tx: tx,
            })
            .map_err(|_| ConversationError::SchedulerGone);
        let sealed = sent.and_then(|()| rx.recv().map_err(|_| ConversationError::SchedulerGone)?);
        let _ = self
            .scheduler_tx
            .send(SchedulerRequest::FreeSequence { sequence_id: slot });
        let seal = sealed?;
        Ok((seal.block_to.saturating_sub(seal.block_from), stream))
    }

    /// The content hash of the sections this one seals against — the same
    /// chain a section of the schema's collection is addressed under.
    fn prefix_hash(&self) -> crate::persistence::content_hash::ContentHash {
        let mut chain = ContentChain::new();
        let view = self.substrate.read();
        for &pid in &self.prefix {
            if self.schema.schema().system_prompt.is_collection_member(pid) {
                continue;
            }
            let tokens = view.section_tokens_of(pid);
            if tokens.is_empty() {
                continue;
            }
            chain.push_section(&tokens);
        }
        chain.prefix()
    }

    /// Release a section that was removed or tombstoned while it was sealing.
    /// The claim was already dropped, so it is retaken under a key of its own —
    /// the owner may have submitted a new section under the same name since.
    fn discard(&self) {
        let key = format!("{}#discarded-{}", self.name, self.section.raw());
        self.substrate
            .register_owned_section(self.owner, &key, self.section);
        if let Err(e) = self
            .substrate
            .release_owned_section(self.owner, self.section)
        {
            tracing::warn!(section = %self.name, "release of a discarded section failed: {e}");
        }
        let _ = self.scheduler_tx.send(SchedulerRequest::RetireSections {
            conversation: self.substrate.clone(),
            sections: vec![self.section],
        });
    }
}

impl Sequence {
    /// Submit a section to a collection of this conversation's schema and return
    /// at once.
    ///
    /// The section is sealed in the background — [`SectionRef::wait`] blocks
    /// until it is — and joins `collection`'s projection candidates from the
    /// next turn after it is sealed. Other conversations never see it. It is
    /// addressed in the substrate by this conversation's timeline, its `name`
    /// and the section's position in the prompt, so it is persisted like any
    /// sealed section: submitting the same `content` under the same `name` after
    /// a restart restores it from disk instead of prefilling it again.
    ///
    /// A collection selects its members by name, so the section can be pinned
    /// with a named selection. `name` is unique among this conversation's
    /// submitted sections and must not collide with a section of the schema;
    /// submitting a name again replaces the earlier section. The collection must
    /// be a top-level collection of the system prompt, not one inside a section
    /// tree.
    pub fn submit_section(
        &mut self,
        collection: &str,
        name: &str,
        content: &str,
        priority: f32,
    ) -> crate::Result<SectionRef> {
        if content.is_empty() {
            return Err(ConversationError::Other(format!(
                "section {name:?} has no content"
            )));
        }
        if priority <= 0.0 {
            return Err(ConversationError::Other(format!(
                "section {name:?} needs a priority above zero, got {priority}"
            )));
        }
        let base = Arc::clone(&self.base_projection);
        if base.id_for_system_section(name).is_some()
            && self.submitted.position_of_name(name).is_none()
        {
            return Err(ConversationError::Other(format!(
                "{name:?} is already a section of the schema"
            )));
        }
        let collection_id = base.id_for_system_collection(collection).ok_or_else(|| {
            ConversationError::Other(format!("the schema has no collection {collection:?}"))
        })?;
        let prefix = prefix_before_collection(&base.schema().system_prompt.items, collection_id)
            .ok_or_else(|| {
                ConversationError::Other(format!(
                    "collection {collection:?} is inside a section tree, which seals each \
                     member once per branch; submit to a top-level collection"
                ))
            })?;

        if let Some(at) = self.submitted.position_of_name(name) {
            let earlier = self.submitted.entries.remove(at);
            self.release_submitted(&earlier);
        }
        self.flush_deferred_releases();

        let owner = self.timeline_id();
        let section = self
            .substrate
            .allocate_owned_section()
            .ok_or_else(|| ConversationError::Other("owned section ids are exhausted".into()))?;
        self.substrate.register_owned_section(owner, name, section);

        let settled = Settled::new();
        let job = SealJob {
            scheduler_tx: self.scheduler_tx.clone(),
            substrate: self.substrate.clone(),
            tokenizer: Arc::clone(&self.tokenizer),
            schema: base,
            owner,
            section,
            name: name.to_string(),
            content: content.to_string(),
            prefix,
            settled: Arc::clone(&settled),
        };
        if let Err(e) = std::thread::Builder::new()
            .name("submitted-section-seal".into())
            .spawn(move || job.run())
        {
            let _ = self.substrate.release_owned_section(owner, section);
            return Err(ConversationError::Other(format!(
                "could not start the seal of section {name:?}: {e}"
            )));
        }

        let entry = Submitted {
            collection: collection_id,
            name: name.to_string(),
            content: content.to_string(),
            priority,
            section,
            settled,
        };
        let handle = entry.handle();
        self.submitted.entries.push(entry);
        Ok(handle)
    }

    /// Remove a submitted section from this conversation: it leaves the
    /// collection, its K/V is dropped from the substrate, and its stream is
    /// tombstoned so a restart does not restore it. Returns whether the
    /// conversation held it.
    ///
    /// While a turn is in flight the substrate keeps the K/V — the turn may be
    /// attending it — and drops it when the next turn is submitted.
    pub fn remove_section(&mut self, section: &SectionRef) -> bool {
        match self.submitted.position_of_section(section.section) {
            Some(at) => self.remove_at(at),
            None => false,
        }
    }

    /// [`Self::remove_section`] by the name the section was submitted under.
    ///
    /// A name this conversation has not submitted since it was opened may still
    /// have been sealed by an earlier run of the process; those streams are
    /// tombstoned too, and the call reports whether it found any.
    pub fn remove_section_named(&mut self, name: &str) -> bool {
        match self.submitted.position_of_name(name) {
            Some(at) => self.remove_at(at),
            None => {
                match self.substrate.release_persisted_owned_sections(
                    self.timeline_id(),
                    Some(name),
                    None,
                ) {
                    Ok(released) => released > 0,
                    Err(e) => {
                        tracing::warn!(section = %name, "release of a persisted section failed: {e}");
                        false
                    }
                }
            }
        }
    }

    /// The sections submitted to this conversation, in submission order.
    pub fn submitted_sections(&self) -> Vec<SectionRef> {
        self.submitted
            .entries
            .iter()
            .map(Submitted::handle)
            .collect()
    }

    fn remove_at(&mut self, at: usize) -> bool {
        let entry = self.submitted.entries.remove(at);
        self.release_submitted(&entry);
        self.rebuild_projection();
        true
    }

    /// Release `entry`'s claim and substrate residence, or hold the release back
    /// until the in-flight turn is over.
    fn release_submitted(&mut self, entry: &Submitted) {
        entry.settled.remove();
        if self.turn_in_flight {
            self.submitted.deferred.push(entry.section);
        } else {
            self.release_now(entry.section);
        }
    }

    fn release_now(&self, section: SectionId) {
        if let Err(e) = self
            .substrate
            .release_owned_section(self.timeline_id(), section)
        {
            tracing::warn!(?section, "release of a submitted section failed: {e}");
        }
        let _ = self.scheduler_tx.send(SchedulerRequest::RetireSections {
            conversation: self.substrate.clone(),
            sections: vec![section],
        });
    }

    fn flush_deferred_releases(&mut self) {
        if self.turn_in_flight {
            return;
        }
        for section in std::mem::take(&mut self.submitted.deferred) {
            self.release_now(section);
        }
    }

    /// Bring the projection up to date with the submitted sections: drop the
    /// failed ones, release anything held back for a turn that has ended, and
    /// merge in the ones that have sealed. Called when a turn is submitted.
    pub(super) fn sync_submitted_sections(&mut self) {
        for section in self.submitted.prune() {
            self.release_now(section);
        }
        self.flush_deferred_releases();
        self.rebuild_projection();
    }

    /// Recompute the projection as the base schema plus the sealed submitted
    /// sections.
    pub(super) fn rebuild_projection(&mut self) {
        self.projection = self.submitted.merged_into(&self.base_projection);
        self.publish_ask_state();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::projection::SelectionRule;

    const YAML: &str = r#"
system_prompt:
  sections:
    - id: alpha
      content: "alpha"
layers:
  - name: dialogue
    window: 4000
    summary:
      turns:
        max_tokens: 256
        user:
          system_prompt: compress
          user_prompt: compress
        assistant:
          system_prompt: compress
          user_prompt: compress
    score_formula: max
    budget:
      priority: 100
    groups:
      - id: convo
        selection:
          kind: conversation
          recent: 4
          historical_top_k: 4
"#;

    fn entry(collection: CollectionId, name: &str, raw: u32) -> Submitted {
        Submitted {
            collection,
            name: name.into(),
            content: format!("content of {name}"),
            priority: 50.0,
            section: SectionId::new(raw),
            settled: Settled::new(),
        }
    }

    fn base_with_collection() -> (Arc<Builder>, CollectionId) {
        let mut b = Builder::from_yaml(YAML).unwrap();
        let c = b
            .add_collection("mail", SelectionRule::AlwaysVisible, 0.0)
            .unwrap();
        (Arc::new(b), c)
    }

    #[test]
    fn a_section_settles_once_and_wait_reports_it() {
        let settled = Settled::new();
        let handle = SectionRef {
            section: SectionId::new(7),
            name: "inbox".into(),
            settled: Arc::clone(&settled),
        };
        assert_eq!(handle.state(), SectionState::Pending);
        assert!(settled.settle(SectionState::Ready { blocks: 3 }));
        assert!(!settled.settle(SectionState::Failed("late".into())));
        assert_eq!(handle.wait().unwrap(), 3);
        assert_eq!(handle.state(), SectionState::Ready { blocks: 3 });
    }

    #[test]
    fn wait_blocks_until_another_thread_settles() {
        let settled = Settled::new();
        let handle = SectionRef {
            section: SectionId::new(7),
            name: "inbox".into(),
            settled: Arc::clone(&settled),
        };
        let sealer = std::thread::spawn(move || {
            std::thread::sleep(std::time::Duration::from_millis(20));
            settled.settle(SectionState::Ready { blocks: 2 });
        });
        assert_eq!(handle.wait().unwrap(), 2);
        sealer.join().unwrap();
    }

    #[test]
    fn wait_reports_a_failed_or_removed_section() {
        let failed = Settled::new();
        failed.settle(SectionState::Failed("no tokens".into()));
        let handle = SectionRef {
            section: SectionId::new(7),
            name: "inbox".into(),
            settled: failed,
        };
        assert!(handle.wait().unwrap_err().to_string().contains("no tokens"));

        let removed = Settled::new();
        removed.remove();
        let handle = SectionRef {
            section: SectionId::new(8),
            name: "gone".into(),
            settled: removed,
        };
        assert!(handle.wait().unwrap_err().to_string().contains("removed"));
    }

    #[test]
    fn a_removed_section_stays_removed() {
        let settled = Settled::new();
        settled.remove();
        assert!(!settled.settle(SectionState::Ready { blocks: 1 }));
        assert_eq!(settled.get(), SectionState::Removed);
    }

    #[test]
    fn only_sealed_sections_merge_into_the_projection() {
        let (base, mail) = base_with_collection();
        let mut submitted = SubmittedSections::default();
        let sealed = entry(mail, "sealed", 0x8000_0001);
        let pending = entry(mail, "pending", 0x8000_0002);
        sealed.settled.settle(SectionState::Ready { blocks: 1 });
        submitted.entries.push(sealed);
        submitted.entries.push(pending);

        let merged = submitted.merged_into(&base);
        assert_eq!(
            merged.id_for_system_section("sealed"),
            Some(SectionId::new(0x8000_0001))
        );
        assert_eq!(merged.id_for_system_section("pending"), None);
        assert_eq!(base.id_for_system_section("sealed"), None);
    }

    #[test]
    fn nothing_sealed_leaves_the_base_schema_untouched() {
        let (base, mail) = base_with_collection();
        let mut submitted = SubmittedSections::default();
        submitted.entries.push(entry(mail, "pending", 0x8000_0001));
        assert!(Arc::ptr_eq(&submitted.merged_into(&base), &base));
    }

    #[test]
    fn merging_twice_gives_the_same_schema() {
        let (base, mail) = base_with_collection();
        let mut submitted = SubmittedSections::default();
        let e = entry(mail, "sealed", 0x8000_0001);
        e.settled.settle(SectionState::Ready { blocks: 1 });
        submitted.entries.push(e);
        let first = submitted.merged_into(&base);
        let second = submitted.merged_into(&base);
        assert_eq!(
            first.id_for_system_section("sealed"),
            second.id_for_system_section("sealed")
        );
    }

    #[test]
    fn pruning_drops_failed_and_removed_entries_and_reports_them() {
        let (_, mail) = base_with_collection();
        let mut submitted = SubmittedSections::default();
        let failed = entry(mail, "failed", 0x8000_0001);
        failed.settled.settle(SectionState::Failed("x".into()));
        let removed = entry(mail, "removed", 0x8000_0002);
        removed.settled.remove();
        let ready = entry(mail, "ready", 0x8000_0003);
        ready.settled.settle(SectionState::Ready { blocks: 1 });
        submitted
            .entries
            .extend([failed, removed, ready, entry(mail, "pending", 0x8000_0004)]);

        let gone = submitted.prune();
        assert_eq!(
            gone,
            vec![SectionId::new(0x8000_0001), SectionId::new(0x8000_0002)]
        );
        let kept: Vec<&str> = submitted.entries.iter().map(|e| e.name.as_str()).collect();
        assert_eq!(kept, ["ready", "pending"]);
    }

    #[test]
    fn entries_are_found_by_name_and_by_section() {
        let (_, mail) = base_with_collection();
        let mut submitted = SubmittedSections::default();
        submitted.entries.push(entry(mail, "a", 0x8000_0001));
        submitted.entries.push(entry(mail, "b", 0x8000_0002));
        assert_eq!(submitted.position_of_name("b"), Some(1));
        assert_eq!(
            submitted.position_of_section(SectionId::new(0x8000_0001)),
            Some(0)
        );
        assert_eq!(submitted.position_of_name("c"), None);
    }
}
