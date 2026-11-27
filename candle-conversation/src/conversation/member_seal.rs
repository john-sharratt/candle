//! Sealing the knowledge base's members: a file's pieces, in order, as shared
//! sections no conversation owns (`docs/provenance_system_collection.md` §5).
//!
//! A chain is sealed behind the prompt before the `files` collection — the
//! birth prefix's sections up to and including `files_open`, so no tool, no
//! journal entry, and never the seed or the close that follow the collection
//! (§12 D4) — and each member behind the members before it: chunk *k* of a
//! file is sealed after its heading and chunks 1…*k*−1, its recurrence run on
//! from where chunk *k*−1's ended, so its K/V is computed as the continuation
//! of its file. A member's address is its content and everything it is sealed
//! after, so the same text after other text is another member, and a member
//! whose stream is already on disk is restored rather than prefilled again.
//!
//! The seal runs on the caller's thread, one member after another: each waits
//! for the one before it, whose end state it starts from.

use std::collections::HashSet;

use super::prefix_state::{chain_of, continue_state, state_after};
use super::{alloc_scratch_slot_on, Sequence};
use crate::error::ConversationError;
use crate::files::{files_owner, member_stream_name};
use crate::persistence::content_hash::{
    boundary_member_prefix, hash_tokens, section_stream_id, shared_section_prefix,
};
use crate::persistence::streams::{ContentAddress, StreamId};
use crate::projection::{
    prefix_before_collection, sections_prefix, DecodePriority, PrefixItem, SectionId,
};
use crate::scheduler::exported_state::SharedState;
use crate::scheduler::SchedulerRequest;
use crate::token_buffer::TokenBuffer;

/// One member to seal: the name it is declared under (after
/// `shared/files/`) and its text.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MemberSpec {
    pub name: String,
    pub content: String,
}

/// The policy a member's K/V is sealed under.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SealAs {
    /// A collection member's: the adaptive member policy.
    Member,
    /// A boundary section's: Q8 K and V, addressed apart from a member's
    /// ([`boundary_member_prefix`]).
    Boundary,
}

/// A member sealed into the substrate.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SealedMember {
    /// Its id in this process's substrate.
    pub section: SectionId,
    /// The stream it is persisted under.
    pub stream: StreamId,
    pub address: ContentAddress,
    /// Its token count.
    pub tokens: usize,
    /// Whether it was restored from its persisted stream rather than
    /// prefilled.
    pub restored: bool,
}

/// What of `before` — everything the schema declares ahead of a collection —
/// the birth prefix holds, in `before`'s order: collection members,
/// `depends_on` sections and empty ones left out, as the birth prefix leaves
/// them out, and every earlier range's hole kept.
fn in_birth_prefix(before: Vec<PrefixItem>, birth: &[PrefixItem]) -> Vec<PrefixItem> {
    let birth: HashSet<SectionId> = birth.iter().filter_map(|i| i.section()).collect();
    before
        .into_iter()
        .filter(|item| item.section().is_none_or(|s| birth.contains(&s)))
        .collect()
}

/// What a collection's members are sealed behind, and its reserved range.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MembersPrefix {
    /// The prompt before the collection, holes of earlier ranges included.
    pub prefix: Vec<PrefixItem>,
    /// The collection's range in positions, when it has one: a run of
    /// members is sealed at its top.
    pub range: Option<usize>,
}

impl MembersPrefix {
    /// What a member is sealed behind when its run's members up to and
    /// including it take `through` positions and `earlier` of them precede
    /// it: the prompt, the part of the range above the run left empty, then
    /// the earlier members.
    fn behind(&self, through: usize, earlier: &[SectionId]) -> Vec<PrefixItem> {
        let mut out = self.prefix.clone();
        if let Some(range) = self.range {
            let hole = range.saturating_sub(through);
            if hole > 0 {
                out.push(PrefixItem::Hole(hole));
            }
        }
        out.extend(earlier.iter().copied().map(PrefixItem::Section));
        out
    }
}

impl Sequence {
    /// The trained window of the engine this conversation runs on, which picks
    /// each reserved range's size; `None` before an engine reports one.
    pub fn native_context(&self) -> Option<usize> {
        self.substrate.read().trained_window()
    }

    /// The token ids `text` encodes to, as a member's seal encodes it: what a
    /// chunk cut checks its seams against (`files::cut_lines`).
    pub fn member_token_ids(&self, text: &str) -> crate::Result<Vec<u32>> {
        Ok(self
            .tokenizer
            .encode(text, false)
            .map_err(|e| ConversationError::Tokenizer(e.to_string()))?
            .get_ids()
            .to_vec())
    }

    /// What a member of `collection` is sealed behind: the birth prefix
    /// before the collection, in order — up to its range's hole when it has a
    /// range, holes of the ranges before it included — and the range.
    pub fn members_prefix(&self, collection: &str) -> crate::Result<MembersPrefix> {
        let base = &self.base_projection;
        let id = base.id_for_system_collection(collection).ok_or_else(|| {
            ConversationError::Other(format!("the schema has no collection {collection:?}"))
        })?;
        if let Some(range) = self.birth_ranges.iter().find(|r| r.collection == id) {
            return Ok(MembersPrefix {
                prefix: range.member_prefix(&self.birth_prefix, range.positions),
                range: range.sealed_at_top(),
            });
        }
        let native = self.substrate.read().trained_window();
        let before = prefix_before_collection(&base.schema().system_prompt.items, id, native)
            .ok_or_else(|| {
                ConversationError::Other(format!(
                    "collection {collection:?} is inside a section tree; its members are \
                     sealed behind the prompt before it, which a tree branches"
                ))
            })?;
        Ok(MembersPrefix {
            prefix: in_birth_prefix(before, &self.birth_prefix),
            range: None,
        })
    }

    /// Seal `chain` as members of `collection`, in order, each behind the
    /// collection's prefix ([`Self::members_prefix`]) and the members before
    /// it, and return them in the same order.
    pub fn seal_members(
        &self,
        collection: &str,
        chain: &[MemberSpec],
    ) -> crate::Result<Vec<SealedMember>> {
        self.seal_members_behind(&self.members_prefix(collection)?, chain, SealAs::Member)
    }

    /// Seal `chain` in order, each behind `prefix` and the members before it,
    /// under `kind`'s policy, and return them in the same order:
    /// [`Self::seal_members`] behind a prefix the caller names — the sections a
    /// prompt actually places ahead of a file, where the birth prefix leaves
    /// some of them out.
    ///
    /// With a range, the chain is sealed at the range's top: each member
    /// behind the part of the range above the chain through it left empty, so
    /// the chain ends at the top (`docs/placed_section_layout.md` §3.1b).
    pub fn seal_members_behind(
        &self,
        prefix: &MembersPrefix,
        chain: &[MemberSpec],
        kind: SealAs,
    ) -> crate::Result<Vec<SealedMember>> {
        let mut out: Vec<SealedMember> = Vec::with_capacity(chain.len());
        // The state the member just prefilled ended on: where the next one
        // starts. `None` after a restored member, whose end the next one
        // works out from the log.
        let mut handed: Option<SharedState> = None;
        for spec in chain {
            let sealed = self.seal_member(prefix, &out, spec, handed.take(), kind)?;
            handed = sealed.1;
            out.push(sealed.0);
        }
        Ok(out)
    }

    /// Seal one member after `before`, starting from `handed` when the member
    /// just before it was prefilled here; returns it and the state its
    /// prefill ended on.
    fn seal_member(
        &self,
        prefix: &MembersPrefix,
        before: &[SealedMember],
        spec: &MemberSpec,
        handed: Option<SharedState>,
        kind: SealAs,
    ) -> crate::Result<(SealedMember, Option<SharedState>)> {
        let encoding = self
            .tokenizer
            .encode(spec.content.as_str(), false)
            .map_err(|e| ConversationError::Tokenizer(e.to_string()))?;
        let tokens = TokenBuffer::from(encoding.get_ids());
        if tokens.is_empty() {
            return Err(ConversationError::Other(format!(
                "member {:?} has no tokens",
                spec.name
            )));
        }
        let earlier: Vec<SectionId> = before.iter().map(|m| m.section).collect();
        let through = before.iter().map(|m| m.tokens).sum::<usize>() + tokens.len();
        let sealed_after = prefix.behind(through, &earlier);
        let chain = chain_of(&self.substrate, &sealed_after).0;
        let address = ContentAddress {
            prefix_hash: match kind {
                SealAs::Member => shared_section_prefix(chain),
                SealAs::Boundary => boundary_member_prefix(chain),
            },
            section_hash: hash_tokens(&tokens),
        };
        let stream = section_stream_id(address);
        let section = self
            .substrate
            .allocate_owned_section()
            .ok_or_else(|| ConversationError::Other("owned section ids are exhausted".into()))?;
        // Named only once it holds its K/V: a member whose seal fails leaves
        // no name resolving to an empty section.
        let register = || {
            self.substrate
                .register_owned_section(files_owner(), &spec.name, section)
        };
        let member = SealedMember {
            section,
            stream,
            address,
            tokens: tokens.len(),
            restored: false,
        };

        if self.substrate.section_stream_is_persisted(stream) {
            let (tx, rx) = flume::bounded(1);
            self.scheduler_tx
                .send(SchedulerRequest::RestoreSection {
                    conversation: self.substrate.clone(),
                    section_id: section,
                    stream_id: stream,
                    tokens: tokens.clone(),
                    response_tx: tx,
                })
                .map_err(|_| ConversationError::SchedulerGone)?;
            match rx.recv().map_err(|_| ConversationError::SchedulerGone)? {
                Ok(_) => {
                    self.substrate.record_section_loads(1, 0);
                    register();
                    return Ok((
                        SealedMember {
                            restored: true,
                            ..member
                        },
                        None,
                    ));
                }
                Err(e) => tracing::warn!(
                    member = %spec.name,
                    "member restore failed ({e}) — sealing it afresh"
                ),
            }
        }

        self.substrate.record_section_loads(0, 1);
        let start = match handed {
            Some(state) => Some(state),
            None => {
                // The state behind the prompt and the empty part of the range,
                // then run on over the earlier members.
                let at_collection = state_after(
                    &self.scheduler_tx,
                    &self.substrate,
                    &prefix.behind(through, &[]),
                )?;
                continue_state(
                    &self.scheduler_tx,
                    &self.substrate,
                    at_collection,
                    &sections_prefix(&earlier),
                )?
            }
        };
        let slot = alloc_scratch_slot_on(&self.scheduler_tx, &self.substrate)?;
        let (tx, rx) = flume::bounded(1);
        let sent = self
            .scheduler_tx
            .send(SchedulerRequest::IngestSection {
                sequence_id: slot,
                section_id: section,
                prefix: sealed_after,
                tokens,
                address,
                debug_name: member_stream_name(&spec.name),
                in_collection: kind == SealAs::Member,
                start,
                band: DecodePriority::Low,
                response_tx: tx,
            })
            .map_err(|_| ConversationError::SchedulerGone);
        let sealed = sent.and_then(|()| rx.recv().map_err(|_| ConversationError::SchedulerGone)?);
        let _ = self
            .scheduler_tx
            .send(SchedulerRequest::FreeSequence { sequence_id: slot });
        let seal = sealed?;
        register();
        Ok((member, seal.end_state.as_ref().map(SharedState::from)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ids(raw: &[u32]) -> Vec<SectionId> {
        raw.iter().map(|&r| SectionId::new(r)).collect()
    }

    /// **A member is sealed behind what the birth prefix holds before its
    /// collection**: a tool member (5) and a `depends_on` section (6) ahead
    /// of the collection are left out, and the seed and the close the birth
    /// prefix holds after it (8, 9) are never in `before`.
    #[test]
    fn a_member_is_sealed_behind_the_birth_prefix_before_its_collection() {
        let mut before = sections_prefix(&ids(&[1, 2, 5]));
        before.push(PrefixItem::Hole(64));
        before.extend(sections_prefix(&ids(&[6, 7])));
        let birth = sections_prefix(&ids(&[1, 2, 7, 8, 9]));
        let mut want = sections_prefix(&ids(&[1, 2]));
        want.push(PrefixItem::Hole(64));
        want.extend(sections_prefix(&ids(&[7])));
        assert_eq!(in_birth_prefix(before, &birth), want);
    }

    /// **A chain in a range ends at its top**: each member behind the prompt,
    /// the part of the range above the chain through it, and its earlier
    /// members.
    #[test]
    fn a_chain_in_a_range_ends_at_its_top() {
        let prompt = sections_prefix(&ids(&[1, 2]));
        let ranged = MembersPrefix {
            prefix: prompt.clone(),
            range: Some(1_000),
        };
        let mut want = prompt.clone();
        want.push(PrefixItem::Hole(600));
        want.extend(sections_prefix(&ids(&[10, 11])));
        assert_eq!(ranged.behind(400, &ids(&[10, 11])), want);
        // A chain past the range leaves no hole.
        let mut full = prompt.clone();
        full.extend(sections_prefix(&ids(&[10])));
        assert_eq!(ranged.behind(1_200, &ids(&[10])), full);
        // Without a range, the prompt and the earlier members alone.
        let plain = MembersPrefix {
            prefix: prompt.clone(),
            range: None,
        };
        let mut bare = prompt;
        bare.extend(sections_prefix(&ids(&[10])));
        assert_eq!(plain.behind(400, &ids(&[10])), bare);
    }
}
