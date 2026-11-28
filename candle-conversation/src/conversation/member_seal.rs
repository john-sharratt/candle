//! Sealing the knowledge base's members: a file's sections, in order, as
//! shared sections no conversation owns (`docs/provenance_system_collection.md`
//! §5, `docs/placed_section_layout.md` §3.5b).
//!
//! A chain is sealed behind the prompt before the `files` collection — the
//! birth prefix's sections up to and including `files_open`, so no tool, no
//! journal entry, and never the close that follows the collection (§12 D4) —
//! and each member behind the members before it: section *k* of a file is
//! sealed after sections 1…*k*−1, its recurrence run on from where section
//! *k*−1's ended, so its K/V is computed as the continuation of its file. A member's address is its content and everything it is sealed
//! after, so the same text after other text is another member, and a member
//! whose stream is already on disk is restored rather than prefilled again.
//!
//! A chain is sealed behind a **lead**: sealed sections placed just ahead of
//! its first member, inside the range, never placed themselves, their K/V and
//! state reused as a seed chain's are. The collection's own lead comes first
//! — the records `projection.yaml` declares for it (§3.5f) — then, for a file,
//! its repository's root listing and its own folder's (§3.5e). Sealed with
//! nothing before it, a section's K/V reads as the only one of its kind: two
//! files' `serde` lines sealed so are told apart by nothing but which sits
//! nearer the question. Behind the listings that name its repository and its
//! folder, it is computed as the part of that repository it is, and not as the
//! first thing read.
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
    SystemPromptItem, MEMBER_KV_LEVEL,
};
use crate::scheduler::exported_state::SharedState;
use crate::scheduler::SchedulerRequest;
use crate::token_buffer::TokenBuffer;
use crate::SectionSeal;

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

/// What of `before` — everything the schema declares ahead of a collection,
/// each item in its slot — the birth prefix holds, in `before`'s order: a
/// collection member or `depends_on` section the birth prefix leaves out is a
/// hole of its `tokens`, so every position after it stands where the birth
/// prefix puts it (`docs/placed_section_layout.md` §3.1c), and every hole is
/// kept.
fn in_birth_prefix(
    before: Vec<PrefixItem>,
    birth: &[PrefixItem],
    tokens: impl Fn(SectionId) -> usize,
) -> Vec<PrefixItem> {
    let birth: HashSet<SectionId> = birth.iter().filter_map(|i| i.section()).collect();
    before
        .into_iter()
        .filter_map(|item| match item.section() {
            Some(s) if !birth.contains(&s) => {
                let positions = tokens(s);
                (positions > 0).then_some(PrefixItem::Hole(positions))
            }
            _ => Some(item),
        })
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
    /// The compression level a member is sealed at: its collection's
    /// [`kv_level`](crate::projection::SectionCollection::kv_level).
    pub level: u8,
    /// The sealed sections the chain is sealed behind, just ahead of its first
    /// member — the collection's lead, then a file's listings — and the
    /// positions they take.
    pub lead: Vec<SectionId>,
    pub lead_tokens: usize,
}

impl MembersPrefix {
    /// The same prefix with the chain sealed behind `lead` as well, after the
    /// lead it has; `lead` takes `tokens` positions.
    pub fn led_by(mut self, lead: Vec<SectionId>, tokens: usize) -> Self {
        self.lead.extend(lead);
        self.lead_tokens += tokens;
        self
    }

    /// What a member is sealed behind when its run's members up to and
    /// including it take `through` positions and `earlier` of them precede
    /// it: the prompt, the part of the range above the lead and the run left
    /// empty, the lead, then the earlier members.
    pub(super) fn behind(&self, through: usize, earlier: &[SectionId]) -> Vec<PrefixItem> {
        let mut out = self.prefix.clone();
        if let Some(range) = self.range {
            let hole = range.saturating_sub(through + self.lead_tokens);
            if hole > 0 {
                out.push(PrefixItem::Hole(hole));
            }
        }
        out.extend(self.lead.iter().copied().map(PrefixItem::Section));
        out.extend(earlier.iter().copied().map(PrefixItem::Section));
        out
    }

    /// What a member's address is computed from: [`Self::behind`] with the
    /// lead left out, its positions part of the hole. The lead is what the
    /// member is read behind, not what it is: an address that named it changed
    /// with whatever else the workspace holds, and with whether the lead's
    /// tokens were loaded when the member was sealed, so a restart prefilled
    /// afresh what it should have restored (`docs/placed_section_layout.md`
    /// §3.5f).
    pub(super) fn addressed(&self, through: usize, earlier: &[SectionId]) -> Vec<PrefixItem> {
        Self {
            lead: Vec::new(),
            lead_tokens: 0,
            ..self.clone()
        }
        .behind(through, earlier)
    }
}

impl Sequence {
    /// The trained window of the engine this conversation runs on, which picks
    /// each reserved range's size; `None` before an engine reports one.
    pub fn native_context(&self) -> Option<usize> {
        self.substrate.read().trained_window()
    }

    /// The tokens a sealed section takes, as the prompt walks size its slot
    /// (`docs/placed_section_layout.md` §3.1c); 0 for one not sealed.
    pub fn section_tokens(&self, section: SectionId) -> usize {
        self.substrate.read().section_tokens_of(section).len()
    }

    /// The token ids `text` encodes to, as a member's seal encodes it.
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
    /// range, holes of the ranges before it included — the range, and the
    /// collection's lead (`docs/placed_section_layout.md` §3.5f).
    pub fn members_prefix(&self, collection: &str) -> crate::Result<MembersPrefix> {
        let base = &self.base_projection;
        let id = base.id_for_system_collection(collection).ok_or_else(|| {
            ConversationError::Other(format!("the schema has no collection {collection:?}"))
        })?;
        let level = self.collection_kv_level(collection);
        let lead = base
            .schema()
            .system_prompt
            .collection_named(collection)
            .map(|c| c.lead_ids())
            .unwrap_or_default();
        let lead_tokens = lead.iter().map(|&s| self.section_tokens(s)).sum();
        if let Some(range) = self.birth_ranges.iter().find(|r| r.collection == id) {
            return Ok(MembersPrefix {
                prefix: range.member_prefix(&self.birth_prefix, range.positions),
                range: range.sealed_at_top(),
                level,
                lead,
                lead_tokens,
            });
        }
        let native = self.substrate.read().trained_window();
        let before =
            prefix_before_collection(&base.schema().system_prompt.items, id, native, |s| {
                self.section_tokens(s)
            })
            .ok_or_else(|| {
                ConversationError::Other(format!(
                    "collection {collection:?} is inside a section tree; its members are \
                     sealed behind the prompt before it, which a tree branches"
                ))
            })?;
        Ok(MembersPrefix {
            prefix: in_birth_prefix(before, &self.birth_prefix, |s| self.section_tokens(s)),
            range: None,
            level,
            lead,
            lead_tokens,
        })
    }

    /// The level `collection`'s members are sealed at: what the schema
    /// declares for it, [`MEMBER_KV_LEVEL`] for a collection it does not hold.
    pub fn collection_kv_level(&self, collection: &str) -> u8 {
        self.base_projection
            .schema()
            .system_prompt
            .items
            .iter()
            .find_map(|item| match item {
                SystemPromptItem::Collection(c) if c.name == collection => Some(c.kv_level),
                _ => None,
            })
            .unwrap_or(MEMBER_KV_LEVEL)
    }

    /// Seal `chain` as members of `collection`, in order, each behind the
    /// collection's prefix ([`Self::members_prefix`]), `lead` and the members
    /// before it, and return them in the same order.
    pub fn seal_members(
        &self,
        collection: &str,
        lead: &[SectionId],
        chain: &[MemberSpec],
    ) -> crate::Result<Vec<SealedMember>> {
        let tokens = lead.iter().map(|&s| self.section_tokens(s)).sum();
        let prefix = self
            .members_prefix(collection)?
            .led_by(lead.to_vec(), tokens);
        self.seal_members_behind(&prefix, chain, SealAs::Member)
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
        let chain = chain_of(&self.substrate, &prefix.addressed(through, &earlier)).0;
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
                seal: match kind {
                    SealAs::Member => SectionSeal::Member {
                        level: prefix.level,
                    },
                    SealAs::Boundary => SectionSeal::Boundary,
                },
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
    /// collection, at the same positions**: a tool member (5, 12 tokens) and a
    /// `depends_on` section (6, 30 tokens) ahead of the collection become holes
    /// of their tokens, and the seed and the close the birth prefix holds after
    /// it (8, 9) are never in `before`.
    #[test]
    fn a_member_is_sealed_behind_the_birth_prefix_before_its_collection() {
        let mut before = sections_prefix(&ids(&[1, 2, 5]));
        before.push(PrefixItem::Hole(64));
        before.extend(sections_prefix(&ids(&[6, 7])));
        let birth = sections_prefix(&ids(&[1, 2, 7, 8, 9]));
        let tokens = |s: SectionId| match s.raw() {
            5 => 12,
            6 => 30,
            _ => 4,
        };
        let mut want = sections_prefix(&ids(&[1, 2]));
        want.push(PrefixItem::Hole(12));
        want.push(PrefixItem::Hole(64));
        want.push(PrefixItem::Hole(30));
        want.extend(sections_prefix(&ids(&[7])));
        assert_eq!(in_birth_prefix(before, &birth, tokens), want);
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
            level: MEMBER_KV_LEVEL,
            lead: Vec::new(),
            lead_tokens: 0,
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
            level: MEMBER_KV_LEVEL,
            lead: Vec::new(),
            lead_tokens: 0,
        };
        let mut bare = prompt;
        bare.extend(sections_prefix(&ids(&[10])));
        assert_eq!(plain.behind(400, &ids(&[10])), bare);
    }

    /// **A led chain is sealed behind its lead, just ahead of its first
    /// member, the range's hole short by the lead's positions**: in a
    /// 1,000-range a lead of 50 positions (20) and a chain through 400 take
    /// 450, so 550 stay a hole above them; the lead stands between the hole
    /// and the earlier members.
    #[test]
    fn a_led_chain_is_sealed_behind_its_lead() {
        let prompt = sections_prefix(&ids(&[1, 2]));
        let led = MembersPrefix {
            prefix: prompt.clone(),
            range: Some(1_000),
            level: MEMBER_KV_LEVEL,
            lead: Vec::new(),
            lead_tokens: 0,
        }
        .led_by(ids(&[20]), 50);
        let mut want = prompt.clone();
        want.push(PrefixItem::Hole(550));
        want.extend(sections_prefix(&ids(&[20, 10, 11])));
        assert_eq!(led.behind(400, &ids(&[10, 11])), want);
        // Its first member is sealed behind the lead alone.
        let mut first = prompt.clone();
        first.push(PrefixItem::Hole(550));
        first.extend(sections_prefix(&ids(&[20])));
        assert_eq!(led.behind(400, &[]), first);
        // A collection's lead (30, 25 positions) stands first, a file's
        // listing after it: 400 + 25 + 50 leave 525.
        let both = MembersPrefix {
            prefix: prompt.clone(),
            range: Some(1_000),
            level: MEMBER_KV_LEVEL,
            lead: ids(&[30]),
            lead_tokens: 25,
        }
        .led_by(ids(&[20]), 50);
        let mut want = prompt.clone();
        want.push(PrefixItem::Hole(525));
        want.extend(sections_prefix(&ids(&[30, 20, 10])));
        assert_eq!(both.behind(400, &ids(&[10])), want);
        // Its address leaves both leads out, their positions in the hole: the
        // same as a chain sealed behind no lead at all.
        let mut addressed = prompt;
        addressed.push(PrefixItem::Hole(600));
        addressed.extend(sections_prefix(&ids(&[10])));
        assert_eq!(both.addressed(400, &ids(&[10])), addressed);
        assert_eq!(
            both.addressed(400, &ids(&[10])),
            MembersPrefix {
                lead: Vec::new(),
                lead_tokens: 0,
                ..both.clone()
            }
            .behind(400, &ids(&[10]))
        );
    }
}
