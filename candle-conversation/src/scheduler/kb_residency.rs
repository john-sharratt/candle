//! A slot's knowledge-base residency, chosen at each boundary
//! (`docs/placed_section_layout.md` §3.5, §3.7b).
//!
//! The range's sections stand beside twice their positions in resident ones
//! (`files::scope::resident`): the target's strongest sections that may enter
//! the range during the request, those the shadow scan read since the last
//! boundary first. Each resident section's index pages are held in the slot's
//! pool at the sampled layers — the rows the shadow scan rates it by at every
//! step — and its K/V is lifted hot, so the step that admits it places it
//! without a PCIe or redo-log wait. The lift is priced as a rebuild's is: below
//! the decode floor the pages are pooled and the K/V stays where it is. The
//! pages come from the sections' page cache (`section_pages`), never the log:
//! on a model the shadow scan rates sections for, a section whose page this
//! process does not hold could never enter at a step, so it is not resident
//! and enters only by a boundary's fill.

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use candle_transformers::models::piece_key::PieceKey;

use super::projection_assembler::{held_sections, KbResident};
use super::Scheduler;
use crate::files::scope::{resident, Candidate};
use crate::files::{FilesPin, SectionKey};
use crate::index_pages::{self, StoredPages};
use crate::projection::{Conversation, SectionId, TimelineId};
use crate::sequence_handle::SequenceId;

impl Scheduler {
    /// Choose `slot`'s resident sections of `target`'s knowledge base, hold
    /// their index pages in its pool and lift their K/V; the page keys the
    /// pool keeps for them. A slot whose target has no free range keeps none.
    pub(super) fn refresh_kb_residency(
        &mut self,
        slot: SequenceId,
        conversation: &Conversation,
        target: TimelineId,
    ) -> Vec<PieceKey> {
        let read_since: HashSet<SectionKey> = self
            .slot_projection_state
            .get_mut(&slot)
            .map(|s| std::mem::take(&mut s.kb_read))
            .unwrap_or_default();
        // The sections, with each one's member and index page.
        let chosen: Vec<(Candidate, SectionId, Option<Arc<StoredPages>>)> = {
            let read = conversation.read();
            let Some(set) = read.files_set_of(target) else {
                self.set_kb_resident(slot, Vec::new());
                return Vec::new();
            };
            let (Some(scope), FilesPin::Free) = (set.scope.as_ref(), &set.pin) else {
                self.set_kb_resident(slot, Vec::new());
                return Vec::new();
            };
            let ranked = set.may_enter();
            let wanted: HashSet<SectionKey> = ranked.iter().map(|(k, _)| *k).collect();
            let held = held_sections(&read, &wanted);
            // On a model the shadow scan rates sections for, one with no page
            // to rate it by could never enter at a step: it is not resident.
            let rated = self.model.carries_positional_state();
            let mut pages: HashMap<SectionKey, Arc<StoredPages>> = HashMap::new();
            let candidates: Vec<Candidate> = ranked
                .iter()
                .filter_map(|(k, _)| {
                    let h = held.get(k)?;
                    let page = read
                        .section_stream(h.id)
                        .and_then(|s| self.section_positional.get(s).cloned());
                    match page {
                        Some(p) => {
                            pages.insert(*k, p);
                        }
                        None if rated => return None,
                        None => {}
                    }
                    Some(Candidate {
                        key: *k,
                        tokens: h.tokens,
                    })
                })
                .collect();
            resident(scope, &candidates, |k| read_since.contains(&k))
                .into_iter()
                .map(|c| (c, held[&c.key].id, pages.remove(&c.key)))
                .collect()
        };
        // Their pages, placed in one call; a page already pooled is kept.
        let pooled: HashSet<PieceKey> = self.model.pooled_keys(slot.0).into_iter().collect();
        let mut residents = Vec::with_capacity(chosen.len());
        let mut items: Vec<(&[u8], PieceKey)> = Vec::new();
        let mut keep: Vec<PieceKey> = Vec::new();
        for (c, id, page) in &chosen {
            let mut pages = Vec::new();
            if let Some(stored) = page {
                if let Ok(framed) = index_pages::decode(stored.bytes()) {
                    for ((width, bytes), key) in framed.into_iter().zip(stored.page_keys()) {
                        pages.push((width, *key));
                        keep.push(*key);
                        if !pooled.contains(key) {
                            items.push((bytes, *key));
                        }
                    }
                }
            }
            residents.push(KbResident {
                key: c.key,
                id: *id,
                tokens: c.tokens,
                pages,
            });
        }
        if !items.is_empty() {
            if let Err(e) = self.model.pool_positional_states(slot.0, &items) {
                tracing::warn!(slot = slot.0, sections = residents.len(), error = %e,
                    "residency: the resident sections' pages could not be pooled; the shadow \
                     scan does not rate them this request");
                keep.clear();
                for r in &mut residents {
                    r.pages.clear();
                }
            }
        }
        drop(items);
        // Their K/V, while the decodes keep their weights.
        let ids: Vec<SectionId> = residents.iter().map(|r| r.id).collect();
        let lift = self.projection_lift(conversation, &ids, &[]);
        let lifted = self.decode_floor_holds_after(lift);
        if lifted {
            self.elevate_projection_working_set(conversation, &ids, &[], "residency");
        }
        tracing::debug!(
            target: "candle_conversation::scheduler::pool",
            slot = slot.0,
            resident = residents.len(),
            tokens = residents.iter().map(|r| r.tokens).sum::<usize>(),
            pages = keep.len(),
            lift_mib = lift >> 20,
            lifted,
            "knowledge base residency"
        );
        self.set_kb_resident(slot, residents);
        keep
    }

    /// `slot`'s resident sections are now `residents`.
    fn set_kb_resident(&mut self, slot: SequenceId, residents: Vec<KbResident>) {
        if let Some(state) = self.slot_projection_state.get_mut(&slot) {
            state.kb_resident = residents;
        }
    }
}
