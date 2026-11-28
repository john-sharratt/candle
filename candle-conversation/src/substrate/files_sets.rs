//! Each dialogue's working set of the knowledge base's sections, on the
//! substrate beside its working set (`docs/provenance_system_collection.md`
//! §4): read by the projection, folded by every scan, its placed order kept
//! by every fill.

use std::collections::HashMap;
use std::sync::Arc;

use super::Substrate;
use crate::files::scope::{Candidate, RangeScope};
use crate::files::{FileKey, FilesPin, FilesSet, FreshSection, SectionKey};
use crate::projection::{GroupId, TimelineId};

impl Substrate {
    /// `target`'s files set, when it has one.
    pub fn files_set_of(&self, target: TimelineId) -> Option<&FilesSet> {
        self.files_sets.get(&target).map(|s| &**s)
    }

    /// Fold one scan's fresh section scores into `target`'s files set.
    pub fn files_observe(
        &mut self,
        target: TimelineId,
        fresh: &[(GroupId, Vec<FreshSection>)],
        beta: f32,
        min_momentum: f32,
    ) {
        Arc::make_mut(self.files_sets.entry(target).or_default()).observe(
            fresh,
            beta,
            min_momentum,
        );
    }

    /// Record `scope` as the range a projection of `target` placed.
    pub fn files_scope(&mut self, target: TimelineId, scope: RangeScope) {
        Arc::make_mut(self.files_sets.entry(target).or_default()).set_scope(scope);
    }

    /// The hit `rates` of `target`'s question's first reading, folded into its
    /// files set's hit momentum before the opening projection orders the range
    /// by it, as the first decode step would (`docs/placed_section_layout.md`
    /// §3.2). Empty `rates` change nothing.
    pub fn files_read_first(&mut self, target: TimelineId, rates: &HashMap<SectionKey, f32>) {
        if let Some(set) = self.files_sets.get_mut(&target) {
            Arc::make_mut(set).observe_hits(rates);
        }
    }

    /// A step of `target`'s range. When `placed_read` — the step counted the
    /// selection's queries, so it says how the placed sections were read — its
    /// hit `rates` fold into the hit momentum. `leaving` leave the range until
    /// the next question ([`FilesSet::evict`]), `candidates` whose rate would
    /// hold the lock's edge enter it at their rank ([`FilesSet::admit`],
    /// [`FilesSet::step_bars`]), and the range is ordered by the momentum
    /// around the sections locked so far, strongest beside the question
    /// ([`FilesSet::reorder`]). Only then are the locks taken again from the
    /// range's shares of attention, none while it is not `engaged`
    /// ([`FilesSet::relock`]): a section the decode has just begun to read
    /// hard climbs first and locks where it arrives. Whether its layout
    /// changed.
    pub fn files_step(
        &mut self,
        target: TimelineId,
        rates: &HashMap<SectionKey, f32>,
        placed_read: bool,
        engaged: bool,
        leaving: &[SectionKey],
        candidates: &[Candidate],
    ) -> bool {
        let Some(set) = self.files_sets.get_mut(&target) else {
            return false;
        };
        let set = Arc::make_mut(set);
        if placed_read {
            set.observe_hits(rates);
        }
        let left = set.evict(leaving);
        let bars = set.step_bars(rates);
        let rate = |k: SectionKey| rates.get(&k).copied().unwrap_or(0.0);
        let entered = set.admit(rate, candidates, &bars);
        // The range in relevance order by the index's momentum, strongest
        // beside the question (`docs/placed_section_layout.md` §3.2).
        let moved = set.reorder();
        if placed_read {
            set.relock(engaged);
        }
        left || !entered.is_empty() || moved
    }

    /// Keep `files`, read by `target`'s turn, from being placed until its
    /// next question ([`FilesSet::withhold`]).
    pub fn files_withhold(&mut self, target: TimelineId, files: &[FileKey]) {
        if !files.is_empty() {
            Arc::make_mut(self.files_sets.entry(target).or_default())
                .withhold(files.iter().copied());
        }
    }

    /// A new question on `target`: its withheld files may be placed again,
    /// and its range is compacted ([`FilesSet::clear_withheld`]).
    pub fn files_clear_withheld(&mut self, target: TimelineId) {
        if let Some(set) = self.files_sets.get_mut(&target) {
            Arc::make_mut(set).clear_withheld();
        }
    }

    /// Pin what `target`'s knowledge base holds ([`FilesPin`]).
    pub fn files_pin(&mut self, target: TimelineId, pin: FilesPin) {
        Arc::make_mut(self.files_sets.entry(target).or_default()).pin = pin;
    }

    /// Drop `target`'s files set with the conversation.
    pub(super) fn forget_files_set(&mut self, target: TimelineId) {
        self.files_sets.remove(&target);
    }

    /// The knowledge base's mandatory files, by workspace path, in order.
    pub fn mandatory_files(&self) -> &[String] {
        &self.mandatory_files
    }

    /// Name the knowledge base's mandatory files (`files::mandatory`).
    pub fn set_mandatory_files(&mut self, paths: Vec<String>) {
        self.mandatory_files = Arc::new(paths);
    }
}
