//! Each dialogue's working set of the knowledge base's files, on the
//! substrate beside its working set (`docs/provenance_system_collection.md`
//! §4): read by the projection, folded by every scan, its placed order kept
//! by every fill.

use std::sync::Arc;

use super::Substrate;
use crate::files::{FileKey, FilesPin, FilesSet};
use crate::projection::{GroupId, TimelineId};

impl Substrate {
    /// `target`'s files set, when it has one.
    pub fn files_set_of(&self, target: TimelineId) -> Option<&FilesSet> {
        self.files_sets.get(&target).map(|s| &**s)
    }

    /// Fold one scan's fresh file scores into `target`'s files set.
    pub fn files_observe(
        &mut self,
        target: TimelineId,
        fresh: &[(GroupId, Vec<(FileKey, f32)>)],
        beta: f32,
        min_momentum: f32,
    ) {
        Arc::make_mut(self.files_sets.entry(target).or_default()).observe(
            fresh,
            beta,
            min_momentum,
        );
    }

    /// Record `order` as where the files a projection of `target` placed sit
    /// ([`FilesSet::order`] computed it).
    pub fn files_placed(&mut self, target: TimelineId, order: Vec<FileKey>) {
        Arc::make_mut(self.files_sets.entry(target).or_default()).set_placed(order);
    }

    /// Keep `files`, read by `target`'s turn, from being placed until its
    /// next question ([`FilesSet::withhold`]).
    pub fn files_withhold(&mut self, target: TimelineId, files: &[FileKey]) {
        if !files.is_empty() {
            Arc::make_mut(self.files_sets.entry(target).or_default())
                .withhold(files.iter().copied());
        }
    }

    /// A new question on `target`: its withheld files may be placed again.
    pub fn files_clear_withheld(&mut self, target: TimelineId) {
        if let Some(set) = self.files_sets.get_mut(&target) {
            if !set.withheld.is_empty() {
                Arc::make_mut(set).clear_withheld();
            }
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
}
