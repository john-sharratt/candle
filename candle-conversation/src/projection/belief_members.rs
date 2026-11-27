//! A section of the knowledge base assembled for the belief scan from its
//! member (`docs/provenance_system_collection.md` §4,
//! `docs/placed_section_layout.md` §3.5c).
//!
//! The scan scores a candidate through its gallery: windows over stored
//! signatures, each with the stream that holds it, grouped into cases. A
//! reading built that gallery from its turns ([`super::belief_files`]); a
//! section of the knowledge base builds it from its member stream, which holds
//! the section's signatures, written at its seal — one window and one case —
//! its calibrations beside it. Each section is a candidate of its own, so a
//! large file no longer gathers every one of its sections' votes into one
//! score. From there the scan is the same scan: the same arena launch, the
//! same vote laws, the same normalization.
//!
//! The scan names each candidate by a [`TimelineId`] ([`FileScan::timeline`]),
//! which here carries the section's [`SectionKey`] ([`scan_id`]): the scan
//! reads the id only to name a result, and the calibration index already maps
//! a calibration to the id its `CALIBRATES_KEY` names, so a calibration that
//! names a section's key votes for that section with no other change. Nothing
//! reads such an id as a conversation.

use std::sync::Arc;

use super::belief_files::FileScan;
use super::ids::{TimelineId, TurnIndex};
use crate::files::SectionKey;
use crate::persistence::streams::StreamId;
use crate::substrate::{FileGallery, Substrate};

/// The id the scan names `section` by ([`SectionKey::scan_id`]).
pub(crate) fn scan_id(section: SectionKey) -> TimelineId {
    section.scan_id()
}

/// The section a scan id names.
pub(crate) fn section_of(id: TimelineId) -> SectionKey {
    SectionKey::of_scan_id(id)
}

/// The gallery of a candidate whose members are `chunks`, in order: each one
/// case, scored by one window over its whole stored signature; a member with
/// no signature keeps its case and scores nothing.
pub(crate) fn member_gallery(sub: &Substrate, chunks: &[StreamId]) -> FileGallery {
    let mut gallery = FileGallery {
        n_slots: chunks.len(),
        ..FileGallery::default()
    };
    for (k, &stream) in chunks.iter().enumerate() {
        gallery.arc_turn.push(TurnIndex(k as u32));
        gallery.ex_ranges.push(k..k + 1);
        let Some(window) = sub.wide_sig_window(stream) else {
            continue;
        };
        let len = window.len();
        if len == 0 {
            continue;
        }
        gallery.arcs_kept.push(window);
        gallery.arc_sids.push(stream);
        gallery
            .windows
            .push((gallery.arcs_kept.len() - 1, 0, len, k));
    }
    gallery
}

/// Every section of `sections` — `(section, its member stream)` — assembled
/// for the scan from `sub`, with its calibrations; a section with no window
/// is left out.
pub(crate) fn assemble_member_scans(
    sub: &Substrate,
    sections: &[(SectionKey, StreamId)],
) -> Vec<FileScan> {
    let records = sub.calibration_snapshot();
    let mut fresh = Vec::new();
    let mut out = Vec::with_capacity(sections.len());
    for &(section, stream) in sections {
        let gallery = member_gallery(sub, &[stream]);
        if gallery.windows.is_empty() {
            continue;
        }
        let id = scan_id(section);
        let cal = sub.calibration_record(id, &records, &mut fresh);
        let own = gallery.arcs_kept.len();
        let cal_windows: Vec<(usize, usize, usize)> = cal.as_ref().map_or(Vec::new(), |r| {
            r.sigs
                .iter()
                .enumerate()
                .map(|(i, w)| (own + i, 0, w.len()))
                .collect()
        });
        out.push(FileScan {
            timeline: id,
            windows: gallery.windows.clone(),
            gallery: Arc::new(gallery),
            cal_windows,
            cal,
        });
    }
    sub.publish_calibration_records(fresh);
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    /// **A section's scan id carries its key**, and the key comes back from
    /// it.
    #[test]
    fn a_sections_scan_id_carries_its_key() {
        let section = SectionKey(0x473e_088d_5653_2900);
        assert_eq!(scan_id(section).raw(), 0x473e_088d_5653_2900);
        assert_eq!(section_of(scan_id(section)), section);
        assert_eq!(scan_id(SectionKey(0)).raw(), 1);
    }

    /// **Members with no stored signature build a gallery of cases and no
    /// window**: every member keeps its case, and the assembly leaves such a
    /// section out.
    #[test]
    fn a_section_without_signatures_keeps_its_case_and_no_window() {
        let sub = Substrate::default();
        let g = member_gallery(&sub, &[StreamId(11), StreamId(12)]);
        assert_eq!(g.n_slots, 2);
        assert_eq!(g.arc_turn, vec![TurnIndex(0), TurnIndex(1)]);
        assert_eq!(g.ex_ranges, vec![0..1, 1..2]);
        assert!(g.windows.is_empty());
        assert!(assemble_member_scans(&sub, &[(SectionKey(5), StreamId(11))]).is_empty());
    }
}
