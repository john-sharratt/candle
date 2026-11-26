//! A file of the knowledge base assembled for the belief scan from its members
//! (`docs/provenance_system_collection.md` §4).
//!
//! The scan scores a file through its gallery: windows over stored
//! signatures, each with the stream that holds it, grouped into cases. A
//! reading built that gallery from its turns ([`super::belief_files`]); a file
//! of the knowledge base builds it from its chunks — each chunk's member
//! stream holds the chunk's signatures, written at its seal — one window and
//! one case a chunk, the file's calibrations beside them. From there the scan
//! is the same scan: the same arena launch, the same vote laws, the same
//! normalization.
//!
//! The scan names each candidate by a [`TimelineId`] ([`FileScan::timeline`]),
//! which here carries the file's [`FileKey`] ([`scan_id`]): the scan reads the
//! id only to name a result, and the calibration index already maps a
//! calibration to the id its `CALIBRATES_KEY` names, so a calibration that
//! names a file's key votes for that file with no other change. Nothing reads
//! such an id as a conversation.

use std::sync::Arc;

use super::belief_files::FileScan;
use super::ids::{TimelineId, TurnIndex};
use crate::files::FileKey;
use crate::persistence::streams::StreamId;
use crate::substrate::{FileGallery, Substrate};

/// The id the scan names `file` by ([`FileKey::scan_id`]).
pub(crate) fn scan_id(file: FileKey) -> TimelineId {
    file.scan_id()
}

/// The file a scan id names.
pub(crate) fn file_of(id: TimelineId) -> FileKey {
    FileKey::of_scan_id(id)
}

/// The gallery of a file whose chunks' members are `chunks`, in line order:
/// each chunk one case, scored by one window over its whole stored signature;
/// a chunk with no signature keeps its case and scores nothing.
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

/// Every file of `files` — `(file, its chunks' member streams)` — assembled
/// for the scan from `sub`, with its calibrations; a file with no window is
/// left out.
pub(crate) fn assemble_member_scans(
    sub: &Substrate,
    files: &[(FileKey, Vec<StreamId>)],
) -> Vec<FileScan> {
    let records = sub.calibration_snapshot();
    let mut fresh = Vec::new();
    let mut out = Vec::with_capacity(files.len());
    for (file, chunks) in files {
        let gallery = member_gallery(sub, chunks);
        if gallery.windows.is_empty() {
            continue;
        }
        let id = scan_id(*file);
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

    /// **A file's scan id carries its key**, and the key comes back from it.
    #[test]
    fn a_files_scan_id_carries_its_key() {
        let file = FileKey(0x473e_088d_5653_2900);
        assert_eq!(scan_id(file).raw(), 0x473e_088d_5653_2900);
        assert_eq!(file_of(scan_id(file)), file);
        assert_eq!(scan_id(FileKey(0)).raw(), 1);
    }

    /// **A file with no stored signature builds a gallery of cases and no
    /// window**: every chunk keeps its case, so a chunk's place in the file is
    /// its case's slot, and the assembly leaves such a file out.
    #[test]
    fn a_file_without_signatures_keeps_its_cases_and_no_window() {
        let sub = Substrate::default();
        let g = member_gallery(&sub, &[StreamId(11), StreamId(12)]);
        assert_eq!(g.n_slots, 2);
        assert_eq!(g.arc_turn, vec![TurnIndex(0), TurnIndex(1)]);
        assert_eq!(g.ex_ranges, vec![0..1, 1..2]);
        assert!(g.windows.is_empty());
        assert!(assemble_member_scans(&sub, &[(FileKey(5), vec![StreamId(11)])]).is_empty());
    }
}
