//! Scanning sample questions against part of a gallery in jobs small in both
//! directions (`docs/incremental_chunk_exclusion.md` §4.6), merging the
//! statistics each job hands back into one summary per question token and
//! fold group.
//!
//! A job is up to [`PROBE_CHUNK`] probe tokens against a slice of up to
//! [`SLICE_TOKENS`] gallery tokens, laid out as segments of at most
//! [`SEGMENT_FILES`] files: the probe chunk keeps a job's upload and output
//! inside the scratch the live scans already hold, the slice bounds its kernel,
//! and the segment bound keeps every backend's per-case table in reach. Files
//! are cases in ascending id within a segment, so the kernel ranks ties as the
//! merge does ([`EntryStats::merge`]).

use std::collections::BTreeMap;

use candle::Result;

use super::gallery::GalleryStream;
use crate::provenance::chunk_exclusion::stats::{EntryStats, FileId};
use crate::provenance::gallery_arena::{
    BeliefArena, GroupSet, Holder, Keep, PagedSegment, PagedWindow,
};
use crate::provenance::{VoteLaw, WideQSig, FOLD_GROUPS};

/// Probe tokens a job scans.
pub(crate) const PROBE_CHUNK: usize = 8_192;
/// Gallery tokens a job scans: with [`PROBE_CHUNK`] probe tokens, ~2 × 10⁹
/// token pairs, ~10 ms of kernel at the measured ~2.5 × 10¹¹ pairs a second.
pub(crate) const SLICE_TOKENS: usize = 262_144;
/// Files a segment holds — under the scalar kernel's shared-memory bound of
/// 1,536 cases.
pub(crate) const SEGMENT_FILES: usize = 1_024;

/// How a competition's gallery is scanned: the fold groups it weighs and the
/// tokens its pages keep.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct ScanShape {
    pub groups: GroupSet,
    pub keep: Keep,
}

/// What scanning probes against streams gave: per probe, per token, per fold
/// group, the merged statistics; and the gallery tokens scanned.
#[derive(Debug, Default, PartialEq)]
pub(crate) struct Scanned {
    pub stats: Vec<Vec<[EntryStats; FOLD_GROUPS]>>,
    pub n: u64,
}

/// Scan `probes` — each one question's full-width tokens — against `streams`
/// on `arena`, in bounded jobs, merging what each hands back. With no probe
/// token nothing is scanned and `n` is not known: `0`.
pub(crate) fn scan(
    arena: &dyn BeliefArena,
    streams: &[&GalleryStream],
    probes: &[&[WideQSig]],
    shape: ScanShape,
) -> Result<Scanned> {
    let mut out = Scanned {
        stats: probes
            .iter()
            .map(|p| vec![[EntryStats::default(); FOLD_GROUPS]; p.len()])
            .collect(),
        n: 0,
    };
    let chunks = probe_chunks(probes);
    if chunks.is_empty() {
        return Ok(out);
    }
    for slice in slices(streams) {
        let (segments, case_files) = segments_of(&slice, shape);
        let mut slice_n: Option<u64> = None;
        for chunk in &chunks {
            let parts: Vec<&[WideQSig]> = chunk.iter().map(|&(p, s, e)| &probes[p][s..e]).collect();
            let stats = arena.scan_stats(&segments, &parts)?;
            if stats.per_probe != parts.iter().map(|p| p.len()).collect::<Vec<_>>() {
                return Err(candle::Error::Msg(
                    "exclusion scan: a probe token was not scanned at full width".into(),
                ));
            }
            slice_n.get_or_insert(stats.seg_tokens.iter().sum());
            let mut row_token = 0usize;
            for &(p, s, e) in chunk {
                for t in s..e {
                    for (place, &g) in stats.groups.iter().enumerate() {
                        for seg in 0..stats.n_segments() {
                            if let Some(row) = stats.row(row_token, place, seg) {
                                let es = EntryStats::of_row(row, |c| {
                                    case_files.get(c as usize).copied().unwrap_or(0)
                                });
                                out.stats[p][t][g].merge(&es);
                            }
                        }
                    }
                    row_token += 1;
                }
            }
        }
        out.n += slice_n.unwrap_or(0);
    }
    Ok(out)
}

/// `probes`' tokens in chunks of at most [`PROBE_CHUNK`]: each part
/// `(probe, start, end)`.
fn probe_chunks(probes: &[&[WideQSig]]) -> Vec<Vec<(usize, usize, usize)>> {
    let mut chunks: Vec<Vec<(usize, usize, usize)>> = Vec::new();
    let mut cur: Vec<(usize, usize, usize)> = Vec::new();
    let mut room = PROBE_CHUNK;
    for (p, probe) in probes.iter().enumerate() {
        let mut at = 0usize;
        while at < probe.len() {
            let take = room.min(probe.len() - at);
            cur.push((p, at, at + take));
            at += take;
            room -= take;
            if room == 0 {
                chunks.push(std::mem::take(&mut cur));
                room = PROBE_CHUNK;
            }
        }
    }
    if !cur.is_empty() {
        chunks.push(cur);
    }
    chunks
}

/// `streams` in file order, cut into slices of at most [`SLICE_TOKENS`] tokens
/// (a stream wider than that alone).
fn slices<'a>(streams: &[&'a GalleryStream]) -> Vec<Vec<&'a GalleryStream>> {
    let mut sorted: Vec<&GalleryStream> = streams.to_vec();
    sorted.sort_by_key(|s| (s.file, s.sid.0));
    let mut out: Vec<Vec<&GalleryStream>> = Vec::new();
    let mut cur: Vec<&GalleryStream> = Vec::new();
    let mut tokens = 0usize;
    for s in sorted {
        let span = s.span();
        if !cur.is_empty() && tokens + span > SLICE_TOKENS {
            out.push(std::mem::take(&mut cur));
            tokens = 0;
        }
        tokens += span;
        cur.push(s);
    }
    if !cur.is_empty() {
        out.push(cur);
    }
    out
}

/// A slice laid out as segments of at most [`SEGMENT_FILES`] files, each
/// file a case in ascending id; and every case's file, in global case order.
fn segments_of<'a>(
    slice: &[&'a GalleryStream],
    shape: ScanShape,
) -> (Vec<PagedSegment<'a>>, Vec<FileId>) {
    let mut by_file: BTreeMap<FileId, Vec<&GalleryStream>> = BTreeMap::new();
    for s in slice {
        by_file.entry(s.file).or_default().push(s);
    }
    let files: Vec<(FileId, Vec<&GalleryStream>)> = by_file.into_iter().collect();
    let mut segments = Vec::new();
    let mut case_files = Vec::new();
    for group in files.chunks(SEGMENT_FILES) {
        let mut windows = Vec::new();
        for (case, (file, streams)) in group.iter().enumerate() {
            case_files.push(*file);
            for s in streams {
                for &(start, end) in &s.windows {
                    windows.push(PagedWindow {
                        sid: s.sid,
                        fingerprint: s.sig.fingerprint(),
                        turn: (&s.sig).into(),
                        start,
                        end,
                        case,
                    });
                }
            }
        }
        segments.push(PagedSegment {
            windows,
            n_cases: group.len(),
            holder: Holder::Shared,
            groups: shape.groups,
            keep: shape.keep,
            law: VoteLaw::SINGLE,
        });
    }
    (segments, case_files)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sig(fill: u64) -> WideQSig {
        WideQSig {
            n_heads: 12,
            words: vec![fill; 24],
        }
    }

    /// **Probe tokens go out in chunks of at most [`PROBE_CHUNK`]**, a probe
    /// split across chunks where it crosses one.
    #[test]
    fn probe_tokens_go_out_in_bounded_chunks() {
        let a: Vec<WideQSig> = (0..PROBE_CHUNK - 3).map(|_| sig(1)).collect();
        let b: Vec<WideQSig> = (0..10).map(|_| sig(2)).collect();
        let chunks = probe_chunks(&[&a, &b]);
        assert_eq!(
            chunks,
            vec![vec![(0, 0, PROBE_CHUNK - 3), (1, 0, 3)], vec![(1, 3, 10)],]
        );
        assert!(probe_chunks(&[&[]]).is_empty());
    }
}
