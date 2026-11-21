//! The types that the paged scan uses: the windows and segments it reads, the
//! per-scan index, and the device arrays the index holds.
//!
//! A segment names its windows, its case count, the groups it weighs, the tokens
//! the card holds and its vote law. The index records the turns it pinned and
//! their run ids, the groups each segment was built for, and the probe rows
//! already scored, so a rescan scores only its new tokens. A device index is
//! uploaded once and read in place by every launch that reuses it.

use super::*;

/// A sub-window of a resident turn that contributes to a scan: `[start, end)`
/// tokens of the turn identified by `sid`/`fingerprint`, voting for `case`.
pub struct PagedWindow<'a> {
    pub sid: StreamId,
    pub fingerprint: u64,
    /// The WHOLE turn window (for residency); the scan reads `[start, end)`.
    pub turn: TurnSigs<'a>,
    pub start: usize,
    pub end: usize,
    pub case: usize,
}

/// One segment (file): its windows, its exchange (case) count, who the turns it
/// reads are held for (`docs/gallery_residency.md` §2), the fold groups its
/// scan weighs (§4) — the groups its turns must hold, and the launch scans —
/// and which of its turns' tokens the card holds (`docs/gallery_dedup.md`),
/// and how a probe token's agreements become votes in it ([`VoteLaw`]).
pub struct PagedSegment<'a> {
    pub windows: Vec<PagedWindow<'a>>,
    pub n_cases: usize,
    pub holder: Holder,
    pub groups: GroupSet,
    pub keep: Keep,
    pub law: VoteLaw,
}

/// `weights` for every one of `segments`.
pub(super) fn every<'w>(segments: &[PagedSegment], weights: &'w [f32]) -> Vec<&'w [f32]> {
    vec![weights; segments.len()]
}

/// The index's arrays on the device — uploaded once, when the index is built,
/// and read in place by every launch that reuses it. Per page and group
/// (`page_ptr`), per entry (`entry_*`, [`IndexEntries`]) and per segment —
/// nothing per scanned token (`docs/gallery_residency.md` §5).
pub(super) struct DeviceIndex {
    pub(super) page_ptr: Held<u64>,
    pub(super) entry_page: Held<u32>,
    pub(super) entry_slots: Held<u32>,
    pub(super) entry_case: Held<u32>,
    pub(super) seg_slot: Held<i32>,
    pub(super) seg_tok: Held<i32>,
    pub(super) seg_case: Held<i32>,
    /// Cases a probe token votes for, per segment.
    pub(super) seg_top_k: Held<i32>,
}

/// The assembled per-scan index, built from the resident arena. `page_ptr`
/// holds absolute device addresses — one per page per launch group, `page ×
/// groups.len() + place`, the address of that group's words in that page —
/// valid while every referenced turn keeps the run it was built against (see
/// `run_ids`).
pub(in super::super) struct PagedIndex {
    /// `None` when the scan covers no token — nothing to launch.
    pub(super) device: Option<DeviceIndex>,
    /// The fold groups the launch scans: every group a segment weighs.
    pub(in super::super) groups: GroupSet,
    /// The groups each segment was built for — the groups its turns hold
    /// pages for. A reader weighing no more in any segment can reuse it.
    pub(super) seg_groups: Vec<GroupSet>,
    /// Scanned tokens.
    pub(in super::super) n_tokens: usize,
    /// Slots the launch walks: entries × 32, scanned or not.
    pub(super) n_slots: usize,
    /// Host copy of the per-segment case prefixes, which the tally reads.
    pub(super) seg_case: Vec<i32>,
    /// Host copy of the per-segment prefix of the tokens each scans — a
    /// segment's `n` in the statistics mode.
    pub(super) seg_tok: Vec<i32>,
    pub(super) n_cases: usize,
    pub(super) n_segments: usize,
    pub(super) max_seg_cases: usize,
    /// Each segment's vote law; its `top_k` is what the kernel ranks to, its
    /// power what the tally raises by.
    pub(super) seg_law: Vec<VoteLaw>,
    /// Votes per kernel output row: the largest `top_k` of any segment.
    pub(super) max_k: usize,
    /// Turns referenced by this scan; pinned for the scan that builds the
    /// index, leased as a set by every scan that reuses it.
    pub(in super::super) pinned_sids: Vec<StreamId>,
    /// `pinned_sids` by their run ids — what a turn's eviction asks of a
    /// leased index, and what a check looks a named turn's run up in.
    pub(in super::super) members: HashMap<StreamId, u64>,
    /// Each referenced turn's run id when its addresses were taken, parallel to
    /// `pinned_sids` — what a later reuse checks the arena against.
    pub(in super::super) run_ids: Vec<u64>,
    /// What scans and the arena's changes agree through (`lease`).
    pub(in super::super) hold: ScanHold,
    /// When its turns' use was last recorded, milliseconds since the arena
    /// began.
    pub(in super::super) touched_ms: AtomicU64,
    /// When a scan last stored or reused it, milliseconds since the arena
    /// began — the cache drops it once idle (`INDEX_IDLE_MS`).
    pub(in super::super) used: AtomicU64,
    /// Every probe token's row a launch over this index has produced, so the
    /// next launch scores only the tokens no earlier one did (see
    /// [`ProbeRows`]).
    pub(super) rows: Mutex<ProbeRows>,
}

impl PagedIndex {
    /// Device bytes the index's arrays hold — what caching it costs the card,
    /// outside the span and the arena's own ceiling.
    pub(in super::super) fn device_bytes(&self) -> u64 {
        self.device.as_ref().map_or(0, |d| {
            (d.page_ptr.len() * 8
                + (d.entry_page.len() + d.entry_slots.len() + d.entry_case.len()) * 4
                + (d.seg_slot.len() + d.seg_tok.len() + d.seg_case.len() + d.seg_top_k.len()) * 4)
                as u64
        })
    }

    /// Whether this index scans every group each segment of a reader weighing
    /// `weighed` (one set per segment) weighs.
    pub(in super::super) fn serves(&self, weighed: &[GroupSet]) -> bool {
        self.seg_groups.len() == weighed.len()
            && self
                .seg_groups
                .iter()
                .zip(weighed)
                .all(|(b, w)| b.covers(*w))
    }

    /// `weighed`, each segment's set widened by the groups this index was
    /// built for it.
    pub(in super::super) fn covering(&self, weighed: &[GroupSet]) -> Vec<GroupSet> {
        weighed
            .iter()
            .enumerate()
            .map(|(s, w)| self.seg_groups.get(s).map_or(*w, |b| b.union(*w)))
            .collect()
    }
}

/// One scan backend. The auto ladder resolves per DEVICE, fastest first: b1
/// BMMA where the hardware has it (sm_75..sm_89), then INT8 IMMA (sm_80+ —
/// Hopper/Blackwell, which dropped b1), then the scalar kernel. The forced
/// variants keep every backend first-class for differential testing and
/// benchmarking — all three produce bit-identical votes.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(super) enum PagedBackend {
    Bmma,
    Imma,
    Scalar,
}
