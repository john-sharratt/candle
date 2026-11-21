//! The public entry points of the paged belief scan: the weighted scan, the
//! forced backends, and the statistics mode.
//!
//! `scan_weighted` ensures residency, builds or reuses the index, launches the
//! kernel and tallies, giving one vote vector per probe. The forced variants keep
//! each backend testable and benchmarkable, and a forced backend reports its
//! launch failure rather than degrading. `tensor_caps` is queried once per arena,
//! on the arena's device.

use super::*;

impl GalleryArena {
    /// Paged belief scan over the resident gallery. Ensures residency, builds the
    /// index, launches the paged kernel, and tallies — one per-case vote vector
    /// per probe. Numerically equivalent to the contiguous
    /// [`BatchedGpuGallery::scan_weighted`](super::super::gpu::BatchedGpuGallery)
    /// on the same windows (only the byte layout differs). Runs on the b1
    /// tensor-core (BMMA) backend when the device has it (sm_75..sm_89) and the
    /// geometry is the locked fold (`gw == 8`); the scalar kernel otherwise.
    pub fn scan_weighted(
        &self,
        segments: &[PagedSegment],
        probes: &[&[WideQSig]],
        group_weights: &[f32],
    ) -> Result<Vec<Vec<f32>>> {
        self.scan_weighted_impl(segments, probes, &every(segments, group_weights), None)
    }

    /// [`Self::scan_weighted`] with each segment's fold groups weighed by its
    /// own `segment_weights[s]` — one launch, the weights applied in the tally.
    pub fn scan_segment_weighted(
        &self,
        segments: &[PagedSegment],
        probes: &[&[WideQSig]],
        segment_weights: &[&[f32]],
    ) -> Result<Vec<Vec<f32>>> {
        if segment_weights.len() != segments.len() {
            return Err(candle::Error::Msg(format!(
                "paged scan: {} segment weightings for {} segments",
                segment_weights.len(),
                segments.len()
            )));
        }
        self.scan_weighted_impl(segments, probes, segment_weights, None)
    }

    /// [`Self::scan_weighted`] forced onto the scalar kernel — the universal
    /// fallback backend, kept first-class for differential testing and
    /// benchmarking (all backends' integer statistics are identical).
    pub fn scan_weighted_scalar(
        &self,
        segments: &[PagedSegment],
        probes: &[&[WideQSig]],
        group_weights: &[f32],
    ) -> Result<Vec<Vec<f32>>> {
        self.scan_weighted_impl(
            segments,
            probes,
            &every(segments, group_weights),
            Some(PagedBackend::Scalar),
        )
    }

    /// [`Self::scan_weighted`] forced onto the b1 tensor-core (BMMA) kernel —
    /// the auto ladder's top rung on sm_75..sm_89. Forcing it surfaces a
    /// backend-specific launch failure as a hard error instead of the ladder's
    /// silent degrade, which is what differential tests and geometry
    /// reproductions need. Errors on devices without b1 BMMA.
    pub fn scan_weighted_bmma(
        &self,
        segments: &[PagedSegment],
        probes: &[&[WideQSig]],
        group_weights: &[f32],
    ) -> Result<Vec<Vec<f32>>> {
        self.scan_weighted_impl(
            segments,
            probes,
            &every(segments, group_weights),
            Some(PagedBackend::Bmma),
        )
    }

    /// [`Self::scan_weighted`] forced onto the INT8 tensor-core (IMMA) kernel —
    /// the backend the auto ladder selects on devices without b1 BMMA (Hopper/
    /// Blackwell). Forcing it keeps the path benchmarkable and differential-
    /// tested on b1 hardware too. Errors on devices without INT8 MMA (below
    /// sm_80).
    pub fn scan_weighted_imma(
        &self,
        segments: &[PagedSegment],
        probes: &[&[WideQSig]],
        group_weights: &[f32],
    ) -> Result<Vec<Vec<f32>>> {
        self.scan_weighted_impl(
            segments,
            probes,
            &every(segments, group_weights),
            Some(PagedBackend::Imma),
        )
    }

    /// This arena device's tensor capabilities `(b1 BMMA, INT8 IMMA)`, queried
    /// once per arena. The FFI probes the calling thread's current CUDA device,
    /// which is the arena's device on every scan path (scans run on threads
    /// holding the arena's context).
    pub(super) fn tensor_caps(&self) -> (bool, bool) {
        *self
            .tensor_caps
            .get_or_init(|| unsafe { (bdp_bmma_supported() != 0, bdp_imma_supported() != 0) })
    }

    /// Every probe token's raw vote in every fold group and segment, before the
    /// needle gate and the tally — one launch, on the backend the auto ladder
    /// picks. For an offline harness that weighs, gates and folds the same
    /// votes under many settings; the live scan is [`Self::scan_weighted`].
    pub fn scan_votes(
        &self,
        segments: &[PagedSegment],
        probes: &[&[WideQSig]],
    ) -> Result<TokenVotes> {
        self.with_index(segments, |idx, weighed| {
            self.token_votes(idx, weighed, probes, None)
                .map(|(votes, _)| votes)
        })
    }

    /// Every probe token's statistics in every segment, in each fold group the
    /// segments weigh — the statistics mode (`docs/incremental_chunk_exclusion.md`
    /// §4.5): what its vote is made of, for a caller merging scans of parts of
    /// a gallery itself.
    ///
    /// **The index is the one the live scans' cache holds for the segments.** A
    /// learner scans each slice of a gallery once per chunk of its probes, so
    /// the slice's index is built for the first chunk and reused, its turns
    /// checked current, by every later one; once the learner moves on it
    /// idles out of the cache like any other.
    pub fn scan_stats(
        &self,
        segments: &[PagedSegment],
        probes: &[&[WideQSig]],
    ) -> Result<TokenStats> {
        self.scan_stats_on(segments, probes, None)
    }

    /// [`Self::scan_stats`] on `force`'s backend, or the auto ladder's.
    pub(super) fn scan_stats_on(
        &self,
        segments: &[PagedSegment],
        probes: &[&[WideQSig]],
        force: Option<PagedBackend>,
    ) -> Result<TokenStats> {
        let dev = match self.device() {
            Device::Cuda(d) => d,
            _ => return Err(candle::Error::Msg("paged scan requires CUDA".into())),
        };
        self.with_index(segments, |idx, _| self.stats_of(dev, idx, probes, force))
    }

    pub(super) fn scan_weighted_impl(
        &self,
        segments: &[PagedSegment],
        probes: &[&[WideQSig]],
        segment_weights: &[&[f32]],
        force: Option<PagedBackend>,
    ) -> Result<Vec<Vec<f32>>> {
        self.with_index(segments, |idx, weighed| {
            self.launch_paged(idx, weighed, probes, segment_weights, force)
        })
    }
}
