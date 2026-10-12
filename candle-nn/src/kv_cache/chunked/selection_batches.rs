//! How a cross-layer format selection is cut to fit the persistence domain.
//!
//! The fused selection stages its tables and carves its result block from the
//! persistence thread's transient span, which is
//! [`MIGRATION_STAGING_CAP_BYTES`] wide. Its cost grows with the chunks it
//! covers — about 11 KB a chunk at 256-wide heads — so a group of ingest
//! documents, tens of thousands of chunks across the layers, needs a quarter of
//! a gigabyte in one piece. That was refused on every pass, and nothing ever
//! left the GPU for the warm tier. These cut the flat layer-major chunk list
//! into the longest runs that fit; chunks never interact across a selection, so
//! the runs' results concatenate to what one selection would have returned.

#[cfg(feature = "cuda")]
use candle::quantized::cuda::Palette4Layout;
use candle::Result;

use super::chunk_ops::MIGRATION_STAGING_CAP_BYTES;
#[cfg(feature = "cuda")]
use super::head_gids::GIDS_PER_HEAD;
#[cfg(feature = "cuda")]
use crate::kv_cache::arena_table::PerHeadEntry;

/// What one run may carve: the domain, less a margin for the bump cursor's
/// alignment between allocations and the stager's own small uploads.
pub(crate) const SELECTION_BUDGET_BYTES: usize = MIGRATION_STAGING_CAP_BYTES - (1 << 20);

/// Device bytes one fused selection over `chunks` chunks carves: the per-head
/// arena table and band gids it stages, and its palette-4 block.
#[cfg(feature = "cuda")]
pub(crate) fn selection_bytes(
    chunks: usize,
    n_kv_head: usize,
    head_dim: usize,
    n_k_cand: usize,
    n_v_cand: usize,
) -> usize {
    let tables = chunks * n_kv_head * (PerHeadEntry::COLS + GIDS_PER_HEAD) * size_of::<i64>();
    let block = Palette4Layout::new(chunks, n_kv_head, head_dim, n_k_cand, n_v_cand).bytes();
    // Each of the three allocations starts on a 256-byte boundary.
    tables + block + 3 * 256
}

/// `total` chunks as consecutive `(start, len)` runs, each the longest whose
/// `bytes` fit `budget`. `bytes` must not shrink as the run grows. A single
/// chunk that does not fit is refused, naming what it needs.
pub(crate) fn selection_batches(
    total: usize,
    budget: usize,
    bytes: impl Fn(usize) -> usize,
) -> Result<Vec<(usize, usize)>> {
    if total == 0 {
        return Ok(Vec::new());
    }
    if bytes(1) > budget {
        candle::bail!(
            "one chunk's format selection needs {} B, more than the {budget} B the \
             persistence domain can give it",
            bytes(1)
        );
    }
    // The longest run that fits, by bisection over `[1, total]`.
    let (mut fits, mut over) = (1usize, total + 1);
    while over - fits > 1 {
        let mid = fits + (over - fits) / 2;
        if bytes(mid) <= budget {
            fits = mid;
        } else {
            over = mid;
        }
    }
    Ok((0..total)
        .step_by(fits)
        .map(|start| (start, fits.min(total - start)))
        .collect())
}

/// The run `[start, start + len)` of the flat layer-major concatenation of
/// `per_layer`, as one (possibly empty) list per layer — so a batch keeps every
/// layer's place and the selection's per-layer table stays aligned.
pub(crate) fn slice_layers<T: Clone>(
    per_layer: &[Vec<T>],
    start: usize,
    len: usize,
) -> Vec<Vec<T>> {
    let end = start + len;
    let mut at = 0usize;
    per_layer
        .iter()
        .map(|layer| {
            let (lo, hi) = (at, at + layer.len());
            at = hi;
            let (a, b) = (start.clamp(lo, hi), end.clamp(lo, hi));
            layer[a - lo..b - lo].to_vec()
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Runs as long as fit, covering every chunk once, in order.
    #[test]
    fn runs_are_the_longest_that_fit_and_cover_everything() {
        // 100 bytes a chunk plus 50 fixed, against 1,000: nine chunks a run.
        let runs = selection_batches(20, 1_000, |c| 50 + 100 * c).unwrap();
        assert_eq!(runs, vec![(0, 9), (9, 9), (18, 2)]);
        assert_eq!(
            selection_batches(9, 1_000, |c| 50 + 100 * c).unwrap(),
            vec![(0, 9)]
        );
        assert_eq!(selection_batches(0, 1_000, |c| c).unwrap(), Vec::new());
    }

    /// A chunk that alone overflows is an error, not an empty plan.
    #[test]
    fn a_chunk_too_big_for_the_domain_is_refused() {
        let err = selection_batches(3, 100, |c| 200 * c)
            .unwrap_err()
            .to_string();
        assert!(err.contains("200 B"), "{err}");
    }

    /// A run that straddles layers takes each layer's part, and the layers it
    /// does not reach are empty but present.
    #[test]
    fn a_run_keeps_every_layers_place() {
        let layers = vec![vec![1, 2, 3], vec![], vec![4, 5], vec![6]];
        assert_eq!(
            slice_layers(&layers, 2, 3),
            vec![vec![3], vec![], vec![4, 5], vec![]]
        );
        assert_eq!(
            slice_layers(&layers, 0, 6),
            vec![vec![1, 2, 3], vec![], vec![4, 5], vec![6]]
        );
        assert_eq!(
            slice_layers(&layers, 5, 1),
            vec![vec![], vec![], vec![], vec![6]]
        );
    }

    /// The measured case: Flash-Next's 2 KV heads of 256, five K and five V
    /// candidates — a run of about five and a half thousand chunks, where the
    /// failing pass asked for twenty-five thousand at once.
    #[cfg(feature = "cuda")]
    #[test]
    fn a_flash_next_run_fits_the_domain() {
        let per = |c| selection_bytes(c, 2, 256, 5, 5);
        let runs = selection_batches(25_000, SELECTION_BUDGET_BYTES, per).unwrap();
        let longest = runs[0].1;
        assert!(per(longest) <= SELECTION_BUDGET_BYTES);
        assert!(per(longest + 1) > SELECTION_BUDGET_BYTES);
        assert!((5_000..6_500).contains(&longest), "{longest}");
        assert_eq!(runs.iter().map(|r| r.1).sum::<usize>(), 25_000);
    }
}
