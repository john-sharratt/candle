//! Tests re-feed cuts, snapshots and the prompt end across rewinds.
//!
//! A re-feed rewinds to the newest block boundary within eight tokens, so only
//! whole blocks and open rows are dropped. A snapshot restores the open rows it
//! copied. A fork or a truncate carries the prompt end with the cache.

use super::*;

// Pinned ranges are lists of ranges, and a list of one is the common case.
#[cfg(test)]
#[allow(clippy::single_range_in_vec_init)]
pub(super) mod tests {
    use super::*;

    /// **A re-feed rewinds back to the newest block boundary within 8 tokens**:
    /// 10 blocks and 2 open rows (42 tokens) keep 9 blocks and re-feed 6; 10
    /// whole blocks keep 8 and re-feed 8; a tail of 5 tokens re-feeds all 5.
    #[test]
    fn a_refeed_rewinds_to_the_newest_block_boundary_within_its_reach() {
        let cut = |blocks, open| refeed_cut(blocks, open, 4, 8);
        assert_eq!(
            cut(10, 2),
            RefeedCut {
                keep_blocks: 9,
                tokens: 6
            }
        );
        assert_eq!(
            cut(10, 0),
            RefeedCut {
                keep_blocks: 8,
                tokens: 8
            }
        );
        assert_eq!(
            cut(1, 1),
            RefeedCut {
                keep_blocks: 0,
                tokens: 5
            }
        );
        assert_eq!(
            cut(0, 0),
            RefeedCut {
                keep_blocks: 0,
                tokens: 0
            }
        );
        // Every cut of a long tail re-feeds 5..=8 tokens: whole blocks only.
        for open in 0..4 {
            let c = cut(20, open);
            assert!((5..=8).contains(&c.tokens), "{open} open: {c:?}");
            assert_eq!(c.keep_blocks * 4 + c.tokens, 80 + open);
        }
    }

    /// **A rewound tail keeps its first blocks and no open rows**, and refuses
    /// to grow.
    #[test]
    fn a_rewound_tail_keeps_its_first_blocks() {
        let mut c = IndexCache::new(8, &Device::Cpu).unwrap();
        c.n_blocks = 5;
        c.n_open = 3;
        c.rewind_live(3).unwrap();
        assert_eq!((c.n_blocks, c.n_open, c.len(4)), (3, 0, 12));
        assert!(c.rewind_live(4).is_err());
    }

    /// A snapshot keeps a COPY of the open rows: the wave after it writes the open
    /// block in place, and restoring — twice, as a partial accept can — puts back
    /// exactly the rows and counters it took.
    #[test]
    fn a_snapshot_restores_the_open_rows_it_copied() {
        let d = 4usize;
        let rows = Q8Keys::quantize(&vec![0f32; 2 * d], d);
        let open_v: Vec<f32> = (0..2 * d).map(|i| i as f32 + 0.5).collect();
        let mut cache = IndexCache::from_rows(&rows, &open_v, d, &Device::Cpu).unwrap();
        let snap = cache.snapshot().unwrap();

        let dirty = |c: &mut IndexCache| {
            c.raw
                .slice_set(
                    &Tensor::full(-1f32, (MAX_RATIO, d), &Device::Cpu).unwrap(),
                    0,
                    0,
                )
                .unwrap();
            c.n_blocks = 5;
            c.n_open = 3;
        };
        for _ in 0..2 {
            dirty(&mut cache);
            cache.restore(&snap).unwrap();
            assert_eq!(cache.seal_shape(), (2, 2));
            assert_eq!(cache.open_rows_host().unwrap(), open_v);
        }
    }

    /// Snapshots draw on the cache's own rewind buffers, which a dropped snapshot
    /// gives back; more outstanding at once than there are buffers is refused by
    /// name. A snapshot of an empty open block needs no buffer at all.
    #[test]
    fn snapshots_draw_on_a_fixed_set_of_rewind_buffers() {
        let d = 4usize;
        let rows = Q8Keys::quantize(&vec![0f32; d], d);
        let bare = IndexCache::from_rows(&rows, &[], d, &Device::Cpu).unwrap();
        let free: Vec<_> = (0..2 * SNAPSHOT_BUFFERS)
            .map(|_| bare.snapshot().unwrap())
            .collect();
        assert!(free.iter().all(|s| s.raw.is_none()));

        let open = vec![1f32; d];
        let cache = IndexCache::from_rows(&rows, &open, d, &Device::Cpu).unwrap();
        let mut held: Vec<_> = (0..SNAPSHOT_BUFFERS)
            .map(|_| cache.snapshot().unwrap())
            .collect();
        let err = cache.snapshot().unwrap_err().to_string();
        assert!(err.contains("outstanding"), "{err}");
        held.pop();
        assert!(
            cache.snapshot().is_ok(),
            "a dropped snapshot gives its buffer back"
        );
    }

    /// The prompt end is a declaration about the slot, so it travels with the
    /// cache: a fork carries it, a truncate below it pulls it back to the cut,
    /// a truncate above it leaves it, and a reset forgets it. The blocks it
    /// covers are the whole indexed blocks below it, through the layout.
    #[test]
    fn the_prompt_end_travels_with_the_cache() -> Result<()> {
        let d = 2;
        // Four complete blocks and one open row at ratio 4: 17 positions.
        let mut c =
            IndexCache::from_rows(&Q8Keys::quantize(&[0.0; 8], d), &[0.0; 2], d, &Device::Cpu)?;
        assert_eq!((c.prompt_end(), c.prompt_blocks(4)), (0, 0));
        c.set_prompt_end(10);
        assert_eq!(c.prompt_blocks(4), 2);
        c.set_prompt_end(12);
        assert_eq!(c.prompt_blocks(4), 3);

        let mut fork = c.fork()?;
        assert_eq!(fork.prompt_end(), 12);
        fork.truncate_to(16)?;
        assert_eq!(fork.prompt_end(), 12, "a cut above the prompt keeps it");
        // A cut restarts the live tail at itself, so with no pages below it no
        // block is indexed under the clamped end until the rows are re-appended.
        c.truncate_to(8)?;
        assert_eq!((c.prompt_end(), c.prompt_blocks(4)), (8, 0));
        fork.reset();
        assert_eq!((fork.prompt_end(), fork.prompt_blocks(4)), (0, 0));
        Ok(())
    }
}
