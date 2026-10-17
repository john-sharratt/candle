//! The grids the QSA index append launches over (`indexer::append_wave`).
//!
//! **Sized by what the wave could complete, not by what it does.** A span of
//! `rows` rows entering with `n_open < ratio` rows already carried completes at
//! most `⌈rows / ratio⌉` blocks and carries rows at most once, so
//! those bounds — summed over the wave's spans and rounded up to a power of two
//! — depend only on the wave's widths. Sized by the counts instead, the append
//! grid moved between zero, one and two blocks from one token to the next, and
//! each launch vanished on the tokens whose count was zero: on `.5`
//! (2026-10-17) the append and carry were the first launch to differ in more
//! than half of 1,298 segments of a new shape, so a recorded segment holding
//! them never recurred. The blocks past the count return at once.

/// The append and carry launches' grids, in blocks.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AppendGrids {
    pub jobs: usize,
    pub carries: usize,
}

impl AppendGrids {
    /// The grids for a wave whose spans contribute `rows` rows each, `ratio`
    /// rows to a block.
    pub fn for_spans(rows: impl IntoIterator<Item = usize>, ratio: usize) -> Self {
        let (mut jobs, mut carries) = (0usize, 0usize);
        for r in rows.into_iter().filter(|&r| r > 0) {
            jobs += r.div_ceil(ratio);
            carries += 1;
        }
        Self {
            jobs: jobs.max(1).next_power_of_two(),
            carries: carries.max(1).next_power_of_two(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// **A decode wave's grids follow its widths alone**: one row a span at
    /// `ratio` 4 can complete one block whatever was carried, so two sessions
    /// launch two append blocks and two carry blocks every token.
    #[test]
    fn decode_grids_follow_the_widths() {
        assert_eq!(
            AppendGrids::for_spans([1, 1], 4),
            AppendGrids {
                jobs: 2,
                carries: 2
            }
        );
    }

    /// The bound covers every entering carry: whatever `n_open < ratio` rows
    /// a span enters with, `plan` completes `(n_open + rows) / ratio` blocks.
    #[test]
    fn the_bound_covers_the_fullest_open_block() {
        for ratio in 1..=8 {
            for rows in 1..=20 {
                let worst = (0..ratio).map(|open| (open + rows) / ratio).max().unwrap();
                assert!(
                    AppendGrids::for_spans([rows], ratio).jobs >= worst,
                    "ratio {ratio} rows {rows}"
                );
            }
        }
    }

    /// Verify widths that differ by a row round to the same grid, and a span
    /// with no rows adds nothing.
    #[test]
    fn nearby_widths_share_a_grid() {
        assert_eq!(
            AppendGrids::for_spans([5], 4),
            AppendGrids::for_spans([6], 4)
        );
        assert_eq!(
            AppendGrids::for_spans([4, 0], 4),
            AppendGrids::for_spans([4], 4)
        );
        assert_eq!(
            AppendGrids::for_spans([0], 4),
            AppendGrids {
                jobs: 1,
                carries: 1
            }
        );
    }
}
