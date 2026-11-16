//! One sample token's scan statistics in one fold group of one competition,
//! kept as an exact, mergeable summary (`docs/incremental_chunk_exclusion.md`
//! §3): the leading and runner-up files with their best agreements, and the
//! agreement sum and sum of squares over every gallery token scanned.
//!
//! **The merge is exact.** The sums add. The leading two are the top two of a
//! max over files, where a file's best only rises as its tokens are added: a
//! file outside both parts' top two is beaten in its own part by two distinct
//! files at least as good, so it cannot place in the merged top two. Ties rank
//! by the lower file id — the order the kernel's ascending case loop gives when
//! a scan's cases are laid out by file id — so folding a gallery a part at a
//! time gives bit-identical statistics to scanning it whole.

use candle_kernels::provenance::BdpStats;

/// A file — the timeline a gallery token belongs to — by its raw id; `0` is no
/// file (timeline ids are non-zero).
pub type FileId = u64;

/// See the module documentation.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct EntryStats {
    /// The leading file's best agreement.
    pub a1: u32,
    /// The leading file; `0` when no file scored.
    pub c1: FileId,
    /// The runner-up file's best agreement.
    pub a2: u32,
    /// The runner-up file; `0` when under two scored.
    pub c2: FileId,
    /// Agreement sum over the gallery tokens scanned.
    pub sum: u64,
    /// Agreement sum of squares over them.
    pub sumsq: u64,
}

impl EntryStats {
    /// A kernel row, its global cases named by `file_of`.
    pub fn of_row(row: &BdpStats, file_of: impl Fn(i32) -> FileId) -> Self {
        Self {
            a1: row.a1,
            c1: if row.c1 < 0 { 0 } else { file_of(row.c1) },
            a2: row.a2,
            c2: if row.c2 < 0 { 0 } else { file_of(row.c2) },
            sum: row.sum,
            sumsq: row.sumsq,
        }
    }

    /// Fold `other` — the statistics of more of the gallery — into these.
    pub fn merge(&mut self, other: &EntryStats) {
        self.merge_leaders(other);
        self.sum += other.sum;
        self.sumsq += other.sumsq;
    }

    /// Fold only `other`'s leading two into these, keeping the sums: a rescan
    /// that finds the leaders again over a gallery whose sums are already held.
    pub fn merge_leaders(&mut self, other: &EntryStats) {
        let mut cands: [(u32, FileId); 4] = [
            (self.a1, self.c1),
            (self.a2, self.c2),
            (other.a1, other.c1),
            (other.a2, other.c2),
        ];
        // A file named twice keeps its best.
        for i in 0..4 {
            for j in i + 1..4 {
                if cands[i].1 != 0 && cands[i].1 == cands[j].1 {
                    cands[i].0 = cands[i].0.max(cands[j].0);
                    cands[j] = (0, 0);
                }
            }
        }
        let mut live: Vec<(u32, FileId)> = cands
            .into_iter()
            .filter(|&(a, c)| c != 0 && a > 0)
            .collect();
        live.sort_by(|x, y| y.0.cmp(&x.0).then(x.1.cmp(&y.1)));
        let (a1, c1) = live.first().copied().unwrap_or((0, 0));
        let (a2, c2) = live.get(1).copied().unwrap_or((0, 0));
        self.a1 = a1;
        self.c1 = c1;
        self.a2 = a2;
        self.c2 = c2;
    }

    /// Take `other`'s sums out of these: a gallery stream removed.
    pub fn subtract_sums(&mut self, other: &EntryStats) {
        self.sum = self.sum.saturating_sub(other.sum);
        self.sumsq = self.sumsq.saturating_sub(other.sumsq);
    }

    /// Whether `file` is the leading or runner-up file.
    pub fn leads(&self, file: FileId) -> bool {
        file != 0 && (self.c1 == file || self.c2 == file)
    }

    /// The vote these make over `n` gallery tokens: `z × margin` for the
    /// leading file, the expression `bdp_vote` computes on the card.
    pub fn vote(&self, n: u64) -> f32 {
        if self.c1 == 0 || n == 0 {
            return 0.0;
        }
        let n = n as f32;
        let mean = self.sum as f32 / n;
        let var = (self.sumsq as f32 / n - mean * mean).max(1e-6);
        let z = ((self.a1 as f32 - mean) / var.sqrt()).max(0.0);
        z * (self.a1 - self.a2) as f32
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn lead(a1: u32, c1: FileId, a2: u32, c2: FileId) -> EntryStats {
        EntryStats {
            a1,
            c1,
            a2,
            c2,
            ..EntryStats::default()
        }
    }

    /// **Merging two parts gives the whole's leaders and sums.** File 7 leads
    /// one part, file 3 the other with more; 7's best over both parts is its
    /// own part's.
    #[test]
    fn merging_parts_gives_the_wholes_leaders_and_sums() {
        let mut a = EntryStats {
            sum: 100,
            sumsq: 4_000,
            ..lead(300, 7, 280, 9)
        };
        let b = EntryStats {
            sum: 50,
            sumsq: 1_000,
            ..lead(310, 3, 290, 7)
        };
        a.merge(&b);
        assert_eq!(
            a,
            EntryStats {
                a1: 310,
                c1: 3,
                a2: 300,
                c2: 7,
                sum: 150,
                sumsq: 5_000
            }
        );
    }

    /// **Ties rank by the lower file**, whichever part holds which.
    #[test]
    fn ties_rank_by_the_lower_file() {
        let mut a = lead(300, 9, 200, 4);
        a.merge(&lead(300, 5, 300, 8));
        assert_eq!(a, lead(300, 5, 300, 8));
        let mut b = lead(300, 5, 300, 8);
        b.merge(&lead(300, 9, 200, 4));
        assert_eq!(b, lead(300, 5, 300, 8));
    }

    /// **Merging is order-independent** over every order of four parts.
    #[test]
    fn merging_is_order_independent() {
        let parts = [
            EntryStats {
                sum: 1,
                sumsq: 1,
                ..lead(120, 11, 110, 12)
            },
            EntryStats {
                sum: 2,
                sumsq: 4,
                ..lead(130, 12, 90, 13)
            },
            EntryStats {
                sum: 3,
                sumsq: 9,
                ..lead(125, 14, 120, 11)
            },
            EntryStats {
                sum: 4,
                sumsq: 16,
                ..lead(100, 15, 0, 0)
            },
        ];
        let orders = [[0, 1, 2, 3], [3, 2, 1, 0], [1, 3, 0, 2], [2, 0, 3, 1]];
        for order in orders {
            let mut s = EntryStats::default();
            for i in order {
                s.merge(&parts[i]);
            }
            assert_eq!(
                s,
                EntryStats {
                    a1: 130,
                    c1: 12,
                    a2: 125,
                    c2: 14,
                    sum: 10,
                    sumsq: 30
                }
            );
        }
    }

    /// **Leaders merge without the sums**, and sums come out exactly.
    #[test]
    fn leaders_merge_alone_and_sums_subtract() {
        let mut s = EntryStats {
            sum: 90,
            sumsq: 900,
            ..lead(200, 4, 150, 6)
        };
        s.merge_leaders(&EntryStats {
            sum: 7,
            sumsq: 7,
            ..lead(210, 8, 0, 0)
        });
        assert_eq!((s.a1, s.c1, s.a2, s.c2, s.sum), (210, 8, 200, 4, 90));
        s.subtract_sums(&EntryStats {
            sum: 40,
            sumsq: 500,
            ..EntryStats::default()
        });
        assert_eq!((s.sum, s.sumsq), (50, 400));
        assert!(s.leads(8) && s.leads(4) && !s.leads(6) && !s.leads(0));
    }

    /// **A kernel row's cases are named by file**; none stays none.
    #[test]
    fn a_rows_cases_are_named_by_file() {
        let row = BdpStats {
            a1: 300,
            c1: 2,
            a2: 0,
            c2: -1,
            sum: 5,
            sumsq: 25,
        };
        let s = EntryStats::of_row(&row, |c| [70, 71, 72][c as usize]);
        assert_eq!(
            s,
            EntryStats {
                a1: 300,
                c1: 72,
                a2: 0,
                c2: 0,
                sum: 5,
                sumsq: 25
            }
        );
    }

    /// **The vote is `z × margin`**: two tokens at agreements 100 and 300 —
    /// mean 200, spread 100 — the leader at 300 over a runner-up at 290 is
    /// z 1 × margin 10.
    #[test]
    fn the_vote_is_z_times_margin() {
        let s = EntryStats {
            sum: 400,
            sumsq: 100_000,
            ..lead(300, 1, 290, 2)
        };
        assert_eq!(s.vote(2), 10.0);
        assert_eq!(EntryStats::default().vote(2), 0.0);
    }
}
