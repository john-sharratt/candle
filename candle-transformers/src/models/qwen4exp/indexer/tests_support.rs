//! Shared fixtures for the index tests: pages, caches, kernel cases and a
//! store-and-resume rig.
//!
//! Caches are built on the CPU where only widths matter, so the page arithmetic
//! is tested without a device. The rig appends hidden states in ragged waves, and
//! it puts a cache through the seal's record shape. The rebuilt cache must behave
//! identically to the original, not merely hold the same bytes.

use super::*;

// Pinned ranges are lists of ranges, and a list of one is the common case.
#[cfg(test)]
#[allow(clippy::single_range_in_vec_init)]
pub(super) mod tests {
    use super::*;

    use crate::models::rope_schedule::plain_inv_freq;

    pub(in super::super) fn cuda() -> Option<Device> {
        match Device::cuda_if_available(0) {
            Ok(d) if d.is_cuda() => Some(d),
            _ => {
                eprintln!("skipping: CUDA device required");
                None
            }
        }
    }

    /// A page of `host` rows (`[rows, d]` row-major), stored as the index
    /// stores them, whose last row covers `last` tokens, placed on `device` the
    /// way an injected record is.
    pub(in super::super) fn placed_page(
        host: &[f32],
        last: usize,
        d: usize,
        device: &Device,
    ) -> Result<Arc<ResidentPage>> {
        let keys = Q8Keys::quantize(host, d);
        let mut pages = ResidentPage::place_host(&[(&keys, last)], d, device)?;
        pages
            .pop()
            .ok_or_else(|| candle::Error::Msg("one page placed".into()))
    }

    /// A page of `rows` zero rows whose last covers `last` tokens, over
    /// slot-shaped chunks on `device` — the shape a closed tail hands over.
    pub(in super::super) fn zero_page(
        rows: usize,
        last: usize,
        d: usize,
        device: &Device,
    ) -> Arc<ResidentPage> {
        let chunks = (0..rows.div_ceil(PAGE_BLOCKS))
            .map(|_| Tensor::zeros((PAGE_BLOCKS, d), DType::F32, device).expect("chunk"))
            .collect();
        ResidentPage::from_key_pages(chunks, rows, last, d).expect("page")
    }

    /// An index cache holding `pages` (by token width) and `open` carried rows,
    /// built on the CPU.
    ///
    /// The page bookkeeping — prefix sums, the ragged walk, the token count — is
    /// pure arithmetic over widths, so it needs no device. Only the flush
    /// kernel does, which is why these tests can pin the contract the page cuts
    /// depend on without a GPU in the loop.
    pub(in super::super) fn caged(widths: &[usize], ratio: usize) -> IndexCache {
        let d = 4usize;
        let mut c = IndexCache::new(d, &Device::Cpu).expect("cpu cache");
        for &w in widths {
            // A page of `w` tokens is `ceil(w / ratio)` rows whose last one
            // covers the remainder — the shape a real seal produces.
            let rows = w.div_ceil(ratio);
            let last = w - (rows - 1) * ratio;
            let base = c.next_base();
            c.push_page(zero_page(rows, last, d, &Device::Cpu), base, ratio)
                .expect("push");
        }
        c
    }

    /// A page `w` tokens wide at `ratio`, as [`caged`] builds them.
    pub(in super::super) fn width_page(w: usize, ratio: usize) -> SplicePiece {
        let rows = w.div_ceil(ratio);
        SplicePiece::Page(zero_page(rows, w - (rows - 1) * ratio, 4, &Device::Cpu))
    }

    /// Two caches resolve every position the same way: the same span, the same
    /// candidates at every position, and every row starting at the same place.
    pub(in super::super) fn assert_same_walk(
        spliced: &IndexCache,
        built: &IndexCache,
        ratio: usize,
    ) {
        assert_eq!(spliced.next_base(), built.next_base(), "tail");
        assert_eq!(spliced.page_count(), built.page_count(), "pages");
        assert_eq!(spliced.page_row_span(), built.page_row_span(), "rows");
        for pos in 0..built.next_base() {
            assert_eq!(
                spliced.candidates_at(pos, ratio),
                built.candidates_at(pos, ratio),
                "candidates at {pos}"
            );
        }
        for row in 0..built.page_row_span() {
            assert_eq!(
                spliced.block_start(row, ratio),
                built.block_start(row, ratio),
                "row {row}"
            );
        }
    }

    pub(in super::super) fn lcg(n: usize, seed: u64, scale: f32) -> Vec<f32> {
        let mut s = seed;
        (0..n)
            .map(|_| {
                s = s
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                ((s >> 33) as f32 / (1u64 << 31) as f32) * scale
            })
            .collect()
    }

    /// What one kernel-against-definition case runs: the queries, the geometry,
    /// the scores' grid, and the strata with its prompt span.
    pub(in super::super) struct KernelCase {
        pub(in super::super) qpos: Vec<usize>,
        pub(in super::super) top_k: usize,
        // `Some(step)` collapses the scores onto a grid of `step`.
        pub(in super::super) quantize: Option<f32>,
        pub(in super::super) seed: u64,
        pub(in super::super) strata: Strata,
        // Row r's prompt spans `prompt_blocks + r · prompt_spread` blocks.
        pub(in super::super) prompt_blocks: u32,
        pub(in super::super) prompt_spread: u32,
        // Every row's pinned block ranges.
        pub(in super::super) pins: Vec<Range<usize>>,
    }

    /// Positions chosen so every row is past the budget and the phases of
    /// `qpos mod ratio` are all represented (the partial-block cut moves with
    /// the phase).
    pub(in super::super) fn shallow_rows() -> Vec<usize> {
        (0..16).map(|i| 40 + i * 7).collect()
    }

    /// Rows deep enough to cut into several small windows: 188 candidate
    /// blocks at the deepest.
    pub(in super::super) fn windowed_rows() -> Vec<usize> {
        (0..16).map(|i| 200 + i * 37).collect()
    }

    pub(in super::super) fn strata(
        window_blocks: usize,
        recent_blocks: usize,
        recent: Recent,
    ) -> Strata {
        Strata {
            window_blocks,
            recent_blocks,
            recent,
            pin_blocks: 0,
        }
    }

    // —— Store and resume ————————————————————————————————————————————————
    //
    // A resumed sequence's index has to be the index it would have had if the
    // process had never stopped. These build a cache by ragged appends, put it
    // through the seal's export shape, rebuild it, and then require the rebuilt
    // one to behave identically — not merely to hold the same bytes.

    /// The test rig the store/resume tests share: weights, the rope table, and
    /// a hidden-state stream long enough to append in ragged waves.
    pub(in super::super) struct Rig {
        pub(in super::super) device: Device,
        pub(in super::super) cfg: IndexerConfig,
        pub(in super::super) w: IndexerWeights,
        pub(in super::super) rope: FactoredRope,
        pub(in super::super) keys: Tensor,
        pub(in super::super) ratio: usize,
        pub(in super::super) eps: f64,
    }

    impl Rig {
        pub(in super::super) fn new(device: Device, tokens: usize) -> Result<Self> {
            let cfg = IndexerConfig {
                n_heads: 2,
                head_dim: 16,
                top_k: 8,
                strata: Strata::WHOLE,
            };
            let (ratio, hidden, eps) = (4usize, 12usize, 1e-6);
            let w = IndexerWeights {
                q_proj: Tensor::from_vec(
                    lcg(cfg.n_heads * cfg.head_dim * hidden, 0x71, 0.6),
                    (cfg.n_heads * cfg.head_dim, hidden),
                    &device,
                )?,
                k_proj: Tensor::from_vec(
                    lcg(cfg.head_dim * hidden, 0x72, 0.6),
                    (cfg.head_dim, hidden),
                    &device,
                )?,
                q_norm: Tensor::from_vec(
                    lcg(cfg.head_dim, 0x73, 0.2)
                        .into_iter()
                        .map(|v| v + 1.0)
                        .collect::<Vec<f32>>(),
                    (cfg.head_dim,),
                    &device,
                )?,
                k_norm: Tensor::from_vec(
                    lcg(cfg.head_dim, 0x74, 0.2)
                        .into_iter()
                        .map(|v| v + 1.0)
                        .collect::<Vec<f32>>(),
                    (cfg.head_dim,),
                    &device,
                )?,
            };
            let x = Tensor::from_vec(lcg(tokens * hidden, 0x75, 1.0), (tokens, hidden), &device)?;
            let rope = FactoredRope::new(&plain_inv_freq(8, 1e6), &device)?;
            let keys = project_keys(&x, &w)?;
            Ok(Self {
                device,
                cfg,
                w,
                rope,
                keys,
                ratio,
                eps,
            })
        }

        /// Append `rows` tokens starting at `start`.
        pub(in super::super) fn append(
            &self,
            cache: &mut IndexCache,
            start: usize,
            rows: usize,
        ) -> Result<()> {
            cache.ensure_capacity(start + rows, self.ratio)?;
            let mut work = [AppendSpan { cache, start, rows }];
            append_wave(&mut work, &self.keys, &self.w, self.ratio, self.eps)
        }

        /// A cache holding the first `tokens` tokens, appended in waves whose
        /// lengths are deliberately not multiples of `ratio`.
        pub(in super::super) fn build(&self, tokens: usize) -> Result<IndexCache> {
            let mut cache = IndexCache::new(self.cfg.head_dim, &self.device)?;
            let mut at = 0usize;
            for step in [7usize, 5, 11, 3].iter().cycle() {
                if at >= tokens {
                    break;
                }
                let rows = (*step).min(tokens - at);
                self.append(&mut cache, at, rows)?;
                at += rows;
            }
            assert_eq!(cache.len(self.ratio), tokens);
            Ok(cache)
        }

        /// The seal's export shape, through the record bytes and back — the
        /// whole persistence path, not a direct field copy.
        pub(in super::super) fn round_trip(&self, cache: &IndexCache) -> Result<IndexCache> {
            use crate::models::qwen4exp::paged_index::{decode_page, encode_page, SealedIndex};
            let sealed = SealedIndex {
                keys: cache.live_rows_host()?,
                last_cells: self.ratio,
                open: cache.open_rows_host()?,
            };
            let back = decode_page(&encode_page(&sealed)?)?;
            IndexCache::from_rows(&back.keys, &back.open, self.cfg.head_dim, &self.device)
        }

        /// Every row's selection, as the model would compute it.
        pub(in super::super) fn select(
            &self,
            cache: &mut IndexCache,
            qpos: &[usize],
        ) -> Result<Vec<Option<Vec<u32>>>> {
            let t = qpos.len();
            let widest = (cache.page_row_span() + cache.n_blocks).max(1);
            let x = Tensor::from_vec(
                lcg(t * self.w.q_proj.dim(1)?, 0x76, 1.0),
                (t, self.w.q_proj.dim(1)?),
                &self.device,
            )?;
            let q = project_queries(
                &x,
                &self.w,
                &self.cfg,
                &self.rope,
                qpos,
                RowRungs::Uniform(0),
                self.eps,
                None,
            )?;
            let scores = Tensor::empty((t, widest), DType::F32, &self.device)?;
            let cand = cache.score_rows(
                &q, qpos, &self.cfg, self.ratio, &self.rope, 0, &scores, widest, 0, None,
            )?;
            let mut table = SelectionTable::new(
                t,
                self.ratio,
                self.cfg.top_k,
                widest,
                &self.cfg.strata,
                &self.device,
                None,
            )?;
            let tail: Vec<u32> = qpos
                .iter()
                .map(|&p| cache.tail_len(p, self.ratio))
                .collect();
            let prompt = vec![cache.prompt_blocks(self.ratio) as u32; t];
            let pins = vec![cache.pin_row(self.ratio, self.cfg.strata.pin_blocks); t];
            table.fill_rows(
                &scores,
                &cand,
                qpos,
                &tail,
                &prompt,
                &pins,
                self.ratio,
                self.cfg.top_k,
                0,
            )?;
            (0..t).map(|r| table.row_to_host(r)).collect()
        }
    }
}
