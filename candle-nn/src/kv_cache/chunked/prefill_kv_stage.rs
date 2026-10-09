//! What one bulk prefill attention launch carves beside its context, and how
//! the launch is cut to keep that bounded at any depth.
//!
//! The int8 prefill kernel (`candle-kernels/src/paged-prefill/`) pre-stages
//! the key positions of every sequence with at least
//! [`PREFILL_KV_STAGE_MIN_Q_LEN`] query rows — decoded, rotated and quantised
//! once (`kv_stage.cuh`) — and its blocks copy tile columns out of the stage.
//! A stage of a sequence's whole kv length grows with context depth, so the
//! launcher cuts the launch in two dimensions, each by a fixed constant:
//!
//! - **Row groups** of [`PREFILL_ATTN_ROW_GROUP_TOKENS`] query tokens — a
//!   contiguous range of the kernel's grid-x blocks, applied to every sequence
//!   of the batch. Blocks are independent, so a row group's launch writes
//!   exactly the bytes the whole grid's launch would for those blocks.
//! - **Key chunks** of [`PREFILL_KV_STAGE_CHUNK_POSITIONS`] positions, inside
//!   each row group, up to that group's causal horizon. Each chunk stages its
//!   positions and runs the attention over them; a block whose walk reaches
//!   past the chunk stops at a tile boundary and leaves its online-softmax
//!   state (O accumulator, running max, running sum — FP32, exact) in the
//!   **carry** and its walk cursor in the **resume** table, and the next chunk's
//!   launch picks both up. The output is the single launch's bit for bit.
//!
//! Row groups exist only to bound the carry: a launch in which nothing carries
//! — every staged sequence fits one key chunk — runs its whole grid as one row
//! group, one pre-staging pass and one attention launch.
//!
//! So a launch carves a stage of at most one chunk per staged sequence, a carry
//! of at most one row group per staged sequence that needs more than one chunk,
//! and the resume table beside it — the same figure at 128K as at 1M.
//! [`PrefillKvStageLayout`] is **the one definition** of that carve: the
//! launcher sizes and places its buffer by it, and the wave plan prices
//! `WaveBuffer::PrefillKvStage` by it.

use super::wave_plan::BUMP_ALIGNMENT;

/// Query rows at which a sequence's prefill launch **pre-stages** its K/V —
/// decodes, rotates and quantises every key position once, ahead of the
/// attention kernel, rather than in every block that selects it
/// (`candle-kernels/src/paged-prefill/kv_stage.cuh`).
///
/// A prefill block serves a couple of query tokens and decodes every position
/// it selects, so a chunk of `q` rows decodes each selected position about
/// `q / 2` times per KV head over; staging decodes each position once, but
/// every position of the sequence. A bulk chunk — hundreds to thousands of
/// rows — is far past where that pays. A verify window — a handful of rows
/// over a long prefix — is far short of it: it would stage a whole context to
/// read a few thousand positions of it. The line sits where a chunk's blocks
/// decode about as many columns as staging its whole context would at the
/// budget's depth, with room either side for the two shapes the engine runs.
pub const PREFILL_KV_STAGE_MIN_Q_LEN: usize = 256;

/// Key positions one launch stages per sequence — the key-chunk width. A
/// multiple of the kernel's 32-position tile, so a chunk boundary never cuts a
/// dense tile or a staged column quad.
pub const PREFILL_KV_STAGE_CHUNK_POSITIONS: usize = 32_768;

/// Query tokens one launch serves per sequence when the launch carries — the
/// row-group height, rounded down to whole kernel blocks
/// ([`prefill_attn_block_tokens`]). Each row group re-stages the key chunks its
/// horizon reaches, so a taller group stages less often for a larger carry.
pub const PREFILL_ATTN_ROW_GROUP_TOKENS: usize = 4_096;

/// The kernel's 32-position tile: key chunks are cut on its grid.
const KV_TILE: usize = 32;

/// Bytes of one carried softmax row: the O accumulator over `head_dim` dims,
/// then the running max and the running sum, all FP32.
const fn carry_row_bytes(head_dim: usize) -> usize {
    (head_dim + 2) * 4
}

/// Bytes of one resume entry: the walk bound and the tile ordinal, `u32` each.
const RESUME_ENTRY_BYTES: usize = 8;

/// The two bounds a launch is cut by. Production uses [`Self::PRODUCTION`];
/// the bit-identity tests force small ones to cut launches the production
/// figures would not.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct PrefillLaunchBounds {
    /// Key positions per chunk — a positive multiple of 32.
    pub chunk_positions: usize,
    /// Query tokens per row group, before rounding down to whole blocks.
    pub row_group_tokens: usize,
}

impl PrefillLaunchBounds {
    pub const PRODUCTION: Self = Self {
        chunk_positions: PREFILL_KV_STAGE_CHUNK_POSITIONS,
        row_group_tokens: PREFILL_ATTN_ROW_GROUP_TOKENS,
    };
}

/// Query tokens one block of the int8 prefill kernel serves: its M rows —
/// 64 through head_dim 128, 32 at 256 (`i8_m_rows`) — over the heads of a GQA
/// group, at least one.
pub const fn prefill_attn_block_tokens(n_head: usize, n_kv_head: usize, head_dim: usize) -> usize {
    let m_rows = if head_dim >= 256 { 32 } else { 64 };
    let hpg = if n_kv_head > 0 && n_head / n_kv_head > 0 {
        n_head / n_kv_head
    } else {
        1
    };
    let tokens = m_rows / hpg;
    if tokens == 0 {
        1
    } else {
        tokens
    }
}

/// Key positions a single uncut launch over `q_lens` at `offsets` would stage:
/// the whole kv length (`offset + q_len`) of every sequence with at least
/// [`PREFILL_KV_STAGE_MIN_Q_LEN`] rows. What the stage-only readback stages.
pub fn prefill_kv_stage_positions(q_lens: &[usize], offsets: &[usize]) -> usize {
    q_lens
        .iter()
        .zip(offsets)
        .filter(|(&q, _)| q >= PREFILL_KV_STAGE_MIN_Q_LEN)
        .map(|(&q, &off)| off + q)
        .sum()
}

/// Bytes of the pre-staged K/V for `positions` key positions over `n_kv_head`
/// heads at `head_dim`: int8 K codes, their FP16 per-32-dim-window scales and
/// FP16 V, each `[n_kv_head][positions][…]`, the first two planes rounded up
/// to [`BUMP_ALIGNMENT`] so each plane starts aligned — `kv_stage_bytes` in
/// `kv_stage.cuh`, which the launcher holds a buffer to.
pub fn prefill_kv_stage_bytes(positions: usize, n_kv_head: usize, head_dim: usize) -> usize {
    let rows = n_kv_head * positions;
    align(rows * head_dim) + align(rows * (head_dim / 32) * 2) + rows * head_dim * 2
}

fn align(bytes: usize) -> usize {
    bytes.div_ceil(BUMP_ALIGNMENT) * BUMP_ALIGNMENT
}

/// The carve of one prefill attention launch — stage, carry and resume, one
/// buffer, each region starting on [`BUMP_ALIGNMENT`] — and the cut it is
/// sized for. `Default` is a launch that stages nothing and carves nothing.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct PrefillKvStageLayout {
    /// Key positions per chunk.
    pub chunk_positions: usize,
    /// Query tokens per kernel block at this geometry.
    pub block_tokens: usize,
    /// Kernel blocks (grid x) per row group — the whole grid when nothing
    /// carries.
    pub group_blocks: usize,
    /// Rows per KV head of the stage: Σ over staged sequences of
    /// `min(kv_len, chunk_positions)` — the widest chunk any launch stages.
    pub stage_positions: usize,
    /// Query tokens the carry holds rows for: Σ over staged sequences deeper
    /// than one chunk of `min(q_len, group tokens)`. Each token carries one
    /// row per query head.
    pub carry_tokens: usize,
    /// Resume entries: Σ over the same sequences of their blocks in one row
    /// group, times the KV heads.
    pub resume_blocks: usize,
    n_head: usize,
    n_kv_head: usize,
    head_dim: usize,
}

impl PrefillKvStageLayout {
    /// A launch that stages and carves nothing.
    pub const NONE: Self = Self {
        chunk_positions: 0,
        block_tokens: 0,
        group_blocks: 0,
        stage_positions: 0,
        carry_tokens: 0,
        resume_blocks: 0,
        n_head: 0,
        n_kv_head: 0,
        head_dim: 0,
    };

    /// The layout of a launch over `q_lens` at `offsets`, cut by `bounds`, for
    /// `n_head` query and `n_kv_head` KV heads at `head_dim`.
    ///
    /// # Panics
    /// When `bounds` is not a positive multiple-of-32 chunk and a positive row
    /// group — a host bug, not a runtime condition.
    pub fn new(
        q_lens: &[usize],
        offsets: &[usize],
        n_head: usize,
        n_kv_head: usize,
        head_dim: usize,
        bounds: PrefillLaunchBounds,
    ) -> Self {
        assert!(
            bounds.chunk_positions > 0 && bounds.chunk_positions.is_multiple_of(KV_TILE),
            "prefill stage chunk of {} positions is not a positive multiple of {KV_TILE}",
            bounds.chunk_positions
        );
        assert!(
            bounds.row_group_tokens > 0,
            "prefill row group of zero tokens"
        );
        let c = bounds.chunk_positions;
        let block_tokens = prefill_attn_block_tokens(n_head, n_kv_head, head_dim);
        let group_blocks = (bounds.row_group_tokens / block_tokens).max(1);
        let group_tokens = group_blocks * block_tokens;
        let mut layout = Self {
            chunk_positions: c,
            block_tokens,
            group_blocks,
            n_head,
            n_kv_head,
            head_dim,
            ..Self::NONE
        };
        for (&q, &off) in q_lens.iter().zip(offsets) {
            if q < PREFILL_KV_STAGE_MIN_Q_LEN {
                continue;
            }
            let kv = off + q;
            layout.stage_positions += kv.min(c);
            if kv > c {
                layout.carry_tokens += q.min(group_tokens);
                layout.resume_blocks += q.div_ceil(block_tokens).min(group_blocks) * n_kv_head;
            }
        }
        // Nothing carries: there is no carry to bound, so the whole grid is one
        // row group — the uncut launch.
        if layout.carry_tokens == 0 {
            let max_q = q_lens.iter().copied().max().unwrap_or(0);
            layout.group_blocks = max_q.div_ceil(block_tokens).max(1);
        }
        layout
    }

    /// Query tokens per row group.
    pub fn group_tokens(&self) -> usize {
        self.group_blocks * self.block_tokens
    }

    /// Bytes of the stage region.
    pub fn stage_bytes(&self) -> usize {
        prefill_kv_stage_bytes(self.stage_positions, self.n_kv_head, self.head_dim)
    }

    /// Bytes of the carry region: one FP32 `[o; head_dim] ++ [m, l]` row per
    /// carried (token, query head).
    pub fn carry_bytes(&self) -> usize {
        self.carry_tokens * self.n_head * carry_row_bytes(self.head_dim)
    }

    /// Bytes of the resume region.
    pub fn resume_bytes(&self) -> usize {
        self.resume_blocks * RESUME_ENTRY_BYTES
    }

    /// Where the carry starts in the buffer.
    pub fn carry_offset(&self) -> usize {
        align(self.stage_bytes())
    }

    /// Where the resume table starts in the buffer.
    pub fn resume_offset(&self) -> usize {
        self.carry_offset() + align(self.carry_bytes())
    }

    /// Bytes the launch carves: the stage alone when nothing carries, else
    /// stage, carry and resume, each region aligned.
    pub fn bytes(&self) -> usize {
        if self.carry_tokens == 0 {
            self.stage_bytes()
        } else {
            self.resume_offset() + self.resume_bytes()
        }
    }

    /// Key chunks each row group of the launch over `q_lens` at `offsets` runs
    /// — the launcher's loop, one entry per row group of the widest sequence.
    ///
    /// A group runs the chunks its staged sequences' causal horizons reach:
    /// for each staged sequence with a block in the group, the furthest
    /// position any of those blocks attends is
    /// `min(kv_len, offset + min(group_end, q_len))`. At least one chunk, which
    /// is the only one when nothing in the group reaches past the first — a
    /// launch in which nothing carries is one group of one chunk, exactly one
    /// kernel launch.
    pub fn group_chunks(&self, q_lens: &[usize], offsets: &[usize]) -> Vec<u32> {
        let max_q = q_lens.iter().copied().max().unwrap_or(0);
        let grid_x = max_q.div_ceil(self.block_tokens.max(1)).max(1);
        let groups = grid_x.div_ceil(self.group_blocks.max(1));
        let gt = self.group_tokens();
        (0..groups)
            .map(|g| {
                let (lo, hi) = (g * gt, (g + 1) * gt);
                let reach = q_lens
                    .iter()
                    .zip(offsets)
                    .filter(|(&q, _)| q >= PREFILL_KV_STAGE_MIN_Q_LEN && q > lo)
                    .map(|(&q, &off)| off + q.min(hi))
                    .max()
                    .unwrap_or(0);
                reach.div_ceil(self.chunk_positions.max(1)).max(1) as u32
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Flash-Next attention: 24 query heads over 2 KV heads at head_dim 256 —
    /// 12 heads a group, 32 M rows, 2 tokens a block.
    const FN_HEADS: (usize, usize, usize) = (24, 2, 256);

    fn flash_next(q_lens: &[usize], offsets: &[usize]) -> PrefillKvStageLayout {
        let (h, kv, hd) = FN_HEADS;
        PrefillKvStageLayout::new(q_lens, offsets, h, kv, hd, PrefillLaunchBounds::PRODUCTION)
    }

    #[test]
    fn block_tokens_follow_the_kernels_m_rows() {
        assert_eq!(prefill_attn_block_tokens(24, 2, 256), 2);
        assert_eq!(prefill_attn_block_tokens(32, 4, 128), 8);
        assert_eq!(prefill_attn_block_tokens(16, 16, 64), 64);
        assert_eq!(prefill_attn_block_tokens(16, 8, 256), 16);
        // 3 heads a group: 64 / 3 rounds down to 21 tokens.
        assert_eq!(prefill_attn_block_tokens(12, 4, 128), 21);
    }

    /// A launch pre-stages the whole kv length of each sequence with a bulk
    /// chunk, and nothing of a short one: a verify window, a decode-shaped
    /// row and a chunk one short of the line stage nothing.
    #[test]
    fn the_whole_stage_counts_bulk_sequences_whole_and_nothing_else() {
        assert_eq!(PREFILL_KV_STAGE_MIN_Q_LEN, 256);
        assert_eq!(
            prefill_kv_stage_positions(&[5], &[4196]),
            0,
            "a verify window"
        );
        assert_eq!(prefill_kv_stage_positions(&[255], &[0]), 0);
        assert_eq!(prefill_kv_stage_positions(&[256], &[0]), 256);
        assert_eq!(
            prefill_kv_stage_positions(&[5, 256, 2048, 255], &[4196, 100, 8192, 0]),
            (100 + 256) + (8192 + 2048),
            "the bulk sequences' prefix and chunk alike; the short ones not at all"
        );
    }

    /// The stage's bytes, to the byte: int8 K codes, FP16 window scales, FP16
    /// V — 784 B a position per KV head at head_dim 256 — the first two planes
    /// rounded up to the bump alignment.
    #[test]
    fn the_stage_is_784_bytes_a_position_at_head_dim_256() {
        assert_eq!(prefill_kv_stage_bytes(131_072 + 8192, 2, 256), 218_365_952);
        // 21,192 rows: K 5,425,152 B; scales 339,072 B, padded to 339,200;
        // V 10,850,304 B.
        assert_eq!(
            prefill_kv_stage_bytes(10_596, 2, 256),
            5_425_152 + 339_200 + 10_850_304
        );
        // Three rows at head_dim 64: K 192 → 256, scales 12 → 256, V 384.
        assert_eq!(prefill_kv_stage_bytes(3, 1, 64), 256 + 256 + 384);
        assert_eq!(prefill_kv_stage_bytes(0, 2, 256), 0);
    }

    /// An 8K chunk at 4K of history fits one key chunk: the stage is its whole
    /// 12,288 positions, nothing carries, and the whole 4,096-block grid is one
    /// row group — the uncut launch and its buffer.
    #[test]
    fn flash_next_at_4k_stages_its_whole_context_and_carries_nothing() {
        let l = flash_next(&[8192], &[4096]);
        assert_eq!(l.block_tokens, 2);
        assert_eq!(l.group_blocks, 4096);
        assert_eq!(l.group_tokens(), 8192);
        assert_eq!(l.stage_positions, 12_288);
        assert_eq!(l.carry_tokens, 0);
        assert_eq!(l.resume_blocks, 0);
        // K 6,291,456 + scales 393,216 + V 12,582,912.
        assert_eq!(l.bytes(), 19_267_584);
        assert_eq!(l.bytes(), prefill_kv_stage_bytes(12_288, 2, 256));
        assert_eq!(l.group_chunks(&[8192], &[4096]), vec![1]);
    }

    /// Past one chunk the carve is one chunk of stage, one row group of carry
    /// and its resume table — 152,862,720 B at 128K, 295K and 1M alike, where
    /// the uncut stage was 218 MB, 475 MB and 1.66 GB.
    #[test]
    fn flash_next_past_one_chunk_carves_the_same_at_any_depth() {
        // Stage: 32,768 positions × 2 heads × 784 B.
        let stage = 51_380_224;
        // Carry: 4,096 tokens × 24 heads × 258 floats × 4 B.
        let carry = 101_449_728;
        // Resume: 2,048 blocks × 2 KV heads × 8 B.
        let resume = 32_768;
        for (depth, uncut) in [
            (131_072, 218_365_952usize),
            (295_000, 475_405_056),
            (1_048_576, 1_657_012_224),
        ] {
            let l = flash_next(&[8192], &[depth]);
            assert_eq!(l.group_blocks, 2048);
            assert_eq!(l.stage_positions, 32_768);
            assert_eq!(l.carry_tokens, 4096);
            assert_eq!(l.resume_blocks, 4096);
            assert_eq!(l.stage_bytes(), stage);
            assert_eq!(l.carry_bytes(), carry);
            assert_eq!(l.resume_bytes(), resume);
            assert_eq!(l.carry_offset(), stage);
            assert_eq!(l.resume_offset(), stage + carry);
            assert_eq!(l.bytes(), 152_862_720);
            assert_eq!(
                prefill_kv_stage_bytes(prefill_kv_stage_positions(&[8192], &[depth]), 2, 256),
                uncut
            );
        }
        // Each row group runs the chunks its horizon reaches.
        assert_eq!(
            flash_next(&[8192], &[131_072]).group_chunks(&[8192], &[131_072]),
            // Horizons 135,168 / 139,264 → 5 chunks each.
            vec![5, 5]
        );
        assert_eq!(
            flash_next(&[8192], &[295_000]).group_chunks(&[8192], &[295_000]),
            vec![10, 10]
        );
        assert_eq!(
            flash_next(&[8192], &[1_048_576]).group_chunks(&[8192], &[1_048_576]),
            // Horizons 1,052,672 / 1,056,768 → 33 chunks (32 reach 1,048,576).
            vec![33, 33]
        );
    }

    /// A row group whose horizon stays inside the first chunk runs one chunk
    /// even when a later group of the same sequence needs two.
    #[test]
    fn a_group_runs_only_the_chunks_its_horizon_reaches() {
        let l = flash_next(&[8192], &[28_000]);
        // Horizons 32,096 (inside 32,768) and 36,192.
        assert_eq!(l.group_chunks(&[8192], &[28_000]), vec![1, 2]);
        assert_eq!(l.stage_positions, 32_768);
        assert_eq!(l.carry_tokens, 4096);
    }

    /// A mixed batch: a deep bulk chunk, an unstaged verify-sized row and a
    /// shallow bulk chunk. Only the deep one carries; the unstaged one stages
    /// nothing.
    #[test]
    fn flash_next_mixed_batch() {
        let (q, off) = ([8192, 100, 300], [295_000, 50_000, 2000]);
        let l = flash_next(&q, &off);
        // 32,768 of the deep sequence + the shallow one's 2,300.
        assert_eq!(l.stage_positions, 35_068);
        // K 17,954,816; scales 1,122,176 → 1,122,304; V 35,909,632.
        assert_eq!(l.stage_bytes(), 54_986_752);
        assert_eq!(l.carry_tokens, 4096);
        assert_eq!(l.resume_blocks, 4096);
        assert_eq!(l.carry_offset(), 54_986_752);
        assert_eq!(l.resume_offset(), 54_986_752 + 101_449_728);
        assert_eq!(l.bytes(), 156_469_248);
        assert_eq!(l.group_chunks(&q, &off), vec![10, 10]);
    }

    /// Small bounds, as the bit-identity tests force them: a 300-row chunk at
    /// 1,000 of history under 512-position chunks and 128-token groups.
    #[test]
    fn forced_bounds_cut_small_launches() {
        let bounds = PrefillLaunchBounds {
            chunk_positions: 512,
            row_group_tokens: 128,
        };
        let l = PrefillKvStageLayout::new(&[300], &[1000], 24, 2, 256, bounds);
        assert_eq!(l.group_blocks, 64);
        assert_eq!(l.stage_positions, 512);
        // min(300, 128) tokens; 64 blocks × 2 KV heads.
        assert_eq!(l.carry_tokens, 128);
        assert_eq!(l.resume_blocks, 128);
        // Groups end at 128, 256, 384 tokens: horizons 1,128 / 1,256 / 1,300.
        assert_eq!(l.group_chunks(&[300], &[1000]), vec![3, 3, 3]);
        // A 300-row prompt from zero fits one chunk: nothing carries, so the
        // whole 150-block grid is one group of one chunk.
        let fresh = PrefillKvStageLayout::new(&[300], &[0], 24, 2, 256, bounds);
        assert_eq!(fresh.carry_tokens, 0);
        assert_eq!(fresh.group_blocks, 150);
        assert_eq!(fresh.group_chunks(&[300], &[0]), vec![1]);
    }

    #[test]
    fn nothing_staged_carves_nothing() {
        let l = flash_next(&[5, 100], &[300_000, 10]);
        assert_eq!(l.bytes(), 0);
        assert_eq!(l.group_chunks(&[5, 100], &[300_000, 10]), vec![1]);
        assert_eq!(PrefillKvStageLayout::default().bytes(), 0);
    }

    #[test]
    #[should_panic(expected = "not a positive multiple of 32")]
    fn a_chunk_off_the_tile_grid_is_refused() {
        let bounds = PrefillLaunchBounds {
            chunk_positions: 100,
            row_group_tokens: 128,
        };
        let _ = PrefillKvStageLayout::new(&[300], &[0], 24, 2, 256, bounds);
    }
}
