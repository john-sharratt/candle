//! Feature-gated KV-zero detector (`kv-zero-check`).
//!
//! A zeroed R16 K/V token would be *soft-dropped* by attention (RoPE of zero is zero,
//! so a fixed-0 score; a zero value injects nothing) — silent, so a corrupt block can
//! go unnoticed. This module makes it loud, but crucially it validates against the
//! **real token window the metadata claims**, not a naive `block×32` layout.
//!
//! The scan consults [`BatchedInferenceSession::provenance_chunk_layout`] — the SAME
//! per-chunk `(offset, len, cum_before)` derivation the decode GPU buffer feeds
//! attention (the writer chunk's length is the `seq_offset`-derived one). For each
//! chunk it checks only the real slots `[offset, offset+len)` and skips the partial-
//! chunk padding tail. So a clean run means every metadata-designated real token has
//! live K/V — the metadata is correct wherever it's consulted — and an error means
//! either genuine corruption OR metadata that points at padding as if it were a token.
//!
//! Positions are true token-space (`cum_before + j`), so the `region` split (own vs
//! prefix, against `turn_start`) is exact, not block-rounded. Hooked at four sites
//! (see `scheduler::mod`): `decode-reproject`, `decode-final-view`,
//! `substrate-write-seal`, `projection-inject`.

use std::collections::BTreeMap;

use candle_nn::CHUNK_SIZE;

/// R16 sub-band palette split — matches `candle_nn::kv_cache::arena_table::N_PALETTE`.
/// A shared structural constant like `CHUNK_SIZE`.
const N_PAL: usize = 4;

/// One gathered layer's blocks: `(block_idx, k_flat, v_flat, q_flat)` as returned by
/// `BatchedInferenceSession::gather_r16_kv_provenance_layers`.
type LayerBlocks = Vec<(usize, Vec<f32>, Vec<f32>, Vec<f32>)>;

/// A contiguous dead-position run in true token-space.
struct DeadRun {
    pos_start: usize,
    pos_end: usize,
    max_dead_layers: u32,
    any_k: bool,
    any_v: bool,
}

/// Scan a gathered R16 dump for zero K/V among the REAL token slots defined by
/// `layout` (`layout[block_idx] = (offset, len, cum_before)` — the attention-visible
/// window), group holes into contiguous token-space runs, and log a `tracing::error`
/// per run with its region relative to `turn_start`. Padding (physical slots outside
/// `[offset, offset+len)`) is never checked.
#[allow(clippy::too_many_arguments)]
pub(crate) fn scan_gathered(
    phase: &str,
    seq: usize,
    dump: &[LayerBlocks],
    layout: &[(u16, u16, usize)],
    n_kv_head: usize,
    head_dim: usize,
    turn_start: usize,
) {
    if n_kv_head == 0 || head_dim == 0 || layout.is_empty() {
        return;
    }
    let sub = (head_dim / N_PAL).max(1);
    let warp_seg = CHUNK_SIZE * sub; // per (head,palette) segment length in the flat buffer

    // Aggregate deadness across layers, keyed by true token-space position.
    let mut at: BTreeMap<usize, (u32, bool, bool)> = BTreeMap::new();
    for blocks in dump {
        for (block_idx, k_flat, v_flat, _q) in blocks {
            if k_flat.is_empty() {
                continue;
            }
            let Some(&(offset, len, cum)) = layout.get(*block_idx) else {
                continue; // no metadata for this chunk — cannot classify, skip
            };
            let offset = offset as usize;
            for j in 0..len as usize {
                let t = offset + j; // physical slot of real token j
                if t >= CHUNK_SIZE {
                    break;
                }
                let k_zero = token_all_zero(k_flat, t, warp_seg, sub);
                let v_zero = token_all_zero(v_flat, t, warp_seg, sub);
                if k_zero || v_zero {
                    let pos = cum + j; // true token-space position
                    let e = at.entry(pos).or_insert((0, false, false));
                    e.0 += 1;
                    e.1 |= k_zero;
                    e.2 |= v_zero;
                }
            }
        }
    }
    if at.is_empty() {
        return;
    }

    // Group consecutive positions into runs.
    let mut runs: Vec<DeadRun> = Vec::new();
    for (&pos, &(dl, k, v)) in &at {
        match runs.last_mut() {
            Some(r) if pos == r.pos_end + 1 => {
                r.pos_end = pos;
                r.max_dead_layers = r.max_dead_layers.max(dl);
                r.any_k |= k;
                r.any_v |= v;
            }
            _ => runs.push(DeadRun {
                pos_start: pos,
                pos_end: pos,
                max_dead_layers: dl,
                any_k: k,
                any_v: v,
            }),
        }
    }

    let total_dead: usize = at.len();
    for r in &runs {
        let tokens = r.pos_end - r.pos_start + 1;
        let region = if r.pos_end < turn_start {
            "prefix" // borrowed/projected context — inherited from an upstream seal
        } else if r.pos_start >= turn_start {
            "own" // this turn's own written span — minted here
        } else {
            "straddle"
        };
        let rel_to_turn_start = r.pos_start as isize - turn_start as isize;
        tracing::error!(
            target: "kv_zero",
            phase,
            seq,
            region,
            pos_start = r.pos_start,
            pos_end = r.pos_end,
            tokens,
            turn_start,
            rel_to_turn_start,
            dead_layers = r.max_dead_layers,
            k = r.any_k,
            v = r.any_v,
            "zero KV run at a REAL token slot — metadata says token, KV is empty"
        );
    }
    tracing::error!(
        target: "kv_zero",
        phase,
        seq,
        runs = runs.len(),
        total_dead,
        turn_start,
        "KV-zero scan summary (real slots only)"
    );
}

/// True iff every element belonging to physical slot `t` (across all head/palette warp
/// segments) is exactly `0.0`. `flat` is laid out `[warp][token][sub]`, so slot `t`
/// occupies `[w*warp_seg + t*sub .. + sub)` in each of `flat.len()/warp_seg` warps.
fn token_all_zero(flat: &[f32], t: usize, warp_seg: usize, sub: usize) -> bool {
    if flat.is_empty() || warp_seg == 0 || flat.len() % warp_seg != 0 {
        return false; // unexpected shape — do not raise a false alarm
    }
    let n_warps = flat.len() / warp_seg;
    for w in 0..n_warps {
        let base = w * warp_seg + t * sub;
        for d in 0..sub {
            if flat[base + d] != 0.0 {
                return false;
            }
        }
    }
    true
}
