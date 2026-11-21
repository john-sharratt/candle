//! Assigns blocks to the four palette slots for one side of the cache, K or V.
//!
//! The side is processed in two passes. The first claims the blocks that pass a
//! slot's format in the amax-descending order, and the second fills any quota the
//! first pass left short from the front of the blocks still unassigned. The head
//! scale is passed in so the error metrics divide by the same capped value the
//! statistics use. The order of claims is the kernel's, so the selection on this
//! side matches the CUDA path on the same data.

use super::*;

/// Phase 4+5 driver for one side (K or V). Returns per-slot format/scale
/// and per-dim palette assignments. `head_amax` is the head scale the error
/// metrics divide by — the head's amax as capped by [`head_scale`].
#[allow(clippy::too_many_arguments)]
pub fn process_side(
    data: &[f32],
    amax: &[f32; HEAD_DIM],
    idx_sorted: &[u16; HEAD_DIM],
    candidates: &[QuantFormat],
    head_amax: f32,
    is_k: bool,
    kthresh: &[f32; HEAD_DIM],
    v_thr_sq: f32,
) -> ([QuantFormat; N_PALETTE], [f32; N_PALETTE], [u8; HEAD_DIM]) {
    let safe_head_amax = head_amax.max(1.0e-8);
    let inv_head_amax = 1.0 / safe_head_amax;
    let inv_head_amax_sq = inv_head_amax * inv_head_amax;
    let fallback_last = *candidates.last().expect("non-empty candidate set");

    let mut alive = [true; HEAD_DIM];
    let mut slot_fmt = [fallback_last; N_PALETTE];
    let mut slot_scale = [1.0f32; N_PALETTE];
    let mut assignments = [255u8; HEAD_DIM];

    for s in 0..N_PALETTE {
        let idx_compact = compact_alive(idx_sorted, &alive);
        if idx_compact.is_empty() {
            slot_fmt[s] = fallback_last;
            slot_scale[s] = 1.0;
            continue;
        }
        let stats = slot_stats(amax, &idx_compact);

        let mut best: Option<ScaleResult> = None;
        let mut fallback: Option<ScaleResult> = None;
        for &fmt in candidates {
            if let Some(b) = best {
                if b.pass_count >= SLOT_QUOTA {
                    break;
                }
            }
            let per_fmt = match search_scales_for_fmt(
                fmt,
                is_k,
                data,
                &idx_compact,
                stats,
                kthresh,
                inv_head_amax,
                inv_head_amax_sq,
                v_thr_sq,
                safe_head_amax,
            ) {
                Some(r) => r,
                None => continue,
            };
            if let Some(fb) = per_fmt.fallback {
                fallback = Some(match fallback {
                    None => fb,
                    Some(prev) => {
                        if fb.max_pass_metric < prev.max_pass_metric {
                            fb
                        } else {
                            prev
                        }
                    }
                });
            }
            if let Some(bw) = per_fmt.best {
                best = Some(match best {
                    None => bw,
                    Some(prev) => {
                        if bw.max_pass_metric < prev.max_pass_metric {
                            bw
                        } else {
                            prev
                        }
                    }
                });
            }
        }

        let winner = best.or(fallback);
        let (chosen_fmt, chosen_scale, pass_mask) = match winner {
            Some(w) => (w.fmt, w.outer, w.pass_mask),
            None => (fallback_last, 1.0f32, 0u128),
        };
        slot_fmt[s] = chosen_fmt;
        slot_scale[s] = chosen_scale;

        // Pass 1: claim passing blocks in sort order. Pass 2: fill any
        // remaining quota from the front of the still-alive set. Matches
        // the kernel's claim_passing_blocks_from_mask + warp-0 fill.
        let mut claimed = 0usize;
        for &b in &idx_compact {
            if claimed >= SLOT_QUOTA {
                break;
            }
            let bit = (pass_mask >> b as u32) & 1u128;
            if bit != 0 {
                assignments[b as usize] = s as u8;
                alive[b as usize] = false;
                claimed += 1;
            }
        }
        if claimed < SLOT_QUOTA {
            for &b in &idx_compact {
                if claimed >= SLOT_QUOTA {
                    break;
                }
                if alive[b as usize] {
                    assignments[b as usize] = s as u8;
                    alive[b as usize] = false;
                    claimed += 1;
                }
            }
        }
    }

    for a in assignments.iter_mut() {
        if *a == 255 {
            *a = (N_PALETTE - 1) as u8;
        }
    }
    (slot_fmt, slot_scale, assignments)
}
