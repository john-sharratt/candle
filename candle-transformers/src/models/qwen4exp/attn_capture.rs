//! The capture at an attention layer's output.
//!
//! An attention output goes non-finite, the layer's input and the drain say it
//! was finite, and two explanations remain that a dump of the output cannot
//! tell apart:
//!
//! * **The stored K/V is bad.** Some chunk the sequence attends holds NaN or
//!   Inf — written that way, or overwritten since. Then the values are there to
//!   be found, and the chunk that holds them names the writer to look for.
//! * **The launch read something that is not the K/V.** A copy into the chunk
//!   that had not landed, a slot freed and re-tenanted under the read, a table
//!   that named the wrong ground. Then the K/V reads finite once the device is
//!   quiet, and rerunning the identical launch over the identical inputs comes
//!   out finite — the bytes changed between the two reads.
//!
//! Both questions need the failing pass's own state, so they are asked here,
//! after the fence and before the dump, rather than left to an offline replay
//! that would see only the output.

use std::fmt::Write as _;

use candle::cuda_backend::CudaDevice;
use candle::{DType, LiveTensor, Result};
use candle_nn::kv_cache::KvCache;

use crate::models::nan_capture::{armed_finding, capture};

/// How many bad blocks a sequence's note lists before it stops.
const BLOCKS_LISTED: usize = 16;

/// The sequence rows `out` holds, as `[rows, width]` host floats.
fn host_rows(t: &LiveTensor<'_>) -> Result<(Vec<f32>, usize)> {
    let width = *t.dims().last().unwrap_or(&1);
    let flat = t.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?;
    Ok((flat, width))
}

/// Which of `rows_per_seq`'s sequences own a non-finite row of `vals`.
fn bad_rows_per_seq(vals: &[f32], width: usize, rows_per_seq: &[usize]) -> Vec<usize> {
    let mut row = 0usize;
    rows_per_seq
        .iter()
        .map(|&n| {
            let bad = (row..row + n)
                .filter(|r| {
                    vals[r * width..(r + 1) * width]
                        .iter()
                        .any(|v| !v.is_finite())
                })
                .count();
            row += n;
            bad
        })
        .collect()
}

/// One sequence's stored K/V, read back as the kernels decode it.
struct KvAudit {
    len: usize,
    blocks: usize,
    k_bad_positions: usize,
    v_bad_positions: usize,
    first_bad: Option<usize>,
    /// `(block, usage)` of the first [`BLOCKS_LISTED`] blocks holding a bad
    /// position.
    bad_blocks: Vec<(usize, u32)>,
}

/// Positions of a `(1, heads, len, head_dim)` read whose row holds a
/// non-finite value in any head.
fn bad_positions(vals: &[f32], heads: usize, len: usize, head_dim: usize) -> Vec<bool> {
    let mut bad = vec![false; len];
    for h in 0..heads {
        for (pos, b) in bad.iter_mut().enumerate() {
            let at = (h * len + pos) * head_dim;
            if vals[at..at + head_dim].iter().any(|v| !v.is_finite()) {
                *b = true;
            }
        }
    }
    bad
}

fn audit_kv(cache: &KvCache) -> Result<Option<KvAudit>> {
    let kc = cache.k_cache();
    let (Some(backing), Some(batch)) = (kc.chunked_backing(), kc.chunked_batch_idx()) else {
        return Ok(None);
    };
    let usage = cache.block_usage();
    let len: usize = usage.iter().map(|&u| u as usize).sum();
    if len == 0 {
        return Ok(Some(KvAudit {
            len,
            blocks: usage.len(),
            k_bad_positions: 0,
            v_bad_positions: 0,
            first_bad: None,
            bad_blocks: Vec::new(),
        }));
    }
    let (k, v) = backing.read_contiguous(batch, 0, len)?;
    let (_, heads, _, head_dim) = k.dims4()?;
    let kb = bad_positions(
        &k.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?,
        heads,
        len,
        head_dim,
    );
    let vb = bad_positions(
        &v.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?,
        heads,
        len,
        head_dim,
    );
    let mut bad_blocks = Vec::new();
    let mut first_bad = None;
    let mut start = 0usize;
    for (blk, &u) in usage.iter().enumerate() {
        let end = start + u as usize;
        if let Some(p) = (start..end).find(|&p| kb[p] || vb[p]) {
            first_bad.get_or_insert(p);
            if bad_blocks.len() < BLOCKS_LISTED {
                bad_blocks.push((blk, u));
            }
        }
        start = end;
    }
    Ok(Some(KvAudit {
        len,
        blocks: usage.len(),
        k_bad_positions: kb.iter().filter(|&&b| b).count(),
        v_bad_positions: vb.iter().filter(|&&b| b).count(),
        first_bad,
        bad_blocks,
    }))
}

/// A checkpoint on an attention group's output.
///
/// Free until a drain has named `name`, like every checkpoint. Once armed and
/// bad, it records which sequences own the bad rows, audits each sequence's
/// stored K/V, reruns the identical launch through `rerun`, and dumps all of it
/// with a verdict before stopping the process. The rerun's side effects — the
/// group's K/V append, at the positions the first launch already wrote — do not
/// matter: the process does not continue past the dump.
///
/// `rows_per_seq` is how many of `out`'s rows each cache's sequence owns, in
/// `caches` order.
pub(super) fn attention_checkpoint<'r>(
    name: &'static str,
    out: &LiveTensor<'_>,
    x: &LiveTensor<'_>,
    rows_per_seq: &[usize],
    caches: &mut [&mut KvCache],
    rerun: impl FnOnce(&mut [&mut KvCache]) -> Result<LiveTensor<'r>>,
    dev: &CudaDevice,
) -> Result<()> {
    let Some(f) = armed_finding(name, out) else {
        return Ok(());
    };
    let mut notes: Vec<(String, String)> = Vec::new();

    let (out_vals, width) = host_rows(out)?;
    let out_bad = bad_rows_per_seq(&out_vals, width, rows_per_seq);
    let (x_vals, x_width) = host_rows(x)?;
    let x_bad = bad_rows_per_seq(&x_vals, x_width, rows_per_seq);

    // Closest in time to the launch that failed, so it goes first.
    let again = rerun(caches)?;
    let (again_vals, _) = host_rows(&again)?;
    let again_bad = bad_rows_per_seq(&again_vals, width, rows_per_seq);
    let identical = again_vals
        .iter()
        .zip(&out_vals)
        .all(|(a, b)| a.to_bits() == b.to_bits());

    let mut kv_bad_total = 0usize;
    for (i, cache) in caches.iter().enumerate() {
        notes.push((format!("seq.{i:03}.rows"), rows_per_seq[i].to_string()));
        notes.push((format!("seq.{i:03}.out_bad_rows"), out_bad[i].to_string()));
        notes.push((format!("seq.{i:03}.x_bad_rows"), x_bad[i].to_string()));
        notes.push((
            format!("seq.{i:03}.rerun_bad_rows"),
            again_bad[i].to_string(),
        ));
        match audit_kv(cache)? {
            None => notes.push((format!("seq.{i:03}.kv"), "not chunked".into())),
            Some(a) => {
                kv_bad_total += a.k_bad_positions + a.v_bad_positions;
                let mut blocks = String::new();
                for (b, u) in &a.bad_blocks {
                    let _ = write!(blocks, "{b}:{u} ");
                }
                notes.push((
                    format!("seq.{i:03}.kv"),
                    format!(
                        "len={} blocks={} k_bad_pos={} v_bad_pos={} first_bad={:?} \
                         bad_blocks(block:usage)=[{}]",
                        a.len,
                        a.blocks,
                        a.k_bad_positions,
                        a.v_bad_positions,
                        a.first_bad,
                        blocks.trim_end()
                    ),
                ));
            }
        }
    }

    let verdict = if kv_bad_total > 0 {
        "STORED K/V IS NON-FINITE: a chunk this group attends holds NaN/Inf now — the \
         blocks are listed per sequence; find what wrote them"
    } else if again_bad.iter().all(|&n| n == 0) {
        "THE BYTES CHANGED UNDER THE LAUNCH: the stored K/V is finite and the identical \
         launch rerun over the identical inputs is finite — the first launch read memory \
         that was not yet, or no longer, this K/V (an unlanded copy, a slot re-tenanted \
         under the read, a table naming the wrong ground)"
    } else {
        "DETERMINISTIC: the stored K/V is finite yet the rerun is non-finite again — the \
         launch addresses something that is not this K/V (a selection or a table), or \
         its own inputs are bad"
    };
    notes.push(("verdict".into(), verdict.into()));
    notes.push(("rerun_bit_identical".into(), identical.to_string()));
    tracing::error!(
        target: "candle_transformers::nan_capture",
        site = name,
        kv_bad_total,
        rerun_identical = identical,
        out_bad = ?out_bad,
        rerun_bad = ?again_bad,
        x_bad = ?x_bad,
        verdict,
        "attention capture"
    );
    capture(name, &f, out, &[("x", x), ("rerun", &again)], &notes, dev)
}

#[cfg(test)]
mod tests {
    use super::{bad_positions, bad_rows_per_seq};

    #[test]
    fn a_bad_row_is_charged_to_the_sequence_that_owns_it() {
        // Three sequences of 2, 1 and 3 rows, width 2; rows 1 and 4 are bad.
        let mut v = vec![0.0f32; 12];
        v[3] = f32::NAN;
        v[9] = f32::INFINITY;
        assert_eq!(bad_rows_per_seq(&v, 2, &[2, 1, 3]), vec![1, 0, 1]);
    }

    #[test]
    fn a_position_is_bad_when_any_head_holds_a_non_finite_value() {
        // Two heads, three positions, head_dim 2, laid out (head, pos, dim).
        let at = |h: usize, pos: usize, d: usize| (h * 3 + pos) * 2 + d;
        let mut v = vec![1.0f32; 12];
        v[at(1, 1, 1)] = f32::NAN;
        v[at(0, 1, 0)] = f32::NEG_INFINITY;
        v[at(1, 2, 0)] = f32::NAN;
        assert_eq!(bad_positions(&v, 2, 3, 2), vec![false, true, true]);
    }
}
