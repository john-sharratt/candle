//! CPU golden reference for the prefill kernel tests: FP32/FP64 causal
//! attention over `[sealed prefix | new tokens]` with GQA and fused
//! non-interleaved RoPE, computed from the *pre-quantization* source values.
//!
//! The reference mirrors the kernel's exact conventions:
//! - K is stored unrotated (position-independent); rotation is applied at
//!   attention time at each token's absolute position, reading cos/sin from
//!   the same `rope_cs` table the kernel uses
//!   (`rope_cs[pos*HEAD_DIM + 2i] = cos`, `…+ 2i + 1] = sin`, pair `(i, i+H/2)`).
//! - Q rows rotate at `prefix_len + t`.
//! - Causal horizon: q row `t` attends columns `0 ..= prefix_len + t`.
//! - GQA: q head `h` reads kv head `h / (N_HEAD / N_KV_HEAD)`.
//! - V is never rotated. Softmax in f64, scale `1/sqrt(HEAD_DIM)`.
//!
//! Against a quantized prefix this is an *upper* reference — the kernel reads
//! the quantizer's arena bytes, so the acceptance band per scenario is the
//! quantization error budget of its compression level. Against an F16
//! identity prefix (`level: None`) the band is FP16-arithmetic-tight, which
//! is what validates the golden itself (RoPE convention included).

use crate::harness::{BuiltCase, HEAD_DIM, N_HEAD, N_KV_HEAD};

/// Rotate one head vector in place at `pos` (non-interleaved half-split).
fn rope_rotate(v: &mut [f32], pos: usize, rope_cs: &[f32]) {
    let half = HEAD_DIM / 2;
    let base = pos * HEAD_DIM;
    for i in 0..half {
        let cos = rope_cs[base + 2 * i];
        let sin = rope_cs[base + 2 * i + 1];
        let x = v[i];
        let y = v[i + half];
        v[i] = x * cos - y * sin;
        v[i + half] = x * sin + y * cos;
    }
}

/// Compute the golden output for every sequence of a built case,
/// concatenated in the same flat `[total_q, N_HEAD, HEAD_DIM]` row order the
/// kernel produces.
pub fn golden(case: &BuiltCase) -> Vec<f32> {
    let scale = 1.0 / (HEAD_DIM as f64).sqrt();
    let hpg = N_HEAD / N_KV_HEAD;
    let mut out = Vec::new();

    for (si, seq) in case.spec.seqs.iter().enumerate() {
        let prefix = seq.prefix_len();
        let kv_len = prefix + seq.q_len;

        // Assemble rotated K and raw V: [kv_len][N_KV_HEAD][HEAD_DIM].
        let mut k_rot = vec![0f32; kv_len * N_KV_HEAD * HEAD_DIM];
        let mut v_all = vec![0f32; kv_len * N_KV_HEAD * HEAD_DIM];
        for t in 0..kv_len {
            for h in 0..N_KV_HEAD {
                let dst = (t * N_KV_HEAD + h) * HEAD_DIM;
                let (src, src_t) = if t < prefix {
                    (&case.prefix_k[si], t)
                } else {
                    (&case.new_k[si], t - prefix)
                };
                let s = (src_t * N_KV_HEAD + h) * HEAD_DIM;
                k_rot[dst..dst + HEAD_DIM].copy_from_slice(&src[s..s + HEAD_DIM]);
                rope_rotate(&mut k_rot[dst..dst + HEAD_DIM], t, &case.rope_cs_host);

                let (srcv, srcv_t) = if t < prefix {
                    (&case.prefix_v[si], t)
                } else {
                    (&case.new_v[si], t - prefix)
                };
                let sv = (srcv_t * N_KV_HEAD + h) * HEAD_DIM;
                v_all[dst..dst + HEAD_DIM].copy_from_slice(&srcv[sv..sv + HEAD_DIM]);
            }
        }

        // Attention per q row per head.
        for t in 0..seq.q_len {
            let horizon = prefix + t + 1; // causal: cols 0..=prefix+t
            for h in 0..N_HEAD {
                let kvh = h / hpg;
                let mut q_row = [0f32; HEAD_DIM];
                let qs = (t * N_HEAD + h) * HEAD_DIM;
                q_row.copy_from_slice(&case.new_q[si][qs..qs + HEAD_DIM]);
                rope_rotate(&mut q_row, prefix + t, &case.rope_cs_host);

                // Scores + online-stable softmax in f64.
                let mut scores = vec![0f64; horizon];
                let mut m = f64::NEG_INFINITY;
                for (c, sc) in scores.iter_mut().enumerate() {
                    let ks = (c * N_KV_HEAD + kvh) * HEAD_DIM;
                    let mut dot = 0f64;
                    for d in 0..HEAD_DIM {
                        dot += q_row[d] as f64 * k_rot[ks + d] as f64;
                    }
                    *sc = dot * scale;
                    m = m.max(*sc);
                }
                let mut l = 0f64;
                for sc in scores.iter_mut() {
                    *sc = (*sc - m).exp();
                    l += *sc;
                }
                let mut acc = [0f64; HEAD_DIM];
                for (c, &p) in scores.iter().enumerate() {
                    let vs = (c * N_KV_HEAD + kvh) * HEAD_DIM;
                    for d in 0..HEAD_DIM {
                        acc[d] += p * v_all[vs + d] as f64;
                    }
                }
                out.extend(acc.iter().map(|&x| (x / l) as f32));
            }
        }
    }
    out
}
