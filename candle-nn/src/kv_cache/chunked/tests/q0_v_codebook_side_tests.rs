//! Q0_V's codebooks against real K and V.
//!
//! Q0_V is calibrated per side: the seal encodes a K band under the K
//! codebook and every read path decodes it under the K codebook, and the same
//! for V. That is only right if each side's codebook reconstructs that side's
//! data at least as well as the other side's does. These tests measure it on
//! the model dumps, with blocks built the way the seal builds them — one dim's
//! 32 tokens, normalised by the head's amax over the chunk.

use super::dump_reader::{load_dump, ChunkData};
use candle::quantized::k_quants::{decode_blocks_q0_v, encode_block_q0_v};

const QWEN3_DUMP: &str = "src/kv_cache/chunked/tests/data/qwen3-kv-data.bin";
const LLAMA_DUMP: &str = "src/kv_cache/chunked/tests/data/llama-kv-data.bin";
const TOKENS: usize = 32;

/// Squared reconstruction error of `block` through Q0_V under side `IS_K`.
fn q0_v_sse<const IS_K: bool>(block: &[f32; TOKENS]) -> f64 {
    let code = encode_block_q0_v::<IS_K>(block);
    let mut recon = [0f32; TOKENS];
    decode_blocks_q0_v::<IS_K>(std::slice::from_ref(&code), &mut recon);
    block
        .iter()
        .zip(&recon)
        .map(|(&x, &r)| ((x - r) as f64).powi(2))
        .sum()
}

/// Per codebook, the summed squared error over every (head, dim) block of one
/// side of the dump, and how many blocks each codebook wins outright.
struct SideScore {
    sse_k: f64,
    sse_v: f64,
    k_wins: usize,
    v_wins: usize,
    blocks: usize,
}

fn score_side(chunks: &[ChunkData], n_kv_head: usize, head_dim: usize, keys: bool) -> SideScore {
    let mut s = SideScore {
        sse_k: 0.0,
        sse_v: 0.0,
        k_wins: 0,
        v_wins: 0,
        blocks: 0,
    };
    for chunk in chunks {
        let data = if keys { &chunk.k } else { &chunk.v };
        for h in 0..n_kv_head {
            let head = &data[h * TOKENS * head_dim..(h + 1) * TOKENS * head_dim];
            let amax = head.iter().fold(0f32, |m, x| m.max(x.abs()));
            if amax < 1e-8 {
                continue;
            }
            let outer = 1.0 / amax;
            for d in 0..head_dim {
                let block: [f32; TOKENS] = std::array::from_fn(|t| head[t * head_dim + d] * outer);
                let (ek, ev) = (q0_v_sse::<true>(&block), q0_v_sse::<false>(&block));
                s.sse_k += ek;
                s.sse_v += ev;
                s.k_wins += usize::from(ek < ev);
                s.v_wins += usize::from(ev < ek);
                s.blocks += 1;
            }
        }
    }
    s
}

fn check_dump(rel: &str) {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join(rel);
    let (header, chunks) =
        load_dump(&path).unwrap_or_else(|| panic!("dump {rel} is present and loads"));
    assert_eq!(header.chunk_size, TOKENS);
    let (nh, hd) = (header.n_kv_head, header.head_dim);
    for (keys, side) in [(true, "K"), (false, "V")] {
        let s = score_side(&chunks, nh, hd, keys);
        let (own, other) = if keys {
            (s.sse_k, s.sse_v)
        } else {
            (s.sse_v, s.sse_k)
        };
        println!(
            "{rel} {side}: {} blocks; SSE under K codebook {:.4}, under V codebook {:.4}; \
             K wins {}, V wins {}",
            s.blocks, s.sse_k, s.sse_v, s.k_wins, s.v_wins
        );
        assert!(
            own <= other,
            "{rel}: {side} data reconstructs worse under its own codebook ({own:.4}) than \
             under the other side's ({other:.4})"
        );
    }
}

#[test]
fn each_sides_codebook_fits_qwen3_data_best() {
    check_dump(QWEN3_DUMP);
}

#[test]
fn each_sides_codebook_fits_llama_data_best() {
    check_dump(LLAMA_DUMP);
}
