//! KV-footprint accounting: what DeepSeek's bounded-window + compressed-corpus
//! attention holds resident vs what a traditional dense-attention FP16 model
//! would hold at the same depth (`docs/deepseek_batched_paged_attention_plan.md`
//! Part IV §H).
//!
//! The reported compression is a **system-level** ratio — bounded attention ×
//! FP8 window storage × corpus pooling — not a per-chunk bits-per-element
//! figure. The baseline is a traditional model: separate full-history K and V,
//! FP16, every token, every layer.

use candle::DType;

use super::config::Config;

/// Storage dtype for the raw sliding-window latents: one fixed format, no
/// adaptive per-block level (the C0–C9 candidate tables assume per-head
/// separate K/V and do not apply to the single 512-d latent).
pub const WINDOW_KV_DTYPE: DType = DType::F8E4M3;

/// Element dtype for the compressed-corpus pair (`[G, head_dim]` attended +
/// `[G, index_head_dim]` scoring) and the trailing partial-group `(m, l, acc)`
/// accumulators. The corpus's job is retrieval, not compression — it stays
/// float.
pub const CORPUS_DTYPE: DType = DType::F32;

/// Per-conversation KV bytes, broken down by where they live.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct KvFootprint {
    /// FP8 sliding-window latents, every layer: `min(n_tokens, window_size)`
    /// rows of `head_dim`.
    pub window_bytes: u64,
    /// Float corpus pair on compression layers (CSA/HCA): completed groups
    /// `G = n_tokens / ratio`, each one `head_dim` attended row + one
    /// `index_head_dim` scoring row.
    pub corpus_bytes: u64,
    /// Trailing partial-group `(m, l, acc)` tails on compression layers whose
    /// group is mid-fill (`n_tokens % ratio != 0`): three floats per channel,
    /// for both the attended (`head_dim`) and scoring (`index_head_dim`)
    /// accumulators.
    pub tail_bytes: u64,
}

impl KvFootprint {
    pub fn total(&self) -> u64 {
        self.window_bytes + self.corpus_bytes + self.tail_bytes
    }
}

/// Bytes a traditional dense-attention FP16 model holds in KV after
/// `n_tokens`: separate full-history K and V (×2), FP16 (2 B/elem), every
/// token, every layer.
pub fn fp16_linear_baseline_bytes(n_tokens: usize, cfg: &Config) -> u64 {
    n_tokens as u64
        * cfg.head_dim as u64
        * 2 // K and V
        * DType::F16.size_in_bytes() as u64
        * cfg.n_layers as u64
}

/// DeepSeek's resident KV bytes after `n_tokens`, summed **per layer kind**:
/// every layer holds the FP8 window; only compression layers (CSA/HCA,
/// `ratio > 0`) add the float corpus pair and, mid-group, the `(m, l, acc)`
/// tails. SWA layers hold the window alone.
pub fn deepseek_kv_footprint(n_tokens: usize, cfg: &Config) -> KvFootprint {
    let window_elem = WINDOW_KV_DTYPE.size_in_bytes() as u64;
    let corpus_elem = CORPUS_DTYPE.size_in_bytes() as u64;

    let mut window_bytes = 0u64;
    let mut corpus_bytes = 0u64;
    let mut tail_bytes = 0u64;

    for layer in 0..cfg.n_layers {
        window_bytes += n_tokens.min(cfg.window_size) as u64 * cfg.head_dim as u64 * window_elem;

        let ratio = cfg.compress_ratio(layer);
        if ratio == 0 {
            continue; // SWA: window only.
        }
        let groups = (n_tokens / ratio) as u64;
        corpus_bytes += groups * (cfg.head_dim as u64 + cfg.index_head_dim as u64) * corpus_elem;
        if !n_tokens.is_multiple_of(ratio) {
            // (m, l, acc) per channel, attended + scoring accumulators.
            tail_bytes += 3 * (cfg.head_dim as u64 + cfg.index_head_dim as u64) * corpus_elem;
        }
    }

    KvFootprint {
        window_bytes,
        corpus_bytes,
        tail_bytes,
    }
}

/// The reported system-level compression: `baseline / actual`. Values > 1 mean
/// DeepSeek holds less than a traditional FP16 dense-attention model would.
pub fn ratio_vs_fp16_linear(n_tokens: usize, cfg: &Config) -> f64 {
    let actual = deepseek_kv_footprint(n_tokens, cfg).total();
    if actual == 0 {
        return f64::INFINITY;
    }
    fp16_linear_baseline_bytes(n_tokens, cfg) as f64 / actual as f64
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Raw expected bytes for the tiny config (head_dim=64, index_head_dim=64,
    /// window=8, layers [SWA, SWA, r4, r128, r4, r128]) at n=1000 — asserted
    /// exactly, never by tolerance.
    #[test]
    fn footprint_raw_bytes_tiny_config() {
        let cfg = Config::tiny();
        let fp = deepseek_kv_footprint(1000, &cfg);

        // Window: 6 layers × min(1000, 8)=8 rows × 64 dims × 1 B (FP8).
        assert_eq!(fp.window_bytes, 6 * 8 * 64);
        // Corpus row = head_dim + index_head_dim = 128 dims × 4 B (f32) = 512 B.
        //   r4 layers (2, 4): G=250 → 250×512 = 128_000 each.
        //   r128 layers (3, 5): G=7 → 7×512 = 3_584 each.
        assert_eq!(fp.corpus_bytes, 2 * 128_000 + 2 * 3_584);
        // Tails: 1000 % 4 == 0 → r4 layers carry none. 1000 % 128 == 104 → the
        // two r128 layers each hold a 3-row (m, l, acc) tail = 3×512 = 1_536.
        assert_eq!(fp.tail_bytes, 2 * 1_536);
        assert_eq!(fp.total(), 3_072 + 263_168 + 3_072);
    }

    #[test]
    fn baseline_raw_bytes_tiny_config() {
        let cfg = Config::tiny();
        // 1000 tokens × 64 dims × 2 (K,V) × 2 B (F16) × 6 layers.
        assert_eq!(fp16_linear_baseline_bytes(1000, &cfg), 1_536_000);
    }

    /// The window term saturates at `window_size` while the corpus grows at
    /// `1/ratio` per compression layer — so the ratio grows monotonically with
    /// depth toward the exact asymptote
    /// `head_dim·2·2·n_layers / Σ_compress((head_dim+index_head_dim)·4/ratio)`.
    /// (Total corpus bytes are linear in N — the *resident* set stays bounded
    /// via tier spill, which is §L's concern, not this accounting's.)
    #[test]
    fn ratio_grows_toward_exact_asymptote() {
        let cfg = Config::tiny();
        let r1k = ratio_vs_fp16_linear(1_000, &cfg);
        let r100k = ratio_vs_fp16_linear(100_000, &cfg);
        assert!(r1k > 1.0, "ratio at 1k should already exceed 1, got {r1k}");
        assert!(
            r100k > r1k,
            "ratio must grow with depth: 1k={r1k}, 100k={r100k}"
        );
        // Tiny config: numerator 64·2·2·6 = 1536 B/token; denominator, with a
        // 128-dim corpus row at 4 B = 512 B/group,
        // 2×(512/4) + 2×(512/128) = 256 + 8 = 264 B/token → 1536/264.
        let asymptote = 1536.0 / 264.0;
        let r10m = ratio_vs_fp16_linear(10_000_000, &cfg);
        assert!(
            (r10m - asymptote).abs() < 0.01,
            "10M-token ratio {r10m} should sit at the asymptote {asymptote}"
        );
    }

    /// Zero tokens: nothing resident, baseline zero — ratio defined as ∞.
    #[test]
    fn empty_conversation_holds_nothing() {
        let cfg = Config::tiny();
        assert_eq!(deepseek_kv_footprint(0, &cfg).total(), 0);
        assert_eq!(fp16_linear_baseline_bytes(0, &cfg), 0);
        assert!(ratio_vs_fp16_linear(0, &cfg).is_infinite());
    }

    /// §L footprint-flat (attended set): the sliding-window term — the bytes
    /// each query actually attends over per layer — is CONSTANT once the depth
    /// reaches `window_size`. This is the O(1)-error bound expressed in
    /// storage: window bytes never grow with N, so at 1M tokens each query's
    /// attended window costs exactly what it costs at `window_size` tokens.
    /// (The corpus term is linear in N by construction — the RESIDENT corpus
    /// stays bounded via warm→cold tier spill, a separate runtime concern; the
    /// baseline FP16 linear model, by contrast, grows its ATTENDED set with N.)
    #[test]
    fn window_bytes_flat_beyond_window_size() {
        let cfg = Config::tiny();
        let w = cfg.window_size;
        // The per-layer window row count saturates at window_size.
        let saturated = deepseek_kv_footprint(w, &cfg).window_bytes;
        let one_layer_full =
            w as u64 * cfg.head_dim as u64 * WINDOW_KV_DTYPE.size_in_bytes() as u64;
        assert_eq!(saturated, cfg.n_layers as u64 * one_layer_full);
        // Every larger depth holds the SAME window bytes — flat, not growing.
        for &n in &[w + 1, w * 4, 10_000, 1_000_000, 1_000_000_000] {
            assert_eq!(
                deepseek_kv_footprint(n, &cfg).window_bytes,
                saturated,
                "window bytes must stay flat at N={n} (attended-set O(1) bound)"
            );
        }
        // The FP16 dense baseline's attended set, in contrast, grows linearly:
        // its footprint at 1M is 1M/window_size times its footprint at
        // window_size. The bound is what separates the two.
        assert!(
            fp16_linear_baseline_bytes(1_000_000, &cfg) > fp16_linear_baseline_bytes(w, &cfg) * 100,
            "dense baseline attended set must grow with depth"
        );
    }
}
