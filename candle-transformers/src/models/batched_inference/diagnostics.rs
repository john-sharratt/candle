//! Formats a sequence's palette and format layout for debugging the paged KV
//! cache.
//!
//! The grid views report each chunk's per-head, per-palette arena format, so a
//! chunk holding mixed formats shows active per-palette routing. Band formats are
//! read from the chunk's own tags, since a shared size-class region would
//! otherwise report every band the same. Palette deviation counts how many of a
//! head's dimensions depart from the identity assignment. The label helpers give
//! a full name for tallies and a three-character name for the grid.

use super::*;

impl BatchedInferenceSession {
    /// Print a compact per-chunk palette4 format distribution for layer 0 of the given sequence.
    ///
    /// For each chunk, tallies how many (head × palette) K and V GID slots use each
    /// arena format. A chunk with mixed formats indicates active per-palette routing.
    #[cfg(feature = "verbose")]
    pub fn print_palette4_stats(&self, seq_idx: usize) {
        use candle_nn::kv_cache::{SampleFormat, N_PALETTE};

        let backing = match self.backings.first() {
            Some(b) => b,
            None => {
                println!("[pal4] seq={seq_idx}: no backings");
                return;
            }
        };

        let chunks = match backing.live_chunks_as_sealed(seq_idx) {
            Some(c) => c,
            None => {
                println!("[pal4] seq={seq_idx}: not allocated");
                return;
            }
        };
        if chunks.is_empty() {
            println!("[pal4] seq={seq_idx}: no chunks");
            return;
        }

        let n_kv_head = backing.n_kv_head();
        let head_dim = backing.head_dim();

        // Band formats come from the chunk's own tags. A grid built from arena
        // state would report every band of a shared size-class region
        // identically, which is exactly the mixed-format variety this grid
        // exists to display. Kept strongly typed — string conversion happens
        // only where a label is needed (tally display or grid short-label).
        let band_fmt = |tags: &[u8], h: usize, p: usize| -> Option<KvFormat> {
            tags.get(h * N_PALETTE + p)
                .copied()
                .and_then(KvFormat::from_tag)
        };
        let is_real_quant = |fmt: Option<KvFormat>| -> bool {
            matches!(fmt, Some(KvFormat::Quantized(qf)) if qf != QuantFormat::R16)
        };

        // Full label for tally/headline output (e.g. "Q8_0", "F16", "NULL").
        let fmt_long_label = |fmt: Option<KvFormat>| -> String {
            match fmt {
                None => "NULL".to_string(),
                Some(KvFormat::Float(dt)) => format!("{:?}", dt),
                Some(KvFormat::Quantized(qf)) => format!("{:?}", qf),
            }
        };

        // 3-char grid label. Delegates to `SampleFormat::grid_label` for the
        // formats it knows about and falls back to the 3-char prefix for any
        // others (R16, exotic dtypes, etc.).
        let fmt_grid_label = |fmt: Option<KvFormat>| -> String {
            match fmt {
                None => "NUL".to_string(),
                Some(KvFormat::Quantized(QuantFormat::R16)) => "R16".to_string(),
                Some(kv) => SampleFormat::from_kv_format(kv)
                    .map(|sf| sf.grid_label().to_string())
                    .unwrap_or_else(|| fmt_long_label(Some(kv)).chars().take(3).collect()),
            }
        };

        // Compact display for a tally: "Q8_0" if uniform, or "F16×4 Q8_0×28" if mixed.
        let tally_desc = |counts: &std::collections::HashMap<Option<KvFormat>, usize>| -> String {
            if counts.len() == 1 {
                fmt_long_label(*counts.keys().next().unwrap())
            } else {
                let mut parts: Vec<(String, usize)> = counts
                    .iter()
                    .map(|(k, v)| (fmt_long_label(*k), *v))
                    .collect();
                parts.sort_by(|a, b| a.0.cmp(&b.0));
                parts
                    .iter()
                    .map(|(k, v)| format!("{k}×{v}"))
                    .collect::<Vec<_>>()
                    .join(" ")
            }
        };

        // Palette deviation: for one head's pal slice, count how many of the head_dim
        // per-dimension 2-bit palette assignments differ from the identity pattern
        // (identity: dim d → palette d/sub_hd).  Returns 0 for an exact identity map
        // and a positive value for any non-identity layout (including balanced shuffles
        // like d%N_PALETTE which have equal per-palette counts but different dim routing).
        let pal_bytes_per_head = (head_dim / 4).max(1);
        let sub_hd = (head_dim / N_PALETTE).max(1);
        let pal_head_deviation = |pal_data: &[u8], h: usize| -> u64 {
            if pal_data.is_empty() {
                return 0; // empty = shared identity palette, deviation = 0
            }
            let base = h * pal_bytes_per_head;
            let end = (base + pal_bytes_per_head).min(pal_data.len());
            let slice = &pal_data[base..end];
            let mut diff_count: u64 = 0;
            for d in 0..head_dim {
                let byte_idx = d / 4;
                let actual_p = if byte_idx < slice.len() {
                    ((slice[byte_idx] >> ((d % 4) * 2)) & 0x3) as usize
                } else {
                    (d / sub_hd).min(N_PALETTE - 1)
                };
                let identity_p = (d / sub_hd).min(N_PALETTE - 1);
                if actual_p != identity_p {
                    diff_count += 1;
                }
            }
            diff_count
        };

        println!(
            "[pal4] seq={seq_idx}  n_chunks={}  n_kv_head={n_kv_head}  N_PALETTE={N_PALETTE}",
            chunks.len()
        );

        // Chunk selection for the dim grid: last chunk that uses a real quant format.
        // R16 is intentionally excluded here.
        let mut last_q_ci: usize = chunks.len().saturating_sub(1);
        let mut last_q_k_pct: u32 = 100;
        let mut last_q_v_pct: u32 = 100;
        let mut last_q_k_grid: Vec<Vec<String>> = Vec::new();
        let mut last_q_v_grid: Vec<Vec<String>> = Vec::new();
        let mut last_q_k_pal_data: Vec<u8> = Vec::new();
        let mut last_q_v_pal_data: Vec<u8> = Vec::new();
        let mut last_q_k_scales: Vec<f32> = Vec::new();
        let mut last_q_v_scales: Vec<f32> = Vec::new();

        for (ci, chunk) in chunks.iter().enumerate() {
            let mut k_counts: std::collections::HashMap<Option<KvFormat>, usize> =
                std::collections::HashMap::new();
            let mut v_counts: std::collections::HashMap<Option<KvFormat>, usize> =
                std::collections::HashMap::new();
            let mut hd_k: Vec<Vec<String>> = Vec::with_capacity(n_kv_head);
            let mut hd_v: Vec<Vec<String>> = Vec::with_capacity(n_kv_head);
            let mut has_quant = false;
            for h in 0..n_kv_head {
                let mut hk_row = Vec::with_capacity(N_PALETTE);
                let mut hv_row = Vec::with_capacity(N_PALETTE);
                for p in 0..N_PALETTE {
                    let kf = band_fmt(&chunk.k_fmt, h, p);
                    let vf = band_fmt(&chunk.v_fmt, h, p);
                    has_quant |= is_real_quant(kf) || is_real_quant(vf);
                    hk_row.push(fmt_grid_label(kf));
                    hv_row.push(fmt_grid_label(vf));
                    *k_counts.entry(kf).or_insert(0) += 1;
                    *v_counts.entry(vf).or_insert(0) += 1;
                }
                hd_k.push(hk_row);
                hd_v.push(hv_row);
            }

            let k_nonid = (0..n_kv_head)
                .filter(|&h| pal_head_deviation(&chunk.k_pal, h) > 0)
                .count();
            let v_nonid = (0..n_kv_head)
                .filter(|&h| pal_head_deviation(&chunk.v_pal, h) > 0)
                .count();
            // k_pct = % of heads with identity palette (100 = all identity).
            let k_pct = 100u32.saturating_sub((k_nonid * 100 / n_kv_head.max(1)) as u32);
            let v_pct = 100u32.saturating_sub((v_nonid * 100 / n_kv_head.max(1)) as u32);
            if has_quant {
                last_q_ci = ci;
                last_q_k_pct = k_pct;
                last_q_v_pct = v_pct;
                last_q_k_grid = hd_k;
                last_q_v_grid = hd_v;
                last_q_k_pal_data = chunk.k_pal.as_ref().clone();
                last_q_v_pal_data = chunk.v_pal.as_ref().clone();
                last_q_k_scales = chunk.k_scale.as_ref().clone();
                last_q_v_scales = chunk.v_scale.as_ref().clone();
            }
            let pct_tag = match (k_pct, v_pct) {
                (100, 100) => String::new(),
                (k, v) => format!("  K={k}% V={v}%"),
            };
            println!(
                "  blk[{ci:3}] toks={:2}  K: {:12}  V: {}{}",
                chunk.token_count,
                tally_desc(&k_counts),
                tally_desc(&v_counts),
                pct_tag
            );
        }

        // Grid view of the last quantized chunk in the sequence.
        if !chunks.is_empty() && !last_q_k_grid.is_empty() {
            let pal_bytes = pal_bytes_per_head;

            let pal_hdr: String = (0..N_PALETTE)
                .map(|p| format!("  p{p}"))
                .collect::<Vec<_>>()
                .join("");
            println!(
                "  grid blk[{last_q_ci:3}] K={last_q_k_pct}% V={last_q_v_pct}%  \
                 (K: {pal_hdr}  | V: {pal_hdr})"
            );
            for h in 0..n_kv_head {
                let k_row: String = last_q_k_grid[h]
                    .iter()
                    .map(|s| format!("{s:>3}"))
                    .collect::<Vec<_>>()
                    .join("  ");
                let v_row: String = last_q_v_grid[h]
                    .iter()
                    .map(|s| format!("{s:>3}"))
                    .collect::<Vec<_>>()
                    .join("  ");
                println!("  h{h:2}: {k_row}  |  {v_row}");
            }
            // Per-head K and V scales (1.0 = no post-dequant scaling).
            let scale_val = |scales: &[f32], h: usize, p: usize| -> f32 {
                scales.get(h * N_PALETTE + p).copied().unwrap_or(1.0)
            };
            for h in 0..n_kv_head {
                let k_sc: String = (0..N_PALETTE)
                    .map(|p| format!("{:.3}", scale_val(&last_q_k_scales, h, p)))
                    .collect::<Vec<_>>()
                    .join("  ");
                let v_sc: String = (0..N_PALETTE)
                    .map(|p| format!("{:.3}", scale_val(&last_q_v_scales, h, p)))
                    .collect::<Vec<_>>()
                    .join("  ");
                println!("  sc{h:2}: {k_sc}  |  {v_sc}");
            }

            // Per-dim palette grid: for K and V, find the head with the largest
            // deviation from the identity mapping and show which sub-band each of the
            // head_dim dims is assigned to. The key shows which format each sub-band uses.
            let print_dim_grid = |side: &str, pal_data: &[u8], hd_grid: &[Vec<String>]| {
                // Pick the head with the most non-identity palette byte assignment.
                let best_h = (0..n_kv_head)
                    .max_by_key(|&h| pal_head_deviation(pal_data, h))
                    .unwrap_or(0);

                // Key: show the format label for each sub-band index in this head.
                // p0=fmt0 p1=fmt1 p2=fmt2 p3=fmt3
                let key: String = (0..N_PALETTE)
                    .map(|p| {
                        let lbl = hd_grid
                            .get(best_h)
                            .and_then(|row| row.get(p))
                            .map(|s| s.as_str())
                            .unwrap_or("?");
                        format!("p{p}={lbl}")
                    })
                    .collect::<Vec<_>>()
                    .join(" ");
                println!("  pal {side}  blk[{last_q_ci:3}] h{best_h}  key: {key}");

                let base = best_h * pal_bytes;
                let end = (base + pal_bytes).min(pal_data.len());
                let slice: &[u8] = if base < end {
                    &pal_data[base..end]
                } else {
                    &[]
                };

                // Each cell shows the raw sub-band index (0/1/2/3) for that dim.
                // The key shows which format each sub-band uses.
                // For the identity palette this produces 32 of each value in order.
                let cols = 8usize;
                let grid_rows = head_dim.div_ceil(cols);
                for r in 0..grid_rows {
                    let row_str: String = (0..cols)
                        .map(|c| {
                            let d = r * cols + c;
                            if d < head_dim {
                                let byte_idx = d / 4;
                                let pal_idx = if byte_idx < slice.len() {
                                    ((slice[byte_idx] >> ((d % 4) * 2)) & 0x3) as usize
                                } else {
                                    (d / sub_hd).min(N_PALETTE - 1)
                                };
                                format!("{pal_idx}")
                            } else {
                                " ".to_string()
                            }
                        })
                        .collect::<Vec<_>>()
                        .join(" ");
                    println!("    {row_str}");
                }
            };

            print_dim_grid("K", &last_q_k_pal_data, &last_q_k_grid);
            print_dim_grid("V", &last_q_v_pal_data, &last_q_v_grid);
        }
    }

    //#[cfg(feature = "verbose")]
    pub fn print_compression_distribution(&self, dist: &ahash::HashMap<GgmlDType, usize>) {
        println!("Compression Distribution (elements)");
        let mut dist = dist.iter().map(|(k, v)| (k, *v)).collect::<Vec<_>>();
        dist.sort_by_key(|(k, _)| k.type_size());
        for (format, cnt) in dist {
            println!("   {:?}: {}", format, cnt);
        }
    }
}
