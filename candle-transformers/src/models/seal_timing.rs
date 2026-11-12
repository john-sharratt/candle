//! What one boundary seal cost: the selection and convert core that turns a
//! cohort's native K/V into its quantized form, timed where
//! [`BatchedInferenceSession::quantize_and_seal_sequences`] already fences it.
//!
//! `docs/kv_quantize_in_the_gap.md` §3.6 and §8 step 1: the gap quantize runs
//! this same core between forwards, so its device time per MiB of native K/V,
//! set against the rate decode produces K/V, says whether the kernels as they
//! are can meet §7's gate (forward time at least 95% of each wave window).
//!
//! [`BatchedInferenceSession::quantize_and_seal_sequences`]:
//!     crate::models::batched_inference::BatchedInferenceSession::quantize_and_seal_sequences

/// Bytes of native K/V one convert descriptor covers: one 32-token chunk of
/// one KV head, K and V, at 16 bits an element.
pub fn native_bytes_per_desc(head_dim: usize) -> u64 {
    (32 * head_dim * 2 * 2) as u64
}

/// One boundary seal's cost.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct SealTiming {
    /// Native K/V the convert read: one descriptor per (chunk, head).
    pub native_bytes: u64,
    /// The format selection across every layer, its host readback included.
    pub select_ms: u64,
    /// Destination allocation on the host.
    pub alloc_ms: u64,
    /// The batched convert, from its launch to the fence that retires it.
    pub convert_ms: f64,
    /// The whole call, the swap back into the sequences included.
    pub total_ms: f64,
}

impl SealTiming {
    /// Native K/V converted, in MiB.
    pub fn native_mib(&self) -> f64 {
        self.native_bytes as f64 / (1024.0 * 1024.0)
    }

    /// The selection and the convert per MiB of native K/V: the work the gap
    /// moves between forwards. `None` when nothing was converted.
    pub fn core_ms_per_mib(&self) -> Option<f64> {
        let mib = self.native_mib();
        (mib > 0.0).then(|| (self.select_ms as f64 + self.convert_ms) / mib)
    }

    /// The share of decode time the gap would spend on what a decode producing
    /// `mib_per_s` of native K/V makes, at this seal's cost per MiB. `None`
    /// when nothing was converted.
    pub fn decode_share(&self, mib_per_s: f64) -> Option<f64> {
        self.core_ms_per_mib().map(|ms| ms * mib_per_s / 1000.0)
    }

    /// The `seal core` line a gate prints for one boundary: `label` names the
    /// row, `boundary` the phase whose K/V was sealed, and `decode_mib_per_s`,
    /// at the decode boundary, the rate that decode produced native K/V.
    pub fn line(&self, label: &str, boundary: &str, decode_mib_per_s: Option<f64>) -> String {
        let mut line = format!(
            "[{label}] seal core ({boundary}): {:.1} MiB native, select {} ms (readback \
             included), convert {:.1} ms, alloc {} ms, total {:.1} ms",
            self.native_mib(),
            self.select_ms,
            self.convert_ms,
            self.alloc_ms,
            self.total_ms,
        );
        if let Some(ms) = self.core_ms_per_mib() {
            line.push_str(&format!(" | {ms:.2} ms/MiB"));
        }
        if let Some((rate, share)) =
            decode_mib_per_s.and_then(|r| self.decode_share(r).map(|s| (r, s)))
        {
            line.push_str(&format!(
                " | decode made {rate:.2} MiB/s, so the gap would take {:.2}% of decode",
                share * 100.0
            ));
        }
        line
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_descriptor_is_one_chunk_of_one_head_k_and_v_at_16_bits() {
        assert_eq!(native_bytes_per_desc(256), 32 * 256 * 4);
        assert_eq!(native_bytes_per_desc(128), 16_384);
    }

    #[test]
    fn the_core_cost_is_selection_plus_convert_per_mib() {
        let t = SealTiming {
            native_bytes: 4 * 1024 * 1024,
            select_ms: 6,
            alloc_ms: 100,
            convert_ms: 2.0,
            total_ms: 200.0,
        };
        assert_eq!(t.native_mib(), 4.0);
        assert_eq!(t.core_ms_per_mib(), Some(2.0));
        // 2 ms a MiB against 10 MiB/s of decode K/V is 2% of decode.
        assert_eq!(t.decode_share(10.0), Some(0.02));
    }

    #[test]
    fn the_decode_line_names_the_rate_and_the_share() {
        let t = SealTiming {
            native_bytes: 4 * 1024 * 1024,
            select_ms: 6,
            alloc_ms: 1,
            convert_ms: 2.0,
            total_ms: 9.5,
        };
        assert_eq!(
            t.line("C5 x8", "decode", Some(10.0)),
            "[C5 x8] seal core (decode): 4.0 MiB native, select 6 ms (readback included), \
             convert 2.0 ms, alloc 1 ms, total 9.5 ms | 2.00 ms/MiB | decode made 10.00 \
             MiB/s, so the gap would take 2.00% of decode"
        );
        assert_eq!(
            SealTiming::default().line("C5 x8", "prefill", None),
            "[C5 x8] seal core (prefill): 0.0 MiB native, select 0 ms (readback included), \
             convert 0.0 ms, alloc 0 ms, total 0.0 ms"
        );
    }

    #[test]
    fn nothing_converted_has_no_rate() {
        let t = SealTiming::default();
        assert_eq!(t.core_ms_per_mib(), None);
        assert_eq!(t.decode_share(10.0), None);
    }
}
