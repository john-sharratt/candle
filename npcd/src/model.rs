//! Which model this daemon runs — the single place npcd decides.
//!
//! The counterpart of `zend/src/model_choice.rs`, and the same decision: cards
//! with at least 24 GiB run Qwen3-30B-A3B at Q6_K (~25 GB, expert-LRU paged),
//! and anything smaller runs Q4_K_M (~17 GB). One place, so the console, the
//! downloader and the engine cannot disagree about what is being run.
//!
//! # Why this does not reuse zend's
//!
//! zend's version calls `candle::quantized::get_total_vram_device0`, which
//! means depending on `candle` — a CUDA toolchain and a twenty-minute build, in
//! order to answer a question this daemon can already answer from the NVML
//! handle it holds for `/v1/telemetry`. The *threshold* is what matters and it
//! is duplicated deliberately, with the constant named the same on both sides;
//! the probe underneath it is whichever one is cheap in that binary.
//!
//! # Selected, not loaded
//!
//! Nothing here loads anything. This says which model the daemon *would* run on
//! the card it can see, which is a real and useful fact — it is what an operator
//! needs before starting a run, and it is decided by hardware they can check.
//! The console labels it as a selection until an engine reports, for the same
//! reason nothing else on that page states more than it measured.

/// A model this daemon knows how to run.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize)]
pub struct ModelSpec {
    /// Family and shape, as a person says it: `Qwen3-30B-A3B`.
    pub name: &'static str,
    pub quant: &'static str,
    /// Total and active parameters. A mixture-of-experts model is misread from
    /// either number alone — 30B sets the memory expectation, 3B sets the
    /// speed one, and the gap between them is the whole point of the
    /// architecture.
    pub params_total: &'static str,
    pub params_active: &'static str,
    pub repo: &'static str,
    pub filename: &'static str,
    /// On-disk size of the GGUF, exact. Pinned rather than measured so the
    /// figure is right before the file has been fetched.
    pub bytes: u64,
}

/// The 24 GiB tier. Q6_K needs ~25 GB resident but the expert LRU pages the
/// overflow, so a 24 GiB card runs it comfortably; a 16 GiB card cannot.
///
/// Named to match `Q6_MIN_TOTAL_VRAM_BYTES` in `zend/src/model_choice.rs`,
/// because the two are one decision and a reader who finds either should be
/// able to find the other.
pub const Q6_MIN_TOTAL_VRAM_BYTES: u64 = 24 * 1024 * 1024 * 1024;

pub const QWEN3_30B_A3B_Q6: ModelSpec = ModelSpec {
    name: "Qwen3-30B-A3B",
    quant: "Q6_K",
    params_total: "30B",
    params_active: "3B",
    repo: "unsloth/Qwen3-30B-A3B-GGUF",
    filename: "Qwen3-30B-A3B-Q6_K.gguf",
    bytes: 25_092_532_800,
};

pub const QWEN3_30B_A3B_Q4: ModelSpec = ModelSpec {
    quant: "Q4_K_M",
    filename: "Qwen3-30B-A3B-Q4_K_M.gguf",
    bytes: 18_556_686_912,
    ..QWEN3_30B_A3B_Q6
};

/// The smaller quant must actually be smaller, or [`choose`] selects the wrong
/// way round. Enforced at compile time rather than by a test, so an edit to
/// either constant cannot break it even in a build nobody runs the tests for.
const _: () = assert!(QWEN3_30B_A3B_Q4.bytes < QWEN3_30B_A3B_Q6.bytes);

/// Which quant fits this much card memory.
///
/// `None` — no card, or a failed probe — picks Q4_K_M: the smaller quant is the
/// one that runs everywhere, so an unknown card gets the answer that is least
/// likely to be wrong.
pub fn choose(total_vram_bytes: Option<u64>) -> ModelSpec {
    match total_vram_bytes {
        Some(total) if total >= Q6_MIN_TOTAL_VRAM_BYTES => QWEN3_30B_A3B_Q6,
        _ => QWEN3_30B_A3B_Q4,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const GIB: u64 = 1024 * 1024 * 1024;

    #[test]
    fn an_unknown_card_gets_the_quant_that_runs_everywhere() {
        assert_eq!(choose(None), QWEN3_30B_A3B_Q4);
    }

    #[test]
    fn the_sixteen_gib_tier_runs_q4() {
        assert_eq!(choose(Some(16 * GIB)), QWEN3_30B_A3B_Q4);
    }

    /// The 3090 and the Blackwell card. This is the boundary the whole module
    /// exists to place, so it is asserted on both sides of the exact byte.
    #[test]
    fn the_twenty_four_gib_tier_runs_q6() {
        assert_eq!(choose(Some(24 * GIB)), QWEN3_30B_A3B_Q6);
        assert_eq!(choose(Some(72 * GIB)), QWEN3_30B_A3B_Q6);
        assert_eq!(choose(Some(Q6_MIN_TOTAL_VRAM_BYTES)), QWEN3_30B_A3B_Q6);
        assert_eq!(choose(Some(Q6_MIN_TOTAL_VRAM_BYTES - 1)), QWEN3_30B_A3B_Q4);
    }

    /// The quants differ only where they should. A struct-update literal makes
    /// it easy to change one field and silently change the family with it.
    #[test]
    fn the_two_quants_are_the_same_model() {
        assert_eq!(QWEN3_30B_A3B_Q4.name, QWEN3_30B_A3B_Q6.name);
        assert_eq!(QWEN3_30B_A3B_Q4.repo, QWEN3_30B_A3B_Q6.repo);
        assert_eq!(
            QWEN3_30B_A3B_Q4.params_active,
            QWEN3_30B_A3B_Q6.params_active
        );
        assert_ne!(QWEN3_30B_A3B_Q4.quant, QWEN3_30B_A3B_Q6.quant);
        assert_ne!(QWEN3_30B_A3B_Q4.filename, QWEN3_30B_A3B_Q6.filename);
        // Their relative size is pinned at compile time — see the `const _`
        // above the `choose` function.
    }

    /// The filename has to name the quant it carries — a mismatch here fetches
    /// the wrong weights, and does it silently.
    #[test]
    fn each_filename_names_its_own_quant() {
        for m in [QWEN3_30B_A3B_Q4, QWEN3_30B_A3B_Q6] {
            assert!(
                m.filename.contains(m.quant),
                "{} does not name quant {}",
                m.filename,
                m.quant
            );
            assert!(m.filename.ends_with(".gguf"));
        }
    }
}
