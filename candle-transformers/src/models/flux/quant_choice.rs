//! Which published FLUX GGUF a card can run.
//!
//! The single place the quantisation is decided, so the choice cannot drift
//! between the downloader that fetches a file and the loader that opens it —
//! the same reason `zend`'s model choice lives in one function.
//!
//! # Why FLUX needs a ladder where the language models do not
//!
//! A MoE language model's resident footprint is its dense weights plus whatever
//! expert working set fits, so a bigger card buys speed rather than
//! feasibility. FLUX is dense: every one of the 12B parameters is read on every
//! one of the fifty steps, so the whole transformer has to be resident and the
//! file size *is* the requirement. That makes the card's capacity a hard gate,
//! and the quant a real decision rather than a preference.
//!
//! # What else has to fit
//!
//! The transformer is the largest tenant but not the only one. A pass needs the
//! T5-v1.1-XXL text encoder (~9.5 GB at bf16) and the autoencoder (~0.3 GB) as
//! well — though not *at the same time*: the encoders run first and are dropped
//! before the transformer is built, so the peak is whichever of those two
//! phases is larger, not their sum. The ladder is sized against the transformer
//! phase, which is the one that grows with the quant.

use candle::quantized::GgmlDType;

/// A published quantisation of FLUX.1-dev.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FluxQuant {
    /// ~6.8 GB. The rung that fits a 16 GB card alongside the autoencoder and
    /// leaves room for a 1024×1024 pass's activations.
    Q4KS,
    /// ~12.7 GB. Near-lossless against the F16 release, and what a 24 GB card
    /// should run — there is no quality argument for Q4 when Q8 fits.
    Q8_0,
}

/// Below this much total VRAM, the Q4 rung.
///
/// Between the two file sizes rather than at either: Q8's 12.7 GB plus the
/// autoencoder and a 1024×1024 pass's activations does not leave a 16 GB card
/// enough headroom, and a card that clears 18 GB has room for Q8 with the same
/// margin Q4 has on 16.
pub const Q8_MIN_TOTAL_VRAM_BYTES: u64 = 18 * (1 << 30);

impl FluxQuant {
    /// The rung for a card with `total_vram_bytes` of memory.
    ///
    /// Total rather than free: the decision is which checkpoint this deployment
    /// runs, and free memory is a reading of the moment that would have a
    /// daemon pick a different model depending on what happened to be resident
    /// when it started.
    pub fn for_vram(total_vram_bytes: u64) -> Self {
        if total_vram_bytes >= Q8_MIN_TOTAL_VRAM_BYTES {
            Self::Q8_0
        } else {
            Self::Q4KS
        }
    }

    /// The repository publishing it.
    ///
    /// Not `black-forest-labs/FLUX.1-dev`: that repo is gated and ships
    /// safetensors, and these are the community GGUF conversions of it. The
    /// weights are the same release; the licence terms of the original still
    /// apply to what is generated with them.
    pub fn repo(&self) -> &'static str {
        "city96/FLUX.1-dev-gguf"
    }

    pub fn filename(&self) -> &'static str {
        match self {
            Self::Q4KS => "flux1-dev-Q4_K_S.gguf",
            Self::Q8_0 => "flux1-dev-Q8_0.gguf",
        }
    }

    /// The published file's exact length, for a downloader's progress total
    /// when the server omits `Content-Length`.
    pub fn bytes(&self) -> u64 {
        match self {
            Self::Q4KS => 6_805_988_640,
            Self::Q8_0 => 12_708_281_504,
        }
    }

    /// The dominant block type in the file, for reporting.
    pub fn dtype(&self) -> GgmlDType {
        match self {
            Self::Q4KS => GgmlDType::Q4_K,
            Self::Q8_0 => GgmlDType::Q8_0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The threshold, from both sides and exactly on it.
    #[test]
    fn the_ladder_turns_at_eighteen_gigabytes() {
        let gb = |n: u64| n * (1 << 30);
        // The fleet's three cards, by the numbers in CLAUDE.md.
        assert_eq!(FluxQuant::for_vram(gb(16)), FluxQuant::Q4KS, "4090 Mobile");
        assert_eq!(FluxQuant::for_vram(gb(24)), FluxQuant::Q8_0, "3090");
        assert_eq!(FluxQuant::for_vram(gb(72)), FluxQuant::Q8_0, "PRO 5000");

        assert_eq!(
            FluxQuant::for_vram(Q8_MIN_TOTAL_VRAM_BYTES),
            FluxQuant::Q8_0
        );
        assert_eq!(
            FluxQuant::for_vram(Q8_MIN_TOTAL_VRAM_BYTES - 1),
            FluxQuant::Q4KS
        );
    }

    /// **A card that cannot hold the file must not be handed it**, and holding
    /// the file is not enough on its own: the autoencoder and a 1024×1024
    /// pass's activations sit beside it. The margin is what makes the threshold
    /// a threshold rather than a restatement of the file size.
    #[test]
    fn the_threshold_leaves_room_for_the_rest_of_the_pass() {
        // The autoencoder plus a 1024×1024 pass — 4096 latent tokens and 512
        // text tokens through 19 double and 38 single blocks.
        const REST_OF_THE_PASS: u64 = 4 * (1 << 30);
        for (q, floor) in [
            // The smallest card in the fleet that takes each rung.
            (FluxQuant::Q4KS, 16 * (1u64 << 30)),
            (FluxQuant::Q8_0, Q8_MIN_TOTAL_VRAM_BYTES),
        ] {
            assert_eq!(FluxQuant::for_vram(floor), q, "the ladder moved");
            assert!(
                q.bytes() + REST_OF_THE_PASS < floor,
                "{q:?} is {:.1} GiB and its floor is {:.1} GiB — no room for the autoencoder and \
                 a pass beside it",
                q.bytes() as f64 / (1u64 << 30) as f64,
                floor as f64 / (1u64 << 30) as f64,
            );
        }
    }

    /// Every rung names a file that exists in the repo. A typo here is a 404
    /// after a deployment has already decided to use it.
    #[test]
    fn every_rung_names_a_published_file() {
        for q in [FluxQuant::Q4KS, FluxQuant::Q8_0] {
            assert!(q.filename().starts_with("flux1-dev-"));
            assert!(q.filename().ends_with(".gguf"));
            assert!(q.bytes() > 1 << 30);
        }
        assert_ne!(FluxQuant::Q4KS.filename(), FluxQuant::Q8_0.filename());
    }
}
