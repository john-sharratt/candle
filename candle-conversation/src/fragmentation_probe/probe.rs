//! What to run, and what the run concluded.
//!
//! [`Probe`] is the whole input: a [`ModelProfile`] plus the timings, which are the
//! only knobs a caller normally touches. It carries no argument parser — the driver
//! that has one converts into this, so a test case and a command line reach the same
//! function by the same route.

use super::batch::BatchTiming;
use super::profile::{ModelProfile, StoryGate};
use candle_transformers::models::batch_test::utils::ExtraRow;

/// One probe run's configuration.
#[derive(Clone, Debug)]
pub struct Probe {
    /// CUDA device ordinal.
    pub device: usize,
    /// The model, and everything that is a property of it — see [`ModelProfile`].
    pub profile: ModelProfile,
    /// Delay between starting conversations, in milliseconds. Lower is more overlap.
    pub stagger_ms: u64,
    /// Raise concurrency and tighten the stagger until the pool saturates.
    pub saturate: bool,
    /// Free regions at or below which the pool counts as saturated.
    pub saturated_free: usize,
    /// Seconds to hold the overlapping churn.
    pub churn_secs: u64,
    /// One retirement in this many is a **straggler**: held for
    /// [`Self::straggler_hold_secs`] before being evicted, instead of retiring with its
    /// burst.
    ///
    /// **This is what keeps the high end of the span alive.** The pinned residents
    /// cannot: they are created before the churn, so they sit at the *lowest* arena
    /// ranks and the frontier rises past them immediately. A straggler is claimed when
    /// the frontier is already high and then survives while everything around it frees
    /// — so it pins the watermark up there while the churn below it opens holes. Each
    /// burst leaves one a little higher than the last, and the frontier ratchets.
    ///
    /// Without this the allocator wins: its free list is lowest-index-first, so a hole
    /// is exactly what the next claim consumes and the live set stays contiguous from
    /// zero. Measured — holes peaked at 34 and settled back to 1.
    pub straggler_every: usize,
    /// How long a straggler holds its KV.
    pub straggler_hold_secs: u64,
    /// Seconds to drain after everything is evicted, watching the frontier fall and the
    /// weight zone grow.
    pub drain_secs: u64,
    /// Tokens each phase-B sequence decodes.
    pub batch_decode: usize,
    /// The engine's KV compression level, `C<n>` — the level every phase runs at.
    pub compression_level: u8,
    /// Sessions in the clean baseline batch, run on the fresh engine before any churn.
    ///
    /// **The baseline is the forward gate's `C5 ×8` row, through the engine.** Same
    /// prompts, same names, same compression level, same width, same token count, timed
    /// with the gate's two windows — so the gap between that gate row and this one is
    /// what the engine costs, and it is the number to optimise against.
    pub baseline_width: usize,
    /// Tokens each baseline session generates, the first included — the generate
    /// count of the gate row it is compared with (`kv_fragmentation`'s long C5 ×8
    /// row, run at this same count).
    ///
    /// **Long enough for decode to be measured, not sampled.** At the ladder's 10
    /// tokens the decode window is ~4 speculative steps, ~80 ms, so one compaction
    /// pass or one scheduling hiccup landing in it moved decode by 20%. At 64 the
    /// window is long enough that per-step cost is what it reports.
    pub baseline_decode: usize,
}

impl Probe {
    /// A run of `profile` with timings that have been measured to reach saturation on
    /// a 72 GiB card.
    ///
    /// Timings rather than sizes: the sizes live on the profile, because they are what
    /// varies with the model. A caller tuning a run adjusts the fields below.
    pub fn new(profile: ModelProfile) -> Self {
        Self {
            device: 0,
            profile,
            stagger_ms: 120,
            saturate: false,
            saturated_free: 24,
            churn_secs: 90,
            straggler_every: 4,
            straggler_hold_secs: 20,
            drain_secs: 40,
            batch_decode: 48,
            compression_level: 5,
            baseline_width: 8,
            baseline_decode: 64,
        }
    }

    /// A short run, for a smoke check rather than a measurement.
    ///
    /// Named rather than left to each caller to shorten, because the figures a short
    /// run produces are not comparable to a full one's: the churn does not reach the
    /// frontier, so its efficiency is the ramp's and not the steady state's. A caller
    /// that wants a quick "does this still work" uses this and reads the story gate.
    pub fn short(mut self) -> Self {
        self.churn_secs = 30;
        self.straggler_hold_secs = 10;
        self.drain_secs = 15;
        self
    }

    pub fn story_gate(&self) -> StoryGate {
        self.profile.story
    }

    /// The model builder this probe's profile describes.
    ///
    /// Sized from the profile, which is where everything model-specific lives — the
    /// context the engine is built for, and the widths the workload will ask of it. A
    /// caller that loads the model itself still takes the builder from here, so the
    /// engine config the two agree on is the same one.
    pub fn builder(&self) -> crate::models::ModelBuilder {
        self.profile
            .model
            .clone()
            .builder()
            .max_concurrent(self.profile.max_concurrency + self.profile.batch + 4)
            .max_seq_len(self.profile.max_seq_len)
            .compression_level(self.compression_level)
    }
}

/// The clean baseline batch: the gate's `C<level> ×width` row, through the engine.
#[derive(Clone, Copy, Debug)]
pub struct BaselineRow {
    pub level: u8,
    pub width: usize,
    pub timing: BatchTiming,
    pub story_pass: usize,
}

impl BaselineRow {
    /// This batch as a row for the batched comparison table, labelled with the gate
    /// row it reproduces.
    pub fn as_table_row(&self) -> ExtraRow {
        ExtraRow {
            label: format!("eng C{}×{}", self.level, self.width),
            contexts: self.width,
            prompt_tokens_per_sec: self.timing.prefill_tps(),
            generate_tokens_per_sec: self.timing.decode_tps(),
            valid: Some((self.story_pass, self.width)),
            compression_ratio: None,
            peak_tokens: self.timing.peak_tokens,
            frontier_regions: None,
            efficiency_pct: None,
        }
    }
}

/// What a probe run measured, and whether it passed.
///
/// Returned rather than panicked so a caller can report every gate before failing on
/// any: the three are the halves of one claim — pack the KV, let the weight side have
/// what packing released, and answer correctly throughout — and failing on the first
/// would hide the evidence for the others.
#[derive(Clone, Debug)]
pub struct ProbeOutcome {
    /// Sessions whose phase-B reply passed the profile's [`StoryGate`], out of the
    /// batch width.
    pub story_pass: usize,
    pub story_total: usize,
    /// The worst VRAM efficiency that persisted across two publishes and whose loss
    /// was large enough to charge, as a percentage. 100 when no sample qualified.
    pub worst_sustained_efficiency: usize,
    /// The worst single sample, whatever its size or persistence — reported so the
    /// floors on the judged figure hide nothing.
    pub worst_single_efficiency: usize,
    /// Share of the ground the KV side released that the weight side took.
    pub weight_uptake_pct: usize,
    /// The weight zone reached its limit in the drain — every slot the model has, or
    /// the most the span lets it hold — so released ground past that has no residency
    /// to buy. Read from the engine's growth ledger.
    pub weight_at_limit: bool,
    /// Phase-B throughput, for comparison against the forward gate's clean rows.
    pub prefill_tps: f64,
    pub decode_tps: f64,
    /// The arena frontier and the tokens live when phase B finished — the geometry the
    /// throughput above was delivered on, so a table row can state both together.
    pub frontier_regions: usize,
    pub efficiency_pct: usize,
    pub peak_tokens: usize,
    /// The clean baseline batch, measured on the fresh engine before phase A.
    pub baseline: BaselineRow,
    /// One line per failed gate, in the order the gates are stated.
    pub failures: Vec<String>,
}

impl ProbeOutcome {
    pub fn passed(&self) -> bool {
        self.failures.is_empty()
    }

    /// This run as a row for the batched comparison table, below the harness's own.
    ///
    /// `label` names what produced it — the shape, not the KV mode, since an engine row
    /// is one workload rather than one format.
    pub fn as_table_row(&self, label: impl Into<String>) -> ExtraRow {
        ExtraRow {
            label: label.into(),
            contexts: self.story_total,
            prompt_tokens_per_sec: self.prefill_tps,
            generate_tokens_per_sec: self.decode_tps,
            valid: Some((self.story_pass, self.story_total)),
            compression_ratio: None,
            peak_tokens: self.peak_tokens,
            frontier_regions: Some(self.frontier_regions),
            efficiency_pct: Some(self.efficiency_pct),
        }
    }

    /// The weight side's verdict as a reader should see it.
    ///
    /// A zone that reached its limit had nothing more for released ground to buy, so
    /// its uptake is the growth it had room for; the label says so rather than leave a
    /// bare low percentage reading as a failure of the gate that just passed.
    pub fn weight_uptake_label(&self) -> String {
        if self.weight_at_limit {
            format!(
                "weight zone at its limit (uptake {}%)",
                self.weight_uptake_pct
            )
        } else {
            format!("weight uptake {}%", self.weight_uptake_pct)
        }
    }

    /// Fail with every gate's verdict in the message, so a test's output says what
    /// held as well as what did not.
    pub fn assert_passed(&self) {
        assert!(
            self.passed(),
            "{} probe gate(s) failed:\n  {}",
            self.failures.len(),
            self.failures.join("\n  "),
        );
    }
}

#[cfg(test)]
mod tests {
    use super::{BaselineRow, BatchTiming, ProbeOutcome};

    fn outcome(weight_uptake_pct: usize, weight_at_limit: bool) -> ProbeOutcome {
        ProbeOutcome {
            story_pass: 20,
            story_total: 20,
            worst_sustained_efficiency: 99,
            worst_single_efficiency: 70,
            weight_uptake_pct,
            weight_at_limit,
            prefill_tps: 0.0,
            decode_tps: 0.0,
            frontier_regions: 0,
            efficiency_pct: 0,
            peak_tokens: 0,
            baseline: BaselineRow {
                level: 5,
                width: 8,
                timing: BatchTiming::from_sessions(&[]),
                story_pass: 8,
            },
            failures: Vec::new(),
        }
    }

    /// The baseline row names the gate row it reproduces and carries its two windows'
    /// rates, so it reads beside that row in the comparison table.
    #[test]
    fn the_baseline_row_is_labelled_with_the_gate_row_it_reproduces() {
        let row = BaselineRow {
            level: 5,
            width: 8,
            timing: BatchTiming {
                prefill_tokens: 8000,
                prefill_s: 2.0,
                decode_tokens: 72,
                decode_s: 0.5,
                complete_s: 0.7,
                streamed_tokens: 80,
                peak_tokens: 9123,
            },
            story_pass: 7,
        }
        .as_table_row();
        assert_eq!(row.label, "eng C5×8");
        assert_eq!(row.contexts, 8);
        assert_eq!(row.prompt_tokens_per_sec, 4000.0);
        assert_eq!(row.generate_tokens_per_sec, 144.0);
        assert_eq!(row.valid, Some((7, 8)));
        assert_eq!(row.peak_tokens, 9123, "every session's whole context");
    }

    #[test]
    fn a_weight_side_at_its_limit_is_named_with_its_uptake() {
        assert_eq!(
            outcome(0, true).weight_uptake_label(),
            "weight zone at its limit (uptake 0%)"
        );
    }

    #[test]
    fn a_weight_side_with_room_reports_its_uptake() {
        assert_eq!(
            outcome(93, false).weight_uptake_label(),
            "weight uptake 93%"
        );
    }
}
