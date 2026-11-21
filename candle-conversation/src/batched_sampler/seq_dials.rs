//! The per-row dials the kernel reads, in the exact layout of its C twin.
//!
//! Each row samples on its own config, so one row's EOS ramp or temperature
//! cannot reach another row in a wave that mixes configs. The kernel reads these
//! entries back as the C struct, so every field is a 4-byte `f32` or `i32` with
//! no padding, and a compile-time check pins the field count and size. Building
//! a row resolves the same optional settings the scalar path resolves, so a
//! row in a mixed wave behaves as it would in a uniform one. A segment-close
//! token of -1 turns that path off for the row alone.

use super::*;

/// Per-sequence sampling dials, uploaded as a `[batch_size]` array and read by
/// the kernel at `seq_dials[row]` so every row samples on its own config.
///
/// **The C twin is `batched_sampling::SeqDials` in `batched_sampling.cuh`.** The
/// kernel reads these bytes back as that struct, so the field order and types
/// here must match it exactly — every field is 4 bytes (`f32`/`i32`), packed
/// with no padding, and `#[repr(C)]` keeps the layout. The EOS id, the row
/// stride and the live vocabulary are not here (they are the same across the
/// wave and stay scalar).
///
/// `typical_draft` is the proposal a
/// speculative verify row tests (-1 on any other row), accepted on
/// `typical_eps`/`typical_delta` when the sample did not land on it — see
/// `candle_transformers::models::speculative_choice::TypicalAcceptance`.
///
/// Before this existed the kernel took these dials as scalars from the first
/// row's config and applied them to the whole launch, so a wave that mixed
/// configs — a deliberating turn beside an impulsive one, a narrator beside a
/// reflection — sampled every row at whichever config sorted first, and one
/// row's EOS ramp could cut another's turn off after a single token.
#[repr(C)]
#[derive(Clone, Copy)]
pub(super) struct SeqDials {
    pub(super) temperature: f32,
    pub(super) top_k: i32,
    pub(super) top_p: f32,
    pub(super) repeat_penalty: f32,
    pub(super) frequency_penalty: f32,
    pub(super) presence_penalty: f32,
    pub(super) dry_multiplier: f32,
    pub(super) dry_base: f32,
    pub(super) dry_allowed_length: i32,
    pub(super) dry_range: i32,
    pub(super) eos_boost: f32,
    pub(super) eos_ramp_start: i32,
    pub(super) eos_ramp_len: i32,
    pub(super) eos_boost_max_multiplier: f32,
    pub(super) cross_turn_penalty: f32,
    pub(super) segment_close_boost: f32,
    pub(super) segment_close_token_id: i32,
    pub(super) segment_close_ramp_start: i32,
    pub(super) segment_close_ramp_len: i32,
    pub(super) segment_close_max_multiplier: f32,
    pub(super) segment_temp_boost: f32,
    pub(super) typical_draft: i32,
    pub(super) typical_eps: f32,
    pub(super) typical_delta: f32,
    /// The row of the logits block this entry samples. A dispatch whose other
    /// rows are resolved on the host (a grammar allow-list) hands the kernel
    /// the whole block and names each kernel row's source here, rather than
    /// gathering the kernel rows into a compacted copy first — a copy that, on
    /// a wave's head span, would claim a block of logits the wave never priced.
    pub(super) logits_row: i32,
    /// The device address of the row's allowed-token bitmap, low word first, or
    /// zero for none (`SamplingConfig::allowed_bits`).
    pub(super) allowed_bits_lo: u32,
    pub(super) allowed_bits_hi: u32,
}

// The kernel reads this struct as a flat run of 4-byte words (all fields are
// f32/i32), one per row of the batch, and casts back to the identically-laid-out
// CUDA `SeqDials`. If the size or field count drifts from the CUDA side the
// kernel reads a row at the wrong stride, so pin it: 27 fields × 4 bytes.
const _: () = assert!(std::mem::size_of::<SeqDials>() == 108);

impl SeqDials {
    /// Read one row's dials from its config, resolving the same Option/gate logic
    /// the scalar path applies (DRY defaults, the dynamic-EOS gate, the
    /// segment-close-active gate) so a per-row wave behaves identically to a
    /// uniform one row-for-row. The row samples `logits_row` of the dispatch's
    /// logits block, and tests no draft until [`Self::with_draft`] gives it one.
    pub(super) fn from_config(c: &SamplingConfig, logits_row: usize) -> Self {
        let (dry_multiplier, dry_base, dry_allowed_length, dry_range) = match &c.dry {
            Some(d) => (d.multiplier, d.base, d.allowed_length, d.range),
            None => (0.0, 1.75, 2, 0),
        };
        let (eos_ramp_start, eos_ramp_len, eos_boost_max_multiplier) = if c.dynamic_eos_boost {
            (c.eos_ramp_start, c.eos_ramp_len, c.eos_boost_max_multiplier)
        } else {
            (0, 0, 0.0)
        };
        let (
            segment_close_boost,
            segment_close_token_id,
            segment_close_ramp_start,
            segment_close_ramp_len,
            segment_close_max_multiplier,
        ) = if c.segment_close_boost != 0.0 && c.segment_close_token_id >= 0 {
            (
                c.segment_close_boost,
                c.segment_close_token_id,
                c.segment_close_ramp_start,
                c.segment_close_ramp_len,
                c.segment_close_max_multiplier,
            )
        } else {
            // Disabled for this row: -1 token id keeps the kernel's per-row
            // segment-close path off even when a co-batched row has it on.
            (0.0, -1, 0, 0, 0.0)
        };
        SeqDials {
            temperature: c.temperature,
            top_k: c.top_k,
            top_p: c.top_p,
            repeat_penalty: c.repeat_penalty,
            frequency_penalty: c.frequency_penalty,
            presence_penalty: c.presence_penalty,
            dry_multiplier,
            dry_base,
            dry_allowed_length,
            dry_range,
            eos_boost: c.eos_boost,
            eos_ramp_start,
            eos_ramp_len,
            eos_boost_max_multiplier,
            cross_turn_penalty: c.cross_turn_penalty,
            segment_close_boost,
            segment_close_token_id,
            segment_close_ramp_start,
            segment_close_ramp_len,
            segment_close_max_multiplier,
            segment_temp_boost: c.segment_temp_boost,
            typical_draft: -1,
            typical_eps: 0.0,
            typical_delta: 0.0,
            logits_row: logits_row as i32,
            allowed_bits_lo: 0,
            allowed_bits_hi: 0,
        }
    }

    /// This row is a speculative verify row testing `draft` on `typical`.
    /// A `None` draft leaves the row a plain one.
    pub(super) fn with_draft(mut self, draft: Option<u32>, typical: TypicalAcceptance) -> Self {
        if let Some(d) = draft {
            self.typical_draft = d as i32;
            self.typical_eps = typical.epsilon;
            self.typical_delta = typical.delta;
        }
        self
    }
}
