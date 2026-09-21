//! The RoPE rung set every paged attention kernel takes (`rope/rope_table.cuh`).

/// Every rung of a model's RoPE schedule, as one launch argument: the rungs'
/// factored tables end to end, each rung's Q rotary scale, the rung count and
/// the rotary pair count. Passed by value; mirrors the CUDA `RopeRungs` field
/// for field (24 bytes, pointers first).
///
/// A kernel takes each sequence's rung from that sequence's own
/// `SlotHeader.rope_rung`, so one launch serves sequences on different rungs
/// without any of them reading another's table.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RopeRungsFfi {
    /// `float2 (sin, cos)[n_rungs][(2048 + 1024) · pairs]`.
    pub tables: u64,
    /// `f32[n_rungs]`: YaRN's `m²` on a rung with temperature, else 1.
    pub q_scale: u64,
    pub n_rungs: u32,
    pub pairs: u32,
}

const _: () = assert!(std::mem::size_of::<RopeRungsFfi>() == 24);
