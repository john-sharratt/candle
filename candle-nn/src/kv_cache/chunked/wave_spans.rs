//! The wave arena's phase spans — how many bytes each bump domain reserves.
//!
//! Separate from [`super::bump_arena`], which owns the CUDA-backed arenas
//! themselves, because these are measurements: three numbers plus the record of
//! what was measured to arrive at them. Everything that consumes them —
//! `region_pool`'s carve arithmetic, `wave_plan`'s row pricing, and the asserts
//! that keep the two consistent — is integer arithmetic over addresses and
//! extents, so it is provable on a machine with no GPU. Same reasoning as
//! [`super::weight_zone`].

/// The attention phase's span.
///
/// Sized from measurement, not symmetry. A generation's bump cursor never
/// rewinds, so a phase needs the **sum** of everything it allocates, not its
/// peak. On Qwen3-30B-A3B at ten concurrent contexts the attention phase sums to
/// 297 MiB, which this rounds up to leave headroom for a wider wave.
///
/// Read that number against the one this constant used to hold. It was sized at
/// 128 MiB from a measured 32 MiB peak — but that peak was the *paged-attention
/// context buffer alone*, because `attention_norm` was accepting a wave and
/// passing `Backing::Owned`, so the norm, the QKV projection, its cast and
/// `o_proj` all inherited the pool instead. A span sized from a chain that is
/// not running measures the chain that is not running, and it reads exactly like
/// a comfortable fit: 24% utilisation, no error, no fallback warning. Seeding
/// the chain moved the phase to 297 MiB in one step.
pub const WAVE_ATTN_BYTES: usize = 384 * 1024 * 1024;

/// The FFN phase's span.
///
/// Four times the attention span because the expert chain is four buffers deep
/// over `rows x experts_per_tok` rows — gather, gate, up, SwiGLU, down, cast —
/// and none of them is reclaimed until the guard drops. This is the number that
/// was clipping: the measured peak sat at 99.87% of a 64 MiB span, so the true
/// demand was unknown until the cap was lifted off it.
pub const WAVE_FFN_BYTES: usize = 512 * 1024 * 1024;

/// The forward-scoped span.
///
/// Three orders of magnitude smaller than the layer phases, because it holds a
/// different *kind* of thing. The phase spans carry activations, which scale
/// with wave width times model width; this carries the metadata a forward builds
/// once and every layer reads — ragged prefill offsets, RoPE tables, gathered
/// position ids — which scale with the *sequence count*. A 64-sequence wave puts
/// a few kilobytes here.
///
/// **One region**, which is also the floor: the transient tier is carved in
/// `region_pool::REGION_BYTES` units and a compile-time assert requires
/// the total to stay region-aligned, so a smaller span would not buy back any
/// memory — it would round up to this anyway. The measured need is ~3 KB, so
/// this is ~5000x headroom, and none of it is wasted: a span that cannot be
/// subdivided cannot be spent on anything else.
pub const WAVE_FORWARD_BYTES: usize = 16 * 1024 * 1024;
