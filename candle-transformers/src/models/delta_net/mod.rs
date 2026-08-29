//! Gated DeltaNet — the generic linear-attention subsystem for hybrid stacks.
//!
//! Everything here is model-agnostic: the mixer's algebra and its CUDA
//! kernels' wrappers, the per-session recurrent state store with its
//! wave-atomicity contract, the hybrid layer↔KV-cache index map, and the
//! quantized layer driver. A model family (the `qwen35` lineage today)
//! supplies configuration, tensor names and the surrounding sweep; nothing in
//! this module reads a GGUF name or a model config.
//!
//! Mirrors the kernel family `candle-kernels/src/delta-net/`, whose parity
//! oracle is [`mix`]'s sequential reference.

#[cfg(feature = "cuda")]
pub mod cuda;
pub mod kv_layout;
pub mod mix;
pub mod quantized;
pub mod state_store;
pub mod types;

pub use kv_layout::KvLayerMap;
pub use mix::{
    causal_conv1d, delta_net_advance_spans, delta_net_layer_forward, delta_net_mix,
    delta_net_mix_spans, delta_recurrence, delta_step, l2_norm, seq_spans, DeltaNetConstants,
    DeltaNetLayerTable, DeltaNetOut, DeltaNetProjections, DeltaNetSeq, DeltaNetSpanTable,
    DeltaNetState, DeltaNetWeights, SeqSpan, SpanOperands, StashSlot,
};
pub use quantized::{
    quantized_delta_net_layer_forward, quantized_delta_net_layer_forward_spans,
    QuantDeltaNetWeights,
};
pub use state_store::{ExportedLayerState, RecurrentStateStore};
pub use types::{DeltaNetDims, LayerKind};
