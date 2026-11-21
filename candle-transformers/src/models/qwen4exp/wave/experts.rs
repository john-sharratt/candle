//! The routed experts of each MoE layer, read through the streamed expert cache.
//!
//! Each MoE layer's routed experts are served by the shared `Qwen35MoeBlock` over the streamed `ExpertCache`, which the layer sweep in `layers.rs` drives over a wave's rows.
