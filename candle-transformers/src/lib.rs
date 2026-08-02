//! Model implementations for the unbounded-context inference engine.
//!
//! This is the model layer of the fork: a large zoo of upstream architecture
//! ports under [`models`] (Llama, Qwen3, Mixtral, Gemma, DeepSeek2, ViT/CLIP
//! families, audio/vision encoders, quantized GGUF variants, ...), plus the
//! fork-specific batched multi-session inference path built on top of it:
//!
//! - [`models::batched_inference`] / [`models::batched_model`] — the
//!   high-level `ManagedBatchedModel` API that steps many concurrent decode
//!   sessions, prefills, and glue-fires through a shared forward pass.
//! - [`models::expert_lre`] — the MoE expert-loading pipeline, including the
//!   Markov expert predictor that prefetches next-layer expert weights from
//!   the prior layer's routing pattern.
//! - [`generation`] — logits processing and sampling strategies
//!   (temperature, top-k, top-p, Gumbel-softmax) shared by all models.
//! - [`models::batch_test`] — the RULER-style long-context integration
//!   harness used to exercise the batched path end to end against fixture
//!   stories and system prompts.
//! - [`pipelines`] — thin task-level wrappers (e.g. text generation) over a
//!   loaded model.
//!
//! Downstream consumers (`candle-examples`, `candle-conversation`, `zend`)
//! call into `models::` for weight loading (`VarBuilder`) and forward passes,
//! and into `generation`/`pipelines` for turning logits into tokens.
pub mod generation;
pub mod models;
pub mod object_detection;
pub mod pipelines;
pub mod quantized_nn;
pub mod quantized_var_builder;
pub mod utils;
