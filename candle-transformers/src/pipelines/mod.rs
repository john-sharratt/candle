//! Task-level pipeline wrappers over a loaded model.
//!
//! Thin orchestration layer above `models::`: given a model, tokenizer, and
//! [`crate::generation::LogitsProcessor`], drive a full task loop rather than
//! a single forward pass. Currently just [`text_generation`], a prompt-in /
//! text-out generation loop.
pub mod text_generation;
