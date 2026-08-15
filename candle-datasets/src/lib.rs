//! Datasets & dataloaders for Candle example/training code.
//!
//! [`batcher`] provides the generic [`Batcher`] iterator adapter for turning
//! a stream of samples into batched tensors. [`hub`] fetches dataset files
//! from the Hugging Face Hub. [`nlp`] and [`vision`] hold task-specific
//! dataset loaders (e.g. TinyStories, CIFAR, Fashion-MNIST, MNIST) that
//! return ready-to-train `Tensor`s. This crate is independent of the KV
//! cache / inference engine — it only supports example and training code.
pub mod batcher;
pub mod hub;
pub mod nlp;
pub mod vision;

pub use batcher::Batcher;
