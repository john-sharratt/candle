//! What a substrate's model-derived state belongs to.
//!
//! Every K/V block, recurrent snapshot, provenance signature, index page and
//! token id in a substrate was computed by one model, and is valid under that
//! model only. A [`ModelKey`] names the model by what makes it compute what it
//! computes — its architecture, its weights, its vocabulary and its RoPE
//! schedule — and a substrate records the key of the model that wrote it and
//! refuses to open under any other (`docs/shared_model_host.md` §4).
//!
//! The key is computed before a single weight is loaded: the weights are named
//! by a fingerprint of each file's header and a sample of its tensor data
//! ([`gguf_fingerprint`]), so a refused start costs seconds rather than a model
//! load.

mod fingerprint;
mod key;
mod rope_key;
mod weight_files;

pub use fingerprint::{file_digest, gguf_fingerprint, SAMPLE_BYTES, SAMPLE_COUNT};
pub use key::{LegacySpec, ModelIdentity, ModelKey, RecordedModel};
pub use rope_key::rope_key;
pub use weight_files::WeightFiles;
