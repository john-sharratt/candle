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
//! by the whole-file SHA-256 of every source their model pack records
//! ([`WeightFiles::digest`]), read from the pack's header, so a refused start
//! costs milliseconds rather than a model load.

mod fingerprint;
mod key;
mod rope_key;
mod weight_files;

pub use fingerprint::file_digest;
pub use key::{LegacySpec, ModelIdentity, ModelKey, RecordedModel};
pub use rope_key::rope_key;
#[cfg(test)]
pub(crate) use weight_files::write_pack_header;
pub use weight_files::WeightFiles;
