//! The model pack: one file per model and numeric mode, holding everything a
//! load reads — `docs/self_contained_model_packs.md` is the design.
//!
//! ```text
//! GGUF part   every tensor but the experts and the streamed projections,
//!             the checkpoint's metadata, the tokenizer, the provenance,
//!             and where each section starts            (alignment 4096)
//! experts     the expert section, every layer's records     (routed)
//! layers      the layer section, every layer's records      (dense, streamed)
//! ```
//!
//! A GGUF reader opens the file as an ordinary GGUF and never sees the sections
//! after it. The loaders read the GGUF part through a mapping and the sections
//! through direct I/O, at offsets the metadata records.
//!
//! A pack is built once, from source checkpoints that are then released
//! ([`resolve::model_pack`]); after that it is the model's only file.

pub mod build;
pub mod cache;
pub mod compose;
pub mod digest;
#[cfg(feature = "cuda")]
pub mod experts;
#[cfg(feature = "cuda")]
pub(crate) mod family;
#[cfg(all(feature = "cuda", any(test, feature = "hub")))]
pub mod hub;
pub mod keys;
#[cfg(feature = "cuda")]
pub mod local;
pub mod open;
pub mod provenance;
pub mod request;
#[cfg(feature = "cuda")]
pub mod resolve;

pub use cache::cache_root;
#[cfg(all(feature = "cuda", any(test, feature = "hub")))]
pub use hub::HubFetch;
#[cfg(feature = "cuda")]
pub use local::{local_label, local_rev, LocalFetch};
pub use open::{ModelPack, SectionRef};
pub use provenance::SourceRecord;
pub use request::{PackFamily, PackRequest, SourceRef};
#[cfg(feature = "cuda")]
pub use resolve::{existing_pack, model_pack, Fetched, SourceFetch};
