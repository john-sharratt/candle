//! The knowledge base: the files provenance places at the end of the system
//! prompt (`docs/provenance_system_collection.md` §3).
//!
//! - [`key`] — what names a file ([`FileKey`]) and a run of its lines
//!   ([`LineSpan`], [`ChunkKey`]);
//! - [`chunk`] and [`cut`] — a file cut into chunks of whole lines whose
//!   seams tokenize as the whole file does;
//! - [`render`] — the notation, Qwen-Agent's, and the pieces the prompt seals
//!   it as.

pub mod chunk;
pub mod cut;
pub mod key;
pub mod render;

pub use chunk::Chunk;
pub use cut::cut_lines;
pub use key::{content_key, ChunkKey, FileKey, LineSpan};
pub use render::{assemble_pieces, knowledge_text, RenderedFile};
