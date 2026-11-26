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
pub mod fill;
pub mod key;
pub mod names;
pub mod owner;
pub mod record;
pub mod render;
pub mod set;
pub mod stream_name;

pub use chunk::Chunk;
pub use cut::cut_lines;
pub use key::{content_key, ChunkKey, FileKey, LineSpan};
pub use owner::files_owner;
pub use record::{path_prefix, ChunkRecord, FileRecord, FileUnit, MemberRef, FILES_PREFIX};
pub use render::{assemble_pieces, knowledge_text, RenderedFile};
pub use set::{FilesPin, FilesSet};
pub use stream_name::{is_member_stream_name, member_stream_name};
