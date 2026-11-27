//! The knowledge base: the files provenance places at the end of the system
//! prompt (`docs/provenance_system_collection.md` §3).
//!
//! - [`key`] — what names a file ([`FileKey`]) and a run of its lines
//!   ([`LineSpan`], [`ChunkKey`]);
//! - [`chunk`] — a section of a file: the lines it holds and its text;
//! - [`mandatory`] — the files every free knowledge base places whole;
//! - [`render`] — the notation, and the pieces the prompt seals it as.

pub mod chunk;
pub mod fill;
pub mod key;
pub mod mandatory;
pub mod names;
pub mod owner;
pub mod placed;
pub mod record;
pub mod render;
pub mod set;
pub mod stream_name;

pub use chunk::Chunk;
pub use key::{content_key, ChunkKey, FileKey, LineSpan};
pub use mandatory::{leaves_room, mandatory_records};
pub use owner::files_owner;
pub use placed::PlacedFile;
pub use record::{path_prefix, ChunkRecord, FileRecord, FileUnit, MemberRef, FILES_PREFIX};
pub use render::{assemble_pieces, knowledge_text, RenderedFile};
pub use set::{FilesPin, FilesSet};
pub use stream_name::{is_member_stream_name, member_stream_name};
