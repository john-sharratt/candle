//! Notes tools: `notes_{write,read,search,list}`.
//!
//! Cross-conversation persistent key-value store for agent memory.  Unlike the
//! VFS (per-session, gone at conversation end), notes persist until explicitly
//! removed.  Use cases: infrastructure naming conventions, service→port mappings,
//! DB schemas, per-person context, design decisions, anything the agent should
//! remember across conversations.
//!
//! # Key format
//!
//! Free-form strings, max 256 bytes.  Hierarchical patterns like `infra/dns/internal`
//! are encouraged for organisation but not enforced.
//!
//! # Deletion
//!
//! Writing empty content to a key tombstones the note.  There is no separate
//! `notes_delete` tool — a single write path avoids accidental data loss from a
//! stray delete call.
//!
//! # Search
//!
//! `notes_search` accepts full-text query (FTS5 syntax: quotes, AND/OR/NOT, `*`
//! prefix) and/or tag filter.  At least one must be provided.
//! `notes_list` enumerates by key prefix or tag without searching content.
//!
//! # Size limits
//!
//! Individual notes: 1 MiB ([`MAX_NOTE_BYTES`]).  Key length: 256 bytes.
//!
//! # Error codes
//!
//! | Code | Cause |
//! |------|-------|
//! | `not_found` | Key does not exist in store (`notes_read`) |
//! | `note_too_large` | Content exceeds 1 MiB |
//! | `key_too_long` | Key exceeds 256 bytes |
//! | `no_search_criteria` | `notes_search` called with neither query nor tags |

use thiserror::Error;

use crate::ToolError;

pub const MAX_NOTE_BYTES: usize = 1024 * 1024; // 1 MiB

pub mod write;
pub mod read;
pub mod search;
pub mod list;

pub use write::NOTES_WRITE;
pub use read::NOTES_READ;
pub use search::NOTES_SEARCH;
pub use list::NOTES_LIST;

#[derive(Debug, Error)]
pub enum NotesError {
    #[error("note not found: {0}")]
    NotFound(String),
    #[error("note content exceeds 1 MiB limit")]
    NoteTooLarge,
    #[error("key too long (max 256 bytes)")]
    KeyTooLong,
    #[error("no search criteria provided")]
    NoSearchCriteria,
}

impl ToolError for NotesError {
    fn code(&self) -> &'static str {
        match self {
            NotesError::NotFound(_) => "not_found",
            NotesError::NoteTooLarge => "note_too_large",
            NotesError::KeyTooLong => "key_too_long",
            NotesError::NoSearchCriteria => "no_search_criteria",
        }
    }
}
