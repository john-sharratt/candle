//! The knowledge base's zend side (`docs/provenance_system_collection.md`):
//! what the daemon renders into it from the workspace.
//!
//! - [`ingest`] — a file cut, sealed as members and recorded;
//! - [`listing`] — a folder as the knowledge base shows it;
//! - [`records`] — the file records in the substrate;
//! - [`sections`] — a file cut where `file_read` pages it, each section its
//!   read;
//! - [`seed`] — the seed every conversation opens on, rendered from the
//!   repositories' roots and installed into the schema at startup.

pub mod ingest;
pub mod listing;
pub mod records;
pub mod sections;
pub mod seed;

/// The collection the knowledge base's files are placed through: filled per
/// projection from the corpus by the fill, never from a content folder.
pub const FILES_COLLECTION: &str = "files";
