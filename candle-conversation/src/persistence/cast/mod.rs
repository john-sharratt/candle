//! npcd's cast as one export: every character's record and the portraits they
//! name, for a substrate wipe-and-rebuild (`docs/hit_driven_projection.md`
//! §13.4).
//!
//! A character is an `Npc` record, not a custom object, so the kept objects'
//! export ([`super::kept_objects`]) does not carry it, and its portrait is a
//! blob only the record names. This carries both, so a substrate a new model
//! key refuses can be exported offline and its cast written into the fresh
//! one before its daemon starts.
//!
//! ```text
//! {"format":"npc-cast","version":1,"taken_at_ms":…}
//! {"npc":{…NpcPayload…}}
//! …
//! {"blob":"<sha256 hex>","bytes_b64":"…"}
//! …
//! ```

mod export;
mod format;
mod import;

pub use export::export;
pub use format::{Header, Line, FORMAT, VERSION};
pub use import::{import, CastImport};

/// The SHA-256 a portrait's image id names: `<64 hex>.<ext>`, the id npcd's
/// images take. `None` for an id of another shape.
pub fn portrait_blob(image_id: &str) -> Option<[u8; 32]> {
    let (hex, _ext) = image_id.split_once('.')?;
    super::blob::parse_hex(hex)
}
