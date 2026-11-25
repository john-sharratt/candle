//! What a member of the knowledge base is declared as in the redo log.
//!
//! A member is a shared section — no conversation's own, addressed by its
//! content and everything sealed before it — so its stream is declared under
//! the shared prefix ([`shared_stream_name`]), and the seal quantizes it at
//! once, as every shared section is. Within the shared streams, the `files/`
//! name tells a member apart: its seal also writes its provenance signatures,
//! which no other section carries, because a member is retrieved by them.

use crate::conversation::{shared_stream_name, SHARED_SECTION_NAME_PREFIX};

/// Where a member's name starts, after the shared prefix.
pub const MEMBER_NAME_PREFIX: &str = "files/";

/// The `debug_name` the member named `name` is declared under:
/// `shared/files/{name}`.
pub fn member_stream_name(name: &str) -> String {
    shared_stream_name(&format!("{MEMBER_NAME_PREFIX}{name}"))
}

/// Whether a stream declared under `debug_name` is a member of the knowledge
/// base's.
pub fn is_member_stream_name(debug_name: &str) -> bool {
    debug_name
        .strip_prefix(SHARED_SECTION_NAME_PREFIX)
        .is_some_and(|rest| rest.starts_with(MEMBER_NAME_PREFIX))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::conversation::is_shared_stream_name;

    /// **A member is a shared stream named under `files/`**, and only such a
    /// stream is a member's.
    #[test]
    fn a_member_is_a_shared_stream_under_files() {
        let name = member_stream_name("candle/zend/src/lib.rs#1-200");
        assert_eq!(name, "shared/files/candle/zend/src/lib.rs#1-200");
        assert!(is_shared_stream_name(&name));
        assert!(is_member_stream_name(&name));
        assert!(!is_member_stream_name("shared/journal/1"));
        assert!(!is_member_stream_name("files/candle/zend/src/lib.rs#1-200"));
        assert!(!is_member_stream_name("conv/77/files/x"));
    }
}
