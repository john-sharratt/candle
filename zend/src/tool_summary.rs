//! Tool-catalog hashing for the deterministic summary's cache record.
//!
//! The catalog summary itself is built **deterministically** from each tool's
//! `category` metadata by [`crate::tools::build_tool_summary`] — no model decode
//! (the previous categorize→assign generation cost two decode stages over ~90
//! tools). This module keeps only the content hash that tags the cached summary
//! record in the substrate redo log, so the projection panel reads back text
//! matching the current catalog and a restart with an unchanged catalog skips
//! the redo-log rewrite.

use candle_conversation::persistence::content_hash::hash_bytes;
use candle_conversation::projection::SectionId;

/// One installed tool: `(name, section_id, json_line)` — the triple
/// [`crate::tools::install_tool_catalog`] returns, in registry order.
pub type InstalledTool = (String, SectionId, String);

/// A 128-bit hash of the ordered tool catalog — each tool's name and full JSON
/// (which includes its parameter schema), concatenated in registry order. Two
/// runs with the same catalog produce the same hash; adding, removing, renaming,
/// or re-parameterising any tool changes it, which is what invalidates the
/// cached summary record.
pub fn catalog_hash(tools: &[InstalledTool]) -> u128 {
    let mut buf: Vec<u8> = Vec::new();
    for (name, _, json) in tools {
        buf.extend_from_slice(name.as_bytes());
        buf.push(0);
        buf.extend_from_slice(json.as_bytes());
        buf.push(0);
    }
    let h = hash_bytes(&buf);
    (u128::from(h.hi) << 64) | u128::from(h.lo)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn catalog_hash_is_stable_and_order_sensitive() {
        let a = (
            "alpha".to_string(),
            SectionId::new(1),
            "{\"name\":\"alpha\",\"parameters\":{}}".to_string(),
        );
        let b = (
            "beta".to_string(),
            SectionId::new(2),
            "{\"name\":\"beta\",\"parameters\":{}}".to_string(),
        );
        let ab = vec![a.clone(), b.clone()];
        // Stable across calls.
        assert_eq!(catalog_hash(&ab), catalog_hash(&ab));
        // Order-sensitive.
        assert_ne!(catalog_hash(&ab), catalog_hash(&[b.clone(), a.clone()]));
        // Parameter change invalidates.
        let b2 = (
            "beta".to_string(),
            SectionId::new(2),
            "{\"name\":\"beta\",\"parameters\":{\"x\":1}}".to_string(),
        );
        assert_ne!(catalog_hash(&ab), catalog_hash(&[a, b2]));
    }
}
