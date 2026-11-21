//! Tests how a committed ingested unit is named in a timeline's key.
//!
//! A unit is named by what it covers: a file's reading by its path, a folder by
//! its folder, and a layer's unit by its subject. A conversation with no content
//! key is named by none of these, so it cannot collide with a unit's key.

use super::*;

#[cfg(test)]
pub(super) mod unit_key_tests {
    use candle_conversation::substrate::pcoll::POrdMap;

    use super::committed_unit_key;

    fn meta(pairs: &[(&str, &str)]) -> POrdMap<String, String> {
        pairs
            .iter()
            .map(|(k, v)| (k.to_string(), v.to_string()))
            .collect()
    }

    /// **A committed unit is named by what it covers**: a file's reading by
    /// its path, a folder by its folder, a layer's unit by its subject — and a
    /// conversation without a content key by none of them.
    #[test]
    fn a_committed_unit_is_named_by_what_it_covers() {
        let read = meta(&[("path", "candle/zend/src/rate.rs"), ("content_key", "k1")]);
        assert_eq!(
            committed_unit_key(&read).map(String::as_str),
            Some("candle/zend/src/rate.rs")
        );
        let review = meta(&[
            ("layer", "code_review"),
            ("subject", "candle/zend/src/rate.rs"),
            ("content_key", "u2:code_review:ab"),
        ]);
        assert_eq!(
            committed_unit_key(&review).map(String::as_str),
            Some("candle/zend/src/rate.rs")
        );
        let partial = meta(&[("layer", "code_review"), ("subject", "candle/x.rs")]);
        assert_eq!(committed_unit_key(&partial), None);
    }
}
