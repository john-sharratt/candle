//! A file as a projection placed it: the file and the lines of it placed
//! (`docs/provenance_system_collection.md` §3).
//!
//! What a projection point records of the knowledge base, so a reader can say
//! which runs of which files the model read at that point — a file is often
//! placed in part. The key travels as [`FileKey::hex`]: a 64-bit hash as a
//! JSON number loses its low bits in a browser, and the panel hands the key
//! back to ask for the file's text.

use serde::{Deserialize, Deserializer, Serialize, Serializer};

use super::{FileKey, LineSpan};

/// One placed file: its key and the line spans of its placed chunks, in the
/// order they sit.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct PlacedFile {
    #[serde(serialize_with = "key_hex", deserialize_with = "key_from_hex")]
    pub file: FileKey,
    pub chunks: Vec<LineSpan>,
}

fn key_hex<S: Serializer>(key: &FileKey, s: S) -> Result<S::Ok, S::Error> {
    s.serialize_str(&key.hex())
}

fn key_from_hex<'de, D: Deserializer<'de>>(d: D) -> Result<FileKey, D::Error> {
    let hex = String::deserialize(d)?;
    FileKey::from_hex(&hex)
        .ok_or_else(|| serde::de::Error::custom(format!("not a file key: {hex:?}")))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// **A placed file's key travels as its 16 hex digits**, whole, and reads
    /// back to the same key.
    #[test]
    fn a_placed_files_key_travels_as_hex() {
        let placed = PlacedFile {
            file: FileKey(0xfedc_ba98_7654_3210),
            chunks: vec![LineSpan {
                first: 1,
                last: 200,
            }],
        };
        let json = serde_json::to_string(&placed).unwrap();
        assert_eq!(
            json,
            r#"{"file":"fedcba9876543210","chunks":[{"first":1,"last":200}]}"#
        );
        assert_eq!(serde_json::from_str::<PlacedFile>(&json).unwrap(), placed);
        assert!(serde_json::from_str::<PlacedFile>(r#"{"file":"xyz","chunks":[]}"#).is_err());
    }
}
