//! The cast export's lines.

use serde::{Deserialize, Serialize};

use crate::persistence::record::NpcPayload;

/// The header's `format`.
pub const FORMAT: &str = "npc-cast";

/// The header's `version`.
pub const VERSION: u32 = 1;

/// The export's first line.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Header {
    pub format: String,
    pub version: u32,
    pub taken_at_ms: u64,
}

impl Header {
    pub fn new(taken_at_ms: u64) -> Self {
        Self {
            format: FORMAT.into(),
            version: VERSION,
            taken_at_ms,
        }
    }
}

/// Every line after the header.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum Line {
    Npc { npc: NpcPayload },
    Blob { blob: String, bytes_b64: String },
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_header_is_the_formats() {
        assert_eq!(
            serde_json::to_string(&Header::new(7)).unwrap(),
            r#"{"format":"npc-cast","version":1,"taken_at_ms":7}"#
        );
        let blob: Line = serde_json::from_str(r#"{"blob":"ab","bytes_b64":"AQI="}"#).unwrap();
        assert_eq!(
            blob,
            Line::Blob {
                blob: "ab".into(),
                bytes_b64: "AQI=".into()
            }
        );
    }
}
