//! A record another member sent, as it arrives: its header line and its
//! payload, with or without the padding that rounds it to its on-disk span.
//! Checked here before it is written anywhere (`docs/replicated_substrate.md`
//! §6.5).

use serde_json::Value;

use super::super::content_hash::ContentHasher;
use super::super::record::{crc32, decode_header, padded_record_len, RecordHeader, RecordType};
use super::super::{PersistenceError, Result};
use super::outcome::{Checked, Refusal};

/// See the module docs.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Incoming {
    pub header: RecordHeader,
    /// The type as the header names it — what a member records for a type its
    /// build does not decode, which has no tag here.
    pub type_name: String,
    pub payload: Vec<u8>,
}

impl Incoming {
    /// Parse `bytes`: one JSON header line, the payload it describes, and
    /// either nothing after it or exactly the zero padding of its on-disk span.
    pub fn parse(bytes: &[u8]) -> Result<Incoming> {
        let (header, head) = decode_header(bytes)?;
        let end = head + header.payload_len as usize;
        if bytes.len() < end {
            return Err(PersistenceError::Truncated {
                need: end,
                have: bytes.len(),
            });
        }
        let padded = padded_record_len(head - 1, header.payload_len);
        if bytes.len() != end && bytes.len() != padded {
            return Err(PersistenceError::Corrupt(format!(
                "a {}-byte record whose payload ends at {end} and whose span is {padded}",
                bytes.len()
            )));
        }
        if bytes[end..].iter().any(|&b| b != 0) {
            return Err(PersistenceError::Corrupt(
                "a record's padding is not zero".to_string(),
            ));
        }
        let line: Value = serde_json::from_slice(&bytes[..head - 1])
            .map_err(|e| PersistenceError::Corrupt(format!("record header JSON parse: {e}")))?;
        let type_name = line
            .get("type")
            .and_then(Value::as_str)
            .ok_or_else(|| PersistenceError::Corrupt("a record header names no type".to_string()))?
            .to_string();
        Ok(Incoming {
            header,
            type_name,
            payload: bytes[head..end].to_vec(),
        })
    }

    /// Check the payload against what its header says of it: the `digest`
    /// over the whole payload, and for any record but a `Chunk` its CRC.
    ///
    /// **A `Chunk`'s golden is not recomputed here.** The digest has already
    /// read every byte of the payload and proved it is what its writer wrote,
    /// and the golden — the writer's checksum over the K/V bytes — is checked
    /// on the device when the chunk is loaded (`persistence::pipeline`). A
    /// second pass over every K/V byte of every replicated chunk proved
    /// nothing the first had not.
    pub fn verify(&self) -> Checked {
        let mut hasher = ContentHasher::new();
        hasher.update(&self.payload);
        if hasher.finish() != self.header.digest {
            return Err(Refusal::Digest);
        }
        if self.header.record_type != RecordType::Chunk && crc32(&self.payload) != self.header.crc {
            return Err(Refusal::Checksum);
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::persistence::identity::RecordIdentity;
    use crate::persistence::record::encode_record;

    fn header() -> RecordHeader {
        RecordHeader {
            record_type: RecordType::Tokens,
            payload_len: 3,
            stream_id: 9,
            ..RecordHeader::NONE
        }
        .with_identity(RecordIdentity {
            origin: 0xB,
            oseq: 4,
            gen: 0,
            epoch: 0,
            hlc: 0x10000,
        })
    }

    /// The record as encoded, exactly — padding and all. The CRC is the
    /// standard CRC-32 of `01 02 03`, `0x55BC801D`.
    #[test]
    fn a_padded_record_parses_and_verifies() {
        let bytes = encode_record(&header(), &[1, 2, 3]);
        assert_eq!(bytes.len(), 4096);
        let line = br#"{"type":"tokens","payload_len":3,"crc":1438416925,"format":0,"stream_id":9,"chunk_index":0,"token_count":0,"origin":11,"oseq":4,"hlc":65536,"digest":{"lo":15264681388316748627,"hi":7131329747664667309}}"#;
        assert_eq!(&bytes[..line.len()], line);
        let incoming = Incoming::parse(&bytes).unwrap();
        assert_eq!(incoming.type_name, "tokens");
        assert_eq!(incoming.payload, vec![1, 2, 3]);
        assert_eq!(incoming.header.oseq, 4);
        assert_eq!(incoming.verify(), Ok(()));
    }

    /// The same record without its padding parses the same.
    #[test]
    fn an_unpadded_record_parses_the_same() {
        let bytes = encode_record(&header(), &[1, 2, 3]);
        let line_end = bytes.iter().position(|&b| b == b'\n').unwrap() + 1;
        let unpadded = &bytes[..line_end + 3];
        assert_eq!(
            Incoming::parse(unpadded).unwrap(),
            Incoming::parse(&bytes).unwrap()
        );
    }

    /// A type this build does not know keeps its name.
    #[test]
    fn an_unknown_type_keeps_its_name() {
        let line =
            br#"{"type":"a_type_from_the_future","payload_len":2,"crc":0,"origin":11,"oseq":7}"#;
        let mut bytes = line.to_vec();
        bytes.push(b'\n');
        bytes.extend_from_slice(&[5, 6]);
        let incoming = Incoming::parse(&bytes).unwrap();
        assert_eq!(incoming.header.record_type, RecordType::Unknown);
        assert_eq!(incoming.type_name, "a_type_from_the_future");
        assert_eq!(incoming.payload, vec![5, 6]);
    }

    /// A short record, a length that is neither exact nor padded, and padding
    /// that is not zero are each refused.
    #[test]
    fn a_misshapen_record_is_refused() {
        let bytes = encode_record(&header(), &[1, 2, 3]);
        let line_end = bytes.iter().position(|&b| b == b'\n').unwrap() + 1;
        assert!(matches!(
            Incoming::parse(&bytes[..line_end + 2]),
            Err(PersistenceError::Truncated { .. })
        ));
        assert!(Incoming::parse(&bytes[..line_end + 4]).is_err());
        let mut dirty = bytes.clone();
        *dirty.last_mut().unwrap() = 1;
        assert!(Incoming::parse(&dirty).is_err());
    }

    /// A payload changed in flight fails its digest; a header whose CRC was
    /// changed fails the checksum.
    #[test]
    fn a_changed_record_fails_verification() {
        let bytes = encode_record(&header(), &[1, 2, 3]);
        let mut incoming = Incoming::parse(&bytes).unwrap();
        incoming.payload[0] = 9;
        assert_eq!(incoming.verify(), Err(Refusal::Digest));
        let mut incoming = Incoming::parse(&bytes).unwrap();
        incoming.header.crc ^= 1;
        assert_eq!(incoming.verify(), Err(Refusal::Checksum));
    }
}
