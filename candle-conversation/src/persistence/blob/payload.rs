//! What a [`RecordType::Blob`](crate::persistence::record::RecordType::Blob)
//! carries: its SHA-256, then its bytes.

use sha2::{Digest, Sha256};

use crate::persistence::{PersistenceError, Result};

/// A blob's name in the record header: the first 128 bits of its SHA-256 —
/// `hi` the first eight bytes, in `stream_id`; `lo` the next eight, in
/// `chunk_index`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct BlobKey {
    pub hi: u64,
    pub lo: u64,
}

impl BlobKey {
    /// The key of the blob whose SHA-256 is `sha256`.
    pub fn of(sha256: &[u8; 32]) -> BlobKey {
        let word = |at: usize| {
            let mut w = [0u8; 8];
            w.copy_from_slice(&sha256[at..at + 8]);
            u64::from_be_bytes(w)
        };
        BlobKey {
            hi: word(0),
            lo: word(8),
        }
    }
}

/// A blob: bytes, and the SHA-256 that names them.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BlobPayload {
    pub sha256: [u8; 32],
    pub bytes: Vec<u8>,
}

impl BlobPayload {
    /// `bytes`, named by their hash.
    pub fn of(bytes: Vec<u8>) -> BlobPayload {
        BlobPayload {
            sha256: Sha256::digest(&bytes).into(),
            bytes,
        }
    }

    pub fn key(&self) -> BlobKey {
        BlobKey::of(&self.sha256)
    }

    /// The hash, then the bytes.
    pub fn encode(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(32 + self.bytes.len());
        out.extend_from_slice(&self.sha256);
        out.extend_from_slice(&self.bytes);
        out
    }

    /// A blob read back — refused unless its bytes still hash to its name.
    pub fn decode(buf: &[u8]) -> Result<BlobPayload> {
        if buf.len() < 32 {
            return Err(PersistenceError::Corrupt(format!(
                "Blob payload is {} bytes, shorter than its hash",
                buf.len()
            )));
        }
        let mut sha256 = [0u8; 32];
        sha256.copy_from_slice(&buf[..32]);
        let bytes = buf[32..].to_vec();
        let held: [u8; 32] = Sha256::digest(&bytes).into();
        if held != sha256 {
            return Err(PersistenceError::Corrupt(
                "Blob bytes do not hash to the name they are filed under".into(),
            ));
        }
        Ok(BlobPayload { sha256, bytes })
    }
}

/// `sha256` as lowercase hex — how a blob is named outside the substrate.
pub fn hex(sha256: &[u8; 32]) -> String {
    sha256.iter().map(|b| format!("{b:02x}")).collect()
}

/// The SHA-256 a 64-digit hex name spells, or `None` for anything else.
pub fn parse_hex(name: &str) -> Option<[u8; 32]> {
    if name.len() != 64 {
        return None;
    }
    let mut out = [0u8; 32];
    for (i, pair) in name.as_bytes().chunks(2).enumerate() {
        let digit = |c: u8| (c as char).to_digit(16);
        out[i] = (digit(pair[0])? * 16 + digit(pair[1])?) as u8;
    }
    Some(out)
}
