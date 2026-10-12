//! SHA-256 of an artifact's bytes, and its hex spelling — how an artifact is
//! named on the wire, in part files and in `build.json` (§3.9).

use std::fs::File;
use std::io::{self, Read};
use std::path::Path;

use sha2::{Digest, Sha256};

/// The SHA-256 of the file at `path`, read in pieces.
pub fn file_sha256(path: &Path) -> io::Result<[u8; 32]> {
    let mut file = File::open(path)?;
    let mut digest = Sha256::new();
    let mut buf = vec![0u8; 1 << 20];
    loop {
        let n = file.read(&mut buf)?;
        if n == 0 {
            break;
        }
        digest.update(&buf[..n]);
    }
    Ok(digest.finalize().into())
}

pub fn sha256(bytes: &[u8]) -> [u8; 32] {
    Sha256::digest(bytes).into()
}

pub fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

/// The 32 bytes a 64-digit hex spelling names.
pub fn unhex(text: &str) -> Option<[u8; 32]> {
    if text.len() != 64 || !text.is_ascii() {
        return None;
    }
    let mut out = [0u8; 32];
    for (i, b) in out.iter_mut().enumerate() {
        *b = u8::from_str_radix(&text[i * 2..i * 2 + 2], 16).ok()?;
    }
    Some(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The digest of the empty input, and of a file, against the published
    /// value; hex round-trips.
    #[test]
    fn digests_are_sha256_and_hex_round_trips() {
        let empty = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855";
        assert_eq!(hex(&sha256(b"")), empty);
        let dir = tempfile::TempDir::new().unwrap();
        let f = dir.path().join("a");
        std::fs::write(&f, b"").unwrap();
        assert_eq!(hex(&file_sha256(&f).unwrap()), empty);
        assert_eq!(unhex(empty), Some(sha256(b"")));
        assert_eq!(unhex("zz"), None);
        assert_eq!(unhex(&"g".repeat(64)), None);
    }
}
