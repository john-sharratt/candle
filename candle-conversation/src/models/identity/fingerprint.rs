//! A small file's identity: the SHA-256 of all of it.

use std::fs::File;
use std::io::{self, Read};
use std::path::Path;

use sha2::{Digest, Sha256};

/// SHA-256 of every byte of the file at `path` — for the small files a model
/// carries beside its pack (an adapter's weights and config), where a sample
/// would save nothing worth the doubt.
pub fn file_digest(path: &Path) -> io::Result<[u8; 32]> {
    let mut file = File::open(path)?;
    let mut hasher = Sha256::new();
    let mut buf = vec![0u8; 1 << 20];
    loop {
        let n = file.read(&mut buf)?;
        if n == 0 {
            break;
        }
        hasher.update(&buf[..n]);
    }
    Ok(hasher.finalize().into())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn hex(d: &[u8; 32]) -> String {
        d.iter().map(|b| format!("{b:02x}")).collect()
    }

    #[test]
    fn a_files_digest_is_its_sha256() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("adapter_config.json");
        std::fs::write(&path, b"abc").unwrap();
        assert_eq!(
            hex(&file_digest(&path).unwrap()),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad",
        );
    }
}
