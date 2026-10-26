//! The group key (§13.1): 32 random bytes, hex-encoded in a file. It only ever
//! keys an HMAC; it is never sent and never logged.

use std::fmt;
use std::fs::OpenOptions;
use std::io::Write;
use std::path::Path;

use hmac::{Hmac, Mac};
use sha2::Sha256;
use thiserror::Error;

type HmacSha256 = Hmac<Sha256>;

#[derive(Debug, Error)]
pub enum KeyError {
    #[error("group key {path}: {source}")]
    Read {
        path: String,
        source: std::io::Error,
    },
    #[error("group key {path}: expected 64 hex digits")]
    Format { path: String },
    #[error("group key {path}: readable by group or other (mode {mode:o}); chmod 600 it")]
    Exposed { path: String, mode: u32 },
    #[error("group key {path}: not written: {source}")]
    Write {
        path: String,
        source: std::io::Error,
    },
}

#[derive(Clone, PartialEq, Eq)]
pub struct GroupKey([u8; 32]);

impl GroupKey {
    pub fn from_bytes(bytes: [u8; 32]) -> Self {
        Self(bytes)
    }

    /// 64 hex digits, surrounding whitespace ignored.
    pub fn from_hex(text: &str) -> Option<Self> {
        let text = text.trim();
        if text.len() != 64 {
            return None;
        }
        let mut out = [0u8; 32];
        for (i, pair) in text.as_bytes().chunks(2).enumerate() {
            let hi = (pair[0] as char).to_digit(16)?;
            let lo = (pair[1] as char).to_digit(16)?;
            out[i] = (hi * 16 + lo) as u8;
        }
        Some(Self(out))
    }

    /// Read the key file, refusing one other users can read.
    pub fn load(path: &Path) -> Result<Self, KeyError> {
        let shown = path.display().to_string();
        check_private(path, &shown)?;
        let text = std::fs::read_to_string(path).map_err(|source| KeyError::Read {
            path: shown.clone(),
            source,
        })?;
        Self::from_hex(&text).ok_or(KeyError::Format { path: shown })
    }

    pub(crate) fn bytes(&self) -> &[u8; 32] {
        &self.0
    }

    /// A key of its own for `purpose`: HMAC-SHA256 of the purpose's name under
    /// this key. Every member holding the group key derives the same one, and
    /// what it signs cannot be passed off as a handshake proof or as another
    /// purpose's signature, so the group key itself never leaves this crate.
    pub fn derive(&self, purpose: &str) -> [u8; 32] {
        let mut m = HmacSha256::new_from_slice(&self.0).expect("HMAC takes a key of any length");
        m.update(purpose.as_bytes());
        m.finalize().into_bytes().into()
    }

    /// 32 bytes from the OS-seeded generator.
    pub fn generate() -> Self {
        let mut bytes = [0u8; 32];
        rand::fill(&mut bytes);
        Self(bytes)
    }

    /// The file's form: 64 hex digits and a newline.
    fn to_file(&self) -> String {
        let mut s: String = self.0.iter().map(|b| format!("{b:02x}")).collect();
        s.push('\n');
        s
    }

    /// Write a fresh key to `path`, which must not exist yet — an estate's
    /// key is replaced deliberately, never by a second run of this. The file
    /// is private to its owner on Unix. The key is not shown anywhere: the
    /// file is what is copied to every gateway and member.
    pub fn create(path: &Path) -> Result<Self, KeyError> {
        let shown = path.display().to_string();
        let key = Self::generate();
        let mut options = OpenOptions::new();
        options.write(true).create_new(true);
        #[cfg(unix)]
        {
            use std::os::unix::fs::OpenOptionsExt;
            options.mode(0o600);
        }
        let written = options.open(path).and_then(|mut f| {
            f.write_all(key.to_file().as_bytes())
                .and_then(|()| f.sync_all())
        });
        written.map_err(|source| KeyError::Write {
            path: shown,
            source,
        })?;
        Ok(key)
    }
}

impl fmt::Debug for GroupKey {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("GroupKey(…)")
    }
}

#[cfg(unix)]
fn check_private(path: &Path, shown: &str) -> Result<(), KeyError> {
    use std::os::unix::fs::PermissionsExt;
    let mode = std::fs::metadata(path)
        .map_err(|source| KeyError::Read {
            path: shown.to_string(),
            source,
        })?
        .permissions()
        .mode();
    if mode & 0o077 != 0 {
        return Err(KeyError::Exposed {
            path: shown.to_string(),
            mode: mode & 0o777,
        });
    }
    Ok(())
}

#[cfg(not(unix))]
fn check_private(_path: &Path, _shown: &str) -> Result<(), KeyError> {
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hex_parses_exactly_64_digits() {
        let k = GroupKey::from_hex(&format!(" {}\n", "0f".repeat(32))).unwrap();
        assert_eq!(k.bytes(), &[0x0f; 32]);
        assert!(GroupKey::from_hex(&"0f".repeat(31)).is_none());
        assert!(GroupKey::from_hex(&format!("{}zz", "0f".repeat(31))).is_none());
        assert_eq!(format!("{k:?}"), "GroupKey(…)");
    }

    #[test]
    fn a_key_file_loads() {
        let dir = std::env::temp_dir().join(format!("cluster-key-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("group.key");
        std::fs::write(&path, "ab".repeat(32)).unwrap();
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o644)).unwrap();
            assert!(matches!(
                GroupKey::load(&path),
                Err(KeyError::Exposed { .. })
            ));
            std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o600)).unwrap();
        }
        assert_eq!(GroupKey::load(&path).unwrap().bytes(), &[0xab; 32]);
        std::fs::remove_dir_all(&dir).unwrap();
    }

    /// **A derived key is the purpose's HMAC under the group key**: the same on
    /// every member, different per purpose, and never the group key itself.
    #[test]
    fn a_derived_key_is_the_purposes_hmac_under_the_group_key() {
        let k = GroupKey::from_bytes([0xab; 32]);
        let mut m = HmacSha256::new_from_slice(&[0xab; 32]).unwrap();
        m.update(b"npcd image links");
        let expected: [u8; 32] = m.finalize().into_bytes().into();
        assert_eq!(k.derive("npcd image links"), expected);
        assert_eq!(
            GroupKey::from_bytes([0xab; 32]).derive("npcd image links"),
            expected,
            "another member holding the same key"
        );
        assert_ne!(k.derive("another purpose"), expected);
        assert_ne!(k.derive("npcd image links"), [0xab; 32]);
        assert_ne!(
            GroupKey::from_bytes([0xac; 32]).derive("npcd image links"),
            expected
        );
    }

    /// A created key is 64 hex digits and a newline, loads back as itself,
    /// and is never written over.
    #[test]
    fn a_created_key_loads_back_and_is_never_overwritten() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("group.key");
        let key = GroupKey::create(&path).unwrap();
        let text = std::fs::read_to_string(&path).unwrap();
        assert_eq!((text.len(), text.ends_with('\n')), (65, true));
        assert_eq!(GroupKey::load(&path).unwrap(), key);
        assert!(matches!(
            GroupKey::create(&path),
            Err(KeyError::Write { .. })
        ));
        assert_eq!(GroupKey::load(&path).unwrap(), key, "unchanged");
        assert_ne!(GroupKey::generate(), GroupKey::generate());
    }
}
