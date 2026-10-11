//! The group key (§13.1): 32 random bytes, hex-encoded in a file. It only ever
//! keys an HMAC; it is never sent and never logged.

use std::fmt;
use std::path::Path;

use thiserror::Error;

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
}
