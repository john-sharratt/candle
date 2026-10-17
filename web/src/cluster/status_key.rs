//! The status key (`docs/replicated_substrate.md` §12): a server-side secret
//! that reads `/gateway/status.json` without a browser session — for a script,
//! or an operator at a shell. `web --new-status-key <PATH>` writes one; the
//! `cluster:` block's `status_key_file` names it; a request presents it as
//! `Authorization: Bearer <64 hex digits>`.
//!
//! **It reads the status and nothing else.** The page, every command, and a
//! member's admission stay what they were: an admin's session, and the group
//! key. A key of its own type, never the group key, so holding one can never
//! admit a member.

use std::fs::OpenOptions;
use std::io::Write;
use std::path::Path;

use anyhow::{anyhow, Context};
use axum::http::{header, HeaderMap};

/// 32 bytes, kept as their file's 64 hex digits.
pub struct StatusKey([u8; 32]);

impl StatusKey {
    /// 64 hex digits, surrounding whitespace ignored.
    pub fn from_hex(text: &str) -> Option<StatusKey> {
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
        Some(StatusKey(out))
    }

    /// The key in `path`.
    pub fn load(path: &Path) -> anyhow::Result<StatusKey> {
        let text = std::fs::read_to_string(path)
            .with_context(|| format!("reading the status key {}", path.display()))?;
        StatusKey::from_hex(&text).ok_or_else(|| anyhow!("{} is not 64 hex digits", path.display()))
    }

    /// Write a fresh key to `path`, which must not exist yet: a key is
    /// replaced deliberately, never by a second run of this.
    pub fn create(path: &Path) -> anyhow::Result<()> {
        let mut bytes = [0u8; 32];
        rand::fill(&mut bytes);
        let mut text: String = bytes.iter().map(|b| format!("{b:02x}")).collect();
        text.push('\n');
        let mut options = OpenOptions::new();
        options.write(true).create_new(true);
        #[cfg(unix)]
        {
            use std::os::unix::fs::OpenOptionsExt;
            options.mode(0o600);
        }
        options
            .open(path)
            .and_then(|mut f| f.write_all(text.as_bytes()).and_then(|()| f.sync_all()))
            .with_context(|| format!("writing the status key {}", path.display()))
    }

    /// Whether `headers` present this key as `Authorization: Bearer <hex>`,
    /// compared in constant time.
    pub fn presented(&self, headers: &HeaderMap) -> bool {
        let Some(token) = headers
            .get(header::AUTHORIZATION)
            .and_then(|v| v.to_str().ok())
            .and_then(|v| v.strip_prefix("Bearer "))
        else {
            return false;
        };
        let Some(given) = StatusKey::from_hex(token) else {
            return false;
        };
        given
            .0
            .iter()
            .zip(self.0.iter())
            .fold(0u8, |diff, (a, b)| diff | (a ^ b))
            == 0
    }
}

#[cfg(test)]
mod tests {
    use axum::http::HeaderValue;
    use tempfile::TempDir;

    use super::*;

    fn bearer(token: &str) -> HeaderMap {
        let mut h = HeaderMap::new();
        h.insert(
            header::AUTHORIZATION,
            HeaderValue::from_str(&format!("Bearer {token}")).unwrap(),
        );
        h
    }

    /// A written key loads back as the bytes its file spells, and its file is
    /// 64 hex digits and a newline.
    #[test]
    fn a_created_key_loads_and_is_never_overwritten() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("status.key");
        StatusKey::create(&path).unwrap();
        let text = std::fs::read_to_string(&path).unwrap();
        assert_eq!(text.len(), 65);
        assert!(text.ends_with('\n'));
        assert!(text[..64].bytes().all(|b| b.is_ascii_hexdigit()));
        let key = StatusKey::load(&path).unwrap();
        assert!(key.presented(&bearer(text.trim())));
        assert!(
            StatusKey::create(&path).is_err(),
            "never a second run over it"
        );
        assert_eq!(std::fs::read_to_string(&path).unwrap(), text);
    }

    /// The key itself is accepted; any other key, a malformed one, another
    /// scheme, or none at all is not.
    #[test]
    fn only_the_key_is_presented() {
        let key = StatusKey::from_hex(&"ab".repeat(32)).unwrap();
        assert!(key.presented(&bearer(&"ab".repeat(32))));
        assert!(
            key.presented(&bearer(&"AB".repeat(32))),
            "hex in either case"
        );
        assert!(!key.presented(&bearer(&format!("{}ac", "ab".repeat(31)))));
        assert!(!key.presented(&bearer(&"ab".repeat(31))));
        assert!(!key.presented(&bearer("not-hex")));
        let mut basic = HeaderMap::new();
        basic.insert(
            header::AUTHORIZATION,
            HeaderValue::from_str(&format!("Basic {}", "ab".repeat(32))).unwrap(),
        );
        assert!(!key.presented(&basic));
        assert!(!key.presented(&HeaderMap::new()));
    }

    #[test]
    fn a_file_that_is_not_a_key_is_refused() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("status.key");
        std::fs::write(&path, "short").unwrap();
        assert!(StatusKey::load(&path).is_err());
        assert!(StatusKey::load(&dir.path().join("absent")).is_err());
    }
}
