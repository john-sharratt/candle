//! A conversation's uploads (`docs/replicated_substrate.md` §10.2): files in
//! its `uploads` store — the folder repository every conversation has — each
//! one a `Replace` or `ReplaceBinary` delta, so it is saved and replicated as
//! a `VfsEvent` with the rest of the conversation's file changes. The store's
//! 10 MiB cap applies, and an upload past it is refused with the cap named.
//!
//! A file's name is made safe and unique within the store; its id is a hash
//! of its path, the handle the files pane opens it by.

use sha2::{Digest, Sha256};
use zend_vfs::vfs::MAX_BYTES;
use zend_vfs::{VfsError, VfsStore};

/// Reduce an uploaded file name to a safe basename that stays inside the
/// uploads store: take the final path component (defeating `../` and
/// absolute paths), keep only `[A-Za-z0-9._-]`, collapse the rest to `_`,
/// and fall back to `upload` if nothing survives. A leading `.` is
/// prefixed with `_` so uploads are never hidden dotfiles, and trailing dots
/// go — Windows strips them on create, so `foo.exe.` cannot dodge the
/// extension gate by materialising as `foo.exe` in a checkout.
pub fn sanitize(name: &str) -> String {
    let base = name.rsplit(['/', '\\']).next().unwrap_or(name);
    let mut out: String = base
        .chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || c == '.' || c == '_' || c == '-' {
                c
            } else {
                '_'
            }
        })
        .collect();
    while out.ends_with('.') {
        out.pop();
    }
    if out.is_empty() {
        return "upload".to_string();
    }
    if out.starts_with('.') {
        out.insert(0, '_');
    }
    out
}

/// `safe` if `store` holds no such file, otherwise the first free
/// `stem-NNN.ext` (`NNN` = 001, 002, …, zero-padded so listings sort). The
/// store's cap bounds how many files it can hold, so a free name is always
/// found.
pub fn unique(store: &VfsStore, safe: &str) -> String {
    let taken = |name: &str| store.read_bytes(name).ok().flatten().is_some();
    if !taken(safe) {
        return safe.to_string();
    }
    let (stem, ext) = match safe.rfind('.') {
        Some(i) if i > 0 => (&safe[..i], &safe[i..]),
        _ => (safe, ""),
    };
    (1u32..)
        .map(|n| format!("{stem}-{n:03}{ext}"))
        .find(|cand| !taken(cand))
        .expect("an unbounded range has a free name")
}

/// The id the files pane opens an upload by: the first eight bytes of the
/// SHA-256 of its path in the uploads store.
pub fn id_of(path: &str) -> u64 {
    let digest = Sha256::digest(path.as_bytes());
    u64::from_be_bytes(digest[..8].try_into().expect("a digest is 32 bytes"))
}

/// Why an upload was not taken.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum UploadRefused {
    #[error("the conversation's files would pass their {cap} MiB limit")]
    OverCap { cap: usize },
    #[error("{0}")]
    Store(String),
}

/// Put `bytes` in `store` under a safe, unused form of `name`; the path it
/// was stored at.
pub fn store(store: &VfsStore, name: &str, bytes: Vec<u8>) -> Result<String, UploadRefused> {
    let path = unique(store, &sanitize(name));
    match store.write_bytes(&path, bytes) {
        Ok(_) => Ok(path),
        Err(VfsError::Full) => Err(UploadRefused::OverCap {
            cap: MAX_BYTES >> 20,
        }),
        Err(e) => Err(UploadRefused::Store(e.to_string())),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn keeps_safe_basenames() {
        assert_eq!(sanitize("notes.txt"), "notes.txt");
        assert_eq!(sanitize("my-file_v2.rs"), "my-file_v2.rs");
    }

    #[test]
    fn strips_path_traversal_and_separators() {
        assert_eq!(sanitize("../../etc/passwd"), "passwd");
        assert_eq!(sanitize("/abs/path/x.md"), "x.md");
        assert_eq!(sanitize("a\\b\\c.json"), "c.json");
        assert_eq!(sanitize(".."), "upload");
        assert_eq!(sanitize("."), "upload");
        assert_eq!(sanitize("foo.exe."), "foo.exe");
    }

    #[test]
    fn collapses_unsafe_chars_and_unhides_dotfiles() {
        assert_eq!(sanitize("weird name!@#.txt"), "weird_name___.txt");
        assert_eq!(sanitize(".env"), "_.env");
        assert_eq!(sanitize(""), "upload");
    }

    /// A taken name gets a counter before its extension; an extensionless
    /// one at its end.
    #[test]
    fn a_taken_name_gets_a_counter() {
        let s = VfsStore::new();
        assert_eq!(store(&s, "notes.txt", b"a".to_vec()).unwrap(), "notes.txt");
        assert_eq!(
            store(&s, "notes.txt", b"b".to_vec()).unwrap(),
            "notes-001.txt"
        );
        assert_eq!(
            store(&s, "../notes.txt", b"c".to_vec()).unwrap(),
            "notes-002.txt"
        );
        store(&s, "README", b"d".to_vec()).unwrap();
        assert_eq!(unique(&s, "README"), "README-001");
        assert_eq!(s.read_bytes("notes-001.txt").unwrap(), Some(b"b".to_vec()));
    }

    /// Binary bytes are held whole; past the cap the upload is refused with
    /// the cap named and nothing is stored.
    #[test]
    fn binary_uploads_are_held_and_the_cap_refuses() {
        let s = VfsStore::new();
        let png = vec![0x89, 0x50, 0x4e, 0x47, 0xff, 0x00, 0x0d];
        let path = store(&s, "logo.png", png.clone()).unwrap();
        assert_eq!(s.read_bytes(&path).unwrap(), Some(png));
        let big = vec![0u8; MAX_BYTES];
        let refused = store(&s, "big.bin", big).unwrap_err();
        assert_eq!(refused, UploadRefused::OverCap { cap: 10 });
        assert_eq!(
            refused.to_string(),
            "the conversation's files would pass their 10 MiB limit"
        );
        assert_eq!(s.read_bytes("big.bin").unwrap(), None);
    }

    /// An id is the path's digest, fixed byte for byte.
    #[test]
    fn an_id_is_the_paths_digest() {
        assert_eq!(id_of("notes.txt"), id_of("notes.txt"));
        assert_ne!(id_of("notes.txt"), id_of("notes-001.txt"));
        // SHA-256("a") begins ca978112ca1bbdca.
        assert_eq!(id_of("a"), 0xca97_8112_ca1b_bdca);
    }
}
