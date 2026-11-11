//! Where diaries live in the journal repository (`docs/private_mode.md`
//! §5.2): one file per diary, `diaries/<id>.bin`, named by a random id that
//! says nothing of its topic, and one salt every passphrase is stretched
//! with, `diaries/salt`.

use std::fmt;

use rand::rngs::OsRng;
use rand::TryRngCore;

/// The folder diaries are kept in, at the top of the repository.
pub const DIR: &str = "diaries";

/// The repository's one salt.
pub const SALT: &str = "diaries/salt";

/// How long the salt is, in bytes.
pub const SALT_LEN: usize = 16;

/// What every diary file's name ends with.
const EXT: &str = ".bin";

/// A diary's id: 16 random bytes, written in hex.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct DiaryId(pub [u8; 16]);

impl DiaryId {
    /// A new id, from the OS.
    pub fn random() -> DiaryId {
        let mut id = [0u8; 16];
        OsRng
            .try_fill_bytes(&mut id)
            .expect("the OS must provide entropy");
        DiaryId(id)
    }

    /// The id `hex` writes: 32 lowercase hex digits.
    pub fn parse(hex: &str) -> Option<DiaryId> {
        let lower_hex = |b: u8| b.is_ascii_digit() || (b'a'..=b'f').contains(&b);
        if hex.len() != 32 || !hex.bytes().all(lower_hex) {
            return None;
        }
        let mut id = [0u8; 16];
        for (i, byte) in id.iter_mut().enumerate() {
            *byte = u8::from_str_radix(&hex[2 * i..2 * i + 2], 16).ok()?;
        }
        Some(DiaryId(id))
    }

    /// The file it is kept in.
    pub fn path(&self) -> String {
        format!("{DIR}/{self}{EXT}")
    }

    /// The diary `path` is the file of, when it is one.
    pub fn of_path(path: &str) -> Option<DiaryId> {
        let name = path.strip_prefix(DIR)?.strip_prefix('/')?;
        DiaryId::parse(name.strip_suffix(EXT)?)
    }
}

impl fmt::Display for DiaryId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for b in self.0 {
            write!(f, "{b:02x}")?;
        }
        Ok(())
    }
}

/// Whether `path` is under [`DIR`]: a diary's, the salt, or anything else
/// there — none of it any journal's, and none of it read as one.
pub fn in_diaries(path: &str) -> bool {
    path == DIR || path.strip_prefix(DIR).is_some_and(|rest| rest.starts_with('/'))
}

/// A new salt, from the OS.
pub fn new_salt() -> [u8; SALT_LEN] {
    let mut salt = [0u8; SALT_LEN];
    OsRng
        .try_fill_bytes(&mut salt)
        .expect("the OS must provide entropy");
    salt
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_diary_is_kept_under_its_hex_id() {
        let id = DiaryId([0xab; 16]);
        assert_eq!(id.path(), format!("diaries/{}.bin", "ab".repeat(16)));
        assert_eq!(DiaryId::of_path(&id.path()), Some(id));
        assert_eq!(DiaryId::of_path("diaries/salt"), None);
        assert_eq!(DiaryId::of_path(&format!("diaries/{}.bin", "AB".repeat(16))), None);
        assert_eq!(DiaryId::of_path(&format!("groups/{}.bin", "ab".repeat(16))), None);
        assert_ne!(DiaryId::random(), DiaryId::random());
    }

    #[test]
    fn only_what_is_under_the_folder_is_in_diaries() {
        assert!(in_diaries("diaries/salt"));
        assert!(in_diaries("diaries/x.bin"));
        assert!(in_diaries("diaries"));
        assert!(!in_diaries("diariesx/salt"));
        assert!(!in_diaries("groups/zen/diaries/x"));
    }
}
