//! Full object ids, and the hash a repository uses.

use std::fmt;

use crate::error::GitError;

/// The hash function a repository's object ids come from.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ObjectFormat {
    Sha1,
    Sha256,
}

impl ObjectFormat {
    /// Hex characters in one of this format's ids.
    pub fn hex_len(self) -> usize {
        match self {
            Self::Sha1 => 40,
            Self::Sha256 => 64,
        }
    }

    /// Parse `rev-parse --show-object-format` output.
    pub fn parse(name: &str) -> Result<Self, GitError> {
        match name {
            "sha1" => Ok(Self::Sha1),
            "sha256" => Ok(Self::Sha256),
            other => Err(GitError::invalid(format!(
                "unknown object format {other:?}"
            ))),
        }
    }
}

/// A full, lowercase object id. Abbreviated ids are refused: git is always
/// asked for full ones, and an abbreviation can become ambiguous as a
/// repository grows.
#[derive(Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Oid(String);

impl Oid {
    pub fn parse(hex: &str) -> Result<Self, GitError> {
        let len_ok = hex.len() == 40 || hex.len() == 64;
        let hex_ok = hex
            .bytes()
            .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b));
        if !len_ok || !hex_ok {
            return Err(GitError::invalid(format!(
                "{hex:?} is not a full lowercase object id"
            )));
        }
        Ok(Self(hex.to_string()))
    }

    /// Parse an id git printed, where all zeros means "no object".
    pub(crate) fn parse_nonzero(hex: &str) -> Result<Option<Self>, GitError> {
        let oid = Self::parse(hex)?;
        Ok((!oid.is_zero()).then_some(oid))
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }

    pub fn format(&self) -> ObjectFormat {
        if self.0.len() == 40 {
            ObjectFormat::Sha1
        } else {
            ObjectFormat::Sha256
        }
    }

    /// The all-zeros id git uses for "absent".
    pub fn is_zero(&self) -> bool {
        self.0.bytes().all(|b| b == b'0')
    }
}

impl fmt::Display for Oid {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl fmt::Debug for Oid {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Oid({})", self.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const EMPTY_TREE: &str = "4b825dc642cb6eb9a060e54bf8d69288fbee4904";

    #[test]
    fn a_full_sha1_id_parses() {
        let oid = Oid::parse(EMPTY_TREE).unwrap();
        assert_eq!(oid.as_str(), EMPTY_TREE);
        assert_eq!(oid.format(), ObjectFormat::Sha1);
        assert!(!oid.is_zero());
    }

    #[test]
    fn a_full_sha256_id_parses() {
        let hex = "6ef19b41225c5369f1c104d45d8d85efa9b057b53b14b4b9b939dd74decc5321";
        assert_eq!(Oid::parse(hex).unwrap().format(), ObjectFormat::Sha256);
    }

    #[test]
    fn abbreviated_uppercase_and_non_hex_ids_are_refused() {
        for bad in [
            "4b825dc",
            "4B825DC642CB6EB9A060E54BF8D69288FBEE4904",
            "4b825dc642cb6eb9a060e54bf8d69288fbee490g",
            "-b825dc642cb6eb9a060e54bf8d69288fbee4904",
            "",
        ] {
            assert!(Oid::parse(bad).is_err(), "{bad:?} should be refused");
        }
    }

    #[test]
    fn the_zero_id_reads_as_absent() {
        let zero = "0".repeat(40);
        assert!(Oid::parse(&zero).unwrap().is_zero());
        assert_eq!(Oid::parse_nonzero(&zero).unwrap(), None);
        assert!(Oid::parse_nonzero(EMPTY_TREE).unwrap().is_some());
    }

    #[test]
    fn object_formats_parse_and_size() {
        assert_eq!(ObjectFormat::parse("sha1").unwrap().hex_len(), 40);
        assert_eq!(ObjectFormat::parse("sha256").unwrap().hex_len(), 64);
        assert!(ObjectFormat::parse("md5").is_err());
    }
}
