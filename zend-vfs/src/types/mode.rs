//! Tree entry modes, in git's octal spelling.

use crate::error::GitError;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum FileMode {
    Regular,
    Executable,
    Symlink,
    Submodule,
    Tree,
}

impl FileMode {
    pub fn parse(octal: &str) -> Result<Self, GitError> {
        match octal {
            "100644" => Ok(Self::Regular),
            "100755" => Ok(Self::Executable),
            "120000" => Ok(Self::Symlink),
            "160000" => Ok(Self::Submodule),
            "040000" | "40000" => Ok(Self::Tree),
            other => Err(GitError::invalid(format!("unknown file mode {other:?}"))),
        }
    }

    /// Parse a mode git printed, where `000000` means "no entry".
    pub(crate) fn parse_present(octal: &str) -> Result<Option<Self>, GitError> {
        if octal.bytes().all(|b| b == b'0') {
            return Ok(None);
        }
        Self::parse(octal).map(Some)
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Self::Regular => "100644",
            Self::Executable => "100755",
            Self::Symlink => "120000",
            Self::Submodule => "160000",
            Self::Tree => "040000",
        }
    }

    /// Whether an entry of this mode holds a blob's bytes.
    pub fn is_blob(self) -> bool {
        matches!(self, Self::Regular | Self::Executable | Self::Symlink)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_mode_round_trips_through_its_octal_form() {
        for mode in [
            FileMode::Regular,
            FileMode::Executable,
            FileMode::Symlink,
            FileMode::Submodule,
            FileMode::Tree,
        ] {
            assert_eq!(FileMode::parse(mode.as_str()).unwrap(), mode);
        }
    }

    #[test]
    fn git_prints_trees_without_a_leading_zero_in_some_places() {
        assert_eq!(FileMode::parse("40000").unwrap(), FileMode::Tree);
    }

    #[test]
    fn zeros_mean_absent_and_unknown_modes_are_refused() {
        assert_eq!(FileMode::parse_present("000000").unwrap(), None);
        assert!(FileMode::parse("100664").is_err());
    }
}
