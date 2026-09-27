//! Workspace-relative paths as tool arguments.
//!
//! The ingest layers key everything by its path relative to the workspace
//! folder — `candle/src/lib.rs` — because that is what the walk, the watcher and
//! the resume cache all see on disk. The file tools address the same file as a
//! repository plus a path inside it: `repo: candle, path: src/lib.rs`. A
//! repository's folder is its name ([`zend_vfs::workspace`]), so the
//! first segment of a key IS its repository and the conversion is a split.

/// `key`'s repository and its path inside that repository. The inner path is
/// empty for the repository's own root (`candle` or `candle/`), and both halves
/// are empty for the workspace root itself (`""`).
pub fn split(key: &str) -> (&str, &str) {
    let key = key.trim_matches('/');
    key.split_once('/').unwrap_or((key, ""))
}

#[cfg(test)]
mod tests {
    use super::split;

    #[test]
    fn a_file_splits_at_its_first_segment() {
        assert_eq!(split("candle/src/lib.rs"), ("candle", "src/lib.rs"));
        assert_eq!(split("mind/README.md"), ("mind", "README.md"));
    }

    #[test]
    fn a_repository_root_has_an_empty_inner_path() {
        assert_eq!(split("candle"), ("candle", ""));
        assert_eq!(split("candle/"), ("candle", ""));
    }

    #[test]
    fn a_folder_key_drops_its_trailing_slash() {
        assert_eq!(split("candle/zend/src/"), ("candle", "zend/src"));
    }

    #[test]
    fn the_workspace_root_is_two_empty_halves() {
        assert_eq!(split(""), ("", ""));
        assert_eq!(split("/"), ("", ""));
    }
}
