//! Workspace-relative paths as tool arguments.
//!
//! The ingest layers key everything by its path relative to the workspace
//! folder — `candle/src/lib.rs` — one key form for every repository's branches
//! and the uploads folder alike. The file tools address the same file as a
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

/// How a view shows an ingested unit's key: `(repository, path inside it)`,
/// the way the tools address it. A folder keeps its trailing `/` and a
/// repository's root reads `/`.
pub fn shown(key: &str) -> (&str, String) {
    let (repo, inner) = split(key);
    let folder = key.ends_with('/');
    let inner = match (inner, folder) {
        ("", _) => "/".to_string(),
        (inner, true) => format!("{inner}/"),
        (inner, false) => inner.to_string(),
    };
    (repo, inner)
}

#[cfg(test)]
mod tests {
    use super::{shown, split};

    #[test]
    fn a_unit_is_shown_as_its_repository_and_the_path_inside_it() {
        assert_eq!(
            shown("candle/CLAUDE.md"),
            ("candle", "CLAUDE.md".to_string())
        );
        assert_eq!(
            shown("candle/zend/src/"),
            ("candle", "zend/src/".to_string())
        );
        assert_eq!(shown("candle/"), ("candle", "/".to_string()));
    }

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
