//! A folder as the knowledge base shows it (`docs/provenance_system_collection.md`
//! §4, §12 D14): its immediate children, one a line, repository-relative, a
//! folder ending in `/`, a file followed by a space and its size in bytes.

/// The listing of a folder whose children are `children` — `(path, bytes,
/// is_dir)`, repository-relative, as `Tree::children` gives them — in the
/// order given, every line ending its `\n`.
pub fn listing_text(children: &[(String, Option<usize>, bool)]) -> String {
    let mut out = String::new();
    for (path, bytes, is_dir) in children {
        if *is_dir {
            out.push_str(path);
            out.push_str("/\n");
        } else {
            out.push_str(path);
            if let Some(bytes) = bytes {
                out.push(' ');
                out.push_str(&bytes.to_string());
            }
            out.push('\n');
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    /// **A folder's listing names each child repository-relative**, a folder
    /// with its `/`, a file with its size.
    #[test]
    fn a_folders_listing_names_each_entry_repository_relative_with_its_size() {
        let children = vec![
            ("zend/src/api".to_string(), None, true),
            ("zend/src/lib.rs".to_string(), Some(2459), false),
            ("zend/src/main.rs".to_string(), Some(88), false),
        ];
        assert_eq!(
            listing_text(&children),
            "zend/src/api/\nzend/src/lib.rs 2459\nzend/src/main.rs 88\n"
        );
        assert_eq!(listing_text(&[]), "");
    }
}
