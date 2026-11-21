//! Tests for the guards: links out of the root, protected directories, and host
//! paths.
//!
//! A folder link leading outside the root is invisible and unwritable, and one
//! into a protected folder is refused, because the protection is on where a path
//! leads rather than how it is spelled. A file symlink to outside the root or to a
//! protected file is refused too. Host paths, drive-prefixed paths and alternate
//! streams reach nothing outside the workspace, every spelling of `secrets/` is
//! protected, and a repository's `.git` is protected at any depth.

use super::*;

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
pub(super) mod tests {
    use super::*;

    use std::path::Path;

    use super::super::tests_upper::tests::{listed, put};

    // ── links ────────────────────────────────────────────────────────────────

    /// Link folder `link` to `target`: a symlink on Unix, a junction on
    /// Windows, which needs no privilege to create.
    pub(in super::super) fn link_dir(target: &Path, link: &Path) {
        #[cfg(unix)]
        std::os::unix::fs::symlink(target, link).unwrap();
        #[cfg(windows)]
        {
            let out = std::process::Command::new("cmd")
                .args(["/C", "mklink", "/J"])
                .arg(link)
                .arg(target)
                .output()
                .unwrap();
            assert!(
                out.status.success(),
                "mklink /J failed: {}",
                String::from_utf8_lossy(&out.stderr)
            );
        }
    }

    /// **A linked folder leading outside the root is invisible and
    /// unwritable.** The daemon's secrets live outside every repository, so a
    /// link from a repository to them is the one way the file tools could
    /// reach them.
    #[test]
    fn a_folder_link_leading_outside_the_root_is_not_read_listed_searched_or_written() {
        let outside = tempfile::tempdir().unwrap();
        put(
            outside.path(),
            "secrets.yaml",
            "tavily_api_key: tvly-live\n",
        );
        let root = tempfile::tempdir().unwrap();
        put(root.path(), "src/lib.rs", "fn f() {}\n");
        link_dir(outside.path(), &root.path().join("escape"));

        let s = VfsStore::with_root(root.path());
        assert_eq!(s.read("escape/secrets.yaml").unwrap(), None);
        assert_eq!(s.read_bytes("escape/secrets.yaml").unwrap(), None);
        assert!(s.read_page("escape/secrets.yaml", 0).unwrap().is_none());
        assert_eq!(s.list_dir("escape").unwrap(), None);
        assert!(!listed(&s, "").contains(&"escape".to_string()));
        assert!(s.paths("escape").is_empty());
        let re = regex::Regex::new("tvly").unwrap();
        assert!(s.grep(&re, "", 10, 10).hits.is_empty());

        let _ = s.write("escape/planted.txt", "x".into());
        assert!(!outside.path().join("planted.txt").exists());
        assert!(!s.delete("escape/secrets.yaml"));
        assert!(outside.path().join("secrets.yaml").exists());
    }

    /// A link into the repository's own protected folder is refused too —
    /// the protection is on where a path leads, not how it is spelled.
    #[test]
    fn a_folder_link_into_a_protected_folder_is_refused() {
        let root = tempfile::tempdir().unwrap();
        put(
            root.path(),
            "secrets/tools.yaml",
            "tavily_api_key: tvly-live\n",
        );
        link_dir(&root.path().join("secrets"), &root.path().join("innocent"));
        let s = VfsStore::with_root(root.path());
        assert_eq!(s.read("innocent/tools.yaml").unwrap(), None);
        assert_eq!(s.read_bytes("innocent/tools.yaml").unwrap(), None);
        assert_eq!(s.list_dir("innocent").unwrap(), None);
    }

    /// **Bytes read the way text does, minus the text rules.** A binary file
    /// comes back whole; a drive-qualified path, a protected path and a
    /// folder do not come back at all.
    #[test]
    fn read_bytes_returns_any_file_and_only_a_file_under_the_root() {
        let root = tempfile::tempdir().unwrap();
        std::fs::write(root.path().join("logo.bin"), [0u8, 159, 146, 150, 255]).unwrap();
        put(root.path(), "src/lib.rs", "fn f() {}\n");
        put(root.path(), "secrets/key.yaml", "k: v\n");
        let outside = tempfile::tempdir().unwrap();
        put(outside.path(), "id_rsa", "private\n");
        let s = VfsStore::with_root(root.path());

        assert_eq!(
            s.read_bytes("logo.bin").unwrap(),
            Some(vec![0u8, 159, 146, 150, 255])
        );
        assert!(s.read("logo.bin").is_err(), "text reads still refuse it");
        assert_eq!(s.read_bytes("src").unwrap(), None);
        assert!(matches!(
            s.read_bytes("secrets/key.yaml"),
            Err(VfsError::Forbidden(_))
        ));
        let abs = outside.path().join("id_rsa");
        let spelled = abs.to_string_lossy().replace('\\', "/");
        assert_eq!(s.read_bytes(&spelled).unwrap(), None, "{spelled}");
    }

    /// A link that stays inside the root, and out of protected folders,
    /// still works.
    #[test]
    fn a_folder_link_inside_the_root_still_reads() {
        let root = tempfile::tempdir().unwrap();
        put(root.path(), "real/a.txt", "hello\n");
        link_dir(&root.path().join("real"), &root.path().join("alias"));
        let s = VfsStore::with_root(root.path());
        assert_eq!(s.read("alias/a.txt").unwrap().as_deref(), Some("hello\n"));
    }

    /// File symlinks, which Windows only creates with privilege.
    #[cfg(unix)]
    #[test]
    fn a_file_symlink_to_outside_or_to_a_protected_file_is_refused() {
        use std::os::unix::fs::symlink;
        let outside = tempfile::tempdir().unwrap();
        put(
            outside.path(),
            "secrets.yaml",
            "tavily_api_key: tvly-live\n",
        );
        let root = tempfile::tempdir().unwrap();
        put(root.path(), "secrets/tools.yaml", "key: live\n");
        put(root.path(), "ok.txt", "fine\n");
        symlink(
            outside.path().join("secrets.yaml"),
            root.path().join("notes.md"),
        )
        .unwrap();
        symlink(
            root.path().join("secrets/tools.yaml"),
            root.path().join("cfg.yaml"),
        )
        .unwrap();
        symlink(root.path().join("ok.txt"), root.path().join("ok-link.txt")).unwrap();
        let s = VfsStore::with_root(root.path());
        assert_eq!(s.read("notes.md").unwrap(), None);
        assert_eq!(s.read("cfg.yaml").unwrap(), None);
        assert_eq!(s.read("ok-link.txt").unwrap().as_deref(), Some("fine\n"));
    }

    /// **The guards hold.** A protected path is refused and left untouched,
    /// and `..` cannot climb out of the workspace — normalisation pins it to
    /// the root, so the write lands on the root's own key.
    #[test]
    fn a_store_cannot_touch_secrets_or_leave_the_root() {
        let outer = tempfile::tempdir().unwrap();
        let root = outer.path().join("ws");
        put(&root, "secrets/tools.yaml", "key: real\n");
        let s = VfsStore::with_root(&root);

        assert!(matches!(
            s.write("secrets/tools.yaml", "key: planted\n".into()),
            Err(VfsError::Forbidden(_))
        ));
        assert!(!s.delete("secrets/tools.yaml"));
        assert_eq!(
            std::fs::read_to_string(root.join("secrets/tools.yaml")).unwrap(),
            "key: real\n"
        );

        s.write("../../escaped.txt", "x".into()).unwrap();
        assert!(!outer.path().join("escaped.txt").exists());
        assert_eq!(s.read("escaped.txt").unwrap().as_deref(), Some("x"));
    }

    /// **A host path is not a workspace path.** An absolute path to a file
    /// outside the workspace — on Windows a drive-prefixed one, which a join
    /// would have taken in place of the root — reads nothing and writes nothing
    /// there, and neither does a drive-relative path or an alternate stream.
    #[test]
    fn a_host_path_cannot_reach_outside_the_workspace() {
        let outer = tempfile::tempdir().unwrap();
        let root = outer.path().join("ws");
        std::fs::create_dir_all(&root).unwrap();
        put(outer.path(), "outside.txt", "host secret\n");
        let outside = outer.path().join("outside.txt");
        let planted = outer.path().join("planted.txt");

        let overlay = VfsStore::with_root(&root);
        assert_eq!(
            overlay.read(&outside.to_string_lossy()).unwrap(),
            None,
            "an absolute host path reads nothing"
        );
        assert!(listed(&overlay, &outer.path().to_string_lossy()).is_empty());
        let _ = overlay.write(&planted.to_string_lossy(), "x".into());
        assert!(!planted.exists(), "a write left the workspace");
        for path in [
            "C:x.txt",
            "C:/x.txt",
            r"\\?\C:\x.txt",
            "notes.txt:hidden",
            "SECRET~1/tools.yaml",
            "docs~2/x.md",
        ] {
            assert!(
                !matches!(overlay.read(path), Ok(Some(_))),
                "{path:?} read through to the host"
            );
        }
    }

    /// **Another spelling of `secrets/` is still `secrets/`.** Windows opens a
    /// name case-insensitively and drops trailing dots and spaces, so each of
    /// these reaches the protected directory there and is refused everywhere.
    #[test]
    fn every_spelling_of_the_protected_directory_is_protected() {
        let outer = tempfile::tempdir().unwrap();
        let root = outer.path().join("ws");
        put(&root, "secrets/tools.yaml", "key: real\n");
        let s = VfsStore::with_root(&root);
        for path in [
            "Secrets/tools.yaml",
            "SECRETS/tools.yaml",
            "secrets./tools.yaml",
            "secrets /tools.yaml",
        ] {
            assert!(
                VfsStore::is_protected(&VfsStore::normalize(path)),
                "{path:?}"
            );
            assert!(s.write(path, "key: planted\n".into()).is_err(), "{path:?}");
            assert!(
                !matches!(s.read(path), Ok(Some(_))),
                "{path:?} read the protected file"
            );
        }
        assert_eq!(
            std::fs::read_to_string(root.join("secrets/tools.yaml")).unwrap(),
            "key: real\n"
        );
        assert!(!VfsStore::is_protected("docs/secretsauce.md"));
    }

    /// **A repository's `.git` is protected, at any depth and by any spelling.**
    ///
    /// Two independent reasons, both measured. `.git/config` holds a remote's
    /// URL with its credentials intact — the git layer redacts exactly that, so
    /// leaving this readable would make the redaction decorative. And on a live
    /// turn the model bypassed `git_refs`, hand-parsed `.git/packed-refs`, and
    /// answered 53 branches and 23 tags where the truth was 56 and 24.
    #[test]
    fn a_repositorys_git_database_is_protected() {
        let outer = tempfile::tempdir().unwrap();
        let root = outer.path().join("ws");
        put(
            &root,
            ".git/config",
            "[remote \"origin\"]\n\turl = https://u:ghp_secrettoken@example.com/a.git\n",
        );
        put(&root, ".git/packed-refs", "abc refs/heads/main\n");
        put(&root, "src/lib.rs", "pub fn ok() {}\n");
        let s = VfsStore::with_root(&root);

        for path in [
            ".git/config",
            ".git/packed-refs",
            ".git/refs/heads/main",
            ".GIT/config",
            ".git./config",
            "nested/repo/.git/config",
        ] {
            assert!(
                VfsStore::is_protected(&VfsStore::normalize(path)),
                "{path:?} is not protected"
            );
            assert!(
                !matches!(s.read(path), Ok(Some(_))),
                "{path:?} was readable"
            );
            assert!(
                s.write(path, "planted".into()).is_err(),
                "{path:?} was written"
            );
        }

        // The token never reaches a caller by any route.
        assert!(s.read(".git/config").is_err());
        let listed = s.list_dir("").unwrap().unwrap_or_default();
        assert!(
            !format!("{listed:?}").contains(".git"),
            "a listing named the git database: {listed:?}"
        );
        let re = regex::Regex::new("ghp_secrettoken").unwrap();
        let hits = s.grep(&re, "", 50, 20);
        assert!(
            !format!("{hits:?}").contains("ghp_secrettoken"),
            "grep reached into .git: {hits:?}"
        );

        // A file merely *called* something gitish is untouched.
        assert!(!VfsStore::is_protected("docs/.gitignore"));
        assert!(!VfsStore::is_protected("src/gitmodules.rs"));
        assert!(matches!(s.read("src/lib.rs"), Ok(Some(_))));
    }
}
