//! Storing file contents as blobs, through the repository's own filters.

use crate::error::GitError;
use crate::runner::utf8;
use crate::types::{FileMode, Oid, RepoPath};
use crate::Repo;

impl Repo {
    /// Store `content` as the blob git would commit for `path`.
    ///
    /// Runs `hash-object --path`, which applies the clean filters,
    /// `core.autocrlf` and `.gitattributes` for that path: a file read from a
    /// Windows checkout holds CRLF where the committed blob holds LF, and this
    /// is what makes the two agree. A symlink's target is stored unfiltered,
    /// as git itself stores it.
    pub fn write_blob(
        &self,
        path: &RepoPath,
        content: Vec<u8>,
        mode: FileMode,
    ) -> Result<Oid, GitError> {
        let _write = self.write_lock();
        self.write_blob_locked(path, content, mode)
    }

    pub(crate) fn write_blob_locked(
        &self,
        path: &RepoPath,
        content: Vec<u8>,
        mode: FileMode,
    ) -> Result<Oid, GitError> {
        let filters = if mode == FileMode::Symlink {
            "--no-filters".to_string()
        } else {
            format!("--path={path}")
        };
        let out = self
            .git("hash-object")
            .args(["-w", "--stdin"])
            .arg(filters)
            .stdin(content)
            .run_ok()?;
        Oid::parse(utf8("hash-object", out)?.trim_end())
    }
}

#[cfg(test)]
mod tests {
    use crate::testing::TestRepo;
    use crate::types::{FileMode, RepoPath};

    /// `hello\n` is the well-known blob `ce0136…`.
    #[test]
    fn a_blob_id_is_the_standard_one() {
        let t = TestRepo::init();
        let oid = t
            .repo()
            .write_blob(
                &RepoPath::parse("x.txt").unwrap(),
                b"hello\n".to_vec(),
                FileMode::Regular,
            )
            .unwrap();
        assert_eq!(oid.as_str(), "ce013625030ba8dba906f756967f9e9ca394464a");
        assert_eq!(t.git(&["cat-file", "blob", oid.as_str()]), "hello\n");
    }

    /// **CRLF content under `core.autocrlf=true` is stored as the LF blob
    /// `git add` would store** — the same id, byte for byte.
    #[test]
    fn autocrlf_converts_like_git_add() {
        let t = TestRepo::init();
        t.git(&["config", "core.autocrlf", "true"]);
        t.write("crlf.txt", b"one\r\ntwo\r\n");
        t.git(&["add", "crlf.txt"]);
        let by_add = t.git(&["ls-files", "-s", "crlf.txt"]);
        let by_add = by_add.split(' ').nth(1).unwrap();

        let oid = t
            .repo()
            .write_blob(
                &RepoPath::parse("crlf.txt").unwrap(),
                b"one\r\ntwo\r\n".to_vec(),
                FileMode::Regular,
            )
            .unwrap();
        assert_eq!(oid.as_str(), by_add);
        assert_eq!(t.git(&["cat-file", "blob", oid.as_str()]), "one\ntwo\n");
    }

    /// `.gitattributes` binary marking stops conversion for that path.
    #[test]
    fn gitattributes_decide_per_path() {
        let t = TestRepo::init();
        t.git(&["config", "core.autocrlf", "true"]);
        t.write(".gitattributes", b"*.bin -text\n");
        t.commit_all("attrs");
        let oid = t
            .repo()
            .write_blob(
                &RepoPath::parse("raw.bin").unwrap(),
                b"a\r\nb\r\n".to_vec(),
                FileMode::Regular,
            )
            .unwrap();
        assert_eq!(t.git(&["cat-file", "blob", oid.as_str()]), "a\r\nb\r\n");
    }
}
