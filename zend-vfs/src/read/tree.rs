//! Tree listings, from `ls-tree -z`.

use crate::error::GitError;
use crate::types::{FileMode, Oid, RepoPath, Rev};
use crate::Repo;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ObjectKind {
    Blob,
    Tree,
    /// A submodule's commit.
    Commit,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TreeEntry {
    pub mode: FileMode,
    pub kind: ObjectKind,
    pub oid: Oid,
    /// Relative to the repository root.
    pub path: RepoPath,
}

/// A tree entry with its blob's size — `None` for a folder or a submodule.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SizedEntry {
    pub entry: TreeEntry,
    pub size: Option<u64>,
}

/// Parse `ls-tree -z -l` output: `<mode> <type> <oid> <size>\t<path>\0`, the
/// size right-aligned in spaces and `-` for anything but a blob.
pub(crate) fn parse_ls_tree_sized(out: &[u8]) -> Result<Vec<SizedEntry>, GitError> {
    let text =
        std::str::from_utf8(out).map_err(|e| GitError::malformed("ls-tree", e.to_string()))?;
    text.split('\0')
        .filter(|r| !r.is_empty())
        .map(|record| {
            let bad = || GitError::malformed("ls-tree", record.to_string());
            let (meta, path) = record.split_once('\t').ok_or_else(bad)?;
            let parts: Vec<&str> = meta.split_whitespace().collect();
            let [mode, kind, oid, size] = parts[..] else {
                return Err(bad());
            };
            let kind = match kind {
                "blob" => ObjectKind::Blob,
                "tree" => ObjectKind::Tree,
                "commit" => ObjectKind::Commit,
                _ => return Err(bad()),
            };
            let size = match size {
                "-" => None,
                n => Some(n.parse().map_err(|_| bad())?),
            };
            Ok(SizedEntry {
                entry: TreeEntry {
                    mode: FileMode::parse(mode)?,
                    kind,
                    oid: Oid::parse(oid)?,
                    path: RepoPath::parse(path)?,
                },
                size,
            })
        })
        .collect()
}

/// Parse `ls-tree -z --full-tree` output: `<mode> <type> <oid>\t<path>\0`.
pub(crate) fn parse_ls_tree(out: &[u8]) -> Result<Vec<TreeEntry>, GitError> {
    let text =
        std::str::from_utf8(out).map_err(|e| GitError::malformed("ls-tree", e.to_string()))?;
    text.split('\0')
        .filter(|r| !r.is_empty())
        .map(|record| {
            let bad = || GitError::malformed("ls-tree", record.to_string());
            let (meta, path) = record.split_once('\t').ok_or_else(bad)?;
            let parts: Vec<&str> = meta.split(' ').collect();
            let [mode, kind, oid] = parts[..] else {
                return Err(bad());
            };
            let kind = match kind {
                "blob" => ObjectKind::Blob,
                "tree" => ObjectKind::Tree,
                "commit" => ObjectKind::Commit,
                _ => return Err(bad()),
            };
            Ok(TreeEntry {
                mode: FileMode::parse(mode)?,
                kind,
                oid: Oid::parse(oid)?,
                path: RepoPath::parse(path)?,
            })
        })
        .collect()
}

impl Repo {
    /// The entries directly inside `dir` at `rev` — the root when `dir` is
    /// `None`. Paths are relative to the repository root.
    pub fn ls_tree(&self, rev: &Rev, dir: Option<&RepoPath>) -> Result<Vec<TreeEntry>, GitError> {
        let mut inv = self
            .git("ls-tree")
            .args(["-z", "--full-tree", "--end-of-options"])
            .arg(rev.spec());
        if let Some(dir) = dir {
            inv = inv.arg("--").arg(format!("{dir}/"));
        }
        let out = inv.read_only().about_rev(rev.spec()).run_ok()?;
        parse_ls_tree(&out)
    }

    /// Every entry of the tree `tree`, at any depth — folders included, each
    /// before what it holds — with every blob's size. One process for the
    /// whole tree.
    pub fn ls_tree_all(&self, tree: &Oid) -> Result<Vec<SizedEntry>, GitError> {
        let out = self
            .git("ls-tree")
            .args(["-r", "-t", "-l", "-z", "--full-tree", "--end-of-options"])
            .arg(tree.as_str())
            .read_only()
            .about_rev(tree.as_str())
            .run_ok()?;
        parse_ls_tree_sized(&out)
    }

    /// The entries at exactly `paths` in `rev` — a file's blob or a folder's
    /// tree. Paths `rev` does not hold are absent from the result.
    pub fn tree_entries(&self, rev: &Rev, paths: &[&RepoPath]) -> Result<Vec<TreeEntry>, GitError> {
        if paths.is_empty() {
            return Ok(Vec::new());
        }
        // `-t`: a folder named alongside a path inside it is listed too — without
        // it, `ls-tree` descends into the folder to reach the inner path and
        // never shows the folder itself.
        let out = self
            .git("ls-tree")
            .args(["-z", "-t", "--full-tree", "--end-of-options"])
            .arg(rev.spec())
            .arg("--")
            .args(paths.iter().map(|p| p.as_str()))
            .read_only()
            .about_rev(rev.spec())
            .run_ok()?;
        let mut entries = parse_ls_tree(&out)?;
        // A pathspec naming a folder lists the folder itself; one naming a file
        // inside a listed folder is already exact. Keep exact matches only.
        entries.retain(|e| paths.iter().any(|p| **p == e.path));
        entries.dedup_by(|a, b| a.path == b.path);
        Ok(entries)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testing::TestRepo;

    #[test]
    fn records_parse_from_raw_bytes() {
        let raw = b"100644 blob ce013625030ba8dba906f756967f9e9ca394464a\ta file.txt\0\
040000 tree 4b825dc642cb6eb9a060e54bf8d69288fbee4904\tsrc\0\
160000 commit ce013625030ba8dba906f756967f9e9ca394464a\tvendor/lib\0";
        let e = parse_ls_tree(raw).unwrap();
        assert_eq!(e.len(), 3);
        assert_eq!(e[0].path.as_str(), "a file.txt");
        assert_eq!((e[1].mode, e[1].kind), (FileMode::Tree, ObjectKind::Tree));
        assert_eq!(
            (e[2].mode, e[2].kind),
            (FileMode::Submodule, ObjectKind::Commit)
        );
    }

    #[test]
    fn sized_records_parse_from_raw_bytes() {
        let raw = b"100644 blob ce013625030ba8dba906f756967f9e9ca394464a       6\ta file.txt\0\
040000 tree 4b825dc642cb6eb9a060e54bf8d69288fbee4904       -\tsrc\0\
100755 blob ce013625030ba8dba906f756967f9e9ca394464a 1234567\tsrc/run.sh\0";
        let e = parse_ls_tree_sized(raw).unwrap();
        assert_eq!(
            e.iter()
                .map(|s| (s.entry.path.as_str(), s.entry.kind, s.size))
                .collect::<Vec<_>>(),
            vec![
                ("a file.txt", ObjectKind::Blob, Some(6)),
                ("src", ObjectKind::Tree, None),
                ("src/run.sh", ObjectKind::Blob, Some(1_234_567)),
            ]
        );
        assert_eq!(e[2].entry.mode, FileMode::Executable);
        assert!(parse_ls_tree_sized(b"100644 blob x\tno-size\0").is_err());
    }

    /// **The whole tree in one listing** — folders before what they hold, and
    /// every blob with its size.
    #[test]
    fn a_whole_tree_lists_every_depth_with_sizes() {
        let t = TestRepo::init();
        t.write("top.txt", b"hello\n");
        t.write("src/deep/x.rs", b"x\n");
        t.commit_all("base");
        let repo = t.repo();
        let tree = Oid::parse(t.git(&["rev-parse", "HEAD^{tree}"]).trim()).unwrap();
        let all: Vec<(String, Option<u64>)> = repo
            .ls_tree_all(&tree)
            .unwrap()
            .into_iter()
            .map(|s| (s.entry.path.to_string(), s.size))
            .collect();
        assert_eq!(
            all,
            vec![
                ("src".to_string(), None),
                ("src/deep".to_string(), None),
                ("src/deep/x.rs".to_string(), Some(2)),
                ("top.txt".to_string(), Some(6)),
            ]
        );
    }

    #[test]
    fn a_listing_shows_one_level_and_exact_lookups_find_files_and_folders() {
        let t = TestRepo::init();
        t.write("top.txt", b"hello\n");
        t.write("src/lib.rs", b"lib\n");
        t.write("src/deep/x.rs", b"x\n");
        t.commit_all("base");
        let repo = t.repo();

        let root: Vec<String> = repo
            .ls_tree(&Rev::Head, None)
            .unwrap()
            .into_iter()
            .map(|e| e.path.to_string())
            .collect();
        assert_eq!(root, vec!["src", "top.txt"]);

        let src = RepoPath::parse("src").unwrap();
        let inside: Vec<String> = repo
            .ls_tree(&Rev::Head, Some(&src))
            .unwrap()
            .into_iter()
            .map(|e| e.path.to_string())
            .collect();
        assert_eq!(inside, vec!["src/deep", "src/lib.rs"]);

        let top = RepoPath::parse("top.txt").unwrap();
        let deep = RepoPath::parse("src/deep/x.rs").unwrap();
        let missing = RepoPath::parse("nope.txt").unwrap();
        let found = repo
            .tree_entries(&Rev::Head, &[&top, &deep, &missing, &src])
            .unwrap();
        let names: Vec<(&str, ObjectKind)> =
            found.iter().map(|e| (e.path.as_str(), e.kind)).collect();
        assert_eq!(
            names,
            vec![
                ("src", ObjectKind::Tree),
                ("src/deep/x.rs", ObjectKind::Blob),
                ("top.txt", ObjectKind::Blob)
            ]
        );
        // "hello\n" — the well-known blob id.
        assert_eq!(
            found[2].oid.as_str(),
            "ce013625030ba8dba906f756967f9e9ca394464a"
        );
    }

    /// A path that looks like pathspec magic names that file literally.
    #[test]
    fn pathspec_magic_is_read_literally() {
        let t = TestRepo::init();
        t.write("a.txt", b"a\n");
        t.commit_all("base");
        let magic = RepoPath::parse(":(glob)*").unwrap();
        assert!(t
            .repo()
            .tree_entries(&Rev::Head, &[&magic])
            .unwrap()
            .is_empty());
    }
}
