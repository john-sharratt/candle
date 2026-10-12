//! The UI directories a build's tenants serve, as one artifact (§3.9): members
//! read their UI from disk, not from the binary, so a build travels with it.
//!
//! The archive is `[8] magic · list<entry>`, an entry `str dir · str file ·
//! blob bytes` in the wire codec's primitives, entries in `(dir, file)` order
//! with `/` between a path's parts. Nothing else goes in — no times, no
//! permissions — so packing one tree twice gives one archive, and a kept
//! build's UI packs again to the SHA-256 it was offered under.

use std::fs;
use std::io;
use std::path::{Component, Path, PathBuf};

use crate::wire::{Reader, Writer};

const MAGIC: &[u8; 8] = b"TKUI\0\0\0\x01";

/// One file of the archive.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Entry {
    /// The UI directory it belongs to, as the tenant names it relative to its
    /// working directory: `zend/web`.
    pub dir: String,
    /// Its path inside that directory: `lib/dom.js`.
    pub file: String,
    pub bytes: Vec<u8>,
}

fn invalid(what: impl Into<String>) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, what.into())
}

/// Whether `path` stays where it is put: relative, with no `..` and no root.
fn contained(path: &str) -> bool {
    !path.is_empty()
        && Path::new(path)
            .components()
            .all(|c| matches!(c, Component::Normal(_)))
}

fn walk(root: &Path, at: &Path, out: &mut Vec<(String, PathBuf)>) -> io::Result<()> {
    for entry in fs::read_dir(at)? {
        let entry = entry?;
        let kind = entry.file_type()?;
        let path = entry.path();
        if kind.is_dir() {
            walk(root, &path, out)?;
        } else if kind.is_file() {
            let rel = path
                .strip_prefix(root)
                .map_err(|_| invalid("a file outside its directory"))?;
            let parts: Vec<String> = rel
                .components()
                .map(|c| c.as_os_str().to_string_lossy().into_owned())
                .collect();
            out.push((parts.join("/"), path));
        }
        // A link is not followed: a UI directory holds files.
    }
    Ok(())
}

/// Every file under `dirs`, each `(name, path on disk)`.
pub fn read(dirs: &[(String, PathBuf)]) -> io::Result<Vec<Entry>> {
    let mut entries = Vec::new();
    for (name, path) in dirs {
        if !contained(name) {
            return Err(invalid(format!("{name:?} is not a relative directory")));
        }
        let mut files = Vec::new();
        walk(path, path, &mut files)?;
        for (file, on_disk) in files {
            entries.push(Entry {
                dir: name.clone(),
                file,
                bytes: fs::read(on_disk)?,
            });
        }
    }
    Ok(entries)
}

/// The archive of `entries`.
pub fn pack(mut entries: Vec<Entry>) -> Vec<u8> {
    entries.sort_by(|a, b| (&a.dir, &a.file).cmp(&(&b.dir, &b.file)));
    let mut w = Writer::new();
    w.fixed(MAGIC).list(&entries, |w, e| {
        w.str(&e.dir).str(&e.file).blob(&e.bytes);
    });
    w.into_bytes()
}

/// The entries of an archive, refused when it is not one or names a path that
/// would leave its directory.
pub fn entries(archive: &[u8]) -> io::Result<Vec<Entry>> {
    let mut r = Reader::new(archive, "UI archive");
    let magic: [u8; 8] = r.fixed().map_err(|e| invalid(e.to_string()))?;
    if &magic != MAGIC {
        return Err(invalid("not a UI archive"));
    }
    let entries = r
        .list(|r| {
            Ok(Entry {
                dir: r.str()?,
                file: r.str()?,
                bytes: r.blob_vec()?,
            })
        })
        .map_err(|e| invalid(e.to_string()))?;
    r.finish().map_err(|e| invalid(e.to_string()))?;
    for e in &entries {
        if !contained(&e.dir) || !contained(&e.file) {
            return Err(invalid(format!(
                "{}/{} would leave its directory",
                e.dir, e.file
            )));
        }
    }
    Ok(entries)
}

/// The directories an archive holds, each once, in order.
pub fn dirs(archive: &[u8]) -> io::Result<Vec<String>> {
    let mut dirs: Vec<String> = entries(archive)?.into_iter().map(|e| e.dir).collect();
    dirs.sort();
    dirs.dedup();
    Ok(dirs)
}

/// Write an archive's files under `root`, each directory at `root/<dir>`,
/// which must not exist yet.
pub fn unpack(archive: &[u8], root: &Path) -> io::Result<()> {
    let entries = entries(archive)?;
    for dir in dirs(archive)? {
        let at = root.join(&dir);
        if at.exists() {
            return Err(io::Error::new(
                io::ErrorKind::AlreadyExists,
                format!("{} is in the way", at.display()),
            ));
        }
    }
    for e in entries {
        let to = root.join(&e.dir).join(&e.file);
        if let Some(parent) = to.parent() {
            fs::create_dir_all(parent)?;
        }
        fs::write(to, e.bytes)?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    use tempfile::TempDir;

    fn tree() -> TempDir {
        let t = TempDir::new().unwrap();
        fs::create_dir_all(t.path().join("zend/web/lib")).unwrap();
        fs::write(t.path().join("zend/web/index.html"), b"<p>").unwrap();
        fs::write(t.path().join("zend/web/lib/a.js"), b"x").unwrap();
        fs::create_dir_all(t.path().join("common")).unwrap();
        fs::write(t.path().join("common/c.css"), b"").unwrap();
        t
    }

    fn named(t: &TempDir) -> Vec<(String, PathBuf)> {
        vec![
            ("zend/web".into(), t.path().join("zend/web")),
            ("common".into(), t.path().join("common")),
        ]
    }

    /// The archive is exact bytes: the magic, then the entries in
    /// `(dir, file)` order whatever order the disk listed them in.
    #[test]
    fn a_tree_packs_to_exact_bytes() {
        let t = tree();
        let archive = pack(read(&named(&t)).unwrap());
        let mut want = b"TKUI\0\0\0\x01".to_vec();
        want.extend(3u32.to_le_bytes());
        for (dir, file, bytes) in [
            ("common", "c.css", &b""[..]),
            ("zend/web", "index.html", &b"<p>"[..]),
            ("zend/web", "lib/a.js", &b"x"[..]),
        ] {
            for s in [dir.as_bytes(), file.as_bytes(), bytes] {
                want.extend((s.len() as u32).to_le_bytes());
                want.extend(s);
            }
        }
        assert_eq!(archive, want);
        assert_eq!(dirs(&archive).unwrap(), ["common", "zend/web"]);
    }

    /// Unpacked elsewhere, the tree is the same files, and packs again to the
    /// same archive.
    #[test]
    fn an_archive_unpacks_to_the_tree_it_came_from() {
        let t = tree();
        let archive = pack(read(&named(&t)).unwrap());
        let out = TempDir::new().unwrap();
        unpack(&archive, out.path()).unwrap();
        assert_eq!(
            fs::read(out.path().join("zend/web/lib/a.js")).unwrap(),
            b"x"
        );
        let again = pack(read(&named(&out)).unwrap());
        assert_eq!(again, archive);
        // A directory already there is never written into.
        assert_eq!(
            unpack(&archive, out.path()).unwrap_err().kind(),
            io::ErrorKind::AlreadyExists
        );
    }

    /// A path that would leave its directory, or bytes that are not an
    /// archive, are refused before anything is written.
    #[test]
    fn a_path_out_of_its_directory_is_refused() {
        let escape = pack(vec![Entry {
            dir: "zend/web".into(),
            file: "../../evil.js".into(),
            bytes: vec![],
        }]);
        let out = TempDir::new().unwrap();
        assert!(unpack(&escape, out.path()).is_err());
        assert!(!out.path().join("zend").exists());
        assert!(entries(b"not an archive").is_err());
        assert!(read(&[("/abs".into(), out.path().to_path_buf())]).is_err());
    }
}
