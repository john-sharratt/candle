//! A checkout's objects as git packs, the form they replicate in through the
//! substrate (`docs/substrate_git.md` §3): a checkout adopted into the log
//! packs its history with [`Repo::pack_objects`], each pack handed over with
//! git's own index of it.

use std::fs;
use std::path::Path;

use crate::error::GitError;
use crate::library::failed;
use crate::runner::utf8;
use crate::types::{Oid, RefName};
use crate::Repo;

/// Where a pack-objects run writes its packs, inside this repository's own git
/// folder, removed when the run ends.
const PACKING_DIR: &str = "zend-packing";

/// The size of a pack's header: `PACK`, the version, then the object count.
const PACK_HEADER: usize = 12;

impl Repo {
    /// Pack every object reachable from `tips` and not from `known`, handing
    /// each pack to `each` in turn with git's index of it; how many it was
    /// handed. Each is at most `max_bytes` (git's `--max-pack-size`, so a pack
    /// holding one object larger than that is larger), and only one is read
    /// into memory at a time, so a whole history is never in memory at once.
    /// None when `known` already reaches everything. A `known` commit this
    /// repository lacks is passed over.
    ///
    /// A checkout's: a checkout adopted into the substrate
    /// (`docs/substrate_git.md` §3). A repository read from the substrate packs
    /// what it made with [`ObjectDb::flush`](crate::ObjectDb::flush).
    ///
    /// The write lock is held until the last pack is handed over: the packs
    /// wait in a folder of this repository's own.
    pub fn pack_objects<E: From<GitError>>(
        &self,
        tips: &[Oid],
        known: &[Oid],
        max_bytes: u64,
        mut each: impl FnMut(Vec<u8>, Vec<u8>) -> Result<(), E>,
    ) -> Result<usize, E> {
        if tips.is_empty() {
            return Ok(0);
        }
        let mut revs = String::new();
        for tip in tips {
            revs.push_str(&format!("{tip}\n"));
        }
        for oid in known {
            if self.has_object(oid)? {
                revs.push_str(&format!("^{oid}\n"));
            }
        }
        let _write = self.write_lock();
        let dir = self.git_dir()?.join(PACKING_DIR);
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).map_err(GitError::Io)?;
        let base = dir.join("pack");
        let made = self
            .git("pack-objects")
            .args(["--revs", "-q"])
            .arg(format!("--max-pack-size={max_bytes}"))
            .arg(base.as_os_str())
            .stdin(revs.into_bytes())
            .run_ok();
        let handed = made
            .and_then(|out| utf8("pack-objects", out))
            .map_err(E::from)
            .and_then(|names| hand_over(&dir, &names, &mut each));
        let _ = fs::remove_dir_all(&dir);
        handed
    }

    /// Whether this repository holds `oid`.
    pub fn has_object(&self, oid: &Oid) -> Result<bool, GitError> {
        if let Some(lib) = self.library() {
            let id = git2::Oid::from_str(oid.as_str()).map_err(|e| failed("from_str", e))?;
            return Ok(lib.odb().map_err(|e| failed("odb", e))?.exists(id));
        }
        let out = self
            .git("cat-file")
            .args(["-e", &oid.to_string()])
            .read_only()
            .run_accepting(&[0, 1, 128])?;
        Ok(out.status == Some(0))
    }

    /// Every branch and tag, with what it points at: the refs a repository
    /// publishes.
    pub fn published_refs(&self) -> Result<Vec<(RefName, Oid)>, GitError> {
        let mut refs = self.refs_under("refs/heads/")?;
        refs.extend(self.refs_under("refs/tags/")?);
        Ok(refs)
    }

    /// Every ref under `folder` (`refs/...`, ending in `/`), with what it points
    /// at.
    pub fn refs_in(&self, folder: &str) -> Result<Vec<(RefName, Oid)>, GitError> {
        self.refs_under(folder)
    }
}

/// Read each pack `pack-objects` named in `names` out of `dir`, with its
/// index, and hand them to `each`, one pack in memory at a time; how many were
/// handed.
fn hand_over<E: From<GitError>>(
    dir: &Path,
    names: &str,
    each: &mut impl FnMut(Vec<u8>, Vec<u8>) -> Result<(), E>,
) -> Result<usize, E> {
    let mut handed = 0;
    for name in names.lines() {
        let path = dir.join(format!("pack-{}.pack", name.trim()));
        let bytes = fs::read(&path).map_err(GitError::Io)?;
        if object_count(&bytes) > 0 {
            let index = fs::read(path.with_extension("idx")).map_err(GitError::Io)?;
            each(bytes, index)?;
            handed += 1;
        }
    }
    Ok(handed)
}

/// How many objects a pack says it holds; zero for bytes too short to be one.
pub fn object_count(pack: &[u8]) -> u32 {
    if pack.len() < PACK_HEADER || &pack[..4] != b"PACK" {
        return 0;
    }
    u32::from_be_bytes([pack[8], pack[9], pack[10], pack[11]])
}

/// A pack's name: its trailing checksum, in hex.
pub fn pack_checksum(pack: &[u8]) -> Option<String> {
    // The trailer is the hash of everything before it: 20 bytes under SHA-1,
    // 32 under SHA-256. The header says neither, so the caller's format
    // decides; SHA-1 is every repository this layer holds in process.
    let len = 20;
    (pack.len() >= PACK_HEADER + len).then(|| {
        pack[pack.len() - len..]
            .iter()
            .map(|b| format!("{b:02x}"))
            .collect()
    })
}

#[cfg(test)]
mod tests {
    use std::slice::from_ref;
    use std::sync::Arc;

    use super::*;
    use crate::odb::memory_store::MemoryStore;
    use crate::odb::{ObjectDb, PackStore, DEFAULT_PACK_BUDGET};
    use crate::testing::TestRepo;

    /// Every pack `pack_objects` hands over, with its index, collected.
    fn packs(repo: &Repo, tips: &[Oid], known: &[Oid]) -> Vec<(Vec<u8>, Vec<u8>)> {
        let mut out = Vec::new();
        let handed = repo
            .pack_objects(tips, known, 64 << 20, |pack, index| {
                out.push((pack, index));
                Ok::<_, GitError>(())
            })
            .unwrap();
        assert_eq!(handed, out.len());
        out
    }

    /// **A pack carries what one repository has and another lacks**, with
    /// git's index of it, and a database that takes it reads every object.
    #[test]
    fn a_pack_carries_new_objects_to_another_repository() {
        let a = TestRepo::init();
        a.write("f", b"one\n");
        let first = a.commit_all("one");
        a.write("f", b"two\n");
        let second = a.commit_all("two");
        let repo = Repo::open(&a.path).unwrap();

        let all = packs(&repo, from_ref(&second), &[]);
        assert_eq!(all.len(), 1);
        let only_new = packs(&repo, from_ref(&second), from_ref(&first));
        assert_eq!(only_new.len(), 1);
        assert!(object_count(&only_new[0].0) < object_count(&all[0].0));

        let store: Arc<dyn PackStore> = Arc::new(MemoryStore::default());
        let db = ObjectDb::open("r", store, DEFAULT_PACK_BUDGET).unwrap();
        let other = Repo::in_substrate(&db).unwrap();
        assert!(!other.has_object(&second).unwrap());
        db.take_pack(all[0].0.clone()).unwrap();
        assert!(other.has_object(&first).unwrap());
        assert!(other.has_object(&second).unwrap());
    }

    /// **A repository read from the substrate packs nothing this way**: it
    /// has no folder for `pack-objects` to run in.
    #[test]
    fn a_repository_in_the_substrate_has_no_pack_objects() {
        let store: Arc<dyn PackStore> = Arc::new(MemoryStore::default());
        let db = ObjectDb::open("r", store, DEFAULT_PACK_BUDGET).unwrap();
        let repo = Repo::in_substrate(&db).unwrap();
        let tip = Oid::parse(&"1".repeat(40)).unwrap();
        let refused = repo.pack_objects(from_ref(&tip), &[], 1 << 20, |_, _| Ok::<_, GitError>(()));
        assert!(
            matches!(refused, Err(GitError::NoFolder { .. })),
            "{refused:?}"
        );
    }

    /// **Nothing new packs to nothing.**
    #[test]
    fn a_tip_already_known_packs_to_nothing() {
        let a = TestRepo::init();
        a.write("f", b"one\n");
        let head = a.commit_all("one");
        let repo = Repo::open(&a.path).unwrap();
        assert!(packs(&repo, from_ref(&head), from_ref(&head)).is_empty());
    }
}
