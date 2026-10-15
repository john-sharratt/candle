//! A repository's objects as git packs, the form they replicate in through the
//! substrate (`docs/substrate_git.md` §3.1). A member packs what it made or
//! fetched ([`Repo::pack_objects`]); every member takes each pack into the
//! object store it derives from its log ([`Repo::take_pack`]).

use std::fs;
use std::path::Path;

use crate::error::GitError;
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
    /// each pack to `each` in turn; how many it was handed. Each is at most
    /// `max_bytes` (git's `--max-pack-size`, so a pack holding one object
    /// larger than that is larger), and only one is read into memory at a
    /// time, so a whole history is never in memory at once. None when
    /// `known` already reaches everything. A `known` commit this repository
    /// lacks is passed over.
    ///
    /// The write lock is held until the last pack is handed over: the packs
    /// wait in a folder of this repository's own.
    pub fn pack_objects<E: From<GitError>>(
        &self,
        tips: &[Oid],
        known: &[Oid],
        max_bytes: u64,
        mut each: impl FnMut(Vec<u8>) -> Result<(), E>,
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
        let dir = self.git_dir().join(PACKING_DIR);
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

    /// Take `pack` into this repository's object store, indexed. A pack
    /// already held is taken again harmlessly.
    pub fn take_pack(&self, pack: &[u8]) -> Result<(), GitError> {
        let _write = self.write_lock();
        self.git("index-pack")
            .arg("--stdin")
            .stdin(pack.to_vec())
            .run_ok()
            .map(|_| ())
    }

    /// Whether this repository holds `oid`.
    pub fn has_object(&self, oid: &Oid) -> Result<bool, GitError> {
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

/// Read each pack `pack-objects` named in `names` out of `dir` and hand it to
/// `each`, one in memory at a time; how many were handed.
fn hand_over<E: From<GitError>>(
    dir: &Path,
    names: &str,
    each: &mut impl FnMut(Vec<u8>) -> Result<(), E>,
) -> Result<usize, E> {
    let mut handed = 0;
    for name in names.lines() {
        let path = dir.join(format!("pack-{}.pack", name.trim()));
        let bytes = fs::read(&path).map_err(GitError::Io)?;
        if object_count(&bytes) > 0 {
            each(bytes)?;
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

    use super::*;
    use crate::testing::TestRepo;

    /// Every pack `pack_objects` hands over, collected.
    fn packs(repo: &Repo, tips: &[Oid], known: &[Oid]) -> Vec<Vec<u8>> {
        let mut out = Vec::new();
        let handed = repo
            .pack_objects(tips, known, 64 << 20, |pack| {
                out.push(pack);
                Ok::<_, GitError>(())
            })
            .unwrap();
        assert_eq!(handed, out.len());
        out
    }

    /// **A pack carries what one repository has and another lacks**, and the
    /// other reads every object of it once taken.
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
        assert!(object_count(&only_new[0]) < object_count(&all[0]));

        let b = TestRepo::init();
        let other = Repo::open(&b.path).unwrap();
        assert!(!other.has_object(&second).unwrap());
        other.take_pack(&all[0]).unwrap();
        assert!(other.has_object(&first).unwrap());
        assert!(other.has_object(&second).unwrap());
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
