//! The mind read from the substrate (`docs/substrate_git.md` §8).
//!
//! With `--mind-repo <name>`, the mind is a repository the cluster's agents
//! hold. Its packs and branches are records in this member's substrate, and
//! its files are read at the commit the log names for `--mind-branch`, from
//! its object database — no checkout, no mirror, no folder. The database is
//! opened once and shared: what reads the mind, what the operations' stores
//! read through, and what keeps it with the log are one database. The first
//! member to need it pulls it through an agent while it loads
//! ([`substrate_git::member::Repos`]); the rest catch it up.

use std::sync::Arc;

use candle_conversation::persistence::SharedSubstrate;
use substrate_git::Store;
use zend_vfs::{BranchName, ObjectDb, Oid, RefName, RemoteName, Repo, ORIGIN};

use crate::mind::files::MindFiles;

/// Which repository and branch the mind is.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MindSource {
    pub name: String,
    pub branch: String,
}

impl MindSource {
    /// Where the store holds the branch as the record: `origin`'s tracking
    /// ref. Refuses a branch name git would not take.
    fn record_ref(&self) -> anyhow::Result<RefName> {
        Ok(RemoteName::parse(ORIGIN)?.tracking(&BranchName::parse(&self.branch)?))
    }
}

/// The mind as this member's log holds it now: its object database brought up
/// to the log, and the files of the branch's commit — `None` while the log
/// names no such branch, or its packs have not all arrived.
pub fn from_log(
    source: &MindSource,
    shared: &SharedSubstrate,
) -> anyhow::Result<(ObjectDb, Option<Arc<MindFiles>>)> {
    let store = Store::open(Arc::new(shared.clone()), &source.name)?;
    store.sync(shared)?;
    let files = at_record(store.repo(), source)?;
    Ok((store.objects().clone(), files))
}

/// The files at the branch's commit as the store `repo` holds it as the
/// record, or `None` when it holds none.
pub fn at_record(repo: &Repo, source: &MindSource) -> anyhow::Result<Option<Arc<MindFiles>>> {
    let tip: Option<Oid> = record_tip(repo, source)?;
    tip.map(|tip| MindFiles::at_commit(repo, &tip).map(Arc::new))
        .transpose()
}

/// The branch's commit as the store `repo` holds it as the record.
pub fn record_tip(repo: &Repo, source: &MindSource) -> anyhow::Result<Option<Oid>> {
    Ok(repo.ref_target(&source.record_ref()?)?)
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_conversation::persistence::SubstratePersistence;
    use candle_conversation::substrate::Substrate;
    use std::path::Path;
    use std::process::Command;

    fn git(dir: &Path, args: &[&str]) {
        let out = Command::new("git")
            .current_dir(dir)
            .args(["-c", "user.name=t", "-c", "user.email=t@t"])
            .args(args)
            .output()
            .unwrap();
        assert!(
            out.status.success(),
            "{}",
            String::from_utf8_lossy(&out.stderr)
        );
    }

    /// **A mind adopted into the log reads from the log alone**: another
    /// member, with no checkout, finds its files at the commit — and its
    /// database names the repository it holds.
    #[test]
    fn a_mind_in_the_log_reads_with_no_checkout() {
        let tmp = tempfile::tempdir().unwrap();
        let mind = tmp.path().join("mind");
        std::fs::create_dir_all(mind.join("layers/eras")).unwrap();
        std::fs::write(mind.join("projection.yaml"), "layers: []\n").unwrap();
        std::fs::write(mind.join("layers/eras/first.md"), "# First\n").unwrap();
        git(&mind, &["init", "-q", "-b", "main"]);
        git(&mind, &["add", "-A"]);
        git(&mind, &["commit", "-q", "-m", "mind"]);

        let sub = tmp.path().join("substrate");
        std::fs::create_dir(&sub).unwrap();
        let mut substrate = Substrate::new();
        let persistence =
            SubstratePersistence::open_in_with_substrate(&sub, &mut substrate).unwrap();
        let shared = SharedSubstrate::new(substrate, persistence);
        substrate_git::adopt(&shared, "battle-mind", &Repo::open(&mind).unwrap()).unwrap();

        let source = MindSource {
            name: "battle-mind".into(),
            branch: "main".into(),
        };
        let (db, files) = from_log(&source, &shared).unwrap();
        let files = files.unwrap();
        assert_eq!(db.name(), "battle-mind");
        assert_eq!(
            files.read("projection.yaml").as_deref(),
            Some("layers: []\n")
        );
        assert!(files.is_file("layers/eras/first.md"));
        assert!(files.folder_root().is_none());

        let other = MindSource {
            name: "battle-mind".into(),
            branch: "elsewhere".into(),
        };
        assert!(from_log(&other, &shared).unwrap().1.is_none());
    }
}
