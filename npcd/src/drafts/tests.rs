use std::path::Path;
use std::process::Command;
use std::sync::Arc;

use candle_conversation::persistence::{SharedSubstrate, SubstratePersistence};
use candle_conversation::substrate::Substrate;
use serde_json::json;
use tempfile::TempDir;

use super::records::{record, records, remove, save};
use super::*;
use crate::opstore::merge::publish::Publish;
use crate::registry::authored::WORLDS;
use crate::registry::Registry;

fn git(dir: &Path, args: &[&str]) -> String {
    let out = Command::new("git")
        .args(args)
        .current_dir(dir)
        .output()
        .unwrap();
    assert!(
        out.status.success(),
        "git {args:?}: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    String::from_utf8(out.stdout).unwrap()
}

fn configure(dir: &Path) {
    for (k, v) in [
        ("core.autocrlf", "false"),
        ("user.name", "Mind"),
        ("user.email", "mind@example.com"),
        ("commit.gpgSign", "false"),
    ] {
        git(dir, &["config", k, v]);
    }
}

const ARDH: &str = "# Ardh, as its author tells it.\nname: Ardh\nsetting: A kingdom.\n";

/// A record: `origin` holding `main` with one world, the mind folder cloned
/// from it, and the stores over the mind with a substrate to keep authors'
/// drafts in.
struct Mind {
    _dir: TempDir,
    origin: std::path::PathBuf,
    root: std::path::PathBuf,
    stores: Arc<MindStores>,
}

fn under_git() -> Mind {
    let dir = TempDir::new().unwrap();
    let base = dir.path();
    let seed = base.join("seed");
    std::fs::create_dir_all(seed.join(WORLDS)).unwrap();
    git(&seed, &["init", "-q", "--template="]);
    git(&seed, &["symbolic-ref", "HEAD", "refs/heads/main"]);
    configure(&seed);
    std::fs::write(seed.join("worlds/ardh.yaml"), ARDH).unwrap();
    git(&seed, &["add", "-A"]);
    git(&seed, &["commit", "-q", "-m", "seed"]);
    git(base, &["clone", "-q", "--bare", "seed", "origin"]);
    git(
        base,
        &["clone", "-q", "-c", "core.autocrlf=false", "origin", "mind"],
    );
    let root = base.join("mind");
    configure(&root);
    let mut substrate = Substrate::new();
    let p = SubstratePersistence::open_in_with_substrate(&base.join("substrate"), &mut substrate)
        .unwrap();
    let stores = MindStores::new(Some(&root));
    stores.set_substrate(SharedSubstrate::new(substrate, p));
    Mind {
        origin: base.join("origin"),
        root,
        stores,
        _dir: dir,
    }
}

impl Mind {
    fn author(&self, account: &str) -> Author {
        Author::open(
            &crate::mind::Mind::new(Some(self.root.clone())),
            &self.root,
            Some(&self.stores),
            account,
        )
    }

    fn worlds(&self) -> Registry {
        Registry::load("world", self.root.join(WORLDS)).unwrap()
    }
}

/// **Over a mind folder that is not a git repository, what is written is
/// what stands**: there are no drafts, and the file is on disk.
#[test]
fn a_plain_folder_is_written_as_it_stands() {
    let dir = TempDir::new().unwrap();
    let stores = MindStores::new(Some(dir.path()));
    let author = Author::open(
        &crate::mind::Mind::new(Some(dir.path().to_path_buf())),
        dir.path(),
        Some(&stores),
        "ab",
    );
    assert!(!author.keeps_drafts());
    save(&author, "world", WORLDS, "ardh", &json!({"name": "Ardh"})).unwrap();
    assert_eq!(
        std::fs::read_to_string(dir.path().join("worlds/ardh.yaml")).unwrap(),
        "name: Ardh\n"
    );
    assert!(author.drafted().is_empty());
    assert!(author.owner().is_none());
}

/// **A mind with no folder is written in the authored folder**, never the
/// process's working directory — and a mind still on its way from the
/// substrate refuses the write outright.
#[test]
fn a_mind_with_no_folder_writes_in_the_authored_folder() {
    let dir = TempDir::new().unwrap();
    let author = Author::open(&crate::mind::Mind::new(None), dir.path(), None, "ab");
    save(&author, "world", WORLDS, "ardh", &json!({"name": "Ardh"})).unwrap();
    assert!(dir.path().join("worlds/ardh.yaml").is_file());

    let pending = crate::mind::Mind::pending(dir.path());
    let author = Author::open(&pending, dir.path(), None, "ab");
    let refused = author.commit(&[("worlds/hess.yaml".into(), Some("name: Hess\n".into()))]);
    assert!(refused.is_err());
    assert!(!dir.path().join("worlds/hess.yaml").exists());
}

/// **A save over a mind under git is the author's draft**: rendered against
/// what they see, so the file's comments survive; seen by them — through a
/// view opened afresh — and by no other author; and nowhere on disk.
#[test]
fn a_save_is_the_authors_own_draft() {
    let m = under_git();
    let author = m.author("ab");
    assert!(author.keeps_drafts());
    save(
        &author,
        "world",
        WORLDS,
        "ardh",
        &json!({"name": "Ardh", "setting": "A republic."}),
    )
    .unwrap();
    save(&author, "world", WORLDS, "fen", &json!({"name": "Low Fen"})).unwrap();

    let again = m.author("ab");
    assert_eq!(
        again.read("worlds/ardh.yaml").as_deref(),
        Some("# Ardh, as its author tells it.\nname: Ardh\nsetting: A republic.\n")
    );
    let reg = m.worlds();
    assert_eq!(
        records(&again, &reg, WORLDS)
            .iter()
            .map(|r| (r.id.as_str(), r.body["name"].as_str().unwrap()))
            .collect::<Vec<_>>(),
        [("ardh", "Ardh"), ("fen", "Low Fen")]
    );
    assert_eq!(
        record(&again, &reg, WORLDS, "ardh").unwrap().body["setting"],
        "A republic."
    );

    let other = m.author("cd");
    assert_eq!(other.read("worlds/ardh.yaml").as_deref(), Some(ARDH));
    assert!(record(&other, &reg, WORLDS, "fen").is_none());
    assert_eq!(
        std::fs::read_to_string(m.root.join("worlds/ardh.yaml")).unwrap(),
        ARDH,
        "the mind folder holds only what is published"
    );
    assert!(!m.root.join("worlds/fen.yaml").exists());
}

/// **A removal is a draft too**: the record is gone from the author's view
/// and still stands for everyone else until it is published.
#[test]
fn a_removal_is_drafted() {
    let m = under_git();
    let author = m.author("ab");
    assert!(remove(&author, WORLDS, "ardh").unwrap());
    assert!(!remove(&author, WORLDS, "never").unwrap());
    let reg = m.worlds();
    assert!(record(&m.author("ab"), &reg, WORLDS, "ardh").is_none());
    assert!(records(&m.author("ab"), &reg, WORLDS).is_empty());
    assert!(record(&m.author("cd"), &reg, WORLDS, "ardh").is_some());
    assert_eq!(
        m.author("ab").drafted_in(WORLDS),
        [("ardh.yaml".to_string(), None)]
    );
}

/// **Publishing lands the drafts on the record as one commit**; following
/// the record brings the mind folder up and says it moved; and the author's
/// store, empty again, reads the record as it now stands.
#[test]
fn published_drafts_land_and_the_mind_follows() {
    let m = under_git();
    let author = m.author("ab");
    save(&author, "world", WORLDS, "fen", &json!({"name": "Low Fen"})).unwrap();

    let owner = author.owner().unwrap().clone();
    let published = m
        .stores
        .merge(&owner, |merge| {
            merge.publish("Published from the console by wren\n")
        })
        .unwrap()
        .unwrap();
    let Publish::Landed { commit, paths } = published else {
        panic!("not landed: {published:?}");
    };
    assert_eq!(paths, ["worlds/fen.yaml"]);
    assert_eq!(git(&m.origin, &["rev-parse", "main"]).trim(), commit);
    assert_eq!(
        git(&m.origin, &["show", "main:worlds/fen.yaml"]),
        "name: Low Fen\n"
    );

    let after = m.author("ab");
    assert!(after.drafted().is_empty(), "{:?}", after.drafted());
}
