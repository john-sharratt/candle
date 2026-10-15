//! A daemon's repositories loaded through a real substrate (feature `member`).
#![cfg(feature = "member")]

use std::path::Path;
use std::process::Command;

use candle_conversation::persistence::{SharedSubstrate, SubstratePersistence};
use candle_conversation::substrate::Substrate;
use substrate_git::member::{branches, Repos, Source};

fn git(dir: &Path, args: &[&str]) -> String {
    let out = Command::new("git")
        .current_dir(dir)
        .args(["-c", "user.name=t", "-c", "user.email=t@t", "-c", "init.defaultBranch=main"])
        .args(args)
        .output()
        .expect("git");
    assert!(out.status.success(), "git {args:?}: {}", String::from_utf8_lossy(&out.stderr));
    String::from_utf8(out.stdout).unwrap().trim().to_string()
}

fn shared(dir: &Path) -> SharedSubstrate {
    let mut substrate = Substrate::new();
    let persistence = SubstratePersistence::open_in_with_substrate(dir, &mut substrate).unwrap();
    SharedSubstrate::new(substrate, persistence)
}

/// **A standalone daemon loads its checkout whole**: adopted into the
/// substrate as packs and branches, and its store holds every branch where
/// the log does — then a second daemon on the same substrate loads it from
/// the log alone, with no checkout.
#[test]
fn a_standalone_checkout_loads_whole_and_reloads_from_the_log() {
    let tmp = tempfile::tempdir().unwrap();
    let work = tmp.path().join("work");
    std::fs::create_dir(&work).unwrap();
    git(&work, &["init", "-q", "-b", "main"]);
    for i in 0..3 {
        std::fs::write(work.join("f"), format!("{i}\n")).unwrap();
        git(&work, &["add", "-A"]);
        git(&work, &["commit", "-q", "-m", &format!("c{i}")]);
    }
    let head = git(&work, &["rev-parse", "HEAD"]);
    let sub = tmp.path().join("substrate");
    std::fs::create_dir(&sub).unwrap();

    {
        let shared = shared(&sub);
        let source = [Source { name: "r".into(), checkout: Some(work.clone()) }];
        let repos = Repos::open(&tmp.path().join("git-a"), &source, shared.clone(), None).unwrap();
        repos.load(&|| false, &|_| {}).unwrap();
        let log = branches(&shared, "r").unwrap();
        assert_eq!(log.values().next().unwrap().as_str(), head);
        assert_eq!(repos.store("r").unwrap().record().unwrap(), log);
    }

    let shared = shared(&sub);
    let source = [Source { name: "r".into(), checkout: None }];
    let repos = Repos::open(&tmp.path().join("git-b"), &source, shared.clone(), None).unwrap();
    repos.load(&|| false, &|_| {}).unwrap();
    assert_eq!(
        repos.store("r").unwrap().record().unwrap(),
        branches(&shared, "r").unwrap()
    );
}
