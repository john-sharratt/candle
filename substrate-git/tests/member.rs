//! A daemon's repositories loaded through a real substrate and an agent in
//! process (feature `member`).
#![cfg(feature = "member")]

use std::path::Path;
use std::process::Command;

use agent::checkouts::Checkouts;
use agent::client::Agents;
use agent::config::AgentConfig;
use agent::jobs::Jobs;
use candle_conversation::persistence::{SharedSubstrate, SubstratePersistence};
use candle_conversation::substrate::Substrate;
use substrate_git::member::{branches, Access, Repos};
use zend_vfs::{CommandPolicy, Priority};

fn git(dir: &Path, args: &[&str]) -> String {
    let out = Command::new("git")
        .current_dir(dir)
        .args([
            "-c",
            "user.name=t",
            "-c",
            "user.email=t@t",
            "-c",
            "init.defaultBranch=main",
        ])
        .args(args)
        .output()
        .expect("git");
    assert!(
        out.status.success(),
        "git {args:?}: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    String::from_utf8(out.stdout).unwrap().trim().to_string()
}

fn shared(dir: &Path) -> SharedSubstrate {
    let mut substrate = Substrate::new();
    let persistence = SubstratePersistence::open_in_with_substrate(dir, &mut substrate).unwrap();
    SharedSubstrate::new(substrate, persistence)
}

/// **A standalone daemon loads a repository whole through its agent**: pulled
/// through the agent's checkout into the substrate as packs and branches —
/// then a second daemon on the same substrate loads it from the log alone.
#[tokio::test(flavor = "multi_thread")]
async fn a_repository_loads_whole_through_an_agent_in_process() {
    let tmp = tempfile::tempdir().unwrap();
    let origin = tmp.path().join("origin.git");
    let work = tmp.path().join("work");
    std::fs::create_dir(&work).unwrap();
    git(
        tmp.path(),
        &[
            "init",
            "-q",
            "--bare",
            "-b",
            "main",
            origin.to_str().unwrap(),
        ],
    );
    git(&work, &["init", "-q", "-b", "main"]);
    for i in 0..3 {
        std::fs::write(work.join("f"), format!("{i}\n")).unwrap();
        git(&work, &["add", "-A"]);
        git(&work, &["commit", "-q", "-m", &format!("c{i}")]);
    }
    git(&work, &["push", "-q", origin.to_str().unwrap(), "main"]);
    let head = git(&work, &["rev-parse", "HEAD"]);

    // The agent: a clone of origin under its own folder.
    let agent_dir = tmp.path().join("agent");
    std::fs::create_dir_all(&agent_dir).unwrap();
    git(&agent_dir, &["clone", "-q", origin.to_str().unwrap(), "r"]);
    let yaml = format!(
        "jobs_dir: '{}'\nslots: 1\ncheckouts:\n  - name: r\n    dir: '{}'\n",
        agent_dir.join("jobs").display(),
        agent_dir.join("r").display(),
    );
    let cfg_file = tmp.path().join("agent.yaml");
    std::fs::write(&cfg_file, yaml).unwrap();
    let cfg = AgentConfig::load(&cfg_file).unwrap();
    let checkouts = Checkouts::open(&cfg, &CommandPolicy::allowing(Vec::<String>::new())).unwrap();
    let jobs: std::sync::Arc<Jobs> = Jobs::new(checkouts, 1, Priority::Below);

    let sub = tmp.path().join("substrate");
    std::fs::create_dir(&sub).unwrap();
    let runtime = tokio::runtime::Handle::current();
    let names = ["r".to_string()];
    let loaded = {
        let shared = shared(&sub);
        let access = Access {
            agents: Agents::in_process(jobs.clone()),
            runtime: runtime.clone(),
            replicator: None,
        };
        let root = tmp.path().join("git-a");
        let shared2 = shared.clone();
        tokio::task::spawn_blocking(move || {
            let repos = Repos::open(&root, &names, shared2, access).unwrap();
            repos.load(&|| false, &|_| {}).unwrap();
            repos.store("r").unwrap().record().unwrap()
        })
        .await
        .unwrap()
    };
    assert_eq!(loaded.values().next().unwrap().as_str(), head);

    let shared = shared(&sub);
    let log = branches(&shared, "r").unwrap();
    assert_eq!(log.values().next().unwrap().as_str(), head);
}
