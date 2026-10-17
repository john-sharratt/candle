//! Packs git itself makes, for the pack layer's tests to hold their reading and
//! writing to: a history whose file changes a few lines a commit, so git's
//! packs hold delta chains.

use std::io::Write;
use std::path::Path;
use std::process::{Command, Stdio};

use crate::testing::{git_in, TestRepo};
use crate::types::Oid;

/// `git <args>` in `dir`, `stdin` fed to it, its raw output.
pub(crate) fn git_bytes(dir: &Path, args: &[&str], stdin: &[u8]) -> Vec<u8> {
    let mut child = Command::new("git")
        .arg("-C")
        .arg(dir)
        .args(args)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("git runs");
    child
        .stdin
        .take()
        .expect("stdin is piped")
        .write_all(stdin)
        .expect("stdin takes the input");
    let out = child.wait_with_output().expect("git finishes");
    assert!(
        out.status.success(),
        "git {args:?}: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    out.stdout
}

/// A repository of `commits` commits, each changing a few lines of one long
/// file, adding a file in a folder, and changing a binary: enough for git to
/// write deltas against earlier versions.
pub(crate) fn history(commits: usize) -> (TestRepo, Vec<Oid>) {
    let t = TestRepo::init();
    let mut lines: Vec<String> = (0..2000)
        .map(|i| format!("line {i} of the long file\n"))
        .collect();
    let mut made = Vec::new();
    for c in 0..commits {
        lines[(c * 37) % 2000] = format!("changed in commit {c}\n");
        lines[(c * 91 + 5) % 2000] = format!("and again in {c}\n");
        t.write("long.txt", lines.concat().as_bytes());
        t.write(
            &format!("dir/file{c}.txt"),
            format!("file {c}\n").as_bytes(),
        );
        let binary: Vec<u8> = (0..4096u32)
            .map(|i| ((i * (c as u32 + 3)) % 251) as u8)
            .collect();
        t.write("bin.dat", &binary);
        made.push(t.commit_all(&format!("commit {c}")));
    }
    (t, made)
}

/// The repository's every object in one pack, as `git repack -adf` writes it,
/// and git's own index of it.
pub(crate) fn full_pack(t: &TestRepo) -> (Vec<u8>, Vec<u8>) {
    git_in(
        &t.path,
        &[
            "repack",
            "-a",
            "-d",
            "-f",
            "-q",
            "--window=50",
            "--depth=50",
        ],
    );
    let dir = t.path.join(".git").join("objects").join("pack");
    let pack = std::fs::read_dir(&dir)
        .unwrap()
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .find(|p| p.extension().is_some_and(|x| x == "pack"))
        .expect("repack wrote a pack");
    let index = pack.with_extension("idx");
    (
        std::fs::read(&pack).unwrap(),
        std::fs::read(&index).unwrap(),
    )
}

/// A thin pack of what `tip` reaches beyond `base`: deltas against objects
/// only `base` holds, named by id.
pub(crate) fn thin_pack(t: &TestRepo, tip: &Oid, base: &Oid) -> Vec<u8> {
    let revs = format!("{tip}\n^{base}\n");
    git_bytes(
        &t.path,
        &[
            "pack-objects",
            "--revs",
            "--thin",
            "--stdout",
            "-q",
            "--delta-base-offset",
        ],
        revs.as_bytes(),
    )
}
