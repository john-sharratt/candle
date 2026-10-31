//! `journals check <dir>`: read a journal repository's working tree as the
//! service would, and list every problem in it — for a hand edit or a job's
//! refactor, before it is committed. Exits 1 when anything is wrong.
//!
//! `journals fmt <dir>`: the same read, then every file the tree is made of
//! rewritten as the service writes it — for a tree written by hand, or in an
//! older form of the files. Writes nothing when anything is wrong.

use std::collections::BTreeMap;
use std::env;
use std::fs;
use std::io;
use std::path::Path;
use std::process::ExitCode;

use journals::tree::load::load;
use journals::tree::render::render_all;
use journals::tree::Tree;

fn read_dir(root: &Path, dir: &Path, out: &mut BTreeMap<String, Vec<u8>>) -> io::Result<()> {
    for entry in fs::read_dir(dir)? {
        let entry = entry?;
        let path = entry.path();
        if entry.file_name() == ".git" {
            continue;
        }
        if entry.file_type()?.is_dir() {
            read_dir(root, &path, out)?;
        } else {
            let rel = path
                .strip_prefix(root)
                .expect("under the root")
                .components()
                .map(|c| c.as_os_str().to_string_lossy().into_owned())
                .collect::<Vec<_>>()
                .join("/");
            out.insert(rel, fs::read(&path)?);
        }
    }
    Ok(())
}

/// `bytes` at `path`, put in place whole: written beside it, then renamed
/// over it, so an interrupted write leaves the old file rather than half a
/// new one.
fn replace(path: &Path, bytes: &[u8]) -> io::Result<()> {
    let tmp = path.with_extension("yaml.tmp");
    fs::write(&tmp, bytes)?;
    fs::rename(&tmp, path)
}

/// Each file of `tree` that `files` holds otherwise, rewritten under `root`;
/// how many.
fn format(root: &Path, files: &BTreeMap<String, Vec<u8>>, tree: &Tree) -> io::Result<usize> {
    let mut changed = 0;
    for (rel, bytes) in render_all(tree) {
        if files.get(&rel) != Some(&bytes) {
            replace(&root.join(&rel), &bytes)?;
            changed += 1;
        }
    }
    Ok(changed)
}

fn main() -> ExitCode {
    let usage = "usage: journals check <dir> | journals fmt <dir>";
    let args: Vec<String> = env::args().collect();
    let [_, cmd, dir] = args.as_slice() else {
        eprintln!("{usage}");
        return ExitCode::from(2);
    };
    if cmd != "check" && cmd != "fmt" {
        eprintln!("{usage}");
        return ExitCode::from(2);
    }
    let root = Path::new(dir);
    let mut files = BTreeMap::new();
    if let Err(e) = read_dir(root, root, &mut files) {
        eprintln!("{dir}: {e}");
        return ExitCode::from(2);
    }
    let tree = load(&files);
    let journals = tree.journals();
    println!(
        "{} persons, {} groups, {} journals, {} lines",
        tree.persons.len(),
        tree.groups.len(),
        journals.len(),
        journals
            .iter()
            .map(|(_, j)| j.all_lines().count())
            .sum::<usize>()
    );
    let problems = tree.all_problems();
    if !problems.is_empty() {
        for p in &problems {
            println!("{p}");
        }
        return ExitCode::from(1);
    }
    println!("no problems");
    if cmd == "fmt" {
        match format(root, &files, &tree) {
            Ok(n) => println!("{n} files rewritten"),
            Err(e) => {
                eprintln!("{dir}: {e}");
                return ExitCode::from(2);
            }
        }
    }
    ExitCode::SUCCESS
}
