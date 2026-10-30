//! `journals check <dir>`: read a journal repository's working tree as the
//! service would, and list every problem in it — for a hand edit or a job's
//! refactor, before it is committed. Exits 1 when anything is wrong.

use std::collections::BTreeMap;
use std::io;
use std::path::Path;
use std::process::ExitCode;

use journals::tree::load::load;

fn read_dir(root: &Path, dir: &Path, out: &mut BTreeMap<String, Vec<u8>>) -> io::Result<()> {
    for entry in std::fs::read_dir(dir)? {
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
            out.insert(rel, std::fs::read(&path)?);
        }
    }
    Ok(())
}

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().collect();
    let [_, cmd, dir] = args.as_slice() else {
        eprintln!("usage: journals check <dir>");
        return ExitCode::from(2);
    };
    if cmd != "check" {
        eprintln!("usage: journals check <dir>");
        return ExitCode::from(2);
    }
    let root = Path::new(dir);
    let mut files = BTreeMap::new();
    if let Err(e) = read_dir(root, root, &mut files) {
        eprintln!("{dir}: {e}");
        return ExitCode::from(2);
    }
    let tree = load(&files);
    let mut problems: Vec<String> = tree.problems.iter().map(ToString::to_string).collect();
    for p in tree.persons.values() {
        if let Err(e) = &p.file {
            problems.push(e.to_string());
        }
        problems.extend(p.problems.iter().map(ToString::to_string));
    }
    for f in tree.families.values() {
        if let Err(e) = &f.file {
            problems.push(e.to_string());
        }
        problems.extend(f.problems.iter().map(ToString::to_string));
    }
    let journals = tree.journals();
    for (_, j) in &journals {
        problems.extend(j.problems.iter().map(ToString::to_string));
    }
    println!(
        "{} persons, {} families, {} journals, {} lines",
        tree.persons.len(),
        tree.families.len(),
        journals.len(),
        journals
            .iter()
            .map(|(_, j)| j.all_lines().count())
            .sum::<usize>()
    );
    if problems.is_empty() {
        println!("no problems");
        ExitCode::SUCCESS
    } else {
        for p in &problems {
            println!("{p}");
        }
        ExitCode::from(1)
    }
}
