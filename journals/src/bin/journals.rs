//! `journals check <dir>`: read a journal repository's working tree as the
//! service would, and list every problem in it — for a hand edit or a job's
//! refactor, before it is committed. Exits 1 when anything is wrong.
//!
//! `journals check <dir> --against <old dir> [--forgets <file>] [--budget
//! <file>]`: the same, and the change from the tree at `<old dir>` held to its
//! rules (`journals::tree::compare`) — no line lost but those `--forgets` names
//! (a line id to a line), and the shape no further than `--budget` allows
//! (`created: n`, `removed: n`, `lines_moved: n`, `summaries_rewritten: n`).
//!
//! `journals fmt <dir>`: the same read, then every file the tree is made of
//! rewritten as the service writes it — for a tree written by hand, or in an
//! older form of the files. Writes nothing when anything is wrong.
//!
//! `journals survey <dir> [--code <checkout>]`: the candidates a maintenance
//! mission starts from (`journals::tree::survey`), one per line; with
//! `--code`, a line's `source` file is looked up in that git checkout's
//! history since the line was written. Reads only; exits 0 with or without
//! candidates.

use std::collections::{BTreeMap, BTreeSet};
use std::env;
use std::fs;
use std::io;
use std::path::Path;
use std::process::{Command, ExitCode};

use journals::line::LineId;
use journals::time::Stamp;
use journals::tree::compare::{lost, shape, Budget, Shape};
use journals::tree::load::load;
use journals::tree::render::render_all;
use journals::tree::survey::survey;
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

/// The tree `dir` holds, read as the service reads it, with its files.
fn read_tree(dir: &str) -> Result<(Tree, BTreeMap<String, Vec<u8>>), String> {
    let root = Path::new(dir);
    let mut files = BTreeMap::new();
    read_dir(root, root, &mut files).map_err(|e| format!("{dir}: {e}"))?;
    Ok((load(&files), files))
}

/// What `check` holds a change to: the tree before it, the lines it forgets
/// by name, and the budget its shape keeps to.
struct Against {
    old: Tree,
    forgets: BTreeSet<LineId>,
    budget: Budget,
}

/// The `--against`, `--forgets` and `--budget` of `flags`.
fn against_of(flags: &[String]) -> Result<Option<Against>, String> {
    let mut old = None;
    let mut forgets = BTreeSet::new();
    let mut budget = Budget(Shape::default());
    let mut it = flags.iter();
    while let Some(flag) = it.next() {
        let value = it
            .next()
            .ok_or_else(|| format!("`{flag}` names a path after it"))?;
        match flag.as_str() {
            "--against" => old = Some(read_tree(value)?.0),
            "--forgets" => {
                let text = fs::read_to_string(value).map_err(|e| format!("{value}: {e}"))?;
                for line in text.lines().map(str::trim).filter(|l| !l.is_empty()) {
                    forgets.insert(LineId::parse(line.trim_start_matches('#'))?);
                }
            }
            "--budget" => {
                let text = fs::read_to_string(value).map_err(|e| format!("{value}: {e}"))?;
                budget = Budget::parse(&text).map_err(|e| format!("{value}: {e}"))?;
            }
            other => return Err(format!("`{other}` is not a flag of `check`")),
        }
    }
    match old {
        Some(old) => Ok(Some(Against {
            old,
            forgets,
            budget,
        })),
        None if flags.is_empty() => Ok(None),
        None => Err("`--forgets` and `--budget` hold a change `--against` names".into()),
    }
}

/// Whether `file` has a commit in the checkout `code` since `at`.
fn changed_in(code: &Path, file: &str, at: Stamp) -> bool {
    Command::new("git")
        .arg("-C")
        .arg(code)
        .args(["log", "-1", "--format=%H"])
        .arg(format!("--since={at}"))
        .args(["--", file])
        .output()
        .is_ok_and(|o| o.status.success() && !o.stdout.trim_ascii().is_empty())
}

/// `journals survey <dir> [--code <checkout>]`.
fn run_survey(dir: &str, flags: &[String]) -> ExitCode {
    let code = match flags {
        [] => None,
        [flag, checkout] if flag == "--code" => Some(Path::new(checkout)),
        _ => {
            eprintln!("usage: journals survey <dir> [--code <checkout>]");
            return ExitCode::from(2);
        }
    };
    let (tree, _) = match read_tree(dir) {
        Ok(t) => t,
        Err(e) => {
            eprintln!("{e}");
            return ExitCode::from(2);
        }
    };
    let changed = |file: &str, at: Stamp| code.is_some_and(|c| changed_in(c, file, at));
    let found = survey(&tree, Stamp::now(), &changed);
    for c in &found {
        println!("{c}");
    }
    println!("{} candidates", found.len());
    ExitCode::SUCCESS
}

fn main() -> ExitCode {
    let usage = "usage: journals check <dir> [--against <old dir> [--forgets <file>] [--budget <file>]] | journals fmt <dir> | journals survey <dir> [--code <checkout>]";
    let args: Vec<String> = env::args().collect();
    let [_, cmd, dir, flags @ ..] = args.as_slice() else {
        eprintln!("{usage}");
        return ExitCode::from(2);
    };
    if cmd == "survey" {
        return run_survey(dir, flags);
    }
    if (cmd != "check" && cmd != "fmt") || (cmd == "fmt" && !flags.is_empty()) {
        eprintln!("{usage}");
        return ExitCode::from(2);
    }
    let against = match against_of(flags) {
        Ok(a) => a,
        Err(e) => {
            eprintln!("{e}\n{usage}");
            return ExitCode::from(2);
        }
    };
    let (tree, files) = match read_tree(dir) {
        Ok(t) => t,
        Err(e) => {
            eprintln!("{e}");
            return ExitCode::from(2);
        }
    };
    let root = Path::new(dir);
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
    let mut problems = tree.all_problems();
    if let Some(a) = &against {
        let moved = shape(&a.old, &tree);
        println!(
            "against the old tree: {} journals created, {} removed, {} lines moved, {} summaries rewritten",
            moved.created, moved.removed, moved.lines_moved, moved.summaries_rewritten
        );
        problems.extend(lost(&a.old, &tree, &a.forgets));
        problems.extend(a.budget.over(moved));
    }
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
