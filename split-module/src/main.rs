//! Split an oversized Rust file into a module folder by a plan.
//!
//! ```text
//! split-module map   <file>          # the items a plan places, one a line
//! split-module check <file> <plan>   # every output's size, nothing written
//! split-module apply <file> <plan>   # write the split; the original is kept
//! split-module docs  <folder> [plan] # each file's items and comments, no code
//! split-module redoc <folder> <docs> # write each file's new module doc
//! split-module tidy  <folder>        # take empty modules out of its mod.rs
//! ```
//!
//! A split is planned from the map and documented from the digest: once the
//! files exist, each one's module doc is written from the comments in it
//! (`digest`) and put in place (`redoc`), touching nothing but that doc.
//!
//! The map is what a planner reads in place of the file; the plan says which
//! file each item goes to (`plan`). Applying moves every item's own bytes —
//! nothing is retyped — so the cost of a split is the map and the plan, not
//! the file. What the moved code needs to build again (an import, a field's
//! visibility) is repaired from the compiler's errors afterwards; the original
//! stays beside the new folder as `<name>.presplit` until that is done.

mod assemble;
mod assign;
mod cuda;
mod digest;
mod layout;
mod map;
mod plan;
mod redoc;
mod rewrite;
mod tidy;
mod write;

use std::fs;
use std::path::Path;
use std::process::ExitCode;

use assemble::Shares;

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    match run(&args) {
        Ok(out) => {
            print!("{out}");
            ExitCode::SUCCESS
        }
        Err(e) => {
            eprintln!("split-module: {e}");
            ExitCode::FAILURE
        }
    }
}

/// `map`, `check` and `apply` for a CUDA or C++ kernel source.
fn run_kernel(
    cmd: &str,
    file: &Path,
    source: &str,
    plan_path: Option<&String>,
) -> Result<String, String> {
    let layout = cuda::layout::layout(source)?;
    let (dir, _, stem) = cuda::split::targets(file)?;
    if cmd == "map" {
        return Ok(cuda::layout::render(source, &layout));
    }
    let plan_path = plan_path.ok_or("check and apply take a plan")?;
    let plan_text = fs::read_to_string(plan_path).map_err(|e| format!("{plan_path}: {e}"))?;
    let plan = plan::parse(&plan_text)?;
    let outputs = cuda::split::split(source, &layout, &plan, &stem)?;
    let mut report = String::new();
    match cmd {
        "check" => {
            report.push_str(&format!(
                "{} parts; the parts put back give the source\n{}\t{}\n",
                outputs.parts.len(),
                outputs.file.len(),
                file.display()
            ));
            for (name, text) in &outputs.parts {
                report.push_str(&format!(
                    "{}\t{}\n",
                    text.len(),
                    dir.join(format!("{name}.cuh")).display()
                ));
            }
        }
        "apply" => {
            for (path, size) in cuda::split::write(file, &outputs)? {
                report.push_str(&format!("{size}\t{}\n", path.display()));
            }
        }
        other => return Err(format!("unknown command `{other}`")),
    }
    Ok(report)
}

fn run(args: &[String]) -> Result<String, String> {
    let (cmd, file) = match args {
        [cmd, file, ..] => (cmd.as_str(), Path::new(file)),
        _ => {
            return Err(
                "usage: split-module map|check|apply <file> [plan] | docs|tidy <folder> | redoc <folder> <docs>"
                    .into(),
            )
        }
    };
    match cmd {
        "docs" => {
            // With the split's plan, only the files it made.
            let only = match args.get(2) {
                Some(plan_path) => {
                    let text =
                        fs::read_to_string(plan_path).map_err(|e| format!("{plan_path}: {e}"))?;
                    Some(
                        plan::parse(&text)?
                            .into_iter()
                            .map(|f| f.name)
                            .filter(|n| n != plan::MOD)
                            .collect::<Vec<_>>(),
                    )
                }
                None => None,
            };
            return digest::digest_dir(file, only.as_deref());
        }
        "tidy" => return Ok(format!("{}\n", tidy::tidy(file)?)),
        "redoc" => {
            let docs_path = args.get(2).ok_or("redoc takes the docs to write")?;
            let text = fs::read_to_string(docs_path).map_err(|e| format!("{docs_path}: {e}"))?;
            let written = redoc::redoc(file, &redoc::parse_docs(&text)?)?;
            return Ok(written.iter().map(|n| format!("{n}\n")).collect());
        }
        _ => {}
    }
    let source = fs::read_to_string(file).map_err(|e| format!("{}: {e}", file.display()))?;
    if cuda::is_kernel(file) {
        return run_kernel(cmd, file, &source, args.get(2));
    }
    let layout = layout::layout(&source)?;
    let targets = write::targets(file)?;
    if cmd == "map" {
        let mut out = String::new();
        // The new files go beside the modules the folder already holds: their
        // names are taken.
        let mut taken = write::taken(&targets)?;
        taken.extend(write::used_names(&source));
        taken.sort();
        taken.dedup();
        if !taken.is_empty() {
            out.push_str(&format!(
                "# names taken, never use them: {}\n",
                taken.join(", ")
            ));
        }
        out.push_str(&map::render(&source, &layout));
        return Ok(out);
    }
    let plan_path = args.get(2).ok_or("check and apply take a plan")?;
    let plan_text = fs::read_to_string(plan_path).map_err(|e| format!("{plan_path}: {e}"))?;
    let plan = plan::parse(&plan_text)?;
    let placed = assign::assign(&layout, &plan)?;
    // A file-less module inside an inline one is `<name>.rs` below the
    // split's folder, or `<name>/mod.rs` when that is the one there.
    let module_file = |below: &str| {
        let folder = format!("{below}/mod.rs");
        if targets.dir.join(&folder).exists() {
            folder
        } else {
            format!("{below}.rs")
        }
    };
    let edits = rewrite::edits(&source, &module_file)?;
    let shares = Shares::new(&source, &edits)?;
    let mut outputs = assemble::assemble(&source, &layout, &plan, &placed, &edits, &shares)?;
    outputs.mod_rs = write::rebase_includes(file, &write::rebase_paths(file, &outputs.mod_rs));
    for (_, text) in outputs.children.iter_mut() {
        *text = write::rebase_includes(file, text);
    }
    let mut report = String::new();
    match cmd {
        "check" => {
            write::refuse_collisions(&targets, &outputs, &source)?;
            report.push_str(&format!(
                "{} files; every id placed once\n",
                outputs.children.len() + 1
            ));
            report.push_str(&format!(
                "{}\t{}\n",
                outputs.mod_rs.len(),
                targets.mod_rs.display()
            ));
            for (name, text) in &outputs.children {
                report.push_str(&format!(
                    "{}\t{}\n",
                    text.len(),
                    targets.dir.join(format!("{name}.rs")).display()
                ));
            }
        }
        "apply" => {
            for (path, size) in write::write(file, &targets, &outputs, &source)? {
                report.push_str(&format!("{size}\t{}\n", path.display()));
            }
            report.push_str(&format!("kept\t{}\n", targets.presplit.display()));
        }
        other => return Err(format!("unknown command `{other}`")),
    }
    Ok(report)
}
