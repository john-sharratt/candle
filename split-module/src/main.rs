//! Split an oversized Rust file into a module folder by a plan.
//!
//! ```text
//! split-module map   <file>          # the items a plan places, one a line
//! split-module check <file> <plan>   # every output's size, nothing written
//! split-module apply <file> <plan>   # write the split; the original is kept
//! split-module docs  <folder>        # each file's items and comments, no code
//! split-module redoc <folder> <docs> # write each file's new module doc
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
mod digest;
mod layout;
mod map;
mod plan;
mod redoc;
mod rewrite;
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

fn run(args: &[String]) -> Result<String, String> {
    let (cmd, file) = match args {
        [cmd, file, ..] => (cmd.as_str(), Path::new(file)),
        _ => {
            return Err(
                "usage: split-module map|check|apply <file> [plan] | docs <folder> | redoc <folder> <docs>"
                    .into(),
            )
        }
    };
    match cmd {
        "docs" => return digest::digest_dir(file),
        "redoc" => {
            let docs_path = args.get(2).ok_or("redoc takes the docs to write")?;
            let text = fs::read_to_string(docs_path).map_err(|e| format!("{docs_path}: {e}"))?;
            let written = redoc::redoc(file, &redoc::parse_docs(&text)?)?;
            return Ok(written.iter().map(|n| format!("{n}\n")).collect());
        }
        _ => {}
    }
    let source = fs::read_to_string(file).map_err(|e| format!("{}: {e}", file.display()))?;
    let layout = layout::layout(&source)?;
    if cmd == "map" {
        return Ok(map::render(&source, &layout));
    }
    let plan_path = args.get(2).ok_or("check and apply take a plan")?;
    let plan_text = fs::read_to_string(plan_path).map_err(|e| format!("{plan_path}: {e}"))?;
    let plan = plan::parse(&plan_text)?;
    let placed = assign::assign(&layout, &plan)?;
    let edits = rewrite::edits(&source)?;
    let shares = Shares::new(&source, &edits)?;
    let outputs = assemble::assemble(&source, &layout, &plan, &placed, &edits, &shares)?;
    let targets = write::targets(file)?;
    let mut report = String::new();
    match cmd {
        "check" => {
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
            for (path, size) in write::write(file, &targets, &outputs)? {
                report.push_str(&format!("{size}\t{}\n", path.display()));
            }
            report.push_str(&format!("kept\t{}\n", targets.presplit.display()));
        }
        other => return Err(format!("unknown command `{other}`")),
    }
    Ok(report)
}
