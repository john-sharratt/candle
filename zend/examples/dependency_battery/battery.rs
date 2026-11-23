//! The battery's questions as `battery.yaml` commits them, and the checks
//! that hold them to the design (`docs/unbounded_agents.md` §9.12): the
//! category proportions, ids unique, every answer file present in the
//! repository, no expected item spelled in its own question, and each
//! question's `names_file` flag the one its text gives.

use std::collections::HashSet;
use std::path::Path;

use anyhow::{bail, Context};
use serde::Deserialize;

/// The three categories, and how many of each the battery holds.
pub const CATEGORIES: [(&str, usize); 3] =
    [("direct", 43), ("transitive", 43), ("architectural", 42)];

#[derive(Deserialize)]
pub struct Battery {
    pub questions: Vec<Question>,
}

#[derive(Deserialize, Clone, Debug, PartialEq)]
pub struct Question {
    pub id: String,
    pub category: String,
    pub question: String,
    /// What a correct answer must name; `a|b` accepts either spelling.
    pub expect: Vec<String>,
    /// The repository files that answer it, relative to the repository root.
    pub files: Vec<String>,
    /// Whether the question names one of its answer files.
    pub names_file: bool,
}

impl Battery {
    pub fn load(path: &Path) -> anyhow::Result<Self> {
        let text =
            std::fs::read_to_string(path).with_context(|| format!("reading {}", path.display()))?;
        serde_yaml::from_str(&text).context("parsing the battery")
    }
}

/// Whether `question` names `file`: its file name, or its stem where the stem
/// is distinctive enough to be a name rather than a word (a `mod.rs` or a
/// `lib.rs` is named by nothing short of its path).
pub fn names(question: &str, file: &str) -> bool {
    let q = question.to_lowercase();
    let base = file.rsplit('/').next().unwrap_or(file).to_lowercase();
    let stem = base.split('.').next().unwrap_or(&base);
    let generic = matches!(stem, "mod" | "lib" | "main" | "build");
    q.contains(&base) || (!generic && stem.len() >= 6 && q.contains(stem))
}

/// Whether `question` names any of its answer files.
pub fn names_any(question: &str, files: &[String]) -> bool {
    files.iter().any(|f| names(question, f))
}

/// Every way `battery` breaks the design, one line each; empty when it holds.
pub fn violations(battery: &Battery, repo: &Path) -> Vec<String> {
    let mut out = Vec::new();
    for (category, want) in CATEGORIES {
        let got = battery
            .questions
            .iter()
            .filter(|q| q.category == category)
            .count();
        if got != want {
            out.push(format!("{got} {category} questions, not {want}"));
        }
    }
    let mut ids = HashSet::new();
    for q in &battery.questions {
        if !ids.insert(q.id.as_str()) {
            out.push(format!("{}: id repeated", q.id));
        }
        if !CATEGORIES.iter().any(|(c, _)| *c == q.category) {
            out.push(format!("{}: no category `{}`", q.id, q.category));
        }
        if q.expect.is_empty() {
            out.push(format!("{}: expects nothing", q.id));
        }
        if q.files.is_empty() {
            out.push(format!("{}: names no answer file", q.id));
        }
        let text = q.question.to_lowercase();
        for item in &q.expect {
            for spelling in item.split('|').map(|s| s.trim().to_lowercase()) {
                if spelling.is_empty() {
                    out.push(format!("{}: an empty spelling in `{item}`", q.id));
                } else if text.contains(&spelling) {
                    out.push(format!("{}: the question spells `{spelling}`", q.id));
                }
            }
        }
        for f in &q.files {
            if !repo.join(f).is_file() {
                out.push(format!("{}: no file {f}", q.id));
            }
        }
        if q.names_file != names_any(&q.question, &q.files) {
            out.push(format!(
                "{}: names_file is {}, the text says {}",
                q.id, q.names_file, !q.names_file
            ));
        }
    }
    out
}

/// `battery`, or every way it breaks the design.
pub fn checked(battery: Battery, repo: &Path) -> anyhow::Result<Battery> {
    let v = violations(&battery, repo);
    if !v.is_empty() {
        bail!("the battery breaks its design:\n  {}", v.join("\n  "));
    }
    Ok(battery)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn repo() -> std::path::PathBuf {
        Path::new(env!("CARGO_MANIFEST_DIR")).join("..")
    }

    #[test]
    fn a_question_names_a_file_by_its_name_or_a_distinctive_stem() {
        assert!(names(
            "Where is `tier_fits` in span_geometry.rs?",
            "a/span_geometry.rs"
        ));
        assert!(names(
            "What does the span_geometry module check?",
            "a/span_geometry.rs"
        ));
        assert!(!names(
            "Where is the boundary checked?",
            "a/span_geometry.rs"
        ));
        assert!(
            !names("Which module holds it?", "a/mod.rs"),
            "`mod` is a word"
        );
        assert!(
            !names("Where does the scan run?", "a/scan.rs"),
            "a short stem is a word"
        );
    }

    /// **The committed battery holds to its design**: 43, 43 and 42 questions,
    /// every answer file in the repository, nothing it expects spelled in the
    /// question, each `names_file` as its text gives it.
    #[test]
    fn the_committed_battery_holds_to_its_design() {
        let path =
            Path::new(env!("CARGO_MANIFEST_DIR")).join("examples/dependency_battery/battery.yaml");
        let battery = Battery::load(&path).unwrap();
        let v = violations(&battery, &repo());
        assert!(v.is_empty(), "{}", v.join("\n"));
    }
}
