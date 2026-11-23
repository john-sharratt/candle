//! The direct questions, generated from the source (`docs/unbounded_agents.md`
//! §9.12, *Questions*): "where is this function defined, and which files call
//! it outside tests?", for functions chosen by rule so their selection carries
//! no judgement.
//!
//! A candidate is a `pub` function of one of [`CRATES`] whose name is defined
//! exactly once in the whole repository — so the name alone identifies it —
//! is at least [`MIN_NAME`] characters with an underscore, and is called from
//! one to [`MAX_CALLERS`] other non-test source files. A file's tests are the
//! text from its first `#[cfg(test)]` on, and a test file is one under
//! `tests/`, `benches/` or `examples/`, or named for tests. Candidates are
//! ordered by a hash of their name under the battery's seed and taken in
//! that order, at most one per defining file and [`PER_CRATE`] per crate.
//!
//! Each expected file is spelled by its file name where the repository holds
//! one file of that name, else by as much of its path as tells it apart.

use std::collections::{BTreeMap, HashMap, HashSet};
use std::path::Path;
use std::sync::LazyLock;

use regex::Regex;

use crate::battery::{names_any, Question};
use crate::stats::SplitMix64;

/// The crates a direct question may ask about.
pub const CRATES: [&str; 10] = [
    "candle-core",
    "candle-nn",
    "candle-transformers",
    "candle-conversation",
    "zend",
    "zend-tools",
    "zend-vfs",
    "cluster",
    "host",
    "web",
];
pub const MIN_NAME: usize = 10;
pub const MAX_CALLERS: usize = 3;
pub const PER_CRATE: usize = 8;

static ANY_FN: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(
        r#"(?m)^\s*(?:pub(?:\([^)]*\))?\s+)?(?:const\s+)?(?:async\s+)?(?:unsafe\s+)?(?:extern\s+"[^"]*"\s+)?fn\s+([a-z_][a-z0-9_]*)"#,
    )
    .expect("a valid pattern")
});
static PUB_FN: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r#"(?m)^\s*pub\s+(?:const\s+)?(?:async\s+)?(?:unsafe\s+)?fn\s+([a-z_][a-z0-9_]*)"#)
        .expect("a valid pattern")
});
static CALLISH: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"\b([a-z_][a-z0-9_]*)\s*(?:::\s*<[^()]*>)?\s*\(").expect("a valid pattern")
});

/// `src` without its tests: everything before its first `#[cfg(test)]`.
pub fn without_tests(src: &str) -> &str {
    src.find("#[cfg(test)]").map_or(src, |i| &src[..i])
}

/// Whether `rel`, a repository-relative path, is a test, bench or example.
pub fn is_test_path(rel: &str) -> bool {
    let name = rel.rsplit('/').next().unwrap_or(rel);
    rel.split('/')
        .any(|c| matches!(c, "tests" | "benches" | "examples"))
        || name.starts_with("test")
        || name.ends_with("_tests.rs")
        || name.ends_with("_test.rs")
}

/// Every function `src` defines.
pub fn all_fns(src: &str) -> impl Iterator<Item = &str> {
    ANY_FN
        .captures_iter(src)
        .map(|c| c.get(1).map_or("", |m| m.as_str()))
}

/// Every `pub` function `src` defines, outside its tests.
pub fn pub_fns(src: &str) -> impl Iterator<Item = &str> {
    PUB_FN
        .captures_iter(without_tests(src))
        .map(|c| c.get(1).map_or("", |m| m.as_str()))
}

/// Every name `src` calls — a name followed by its argument list, with or
/// without a turbofish — that is not its own definition.
pub fn called(src: &str) -> HashSet<&str> {
    CALLISH
        .captures_iter(src)
        .filter_map(|c| {
            let m = c.get(1)?;
            let before = src[..m.start()].trim_end();
            (!before.ends_with("fn")).then_some(m.as_str())
        })
        .collect()
}

/// One generated question's source: the function, its file, its callers.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Candidate {
    pub name: String,
    pub krate: String,
    pub def: String,
    pub callers: Vec<String>,
}

/// The candidates among `files`, each `(repository-relative path, text)`.
pub fn candidates(files: &[(String, String)]) -> Vec<Candidate> {
    let mut defined: HashMap<&str, usize> = HashMap::new();
    for (_, src) in files {
        for f in all_fns(src) {
            *defined.entry(f).or_default() += 1;
        }
    }
    let calls: Vec<(&str, HashSet<&str>)> = files
        .iter()
        .filter(|(rel, _)| !is_test_path(rel))
        .map(|(rel, src)| (rel.as_str(), called(without_tests(src))))
        .collect();
    let mut out = Vec::new();
    for (rel, src) in files {
        let krate = rel.split('/').next().unwrap_or("");
        if !CRATES.contains(&krate) || is_test_path(rel) {
            continue;
        }
        for name in pub_fns(src) {
            if name.len() < MIN_NAME || !name.contains('_') || defined.get(name) != Some(&1) {
                continue;
            }
            let callers: Vec<String> = calls
                .iter()
                .filter(|(other, c)| *other != rel.as_str() && c.contains(name))
                .map(|(other, _)| other.to_string())
                .collect();
            if (1..=MAX_CALLERS).contains(&callers.len()) {
                out.push(Candidate {
                    name: name.to_string(),
                    krate: krate.to_string(),
                    def: rel.clone(),
                    callers,
                });
            }
        }
    }
    out
}

/// The shortest trailing part of `rel` no other path of `all` ends with.
pub fn spelling(rel: &str, all: &[String]) -> String {
    let parts: Vec<&str> = rel.split('/').collect();
    for n in 1..=parts.len() {
        let tail = parts[parts.len() - n..].join("/");
        let clashes = all
            .iter()
            .filter(|p| {
                p.as_str() != rel && (p.as_str() == tail || p.ends_with(&format!("/{tail}")))
            })
            .count();
        if clashes == 0 {
            return tail;
        }
    }
    rel.to_string()
}

/// A name's place in the seeded order.
fn rank(name: &str, seed: u64) -> u64 {
    // FNV-1a folds the name into the seed; SplitMix64 spreads it.
    let mut h: u64 = 0xcbf2_9ce4_8422_2325 ^ seed;
    for b in name.bytes() {
        h = (h ^ b as u64).wrapping_mul(0x0100_0000_01b3);
    }
    SplitMix64::new(h).next_u64()
}

/// `n` questions from `candidates`, in the seeded order, at most one per
/// defining file and [`PER_CRATE`] per crate; `all` is every source path, for
/// spelling the files.
pub fn pick(mut candidates: Vec<Candidate>, n: usize, seed: u64, all: &[String]) -> Vec<Question> {
    candidates.sort_by_key(|c| (rank(&c.name, seed), c.name.clone()));
    let mut files = HashSet::new();
    let mut crates: BTreeMap<String, usize> = BTreeMap::new();
    let mut out = Vec::new();
    for c in candidates {
        if out.len() == n {
            break;
        }
        let in_crate = crates.entry(c.krate.clone()).or_default();
        if *in_crate >= PER_CRATE || !files.insert(c.def.clone()) {
            continue;
        }
        *in_crate += 1;
        let question = format!(
            "In the {} crate, where is the function `{}` defined, and which source files call it \
             outside of tests?",
            c.krate, c.name
        );
        let mut expect = vec![spelling(&c.def, all)];
        expect.extend(c.callers.iter().map(|f| spelling(f, all)));
        let mut answer_files = vec![c.def.clone()];
        answer_files.extend(c.callers.iter().cloned());
        out.push(Question {
            id: format!("d{:02}", out.len() + 1),
            category: "direct".into(),
            names_file: names_any(&question, &answer_files),
            question,
            expect,
            files: answer_files,
        });
    }
    out
}

/// Every Rust source of `repo`, `.gitignore` honoured, as `(relative path,
/// text)` in path order.
pub fn sources(repo: &Path) -> anyhow::Result<Vec<(String, String)>> {
    let mut out = Vec::new();
    for entry in ignore::WalkBuilder::new(repo).build() {
        let entry = entry?;
        let path = entry.path();
        if path.extension().and_then(|e| e.to_str()) != Some("rs") || !path.is_file() {
            continue;
        }
        let rel = path
            .strip_prefix(repo)?
            .to_string_lossy()
            .replace('\\', "/");
        if let Ok(text) = std::fs::read_to_string(path) {
            out.push((rel, text));
        }
    }
    out.sort();
    Ok(out)
}

/// The questions as `battery.yaml` lists them.
pub fn yaml(questions: &[Question]) -> String {
    let mut out = String::new();
    for q in questions {
        let quote = |s: &str| format!("\"{}\"", s.replace('"', "\\\""));
        out.push_str(&format!(
            "  - id: {}\n    category: direct\n    question: >-\n      {}\n    expect: [{}]\n    files:\n{}    names_file: {}\n\n",
            q.id,
            q.question,
            q.expect.iter().map(|e| quote(e)).collect::<Vec<_>>().join(", "),
            q.files
                .iter()
                .map(|f| format!("      - {f}\n"))
                .collect::<String>(),
            q.names_file
        ));
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn file(rel: &str, src: &str) -> (String, String) {
        (rel.to_string(), src.to_string())
    }

    #[test]
    fn tests_are_cut_and_test_paths_known() {
        assert_eq!(without_tests("a\n#[cfg(test)]\nb"), "a\n");
        assert!(is_test_path("zend/tests/x.rs"));
        assert!(is_test_path("zend/src/tests_workers.rs"));
        assert!(is_test_path("zend/examples/b/main.rs"));
        assert!(!is_test_path("zend/src/attest.rs"));
    }

    /// **A call is a name before its arguments, never its definition.**
    #[test]
    fn calls_are_told_from_definitions() {
        let c = called("pub fn alpha_beta() {}\nfn x() { alpha_beta(); gamma::<u8>(1); }");
        assert!(c.contains("alpha_beta"), "the call in x counts");
        assert!(c.contains("gamma"), "a turbofish call counts");
        let only_def = called("pub fn alpha_beta() {}");
        assert!(!only_def.contains("alpha_beta"));
    }

    /// **A candidate is unique by name and called from one to three non-test
    /// files**: a name defined twice, one called only from tests, and one
    /// called from four files are all refused.
    #[test]
    fn candidates_follow_the_rule() {
        let files = vec![
            file("zend/src/a.rs", "pub fn placement_of_x() {}\npub fn defined_twice() {}\npub fn only_tested_fn() {}\npub fn called_by_four() {}"),
            file("zend/src/b.rs", "fn defined_twice() {}\nfn f() { placement_of_x(); called_by_four(); }\n#[cfg(test)]\nfn t() { only_tested_fn(); }"),
            file("zend/src/c.rs", "fn g() { called_by_four(); }"),
            file("zend/src/d.rs", "fn g() { called_by_four(); }"),
            file("zend/src/e.rs", "fn g() { called_by_four(); }"),
            file("zend/tests/t.rs", "fn g() { only_tested_fn(); }"),
        ];
        let got = candidates(&files);
        assert_eq!(
            got,
            vec![Candidate {
                name: "placement_of_x".into(),
                krate: "zend".into(),
                def: "zend/src/a.rs".into(),
                callers: vec!["zend/src/b.rs".into()],
            }]
        );
    }

    #[test]
    fn a_file_is_spelled_by_as_little_path_as_tells_it_apart() {
        let all: Vec<String> = ["a/x/launch.rs", "a/y/launch.rs", "a/y/span_geometry.rs"]
            .iter()
            .map(|s| s.to_string())
            .collect();
        assert_eq!(spelling("a/y/span_geometry.rs", &all), "span_geometry.rs");
        assert_eq!(spelling("a/x/launch.rs", &all), "x/launch.rs");
    }

    /// **The pick is the seed's**: the same candidates and seed give the same
    /// questions, numbered in order, one per defining file.
    #[test]
    fn the_pick_is_reproducible_and_one_per_file() {
        let c = |name: &str, def: &str| Candidate {
            name: name.into(),
            krate: "zend".into(),
            def: def.into(),
            callers: vec!["zend/src/z.rs".into()],
        };
        let cands = vec![
            c("first_function", "zend/src/a.rs"),
            c("second_function", "zend/src/a.rs"),
            c("third_function", "zend/src/b.rs"),
        ];
        let all = vec![
            "zend/src/a.rs".to_string(),
            "zend/src/b.rs".into(),
            "zend/src/z.rs".into(),
        ];
        let a = pick(cands.clone(), 5, 7, &all);
        assert_eq!(a, pick(cands, 5, 7, &all));
        assert_eq!(a.len(), 2, "one per defining file");
        assert_eq!((a[0].id.as_str(), a[1].id.as_str()), ("d01", "d02"));
        assert_eq!(a[0].expect.len(), 2);
        assert!(!a[0].names_file);
    }
}
