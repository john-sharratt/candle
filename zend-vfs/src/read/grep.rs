//! Searching file contents at a revision, without checking it out.

use crate::error::GitError;
use crate::types::{RepoPath, Rev};
use crate::Repo;

/// What to search for.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GrepQuery {
    /// A POSIX extended regular expression, or a literal string when
    /// `fixed` is set.
    pub pattern: String,
    pub fixed: bool,
    pub ignore_case: bool,
    /// Limit the search to these paths (files or folders); all when empty.
    pub paths: Vec<RepoPath>,
    /// At most this many matches per file.
    pub max_per_file: Option<u32>,
}

impl GrepQuery {
    /// A case-sensitive regular-expression search of every file.
    pub fn regex(pattern: &str) -> Self {
        Self {
            pattern: pattern.to_string(),
            fixed: false,
            ignore_case: false,
            paths: Vec::new(),
            max_per_file: None,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GrepHit {
    pub path: RepoPath,
    /// 1-based.
    pub line: u32,
    /// The matching line's bytes, without its newline.
    pub text: Vec<u8>,
}

/// Parse `grep -z -n` output for a revision: `<rev>:<path>\0<line>:<text>`
/// per line. Binary files are skipped by `-I`, so no text holds a newline.
pub(crate) fn parse_grep(out: &[u8], rev_prefix: &str) -> Result<Vec<GrepHit>, GitError> {
    let bad = |what: &[u8]| GitError::malformed("grep", String::from_utf8_lossy(what).into_owned());
    let mut hits = Vec::new();
    for record in out.split(|&b| b == b'\n').filter(|r| !r.is_empty()) {
        let nul = record
            .iter()
            .position(|&b| b == 0)
            .ok_or_else(|| bad(record))?;
        let name = std::str::from_utf8(&record[..nul]).map_err(|_| bad(record))?;
        let path = name.strip_prefix(rev_prefix).ok_or_else(|| bad(record))?;
        let rest = &record[nul + 1..];
        let digits = rest.iter().take_while(|b| b.is_ascii_digit()).count();
        let line: u32 = std::str::from_utf8(&rest[..digits])
            .ok()
            .and_then(|d| d.parse().ok())
            .ok_or_else(|| bad(record))?;
        // `:` normally, NUL under some versions' `-z`.
        let text = rest.get(digits + 1..).ok_or_else(|| bad(record))?;
        hits.push(GrepHit {
            path: RepoPath::parse(path)?,
            line,
            text: text.to_vec(),
        });
    }
    Ok(hits)
}

impl Repo {
    /// Every line matching `query` in the files of `rev`. Binary files are
    /// skipped.
    pub fn grep(&self, rev: &Rev, query: &GrepQuery) -> Result<Vec<GrepHit>, GitError> {
        let spec = rev.spec();
        let mut inv = self.git("grep").args([
            "-z",
            "-n",
            "-I",
            "--no-textconv",
            "--no-color",
            "--full-name",
            if query.fixed { "-F" } else { "-E" },
        ]);
        if query.ignore_case {
            inv = inv.arg("-i");
        }
        // The pattern goes through `-e`, so it is never read as a flag, and
        // no `Rev` spelling begins with `-`. `grep` takes no
        // `--end-of-options` after `-e` before 2.30.
        inv = inv.arg("-e").arg(&query.pattern).arg(&spec);
        if !query.paths.is_empty() {
            inv = inv.arg("--").args(query.paths.iter().map(|p| p.as_str()));
        }
        // Exit 1: no match.
        let out = inv
            .read_only()
            .about_rev(spec.clone())
            .run_accepting(&[0, 1])?;
        let mut hits = parse_grep(&out.stdout, &format!("{spec}:"))?;
        // The per-file limit is applied here: `grep --max-count` needs 2.38.
        // Hits arrive grouped by file, in line order.
        if let Some(max) = query.max_per_file {
            let mut seen: Option<(RepoPath, u32)> = None;
            hits.retain(|h| {
                let count = match &seen {
                    Some((path, n)) if *path == h.path => n + 1,
                    _ => 1,
                };
                seen = Some((h.path.clone(), count));
                count <= max
            });
        }
        Ok(hits)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testing::TestRepo;
    use crate::types::BranchName;

    #[test]
    fn records_parse_from_raw_bytes() {
        let out = b"HEAD:src/a b.rs\x0012:fn main() {\nHEAD:x.rs\x003:let x = 1;\n";
        let hits = parse_grep(out, "HEAD:").unwrap();
        assert_eq!(
            hits,
            vec![
                GrepHit {
                    path: RepoPath::parse("src/a b.rs").unwrap(),
                    line: 12,
                    text: b"fn main() {".to_vec()
                },
                GrepHit {
                    path: RepoPath::parse("x.rs").unwrap(),
                    line: 3,
                    text: b"let x = 1;".to_vec()
                },
            ]
        );
    }

    fn repo_with_branch() -> TestRepo {
        let t = TestRepo::init();
        t.write("src/lib.rs", b"pub fn alpha() {}\npub fn beta() {}\n");
        t.write("docs/notes.md", b"Alpha and ALPHA\n");
        t.write("bin.dat", b"alpha\0binary");
        t.commit_all("base");
        t.git(&["checkout", "-q", "-b", "other"]);
        t.write("src/lib.rs", b"pub fn gamma() {}\n");
        t.commit_all("other");
        t.git(&["checkout", "-q", "main"]);
        t
    }

    #[test]
    fn a_revision_is_searched_without_checking_it_out() {
        let t = repo_with_branch();
        let repo = t.repo();
        let other = Rev::Branch(BranchName::parse("other").unwrap());
        let hits = repo.grep(&other, &GrepQuery::regex("fn [a-z]+")).unwrap();
        assert_eq!(
            hits,
            vec![GrepHit {
                path: RepoPath::parse("src/lib.rs").unwrap(),
                line: 1,
                text: b"pub fn gamma() {}".to_vec()
            }]
        );
        // The checkout is still on main, untouched.
        assert_eq!(
            t.read("src/lib.rs"),
            b"pub fn alpha() {}\npub fn beta() {}\n"
        );
    }

    #[test]
    fn case_fixed_strings_paths_and_limits_apply() {
        let t = repo_with_branch();
        let repo = t.repo();
        let count = |q: &GrepQuery| repo.grep(&Rev::Head, q).unwrap().len();

        assert_eq!(count(&GrepQuery::regex("alpha")), 1, "binary skipped");
        let i = GrepQuery {
            ignore_case: true,
            ..GrepQuery::regex("alpha")
        };
        assert_eq!(count(&i), 2);
        let docs = GrepQuery {
            paths: vec![RepoPath::parse("docs").unwrap()],
            ..i.clone()
        };
        let hits = repo.grep(&Rev::Head, &docs).unwrap();
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].path.as_str(), "docs/notes.md");
        let fixed = GrepQuery {
            fixed: true,
            ..GrepQuery::regex("fn alpha()")
        };
        assert_eq!(count(&fixed), 1);
        let regex_only = GrepQuery::regex("fn (alpha|beta)\\(");
        assert_eq!(count(&regex_only), 2, "alternation and escapes are ERE");
        let limited = GrepQuery {
            max_per_file: Some(1),
            ..GrepQuery::regex("pub fn")
        };
        assert_eq!(count(&limited), 1);
    }

    #[test]
    fn no_match_is_empty_and_a_pattern_starting_with_a_dash_is_a_pattern() {
        let t = repo_with_branch();
        t.write("dash.txt", b"--verbose flag\n");
        t.commit_all("dash");
        let repo = t.repo();
        assert!(repo
            .grep(&Rev::Head, &GrepQuery::regex("zzz"))
            .unwrap()
            .is_empty());
        let hits = repo
            .grep(
                &Rev::Head,
                &GrepQuery {
                    fixed: true,
                    ..GrepQuery::regex("--verbose")
                },
            )
            .unwrap();
        assert_eq!(hits.len(), 1);
    }
}
