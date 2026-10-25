//! The order a pass reads its queued files in.
//!
//! A first pass over a repository reads every file once, and that takes hours.
//! Whatever is read first is what projections can draw on first, so the files
//! that explain the codebase come before the files that exercise it: the
//! documentation, then the source, then tests, benches, examples and fixtures.
//! Within each class the walk's order stands.

/// How early a file at `path` is read: lower first.
pub fn read_rank(path: &str) -> u8 {
    let lower = path.to_ascii_lowercase();
    let exercise = [
        "/tests/",
        "/test/",
        "/benches/",
        "/examples/",
        "/fixtures/",
        "/testdata/",
    ];
    if exercise.iter().any(|d| lower.contains(d))
        || lower.ends_with("_tests.rs")
        || lower.ends_with("_test.rs")
    {
        return 2;
    }
    let explains = [".md", ".markdown", ".rst", ".txt"];
    if explains.iter().any(|e| lower.ends_with(e)) {
        return 0;
    }
    1
}

/// `jobs` in read order: by [`read_rank`] of each one's path, the walk's
/// order within a rank.
pub fn in_read_order<T>(mut jobs: Vec<T>, path: impl Fn(&T) -> &str) -> Vec<T> {
    jobs.sort_by_key(|j| read_rank(path(j)));
    jobs
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn documentation_then_source_then_what_exercises_it() {
        assert_eq!(read_rank("candle/docs/unbounded_agents.md"), 0);
        assert_eq!(read_rank("candle/README.md"), 0);
        assert_eq!(read_rank("candle/candle-nn/src/lib.rs"), 1);
        assert_eq!(read_rank("candle/Cargo.toml"), 1);
        assert_eq!(read_rank("candle/candle-nn/tests/kv.rs"), 2);
        assert_eq!(read_rank("candle/candle-core/benches/bench_main.rs"), 2);
        assert_eq!(
            read_rank("candle/candle-examples/examples/llama/main.rs"),
            2
        );
        assert_eq!(read_rank("candle/a/src/thing_tests.rs"), 2);
        // A markdown file among the tests is read with them.
        assert_eq!(read_rank("candle/candle-nn/tests/README.md"), 2);
    }

    #[test]
    fn the_walk_order_stands_within_a_rank() {
        let jobs = vec![
            "r/z/tests/a.rs",
            "r/b/src/x.rs",
            "r/docs/y.md",
            "r/a/src/w.rs",
            "r/README.md",
        ];
        assert_eq!(
            in_read_order(jobs, |p| p),
            vec![
                "r/docs/y.md",
                "r/README.md",
                "r/b/src/x.rs",
                "r/a/src/w.rs",
                "r/z/tests/a.rs"
            ]
        );
    }
}
