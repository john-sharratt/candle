//! Whether a job on the cluster was asked for (`docs/zen_everywhere.md` §5).
//!
//! **A build or a test takes a machine's memory from whatever else it is
//! doing**, so the mind starts one only for somebody who asked for one. Told
//! so in its personality, Zen still started `cargo test` on BlackWorld and on
//! WhiteWorld on idle turns and while it was answering the time in Hanoi —
//! on BlackWorld while it re-ingested a substrate. The rule is kept where the
//! act is performed: a `cluster_job_start` is refused unless what a person
//! asked, and has not been told the outcome of, asks for a job.

/// The words that ask for a job, matched as whole words, any case.
const ASKING: &[&str] = &[
    "build",
    "builds",
    "built",
    "test",
    "tests",
    "tested",
    "compile",
    "compiles",
    "cargo",
    "bench",
    "benchmark",
    "clippy",
    "fmt",
    "rebuild",
    "retest",
];

/// Whether `request` — what a person asked and is still owed — asks for a job.
pub fn asks_for_a_job(request: &str) -> bool {
    request
        .split(|c: char| !c.is_alphanumeric())
        .any(|w| ASKING.iter().any(|a| a.eq_ignore_ascii_case(w)))
}

/// What the mind is told of a job nobody asked for.
pub fn refused(request: Option<(&str, &str)>) -> String {
    match request {
        Some((party, asked)) => format!(
            "Nobody asked for a job: {party} asked \"{asked}\", and a build or a test answers \
             none of that. A job takes a machine from whatever else it is doing, so it is \
             started only for somebody who asked for one."
        ),
        None => "Nobody asked for a job: a build or a test is started only for somebody who \
                 asked for one, and a job takes a machine from whatever else it is doing."
            .to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// **A job is asked for in so many words**; a question about something
    /// else, or nothing asked at all, asks for none.
    #[test]
    fn a_job_is_asked_for_in_so_many_words() {
        for asked in [
            "run the npcd tests",
            "does candle build at the moment",
            "Can you cargo check the web crate?",
            "benchmark the scan",
            "Test it on WhiteWorld",
        ] {
            assert!(asks_for_a_job(asked), "{asked}");
        }
        for asked in [
            "what time is it in Hanoi",
            "what's 1250 times 12, plus 10%",
            "is the cluster up",
            "the contest results",
            "a rebuilt house",
        ] {
            assert!(!asks_for_a_job(asked), "{asked}");
        }
    }

    #[test]
    fn a_refusal_says_what_was_asked_instead() {
        assert_eq!(
            refused(Some(("John", "what time is it in Hanoi"))),
            "Nobody asked for a job: John asked \"what time is it in Hanoi\", and a build or a \
             test answers none of that. A job takes a machine from whatever else it is doing, \
             so it is started only for somebody who asked for one."
        );
        assert!(refused(None).starts_with("Nobody asked for a job: a build or a test"));
    }
}
