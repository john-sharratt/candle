//! A finished task's citation audit (`docs/mcp_guides.md` §4): each URL its
//! final response cites, against what the tools actually returned to the
//! worker — and to the parts it started — in their turns.
//!
//! | Mark | Meaning |
//! |---|---|
//! | opened | a `web_fetch` of it returned the page: the URL it asked for, or the one it reached |
//! | seen | appeared only inside a tool's result — a search hit, a link on a page — never fetched |
//! | never seen | cited, but no tool result held it |
//!
//! Only what tools returned counts: the `<tool_response>` blocks of each
//! turn's user half. A URL named in a prompt — Claude's, or the correction
//! that sends a worker back — is not evidence of anything, and a fetch that
//! failed returned no page. Deterministic and cheap. A task whose answer cites
//! anything not opened is sent back once to open it ([`Audit::correction`]).

use std::collections::{BTreeMap, BTreeSet};
use std::sync::LazyLock;

use regex::Regex;
use serde_json::Value;

use crate::fast_path::coverage::response_bodies;

/// A URL as it appears in text: up to the first character that ends one.
/// Parentheses are kept — a path may hold them — and an unbalanced closing
/// one is taken off by [`trimmed`].
static URL: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r#"https?://[^\s<>"'`\[\]{}|\\^]+"#).expect("a valid pattern"));

/// What a turn's user half holds, to the audit.
pub type Turn<'a> = &'a str;

/// What a final response cites, sorted by what the run did with each.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Audit {
    pub opened: Vec<String>,
    /// Cited under an address that differs only in its separators from a page
    /// the run opened — `2026-10-01/` for `2026/10/01/`: a slip in copying,
    /// each with the address that was opened.
    pub miscopied: Vec<(String, String)>,
    pub seen: Vec<String>,
    pub never_seen: Vec<String>,
}

/// The address two copyings of one page share however their word separators
/// slip: the path with every `-`, `_` and `/` read as one separator, in lower
/// case — so `2026-10-01/` and `2026/10/01/` meet, while pages that differ in
/// any other character, or in how their digits group, do not.
fn loose(url: &str) -> String {
    let lowered = normal(url).to_lowercase();
    let mut out = String::with_capacity(lowered.len());
    for c in lowered.chars() {
        let c = if matches!(c, '-' | '_') { '/' } else { c };
        if !(c == '/' && out.ends_with('/')) {
            out.push(c);
        }
    }
    out
}

/// A URL with the punctuation a sentence ends it with taken off, and a
/// closing parenthesis that closes the sentence's, not the URL's.
fn trimmed(url: &str) -> &str {
    const END: [char; 8] = ['.', ',', ';', ':', '!', '?', '*', '_'];
    let mut u = url.trim_end_matches(END);
    while u.ends_with(')') && u.matches('(').count() < u.matches(')').count() {
        u = u[..u.len() - 1].trim_end_matches(END);
    }
    u
}

/// The URL as two citations of one page compare: no scheme, no `www.`, no
/// fragment, no trailing `/`, the host in lower case.
pub fn normal(url: &str) -> String {
    let url = trimmed(url);
    let rest = url
        .strip_prefix("https://")
        .or_else(|| url.strip_prefix("http://"))
        .unwrap_or(url);
    let rest = rest.strip_prefix("www.").unwrap_or(rest);
    let rest = rest.split('#').next().unwrap_or(rest);
    let rest = rest.trim_end_matches('/');
    match rest.split_once('/') {
        Some((host, path)) => format!("{}/{path}", host.to_ascii_lowercase()),
        None => rest.to_ascii_lowercase(),
    }
}

/// Every URL in `text`, as written, each once, in order.
pub fn urls_in(text: &str) -> Vec<String> {
    let mut seen = BTreeSet::new();
    URL.find_iter(text)
        .map(|m| trimmed(m.as_str()).to_string())
        .filter(|u| seen.insert(normal(u)))
        .collect()
}

/// What a tool result is, to the audit.
enum Returned {
    /// A page came back — the URLs asked for and reached.
    Page(Vec<String>),
    /// A part's `sub_run` answer: a model's words, not a tool's finding.
    Part,
    /// Anything else a tool returned.
    Other,
}

/// What the tool result `body` is. A page came back when a `web_fetch`
/// returned one — only a fetched page carries a `final_url` — or an
/// `http_request` answered it with a success status; a failed fetch is an
/// error with neither.
fn classify(body: &str) -> Returned {
    let Ok(v) = serde_json::from_str::<Value>(body.trim()) else {
        return Returned::Other;
    };
    if ["path", "state", "result"]
        .iter()
        .all(|k| v.get(k).is_some())
    {
        return Returned::Part;
    }
    let Some(reached) = v.get("final_url").and_then(Value::as_str) else {
        return Returned::Other;
    };
    if v.get("status")
        .and_then(Value::as_u64)
        .is_some_and(|s| !(200..300).contains(&s))
    {
        return Returned::Other;
    }
    let mut out = vec![reached.to_string()];
    out.extend(v.get("url").and_then(Value::as_str).map(str::to_string));
    Returned::Page(out)
}

impl Audit {
    /// The audit of `answer` against the user halves of a run's turns.
    pub fn of(answer: &str, turns: &[Turn<'_>]) -> Self {
        let mut opened = BTreeSet::new();
        // Each opened page by its loose address, with the address it was
        // reached at — what a miscopied citation should have said.
        let mut opened_loosely: BTreeMap<String, String> = BTreeMap::new();
        let mut touched = BTreeSet::new();
        for body in turns.iter().flat_map(|user| response_bodies(user)) {
            match classify(body) {
                // The part's own turns are audited with the worker's; what it
                // says it found is not evidence.
                Returned::Part => continue,
                Returned::Page(urls) => {
                    for url in urls {
                        opened.insert(normal(&url));
                        opened_loosely
                            .entry(loose(&url))
                            .or_insert_with(|| trimmed(&url).to_string());
                    }
                }
                Returned::Other => {}
            }
            for url in urls_in(body) {
                touched.insert(normal(&url));
            }
        }
        let mut audit = Self::default();
        for cited in urls_in(answer) {
            let key = normal(&cited);
            if opened.contains(&key) {
                audit.opened.push(cited);
            } else if let Some(was) = opened_loosely.get(&loose(&cited)) {
                audit.miscopied.push((cited, was.clone()));
            } else if touched.contains(&key) {
                audit.seen.push(cited);
            } else {
                audit.never_seen.push(cited);
            }
        }
        audit
    }

    /// Whether the answer cites anything.
    pub fn cites(&self) -> bool {
        !(self.opened.is_empty()
            && self.miscopied.is_empty()
            && self.seen.is_empty()
            && self.never_seen.is_empty())
    }

    /// Whether it cites anything not by an address it opened.
    pub fn unopened(&self) -> bool {
        !(self.miscopied.is_empty() && self.seen.is_empty() && self.never_seen.is_empty())
    }

    /// The audit as a task's answer carries it.
    pub fn render(&self) -> String {
        let mut out = format!(
            "Citation audit: {} cited — {} opened, {} seen only in tool results, {} never seen \
             in the run.",
            self.opened.len() + self.miscopied.len() + self.seen.len() + self.never_seen.len(),
            self.opened.len() + self.miscopied.len(),
            self.seen.len(),
            self.never_seen.len()
        );
        if !self.miscopied.is_empty() {
            out.push_str(&format!(
                "\nOpened, but cited under a miscopied address: {}",
                self.miscopied
                    .iter()
                    .map(|(cited, was)| format!("{cited} (opened as {was})"))
                    .collect::<Vec<_>>()
                    .join(", ")
            ));
        }
        if !self.seen.is_empty() {
            out.push_str(&format!(
                "\nSeen only in tool results — a search hit, a link, an error — never opened: {}",
                self.seen.join(", ")
            ));
        }
        if !self.never_seen.is_empty() {
            out.push_str(&format!(
                "\nNever seen in the run (inferred or invented): {}",
                self.never_seen.join(", ")
            ));
        }
        out
    }

    /// The message that sends the worker back to open what it cited, and to
    /// cite what it opened by the address it opened it at.
    pub fn correction(&self) -> String {
        let mut parts = Vec::new();
        if !self.miscopied.is_empty() {
            parts.push(format!(
                "Your answer cites pages you opened under addresses that differ from the ones \
                 you opened them at: {}. Cite each by the exact address you opened — do not \
                 fetch the miscopied one, which may not exist.",
                self.miscopied
                    .iter()
                    .map(|(cited, was)| format!("{cited} → {was}"))
                    .collect::<Vec<_>>()
                    .join(", ")
            ));
        }
        let unopened: Vec<&str> = self
            .seen
            .iter()
            .chain(&self.never_seen)
            .map(String::as_str)
            .collect();
        if !unopened.is_empty() {
            parts.push(format!(
                "Your answer cites these without having opened them: {}. Fetch each one with \
                 web_fetch and check the claim it supports. Correct your answer where the page \
                 says otherwise; remove a citation whose page does not exist; mark UNVERIFIED \
                 anything you still cannot confirm.",
                unopened.join(", ")
            ));
        }
        format!(
            "{}\n\nAnswer again, whole, in the same shape as before, with a CHANGES section saying \
             what you corrected and why.",
            parts.join("\n\n")
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn response(body: &str) -> String {
        format!("<tool_response>{body}</tool_response>\n")
    }

    #[test]
    fn two_citations_of_one_page_compare_equal() {
        assert_eq!(
            normal("https://www.Example.com/a/b/#top"),
            "example.com/a/b"
        );
        assert_eq!(normal("http://example.com/a/b."), "example.com/a/b");
        assert_eq!(normal("https://EXAMPLE.com"), "example.com");
    }

    #[test]
    fn urls_are_found_once_without_the_punctuation_around_them() {
        assert_eq!(
            urls_in("See https://a.com/x. Also (https://b.org/y) and https://a.com/x/ again."),
            ["https://a.com/x", "https://b.org/y"]
        );
    }

    /// **A path's own parentheses stay**: only one that closes the
    /// sentence's is taken off, and markdown's emphasis around a URL too.
    #[test]
    fn a_urls_own_parentheses_are_kept() {
        assert_eq!(
            urls_in(
                "Mercury (https://en.wikipedia.org/wiki/Mercury_(planet)), [x](https://a.com/y), \
                 __https://b.com/z__."
            ),
            [
                "https://en.wikipedia.org/wiki/Mercury_(planet)",
                "https://a.com/y",
                "https://b.com/z"
            ]
        );
    }

    /// **Only a page that came back is opened**: an `http_request` that
    /// answered an error opened nothing, and a part's `sub_run` answer is its
    /// words, not a finding — what it cites is not even seen.
    #[test]
    fn a_failed_request_and_a_parts_answer_are_not_evidence() {
        let failed =
            response(r#"{"status":404,"headers":{},"body":"","final_url":"https://x.com/paper"}"#);
        let ok =
            response(r#"{"status":200,"headers":{},"body":"…","final_url":"https://x.com/data"}"#);
        let part = response(
            r#"{"path":"T:r/v/read","state":"done","result":"see https://made-up.example/report"}"#,
        );
        let turns = [failed.as_str(), ok.as_str(), part.as_str()];
        assert_eq!(
            Audit::of(
                "https://x.com/paper https://x.com/data https://made-up.example/report",
                &turns
            ),
            Audit {
                opened: vec!["https://x.com/data".into()],
                miscopied: Vec::new(),
                seen: vec!["https://x.com/paper".into()],
                never_seen: vec!["https://made-up.example/report".into()],
            }
        );
    }

    /// **Each citation is marked by what the tools returned**: a fetched page
    /// is opened — by the URL asked or the one reached — a URL inside any
    /// result is seen, and anything else was never seen.
    #[test]
    fn each_citation_is_marked_by_what_the_tools_returned() {
        let fetched_page = response(
            r#"{"url":"https://redirect.example/x","final_url":"https://inception.ai/mercury-3","title":"Mercury","content":"…","truncated":false,"images":0}"#,
        );
        let wiki = response(
            r#"{"url":"https://en.wikipedia.org/wiki/Mercury_(planet)","final_url":"https://en.wikipedia.org/wiki/Mercury_(planet)","title":"Mercury","content":"","truncated":false,"images":0}"#,
        );
        let search = response(
            "[2/2 web_search]\n1. Kimi K3.1 — https://news.example.com/kimi 2. https://llm-stats.com/models",
        );
        let turns = [
            "research the releases; start at https://hf.co/moonshot",
            fetched_page.as_str(),
            wiki.as_str(),
            search.as_str(),
        ];
        let answer = "Mercury (https://inception.ai/mercury-3/), via https://redirect.example/x, \
                      the planet (https://en.wikipedia.org/wiki/Mercury_(planet)), scores per \
                      https://llm-stats.com/models, weights at https://hf.co/moonshot.";
        assert_eq!(
            Audit::of(answer, &turns),
            Audit {
                opened: vec![
                    "https://inception.ai/mercury-3/".into(),
                    "https://redirect.example/x".into(),
                    "https://en.wikipedia.org/wiki/Mercury_(planet)".into(),
                ],
                miscopied: Vec::new(),
                seen: vec!["https://llm-stats.com/models".into()],
                never_seen: vec!["https://hf.co/moonshot".into()],
            },
            "a URL named in a prompt is not evidence"
        );
    }

    /// **A page opened but cited under a miscopied address is said as such**
    /// — the address it was opened at given — and the correction asks for
    /// that address, never to fetch the miscopied one.
    #[test]
    fn a_miscopied_citation_of_an_opened_page_is_named_with_its_real_address() {
        let page = response(
            r#"{"url":"https://blog.rust-lang.org/2026/10/01/Rust-1.99.0/","final_url":"https://blog.rust-lang.org/2026/10/01/Rust-1.99.0/","title":"Rust 1.99.0","content":"","truncated":false,"images":0}"#,
        );
        let a = Audit::of(
            "See https://blog.rust-lang.org/2026-10-01/Rust-1.99.0/ and https://other.org/x.",
            &[page.as_str()],
        );
        assert_eq!(
            a,
            Audit {
                miscopied: vec![(
                    "https://blog.rust-lang.org/2026-10-01/Rust-1.99.0/".into(),
                    "https://blog.rust-lang.org/2026/10/01/Rust-1.99.0/".into()
                )],
                never_seen: vec!["https://other.org/x".into()],
                ..Audit::default()
            }
        );
        assert_eq!(
            a.render(),
            "Citation audit: 2 cited — 1 opened, 0 seen only in tool results, 1 never seen in the \
             run.\nOpened, but cited under a miscopied address: \
             https://blog.rust-lang.org/2026-10-01/Rust-1.99.0/ (opened as \
             https://blog.rust-lang.org/2026/10/01/Rust-1.99.0/)\nNever seen in the run (inferred \
             or invented): https://other.org/x"
        );
        assert!(a.correction().starts_with(
            "Your answer cites pages you opened under addresses that differ from the ones you \
             opened them at: https://blog.rust-lang.org/2026-10-01/Rust-1.99.0/ → \
             https://blog.rust-lang.org/2026/10/01/Rust-1.99.0/."
        ));
        // Only word separators slip: other characters, and how digits group,
        // tell pages apart.
        assert_ne!(
            loose("https://en.wikipedia.org/wiki/東京"),
            loose("https://en.wikipedia.org/wiki/大阪")
        );
        assert_ne!(loose("https://x.com/a/1/23"), loose("https://x.com/a/12/3"));
        assert_eq!(loose("https://x.com/a_b/c"), loose("https://x.com/a-b-c/"));
    }

    /// **A fetch that failed opened nothing**, and the correction that names
    /// a URL does not make it seen: an invented citation stays invented after
    /// the worker is sent back.
    #[test]
    fn a_failed_fetch_and_the_correction_open_nothing() {
        let first = Audit {
            never_seen: vec!["https://hf.co/moonshot".into()],
            ..Audit::default()
        };
        let correction = first.correction();
        let failed = response("[1/1 web_fetch]\nHTTP error 404: https://hf.co/moonshot not found");
        let turns = [correction.as_str(), failed.as_str()];
        assert_eq!(
            Audit::of("weights at https://hf.co/moonshot", &turns),
            Audit {
                seen: vec!["https://hf.co/moonshot".into()],
                ..Audit::default()
            },
            "the error names it, so it is seen, never opened"
        );
        assert_eq!(
            Audit::of("weights at https://hf.co/moonshot", &[correction.as_str()]),
            first
        );
    }

    #[test]
    fn the_audit_says_what_was_not_opened_and_the_correction_names_it() {
        let a = Audit {
            opened: vec!["https://a.com".into()],
            miscopied: Vec::new(),
            seen: vec!["https://b.com".into()],
            never_seen: vec!["https://c.com".into()],
        };
        assert_eq!(
            a.render(),
            "Citation audit: 3 cited — 1 opened, 1 seen only in tool results, 1 never seen in the \
             run.\nSeen only in tool results — a search hit, a link, an error — never opened: \
             https://b.com\nNever seen in the run (inferred or invented): https://c.com"
        );
        assert!(a.unopened() && a.cites());
        assert!(a.correction().starts_with(
            "Your answer cites these without having opened them: https://b.com, https://c.com."
        ));
        let clean = Audit {
            opened: vec!["https://a.com".into()],
            ..Audit::default()
        };
        assert!(!clean.unopened());
        assert!(!Audit::default().cites());
    }
}
