//! A finished task's citation audit (`docs/mcp_guides.md` §4): each URL its
//! final response cites, against what the worker — and the parts it started —
//! actually did in their turns.
//!
//! | Mark | Meaning |
//! |---|---|
//! | opened | fetched with `web_fetch`: the URL it asked for, or the one it reached |
//! | seen | appeared only in a tool's result — a search hit, a snippet — never fetched |
//! | never seen | cited, but nothing in the run touched it |
//!
//! Deterministic and cheap: the calls are read back out of each turn's
//! assistant half ([`extract_tool_calls`]), and their results are the turns'
//! user halves. A task whose answer cites anything not opened is sent back
//! once to open it ([`Audit::correction`]).

use std::collections::BTreeSet;
use std::sync::LazyLock;

use regex::Regex;
use serde_json::Value;

use crate::tools::extract_tool_calls;

/// A URL as it appears in text: up to the first character that ends one.
static URL: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r#"https?://[^\s<>"'`()\[\]{}|\\^]+"#).expect("a valid pattern"));

/// A fetch's `final_url`, as its JSON result carries it.
static FINAL_URL: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r#""final_url"\s*:\s*"([^"]+)""#).expect("a valid pattern"));

/// One turn's two halves.
pub struct Turn<'a> {
    pub user: &'a str,
    pub assistant: &'a str,
}

/// What a final response cites, sorted by what the run did with each.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Audit {
    pub opened: Vec<String>,
    pub seen: Vec<String>,
    pub never_seen: Vec<String>,
}

/// A URL with the punctuation a sentence ends it with taken off.
fn trimmed(url: &str) -> &str {
    url.trim_end_matches(['.', ',', ';', ':', '!', '?', '*', '_'])
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

impl Audit {
    /// The audit of `answer` against `turns`.
    pub fn of(answer: &str, turns: &[Turn<'_>]) -> Self {
        let mut opened = BTreeSet::new();
        let mut touched = BTreeSet::new();
        for t in turns {
            for call in extract_tool_calls(t.assistant) {
                if call.name == "web_fetch" {
                    if let Some(url) = call.arguments.get("url").and_then(Value::as_str) {
                        opened.insert(normal(url));
                    }
                }
            }
            for c in FINAL_URL.captures_iter(t.user) {
                opened.insert(normal(&c[1]));
            }
            for u in urls_in(t.user) {
                touched.insert(normal(&u));
            }
        }
        let mut audit = Self::default();
        for cited in urls_in(answer) {
            let key = normal(&cited);
            if opened.contains(&key) {
                audit.opened.push(cited);
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
        !(self.opened.is_empty() && self.seen.is_empty() && self.never_seen.is_empty())
    }

    /// Whether it cites anything it did not open.
    pub fn unopened(&self) -> bool {
        !(self.seen.is_empty() && self.never_seen.is_empty())
    }

    /// The audit as a task's answer carries it.
    pub fn render(&self) -> String {
        let mut out = format!(
            "Citation audit: {} cited — {} opened, {} seen only in search results, {} never seen \
             in the run.",
            self.opened.len() + self.seen.len() + self.never_seen.len(),
            self.opened.len(),
            self.seen.len(),
            self.never_seen.len()
        );
        if !self.seen.is_empty() {
            out.push_str(&format!(
                "\nSeen only in search results (the snippet, not the page): {}",
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

    /// The message that sends the worker back to open what it cited.
    pub fn correction(&self) -> String {
        let unopened: Vec<&str> = self
            .seen
            .iter()
            .chain(&self.never_seen)
            .map(String::as_str)
            .collect();
        format!(
            "Your answer cites these without having opened them: {}.\n\nFetch each one with \
             web_fetch and check the claim it supports. Correct your answer where the page says \
             otherwise; remove a citation whose page does not exist; mark UNVERIFIED anything you \
             still cannot confirm. Answer again, whole, in the same shape as before, with a \
             CHANGES section saying what you corrected and why.",
            unopened.join(", ")
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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

    /// **Each citation is marked by what the run did with it**: fetched is
    /// opened — by the URL asked or the one reached — a search hit is seen,
    /// and anything else was never seen.
    #[test]
    fn each_citation_is_marked_by_what_the_run_did_with_it() {
        let fetch = r#"<tool_call>{"name": "web_fetch", "arguments": {"url": "https://kimi.ai/blog/k3"}}</tool_call>"#;
        let turns = [
            Turn {
                user: "research the releases",
                assistant: fetch,
            },
            Turn {
                user: r#"{"url":"https://redirect.example/x","final_url":"https://inception.ai/mercury-3","title":"Mercury"}"#,
                assistant: "",
            },
            Turn {
                user:
                    "1. Kimi K3.1 — https://news.example.com/kimi 2. https://llm-stats.com/models",
                assistant: "",
            },
        ];
        let answer = "Kimi (https://kimi.ai/blog/k3), Mercury (https://inception.ai/mercury-3/), \
                      scores per https://llm-stats.com/models, weights at https://hf.co/moonshot.";
        assert_eq!(
            Audit::of(answer, &turns),
            Audit {
                opened: vec![
                    "https://kimi.ai/blog/k3".into(),
                    "https://inception.ai/mercury-3/".into()
                ],
                seen: vec!["https://llm-stats.com/models".into()],
                never_seen: vec!["https://hf.co/moonshot".into()],
            }
        );
    }

    #[test]
    fn the_audit_says_what_was_not_opened_and_the_correction_names_it() {
        let a = Audit {
            opened: vec!["https://a.com".into()],
            seen: vec!["https://b.com".into()],
            never_seen: vec!["https://c.com".into()],
        };
        assert_eq!(
            a.render(),
            "Citation audit: 3 cited — 1 opened, 1 seen only in search results, 1 never seen in \
             the run.\nSeen only in search results (the snippet, not the page): https://b.com\n\
             Never seen in the run (inferred or invented): https://c.com"
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
