//! Gate 4: the paper conversation broken into its projections
//! (`docs/provenance_scoped_signatures.md` §3.4, `docs/paper_conversation_replay.md`).
//!
//! The replayed conversation's own turns are cut by the shared boundary rule
//! into the probes its projections asked — every boundary of every turn the
//! dialogue sealed — and each probe ranks the `code_reading` files offline.
//! Two kinds of expectation grade it:
//!
//! - **written**, per user message, in the expectations file: the files the
//!   working set should hold in its top K for that message's projections, and
//!   the ones it must not;
//! - **mechanical**: a probe holding a tool call that reads a file should rank
//!   that file in its top K.
//!
//! A turn's reasoning is not sealed into its signature, so a thinking block's
//! own boundaries are not among the probes; the probes are the user turns, the
//! visible answers, the tool calls and the tool responses.

use std::collections::{BTreeMap, HashMap};
use std::path::Path;

use candle_conversation::provenance::boundaries::BoundaryRule;
use serde::Deserialize;
use tokenizers::Tokenizer;

use crate::corpus::{gather, ByLayer, File, DIALOGUE};
use crate::gpu_gallery::GpuGallery;
use crate::ranking::order;
use crate::rules::{Scorer, RULES};

/// The expectations file.
#[derive(Deserialize)]
pub struct Expectations {
    pub top_k: usize,
    pub messages: Vec<Message>,
}

/// One user message's expectations.
#[derive(Deserialize)]
pub struct Message {
    /// The message's opening words, which find its turn.
    pub starts: String,
    /// Workspace paths or `*` patterns the top K should hold.
    pub should: Vec<String>,
    /// Patterns the top K must not hold.
    pub should_not: Vec<String>,
}

/// Whether `path` matches `pattern`, where `*` matches any run of characters,
/// `/` included.
pub fn matches(pattern: &str, path: &str) -> bool {
    match pattern.split_once('*') {
        None => pattern == path,
        Some((head, rest)) => {
            let Some(after) = path.strip_prefix(head) else {
                return false;
            };
            (0..=after.len())
                .filter(|&i| after.is_char_boundary(i))
                .any(|i| matches(rest, &after[i..]))
        }
    }
}

/// The `repo/path` keys of the `file_read` calls in `text`: each call's
/// `"repo"` and `"path"` arguments.
pub fn called_paths(text: &str) -> Vec<String> {
    let arg = |call: &str, name: &str| -> Option<String> {
        let at = call.find(&format!("\"{name}\""))?;
        let rest = call[at + name.len() + 2..].trim_start();
        let rest = rest.strip_prefix(':')?.trim_start().strip_prefix('"')?;
        Some(rest[..rest.find('"')?].to_string())
    };
    text.split("<tool_call>")
        .skip(1)
        .filter_map(|call| {
            let call = call.split("</tool_call>").next()?;
            Some(format!("{}/{}", arg(call, "repo")?, arg(call, "path")?))
        })
        .collect()
}

pub fn run(
    files: &[File],
    by_layer: &mut ByLayer,
    rule: &BoundaryRule,
    tokenizer: &Tokenizer,
    timeline: u64,
    expectations: &Path,
) -> anyhow::Result<()> {
    let exp: Expectations = serde_yaml::from_str(&std::fs::read_to_string(expectations)?)?;
    let gpu = GpuGallery::new(files)?;
    let scorer = Scorer::new(&gpu, files, rule)?;
    let turns = by_layer
        .remove(&DIALOGUE)
        .and_then(|mut d| d.remove(&timeline))
        .ok_or_else(|| anyhow::anyhow!("no dialogue turns for timeline {timeline:016x}"))?;
    let path_of: Vec<String> = files
        .iter()
        .map(|f| {
            f.path
                .clone()
                .unwrap_or_else(|| format!("{:016x}", f.timeline))
        })
        .collect();
    let index_of: HashMap<&str, usize> = path_of
        .iter()
        .enumerate()
        .map(|(i, p)| (p.as_str(), i))
        .collect();
    for (m, msg) in exp.messages.iter().enumerate() {
        for p in msg.should.iter().filter(|p| !p.contains('*')) {
            if !index_of.contains_key(p.as_str()) {
                println!("message {m}: `{p}` is not in the gallery yet");
            }
        }
    }
    let rules: Vec<&str> = RULES.iter().map(|r| r.name).collect();
    let k = exp.top_k;
    // [message][rule] → (projections, should hits, should total, should_not hits).
    let mut graded: BTreeMap<(usize, usize), (usize, usize, usize, usize)> = BTreeMap::new();
    // [rule] → (tool calls, the read file in the top K).
    let mut calls: BTreeMap<usize, (usize, usize)> = BTreeMap::new();
    let mut message: Option<usize> = None;
    for (ti, turn) in &turns {
        if !turn.covered() {
            println!(
                "dialogue turn {ti}: not covered ({} tokens, {} signatures)",
                turn.tokens.len(),
                turn.sigs.len()
            );
            continue;
        }
        let text = tokenizer.decode(&turn.tokens, false).unwrap_or_default();
        if let Some(m) = exp
            .messages
            .iter()
            .position(|msg| text.contains(&msg.starts))
        {
            message = Some(m);
        }
        let Some(m) = message else {
            continue;
        };
        for (wi, w) in turn.windows(rule).iter().enumerate() {
            let probe = gather(&turn.sigs, w);
            let window_tokens: Vec<u32> = w
                .ranges()
                .flat_map(|r| turn.tokens[r].iter().copied())
                .collect();
            let window_text = tokenizer.decode(&window_tokens, false).unwrap_or_default();
            // Only a read of a file the gallery holds can be found: a listing
            // of a folder, or a file not yet read, is no expectation.
            let read: Vec<String> = called_paths(&window_text)
                .into_iter()
                .filter(|p| index_of.contains_key(p.as_str()))
                .collect();
            let per_rule = scorer.score(&probe, &|_, _| false, &|_, _| false)?;
            let mut tops: Vec<String> = Vec::new();
            for (ri, s) in per_rule.iter().enumerate() {
                let top: Vec<&str> = order(s)
                    .into_iter()
                    .take(k)
                    .map(|f| path_of[f].as_str())
                    .collect();
                let msg = &exp.messages[m];
                let should_hits = msg
                    .should
                    .iter()
                    .filter(|p| top.iter().any(|t| matches(p, t)))
                    .count();
                let should_not_hits = top
                    .iter()
                    .filter(|t| msg.should_not.iter().any(|p| matches(p, t)))
                    .count();
                let e = graded.entry((m, ri)).or_default();
                e.0 += 1;
                e.1 += should_hits;
                e.2 += msg.should.len();
                e.3 += should_not_hits;
                for p in &read {
                    let c = calls.entry(ri).or_default();
                    c.0 += 1;
                    c.1 += usize::from(top.contains(&p.as_str()));
                }
                // Today's, and the live rule with its questions.
                if ri == 0 || ri + 1 == RULES.len() {
                    tops.push(format!(
                        "    {}: {}",
                        rules[ri],
                        top.iter().take(5).copied().collect::<Vec<_>>().join(", ")
                    ));
                }
            }
            println!(
                "message {m} turn {ti} window {wi}{}",
                if read.is_empty() {
                    String::new()
                } else {
                    format!(" reads {}", read.join(", "))
                }
            );
            for t in tops {
                println!("{t}");
            }
        }
    }
    println!(
        "\n{:<8} {:<40} {:>6} {:>9} {:>11}",
        "message", "rule", "probes", "should", "should_not"
    );
    for ((m, ri), (n, hits, total, bad)) in &graded {
        println!(
            "{m:<8} {:<40} {n:>6} {:>8.0}% {:>10.2}",
            rules[*ri],
            100.0 * *hits as f32 / (*total).max(1) as f32,
            *bad as f32 / (*n).max(1) as f32
        );
    }
    println!();
    for (ri, (n, hit)) in &calls {
        println!(
            "tool calls {:<40} the read file in the top {k}: {hit}/{n}",
            rules[*ri]
        );
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_star_matches_any_run_slashes_included() {
        assert!(matches("candle/docs/*.md", "candle/docs/performance.md"));
        assert!(matches("*/tests/*", "candle/candle-nn/tests/kv.rs"));
        assert!(matches("*_tests.rs", "candle/a/b_tests.rs"));
        assert!(!matches("*/tests/*", "candle/candle-nn/src/kv.rs"));
        assert!(matches("candle/README.md", "candle/README.md"));
        assert!(!matches("candle/README.md", "candle/README.md.bak"));
    }

    #[test]
    fn a_tool_call_names_its_repo_and_path() {
        let text = r#"x<tool_call>{"name": "file_read", "arguments": {"repo": "candle", "path": "docs/unbounded_agents.md", "page": 2}}</tool_call>y<tool_call>{"name":"file_read","arguments":{"repo":"candle","path":"README.md"}}</tool_call>"#;
        assert_eq!(
            called_paths(text),
            vec!["candle/docs/unbounded_agents.md", "candle/README.md"]
        );
        assert!(called_paths("no calls").is_empty());
    }
}
