//! `drive_tool_cases` — drive the running zend daemon to build the per-tool
//! invocation baseline.
//!
//! For each tool's train + holdout prompt (from `tool_cases/prompts.json`) it
//! POSTs to `/v1/chat/completions` with `force_high_resolution: "tools"` — so the
//! full tool catalog is materialised (projection + reprojection don't filter it)
//! and capture mode seals the invocation turn without executing the tool. Each
//! conversation lands in the substrate under a stable `conv_id`; the prompt,
//! response, and parsed invocation are recorded to `tool_cases/test_config.json`.
//!
//! Start the daemon first (it serves 127.0.0.1:8080), then:
//! ```sh
//! cargo run -p zend --example drive_tool_cases --release
//! ```

use std::path::PathBuf;

use futures::stream::{self, StreamExt};
use serde::{Deserialize, Serialize};
use serde_json::json;
use zend_tools::registry;

/// Seed the assistant response with the tool-call opener so the decode is forced
/// into the tool-call grammar — the model can't refuse, narrate, or fabricate a
/// result. Passed verbatim as `assistant_prefill` in each chat request. The
/// model still chooses the tool name and arguments, so tool selection stays a
/// real signal; only the call-vs-prose decision is taken off the table.
const ASSISTANT_PREFILL: &str = "<tool_call>";

#[derive(Deserialize)]
struct PromptCase {
    tool: String,
    train: String,
    holdout: String,
    /// Per-case override of [`ASSISTANT_PREFILL`]. Defaults to the bare
    /// `<tool_call>` opener; set it to pin more of the call (e.g. the tool name)
    /// for a case the model otherwise mis-routes. `prefill_note` records why.
    #[serde(default)]
    prefill: Option<String>,
    /// Human-readable justification for a non-default `prefill`, kept beside the
    /// case (JSON has no comments) and echoed at startup so the reason travels
    /// with the data rather than living only in a commit message.
    #[serde(default)]
    prefill_note: Option<String>,
}
#[derive(Deserialize)]
struct PromptsFile {
    cases: Vec<PromptCase>,
}

#[derive(Serialize)]
struct CaseResult {
    tool: String,
    split: String,
    conv_id: String,
    prompt: String,
    response: String,
    /// Raw tool name the model emitted (may be an alias).
    invoked_tool: Option<String>,
    /// Canonical tool the emitted name resolves to via `registry::find` (aliases
    /// included); `None` if no tool call or an unknown name.
    resolved_tool: Option<String>,
    correct: bool,
    /// Whether the extracted tool-call object parses as well-formed JSON.
    json_valid: bool,
    /// Keys that appear more than once within a single object of the call —
    /// valid to serde (last wins) but a quality wobble worth surfacing.
    duplicate_keys: Vec<String>,
}

/// Pull the tool name the model emitted out of a response. Preamble-aware: the
/// model may say something before the call, so search begins at the `<tool_call>`
/// tag when present (else anywhere, for the no-wrapper variant).
fn parse_invoked_tool(text: &str) -> Option<String> {
    let start = text
        .find("<tool_call>")
        .map(|i| i + "<tool_call>".len())
        .unwrap_or(0);
    let s = &text[start..];
    let key = s.find("\"name\"")?;
    let after_colon = s[key..].find(':')? + key;
    let rest = &s[after_colon..];
    let open = rest.find('"')? + 1;
    let inner = &rest[open..];
    let close = inner.find('"')?;
    Some(inner[..close].to_string())
}

/// The first balanced `{...}` object in `text` (the tool-call body), string- and
/// escape-aware so braces inside JSON strings don't throw off the depth count.
fn first_json_object(text: &str) -> Option<&str> {
    let b = text.as_bytes();
    let start = text.find('{')?;
    let (mut depth, mut in_str, mut esc) = (0usize, false, false);
    for i in start..b.len() {
        let c = b[i];
        if in_str {
            if esc {
                esc = false;
            } else if c == b'\\' {
                esc = true;
            } else if c == b'"' {
                in_str = false;
            }
        } else if c == b'"' {
            in_str = true;
        } else if c == b'{' {
            depth += 1;
        } else if c == b'}' {
            depth -= 1;
            if depth == 0 {
                return Some(&text[start..=i]);
            }
        }
    }
    None
}

/// Keys that appear more than once within a single JSON object of `json`. serde
/// silently keeps the last value for a duplicate key, so this scan catches the
/// wobble that parsing alone would hide. A stack of seen-key sets handles nesting.
fn duplicate_keys(json: &str) -> Vec<String> {
    use std::collections::HashSet;
    let b = json.as_bytes();
    let mut dups = Vec::new();
    let mut stack: Vec<HashSet<String>> = Vec::new();
    let (mut in_str, mut esc) = (false, false);
    let mut cur: Vec<u8> = Vec::new();
    let mut last_string: Option<String> = None;
    for &c in b {
        if in_str {
            if esc {
                cur.push(c);
                esc = false;
            } else if c == b'\\' {
                cur.push(c);
                esc = true;
            } else if c == b'"' {
                in_str = false;
                last_string = Some(String::from_utf8_lossy(&cur).into_owned());
            } else {
                cur.push(c);
            }
            continue;
        }
        match c {
            b'"' => {
                in_str = true;
                cur.clear();
            }
            b'{' => {
                stack.push(HashSet::new());
                last_string = None;
            }
            b'}' => {
                stack.pop();
                last_string = None;
            }
            // A string immediately followed by `:` is a key in the current object.
            b':' => {
                if let (Some(set), Some(k)) = (stack.last_mut(), last_string.take()) {
                    if !set.insert(k.clone()) {
                        dups.push(k);
                    }
                }
            }
            b',' => last_string = None,
            _ => {}
        }
    }
    dups
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let url = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "http://127.0.0.1:8080/v1/chat/completions".to_string());
    // 2nd arg: max concurrent requests (queue width). 1 = sequential.
    let concurrency: usize = std::env::args()
        .nth(2)
        .and_then(|s| s.parse().ok())
        .unwrap_or(16);
    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("candle-conversation")
        .join("tests")
        .join("tool_cases");
    let prompts: PromptsFile =
        serde_json::from_str(&std::fs::read_to_string(dir.join("prompts.json"))?)?;
    println!("driving {} tool cases against {url}", prompts.cases.len());

    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(300))
        .build()?;

    println!("concurrency: {concurrency}");
    println!("assistant_prefill (default): {ASSISTANT_PREFILL:?}");
    // Echo any per-case prefill overrides + their justification, so the reason
    // travels with the run output and not just the data file.
    for c in &prompts.cases {
        if let Some(p) = &c.prefill {
            let why = c
                .prefill_note
                .as_deref()
                .map(|n| format!(" — {n}"))
                .unwrap_or_default();
            println!("  prefill override for {}: {p:?}{why}", c.tool);
        }
    }

    // Flatten every (tool, split) into an indexed work queue. Each task carries
    // the prefill to seed: the case's override if set, else the default opener.
    let tasks: Vec<(usize, String, String, String, String)> = prompts
        .cases
        .iter()
        .flat_map(|c| {
            let prefill = c
                .prefill
                .clone()
                .unwrap_or_else(|| ASSISTANT_PREFILL.to_string());
            [("train", &c.train), ("holdout", &c.holdout)]
                .into_iter()
                .map(move |(split, prompt)| {
                    (
                        c.tool.clone(),
                        split.to_string(),
                        prompt.clone(),
                        prefill.clone(),
                    )
                })
        })
        .enumerate()
        .map(|(i, (tool, split, prompt, prefill))| (i, tool, split, prompt, prefill))
        .collect();
    let total = tasks.len();
    let done = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));

    // Run the queue CONCURRENCY-at-a-time; buffer_unordered keeps the window
    // full, starting a new request as each completes.
    let mut indexed: Vec<(usize, CaseResult)> = stream::iter(tasks)
        .map(|(idx, tool, split, prompt, prefill)| {
            let client = client.clone();
            let url = url.clone();
            // Base for sibling endpoints (e.g. archive), derived from the chat URL.
            let base = url
                .trim_end_matches("/v1/chat/completions")
                .trim_end_matches('/')
                .to_string();
            let done = std::sync::Arc::clone(&done);
            async move {
                let conv_id = format!("toolcase_{tool}_{split}");
                let content = {
                    // Emit ONLY this tool's section (tools/<tool>).
                    let body = json!({
                        "model": "zen-code",
                        "messages": [{ "role": "user", "content": &prompt }],
                        "stream": false,
                        "conv_id": &conv_id,
                        "force_high_resolution": format!("tools/{tool}"),
                        "assistant_prefill": &prefill,
                        // Seal these capture turns without KV quantization so the
                        // provenance work gets full-resolution (native R16/F16) keys.
                        "lossless_kv": true,
                        "max_tokens": 128,
                    });
                    match client.post(&url).json(&body).send().await {
                        Ok(resp) => resp
                            .json::<serde_json::Value>()
                            .await
                            .unwrap_or(serde_json::Value::Null)["choices"][0]["message"]["content"]
                            .as_str()
                            .unwrap_or("")
                            .to_string(),
                        Err(e) => {
                            eprintln!("  {conv_id}: request failed: {e}");
                            String::new()
                        }
                    }
                };
                // The prefilled prefix is pinned into the turn's K/V server-side
                // but isn't part of the decoded body the daemon returns; prepend it
                // so the recorded exemplar reads as the full <tool_call>{...}.
                let content = format!("{prefill}{content}");
                // Archive this capture conversation so it doesn't bloat the
                // sidebar / active set (best-effort — the capture is what matters).
                let _ = client
                    .post(format!("{base}/v1/conversations/{conv_id}/archive"))
                    .send()
                    .await;

                let invoked = parse_invoked_tool(&content);
                // Resolve the emitted name (alias-aware) to its canonical tool —
                // an alias invocation still counts as hitting the right tool.
                let resolved = invoked
                    .as_deref()
                    .and_then(registry::find)
                    .map(|t| t.name.to_string());
                let ok = resolved.as_deref() == Some(tool.as_str());
                // Capture-quality checks on the extracted call object (separate
                // axis from tool selection): does it parse, and are any keys dup'd?
                let call_json = first_json_object(&content);
                let json_valid = call_json
                    .map(|j| serde_json::from_str::<serde_json::Value>(j).is_ok())
                    .unwrap_or(false);
                let dup_keys = call_json.map(duplicate_keys).unwrap_or_default();
                let flag = if !json_valid {
                    " !badjson".to_string()
                } else if !dup_keys.is_empty() {
                    format!(" !dupkey:{}", dup_keys.join(","))
                } else {
                    String::new()
                };
                let n = done.fetch_add(1, std::sync::atomic::Ordering::Relaxed) + 1;
                println!(
                    "[{n:>3}/{total}] {conv_id:<38} {}{flag} invoked={:<26} {}",
                    if ok { "OK " } else { " ? " },
                    invoked.clone().unwrap_or_else(|| "<none>".into()),
                    content
                        .chars()
                        .take(160)
                        .collect::<String>()
                        .replace('\n', "\\n"),
                );
                (
                    idx,
                    CaseResult {
                        tool,
                        split,
                        conv_id,
                        prompt,
                        response: content,
                        invoked_tool: invoked,
                        resolved_tool: resolved,
                        correct: ok,
                        json_valid,
                        duplicate_keys: dup_keys,
                    },
                )
            }
        })
        .buffer_unordered(concurrency)
        .collect()
        .await;

    // Restore declaration order for a stable test config.
    indexed.sort_by_key(|(i, _)| *i);
    let results: Vec<CaseResult> = indexed.into_iter().map(|(_, r)| r).collect();
    let correct = results.iter().filter(|r| r.correct).count();

    let malformed: Vec<&CaseResult> = results.iter().filter(|r| !r.json_valid).collect();
    let dup_keyed: Vec<&CaseResult> = results
        .iter()
        .filter(|r| r.json_valid && !r.duplicate_keys.is_empty())
        .collect();

    let out = dir.join("test_config.json");
    std::fs::write(&out, serde_json::to_string_pretty(&results)?)?;
    println!("\n{correct}/{total} cases invoked the expected tool");
    println!(
        "capture quality: {} malformed JSON, {} with duplicate keys",
        malformed.len(),
        dup_keyed.len()
    );
    for r in &malformed {
        println!("  !badjson  {}_{}", r.tool, r.split);
    }
    for r in &dup_keyed {
        println!(
            "  !dupkey   {}_{} ({})",
            r.tool,
            r.split,
            r.duplicate_keys.join(",")
        );
    }
    println!("test config → {}", out.display());
    Ok(())
}
