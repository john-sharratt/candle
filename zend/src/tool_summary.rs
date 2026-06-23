//! Tool-catalog summary: the execution path for the `tools` collection's
//! categorize→assign [`GroupSummary`].
//!
//! The catalog rarely changes between restarts, and the summary is expensive to
//! generate (two model stages over ~90 tools), so it is cached in the substrate
//! redo log keyed by a hash of the ordered catalog. At startup the daemon
//! computes [`catalog_hash`] over the freshly-injected tools and compares it to
//! the persisted summary's hash (`Substrate::tool_summary_hash`); only on a
//! mismatch does it regenerate via [`generate_tool_summary`] and persist the new
//! value (which supersedes the old — the compactor reclaims it).
//!
//! The generation mirrors the catalog-from-config split proven in
//! `examples/compress_tools.rs`: stage 1 the model proposes the category labels,
//! stage 2 it assigns each tool to a fixed category *by number* (so it can never
//! write a tool name and therefore can never invent one), with a deterministic
//! name-token fallback closing the last gap. The names are filled in from the
//! real catalog, never from the model.

use std::collections::HashSet;

use candle_conversation::persistence::content_hash::hash_bytes;
use candle_conversation::projection::{GroupSummary, SectionId};
use candle_conversation::{
    ConversationEngine, SamplingConfig, Sequence, SequenceConfig, TurnHandle,
};

/// Run one throwaway `(system, user)` exchange and return the reply text. The
/// scratch conversation's timeline is tombstoned immediately — these
/// categorize/assign turns are scaffolding, not conversation history, and must
/// not pollute the workspace (the compactor reclaims the tombstoned records).
fn run_scratch(
    engine: &ConversationEngine,
    system: &str,
    user: &str,
    cfg: &SequenceConfig,
) -> anyhow::Result<String> {
    let mut conv = engine
        .new_conversation(system, cfg.clone())
        .map_err(|e| anyhow::anyhow!("tool-summary scratch conversation: {e}"))?;
    let timeline = conv.timeline_id();
    // Exempt this throwaway timeline from the summariser before its turn seals:
    // it is tombstoned below, so a wave-driven compression pass on it would burn
    // a decode mid-startup and race the tombstone. Must precede `send_turn`.
    engine.set_timeline_summarize(timeline, false);
    let result = conv.send_turn(user);
    drop(conv);
    if let Err(e) = engine.tombstone_timeline(timeline) {
        tracing::warn!("tool-summary: tombstone scratch timeline {timeline}: {e}");
    }
    Ok(result
        .map_err(|e| anyhow::anyhow!("tool-summary scratch send: {e}"))?
        .text)
}

/// Submit `users` as concurrent turns — one per fork of `base` — so their
/// decodes batch in the scheduler instead of running one-at-a-time. Each fork
/// shares `base`'s already-prefilled system-prompt KV (no re-prefill) and is
/// tombstoned once its reply lands. Returns the reply text per input, in order.
///
/// The forks are scaffolding: summarisation is disabled before the turn seals
/// (so the wave-driven summariser never spends a decode on them), and
/// `finish_turn` is skipped — the scheduler auto-finalises the view and the slot
/// frees on drop, so there is no point paying the per-turn finalize + next-user
/// header prefill that `send_turn` would.
fn run_forked_batch(
    engine: &ConversationEngine,
    base: &Sequence,
    users: &[String],
) -> anyhow::Result<Vec<String>> {
    // Submit every fork first (non-blocking) so they are all in flight before we
    // block on any reply — that is what lets the scheduler co-batch their decodes.
    let mut pending: Vec<(Sequence, TurnHandle)> = Vec::with_capacity(users.len());
    for user in users {
        let mut fork = base
            .fork()
            .map_err(|e| anyhow::anyhow!("tool-summary fork: {e}"))?;
        engine.set_timeline_summarize(fork.timeline_id(), false);
        let handle = fork
            .submit_turn(user)
            .map_err(|e| anyhow::anyhow!("tool-summary fork submit: {e}"))?;
        pending.push((fork, handle));
    }
    let mut out = Vec::with_capacity(users.len());
    for (fork, handle) in pending {
        let resp = handle
            .wait()
            .map_err(|e| anyhow::anyhow!("tool-summary fork wait: {e}"))?;
        out.push(resp.text);
        let timeline = fork.timeline_id();
        drop(fork);
        let _ = engine.tombstone_timeline(timeline);
    }
    Ok(out)
}

/// One installed tool: `(name, section_id, json_line)` — the triple
/// [`crate::tools::install_tool_catalog`] returns, in registry order.
pub type InstalledTool = (String, SectionId, String);

/// A 128-bit hash of the ordered tool catalog — each tool's name and full JSON
/// (which includes its parameter schema), concatenated in registry order. Two
/// runs with the same catalog produce the same hash; adding, removing, renaming,
/// or re-parameterising any tool changes it, which is what invalidates the cache.
pub fn catalog_hash(tools: &[InstalledTool]) -> u128 {
    let mut buf: Vec<u8> = Vec::new();
    for (name, _, json) in tools {
        buf.extend_from_slice(name.as_bytes());
        buf.push(0);
        buf.extend_from_slice(json.as_bytes());
        buf.push(0);
    }
    let h = hash_bytes(&buf);
    (u128::from(h.hi) << 64) | u128::from(h.lo)
}

/// Run the `tools` collection's categorize→assign workflow over the real catalog
/// and return the grouped summary text (`## <category>` headers, each followed by
/// its tools by name). Uses generative sampling — argmax drives this model into a
/// self-correction meltdown on the categorize step.
pub fn generate_tool_summary(
    engine: &ConversationEngine,
    tools: &[InstalledTool],
    gs: &GroupSummary,
    config: &SequenceConfig,
) -> anyhow::Result<String> {
    // Categorizing is generative, not lossy compression.
    let mut cfg = config.clone();
    cfg.sampling = SamplingConfig::top_k_top_p(40, 0.9, 0.5).with_repeat_penalty(1.1);

    let names: Vec<String> = tools.iter().map(|(n, _, _)| n.clone()).collect();

    // Numbered (number, name, description) — the model references tools by
    // number, so it cannot invent one.
    let mut numbered = String::new();
    for (i, (name, _, json)) in tools.iter().enumerate() {
        let desc = serde_json::from_str::<serde_json::Value>(json.trim())
            .ok()
            .and_then(|v| {
                v.get("description")
                    .and_then(|d| d.as_str())
                    .map(str::to_string)
            })
            .unwrap_or_default();
        numbered.push_str(&format!("{}. {} — {}\n", i + 1, name, desc));
    }

    // ── Stage 1: propose the category labels.
    let resp1 = run_scratch(
        engine,
        &gs.categorize.prompt.system_prompt.content,
        &format!("{numbered}\n{}", gs.categorize.prompt.user_prompt),
        &cfg,
    )?;
    let categories = parse_category_list(&resp1);
    if categories.is_empty() {
        anyhow::bail!("tool-summary stage 1 produced no categories:\n{resp1}");
    }

    // ── Stage 2: assign each tool to a fixed category, by number, in chunks.
    //
    // Every chunk runs the *same* assign system prompt, so prefill it once into a
    // base conversation and fork per chunk: the forks share that system-prompt KV
    // (no re-prefill) and their decodes batch in the scheduler rather than running
    // one-at-a-time. The categorize→assign→missing stages stay ordered (genuine
    // data dependencies); only the independent chunks within a stage fan out.
    let cat_list: String = categories
        .iter()
        .enumerate()
        .map(|(i, c)| format!("{}={}", i + 1, c))
        .collect::<Vec<_>>()
        .join("; ");
    let asn_sys = &gs.assign.prompt.system_prompt.content;
    let chunk = gs.chunk.max(1);

    // Per-chunk `(first_tool_number - 1, last_tool_number, user_prompt)`.
    let chunks: Vec<(usize, usize, String)> = (0..names.len())
        .step_by(chunk)
        .map(|start| {
            let end = (start + chunk).min(names.len());
            let mut sub = String::new();
            for (i, name) in names.iter().enumerate().take(end).skip(start) {
                sub.push_str(&format!("{}. {}\n", i + 1, name));
            }
            let user = format!(
                "FIXED categories (use ONLY these numbers, never invent any):\n{cat_list}\n\n\
                 Tools:\n{sub}\n{}\nExample: [{}=1][{}=4].",
                gs.assign.prompt.user_prompt,
                start + 1,
                start + 2
            );
            (start, end, user)
        })
        .collect();

    // Base holds the prefilled assign system prompt; every fork shares it.
    let base = engine
        .new_conversation(asn_sys, cfg.clone())
        .map_err(|e| anyhow::anyhow!("tool-summary assign base conversation: {e}"))?;
    let base_timeline = base.timeline_id();
    engine.set_timeline_summarize(base_timeline, false);

    // Best-of-3 per chunk, batched: each round forks every still-incomplete chunk
    // at once so their decodes ride one wave. Keep the longest parse seen.
    let mut got: Vec<Vec<(usize, usize)>> = vec![Vec::new(); chunks.len()];
    for _ in 0..3 {
        let todo: Vec<usize> = (0..chunks.len())
            .filter(|&i| got[i].len() < chunks[i].1 - chunks[i].0)
            .collect();
        if todo.is_empty() {
            break;
        }
        let users: Vec<String> = todo.iter().map(|&i| chunks[i].2.clone()).collect();
        let texts = run_forked_batch(engine, &base, &users)?;
        for (k, &i) in todo.iter().enumerate() {
            let (start, end, _) = chunks[i];
            let part: Vec<(usize, usize)> = parse_numeric_assignments(&texts[k])
                .into_iter()
                .filter(|(n, _)| *n > start && *n <= end)
                .collect();
            if part.len() > got[i].len() {
                got[i] = part;
            }
        }
    }
    let mut assigns: Vec<(usize, usize)> = got.into_iter().flatten().collect();

    // Final pass over any tool no chunk resolved — one fork per retry (still
    // sharing the base prefill), accumulating until every gap is closed.
    let mut have: HashSet<usize> = assigns
        .iter()
        .filter(|(n, k)| *n >= 1 && *n <= names.len() && *k >= 1 && *k <= categories.len())
        .map(|(n, _)| *n)
        .collect();
    let missing: Vec<usize> = (1..=names.len()).filter(|n| !have.contains(n)).collect();
    if !missing.is_empty() {
        let mut sub = String::new();
        for &n in &missing {
            sub.push_str(&format!("{}. {}\n", n, names[n - 1]));
        }
        let user = format!(
            "FIXED categories (use ONLY these numbers, never invent any):\n{cat_list}\n\n\
             Tools:\n{sub}\nOutput exactly one bracket [<tool-number>=<category-number>] per \
             tool, using only category numbers 1-{}. Brackets only.",
            categories.len()
        );
        for _ in 0..3 {
            let text = run_forked_batch(engine, &base, std::slice::from_ref(&user))?
                .pop()
                .unwrap_or_default();
            for (n, k) in parse_numeric_assignments(&text) {
                if missing.contains(&n) && !have.contains(&n) {
                    have.insert(n);
                    assigns.push((n, k));
                }
            }
            if missing.iter().all(|n| have.contains(n)) {
                break;
            }
        }
    }

    // Base no longer needed — its forks carried the work.
    let _ = engine.tombstone_timeline(base_timeline);

    // Deterministic fallback: place any still-unassigned tool with the category
    // whose tools share the most underscore-separated name tokens with it.
    let placed: Vec<(usize, usize)> = assigns
        .iter()
        .copied()
        .filter(|(n, k)| *n >= 1 && *n <= names.len() && *k >= 1 && *k <= categories.len())
        .collect();
    let still: Vec<usize> = (1..=names.len())
        .filter(|n| !placed.iter().any(|(m, _)| m == n))
        .collect();
    for n in still {
        let tn = name_tokens(&names[n - 1]);
        let mut best = (0usize, 0usize); // (score, category)
        for c in 1..=categories.len() {
            let mut score = 0usize;
            for (m, k) in &placed {
                if *k != c {
                    continue;
                }
                let tm = name_tokens(&names[m - 1]);
                score += tn.iter().filter(|t| tm.contains(t)).count();
            }
            if score > best.0 {
                best = (score, c);
            }
        }
        if best.1 >= 1 {
            assigns.push((n, best.1));
        }
    }

    // Render `## <category>` blocks, names filled from the real catalog.
    let mut assigned = vec![false; names.len()];
    let mut out = String::new();
    for (ci, label) in categories.iter().enumerate() {
        let cat_num = ci + 1;
        let mut listed: Vec<&str> = Vec::new();
        for (n, k) in &assigns {
            if *k != cat_num {
                continue;
            }
            if let Some(idx) = n.checked_sub(1) {
                if let Some(name) = names.get(idx) {
                    if !assigned[idx] {
                        assigned[idx] = true;
                        listed.push(name.as_str());
                    }
                }
            }
        }
        if listed.is_empty() {
            continue;
        }
        out.push_str(&format!("## {label}\n  {}\n", listed.join(", ")));
    }
    let leftover: Vec<&str> = names
        .iter()
        .enumerate()
        .filter(|(i, _)| !assigned[*i])
        .map(|(_, n)| n.as_str())
        .collect();
    if !leftover.is_empty() {
        out.push_str(&format!("## Uncategorized\n  {}\n", leftover.join(", ")));
    }
    Ok(out.trim_end().to_string())
}

/// Underscore-separated tokens of a tool name (`sql_session_query` → sql,
/// session, query) — the deterministic fallback's name-overlap key.
fn name_tokens(s: &str) -> Vec<&str> {
    s.split('_').collect()
}

/// Parse stage 1's numbered category list into clean labels. Locates each
/// ordinal `N.` marker (so it works whether the model used newlines or ran them
/// onto one line) and slices the label between consecutive markers.
fn parse_category_list(text: &str) -> Vec<String> {
    let mut marks: Vec<(usize, usize)> = Vec::new();
    let mut n = 1usize;
    let mut from = 0usize;
    while n <= 12 {
        let pat = format!("{n}.");
        let Some(rel) = text[from..].find(&pat) else {
            break;
        };
        let pos = from + rel;
        let boundary = pos == 0
            || text[..pos]
                .chars()
                .next_back()
                .map(char::is_whitespace)
                .unwrap_or(true);
        if boundary {
            marks.push((pos, pos + pat.len()));
            from = pos + pat.len();
            n += 1;
        } else {
            from = pos + pat.len();
        }
    }

    let mut cats = Vec::new();
    for i in 0..marks.len() {
        let start = marks[i].1;
        let end = marks.get(i + 1).map(|m| m.0).unwrap_or(text.len());
        let raw = text[start..end].trim();
        let label = raw
            .split(" - ")
            .next()
            .unwrap_or(raw)
            .split(" — ")
            .next()
            .unwrap_or(raw)
            .split(':')
            .next()
            .unwrap_or(raw)
            .trim();
        if !label.is_empty() && label.len() < 80 {
            cats.push(label.to_string());
        }
    }
    cats
}

/// Parse `<tool-number>=<category-number>` assignments — with or without
/// brackets, tolerant of spaces — by scanning each `=` for a number on both
/// sides. A non-numeric RHS (`1=Date`) is ignored, so stray category lists don't
/// match.
fn parse_numeric_assignments(text: &str) -> Vec<(usize, usize)> {
    let chars: Vec<char> = text.chars().collect();
    let mut out = Vec::new();
    for (i, &c) in chars.iter().enumerate() {
        if c != '=' {
            continue;
        }
        let mut l = i;
        while l > 0 && chars[l - 1].is_whitespace() {
            l -= 1;
        }
        let mut ls = l;
        while ls > 0 && chars[ls - 1].is_ascii_digit() {
            ls -= 1;
        }
        if ls == l {
            continue;
        }
        let mut r = i + 1;
        while r < chars.len() && chars[r].is_whitespace() {
            r += 1;
        }
        let mut re = r;
        while re < chars.len() && chars[re].is_ascii_digit() {
            re += 1;
        }
        if re == r {
            continue;
        }
        let lnum: usize = chars[ls..l].iter().collect::<String>().parse().unwrap_or(0);
        let rnum: usize = chars[r..re].iter().collect::<String>().parse().unwrap_or(0);
        if lnum >= 1 {
            out.push((lnum, rnum));
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn catalog_hash_is_stable_and_order_sensitive() {
        let a = (
            "alpha".to_string(),
            SectionId::new(1),
            "{\"name\":\"alpha\",\"parameters\":{}}".to_string(),
        );
        let b = (
            "beta".to_string(),
            SectionId::new(2),
            "{\"name\":\"beta\",\"parameters\":{}}".to_string(),
        );
        let ab = vec![a.clone(), b.clone()];
        // Stable across calls.
        assert_eq!(catalog_hash(&ab), catalog_hash(&ab));
        // Order-sensitive.
        assert_ne!(catalog_hash(&ab), catalog_hash(&[b.clone(), a.clone()]));
        // Parameter change invalidates.
        let b2 = (
            "beta".to_string(),
            SectionId::new(2),
            "{\"name\":\"beta\",\"parameters\":{\"x\":1}}".to_string(),
        );
        assert_ne!(catalog_hash(&ab), catalog_hash(&[a, b2]));
    }

    #[test]
    fn parse_assignments_ignores_non_numeric_rhs() {
        assert_eq!(
            parse_numeric_assignments("[9=2][10=3] 11=4 1=Date"),
            vec![(9, 2), (10, 3), (11, 4)]
        );
    }

    #[test]
    fn parse_categories_handles_inline_and_descriptions() {
        let cats = parse_category_list("1. Files — manage files 2. Network: sockets 3. Crypto");
        assert_eq!(cats, vec!["Files", "Network", "Crypto"]);
    }
}
