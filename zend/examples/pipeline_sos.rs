//! `pipeline_sos` — a pipelined summary-of-summaries for the structural layers,
//! mirroring the `compress_tools` approach: the model does the *judgment*
//! (group / abstract the child skeletons one level up); a deterministic pass
//! does the *faithful reproduction* (validate every name against the real
//! children, dropping anything invented).
//!
//! Motivation: a single-pass SoS over directory listings either accumulates
//! (no abstraction) or fabricates paths (`candle-llama/0.12.3`, `models/mistral`).
//! Separating the two — like categorize→assign for tools — gives a clean,
//! abstracted, *and* fabrication-free roll-up.
//!
//! ```bash
//! cargo run -p zend --example pipeline_sos --features cuda --release
//! ```

use std::collections::BTreeSet;
use std::time::Instant;

use candle_conversation::models::Model;
use candle_conversation::SamplingConfig;

/// The child summaries this SoS rolls up — already-compressed structural
/// skeletons (a directory path then its files/modules by name), exactly the
/// shape the leaf compression produces.
fn children() -> Vec<&'static str> {
    vec![
        "(workspace root) - CHANGELOG.md - CLAUDE.md - Cargo.toml - README.md - candle-core/ - \
         candle-nn/ - candle-transformers/ - candle-kernels/ - candle-conversation/ - \
         candle-examples/",
        "candle-core/src/ - tensor.rs - device.rs - dtype.rs - shape.rs - cpu_backend/ - \
         cuda_backend/ - metal_backend/ - quantized/",
        "candle-nn/src/kv_cache/ - mod.rs - cache.rs - rotating.rs - arena_table.rs - chunked/",
        "candle-transformers/src/models/ - qwen3.rs - llama.rs - quantized_qwen3.rs - mod.rs",
        "candle-kernels/src/ - lib.rs - build.rs - paged-decode/ - paged-prefill/ - quantized/",
    ]
}

/// A token is a name if it carries a name-ish character. Strips list markers and
/// trailing punctuation so `candle-core/`, `- tensor.rs`, `(workspace` normalise.
fn normalize(tok: &str) -> Option<String> {
    let t = tok
        .trim()
        .trim_start_matches(['-', '*', '#', '(', ' '])
        .trim_end_matches([')', ',', ':', ' '])
        .trim_end_matches('/');
    if t.is_empty() {
        return None;
    }
    // Keep things that look like a file or a path/module name.
    if t.contains('.')
        || t.contains('/')
        || t.chars()
            .all(|c| c.is_ascii_alphanumeric() || c == '_' || c == '-')
    {
        Some(t.to_ascii_lowercase())
    } else {
        None
    }
}

/// Every real name that appears anywhere in the children.
fn valid_names(children: &[&str]) -> BTreeSet<String> {
    let mut set = BTreeSet::new();
    for c in children {
        for tok in c.split_whitespace() {
            if let Some(n) = normalize(tok) {
                set.insert(n);
            }
            // also index path components (candle-nn/src/kv_cache → each part)
            if let Some(n) = normalize(tok) {
                for part in n.split('/') {
                    if !part.is_empty() {
                        set.insert(part.to_string());
                    }
                }
            }
        }
    }
    set
}

fn main() -> anyhow::Result<()> {
    let device = match candle::Device::cuda_if_available(0) {
        Ok(d) if d.is_cuda() => d,
        _ => {
            eprintln!("pipeline_sos needs a CUDA device. Aborting.");
            return Ok(());
        }
    };

    let tmp = tempfile::tempdir()?;
    let mut builder = Model::Qwen3_30B_A3B_Q6
        .builder()
        .sampling(SamplingConfig::top_k_top_p(40, 0.9, 0.5).with_repeat_penalty(1.1))
        .seed(42)
        .max_response_tokens(512)
        .max_concurrent(4)
        .workspace_path(tmp.path());
    let config = builder.conversation_config();

    eprintln!("=== Loading {} ===", Model::Qwen3_30B_A3B_Q6);
    let t0 = Instant::now();
    let engine = builder
        .engine(&device)
        .map_err(|e| anyhow::anyhow!("engine: {e}"))?;
    eprintln!("    loaded in {:.1}s", t0.elapsed().as_secs_f64());

    let children = children();
    let valid = valid_names(&children);
    let numbered: String = children
        .iter()
        .enumerate()
        .map(|(i, c)| format!("{}. {c}", i + 1))
        .collect::<Vec<_>>()
        .join("\n");

    // ── Stage 1: the model groups/abstracts the child skeletons one level up.
    let sys = "You summarise a software repository's structure one level up. You are given \
        several directory listings (a directory path, then its files/modules by name). Produce \
        ONE higher-level structural skeleton: a line per top-level crate or directory, each \
        followed by the significant module and file names under it. You may drop minor files, \
        but use ONLY names that appear in the listings — never invent a file, module, crate, \
        version, or path. Output `## <directory>` then its names, comma-separated. No prose, no \
        descriptions, no versions.";
    let mut conv = engine
        .new_conversation(sys, config)
        .map_err(|e| anyhow::anyhow!("conversation: {e}"))?;
    eprintln!("Stage 1: abstracting {} child skeletons...", children.len());
    let resp = conv
        .send_turn(&format!(
            "{numbered}\nProduce one grouped higher-level skeleton of the structure above."
        ))
        .map_err(|e| anyhow::anyhow!("send: {e}"))?;

    println!("\n================ STAGE 1 — RAW MODEL ROLL-UP ================");
    println!("{}", resp.text.trim());

    // ── Stage 2: validate every name against the real children; drop inventions.
    println!("\n================ STAGE 2 — VALIDATED (fabrications dropped) ================");
    let mut kept = 0usize;
    let mut dropped: Vec<String> = Vec::new();
    for block in resp.text.split("##").skip(1) {
        // The model runs blocks onto one line, so split the header (first token)
        // from its names (the rest) rather than relying on newlines.
        let block = block.trim();
        if block.is_empty() {
            continue;
        }
        let mut it = block.splitn(2, char::is_whitespace);
        let header_raw = it.next().unwrap_or("").trim();
        let rest = it.next().unwrap_or("");
        let header_ok = normalize(header_raw)
            .map(|h| h.split('/').all(|p| p.is_empty() || valid.contains(p)))
            .unwrap_or(false);
        if !header_ok {
            if !header_raw.is_empty() {
                dropped.push(format!("dir:{header_raw}"));
            }
            continue;
        }
        let mut names: Vec<String> = Vec::new();
        for tok in rest.split([',', ' ']) {
            let Some(n) = normalize(tok) else {
                continue;
            };
            let leaf = n.split('/').next_back().unwrap_or(&n);
            if valid.contains(&n) || valid.contains(leaf) {
                let clean = tok.trim().trim_matches(',').to_string();
                if !clean.is_empty() && !names.contains(&clean) {
                    names.push(clean);
                    kept += 1;
                }
            } else {
                dropped.push(n);
            }
        }
        println!("\n## {header_raw}");
        if !names.is_empty() {
            println!("  {}", names.join(", "));
        }
    }
    println!(
        "\nkept {kept} real names; dropped {} fabricated tokens{}",
        dropped.len(),
        if dropped.is_empty() {
            String::new()
        } else {
            format!(": {}", dropped.join(", "))
        }
    );
    println!("\n=============================================================");
    Ok(())
}
