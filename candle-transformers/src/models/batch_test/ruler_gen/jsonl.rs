//! Loads RULER samples from the canonical JSONL files produced by the reference data
//! generator, using a minimal parser.
//!
//! Each line carries an input prompt and its expected outputs. Lines missing either
//! field are skipped. The parser reads only those two fields and does not depend on
//! a JSON library, since the RULER outputs are simple identifiers with no embedded
//! commas or brackets. Files are named by task and context length, so a sample set
//! is found from its task and length alone.

use super::*;

// ── JSONL loader ──────────────────────────────────────────────────────────────

/// Load RULER samples from a canonical JSONL file.
///
/// Expected line format (RULER data-gen output):
/// ```json
/// {"index": 0, "input": "<prompt>", "outputs": ["42"]}
/// ```
///
/// Lines missing `input` or `outputs` are silently skipped.
pub fn load_ruler_samples_jsonl(path: &std::path::Path) -> Result<Vec<RulerSample>> {
    let text = std::fs::read_to_string(path)
        .map_err(|e| candle::Error::Msg(format!("JSONL read {:?}: {}", path, e)))?;
    let mut samples = Vec::new();
    for (lineno, line) in text.lines().enumerate() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        // Minimal JSON parse — avoid pulling in serde_json as a dep.
        // We only need "input" (string) and "outputs" (array of strings).
        let input = extract_json_str(line, "input").ok_or_else(|| {
            candle::Error::Msg(format!(
                "JSONL {:?} line {}: missing \"input\" field",
                path,
                lineno + 1
            ))
        })?;
        let outputs = extract_json_str_array(line, "outputs").ok_or_else(|| {
            candle::Error::Msg(format!(
                "JSONL {:?} line {}: missing \"outputs\" field",
                path,
                lineno + 1
            ))
        })?;
        samples.push(RulerSample { input, outputs });
    }
    Ok(samples)
}

/// Extract a single JSON string value by key from a flat JSON object line.
/// Returns `None` if not found. Handles basic `\"` escape sequences.
pub(super) fn extract_json_str(json: &str, key: &str) -> Option<String> {
    let needle = format!("\"{key}\":");
    let start = json.find(&needle)? + needle.len();
    let rest = json[start..].trim_start();
    if !rest.starts_with('"') {
        return None;
    }
    let inner = &rest[1..];
    let mut value = String::new();
    let mut escaped = false;
    for ch in inner.chars() {
        if escaped {
            match ch {
                'n' => value.push('\n'),
                't' => value.push('\t'),
                'r' => value.push('\r'),
                other => value.push(other),
            }
            escaped = false;
        } else if ch == '\\' {
            escaped = true;
        } else if ch == '"' {
            return Some(value);
        } else {
            value.push(ch);
        }
    }
    None
}

/// Extract a JSON array of strings `["a","b",...]` by key.
pub(super) fn extract_json_str_array(json: &str, key: &str) -> Option<Vec<String>> {
    let needle = format!("\"{key}\":");
    let start = json.find(&needle)? + needle.len();
    let rest = json[start..].trim_start();
    if !rest.starts_with('[') {
        return None;
    }
    let inner = &rest[1..];
    let end = inner.find(']')?;
    let items_str = &inner[..end];
    // Split on `","` boundaries (naive but sufficient for RULER outputs which
    // are simple identifiers — no embedded commas or brackets in values).
    let mut result = Vec::new();
    let mut remaining = items_str;
    loop {
        remaining = remaining.trim_start();
        if remaining.is_empty() {
            break;
        }
        if remaining.starts_with('"') {
            remaining = &remaining[1..];
            let close = remaining.find('"').unwrap_or(remaining.len());
            result.push(remaining[..close].replace("\\\"", "\""));
            remaining = &remaining[close..];
            if remaining.starts_with('"') {
                remaining = &remaining[1..];
            }
        }
        // Skip past comma
        if let Some(comma) = remaining.find(',') {
            remaining = &remaining[comma + 1..];
        } else {
            break;
        }
    }
    if result.is_empty() {
        None
    } else {
        Some(result)
    }
}

/// Build the JSONL filename for a (task, ctx_len) pair, e.g. `niah_single_1_4096.jsonl`.
pub fn ruler_jsonl_filename(task: RulerTask, ctx_len: usize) -> String {
    format!("{task}_{ctx_len}.jsonl")
}

pub(super) fn extract_json_bool(json: &str, key: &str) -> Option<bool> {
    let needle = format!("\"{key}\":");
    let start = json.find(&needle)? + needle.len();
    let rest = json[start..].trim_start();
    if rest.starts_with("true") {
        Some(true)
    } else if rest.starts_with("false") {
        Some(false)
    } else {
        None
    }
}

pub(super) fn extract_json_u64(json: &str, key: &str) -> Option<u64> {
    let needle = format!("\"{key}\":");
    let start = json.find(&needle)? + needle.len();
    let rest = json[start..].trim_start();
    let end = rest
        .find(|c: char| !c.is_ascii_digit())
        .unwrap_or(rest.len());
    if end == 0 {
        return None;
    }
    rest[..end].parse().ok()
}
