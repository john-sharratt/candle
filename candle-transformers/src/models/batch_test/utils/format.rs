//! Formats the prediction, regret and latency figures the reports print.
//!
//! Each figure is a count over a total, or `-` when nothing was judged, so a
//! measurement that did not happen cannot read as a zero. The think-block
//! stripper leaves only a run's final answer for comparison and printing, and
//! an unclosed block drops everything after its opening tag. The diff helper
//! shows only where two strings differ, and it cuts at a character boundary so
//! it never splits a multi-byte character.

use super::*;

/// One prediction figure per hop as `a/b/c/d/e` percentages — `hits / of` each
/// hop — or `-` when nothing was judged.
pub(super) fn per_hop(hits: &[usize], of: &[usize]) -> String {
    if of.iter().all(|&n| n == 0) {
        return "-".to_string();
    }
    hits.iter()
        .zip(of)
        .map(|(&h, &n)| format!("{:.0}", PipelineStats::percent(h, n)))
        .collect::<Vec<_>>()
        .join("/")
}

/// A share of judged claims as `p% of n`, or `-` when none were judged.
pub(super) fn regret(hits: usize, judged: usize) -> String {
    if judged == 0 {
        return "-".to_string();
    }
    format!("{:.1}% of {judged}", PipelineStats::percent(hits, judged))
}

/// One read source's cell in the expert-pipeline table: reads, the mean, the
/// slowest, and how many were slow.
pub(super) fn read_latency_cell(l: &ReadLatency) -> String {
    match l.mean_secs() {
        None => "-".to_string(),
        Some(mean) => format!(
            "{} · mean {:.2} ms · max {:.1} ms · {} slow",
            l.reads,
            mean * 1e3,
            l.max().as_secs_f64() * 1e3,
            l.slow
        ),
    }
}

/// Remove all `<think>…</think>` blocks from `text`.
/// Used to strip chain-of-thought reasoning from model output when `suppress_thinking` is set,
/// so that comparison and printing only see the final answer.
pub(super) fn strip_thinking_blocks(text: &str) -> String {
    const OPEN: &str = "<think>";
    const CLOSE: &str = "</think>";
    let mut out = String::with_capacity(text.len());
    let mut rest = text;
    while let Some(open_pos) = rest.find(OPEN) {
        out.push_str(&rest[..open_pos]);
        rest = &rest[open_pos + OPEN.len()..];
        if let Some(close_pos) = rest.find(CLOSE) {
            rest = &rest[close_pos + CLOSE.len()..];
        } else {
            // Unclosed <think> — drop everything from here.
            break;
        }
    }
    out.push_str(rest);
    out
}

/// Format a diff between expected and actual strings (compact, differences only)
pub(super) fn format_diff(expected: &str, actual: &str) -> String {
    // Truncate expected to match actual length for fair comparison
    // Use floor_char_boundary to avoid slicing in the middle of a UTF-8 character
    let expected_truncated = if expected.len() > actual.len() {
        let boundary = actual.len();
        // Find the last valid char boundary at or before the target position
        let safe_boundary = expected
            .char_indices()
            .take_while(|(i, _)| *i <= boundary)
            .last()
            .map(|(i, _)| i)
            .unwrap_or(0);
        &expected[..safe_boundary]
    } else {
        expected
    };

    let exp_lines: Vec<&str> = expected_truncated.lines().take(10).collect();
    let act_lines: Vec<&str> = actual.lines().take(10).collect();

    let mut output = String::from("\n");
    let mut diff_count = 0;
    for (i, (exp, act)) in exp_lines.iter().zip(act_lines.iter()).enumerate() {
        if exp != act {
            output.push_str(&format!("Line {}: - {}\n        + {}\n", i + 1, exp, act));
            diff_count += 1;
            if diff_count >= 3 {
                break;
            }
        }
    }
    if exp_lines.len() != act_lines.len() {
        output.push_str(&format!(
            "Length: expected {} lines, got {}\n",
            exp_lines.len(),
            act_lines.len()
        ));
    }
    output
}
