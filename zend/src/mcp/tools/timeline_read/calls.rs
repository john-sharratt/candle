//! `view: calls`: each turn's tool calls, one line each — the name and its
//! arguments as written — with a call identical to one the conversation already
//! made flagged as a repeat, and where it was first made. A run that went round
//! in circles reads as a column of repeats; a call that dropped the argument
//! its reasoning named reads as the same call over and over.

use std::collections::HashMap;

use host::substrate_views::TimelineDetail;

use super::head;
use crate::round_timing::{rounds, total};
use crate::tools::extract_tool_calls;

/// How much of a call's arguments a line shows.
const ARGS: usize = 200;

/// A call's arguments as one line, cut at [`ARGS`] characters, and whether
/// they were cut.
fn args_line(args: &serde_json::Value) -> (String, bool) {
    let text = args.to_string();
    match text.char_indices().nth(ARGS) {
        Some((at, _)) => (format!("{}…", &text[..at]), true),
        None => (text, false),
    }
}

/// The page's last line when a call on it was cut short: where its whole
/// arguments are.
const WHOLE: &str = "A call ending in … was cut short here: `view: turn` with that `turn` \
                     reads it whole.";

/// The calls of turns `offset..offset + limit`, a line each, repeats of any
/// earlier call of the conversation flagged with where it was first made.
pub fn render(t: &TimelineDetail, offset: usize, limit: usize) -> String {
    let mut first: HashMap<String, u32> = HashMap::new();
    let (mut calls, mut repeats) = (0, 0);
    let mut cut = false;
    let mut lines = Vec::new();
    let timed = rounds(&t.custom);
    for (at, turn) in t.turns.iter().enumerate() {
        let shown = (offset..offset + limit).contains(&at);
        if let (true, Some(timing)) = (shown, timed.get(&turn.index)) {
            lines.push(format!("#{} round: {timing}", turn.index));
        }
        let made = extract_tool_calls(&turn.assistant);
        if shown && made.is_empty() {
            lines.push(format!("#{} no calls", turn.index));
        }
        for c in made {
            let key = format!("{} {}", c.name, c.arguments);
            let (args, was_cut) = args_line(&c.arguments);
            let line = format!("{} {args}", c.name);
            let seen = first.get(&key).copied();
            if seen.is_none() {
                first.insert(key, turn.index);
            }
            if !shown {
                continue;
            }
            cut |= was_cut;
            calls += 1;
            match seen {
                Some(was) => {
                    repeats += 1;
                    lines.push(format!("#{} {line} — repeat of #{was}'s", turn.index));
                }
                None => lines.push(format!("#{} {line}", turn.index)),
            }
        }
    }
    let mut out = head(t);
    let (n, sum) = total(timed.values());
    if n > 0 {
        out.push_str(&format!("\ntime over {n} timed rounds: {sum}"));
    }
    out.push_str(&format!(
        "\ncalls of turns {}–{}: {calls}, of which {repeats} repeat a call already made",
        offset.min(t.turns.len()),
        (offset + limit).min(t.turns.len())
    ));
    for l in lines {
        out.push('\n');
        out.push_str(&l);
    }
    if cut {
        out.push('\n');
        out.push_str(WHOLE);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::super::tests::{timeline, turn_view};
    use super::*;

    /// **A repeated call is flagged with where it was first made**, whether
    /// the repeat is in the same turn or a later one; a turn with no calls
    /// says so.
    #[test]
    fn a_repeated_call_is_flagged_with_where_it_was_first_made() {
        let call = |args: &str| {
            format!("<tool_call>\n{{\"name\": \"file_list\", \"arguments\": {args}}}\n</tool_call>")
        };
        let root = call(r#"{"repo": "candle", "page": 1}"#);
        let docs = call(r#"{"repo": "candle", "path": "docs", "page": 1}"#);
        let mut t = timeline();
        t.turns = vec![
            turn_view(0, "tour", &format!("<think>x</think>\n{root}\n{root}")),
            turn_view(1, "r", &format!("{docs}\n{root}")),
            turn_view(2, "r", "Here it is."),
        ];
        let text = render(&t, 0, 10);
        assert!(
            text.contains("calls of turns 0–3: 4, of which 2 repeat a call already made"),
            "{text}"
        );
        let lines: Vec<&str> = text
            .lines()
            .skip_while(|l| !l.starts_with("calls of"))
            .skip(1)
            .collect();
        assert_eq!(lines.len(), 5, "{text}");
        assert!(
            lines[0].starts_with("#0 file_list {") && !lines[0].contains("repeat"),
            "{text}"
        );
        assert!(
            lines[1].starts_with("#0 file_list {") && lines[1].ends_with("— repeat of #0's"),
            "{text}"
        );
        assert!(
            lines[2].starts_with("#1 file_list {") && lines[2].contains("\"path\":\"docs\""),
            "{text}"
        );
        assert!(!lines[2].contains("repeat"), "{text}");
        assert!(
            lines[3].starts_with("#1 file_list {") && lines[3].ends_with("— repeat of #0's"),
            "{text}"
        );
        assert_eq!(lines[4], "#2 no calls");
    }

    /// **A call cut short names where it reads whole**, once, as the page's
    /// last line; a page with no cut call says nothing of it.
    #[test]
    fn a_cut_call_names_where_it_reads_whole() {
        let long = "x".repeat(ARGS + 10);
        let call = format!(
            "<tool_call>\n{{\"name\": \"write\", \"arguments\": {{\"content\": \"{long}\"}}}}\n</tool_call>"
        );
        let mut t = timeline();
        t.turns = vec![turn_view(0, "a", &call), turn_view(1, "b", &call)];
        let text = render(&t, 0, 2);
        assert_eq!(text.lines().last(), Some(WHOLE), "{text}");
        assert_eq!(text.matches(WHOLE).count(), 1, "{text}");
        let short = "<tool_call>\n{\"name\": \"datetime\", \"arguments\": {}}\n</tool_call>";
        t.turns = vec![turn_view(0, "a", short)];
        assert!(!render(&t, 0, 1).contains(WHOLE));
    }

    /// **A timed round shows where its time went**, before its calls, and the
    /// conversation's rounds are totalled under the header.
    #[test]
    fn a_rounds_time_shows_beside_its_calls() {
        use crate::round_timing::{key, RoundTiming};
        let call = "<tool_call>\n{\"name\": \"datetime\", \"arguments\": {}}\n</tool_call>";
        let mut t = timeline();
        t.turns = vec![turn_view(0, "a", call), turn_view(1, "b", "Done.")];
        let timing = RoundTiming::of(10_000, 512, 1_400, 210, 6_100);
        t.custom.insert(key(0), timing.value());
        let text = render(&t, 0, 2);
        assert!(
            text.contains(
                "\ntime over 1 timed rounds: queued 2.5 s, prefill 512 tok in 1.4 s, decode 210 \
                 tok in 6.1 s\n"
            ),
            "{text}"
        );
        assert!(
            text.contains(
                "\n#0 round: queued 2.5 s, prefill 512 tok in 1.4 s, decode 210 tok in 6.1 s\n#0 \
                 datetime {}"
            ),
            "{text}"
        );
    }

    /// **A repeat of a call made before the page is still a repeat.**
    #[test]
    fn a_repeat_of_a_call_before_the_page_is_still_flagged() {
        let root = "<tool_call>\n{\"name\": \"datetime\", \"arguments\": {}}\n</tool_call>";
        let mut t = timeline();
        t.turns = vec![turn_view(0, "a", root), turn_view(1, "b", root)];
        let text = render(&t, 1, 1);
        assert!(text.contains("#1 datetime {} — repeat of #0's"), "{text}");
        assert!(!text.contains("\n#0 "), "{text}");
    }
}
