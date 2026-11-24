//! What the model had in view of each expected item when it began to answer.
//!
//! An answer file *placed* (`placement.rs`) says some turn of a reading of it
//! was in the slot. That is not the same as the item the answer must name
//! being there: a reading is many turns, and a projection places some of
//! them. Here each `expect` item is looked for, by the same spellings recall
//! matches (`score::names`), in the text the model was actually given:
//!
//! - for a memory arm, the turns the point governing the answer's first token
//!   injected — its materialized spine, each turn's sealed body as the
//!   projection panel reads it (`GET /v1/conversations/{id}/projections/{turn}/{event}`);
//! - for R1, the files block its prompt carried.
//!
//! `in_files` looks only in the text of the question's answer files — the
//! turns of their readings, or their blocks — and `anywhere` in everything
//! the model was shown. The two arms are measured the same way, so an item in
//! view and still not named is a reading failure on either side.

use std::collections::{HashMap, HashSet};

use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::placement::is_file;
use crate::rag::context::{first_chars, header, Packed};
use crate::score::names;

/// One answer's view of its expected items, in `expect` order.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct View {
    /// Whether the item was in the text of an answer file the model was given.
    pub in_files: Vec<bool>,
    /// Whether the item was anywhere in what the model was given.
    pub anywhere: Vec<bool>,
    /// Whether the answer named the item.
    pub named: Vec<bool>,
    /// How much of each answer file's reading the point placed, in `files`
    /// order; empty for R1, whose files are whole or cut by the budget.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub files: Vec<FileShare>,
}

/// The turns of one answer file's reading the point weighed and placed.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct FileShare {
    pub turns: usize,
    pub placed: usize,
}

/// The index of the point of `starts` governing token `t`: the last that
/// starts at or before it.
pub fn governing(starts: &[u32], t: u32) -> Option<usize> {
    starts.iter().rposition(|&s| s <= t)
}

/// `expect` looked for in `files_text` and `all_text`.
fn looked_up(expect: &[String], files_text: &str, all_text: &str) -> (Vec<bool>, Vec<bool>) {
    (
        expect.iter().map(|e| names(files_text, e)).collect(),
        expect.iter().map(|e| names(all_text, e)).collect(),
    )
}

/// The text of one turn as the panel gives it: the whole turn, or its halves
/// when its token record was lost.
fn turn_text(t: &Value) -> String {
    match t.get("text").and_then(Value::as_str) {
        Some(text) => text.to_string(),
        None => {
            let half = |k: &str| t.get(k).and_then(Value::as_str).unwrap_or_default();
            format!("{}\n{}", half("user"), half("assistant"))
        }
    }
}

/// `view` with what `answer` named of `expect`.
pub fn with_named(mut view: View, answer: &str, expect: &[String]) -> View {
    view.named = expect.iter().map(|e| names(answer, e)).collect();
    view
}

/// The view a full projection point gives of `expect`, its answer files
/// `files`.
pub fn of_point(detail: &Value, files: &[String], expect: &[String]) -> View {
    let empty = Vec::new();
    let arr = |v: Option<&Value>| v.and_then(Value::as_array).unwrap_or(&empty).clone();
    let span = detail.get("span");
    let pieces = arr(span.and_then(|s| s.get("pieces")));
    let answer_timelines: HashSet<u64> = pieces
        .iter()
        .filter(|p| p.get("kind").and_then(Value::as_str) == Some("Reading"))
        .filter(|p| {
            p.get("path")
                .and_then(Value::as_str)
                .is_some_and(|path| files.iter().any(|f| is_file(path, f)))
        })
        .filter_map(|p| p.get("timeline").and_then(Value::as_u64))
        .collect();
    let bodies: HashMap<(u64, u64), String> = arr(detail.get("turn_content"))
        .iter()
        .filter_map(|t| {
            let tl = t.get("timeline")?.as_u64()?;
            let ix = t.get("index")?.as_u64()?;
            Some(((tl, ix), turn_text(t)))
        })
        .collect();
    let mut files_text = String::new();
    let mut all_text = String::new();
    for m in arr(span.and_then(|s| s.get("materialized"))) {
        if m.get("kind").and_then(Value::as_str) != Some("turn") {
            continue;
        }
        let Some(turn) = m.get("turn") else { continue };
        let (Some(tl), Some(ix)) = (
            turn.get("timeline").and_then(Value::as_u64),
            turn.get("index").and_then(Value::as_u64),
        ) else {
            continue;
        };
        let Some(body) = bodies.get(&(tl, ix)) else {
            continue;
        };
        if answer_timelines.contains(&tl) {
            files_text.push_str(body);
            files_text.push('\n');
        }
        all_text.push_str(body);
        all_text.push('\n');
    }
    for s in arr(detail.get("section_content")) {
        all_text.push_str(s.get("content").and_then(Value::as_str).unwrap_or_default());
        all_text.push('\n');
    }
    let (in_files, anywhere) = looked_up(expect, &files_text, &all_text);
    let share = |f: &String| {
        pieces
            .iter()
            .find(|p| {
                p.get("kind").and_then(Value::as_str) == Some("Reading")
                    && p.get("path")
                        .and_then(Value::as_str)
                        .is_some_and(|path| is_file(path, f))
            })
            .map(|p| FileShare {
                turns: p.get("turns").and_then(Value::as_u64).unwrap_or(0) as usize,
                placed: p.get("placed").and_then(Value::as_u64).unwrap_or(0) as usize,
            })
            .unwrap_or_default()
    };
    View {
        in_files,
        anywhere,
        named: Vec::new(),
        files: files.iter().map(share).collect(),
    }
}

/// The view R1's files block gives of `expect`, its answer files `files`.
pub fn of_packed(packed: &Packed, files: &[String], expect: &[String]) -> View {
    let mut files_text = String::new();
    let mut at = 0;
    for f in &packed.files {
        let head = header(&f.path);
        let Some(found) = packed.text[at..].find(&head) else {
            continue;
        };
        let start = at + found + head.len();
        let body = first_chars(&packed.text[start..], f.chars);
        if files.iter().any(|a| is_file(&f.path, a)) {
            files_text.push_str(&head);
            files_text.push_str(body);
            files_text.push('\n');
        }
        at = start + body.len();
    }
    let (in_files, anywhere) = looked_up(expect, &files_text, &packed.text);
    View {
        in_files,
        anywhere,
        named: Vec::new(),
        files: Vec::new(),
    }
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;
    use crate::rag::context::RetrievedFile;

    fn expect() -> Vec<String> {
        vec!["carve.rs".into(), "work_dtype".into(), "gone|absent".into()]
    }

    #[test]
    fn the_answer_is_governed_by_the_last_point_at_or_before_it() {
        assert_eq!(governing(&[0, 0, 65, 132, 301], 301), Some(4));
        assert_eq!(governing(&[0, 0, 65, 132, 301], 300), Some(3));
        assert_eq!(governing(&[0, 65], 0), Some(0));
        assert_eq!(governing(&[], 10), None);
    }

    /// **Only the spine's turns are in view**, and an answer file's text is
    /// what its reading's injected turns say: the selected-but-not-injected
    /// body, and the turn of another file, count for `in_files` not at all.
    #[test]
    fn a_point_shows_what_its_spine_injected() {
        let detail = json!({
            "span": {
                "pieces": [
                    {"timeline": 7, "kind": "Reading", "path": "candle/nn/carve.rs",
                     "turns": 5, "placed": 2},
                    {"timeline": 9, "kind": "Reading", "path": "candle/nn/other.rs",
                     "turns": 3, "placed": 1},
                ],
                "materialized": [
                    {"kind": "glue", "text": "<|im_start|>"},
                    {"kind": "turn", "turn": {"timeline": 7, "index": 1}},
                    {"kind": "turn", "turn": {"timeline": 9, "index": 0}},
                ],
            },
            "turn_content": [
                {"timeline": 7, "index": 1, "text": "fn carve() { work_dtype() }", "user": "", "assistant": ""},
                {"timeline": 7, "index": 2, "text": "gone", "user": "", "assistant": ""},
                {"timeline": 9, "index": 0, "user": "see carve.rs", "assistant": "absent"},
            ],
            "section_content": [{"name": "seed", "content": "nothing"}],
        });
        let v = of_point(&detail, &["nn/carve.rs".to_string()], &expect());
        assert_eq!(v.in_files, vec![false, true, false]);
        assert_eq!(v.anywhere, vec![true, true, true]);
        assert_eq!(
            v.files,
            vec![FileShare {
                turns: 5,
                placed: 2
            }]
        );
    }

    /// An answer file the point never weighed shares nothing.
    #[test]
    fn an_unweighed_file_has_no_share() {
        let v = of_point(&json!({}), &["a.rs".to_string()], &expect());
        assert_eq!(v.files, vec![FileShare::default()]);
        assert_eq!(v.in_files, vec![false; 3]);
    }

    /// **R1's answer file is its block, header included**, and only the
    /// characters the budget let in: a file cut before the item does not
    /// hold it.
    #[test]
    fn a_packed_block_holds_what_the_budget_let_in() {
        let text = "=== x/carve.rs ===\nwork_d\n\n=== y.rs ===\nabsent\n\n".to_string();
        let file = |path: &str, rank, chars, cut| RetrievedFile {
            path: path.into(),
            rank,
            score: 1.0,
            chars,
            cut,
            answer_file: false,
        };
        let packed = Packed {
            text,
            files: vec![file("x/carve.rs", 1, 6, true), file("y.rs", 2, 6, false)],
        };
        let v = of_packed(&packed, &["x/carve.rs".to_string()], &expect());
        assert_eq!(v.in_files, vec![true, false, false]);
        assert_eq!(v.anywhere, vec![true, false, true]);
        assert!(v.files.is_empty());
    }

    /// An item is named by any of its spellings, as recall scores it.
    #[test]
    fn the_view_records_what_the_answer_named() {
        let v = with_named(
            View::default(),
            "In `carve.rs`; the rows are ABSENT.",
            &expect(),
        );
        assert_eq!(v.named, vec![true, false, true]);
    }
}
