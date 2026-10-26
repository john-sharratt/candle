//! `timeline_read {id, offset?, limit?, turn?, from?}`: one conversation in the
//! substrate (`docs/claude_zend_gateway.md` §8.5) — `GET
//! /v1/substrate/zend/timeline/:tl`, read in process: what it is, its
//! metadata, its lineage, and its turns a page at a time, each cut to its
//! first line; or, with `turn`, that one turn in full, a page of characters at
//! a time. Admins only.

use serde_json::{json, Value};

use super::{Answer, Ctx, LOADING};
use crate::api::substrate::TimelineDetail;
use crate::mcp::ops;
use crate::mcp::workers::first_line;

pub const NAME: &str = "timeline_read";

const DEFAULT_LIMIT: usize = 40;
const MAX_LIMIT: usize = 400;
/// How much of each half of a turn a page shows.
const HALF: usize = 160;
/// How much of a metadata value the head shows.
const META_VALUE: usize = 80;
/// How many characters of one turn a `turn` read answers at a time.
const TURN_PAGE: usize = 16_000;

pub fn definition() -> Value {
    json!({
        "name": NAME,
        "description": "Read one stored conversation — a timeline in zend's substrate, the store everything \
            zend keeps lives in — by the id `layer_read` lists, or a worker's own: its label, layer, tokens, \
            metadata (the dials a worker ran with among it) and lineage, then its turns a page at a time — \
            each with its index, kind (normal, or a summary of turns or of summaries), tokens, and the first \
            line of each half. With `turn`, that one turn in full instead: the user half (a tool call's \
            results arrive here) and the assistant half (its thinking and the tool calls it made), a page \
            of characters at a time from `from`. Admins only.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "id": {"type": "string", "description": "The timeline id, a decimal number."},
                "offset": {"type": "integer", "minimum": 0, "description": "The first turn (0 by default)."},
                "limit": {"type": "integer", "minimum": 1, "maximum": MAX_LIMIT, "description": "How many turns (40 by default)."},
                "turn": {"type": "integer", "minimum": 0, "description": "One turn's index, to read it in full."},
                "from": {"type": "integer", "minimum": 0, "description": "With `turn`: the character to read from (0 by default)."}
            },
            "required": ["id"],
            "additionalProperties": false
        }
    })
}

/// A timeline's head and a page of its turns.
pub fn render(t: &TimelineDetail, offset: usize, limit: usize) -> String {
    let mut out = format!(
        "timeline {} — {} ({}), layer {} group {}, {} tokens, {} turns{}",
        t.timeline,
        if t.label.is_empty() {
            "untitled"
        } else {
            &t.label
        },
        if t.conv_id.is_empty() {
            "no conv_id"
        } else {
            &t.conv_id
        },
        t.layer,
        t.group,
        t.total_tokens,
        t.turn_count,
        if t.archived { ", archived" } else { "" }
    );
    if t.inherited_chain.len() > 1 {
        out.push_str(&format!("\nlineage: {}", t.inherited_chain.join(" → ")));
    }
    if !t.custom.is_empty() {
        // The values too: the dials a worker ran with (`dial_effort`,
        // `dial_think`, …) are what a review of its run asks after.
        let pairs: Vec<String> = t
            .custom
            .iter()
            .map(|(k, v)| format!("{k}={}", first_line(v, META_VALUE)))
            .collect();
        out.push_str(&format!("\nmetadata: {}", pairs.join(", ")));
    }
    out.push_str(&format!(
        "\nturns {}–{}:",
        offset.min(t.turns.len()),
        (offset + limit).min(t.turns.len())
    ));
    for turn in t.turns.iter().skip(offset).take(limit) {
        out.push_str(&format!(
            "\n#{} {} {} tokens\n  user: {}\n  assistant: {}",
            turn.index,
            turn.kind,
            turn.tokens,
            first_line(&turn.user, HALF),
            first_line(&turn.assistant, HALF)
        ));
    }
    out
}

/// One turn in full — both halves, the tool calls an assistant half makes and
/// the results a user half carries included — a page of [`TURN_PAGE`]
/// characters from character `from`, and where to read on.
pub fn render_turn(t: &TimelineDetail, index: u32, from: usize) -> Result<String, String> {
    let Some(turn) = t.turns.iter().find(|turn| turn.index == index) else {
        return Err(format!(
            "timeline {} has no turn {index}; it has {} turns",
            t.timeline, t.turn_count
        ));
    };
    let whole = format!("--- user\n{}\n--- assistant\n{}", turn.user, turn.assistant);
    let total = whole.chars().count();
    let page: String = whole.chars().skip(from).take(TURN_PAGE).collect();
    let to = (from + TURN_PAGE).min(total);
    let mut out = format!(
        "timeline {} turn #{} {}, {} tokens — characters {}–{to} of {total}",
        t.timeline,
        turn.index,
        turn.kind,
        turn.tokens,
        from.min(total)
    );
    if to < total {
        out.push_str(&format!(" (call again with `from: {to}` to read on)"));
    }
    out.push('\n');
    out.push_str(&page);
    Ok(out)
}

pub fn call(ctx: &Ctx, args: &Value) -> Answer {
    if let Err(e) = ops::admin_only(&ctx.caller, NAME) {
        return Answer::error(e);
    }
    let Some(raw) = args
        .get("id")
        .and_then(|v| {
            v.as_str()
                .map(str::to_string)
                .or_else(|| v.as_u64().map(|n| n.to_string()))
        })
        .and_then(|s| s.parse::<u64>().ok())
    else {
        return Answer::error("`id` is a timeline id, a decimal number as `layer_read` lists it");
    };
    let offset = args.get("offset").and_then(Value::as_u64).unwrap_or(0) as usize;
    let limit = (args
        .get("limit")
        .and_then(Value::as_u64)
        .unwrap_or(DEFAULT_LIMIT as u64) as usize)
        .clamp(1, MAX_LIMIT);
    if !ctx.session.is_loaded() {
        return Answer::error(LOADING);
    }
    let Some(t) = ctx.session.substrate_timeline(raw) else {
        return Answer::error(format!("there is no timeline {raw}"));
    };
    match args.get("turn").and_then(Value::as_u64) {
        Some(index) => {
            let from = args.get("from").and_then(Value::as_u64).unwrap_or(0) as usize;
            match u32::try_from(index) {
                Ok(index) => match render_turn(&t, index, from) {
                    Ok(text) => Answer::ok(text),
                    Err(e) => Answer::error(e),
                },
                Err(_) => Answer::error(format!("there is no turn {index}")),
            }
        }
        None => Answer::ok(render(&t, offset, limit)),
    }
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use super::*;
    use crate::api::substrate::TurnView;

    #[test]
    fn a_timeline_reads_as_its_head_then_its_turns() {
        let t = timeline();
        assert_eq!(
            render(&t, 0, 10),
            "timeline 77 — Verify (mcp:o:review/verify), layer dialogue group chat, 1200 tokens, \
             1 turns\nlineage: 1 → 77\nmetadata: decode_band=normal\nturns 0–1:\n#0 normal 40 \
             tokens\n  user: check the expiry\n  assistant: It is inclusive."
        );
    }

    fn timeline() -> TimelineDetail {
        TimelineDetail {
            timeline: "77".into(),
            conv_id: "mcp:o:review/verify".into(),
            label: "Verify".into(),
            archived: false,
            layer: "dialogue".into(),
            group: "chat".into(),
            total_tokens: 1200,
            custom: BTreeMap::from([("decode_band".into(), "normal".into())]),
            inherited_chain: vec!["1".into(), "77".into()],
            peaks: vec![],
            couplings: vec![],
            turn_count: 1,
            turns: vec![TurnView {
                index: 0,
                kind: "normal",
                height: 0,
                tokens: 40,
                no_think: false,
                children: vec![],
                coupled: false,
                peak: true,
                user: "check the expiry\nmore".into(),
                assistant: "It is inclusive.".into(),
                layout: None,
            }],
        }
    }

    /// **One turn reads in full, a page at a time**: both halves whole, and
    /// where to read on when they run past a page.
    #[test]
    fn one_turn_reads_in_full_a_page_at_a_time() {
        let mut t = timeline();
        assert_eq!(
            render_turn(&t, 0, 0).unwrap(),
            "timeline 77 turn #0 normal, 40 tokens — characters 0–61 of 61\n--- user\ncheck the \
             expiry\nmore\n--- assistant\nIt is inclusive."
        );
        assert_eq!(
            render_turn(&t, 3, 0).unwrap_err(),
            "timeline 77 has no turn 3; it has 1 turns"
        );
        t.turns[0].user = "x".repeat(TURN_PAGE + 5);
        let first = render_turn(&t, 0, 0).unwrap();
        assert!(first.contains(&format!(
            "characters 0–{TURN_PAGE} of {} (call again with `from: {TURN_PAGE}` to read on)",
            TURN_PAGE + 5 + 40
        )));
        let rest = render_turn(&t, 0, TURN_PAGE).unwrap();
        assert!(
            rest.ends_with("xxx\n--- assistant\nIt is inclusive."),
            "{rest}"
        );
    }
}
