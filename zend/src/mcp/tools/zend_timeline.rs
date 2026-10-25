//! `zend_timeline {id, offset?, limit?}`: one conversation in the substrate
//! (`docs/claude_zend_gateway.md` §8.5) — `GET
//! /v1/substrate/zend/timeline/:tl`, read in process: what it is, its lineage,
//! and its turns a page at a time, each cut to its first line. An Admin's.

use serde_json::{json, Value};

use super::{Answer, Ctx, LOADING};
use crate::api::substrate::TimelineDetail;
use crate::mcp::ops;
use crate::mcp::workers::first_line;

pub const NAME: &str = "zend_timeline";

const DEFAULT_LIMIT: usize = 40;
const MAX_LIMIT: usize = 400;
/// How much of each half of a turn a page shows.
const HALF: usize = 160;

pub fn definition() -> Value {
    json!({
        "name": NAME,
        "description": "Read one conversation in zend's substrate by its timeline id, as `zend_layer` lists \
            it: its label, layer, tokens, metadata and lineage, then its turns a page at a time — each \
            turn's index, kind (normal, or a summary of turns or of summaries), tokens, and the first line \
            of each half. An Admin's.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "id": {"type": "string", "description": "The timeline id, a decimal number."},
                "offset": {"type": "integer", "minimum": 0, "description": "The first turn (0 by default)."},
                "limit": {"type": "integer", "minimum": 1, "maximum": MAX_LIMIT, "description": "How many turns (40 by default)."}
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
        let keys: Vec<&str> = t.custom.keys().map(String::as_str).collect();
        out.push_str(&format!("\nmetadata keys: {}", keys.join(", ")));
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
        return Answer::error("`id` is a timeline id, a decimal number as `zend_layer` lists it");
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
    match ctx.session.substrate_timeline(raw) {
        Some(t) => Answer::ok(render(&t, offset, limit)),
        None => Answer::error(format!("there is no timeline {raw}")),
    }
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use super::*;
    use crate::api::substrate::TurnView;

    #[test]
    fn a_timeline_reads_as_its_head_then_its_turns() {
        let t = TimelineDetail {
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
        };
        assert_eq!(
            render(&t, 0, 10),
            "timeline 77 — Verify (mcp:o:review/verify), layer dialogue group chat, 1200 tokens, \
             1 turns\nlineage: 1 → 77\nmetadata keys: decode_band\nturns 0–1:\n#0 normal 40 tokens\n  \
             user: check the expiry\n  assistant: It is inclusive."
        );
    }
}
