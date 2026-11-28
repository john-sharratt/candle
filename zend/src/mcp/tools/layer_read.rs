//! `layer_read {name, offset?, limit?}`: the conversations one projection layer
//! holds (`docs/claude_zend_gateway.md` §8.5) — `GET
//! /v1/substrate/zend/layer/:name`, read in process. Admins only.

use host::substrate_views::{LayerConversations, PathNode};
use serde_json::{json, Value};

use super::{string, Answer, Ctx, LOADING};
use crate::mcp::ops;
use crate::mcp::ops_tools::OpsCall;
use crate::session::ZendSession;

pub const NAME: &str = "layer_read";

const DEFAULT_LIMIT: usize = 50;
const MAX_LIMIT: usize = 500;

pub fn definition() -> Value {
    json!({
        "name": NAME,
        "annotations": {"readOnlyHint": true},
        "description": "Read what one of the projection layers holds. A layer is one kind of analysis kept \
            about the repositories — `repo_map`, `code_reading`, `dialogue` — and workers read from it; \
            each piece of it is a stored \
            conversation, a timeline. Lists the layer's timelines, each with its id, label, repository, \
            turns and tokens, a page at a time; read one with `timeline_read`. Admins only.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "The layer."},
                "offset": {"type": "integer", "minimum": 0, "description": "The first conversation (0 by default)."},
                "limit": {"type": "integer", "minimum": 1, "maximum": MAX_LIMIT, "description": "How many (50 by default)."}
            },
            "required": ["name"],
            "additionalProperties": false
        }
    })
}

/// A page of a layer's conversations, one line each, then the repositories a
/// layer of knowledge-base records holds (`repos`, the top of its path view).
pub fn render(
    name: &str,
    layer: &LayerConversations,
    repos: &[PathNode],
    offset: usize,
    limit: usize,
) -> String {
    let all = &layer.conversations;
    let page: Vec<String> = all
        .iter()
        .skip(offset)
        .take(limit)
        .map(|c| {
            let place = if c.repo.is_empty() {
                c.label.clone()
            } else {
                format!("{}/{}", c.repo, c.label.trim_start_matches('/'))
            };
            format!(
                "- {} {}{}: {} turns, {} tokens{}",
                c.timeline,
                place,
                if c.branches.is_empty() {
                    String::new()
                } else {
                    format!(" [{}]", c.branches.join(", "))
                },
                c.turn_count,
                c.tokens,
                if c.archived { ", archived" } else { "" }
            )
        })
        .collect();
    let mut out = format!(
        "{name}: {} conversations; {}–{}",
        all.len(),
        offset.min(all.len()),
        (offset + page.len()).min(all.len())
    );
    for line in page {
        out.push('\n');
        out.push_str(&line);
    }
    // A layer of the knowledge base's records holds them by path, not as
    // conversations: each repository with what it holds.
    for repo in repos {
        out.push_str(&format!(
            "\n- {}: {} units, {} tokens",
            repo.path, repo.unit_count, repo.tokens
        ));
    }
    out
}

pub fn call(ctx: &Ctx, args: &Value) -> Answer {
    if let Err(e) = ops::admin_only(&ctx.caller, NAME) {
        return Answer::error(e);
    }
    read(&ctx.session, args)
}

/// The read for a conversation of Zen's, through the operations runner, which
/// has checked the caller is an Admin.
pub fn run(call: &OpsCall, args: &Value) -> Answer {
    read(&call.session, args)
}

fn read(session: &ZendSession, args: &Value) -> Answer {
    let name = match string(args, "name") {
        Ok(n) => n,
        Err(e) => return Answer::error(e),
    };
    let offset = args.get("offset").and_then(Value::as_u64).unwrap_or(0) as usize;
    let limit = (args
        .get("limit")
        .and_then(Value::as_u64)
        .unwrap_or(DEFAULT_LIMIT as u64) as usize)
        .clamp(1, MAX_LIMIT);
    if !session.is_loaded() {
        return Answer::error(LOADING);
    }
    match session.substrate_layer(name) {
        Some(layer) => {
            let repos = session
                .substrate_layer_path(name, "")
                .map(|top| top.children)
                .unwrap_or_default();
            Answer::ok(render(name, &layer, &repos, offset, limit))
        }
        None => Answer::error(format!("there is no layer `{name}`")),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use host::substrate_views::{path_level, ConvView, PathUnit, PathUnitView};

    fn conv(timeline: &str, repo: &str, label: &str) -> ConvView {
        ConvView {
            timeline: timeline.into(),
            conv_id: String::new(),
            label: label.into(),
            repo: repo.into(),
            archived: false,
            group: "files".into(),
            turn_count: 3,
            tokens: 900,
            summary_nodes: 0,
            branches: vec!["main".into()],
            commit: String::new(),
        }
    }

    #[test]
    fn a_layer_is_paged_one_conversation_a_line() {
        let layer = LayerConversations {
            conversations: vec![conv("11", "candle", "/src/lib.rs"), conv("12", "", "Root")],
        };
        assert_eq!(
            render("code_reading", &layer, &[], 0, 1),
            "code_reading: 2 conversations; 0–1\n- 11 candle/src/lib.rs [main]: 3 turns, 900 tokens"
        );
        assert_eq!(
            render("code_reading", &layer, &[], 1, 5),
            "code_reading: 2 conversations; 1–2\n- 12 Root [main]: 3 turns, 900 tokens"
        );
    }

    /// A layer of records lists each repository it holds, with its units.
    #[test]
    fn a_layer_of_records_lists_its_repositories() {
        let layer = LayerConversations {
            conversations: Vec::new(),
        };
        let top = path_level(
            "",
            vec![
                PathUnit {
                    path: "candle/a.rs".into(),
                    unit: PathUnitView {
                        blob: "b".into(),
                        branches: vec!["main".into()],
                        lines: 4,
                        sections: 1,
                        tokens: 40,
                        placed: true,
                    },
                },
                PathUnit {
                    path: "yenbot/b.rs".into(),
                    unit: PathUnitView {
                        blob: "c".into(),
                        branches: vec!["main".into()],
                        lines: 2,
                        sections: 1,
                        tokens: 9,
                        placed: false,
                    },
                },
            ],
        );
        assert_eq!(
            render("code_reading", &layer, &top.children, 0, 50),
            "code_reading: 0 conversations; 0–0\n- candle/: 1 units, 40 tokens\n- yenbot/: 1 units, 9 tokens"
        );
    }
}
