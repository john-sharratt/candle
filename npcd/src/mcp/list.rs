//! `bot_list {state?, world?, q?}`: the caller's bots — `GET /v1/npc`.

use std::collections::HashMap;
use std::sync::Arc;

use axum::extract::{Query, State};
use axum::http::HeaderMap;
use serde_json::{json, Value};

use super::args::optional;
use super::route::read;
use super::{Answer, Tool};
use crate::api::{list_npcs, Authored};
use crate::engine::bodies::NpcList;

pub struct List;

impl Tool for List {
    const NAME: &'static str = "bot_list";

    fn definition() -> Value {
        json!({
            "name": Self::NAME,
            "description": "List your bots — autonomous agents, each living in a world and acting on \
                its own: each one's id (what the other bot_* tools name it by), name, world, state and \
                mood. Filter by `state` (active, idle, asleep, \
                suspended), by `world`, or by text in its name with `q`.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "state": {"type": "string", "enum": ["active", "idle", "asleep", "suspended", "any"], "description": "Only bots in this state."},
                    "world": {"type": "string", "description": "Only bots in this world."},
                    "q": {"type": "string", "description": "Only bots whose name holds this text."}
                },
                "additionalProperties": false
            }
        })
    }

    async fn call(s: &Arc<Authored>, headers: &HeaderMap, args: &Value) -> Answer {
        let mut query = HashMap::new();
        for (arg, param) in [("state", "state"), ("world", "world_id"), ("q", "q")] {
            if let Some(v) = optional(args, arg) {
                query.insert(param.to_string(), v.to_string());
            }
        }
        let res = list_npcs(State(Arc::clone(s)), headers.clone(), Query(query)).await;
        Answer::of(read::<NpcList>(res).await.map(|l| render(&l)))
    }
}

/// A line per bot.
pub fn render(list: &NpcList) -> String {
    if list.items.is_empty() {
        return "You have no bots that match.".into();
    }
    list.items
        .iter()
        .map(|n| {
            let mut line = format!("{} — bot {}, {}", n.name, n.npc_id, n.state);
            if let Some(w) = n.world_id.as_deref().filter(|w| !w.is_empty()) {
                line.push_str(&format!(", in {w}"));
            }
            if let Some(m) = &n.mood {
                line.push_str(&format!(", mood {m}"));
            }
            line
        })
        .collect::<Vec<_>>()
        .join("\n")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::bodies::Npc;

    #[test]
    fn a_bot_is_a_line_naming_its_id() {
        let list = NpcList {
            items: vec![
                Npc {
                    npc_id: "1z".into(),
                    name: "Paxon Vael".into(),
                    world_id: Some("battle-cities".into()),
                    state: "active".into(),
                    mood: Some("wary".into()),
                    personality_name: None,
                },
                Npc {
                    npc_id: "2a".into(),
                    name: "Ione".into(),
                    world_id: None,
                    state: "asleep".into(),
                    mood: None,
                    personality_name: None,
                },
            ],
        };
        assert_eq!(
            render(&list),
            "Paxon Vael — bot 1z, active, in battle-cities, mood wary\nIone — bot 2a, asleep"
        );
        assert_eq!(
            render(&NpcList { items: vec![] }),
            "You have no bots that match."
        );
    }
}
