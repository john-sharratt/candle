//! Starting characters over (`POST /v1/pulse/start-over`): the missions each
//! carries and has waiting are called off, and its conversations — the one it
//! thinks in, its narrator's, and every one it holds with a person or another
//! character — are retired, so its next turn opens a fresh one. Asked for by
//! name, its journal is emptied and its dreams retired as well.
//!
//! **By character, never by cast.** `--forget-conversations` and its kin start
//! the whole cast over at a start; this names who, while the daemon runs, so
//! one world's characters begin again and another's work goes on untouched.
//!
//! Its beliefs, relationships, portrait and place in the world are kept: the
//! character is the same one, with nothing behind it.
//!
//! The route answers `202` with who is being started over, and the work runs
//! on a task of its own; what each character took is logged as it finishes.

use std::sync::Arc;

use axum::extract::State;
use axum::http::{HeaderMap, StatusCode};
use axum::response::{IntoResponse, Response};
use axum::Json;
use serde::{Deserialize, Serialize};
use serde_json::json;

use crate::api::{err, owner_of, Authored};
use crate::engine::no_engine;
use crate::engine::runtime::Runtime;
use crate::npcs::{npc_id_of_wire, npc_id_wire};

/// Who starts over, and how far back.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct StartOver {
    /// The characters, by the ids the API names them with.
    pub npcs: Vec<String>,
    /// Empty each one's journal too.
    #[serde(default)]
    pub journal: bool,
    /// Retire each one's dreams too.
    #[serde(default)]
    pub dreams: bool,
}

/// What starting one character over took.
#[derive(Debug, Default, PartialEq, Eq, Serialize)]
pub struct Started {
    pub npc: String,
    /// Missions called off or dropped.
    pub missions: usize,
    /// Conversations retired, of every kind.
    pub conversations: usize,
    /// Conversation records whose journal was emptied.
    pub journal: usize,
    /// Dreams retired.
    pub dreams: usize,
}

impl Runtime {
    /// Start `npc_id` over — see the module.
    ///
    /// The missions first, so nothing is carried into the fresh conversation;
    /// then the journal, emptied on the records the conversations still hold;
    /// then the conversations themselves. A turn in flight finishes on the
    /// conversation it began on, which is retired once it lets go.
    pub async fn start_over(&self, npc_id: u64, journal: bool, dreams: bool) -> Started {
        let mut took = Started {
            npc: npc_id_wire(npc_id),
            ..Started::default()
        };
        if let Some((hosted, body)) = self.body_of(npc_id) {
            if self.standing.writes(hosted.id()) {
                took.missions = hosted.with_sim(|s| {
                    // Nothing said before now is its to read back: a thread's
                    // history would hand the fresh conversation the old one.
                    s.threads.start_over(&body);
                    s.terminals.disconnect(&body);
                    s.missions.start_over(&body)
                });
            }
        }
        let minds = self.minds.read().unwrap().clone();
        let Some(minds) = minds else {
            return took;
        };
        if journal {
            took.journal = minds.forget_journal(npc_id);
        }
        let talks = minds.retire_npc(npc_id).await;
        took.conversations = talks + minds.forget_conversations(npc_id);
        if dreams {
            took.dreams = minds.forget_dreams(npc_id);
        }
        tracing::info!(
            npc = npc_id,
            missions = took.missions,
            conversations = took.conversations,
            journal = took.journal,
            dreams = took.dreams,
            "a character was started over"
        );
        took
    }
}

/// `POST /v1/pulse/start-over` — start the named characters over. Admin: it
/// reaches characters the caller does not own, and none of it can be undone.
pub async fn start_over(
    State(s): State<Arc<Authored>>,
    headers: HeaderMap,
    Json(body): Json<StartOver>,
) -> Response {
    if let Err(r) = owner_of(&s, &headers).await {
        return *r;
    }
    let Some(rt) = s.runtime.as_ref() else {
        return no_engine("starting a character over");
    };
    let mut ids = Vec::with_capacity(body.npcs.len());
    for wire in &body.npcs {
        match npc_id_of_wire(wire).filter(|id| rt.body_of(*id).is_some()) {
            Some(id) => ids.push(id),
            None => {
                return err(
                    StatusCode::NOT_FOUND,
                    "npc_not_found",
                    &format!(
                        "no character `{wire}` stands in a world here; nobody was started over"
                    ),
                )
            }
        }
    }
    // On a task of its own, and answered at once: a character mid-turn is
    // waited for, which can outlast any request, and a request dropped part
    // way would leave a character with its missions gone and its
    // conversation still standing. What each one took is in the log.
    let (rt, journal, dreams) = (Arc::clone(rt), body.journal, body.dreams);
    let starting: Vec<String> = ids.iter().map(|id| npc_id_wire(*id)).collect();
    tokio::spawn(async move {
        for id in ids {
            rt.start_over(id, journal, dreams).await;
        }
    });
    (StatusCode::ACCEPTED, Json(json!({ "starting": starting }))).into_response()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The request names who, and asks for the journal and the dreams only by
    /// name: left out, they are kept.
    #[test]
    fn the_request_keeps_what_it_does_not_name() {
        let plain: StartOver = serde_json::from_str(r#"{"npcs":["3fesnh4htpl0q"]}"#).unwrap();
        assert_eq!(plain.npcs, ["3fesnh4htpl0q"]);
        assert!(!plain.journal && !plain.dreams);
        let whole: StartOver =
            serde_json::from_str(r#"{"npcs":["128j9dsgheutk"],"journal":true,"dreams":true}"#)
                .unwrap();
        assert!(whole.journal && whole.dreams);
        assert!(serde_json::from_str::<StartOver>(r#"{"npcs":[],"world":"moon"}"#).is_err());
    }

    /// What it took is reported under the character's wire id, as raw JSON.
    #[test]
    fn what_it_took_is_reported_by_wire_id() {
        let took = Started {
            npc: npc_id_wire(16243565697050073002),
            missions: 2,
            conversations: 5,
            journal: 0,
            dreams: 0,
        };
        assert_eq!(
            serde_json::to_string(&took).unwrap(),
            r#"{"npc":"3fesnh4htpl0q","missions":2,"conversations":5,"journal":0,"dreams":0}"#
        );
    }
}
