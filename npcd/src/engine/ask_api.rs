//! The operator's way to look inside a running character and to steer it:
//! ask it a question and force an answer, or put a thought in its head.
//!
//! `POST /v1/npc/:nid/ask` puts a question to the character between two of its
//! turns and returns what it answered, without the exchange entering its
//! memory — see [`crate::engine::mind::Minds::ask`]. `POST
//! /v1/npc/:nid/mind_control` delivers a thought that arrives as the character's
//! own reasoning. Both are debugging instruments first, and what a health
//! monitor is built from second.

use std::sync::Arc;

use axum::extract::{Path, State};
use axum::http::{HeaderMap, StatusCode};
use axum::response::{IntoResponse, Response};
use axum::Json;
use serde::Deserialize;
use serde_json::json;

use crate::api::{err, Authored};
use crate::engine::check;
use crate::engine::event::{Event, EventKind, Salience};
use crate::engine::{no_engine, owned};

/// The body of an `ask`: a standing question by name, or the caller's own.
#[derive(Debug, Deserialize)]
pub struct AskBody {
    /// A standing question — see [`check::ALL`]. Wins over `question`.
    #[serde(default)]
    check: Option<String>,
    /// The caller's own question.
    #[serde(default)]
    question: Option<String>,
    /// The answers it may give. Absent or empty means free text. Ignored when
    /// `check` names a standing question, which carries its own.
    #[serde(default)]
    choices: Vec<String>,
}

/// The question and the answers it admits, or why the body names neither.
fn resolve(body: AskBody) -> Result<(String, Vec<String>), String> {
    if let Some(name) = body.check.as_deref() {
        let check = check::named(name).ok_or_else(|| {
            let known: Vec<&str> = check::ALL.iter().map(|c| c.name).collect();
            format!(
                "no standing question `{name}`; they are {}",
                known.join(", ")
            )
        })?;
        let choices = check.choices.iter().map(|c| c.to_string()).collect();
        return Ok((check.question.to_string(), choices));
    }
    let question = body.question.as_deref().map(str::trim).unwrap_or_default();
    if question.is_empty() {
        return Err("give a `check` by name or a `question`".to_string());
    }
    let choices: Vec<String> = body
        .choices
        .into_iter()
        .map(|c| c.trim().to_string())
        .filter(|c| !c.is_empty())
        .collect();
    Ok((question.to_string(), choices))
}

/// `POST /v1/npc/:nid/ask` — put a question to a character and force an answer.
///
/// Ownership is checked: this reads a mind somebody owns. It waits for the
/// character's own turn to finish, so on a busy character it takes as long as
/// that turn does.
pub async fn ask(
    State(s): State<Arc<Authored>>,
    Path(nid): Path<String>,
    headers: HeaderMap,
    Json(body): Json<AskBody>,
) -> Response {
    let npc_id = match owned(&s, &headers, &nid).await {
        Ok(id) => id,
        Err(r) => return *r,
    };
    let Some(rt) = s.runtime.as_ref() else {
        return no_engine("asking a character a question");
    };
    let (question, choices) = match resolve(body) {
        Ok(v) => v,
        Err(why) => return err(StatusCode::BAD_REQUEST, "bad_question", &why),
    };
    match rt.ask(npc_id, &question, &choices).await {
        Ok(answer) => Json(json!({
            "question": question,
            "choices": choices,
            "answer": answer.answer,
            "reason": answer.reason,
            "raw": answer.raw,
            "ms": answer.ms,
        }))
        .into_response(),
        Err(e) => err(StatusCode::CONFLICT, "not_answered", &format!("{e:#}")),
    }
}

/// The body of a `mind_control`.
#[derive(Debug, Deserialize)]
pub struct MindControlBody {
    /// The thought, worded as the character's own — what it can't leave alone.
    text: String,
    /// How hard it lands. Absent is [`Salience::URGENT`]: a thought put there to
    /// steer a character that has gone wrong should be read now.
    #[serde(default)]
    salience: Option<f32>,
}

/// The event a `mind_control` becomes. `None` for a thought with nothing in it.
fn thought_event(body: MindControlBody) -> Option<(Salience, EventKind)> {
    let text = body.text.trim();
    if text.is_empty() {
        return None;
    }
    let salience = body.salience.map_or(Salience::URGENT, Salience::new);
    Some((
        salience,
        EventKind::MindControl {
            text: text.to_string(),
        },
    ))
}

/// `POST /v1/npc/:nid/mind_control` — put a thought in a character's head.
///
/// It reaches the character's main conversation as an event, so it is read at the
/// character's next turn and acted on as its own reasoning.
pub async fn mind_control(
    State(s): State<Arc<Authored>>,
    Path(nid): Path<String>,
    headers: HeaderMap,
    Json(body): Json<MindControlBody>,
) -> Response {
    let npc_id = match owned(&s, &headers, &nid).await {
        Ok(id) => id,
        Err(r) => return *r,
    };
    let Some(rt) = s.runtime.as_ref() else {
        return no_engine("putting a thought in a character's head");
    };
    let Some((salience, kind)) = thought_event(body) else {
        return err(
            StatusCode::BAD_REQUEST,
            "empty_thought",
            "a thought needs something in it",
        );
    };
    let world_ms = s.world_ms(npc_id).await;
    let prose = Event::new(0, world_ms, salience, kind.clone()).prose();
    if !rt.scheduler.deliver(npc_id, world_ms, salience, kind) {
        return err(
            StatusCode::SERVICE_UNAVAILABLE,
            "not_awake",
            "the character exists but is not in the scheduler — the engine is still loading",
        );
    }
    Json(json!({
        "delivered": true,
        "salience": salience.get(),
        "preempts": salience.preempts(),
        "prose": prose,
    }))
    .into_response()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ask_body(check: Option<&str>, question: Option<&str>, choices: &[&str]) -> AskBody {
        AskBody {
            check: check.map(str::to_string),
            question: question.map(str::to_string),
            choices: choices.iter().map(|c| c.to_string()).collect(),
        }
    }

    #[test]
    fn a_standing_question_brings_its_own_choices() {
        let (q, choices) = resolve(ask_body(Some("looping"), Some("ignored"), &["x"])).unwrap();
        assert_eq!(q, check::LOOPING.question);
        assert_eq!(
            choices,
            vec!["making progress", "repeating myself", "stuck"]
        );
    }

    #[test]
    fn the_mission_check_is_free_text() {
        let (q, choices) = resolve(ask_body(Some("mission"), None, &[])).unwrap();
        assert_eq!(q, check::MISSION.question);
        assert!(choices.is_empty());
    }

    #[test]
    fn an_unknown_standing_question_names_the_known_ones() {
        let why = resolve(ask_body(Some("mood"), None, &[])).unwrap_err();
        assert!(why.contains("looping") && why.contains("lost"), "{why}");
    }

    #[test]
    fn an_ad_hoc_question_is_trimmed_and_blank_choices_dropped() {
        let (q, choices) =
            resolve(ask_body(None, Some("  Where are you? "), &[" a ", "", "b"])).unwrap();
        assert_eq!(q, "Where are you?");
        assert_eq!(choices, vec!["a", "b"]);
    }

    #[test]
    fn a_body_with_no_question_is_refused() {
        assert!(resolve(ask_body(None, None, &[])).is_err());
        assert!(resolve(ask_body(None, Some("   "), &[])).is_err());
    }

    #[test]
    fn a_thought_is_urgent_by_default_and_reads_as_the_characters_own() {
        let (salience, kind) = thought_event(MindControlBody {
            text: " go and find Wren ".into(),
            salience: None,
        })
        .unwrap();
        assert_eq!(salience, Salience::URGENT);
        assert_eq!(
            Event::new(0, 0, salience, kind).prose(),
            "A thought you can't shake: go and find Wren"
        );
    }

    #[test]
    fn an_empty_thought_is_refused() {
        assert!(thought_event(MindControlBody {
            text: "  ".into(),
            salience: None
        })
        .is_none());
    }
}
