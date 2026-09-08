//! A scenario put to a character that does not exist, so the prompt can be
//! tested without waiting for one that does.
//!
//! # Why this is a route and not a test
//!
//! Everything this answers is a property of a live engine holding a real
//! checkpoint: whether the frame produces a call at all, whether provenance
//! brings the *right* acts into focus, whether a reasoning block opens when the
//! work needs one and stays shut when it does not, whether the call comes out
//! clean. None of that can be asserted without the model, and the model is the
//! thing the daemon is holding.
//!
//! So the harness talks to the running daemon instead of standing up its own.
//! One card, one engine, one substrate — and the cast goes on thinking while the
//! scenarios run beside them.
//!
//! # Why it is worth having at all
//!
//! Because the failure this exists to catch is **silent**. A character that
//! emitted no act and a character that chose to do nothing are identical in
//! every view the daemon has: same empty act list, same successful tick, no
//! error anywhere. The one time it mattered, the decode had opened a `<think>`
//! it never closed and run to the token ceiling, and the entire turn was
//! discarded as reasoning — for hours, across every character, with the log
//! reporting healthy ticks throughout.
//!
//! [`Probe`](crate::engine::mind::Probe) reports the raw decode and both halves
//! of the reasoning block precisely so that failure has somewhere to show.
//!
//! # Deterministic provenance
//!
//! A scenario names what is true and nothing scores it: which personality, which
//! world, what has been asked, and what the character can perceive this moment.
//! That is the same discipline the environment follows for a real character —
//! the situation is *computed from the world* rather than selected for
//! relevance — and it is what makes two runs of one scenario comparable.

use std::sync::Arc;

use axum::extract::State;
use axum::http::{HeaderMap, StatusCode};
use axum::response::{IntoResponse, Response};
use axum::Json;
use serde::{Deserialize, Serialize};

use crate::api::{err, Authored};
use crate::engine::identity::Deliberation;
use crate::engine::prompt::Persona;
use crate::engine::tools::Mode;

/// One scenario.
#[derive(Debug, Deserialize)]
pub struct Scenario {
    /// The personality whose floor the character reads. Unknown ones fall
    /// through to the generic member, which is itself worth testing.
    #[serde(default)]
    pub personality: String,
    #[serde(default)]
    pub world_id: String,
    /// What the character is called. Only reaches the rendered-prompt path — a
    /// probe under the projection is nobody, deliberately, so what is under test
    /// is the frame rather than whose name is on it.
    #[serde(default)]
    pub name: String,
    /// What has been asked of it. Absent is the ordinary case and means the
    /// standing instruction applies.
    #[serde(default)]
    pub mission: Option<String>,
    /// How hard to think — a property of the work, not of the character.
    /// `off` | `quick` | `balanced` | `deep` | `exhaustive`.
    #[serde(default)]
    pub thinking: Option<String>,
    /// What the character perceives this moment, already in the narrator's
    /// voice — exactly what the environment would have handed it.
    pub perceive: String,
    /// Who is standing here, by name.
    ///
    /// **Part of the provenance, not decoration.** The grammar is built from
    /// this: alone, the acts that need company are not in the mask at all, and
    /// in company an addressee may only be one of these names. A scenario that
    /// says somebody is here in its `perceive` prose and leaves this empty is
    /// describing a room the character cannot act in.
    #[serde(default)]
    pub company: Vec<String>,
    /// Which of [`Self::company`] are already waiting on this character.
    ///
    /// They may still be spoken to — being waited on is the best reason to
    /// speak to somebody — but they may not be waited on back, so a scenario
    /// that sets this is testing that the character does something rather than
    /// returning the stare.
    #[serde(default)]
    pub waited_on_by: Vec<String>,
    /// Where the character may walk, never including where it stands.
    ///
    /// Empty takes `move_to` out of the grammar entirely, which is the honest
    /// reading of a scenario that names nowhere to go — and the only way to
    /// test a character that has to deal with the room it is in.
    #[serde(default)]
    pub places: Vec<String>,
    /// Which assembly to put under test. Both are real paths a character can run
    /// under, and comparing them is the point.
    #[serde(default = "yes")]
    pub projected: bool,
}

fn yes() -> bool {
    true
}

/// What the scenario produced.
#[derive(Debug, Serialize)]
pub struct Outcome {
    /// The acts, in order, as the character would read them back.
    pub acts: Vec<String>,
    /// Calls that tried and failed, in the character's own second person.
    pub rejected: Vec<String>,
    /// Anything written that was not a call.
    pub narration: String,
    pub raw: String,
    pub opened_think: bool,
    pub closed_think: bool,
    /// **The failure worth naming.** A block that opened and never closed took
    /// the whole decode with it.
    pub runaway_think: bool,
    pub prompt_bytes: usize,
    pub projected: bool,
    pub ms: u64,
}

pub async fn run(
    State(s): State<Arc<Authored>>,
    headers: HeaderMap,
    Json(scenario): Json<Scenario>,
) -> Response {
    // Admin-only. A scenario runs a real decode on the one card the whole cast
    // shares, so it is a way to take the engine's time — and it reports the
    // system prompt's size and content, which is the daemon's own configuration
    // rather than any caller's data.
    let _ = &headers;
    let Some(rt) = s.runtime.as_ref() else {
        return crate::engine::no_engine("simulating a conversation");
    };
    let Some(minds) = rt.minds.read().unwrap().clone() else {
        return crate::engine::no_engine("simulating a conversation");
    };

    let thinking = match scenario.thinking.as_deref() {
        None | Some("off") => Deliberation::None,
        Some("quick") => Deliberation::Quick,
        Some("balanced") => Deliberation::Balanced,
        Some("deep") => Deliberation::Deep,
        Some("exhaustive") => Deliberation::Exhaustive,
        Some(other) => {
            return err(
                StatusCode::BAD_REQUEST,
                "unknown_thinking",
                &format!(
                    "`{other}` is not a deliberation level — use off, quick, balanced, deep or \
                     exhaustive"
                ),
            )
        }
    };

    // The world's setting, so the rendered-prompt path reads what the projected
    // one selects. A probe that compared two assemblies given different content
    // would be measuring the content.
    let world = s
        .worlds
        .read()
        .await
        .get(&scenario.world_id)
        .and_then(|r| r.body.get("setting").and_then(|d| d.as_str()))
        .unwrap_or_default()
        .to_string();
    let mission = scenario.mission.clone().unwrap_or_default();

    // Blocking: a decode is seconds and this is an axum worker. Moved off it so
    // a scenario does not stall the console's polling for the length of a
    // generation.
    let probe = tokio::task::spawn_blocking(move || {
        let persona = persona_of(&scenario, &world, &mission);
        let within = crate::engine::tools::Within {
            company: scenario.company.clone(),
            places: scenario.places.clone(),
            waited_on_by: scenario.waited_on_by.clone(),
        };
        minds.probe(
            &persona,
            Mode::Physical,
            thinking,
            &scenario.perceive,
            scenario.projected,
            &within,
        )
    })
    .await;

    match probe {
        Ok(Ok(p)) => Json(Outcome {
            acts: p.parsed.acts.iter().map(|a| a.summary()).collect(),
            rejected: p.parsed.rejected.iter().map(|r| r.line()).collect(),
            narration: p.parsed.narration.clone(),
            runaway_think: p.opened_think && !p.closed_think,
            opened_think: p.opened_think,
            closed_think: p.closed_think,
            prompt_bytes: p.prompt_bytes,
            projected: p.projected,
            ms: p.ms,
            raw: p.raw,
        })
        .into_response(),
        Ok(Err(e)) => err(
            StatusCode::INTERNAL_SERVER_ERROR,
            "probe_failed",
            &format!("{e:#}"),
        ),
        Err(e) => err(
            StatusCode::INTERNAL_SERVER_ERROR,
            "probe_panicked",
            &format!("{e}"),
        ),
    }
}

/// The persona a scenario describes, rebuilt inside the blocking task.
///
/// Rebuilt rather than moved because [`Persona`] borrows every field, and the
/// scenario that owns them has to cross a thread boundary to reach the decode.
fn persona_of<'a>(s: &'a Scenario, world: &'a str, mission: &'a str) -> Persona<'a> {
    Persona {
        name: &s.name,
        personality: &s.personality,
        world_id: &s.world_id,
        identity: "",
        manner: "",
        beliefs: &[],
        relationships: &[],
        intent: (!mission.is_empty()).then_some(mission),
        situation: "",
        world,
        place: "",
    }
}
