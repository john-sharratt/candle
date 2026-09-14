//! Prose from the resident model — names, descriptions, narration, and the judge
//! that reads an image prompt before it is drawn.
//!
//! # Why the resident model, and not a guest
//!
//! These were served by a co-resident prose model loaded between the engine's
//! waves. Every job stopped the world: the scheduler evicted the engine's working
//! set, a second checkpoint crossed the PCIe link, the backlog ran, and the ground
//! went back — so a single name paid for a model load, and every character in
//! every world stopped thinking while it did.
//!
//! The resident model writes this prose well on its own, so a job is a turn in a
//! throwaway conversation instead ([`crate::engine::prose`]). It rides the
//! engine's waves beside the cast: a description is one more sequence in a wave
//! that was running anyway, and no character waits for it. A deployment needs no
//! second checkpoint and no `guests.yaml` for any of it.
//!
//! # What a caller gets back
//!
//! An [`Answer`] — the text, the tokens it cost and the seed that drew it, so a
//! draft an author liked can be asked for again. A failure is a [`ProseError`],
//! and [`refusal`] maps it to a status in one place, so every route agrees about
//! what is the caller's to fix and what is worth retrying.

use std::sync::Arc;

use axum::extract::State;
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use axum::Json;
use candle_conversation::guest::resolve_seed;
use serde::{Deserialize, Serialize};
use serde_json::json;

use crate::api::Authored;

/// The most tokens one prose job may decode.
///
/// A job is a sequence in the engine's waves like any character's turn, so this
/// bounds how long one ask can hold a slot in them.
pub const MAX_TOKENS: u32 = 4096;

/// The voice a job gets when the caller names none.
pub const DEFAULT_SYSTEM: &str =
    "You are a careful writer. Answer exactly what is asked, in plain prose, and nothing else.";

/// Prose to generate.
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
pub struct Request {
    /// The voice. Empty takes [`DEFAULT_SYSTEM`].
    #[serde(default)]
    pub system: String,
    pub prompt: String,
    pub max_tokens: u32,
    #[serde(default)]
    pub temperature: Option<f32>,
    #[serde(default)]
    pub seed: Option<u64>,
    /// Constrain the answer to exactly one of these strings.
    ///
    /// A **stencil**, in the sense of [`candle_conversation::stencil`]: the sampler
    /// is masked to the tokens the grammar permits, so an answer outside the set is
    /// not improbable, it is unreachable. `None` decodes freely.
    ///
    /// **Prefer single-token choices.** A multi-token arm commits the walk on its
    /// first token and then forces the remainder, so a model that would have
    /// changed its mind after one token cannot — it is stuck down a path it did
    /// not want. One token per arm makes the whole decision a single masked decode
    /// with nothing to commit to.
    #[serde(default)]
    pub choices: Option<Vec<String>>,
}

impl Request {
    /// Refuse a request that cannot be served, before any engine time is spent on
    /// it. The caller gets a synchronous answer instead of a failed decode.
    pub fn check(&self) -> Result<(), String> {
        if self.prompt.trim().is_empty() {
            return Err("a prose request needs a prompt".into());
        }
        if self.max_tokens == 0 {
            return Err("a prose request needs a non-zero token budget".into());
        }
        if self.max_tokens > MAX_TOKENS {
            return Err(format!(
                "{} tokens is past the {MAX_TOKENS} limit",
                self.max_tokens
            ));
        }
        if let Some(t) = self.temperature {
            if !t.is_finite() || t < 0.0 {
                return Err(format!("temperature {t} is not a usable number"));
            }
        }
        if self.choices.as_ref().is_some_and(|c| c.is_empty()) {
            return Err("an empty set of choices is a grammar with nothing to walk".into());
        }
        Ok(())
    }
}

/// What a finished job produced.
#[derive(Clone, Debug, PartialEq)]
pub struct Answer {
    pub text: String,
    /// Tokens the model decoded for it.
    pub tokens: u32,
    /// The seed actually used, whether the caller pinned one or not — a draft an
    /// author liked is a one-off unless the seed that drew it comes back with it.
    pub seed: u64,
}

/// Why a job produced nothing.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ProseError {
    /// The ask itself is not servable. Carries what to change.
    Refused(String),
    /// There is no engine to decode on — it is still loading, or shutting down.
    /// Retryable: the first case resolves on its own.
    Unavailable,
    /// The decode ran and failed.
    Failed(String),
}

impl ProseError {
    /// The stable code a console branches on.
    pub fn code(&self) -> &'static str {
        match self {
            Self::Refused(_) => "bad_request",
            Self::Unavailable => "engine_unavailable",
            Self::Failed(_) => "prose_failed",
        }
    }

    /// Whether asking again later can succeed.
    pub fn retry(&self) -> bool {
        matches!(self, Self::Unavailable)
    }

    pub fn status(&self) -> StatusCode {
        match self {
            Self::Refused(_) => StatusCode::BAD_REQUEST,
            Self::Unavailable => StatusCode::SERVICE_UNAVAILABLE,
            Self::Failed(_) => StatusCode::INTERNAL_SERVER_ERROR,
        }
    }
}

impl std::fmt::Display for ProseError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Refused(why) => f.write_str(why),
            Self::Unavailable => {
                f.write_str("the engine is not running yet — it is still loading, or shutting down")
            }
            Self::Failed(e) => write!(f, "the prose decode failed: {e}"),
        }
    }
}

impl std::error::Error for ProseError {}

/// The response every prose route answers a failed job with.
///
/// One mapping for all of them: a caller that learns "retry" from one route and
/// "give up" from another for the same condition cannot act on either.
pub fn refusal(e: &ProseError) -> Response {
    (
        e.status(),
        Json(json!({ "error": e.code(), "detail": e.to_string(), "retry": e.retry() })),
    )
        .into_response()
}

/// Decode `request` on the resident model and wait for the answer.
pub async fn run(s: &Arc<Authored>, request: Request) -> Result<Answer, ProseError> {
    run_streamed(s, request, |_: &str| {}).await
}

/// The same, with each new fragment of text handed to `on_fragment` as it lands.
///
/// The fragments are a **preview**: tokenizer cleanup can revise a character
/// already shown, so they concatenate to within a character or two of the final
/// text, and the [`Answer`] is authoritative. `on_fragment` runs between decode
/// awaits, so it must push and return.
pub async fn run_streamed(
    s: &Arc<Authored>,
    request: Request,
    mut on_fragment: impl FnMut(&str) + Send + 'static,
) -> Result<Answer, ProseError> {
    request.check().map_err(ProseError::Refused)?;
    let Some(rt) = s.runtime.as_ref() else {
        return Err(ProseError::Unavailable);
    };
    let minds = rt.minds.read().unwrap().clone();
    let Some(minds) = minds else {
        return Err(ProseError::Unavailable);
    };
    let seed = resolve_seed(request.seed);
    let (engine, cfg) = (minds.engine(), minds.base_config());
    // Awaited directly: the decode yields on the turn's event channel, so a
    // long generation parks a future rather than a thread — and a caller that
    // gives up drops the future, which stops the decode.
    crate::engine::prose::decode(&engine, &cfg, &request, seed, &mut on_fragment)
        .await
        .map_err(|e| ProseError::Failed(format!("{e:#}")))
}

/// What a caller posts to `/v1/generate/prose`.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ProseBody {
    pub prompt: String,
    #[serde(default)]
    pub system: String,
    #[serde(default = "default_tokens")]
    pub max_tokens: u32,
    #[serde(default)]
    pub temperature: Option<f32>,
    #[serde(default)]
    pub seed: Option<u64>,
}

fn default_tokens() -> u32 {
    512
}

impl ProseBody {
    /// The request this body asks for — one function, so the route and its test
    /// build the same thing.
    fn into_request(self) -> Request {
        Request {
            system: self.system,
            prompt: self.prompt,
            max_tokens: self.max_tokens,
            temperature: self.temperature,
            seed: self.seed,
            // The public route decodes freely. A stencil is a grammar the caller
            // would have to supply, and `ProseBody` deliberately has no field for
            // one — an arbitrary grammar from an untrusted caller is a way to make
            // the model decode something nobody reviewed.
            choices: None,
        }
    }
}

/// `POST /v1/generate/prose` — free prose from the resident model.
pub async fn post_prose(State(s): State<Arc<Authored>>, Json(body): Json<ProseBody>) -> Response {
    match run(&s, body.into_request()).await {
        // The seed rides back with the text because a draft an author liked is a
        // one-off unless the draw can be repeated.
        Ok(a) => {
            Json(json!({ "text": a.text, "tokens": a.tokens, "seed": a.seed })).into_response()
        }
        Err(e) => refusal(&e),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ask(prompt: &str, max_tokens: u32) -> Request {
        Request {
            system: String::new(),
            prompt: prompt.into(),
            max_tokens,
            temperature: None,
            seed: None,
            choices: None,
        }
    }

    #[test]
    fn an_ordinary_ask_is_accepted() {
        assert!(ask("Describe the yard.", 256).check().is_ok());
    }

    /// **A refusal costs nothing.** An ask that was never servable is answered at
    /// submission rather than after a decode has been started for it.
    #[test]
    fn an_unservable_ask_is_refused_before_it_runs() {
        assert!(ask("   ", 64).check().is_err(), "an empty prompt");
        assert!(ask("x", 0).check().is_err(), "a zero budget");
        assert!(ask("x", MAX_TOKENS + 1).check().is_err(), "past the limit");
        assert!(ask("x", MAX_TOKENS).check().is_ok(), "at the limit");
    }

    /// A temperature that is not a number would reach the sampler and draw
    /// uniformly over the vocabulary — prose unrelated to the prompt, with nothing
    /// reporting why.
    #[test]
    fn a_nonsense_temperature_is_refused() {
        let with = |t: f32| {
            Request {
                temperature: Some(t),
                ..ask("x", 8)
            }
            .check()
        };
        assert!(with(f32::NAN).is_err());
        assert!(with(-1.0).is_err());
        assert!(with(0.0).is_ok(), "greedy is a legitimate ask");
    }

    /// An empty choice set would compile to a grammar with no arm to take.
    #[test]
    fn an_empty_choice_set_is_refused() {
        let with = |c: Vec<String>| {
            Request {
                choices: Some(c),
                ..ask("x", 1)
            }
            .check()
        };
        assert!(with(Vec::new()).is_err());
        assert!(with(vec!["Y".into(), "N".into()]).is_ok());
    }

    /// **Three failures, three statuses.** A single 500 would tell a caller
    /// nothing about whether to fix the ask, retry, or stop asking — and the
    /// retryable one is the only one that resolves on its own.
    #[test]
    fn each_failure_carries_a_status_a_caller_can_act_on() {
        let refused = ProseError::Refused("too long".into());
        let failed = ProseError::Failed("boom".into());
        assert_eq!(refusal(&refused).status(), StatusCode::BAD_REQUEST);
        assert_eq!(
            refusal(&ProseError::Unavailable).status(),
            StatusCode::SERVICE_UNAVAILABLE
        );
        assert_eq!(refusal(&failed).status(), StatusCode::INTERNAL_SERVER_ERROR);
        assert!(ProseError::Unavailable.retry());
        assert!(!refused.retry());
        assert!(!failed.retry());
    }

    /// The route's defaults are what a caller who sent only a prompt gets, so
    /// they have to be a request the engine will actually accept.
    #[test]
    fn the_body_defaults_are_a_servable_request() {
        let body: ProseBody = serde_json::from_str(r#"{"prompt":"the yard"}"#).unwrap();
        let r = body.into_request();
        assert!(r.check().is_ok(), "{:?}", r.check());
        assert!(r.choices.is_none(), "the public route must decode freely");
    }

    /// A prompt is the one thing with no default — a request without one is not
    /// a request, and it is refused before it reaches the engine.
    #[test]
    fn a_body_with_no_prompt_is_refused_at_the_boundary() {
        assert!(serde_json::from_str::<ProseBody>(r#"{"max_tokens":16}"#).is_err());
    }
}
