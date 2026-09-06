//! Generating a character's portrait.
//!
//! # Where the prompt comes from
//!
//! Three sources, in this order, and the order is the whole design:
//!
//! 1. **The request**, if it carries one. This is the console's prompt box —
//!    somebody looking at a portrait and adjusting the words that drew it.
//! 2. **The personality's authored prompt**, if it has one. See
//!    [`crate::personality_portrait`]: a personality is art-directed once, in
//!    the mind, and every character struck from it inherits that direction.
//! 3. **The description**, turned into a portrait prompt by [`prompt_for`].
//!
//! There used to be only the third, and a header here explaining that a prompt
//! field would let a character's picture drift from its description. That risk
//! is real and it has not gone away — but it was being paid for by a worse one.
//! A description is written to be *read*: it is the character's identity in the
//! system prompt, prose about who somebody is. A prompt is written to be
//! *drawn*: framing, lens, light, what fills the frame. Forcing one sentence to
//! do both jobs got a worse version of each, and left no way to keep a portrait
//! somebody had spent an afternoon getting right.
//!
//! What keeps the two from drifting now is that the prompt is **authored beside
//! the character**, in the same file, under the same review — not typed into a
//! box that vanishes when the page closes.
//!
//! Whichever source wins, the prompt reaches the image guest the same way and
//! is checked the same way — see [`crate::compliance`].
//!
//! # Why an upload outranks this permanently
//!
//! [`crate::npcs::Npcs::set_portrait`] records an origin. A generated portrait
//! is written with `"generated"`; an uploaded one with `"uploaded"`. This route
//! refuses to overwrite an upload unless the caller says `force`, because
//! "regenerate everything missing" is a button an operator presses without
//! reading, and the picture they chose is the one thing here they cannot get
//! back.

use std::sync::Arc;

use axum::extract::{Path, State};
use axum::http::{HeaderMap, StatusCode};
use axum::response::{IntoResponse, Response};
use axum::Json;
use candle_conversation::guest::{GuestError, GuestOutcome, GuestRequest, ImageLora, ImageRequest};
use serde::Deserialize;
use serde_json::json;

use crate::api::Authored;

/// The framing a portrait needs and a character description does not carry.
///
/// Prepended rather than mixed in, so the description's own words stay the
/// subject of the sentence — a prompt that opens with style directives gets a
/// picture of the style with somebody in it.
const FRAMING: &str =
    "character portrait, head and shoulders, centred, neutral background, painterly, \
     detailed face, natural lighting";

/// Portraits are square, because every place the console shows one is square.
///
/// **512 on a model whose native size is 1,024**, which is a budget rather than
/// a quality ceiling: the console shows a portrait small, and the sequence the
/// transformer attends over is the image's area over 256 — so 1,024 is not twice
/// the drain, it is five times it. The face holds up at 512; a landscape would
/// not, and is not what this route draws.
const SIDE: u32 = 512;

/// Held at compile time: the latent is the image over 8 and the transformer
/// patches that by 2, and a size that does not divide is rounded away rather
/// than refused — so the caller would get an image of a different size than the
/// record says.
const _: () = assert!(SIDE.is_multiple_of(16));

/// Denoising steps.
///
/// Twelve, against the eight Z-Image-Turbo's schedule is trained at — for the
/// detail, not for variety. See [`crate::guest_routes::default_steps`] for the
/// measurement.
///
/// **Deliberately below that route's twenty-four.** A portrait is 512 px, drawn
/// for a record rather than for a look, and paid once per character across a
/// whole cast — where an image on the Images page is one picture somebody chose
/// to make and will keep. Tripling the drain of every portrait in a roster to
/// buy finish nobody is looking closely at is the wrong side of the same trade.
///
/// **What that measurement means for this route.** A portrait's prompt is built
/// from the character's own description, and on this model the description is
/// what decides the face — the seed barely touches it. Two characters whose
/// descriptions are both "a guard at the north gate" will come back as the same
/// man however their seeds differ, and no step count or sampler setting changes
/// that. A cast that looks like one person is a cast whose descriptions do not
/// distinguish them; the fix is in the writing, which is where this route gets
/// its prompt from.
///
/// Not a dial worth exposing per request — a portrait is drawn *for a record*,
/// and a per-call step count would make two portraits of the same character
/// differ for a reason the record does not carry.
const STEPS: u32 = 12;

#[derive(Debug, Default, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GenerateBody {
    /// Replace a portrait the owner uploaded. Off by default — see the module
    /// header.
    #[serde(default)]
    pub force: bool,
    /// Pin the draw, so a portrait an operator liked can be reproduced.
    #[serde(default)]
    pub seed: Option<u64>,
    /// Draw from these words instead of the character's own.
    ///
    /// The console's prompt box, which opens holding whatever this route would
    /// have used anyway — so sending it back unchanged is the same request as
    /// omitting it. Blank or whitespace is treated as absent rather than as an
    /// instruction to draw nothing.
    ///
    /// It is **not** stored on the character. A prompt worth keeping belongs in
    /// the personality document, where it is authored and reviewed; this is the
    /// one-off, and the record keeps the picture rather than the words.
    #[serde(default)]
    pub prompt: Option<String>,
}

/// Which words this draw uses, of the three that could supply them.
///
/// The order is the module header's: the request, then the personality's
/// authored prompt, then the description. Separate from the route and tested
/// directly — the route can only report a refusal, so a test driving it end to
/// end cannot see *which* prompt was chosen, and a test that re-implemented
/// this decision would be free to disagree with the one that ships.
///
/// **Blank is absent.** A cleared prompt box falls through to the character's
/// own words rather than sending the model an empty string, which draws
/// something — just nothing anybody asked for.
pub fn choose_prompt(
    requested: Option<&str>,
    authored: Option<&str>,
    description: &str,
    name: &str,
) -> String {
    for candidate in [requested, authored] {
        if let Some(words) = candidate.map(str::trim) {
            if !words.is_empty() {
                return words.to_string();
            }
        }
    }
    prompt_for(description, name)
}

/// The prompt a description becomes.
///
/// Separate from the route and public to the crate so a test can read it
/// without a card: this is the one piece of the portrait path with a decision
/// in it, and it is decided here rather than by the model.
pub fn prompt_for(description: &str, name: &str) -> String {
    let d = description.trim();
    if d.is_empty() {
        // A character with no description still has a name, and a portrait of
        // "a person" is more useful than a refusal for a record that is allowed
        // to have an empty description.
        return format!("{FRAMING}, a person named {name}");
    }
    format!("{FRAMING}, {d}")
}

/// `POST /v1/npc/:nid/portrait/generate`
///
/// Blocks for the length of the drain — see [`crate::guest_routes`] for why the
/// wait is on a blocking thread — and answers with the character's record,
/// exactly as the upload route does, so the console has one shape to render.
/// **The body is required, and `{}` is the ordinary one.**
///
/// It was `Option<Json<GenerateBody>>`, so a bodyless POST would work — and
/// that quietly defeated `deny_unknown_fields`: axum's `Option` extractor turns
/// a *parse failure* into `None`, so a caller sending `{"prompt": "a wizard"}`
/// got the defaults and a portrait drawn from the description, which looks
/// exactly like the prompt having been honoured.
pub async fn post_generate(
    State(s): State<Arc<Authored>>,
    headers: HeaderMap,
    Path(nid): Path<String>,
    Json(body): Json<GenerateBody>,
) -> Response {
    let (_, owner) = match crate::api::owner_of(&s, &headers).await {
        Ok(v) => v,
        Err(r) => return *r,
    };
    let Ok(npc_id) = nid.parse::<u64>() else {
        return refused(
            StatusCode::NOT_FOUND,
            "npc_not_found",
            "no such character",
            false,
        );
    };

    // The description and the existing portrait's origin, read under a *read*
    // lock and released before the drain. Holding the cast's write lock across
    // a stop-the-world image generation would block every character's tick, the
    // console's listing and every other write in the daemon for the duration.
    let (description, name, personality_id, existing_origin) = {
        let npcs = s.npcs.read().await;
        let Ok(record) = npcs.get(npc_id, &owner) else {
            return refused(
                StatusCode::NOT_FOUND,
                "npc_not_found",
                "no such character",
                false,
            );
        };
        (
            record["persona"]["description"]
                .as_str()
                .unwrap_or_default()
                .to_string(),
            record["name"]
                .as_str()
                .unwrap_or("this character")
                .to_string(),
            record["personality_id"]
                .as_str()
                .unwrap_or_default()
                .to_string(),
            record["portrait"]["origin"].as_str().map(str::to_string),
        )
    };

    if existing_origin.as_deref() == Some("uploaded") && !body.force {
        return refused(
            StatusCode::CONFLICT,
            "portrait_uploaded",
            "this character has a portrait its owner uploaded, and an upload outranks the \
             generator — send `force: true` to replace it",
            false,
        );
    }

    // The three sources, in the order the module header sets out: the request's
    // own words, then the personality's authored direction, then the
    // description. The personality is read under its own read lock and released
    // immediately — the drain below must not hold it.
    let authored = {
        let reg = s.personalities.read().await;
        reg.get(&personality_id)
            .and_then(|r| crate::personality_portrait::prompt(&r.body).map(str::to_string))
    };
    let prompt = choose_prompt(
        body.prompt.as_deref(),
        authored.as_deref(),
        &description,
        &name,
    );

    // Whichever source won, the prompt is caller-reachable text — a request
    // body, or a document edited through the console — so this route reaches
    // the generator exactly as `/v1/image/generate` does and is checked the same
    // way. Gating only the direct route would leave the character editor as the
    // way around it. See [`crate::compliance`].
    if let Err(denial) = crate::compliance::check(&s, &prompt).await {
        let (status, code, retry) = match denial {
            crate::compliance::Denial::Refused => (StatusCode::FORBIDDEN, "prompt_declined", false),
            crate::compliance::Denial::Unavailable(_) => {
                (StatusCode::SERVICE_UNAVAILABLE, "check_unavailable", true)
            }
        };
        return refused(status, code, &denial.to_string(), retry);
    }

    // Portraits stay on the standing checkpoint deliberately: they are the
    // world's own record art, and every character's should come off one brush.
    let request = GuestRequest::Image(ImageRequest {
        prompt,
        width: SIDE,
        height: SIDE,
        steps: STEPS,
        seed: body.seed,
        lora: ImageLora::default(),
        // A portrait is drawn *from the description*, so there is nothing to
        // start from but noise — and the deployment's own schedule, because
        // every character's portrait should come off one brush.
        reference: None,
        shift: None,
    });

    let png = match crate::guest_routes::run_guest(&s, request).await {
        Ok(GuestOutcome::Image(i)) => i,
        Ok(other) => {
            return refused(
                StatusCode::INTERNAL_SERVER_ERROR,
                "wrong_guest",
                &format!(
                    "the image request came back as {} — the queue routed by kind and should \
                     not have",
                    other.guest()
                ),
                false,
            )
        }
        Err(e) => return guest_refusal(&e),
    };

    // Stored before the record is touched: an id that names no bytes would be a
    // character pointing at a portrait that 404s, and the store is
    // content-addressed so a repeat draw of the same image costs nothing.
    let image_id = match s.images.put(&png.png) {
        Ok(id) => id,
        Err(e) => {
            return refused(
                StatusCode::INTERNAL_SERVER_ERROR,
                "image_store",
                &format!("the portrait was generated but could not be stored: {e:?}"),
                false,
            )
        }
    };

    let mut npcs = s.npcs.write().await;
    match npcs.set_portrait(npc_id, &owner, image_id, "generated", crate::api::now_ms()) {
        Ok(mut record) => {
            crate::api::name_personality(&mut record, &*s.personalities.read().await);
            // The seed rides along so an operator who liked a draw can ask for
            // it again. It is not on the record — it is a property of this
            // generation, not of the character.
            record["portrait"]["seed"] = json!(png.seed);
            Json(record).into_response()
        }
        Err(e) => crate::api::npc_err(e),
    }
}

/// `GET /v1/image/models` — what this daemon can actually draw with.
///
/// Registered in `engine::routes`, where the placeholder it replaces already
/// lived. A second registration in `api::routes` is not an override — axum
/// panics at startup with "overlapping method route".
///
/// The console has always called this and always got a 404, so it fell back to
/// "no image model is loaded" — which happened to be true and was not being
/// *reported*, it was being guessed at. Now it is the registry's own answer.
pub async fn get_models(State(s): State<Arc<Authored>>) -> Response {
    let configured = crate::guest_routes::configured(&s).await;
    let has_image = configured.contains(&candle_conversation::guest::Guest::Image);
    Json(json!({
        "models": if has_image {
            // One entry, because a deployment configures one image guest — the
            // kind is what the queue routes by, so there is nothing to pick
            // between. The console renders a picker when there is more than
            // one and this shape lets it keep doing that without a special case.
            vec![json!({
                "id": "guest-image",
                "display": "Z-Image-Turbo (co-resident)",
                // Not a fixed figure: the guest claims span ground per drain,
                // sized from the jobs in it. Reported as null rather than as a
                // number that would be wrong for every size but one.
                "vram_gib": serde_json::Value::Null,
                "loaded": true,
                "default": true,
            })]
        } else {
            Vec::new()
        },
        // What "loaded" means here, so the console is not claiming residency it
        // cannot see: the guest is *configured and ready to be loaded*, which is
        // the only state that matters to somebody deciding whether to press the
        // button. It is resident only during a drain.
        "resident_between_drains": false,
    }))
    .into_response()
}

fn refused(status: StatusCode, error: &str, detail: &str, retry: bool) -> Response {
    (
        status,
        Json(json!({ "error": error, "detail": detail, "retry": retry })),
    )
        .into_response()
}

/// A guest refusal, as the portrait route reports it.
///
/// The same status mapping [`crate::guest_routes`] uses, because a caller that
/// learns "retry this" from one route and "give up" from the other for the same
/// condition cannot act on either.
fn guest_refusal(e: &GuestError) -> Response {
    let (status, code) = match e {
        GuestError::Refused(_) => (StatusCode::BAD_REQUEST, "bad_request"),
        GuestError::NoRoom { .. } => (StatusCode::SERVICE_UNAVAILABLE, "no_room"),
        GuestError::Unavailable(_) => (StatusCode::NOT_IMPLEMENTED, "no_image_model"),
        GuestError::Failed(_) => (StatusCode::INTERNAL_SERVER_ERROR, "guest_failed"),
        GuestError::Abandoned => (StatusCode::SERVICE_UNAVAILABLE, "shutting_down"),
    };
    let retry = matches!(e, GuestError::NoRoom { .. } | GuestError::Abandoned);
    refused(status, code, &e.to_string(), retry)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// **The description is the subject, and the framing comes first.**
    ///
    /// A prompt that opens with the description gets a picture composed the way
    /// the description reads — a scene, a full figure, whatever the author was
    /// describing — because nothing has said "portrait" yet. Leading with the
    /// framing and following with the description is what makes it a portrait
    /// *of* them.
    #[test]
    fn a_description_becomes_a_portrait_prompt_with_the_framing_first() {
        let p = prompt_for("a quartermaster in his fifties, scarred left hand", "Hess");
        assert!(p.starts_with("character portrait, head and shoulders"));
        assert!(p.contains("a quartermaster in his fifties, scarred left hand"));
    }

    /// A character is allowed an empty description, so the prompt has to be
    /// buildable from a name alone — a refusal here would make the button fail
    /// on exactly the characters somebody is still writing.
    #[test]
    fn an_empty_description_still_makes_a_prompt() {
        let p = prompt_for("   ", "Hess");
        assert!(p.contains("Hess"));
        assert!(p.starts_with("character portrait"));
    }

    /// The description is trimmed, not passed through: a trailing newline from
    /// a textarea would otherwise end the prompt on whitespace the tokenizer
    /// spends a token on and the model reads as a pause.
    #[test]
    fn surrounding_whitespace_does_not_reach_the_prompt() {
        assert_eq!(
            prompt_for("\n  a woman with grey eyes \n", "X"),
            prompt_for("a woman with grey eyes", "X")
        );
    }

    /// **The framing carries what a negative prompt used to.**
    ///
    /// Z-Image-Turbo is guidance-distilled: it runs at `guidance_scale=0.0`, so
    /// there is no unconditioned branch to push away from and nothing a negative
    /// prompt could attach to. The constraints that mattered — one person, head
    /// and shoulders, not a full figure — are therefore stated positively or not
    /// at all, and this pins the ones that are.
    #[test]
    fn the_framing_states_what_a_portrait_is() {
        for must in ["character portrait", "head and shoulders", "centred"] {
            assert!(FRAMING.contains(must), "the framing drops {must:?}");
        }
    }

    /// The request this route builds must be one the engine accepts — a
    /// refusal here would only be discovered by pressing the button.
    #[test]
    fn the_request_it_builds_is_servable() {
        let r = GuestRequest::Image(ImageRequest {
            prompt: prompt_for("a tall man", "Hess"),
            width: SIDE,
            height: SIDE,
            steps: STEPS,
            seed: None,
            lora: ImageLora::default(),
            reference: None,
            shift: None,
        });
        assert!(r.check().is_ok(), "{:?}", r.check());
    }

    /// An empty body is a valid request — the console's button sends nothing.
    #[test]
    fn the_body_is_optional_and_defaults_to_not_forcing() {
        let b: GenerateBody = serde_json::from_str("{}").unwrap();
        assert!(!b.force);
        assert!(b.seed.is_none());
        assert!(GenerateBody::default().seed.is_none());
    }

    /// An unknown key is still refused rather than ignored — a caller sending a
    /// field this route does not have is asking for something it will not do,
    /// and drawing from the description anyway would look like it was honoured.
    #[test]
    fn an_unknown_field_is_refused_rather_than_ignored() {
        assert!(serde_json::from_str::<GenerateBody>(r#"{"stlye":"a wizard"}"#).is_err());
        assert!(serde_json::from_str::<GenerateBody>(r#"{"lora":"restricted"}"#).is_err());
    }

    /// A prompt in the body is read, because it is now a field this route has.
    #[test]
    fn a_prompt_in_the_body_is_accepted() {
        let b: GenerateBody = serde_json::from_str(r#"{"prompt":"a wizard"}"#).unwrap();
        assert_eq!(b.prompt.as_deref(), Some("a wizard"));
    }

    /// The request outranks the personality, which outranks the description.
    #[test]
    fn the_request_outranks_the_personality_which_outranks_the_description() {
        assert_eq!(
            choose_prompt(Some("a wizard"), Some("authored"), "a guard", "Keeper"),
            "a wizard"
        );
        assert_eq!(
            choose_prompt(None, Some("authored"), "a guard", "Keeper"),
            "authored"
        );
        let fallen = choose_prompt(None, None, "a guard", "Keeper");
        assert!(fallen.contains("a guard"), "{fallen}");
        assert!(fallen.starts_with("character portrait"), "{fallen}");
    }

    /// **Blank is absent, not an instruction to draw nothing.**
    ///
    /// The console's prompt box can be cleared, and a cleared box must fall
    /// back to the character's own words rather than sending the model an empty
    /// string — which draws something, just nothing anybody asked for. The same
    /// holds one rung down: a personality whose `prompt:` was started and left
    /// blank falls through to the description.
    #[test]
    fn a_blank_prompt_falls_back_rather_than_drawing_nothing() {
        for blank in ["", "   ", "\n\t "] {
            assert_eq!(
                choose_prompt(Some(blank), Some("authored"), "a guard", "Keeper"),
                "authored",
                "a blank request did not fall through"
            );
            let fallen = choose_prompt(Some(blank), Some(blank), "a guard", "Keeper");
            assert!(
                fallen.contains("a guard"),
                "a blank personality prompt did not fall through: {fallen}"
            );
        }
    }

    /// The chosen prompt is trimmed whichever rung it came from — a trailing
    /// newline out of a YAML block scalar or a textarea would otherwise end the
    /// prompt on whitespace the tokenizer spends a token on.
    #[test]
    fn the_chosen_prompt_is_trimmed_whichever_source_won() {
        assert_eq!(
            choose_prompt(Some("  a wizard\n"), None, "a guard", "Keeper"),
            "a wizard"
        );
        assert_eq!(
            choose_prompt(None, Some("\na guardian  "), "a guard", "Keeper"),
            "a guardian"
        );
    }
}
