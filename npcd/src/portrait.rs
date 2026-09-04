//! Generating a character's portrait from its own description.
//!
//! # Why there is no prompt field
//!
//! The console's create step says it, and it is a real decision rather than a
//! simplification: the portrait derives from `persona.description`, so there is
//! nowhere for the two to drift apart. A character whose description says
//! "a quartermaster in his fifties, scarred left hand" and whose portrait shows
//! a young woman is a character nobody can use, and the way that happens is a
//! prompt field somebody edited once and forgot.
//!
//! What this module does is turn the description into an image prompt — adding
//! the framing a *portrait* needs and the description does not carry — and hand
//! it to the image guest.
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
    let (description, name, existing_origin) = {
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

    // The description is the prompt, and a description is written by whoever
    // owns the character — so this route reaches the generator with caller text
    // exactly as `/v1/image/generate` does, and is checked the same way. Gating
    // only the direct route would leave the character editor as the way around
    // it. See [`crate::compliance`].
    let prompt = prompt_for(&description, &name);
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

    /// An unknown key is refused rather than ignored: a caller sending
    /// `{"prompt": "..."}` is asking for something this route deliberately does
    /// not offer, and silently drawing from the description instead would look
    /// like the prompt was honoured.
    #[test]
    fn an_unknown_field_is_refused_rather_than_ignored() {
        assert!(serde_json::from_str::<GenerateBody>(r#"{"prompt":"a wizard"}"#).is_err());
    }
}
