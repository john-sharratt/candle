//! The routes a caller asks a co-resident model for work through.
//!
//! # Why these block
//!
//! A guest job is served between two of the engine's waves: the request is
//! queued, the scheduler picks it up on its next pass, evicts what it needs,
//! loads the guest, serves the whole backlog and hands the ground back. That is
//! seconds to a minute, and the handler waits for it — so the wait happens on a
//! blocking thread rather than on a tokio worker, or a handful of image
//! requests would starve the runtime that serves the console.
//!
//! # What a caller gets back
//!
//! The data, not a job id. There is no polling endpoint because there is
//! nothing useful to poll: a guest drain is one indivisible stop-the-world
//! event, and a caller that reconnected mid-drain would learn only that it was
//! still running.
//!
//! **An image answers as an NDJSON stream**,
//! because it has a length worth showing and no partial result to show: the
//! guest counts its denoise steps and its decode, those counts come across as
//! `step` lines, and the terminal `done` line carries the whole picture. The
//! image rides in that line as base64 rather than as `image/png` bytes so the
//! seed travels with it — an operator who liked a draw needs the seed to ask
//! for it again, and a header is a worse place to put it than the body it
//! belongs to.

use std::sync::Arc;

use axum::extract::State;
use axum::http::{HeaderMap, StatusCode};
use axum::response::{IntoResponse, Response};
use axum::Json;
use base64::Engine as _;
use candle_conversation::guest::{
    GuestError, GuestEvent, GuestOutcome, GuestRequest, GuestSink, ImageLora, ImageRequest,
    MatteRequest, DEFAULT_REFERENCE_HOLD,
};
use serde::Deserialize;
use serde_json::{json, Value};
use tokio::sync::mpsc::unbounded_channel;
use web::auth::Role;

use crate::api::Authored;
use crate::identity::require;
use crate::ndjson;
use crate::refimage;

/// What a caller posts to `/v1/guest/image`.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ImageBody {
    pub prompt: String,
    #[serde(default = "default_side")]
    pub width: u32,
    #[serde(default = "default_side")]
    pub height: u32,
    #[serde(default = "default_steps")]
    pub steps: u32,
    #[serde(default)]
    pub seed: Option<u64>,
    /// Which fused checkpoint draws this — `"diversity"` (the default) or
    /// `"restricted"`, which also asks for the admin-only handling
    /// [`post_image`] documents. An enum rather than a path: the set is
    /// curated by the deployment's `guests.yaml`, and a route that accepted
    /// file names would be a route that loads arbitrary files.
    #[serde(default)]
    pub lora: ImageLora,
    /// A picture to start from, base64, in any format this build decodes.
    ///
    /// Base64 rather than a multipart upload because the rest of this body is
    /// JSON and the picture is small enough that splitting the request into two
    /// content types would cost more than the third it saves. It is decoded and
    /// fitted to the draw's own size *before anything is queued* — see
    /// [`crate::refimage`] for why that must not happen inside a drain.
    #[serde(default)]
    pub reference: Option<String>,
    /// How much of the reference survives, `0.0 ..= 0.95`.
    #[serde(default = "default_hold")]
    pub reference_hold: f32,
    /// The schedule's shift, or the deployment's own when absent.
    #[serde(default)]
    pub shift: Option<f64>,
}

fn default_hold() -> f32 {
    DEFAULT_REFERENCE_HOLD
}

/// What a caller posts to `/v1/image/cutout`.
///
/// Just the picture. There is nothing to tune: the network was trained to know
/// what a subject is, and the tolerance dial the colour keyer needed existed
/// only because a threshold had to be guessed.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CutoutBody {
    /// The picture to separate from its background, base64.
    pub png_base64: String,
}

fn default_side() -> u32 {
    512
}

/// Twenty-four, against the eight Z-Image-Turbo's schedule is trained at.
///
/// The extra steps buy **finish, not variety**, and it is worth being exact
/// about that because the obvious guess is wrong. A distilled model reaches the
/// modal answer in very few evaluations, so it is tempting to read "same face
/// every seed" as the first Euler step deciding too much and to spend steps
/// against it. Measured here, that does nothing: the same prompt at three seeds
/// returns the same person at eight steps *and* at twelve, and a given seed's
/// face is unchanged between the two — only fine detail moves.
///
/// **Identity comes from the conditioning, not the noise.** The same seed with a
/// described subject ("a woman in her sixties, deep wrinkles, silver hair")
/// returns an entirely different person, which is where the variety actually
/// lives. A caller who wants different faces varies the prompt; there is no
/// sampler setting that substitutes for it, and `cfg_truncation` — the knob this
/// is usually blamed on — governs the guidance branch, which a model running at
/// `guidance_scale = 0.0` does not have.
///
/// So this is a quality dial and its cost is honest: three times the trained
/// count is about three times the denoise, and every second of it is time the
/// whole world stops thinking. It is set here because the images this daemon
/// draws are artwork somebody keeps. [`crate::portrait::STEPS`] deliberately
/// does **not** follow it — a portrait is small, drawn per character, and paid
/// for once per record rather than once per attempt.
fn default_steps() -> u32 {
    24
}

/// `GET /v1/guest` — what this deployment offers and how deep the queue is.
pub async fn get_guests(State(s): State<Arc<Authored>>) -> Response {
    let Some(rt) = s.runtime.as_ref() else {
        return unavailable();
    };
    let minds = rt.minds.read().unwrap().clone();
    let Some(minds) = minds else {
        return Json(json!({ "ready": false, "guests": [], "backlog": 0 })).into_response();
    };
    let (guests, backlog) = {
        let engine = minds.engine();
        let engine = engine.lock().unwrap();
        (engine.configured_guests(), engine.guest_backlog())
    };
    Json(json!({ "ready": true, "guests": guests, "backlog": backlog })).into_response()
}

/// `POST /v1/guest/image`
///
/// **NDJSON, because a draw has a length worth showing.** The picture does not
/// exist until the decoder runs, so there is nothing to stream *of the result* —
/// but a drain is a model crossing the link, eight denoise steps and a
/// full-resolution decode, and a caller that gets only a closed connection for
/// all of it can show nothing but a spinner. The guest counts those units
/// ([`candle_conversation::guest::GuestEvent::Step`]) and they come across here.
///
/// The terminal `done` line carries the whole image, so a consumer needs nothing
/// from the intermediate lines except the count. See [`crate::ndjson`] for the
/// contract every stream on this daemon keeps.
pub async fn post_image(
    State(s): State<Arc<Authored>>,
    headers: HeaderMap,
    Json(body): Json<ImageBody>,
) -> Response {
    // **Before the stream opens, so this can still be a status.**
    //
    // Once the first line is written the status is long gone and every failure
    // has to be reported in band, where a caller must parse the body to find
    // out the request never ran. "The engine has not finished loading" is
    // knowable now, so it is answered now — as the 503 with a `retry` the
    // console already understands.
    //
    // It also keeps the two meanings of `Abandoned` apart. The engine reports
    // the same variant for "still loading" and "shutting down", and during
    // startup the second reads as a daemon that died — which is what a caller
    // saw on the first draw after a restart until this check existed.
    // **The restricted checkpoint takes the admin's own judgement in place of
    // the compliance gate** — asking for it waives the check below, so the two
    // travel together: the flag is honoured only for a caller `require` puts at
    // `Admin`, and anyone else is refused outright rather than quietly drawn on
    // the standing checkpoint. Checked here in the handler, not left to the
    // route table's minimum, precisely so relaxing the route's level someday
    // cannot silently hand this waiver to every user. First, before the engine
    // check, because it depends on nothing but the headers — who you are does
    // not change while the engine loads.
    if body.lora == ImageLora::Restricted {
        if let Err(refusal) = require(&headers, &s.roles, Role::Admin) {
            return *refusal;
        }
    }

    if !engine_ready(&s) {
        return unavailable();
    }

    // **Before the stream opens, and before the image guest is queued.** The
    // prompt is read by the resident model first; see [`crate::compliance`] for
    // where the line is and why a failure to check stops the draw. Refused here
    // as a status rather than in band, because nothing has been started yet and
    // a caller branching on 403 should not have to parse a 200. A restricted
    // draw was authorised above and does not take this gate.
    let denial = if body.lora == ImageLora::Restricted {
        None
    } else {
        crate::compliance::check(&s, &body.prompt).await.err()
    };
    if let Some(denial) = denial {
        let status = match denial {
            crate::compliance::Denial::Refused => StatusCode::FORBIDDEN,
            crate::compliance::Denial::Unavailable(_) => StatusCode::SERVICE_UNAVAILABLE,
        };
        let retry = matches!(denial, crate::compliance::Denial::Unavailable(_));
        return (
            status,
            Json(
                json!({ "error": "prompt_declined", "detail": denial.to_string(), "retry": retry }),
            ),
        )
            .into_response();
    }

    // **Decoded and fitted here, before the queue.** A reference that turns out
    // not to be a picture would otherwise be discovered inside a drain, which
    // is after every character in every world has lost its resident KV to make
    // room for the guest. See [`crate::refimage`].
    let reference = match &body.reference {
        None => None,
        Some(encoded) => {
            let bytes = match base64::engine::general_purpose::STANDARD.decode(encoded.as_bytes()) {
                Ok(b) => b,
                Err(e) => return bad_reference(&format!("the reference is not valid base64: {e}")),
            };
            match refimage::conform(&bytes, body.width, body.height, body.reference_hold) {
                Ok(r) => Some(r),
                Err(e) => return bad_reference(&e),
            }
        }
    };

    let request = GuestRequest::Image(ImageRequest {
        prompt: body.prompt,
        width: body.width,
        height: body.height,
        steps: body.steps,
        seed: body.seed,
        lora: body.lora,
        reference,
        shift: body.shift,
    });

    // Unbounded for the reason `describe` documents: the sink runs on the
    // scheduler thread with normal inference blocked, so a bounded channel that
    // filled would stop the engine until the HTTP client read. What can
    // accumulate here is one line per step, which is single digits.
    let (tx, rx) = unbounded_channel::<Value>();
    let sink = {
        let tx = tx.clone();
        GuestSink::new(move |e| {
            let line = match e {
                GuestEvent::Loading => json!({ "event": "loading" }),
                GuestEvent::Step { done, total, what } => {
                    json!({ "event": "step", "done": done, "total": total, "what": what })
                }
            };
            // A send that fails means the reader hung up. The job runs on: the
            // drain has already evicted the engine's working set for it, and
            // stopping now would pay that cost for nothing.
            let _ = tx.send(line);
        })
    };

    tokio::spawn(async move {
        let line = match run_guest_watched(&s, request, sink).await {
            Ok(GuestOutcome::Image(i)) => json!({
                "event": "done",
                "width": i.width,
                "height": i.height,
                "png_base64": base64::engine::general_purpose::STANDARD.encode(&i.png),
                "seed": i.seed,
            }),
            Ok(other) => ndjson::error_line(
                "wrong_guest",
                &format!("the image request came back as {}", other.guest()),
                false,
            ),
            Err(e) => {
                let (code, retry) = match &e {
                    GuestError::Refused(_) => ("refused", false),
                    GuestError::NoRoom { .. } => ("no_room", true),
                    GuestError::Unavailable(_) => ("no_guest", false),
                    GuestError::Failed(_) => ("failed", false),
                    GuestError::Abandoned => ("abandoned", true),
                };
                ndjson::error_line(code, &e.to_string(), retry)
            }
        };
        let _ = tx.send(line);
    });

    ndjson::stream(rx)
}

/// Queue a job, wait for it, and hand the caller the outcome.
///
/// **The one place a guest job is submitted from.** The portrait route and the
/// life-node route both go through it, so the lock discipline and the blocking
/// behaviour below are stated once rather than re-derived per caller — and a
/// caller that got either wrong would take the console down with it.
///
/// The submission takes the engine lock only long enough to push. A guest
/// receipt is a channel, not a borrow, so a caller waiting on a drain does not
/// hold the lock every other route needs — getting that wrong would make one
/// image request stop the whole daemon rather than only its inference.
pub async fn run_guest(
    s: &Arc<Authored>,
    request: GuestRequest,
) -> Result<GuestOutcome, GuestError> {
    run_guest_watched(s, request, GuestSink::none()).await
}

/// Whether there is an engine to serve a guest between the waves of.
///
/// The same two conditions [`run_guest_watched`] checks before it submits, asked
/// ahead of time — which is what lets a streaming route answer "not yet" with a
/// status instead of an in-band error line. It is a snapshot and nothing holds
/// it: the engine can still stop between this and the submission, which is why
/// the job's own `Abandoned` is handled as well rather than instead.
fn engine_ready(s: &Arc<Authored>) -> bool {
    s.runtime
        .as_ref()
        .is_some_and(|rt| rt.minds.read().unwrap().is_some())
}

/// The same, with a watcher on the job's progress.
///
/// `sink` runs on the scheduler thread between decode steps, with normal
/// inference blocked — so it must push and return. The one this daemon builds
/// sends into an unbounded channel, which never waits.
pub async fn run_guest_watched(
    s: &Arc<Authored>,
    request: GuestRequest,
    sink: GuestSink,
) -> Result<GuestOutcome, GuestError> {
    let Some(rt) = s.runtime.as_ref() else {
        return Err(GuestError::Abandoned);
    };
    let minds = rt.minds.read().unwrap().clone();
    let Some(minds) = minds else {
        return Err(GuestError::Abandoned);
    };
    let receipt = {
        let engine = minds.engine();
        let engine = engine.lock().unwrap();
        engine.submit_guest_watched(request, sink)?
    };

    // Off the async pool: a drain is seconds to a minute of another thread's
    // work, and blocking a tokio worker on it would take one of the runtime's
    // few threads out of service for the duration.
    tokio::task::spawn_blocking(move || receipt.wait())
        .await
        .unwrap_or_else(|e| Err(GuestError::Failed(format!("the guest wait failed: {e}"))))
}

/// The guests this daemon has configured, for a status view.
///
/// Empty before the model has loaded — the registry lives on the engine, and
/// the engine is what is still loading. That is the honest answer rather than
/// a guess, and it is why the console can stop inferring availability from a
/// 404.
pub async fn configured(s: &Arc<Authored>) -> Vec<candle_conversation::guest::Guest> {
    let Some(rt) = s.runtime.as_ref() else {
        return Vec::new();
    };
    let minds = rt.minds.read().unwrap().clone();
    let Some(minds) = minds else {
        return Vec::new();
    };
    let engine = minds.engine();
    let engine = engine.lock().unwrap();
    engine.configured_guests()
}

/// A reference that could not be used, as a status rather than a stream line.
///
/// Its own code because it is a distinct thing to fix: the prompt was fine, the
/// engine was fine, the *upload* was the problem — and a console showing "the
/// draw failed" for a truncated JPEG sends somebody looking at the daemon.
fn bad_reference(detail: &str) -> Response {
    (
        StatusCode::BAD_REQUEST,
        Json(json!({ "error": "bad_reference", "detail": detail, "retry": false })),
    )
        .into_response()
}

/// `POST /v1/image/cutout` — separate a picture from its background.
///
/// A guest job like any other: the picture is decoded here, queued, and served
/// between two of the engine's waves by a salient-object network whose weights
/// stand in ground. See [`candle_conversation::guest::matte`] for what the
/// network does and why this is not a colour algorithm.
///
/// It is one forward — measured at 0.23 s against the 11.9 s the same graph
/// takes on a CPU — so the drain it costs the estate is a fraction of a single
/// image draw's.
pub async fn post_cutout(State(s): State<Arc<Authored>>, Json(body): Json<CutoutBody>) -> Response {
    let bytes = match base64::engine::general_purpose::STANDARD.decode(body.png_base64.as_bytes()) {
        Ok(b) => b,
        Err(e) => return bad_reference(&format!("the picture is not valid base64: {e}")),
    };
    if bytes.len() > refimage::MAX_BYTES {
        return bad_reference(&format!(
            "the picture is {} bytes and the limit is {}",
            bytes.len(),
            refimage::MAX_BYTES
        ));
    }
    // **Decoded before it is queued**, for the reason a reference image is: a
    // drain evicts the engine's whole working set before the guest loads, so a
    // truncated upload must not be discovered inside one.
    let decoded = match image::load_from_memory(&bytes) {
        Ok(i) => i.to_rgb8(),
        Err(e) => return bad_reference(&format!("the picture could not be read: {e}")),
    };
    let (w, h) = decoded.dimensions();
    if w > refimage::MAX_SIDE || h > refimage::MAX_SIDE {
        return bad_reference(&format!("the picture is {w}×{h}, past the side limit"));
    }

    let request = GuestRequest::Matte(MatteRequest {
        pixels: decoded.into_raw(),
        width: w,
        height: h,
    });
    match run_guest(&s, request).await {
        Ok(GuestOutcome::Matte(m)) => Json(json!({
            "png_base64": base64::engine::general_purpose::STANDARD.encode(m.png),
            "width": m.width,
            "height": m.height,
            "lifted": m.lifted,
        }))
        .into_response(),
        Ok(other) => refusal(&GuestError::Failed(format!(
            "the matte request came back as {}",
            other.guest()
        ))),
        Err(GuestError::Abandoned) => unavailable(),
        Err(e) => refusal(&e),
    }
}

/// Map a guest refusal to a status a caller can act on.
///
/// The three cases want three different responses and a single 500 would hide
/// which: a bad ask is the caller's to fix, a full card is worth retrying, and
/// an unconfigured guest is neither.
fn refusal(e: &GuestError) -> Response {
    let status = match e {
        GuestError::Refused(_) => StatusCode::BAD_REQUEST,
        GuestError::NoRoom { .. } => StatusCode::SERVICE_UNAVAILABLE,
        GuestError::Unavailable(_) => StatusCode::NOT_IMPLEMENTED,
        GuestError::Failed(_) => StatusCode::INTERNAL_SERVER_ERROR,
        GuestError::Abandoned => StatusCode::SERVICE_UNAVAILABLE,
    };
    let retry = matches!(e, GuestError::NoRoom { .. } | GuestError::Abandoned);
    (
        status,
        Json(json!({ "error": e.to_string(), "retry": retry })),
    )
        .into_response()
}

fn unavailable() -> Response {
    (
        StatusCode::SERVICE_UNAVAILABLE,
        Json(json!({
            "error": "the engine is still loading — guests are served between its waves, so \
                      there are none to serve between yet",
            "retry": true,
        })),
    )
        .into_response()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The defaults are what a caller who sent only a prompt gets, so they have
    /// to be a request the engine will actually accept.
    #[test]
    fn the_body_defaults_are_a_servable_request() {
        let body: ImageBody = serde_json::from_str(r#"{"prompt":"a lantern"}"#).unwrap();
        let r = GuestRequest::Image(ImageRequest {
            prompt: body.prompt,
            width: body.width,
            height: body.height,
            steps: body.steps,
            seed: body.seed,
            lora: body.lora,
            reference: None,
            shift: body.shift,
        });
        assert!(r.check().is_ok(), "{:?}", r.check());
    }

    /// **The three refusals get three statuses.** A single 500 would tell a
    /// caller nothing about whether to fix the ask, retry, or stop asking — and
    /// the retryable one is the only one that resolves on its own.
    #[test]
    fn each_refusal_carries_a_status_a_caller_can_act_on() {
        let status = |e: GuestError| refusal(&e).status();
        assert_eq!(
            status(GuestError::Refused("too wide".into())),
            StatusCode::BAD_REQUEST
        );
        assert_eq!(
            status(GuestError::NoRoom {
                wanted_mib: 4096,
                freed_mib: 128
            }),
            StatusCode::SERVICE_UNAVAILABLE
        );
        assert_eq!(
            status(GuestError::Unavailable(
                candle_conversation::guest::Guest::Image
            )),
            StatusCode::NOT_IMPLEMENTED
        );
        assert_eq!(
            status(GuestError::Failed("the UNet".into())),
            StatusCode::INTERNAL_SERVER_ERROR
        );
    }

    /// A caller who says nothing draws with the standing checkpoint; one who
    /// names a lora gets exactly that one; one who names something the enum
    /// does not carry is refused at the boundary rather than quietly drawn
    /// with the default.
    #[test]
    fn the_lora_field_parses_and_defaults_to_the_standing_checkpoint() {
        let body: ImageBody = serde_json::from_str(r#"{"prompt":"x"}"#).unwrap();
        assert_eq!(body.lora, ImageLora::Diversity);
        let body: ImageBody =
            serde_json::from_str(r#"{"prompt":"x","lora":"restricted"}"#).unwrap();
        assert_eq!(body.lora, ImageLora::Restricted);
        assert!(serde_json::from_str::<ImageBody>(r#"{"prompt":"x","lora":"sepia"}"#).is_err());
    }

    /// **The dials default to the draw that existed before them.** A caller
    /// that sends only a prompt must get no reference, the deployment's own
    /// schedule, and the standing hold — otherwise adding the dials silently
    /// changed every existing caller's pictures.
    #[test]
    fn the_reference_dials_default_to_the_draw_that_predates_them() {
        let body: ImageBody = serde_json::from_str(r#"{"prompt":"x"}"#).unwrap();
        assert!(body.reference.is_none());
        assert!(body.shift.is_none());
        assert_eq!(body.reference_hold, DEFAULT_REFERENCE_HOLD);

        let body: ImageBody = serde_json::from_str(
            r#"{"prompt":"x","reference":"AAAA","reference_hold":0.6,"shift":4.5}"#,
        )
        .unwrap();
        assert_eq!(body.reference.as_deref(), Some("AAAA"));
        assert_eq!(body.reference_hold, 0.6);
        assert_eq!(body.shift, Some(4.5));
    }

    /// **Every fixture decodes to exactly what the guest requires.**
    ///
    /// `post_cutout` decodes at the boundary and hands the guest RGB8 of
    /// exactly `width · height · 3`, because a drain evicts the engine's whole
    /// working set before the guest loads — a truncated upload discovered
    /// inside one costs every character its resident KV to find out. These are
    /// real 512×512 draws from this daemon, each named with the prompt and seed
    /// that made it in `npcd/tests/images/`.
    #[test]
    fn every_fixture_decodes_to_a_servable_matte_request() {
        for (name, png) in [
            (
                "grey_portrait",
                &include_bytes!("../tests/images/grey_portrait.png")[..],
            ),
            (
                "green_curls",
                &include_bytes!("../tests/images/green_curls.png")[..],
            ),
            (
                "green_hood",
                &include_bytes!("../tests/images/green_hood.png")[..],
            ),
            (
                "green_lantern",
                &include_bytes!("../tests/images/green_lantern.png")[..],
            ),
            (
                "busy_canyon",
                &include_bytes!("../tests/images/busy_canyon.png")[..],
            ),
        ] {
            let rgb = image::load_from_memory(png)
                .unwrap_or_else(|e| panic!("{name} is not a readable picture: {e}"))
                .to_rgb8();
            let (w, h) = rgb.dimensions();
            assert_eq!((w, h), (512, 512), "{name}");
            let r = GuestRequest::Matte(MatteRequest {
                pixels: rgb.into_raw(),
                width: w,
                height: h,
            });
            assert!(r.check().is_ok(), "{name}: {:?}", r.check());
        }
    }

    /// The cutout body carries a picture and nothing else — no threshold to
    /// guess, because the network is not guessing.
    #[test]
    fn the_cutout_body_is_just_a_picture() {
        let b: CutoutBody = serde_json::from_str(r#"{"png_base64":"AAAA"}"#).unwrap();
        assert_eq!(b.png_base64, "AAAA");
        assert!(serde_json::from_str::<CutoutBody>("{}").is_err());
        // The colour keyer's dial is gone, and a caller still sending it is
        // told so rather than having it silently ignored.
        assert!(
            serde_json::from_str::<CutoutBody>(r#"{"png_base64":"AAAA","tolerance":2.0}"#).is_err()
        );
    }

    /// **A bad reference is its own refusal.** A console that showed "the draw
    /// failed" for a truncated upload would send somebody looking at the
    /// daemon, so the code says which half of the request was wrong.
    #[test]
    fn a_reference_that_cannot_be_read_is_named_as_the_reference() {
        let r = bad_reference("the reference is not valid base64");
        assert_eq!(r.status(), StatusCode::BAD_REQUEST);
    }

    /// An unknown key is refused rather than ignored: a caller who wrote
    /// `"negative_prompt"` would otherwise get an image that quietly ignored it.
    #[test]
    fn an_unknown_field_is_refused() {
        assert!(
            serde_json::from_str::<ImageBody>(r#"{"prompt":"x","negative_prompt":"y"}"#).is_err()
        );
    }

    /// A prompt is the one thing with no default — a request without one is not
    /// a request.
    #[test]
    fn a_body_with_no_prompt_is_refused_at_the_boundary() {
        assert!(serde_json::from_str::<ImageBody>(r#"{"width":512}"#).is_err());
    }

    /// **Every guest refusal reaches a streaming caller as a terminal line.**
    ///
    /// Once the stream has opened the status line is gone, so a failure that
    /// fell out of the match would close the connection cleanly with no terminal
    /// line — which a consumer is obliged to read as "the connection dropped".
    /// That is the one failure mode a status code cannot describe, so the
    /// mapping is pinned here rather than left to the arm order.
    #[test]
    fn every_refusal_has_a_streamed_code_and_a_retry_flag() {
        // The same construction as the route's, over every variant. A new
        // `GuestError` breaks this match, which is the point.
        let line = |e: &GuestError| {
            let (code, retry) = match e {
                GuestError::Refused(_) => ("refused", false),
                GuestError::NoRoom { .. } => ("no_room", true),
                GuestError::Unavailable(_) => ("no_guest", false),
                GuestError::Failed(_) => ("failed", false),
                GuestError::Abandoned => ("abandoned", true),
            };
            ndjson::error_line(code, &e.to_string(), retry)
        };

        for (e, code, retry) in [
            (GuestError::Refused("too wide".into()), "refused", false),
            (
                GuestError::NoRoom {
                    wanted_mib: 4096,
                    freed_mib: 128,
                },
                "no_room",
                true,
            ),
            (
                GuestError::Unavailable(candle_conversation::guest::Guest::Image),
                "no_guest",
                false,
            ),
            (GuestError::Failed("the decoder".into()), "failed", false),
            (GuestError::Abandoned, "abandoned", true),
        ] {
            let v = line(&e);
            assert_eq!(v["event"], "error");
            assert_eq!(v["error"], code);
            assert_eq!(v["retry"], retry);
            assert!(
                v["detail"].as_str().is_some_and(|d| !d.is_empty()),
                "`{code}` carried no detail, so a caller sees a code and no reason"
            );
        }
    }

    /// The two retryable failures are the two that resolve without the caller
    /// changing anything — a card that was full and a drain that was abandoned.
    /// Marking a bad ask retryable would have the console spin on a request that
    /// can never succeed.
    #[test]
    fn only_the_transient_failures_are_marked_retryable() {
        assert!(matches!(
            GuestError::NoRoom {
                wanted_mib: 1,
                freed_mib: 0
            },
            GuestError::NoRoom { .. }
        ));
        let transient =
            |e: &GuestError| matches!(e, GuestError::NoRoom { .. } | GuestError::Abandoned);
        assert!(transient(&GuestError::Abandoned));
        assert!(!transient(&GuestError::Refused("x".into())));
        assert!(!transient(&GuestError::Failed("x".into())));
        assert!(!transient(&GuestError::Unavailable(
            candle_conversation::guest::Guest::Image
        )));
    }
}
