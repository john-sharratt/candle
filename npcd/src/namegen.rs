//! Naming a character, in the register of the world it belongs to.
//!
//! # Why a name is generated separately, and first
//!
//! A description used to invent its own name mid-paragraph, which meant the two
//! could not be written independently: an author who liked the name and not the
//! prose had to regenerate both and lose the name. Naming first inverts that —
//! the name is a fact by the time the description is written, and
//! [`crate::describe`] writes *about that person* rather than inventing one.
//!
//! It also fills the create form immediately. A name is a handful of tokens, so
//! it arrives while the author is still reading the world they picked, and the
//! description follows into a form that is no longer empty.
//!
//! # What it is written against
//!
//! The world's `setting` — five or six sentences of high-level summary that
//! every world in the mind carries, deliberately kept short because the body of
//! the canon lives in `layers/world/` and is gathered by relevance instead.
//! That summary is exactly the right size for a naming prompt: enough to fix the
//! register (uploaded consciousness in towers, or a quiet contemporary house)
//! without spending the context on canon a name cannot express.

use std::sync::Arc;

use axum::extract::State;
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use axum::Json;
use candle_conversation::guest::{
    resolve_seed, GuestOutcome, GuestRequest, GuestSink, ProseRequest, Seeded,
};
use serde::Deserialize;
use serde_json::json;

use crate::api::Authored;
use crate::describe::INITIALS;

/// The voice, and the shape of the answer.
///
/// "Only the name" is stated three ways because a chat-tuned model wants to
/// introduce its answer, and a name arriving as *Certainly! Here is a name:
/// Aelis* lands in the form field verbatim.
const VOICE: &str = "You name characters for a fiction engine. Given a setting, invent ONE \
     person's name that belongs in that world. Reply with the name and nothing else — no \
     preamble, no explanation, no quotation marks, no punctuation after it. Two or three words \
     at most.";

/// A name is a few tokens; this is the ceiling that stops a model which ignores
/// the instruction from writing a paragraph into a name field.
const MAX_TOKENS: u32 = 24;

/// Warm, but not as warm as it was.
///
/// This was 1.0 on the reasoning that a name is one short draw and variety is
/// the point. The variety comes from the initial in the prompt, not from the
/// sampler, and 1.0 mostly bought derailment — the model reached the end of the
/// name and kept going into whatever was next most likely.
const TEMPERATURE: f32 = 0.85;

/// The longest name accepted before it is treated as prose that got away.
const MAX_NAME_CHARS: usize = 48;

#[derive(Debug, Default, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct NameBody {
    /// The world the character will live in. Its `setting` is the whole of the
    /// context, so a request without one is refused rather than answered from
    /// nowhere.
    #[serde(default)]
    pub world_id: String,
    /// The personality they will be cast as, if one is chosen yet. Optional.
    #[serde(default)]
    pub personality_id: String,
    /// Pin the draw, so a name an author liked can be reproduced.
    #[serde(default)]
    pub seed: Option<u64>,
}

/// What the model is told, and what it is asked.
///
/// Public to the crate and separate from the route for the same reason
/// [`crate::describe::brief_for`] is: it is the whole of the decision here, and
/// it is testable without a card, a checkpoint or a daemon.
pub fn name_brief(
    world: &serde_json::Value,
    personality: Option<&serde_json::Value>,
    seed: u64,
) -> crate::describe::Brief {
    let mut context = String::new();
    let name = world["name"].as_str().unwrap_or("this world");
    context.push_str(&format!("THE WORLD\n{name}\n"));
    if let Some(setting) = world["setting"].as_str().filter(|s| !s.trim().is_empty()) {
        context.push_str(&format!("\n{}\n", setting.trim()));
    }
    /* **The anchor, not just the label.**
     *
     * This named the personality and stopped — "THEY WILL BE CAST AS Keeper" —
     * which tells a naming model almost nothing. A personality's *name* is a
     * slug an author chose; its anchor is who the character is, and that is what
     * a name has to suit. An ancient tower intelligence and a frontier
     * quartermaster want different names, and only one of those facts was
     * reaching the model.
     *
     * Same extract the description uses, so the two prompts agree about what a
     * personality is rather than each taking their own slice of it.
     */
    if let Some(p) = personality {
        let pname = crate::describe::personality_label(p);
        context.push_str(&format!("\nTHEY WILL BE CAST AS\n{pname}\n"));
        if let Some(anchor) = p["anchor"].as_str().filter(|s| !s.trim().is_empty()) {
            context.push_str(&format!("{}\n", crate::describe::anchor_extract(anchor)));
        }
    }

    // **The same first-token problem the description had, and worse.** A name
    // *is* the first token, so with a fixed prompt one draw in three came back
    // identical whatever the seed. Naming an initial moves the distribution
    // rather than re-rolling against it — see [`INITIALS`].
    let mut spice = Seeded::new(seed);
    let initial = spice.pick(&INITIALS).copied().unwrap_or('A');
    let task = format!("Invent one person's name for {name}. It begins with the letter {initial}.");

    crate::describe::Brief { context, task }
}

/// The most words a name may have before the rest is treated as the model
/// carrying on past the answer.
const MAX_NAME_WORDS: usize = 4;

/// Lowercase words that belong *inside* a name.
///
/// The only lowercase words a name may contain. Everything else lowercase ends
/// the name, which is what stops a sentence from being read as one.
const PARTICLES: [&str; 14] = [
    "van", "von", "de", "del", "della", "der", "den", "di", "du", "la", "le", "af", "bin", "al",
];

/// Reduce whatever the model said to a name.
///
/// **A twenty-token budget is not a stop instruction.** Asked for a name, the
/// model gives one and then keeps going, because nothing in the decode ends at
/// the end of the answer: real outputs here were `Gaelen<REASONING>`,
/// `Narvius})\` and `Orionisors 다운받기`. Every one of those has a good name at
/// the front and debris behind it, so the debris is cut rather than the whole
/// answer refused.
///
/// The cut is by shape, not by a list of things seen once. A name is a short run
/// of capitalised words: take the leading run of characters a name can contain,
/// then keep words while they *look* like name words. That ends the answer at
/// `<`, at `}`, and at a word in a script with no capitals — without needing to
/// have anticipated any of them.
pub fn clean_name(raw: &str) -> String {
    let line = raw
        .lines()
        .map(str::trim)
        .find(|l| !l.is_empty())
        .unwrap_or("");
    // Quotes and list markers, then a colon meaning it introduced itself —
    // "Name: Aelis" — so take what follows.
    let line = line
        .trim_start_matches(['-', '*', '•', '#'])
        .trim()
        .trim_matches(['"', '\'', '“', '”', '‘', '’'])
        .trim();
    let line = line.rsplit(':').next().unwrap_or(line).trim();

    // The leading run of characters that can appear in a name. Stops dead at
    // the first bracket, slash or angle — which is where the debris starts.
    let head: String = line
        .chars()
        .take_while(|c| c.is_alphabetic() || matches!(c, ' ' | '\'' | '’' | '-' | '.'))
        .collect();

    // Then word by word: a name's words are capitalised. This is what ends
    // `Orionisors 다운받기` at the first word, since Hangul has no upper case —
    // and it costs nothing on a real name, whose every word has one.
    let mut words: Vec<&str> = Vec::new();
    for w in head.split_whitespace() {
        let first = w.chars().next().unwrap_or(' ');
        // Lowercase particles are ordinary inside a name but never open one.
        // Named explicitly rather than by length: "is" and "a" are two letters
        // too, and `Hess is a quartermaster` would otherwise read as a name.
        let particle = PARTICLES.contains(&w.to_lowercase().as_str());
        let ok = first.is_uppercase() || (!words.is_empty() && particle);
        if !ok || words.len() == MAX_NAME_WORDS {
            break;
        }
        words.push(w);
    }
    let name = words.join(" ");
    let name = name.trim_end_matches(['.', ',', '-', '\'']).trim();

    if name.is_empty()
        || name.chars().count() > MAX_NAME_CHARS
        || !name.chars().any(char::is_alphabetic)
    {
        return String::new();
    }
    name.to_string()
}

/// `POST /v1/generate/name`
///
/// Blocks for the length of a drain, like every guest route — but a name is
/// twenty tokens, so the drain is dominated by the model load the description
/// that follows will reuse.
pub async fn post_name(State(s): State<Arc<Authored>>, Json(body): Json<NameBody>) -> Response {
    if body.world_id.trim().is_empty() {
        return fail(
            StatusCode::BAD_REQUEST,
            "no_world",
            "a name is written against a world's setting — name one",
        );
    }
    let world = match s.worlds.read().await.get(&body.world_id) {
        Some(w) => w.body.clone(),
        None => return fail(StatusCode::NOT_FOUND, "world_not_found", &body.world_id),
    };
    let personality = if body.personality_id.trim().is_empty() {
        None
    } else {
        match s.personalities.read().await.get(&body.personality_id) {
            // With its id — see [`crate::describe::personality_label`].
            Some(p) => Some(crate::api::with_id("personality_id", &p.id, &p.body)),
            None => {
                return fail(
                    StatusCode::NOT_FOUND,
                    "personality_not_found",
                    &body.personality_id,
                )
            }
        }
    };

    let seed = resolve_seed(body.seed);
    let brief = name_brief(&world, personality.as_ref(), seed);
    let request = GuestRequest::Prose(ProseRequest {
        system: format!("{VOICE}\n\n{}", brief.context),
        prompt: brief.task,
        max_tokens: MAX_TOKENS,
        temperature: Some(TEMPERATURE),
        seed: Some(seed),
        // A name is free prose — the set of good ones is not enumerable.
        choices: None,
    });

    match crate::guest_routes::run_guest_watched(&s, request, GuestSink::none()).await {
        Ok(GuestOutcome::Prose { text, seed, .. }) => {
            let name = clean_name(&text);
            if name.is_empty() {
                return fail(
                    StatusCode::BAD_GATEWAY,
                    "empty_draft",
                    "the model produced no usable name",
                );
            }
            Json(json!({
                "name": name,
                "seed": seed,
                "world_id": body.world_id,
                "personality_id": body.personality_id,
            }))
            .into_response()
        }
        Ok(other) => fail(
            StatusCode::INTERNAL_SERVER_ERROR,
            "wrong_guest",
            &format!("the prose request came back as {}", other.guest()),
        ),
        Err(e) => crate::describe::guest_refusal(&e),
    }
}

fn fail(status: StatusCode, error: &str, detail: &str) -> Response {
    (status, Json(json!({ "error": error, "detail": detail }))).into_response()
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use std::collections::HashSet;

    fn world() -> serde_json::Value {
        json!({
            "id": "battle-cities",
            "name": "Battle Cities",
            "setting": "Three centuries after the Great War, humanity survives as uploaded \
                        consciousness in the towers.",
        })
    }

    /// **The world's summary is the whole of the context.** A name invented
    /// without it is a name from nowhere, and it is the first thing an author
    /// sees on the create form.
    #[test]
    fn the_worlds_setting_is_the_context() {
        let b = name_brief(&world(), None, 4);
        assert!(b.context.contains("Battle Cities"));
        assert!(b.context.contains("uploaded"));
        assert!(b.task.contains("Battle Cities"));
    }

    /// The same first-token clustering the description had — worse here,
    /// because the name *is* the first token.
    #[test]
    fn the_brief_varies_with_the_seed() {
        let tasks: HashSet<String> = (0..200)
            .map(|s| name_brief(&world(), None, s).task)
            .collect();
        assert!(
            tasks.len() > 20,
            "200 seeds produced only {} distinct briefs",
            tasks.len()
        );
    }

    /// The same seed rebuilds the same brief, or a reported seed reproduces
    /// nothing.
    #[test]
    fn one_seed_rebuilds_one_brief() {
        for seed in [0u64, 7, 99, u64::MAX] {
            assert_eq!(
                name_brief(&world(), None, seed).task,
                name_brief(&world(), None, seed).task
            );
        }
    }

    /// **A chat-tuned model introduces its answer.** Every one of these is a
    /// shape a model actually reaches for, and each would otherwise land in the
    /// form's name field verbatim.
    #[test]
    fn the_models_wrapping_is_stripped() {
        for (raw, want) in [
            ("Aelis Maelstrom", "Aelis Maelstrom"),
            ("\"Aelis Maelstrom\"", "Aelis Maelstrom"),
            ("Aelis Maelstrom.", "Aelis Maelstrom"),
            ("- Aelis Maelstrom", "Aelis Maelstrom"),
            ("Name: Aelis Maelstrom", "Aelis Maelstrom"),
            ("  Aelis Maelstrom  \nsomething else", "Aelis Maelstrom"),
            ("“Aelis Maelstrom”", "Aelis Maelstrom"),
        ] {
            assert_eq!(clean_name(raw), want, "for {raw:?}");
        }
    }

    /// **The model does not stop at the end of the name.** Every one of these
    /// came off the real guest: a good name with the model's continuation
    /// welded to it, because a token budget ends a decode and nothing ends the
    /// *answer*. The name is in front, so the debris is cut rather than the
    /// whole draw thrown away.
    #[test]
    fn the_models_continuation_is_cut_off_the_end() {
        for (raw, want) in [
            ("Gaelen<REASONING>", "Gaelen"),
            ("Narvius})\\", "Narvius"),
            ("Orionisors 다운받기", "Orionisors"),
            ("Zephyrion", "Zephyrion"),
            ("Aelis Maelstrom, of House Cyclone", "Aelis Maelstrom"),
            ("Hess (a quartermaster)", "Hess"),
            ("Tam Sorrel\nHe is a", "Tam Sorrel"),
        ] {
            assert_eq!(clean_name(raw), want, "for {raw:?}");
        }
    }

    /// A lowercase particle belongs inside a name and never opens one, so the
    /// run ends at the first ordinary word — which is what stops a sentence
    /// being read as a very long name.
    #[test]
    fn a_particle_is_kept_and_a_sentence_is_not() {
        assert_eq!(clean_name("Ivo van Renn"), "Ivo van Renn");
        assert_eq!(clean_name("Hess is a quartermaster"), "Hess");
        assert_eq!(clean_name("the quartermaster"), "");
    }

    /// A paragraph is not a name. Returning an empty string is what lets the
    /// route answer "no usable name" rather than writing prose into a field
    /// that is one line tall.
    #[test]
    fn prose_that_got_away_is_refused() {
        assert_eq!(clean_name("   "), "");
        assert_eq!(clean_name(""), "");
        assert_eq!(clean_name("<REASONING>"), "");
        assert_eq!(clean_name("다운받기"), "");
    }

    #[test]
    fn an_unknown_field_is_refused() {
        assert!(serde_json::from_str::<NameBody>(r#"{"prompt":"x"}"#).is_err());
        let ok: NameBody = serde_json::from_str(r#"{"world_id":"w"}"#).unwrap();
        assert_eq!(ok.world_id, "w");
    }
}
