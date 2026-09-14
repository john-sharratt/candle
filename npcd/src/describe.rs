//! Writing a character's description, in the voice of the world it belongs to.
//!
//! # Why the world is the system prompt
//!
//! A description is the character's identity section in every prompt it will
//! ever think under, and it has to *fit*. A quartermaster invented without the
//! setting is a quartermaster from nowhere: the console's create page offers a
//! world and a personality before it offers this field precisely because those
//! two are what the description has to be consistent with.
//!
//! So the world's own `setting` prose — three centuries after the Great War,
//! uploaded consciousness, the Four Houses — goes in the system turn, together
//! with the personality's anchor. The user turn asks for one person. That split
//! is the same one [`crate::lifegen::narrate`] makes and for the same reason:
//! background the model writes *from* belongs in the system turn, and the thing
//! it must produce belongs in the turn it answers.
//!
//! # Why a throwaway conversation
//!
//! The resident model spends its day *being* characters — it perceives, decides
//! and emits acts under a character's identity. Asked to describe one from inside
//! that frame it would answer in character rather than write a profile. So the
//! description is written in a conversation of its own ([`crate::prose`]): the
//! voice and the world above, one request below, and nothing of any character's
//! identity, acts or history in between. It rides the engine's waves beside the
//! cast, and nobody in the world stops thinking for it.

use std::sync::Arc;

use axum::extract::State;
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use axum::Json;
use candle_conversation::guest::{resolve_seed, Seeded};
use serde::Deserialize;
use serde_json::{json, Value};
use tokio::sync::mpsc::unbounded_channel;

use crate::api::Authored;
use crate::ndjson;
use crate::prose;

/// The voice, and the shape of the answer.
///
/// Explicit about length and form because the field it fills is a textarea an
/// author then edits: a description that arrives as a bulleted sheet has to be
/// rewritten before it can be used, and one that arrives as three pages will be
/// truncated by whoever reads it.
const VOICE: &str = "You write character profiles for a fiction engine. Given a setting and a \
     personality, invent ONE person who belongs in that world and write them as a short \
     present-day description: who they are, what they look like, what they do, one habit or \
     scar that makes them specific. Two or three sentences, plain prose, no headings, no lists, \
     no preamble. Write only the description.";

/// What to call a personality in a prompt.
///
/// **A personality document has no `name` field.** It has `id` and `category`
/// and the authored prose; the console derives its label from the slug. So
/// `p["name"]` was absent for every real personality in the mind and this read
/// "THE PERSONALITY THEY ARE CAST AS / this personality" every time — while the
/// task's "who could be cast as …" clause, built from the same missing field,
/// collapsed to an empty string and never mentioned the personality at all.
///
/// It survived because the test fixture carries a `name`, which no document in
/// the corpus does. Preferring the field and falling back to the id means an
/// authored `name:` still wins if anybody adds one.
pub fn personality_label(p: &serde_json::Value) -> String {
    if let Some(n) = p["name"].as_str().map(str::trim).filter(|s| !s.is_empty()) {
        return n.to_string();
    }
    let id = p["personality_id"]
        .as_str()
        .map(str::trim)
        .filter(|s| !s.is_empty());
    match id {
        // `the-dapper-lich` → `The Dapper Lich`, the same reading the console
        // shows in its selector, so the prompt and the picker agree.
        Some(id) => id
            .split('-')
            .filter(|w| !w.is_empty())
            .map(|w| {
                let mut c = w.chars();
                match c.next() {
                    Some(f) => f.to_uppercase().collect::<String>() + c.as_str(),
                    None => String::new(),
                }
            })
            .collect::<Vec<_>>()
            .join(" "),
        None => "this personality".to_string(),
    }
}

/// How much of a personality's anchor reaches a naming or describing prompt.
///
/// **A character budget, not a line count.** This was `lines().take(6)`, which
/// measures the author's wrapping rather than the text: six lines of Keeper is
/// most of an anchor, six lines of a personality written in long paragraphs is
/// the whole thing, and six lines of one written in short ones is a fragment
/// that stops mid-thought. Characters are at least proportional to what the
/// model actually reads.
///
/// Twelve hundred takes every anchor in the corpus whole except the longest —
/// Keeper's is 848, the ordinary ones are under 350 — while still bounding the
/// outlier at roughly 300 tokens.
///
/// The cap survives because an anchor is *unbounded authored text*: it is a
/// whole system prompt with sections and examples, and pasting an arbitrarily
/// long one makes the description a summary of the anchor rather than a person
/// who fits it.
const ANCHOR_BUDGET_CHARS: usize = 1_200;

/// As much of an anchor as the budget allows, cut on a line boundary.
///
/// Never mid-line: a prompt that stops in the middle of a sentence reads as a
/// transmission error, and a model completing the thought is a model writing
/// from something the author did not say.
pub fn anchor_extract(anchor: &str) -> String {
    let anchor = anchor.trim();
    if anchor.len() <= ANCHOR_BUDGET_CHARS {
        return anchor.to_string();
    }
    let mut out = String::new();
    for line in anchor.lines() {
        if out.len() + line.len() + 1 > ANCHOR_BUDGET_CHARS {
            break;
        }
        if !out.is_empty() {
            out.push('\n');
        }
        out.push_str(line);
    }
    // A single line longer than the whole budget would leave nothing at all,
    // and no anchor is better than a short one.
    if out.is_empty() {
        out = anchor.chars().take(ANCHOR_BUDGET_CHARS).collect();
    }
    out
}

/// Tokens one description gets.
///
/// Two or three sentences is well under this; the ceiling is here so a model
/// that ignores the instruction cannot hold the card for a page.
const MAX_TOKENS: u32 = 220;

/// Warm and specific rather than clinical — a description is read by an author
/// choosing whether to keep it.
const TEMPERATURE: f32 = 0.9;

/// Initials the invented name may take.
///
/// **Why the prompt names a letter at all.** A name is the *first* thing
/// decoded, with no generated context in front of it, so its distribution is
/// whatever the fixed prompt implies — and that distribution is sharply peaked.
/// Ten unseeded drafts of one world came back as Eira, Eira, Eira, Aria, Aria,
/// Ella, Talia, Levi, Zara, Lei: a different seed each time, drawing from the
/// same handful of high-probability first tokens. The seed was doing its job.
/// Sampling harder cannot fix it either — raising temperature to flatten that
/// one token degrades every token after it.
///
/// So the variety goes in the prompt, where it changes the distribution instead
/// of re-rolling against it. Naming an initial moves nearly all of the mass:
/// the model still writes the name it wants, in the world's own register, but
/// out of a different corner of its prior each time.
///
/// Q and X are left out. Both are so thin in most name priors that the model
/// either ignores the instruction or reaches for the same two or three names,
/// which reintroduces exactly the clustering this is here to break.
pub(crate) const INITIALS: [char; 24] = [
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'R', 'S', 'T',
    'U', 'V', 'W', 'Y', 'Z',
];

/// Walks of life, as a second independent axis.
///
/// The initial decorrelates the name; this decorrelates the person. Deliberately
/// abstract — the world's own setting prose is what makes "someone who keeps
/// records" a tower archivist or a ship's purser, so one list serves every
/// world without knowing anything about any of them.
const STATIONS: [&str; 16] = [
    "someone who repairs what other people break",
    "someone who keeps records nobody reads until they matter",
    "someone who moves goods from where they are to where they are wanted",
    "someone who trains the ones coming up behind them",
    "someone who has held one post far longer than anyone expected",
    "someone recently arrived, still learning what is normal here",
    "someone who deals with the dead, or with what is left behind",
    "someone who feeds people",
    "someone who carries messages between parties who do not speak",
    "someone who settles disputes without any authority to do so",
    "someone who builds or maintains the physical fabric of the place",
    "someone who watches a boundary",
    "someone who was demoted, and stayed",
    "someone who makes or performs something the place values",
    "someone who knows where everything is kept",
    "someone whose work is dangerous and routine at once",
];

#[derive(Debug, Default, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DescribeBody {
    /// The world the character will live in. Its `setting` is the whole of the
    /// context, so a request without one is refused rather than answered from
    /// nowhere.
    #[serde(default)]
    pub world_id: String,
    /// The personality it will be cast as, if one is chosen yet. Optional: the
    /// create page lets an author generate a description before settling on a
    /// personality, and a profile that fits the world is still useful then.
    #[serde(default)]
    pub personality_id: String,
    /// The character's name, when there is one.
    ///
    /// The create form generates a name first and passes it here, so the
    /// description is written *about that person* rather than inventing a second
    /// one — the two used to disagree, because a description that names its own
    /// subject had no idea what the form's name field said. Empty is fine and
    /// means "invent one".
    #[serde(default)]
    pub name: String,
    /// Pin the draw, so a description an author liked can be reproduced.
    #[serde(default)]
    pub seed: Option<u64>,
}

/// What the model is told, and what it is asked.
///
/// Separate from the route and public to the crate because it is the whole of
/// the decision here — which context a description is written against — and it
/// is testable without a card, a checkpoint or a daemon.
pub struct Brief {
    pub context: String,
    pub task: String,
}

/// Build the brief from a world record and, if there is one, a personality.
///
/// Both arrive as the wire JSON the registries already serve, so this reads the
/// same fields the console shows rather than a second projection of them.
pub fn brief_for(
    world: &serde_json::Value,
    personality: Option<&serde_json::Value>,
    seed: u64,
    named: &str,
) -> Brief {
    let mut context = String::new();
    let name = world["name"].as_str().unwrap_or("this world");
    context.push_str(&format!("THE WORLD\n{name}\n"));
    if let Some(setting) = world["setting"].as_str().filter(|s| !s.trim().is_empty()) {
        context.push_str(&format!("\n{}\n", setting.trim()));
    }

    // The world's `excludes` used to be restated here as "do not write anything
    // sexual, intimate". It is gone, and it was doing less than it looked like.
    // `excludes` gates which *section categories* of the mind a world admits
    // (see [`crate::collections`]) — it is a retrieval filter, not a style
    // guide, and borrowing it as one put a list of registers in front of a model
    // that had not been going to write them. Naming a thing you do not want is
    // how it ends up in the frame. The register a description should be in is
    // already carried by the setting prose and by `VOICE`.

    if let Some(p) = personality {
        let pname = personality_label(p);
        context.push_str(&format!("\nTHE PERSONALITY THEY ARE CAST AS\n{pname}\n"));
        if let Some(anchor) = p["anchor"].as_str().filter(|s| !s.trim().is_empty()) {
            context.push_str(&format!("{}\n", anchor_extract(anchor)));
        }
    }

    // Drawn from the seed, so the same seed rebuilds the same brief as well as
    // replaying the same sampling — reproducing half of what made a draft would
    // not reproduce the draft. `Seeded` splits the value so the two axes are
    // independent; taken straight off the seed they would move together, and
    // the second would be doing nothing.
    let mut spice = Seeded::new(seed);
    let initial = spice.pick(&INITIALS).copied().unwrap_or('A');
    let station = spice.pick(&STATIONS).copied().unwrap_or(STATIONS[0]);

    let cast = match personality {
        Some(p) => format!(" who could be cast as {}", personality_label(p)),
        None => String::new(),
    };

    /* **The station is what to do when there is no personality, not a second
     * opinion alongside one.**
     *
     * `STATIONS` exists to decorrelate draws: without it the model returns the
     * same handful of people whatever the seed. But it was stated in the *task*
     * — "Make them someone who feeds people" — while the personality sat in the
     * system turn as background, and a concrete imperative in the turn being
     * answered beats background every time. An anxious dragon drew "someone who
     * builds or maintains the physical fabric" and came back a broad-shouldered
     * carpenter; a dapper lich drew "someone who feeds people" and came back an
     * elderly cook. The station is picked from the seed alone and has no idea a
     * personality was chosen, so it was overwriting one at random.
     *
     * A chosen personality *is* the decorrelation, and a much better one. So the
     * station is used only when there is none, and where a personality exists
     * the task names it and lets the anchor decide the walk of life. The seeded
     * initial still separates the names either way.
     */
    let walk = match personality {
        Some(_) => String::new(),
        None => format!(" Make them {station}."),
    };

    // **Named or not are different jobs.** With a name the description is
    // *about somebody* — the create form names the character first (see
    // [`crate::namegen`]) so an author who likes the name and not the prose can
    // rewrite one without losing the other. The initial is dropped there: it
    // exists only to decorrelate a name this prompt would otherwise invent, and
    // naming a letter that contradicts the name already given is worse than
    // saying nothing.
    let named = named.trim();
    let task = if named.is_empty() {
        format!(
            "Invent one person who lives in {name}{cast}.{walk} Their name begins with the letter \
             {initial}. Write their description."
        )
    } else {
        format!(
            "Write the description of {named}, who lives in {name}{cast}.{walk} Use the name \
             {named} and do not rename them."
        )
    };

    Brief { context, task }
}

/// `POST /v1/generate/description`
///
/// The console's create page has called this since it was written; the daemon
/// answered 404 and the page fell back to "generation unavailable — write one
/// yourself". It now runs on the prose guest.
///
/// Blocks for the length of the drain, on a blocking thread — see
/// [`crate::guest_routes::run_guest`].
pub async fn post_describe(
    State(s): State<Arc<Authored>>,
    Json(body): Json<DescribeBody>,
) -> Response {
    if body.world_id.trim().is_empty() {
        return fail(
            StatusCode::BAD_REQUEST,
            "no_world",
            "a description is written against a world's setting — name one",
        );
    }
    // Cloned out of the registries rather than held: the guest drain below
    // takes seconds to a minute, and a read guard on the world registry held
    // across it would block every listing the console makes meanwhile.
    let world = match s.worlds.read().await.get(&body.world_id) {
        Some(w) => w.body.clone(),
        None => return fail(StatusCode::NOT_FOUND, "world_not_found", &body.world_id),
    };
    // A personality that is named must exist; one that is not is fine. The
    // create page lets an author write a description before settling on one.
    let personality = if body.personality_id.trim().is_empty() {
        None
    } else {
        match s.personalities.read().await.get(&body.personality_id) {
            // With its id, because a personality document carries no `name` and
            // the id is what [`personality_label`] falls back to. Same shape the
            // listings hand the console.
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

    // **Resolved here, not left to the guest.** The seed decides the brief as
    // well as the sampling — see [`INITIALS`] — so this route has to know it
    // before it can build the prompt. Passing it on explicitly is what keeps the
    // two halves in agreement: a guest drawing its own would sample a brief this
    // route wrote against a different one, and a reported seed would then
    // reproduce neither.
    let seed = resolve_seed(body.seed);
    let brief = brief_for(&world, personality.as_ref(), seed, &body.name);
    let request = prose::Request {
        system: format!("{VOICE}\n\n{}", brief.context),
        prompt: brief.task,
        max_tokens: MAX_TOKENS,
        temperature: Some(TEMPERATURE),
        seed: Some(seed),
        // Free prose — there is no fixed set of answers to constrain it to.
        choices: None,
    };

    match prose::run(&s, request).await {
        Ok(prose::Answer { text, tokens, seed }) => {
            let description = text.trim();
            if description.is_empty() {
                return fail(
                    StatusCode::BAD_GATEWAY,
                    "empty_draft",
                    "the model produced no description",
                );
            }
            Json(json!({
                "description": description,
                "tokens": tokens,
                // The seed that produced it, so a draft an author liked can be
                // asked for again. Without it every good description is a
                // one-off — and the guest used to answer every unseeded request
                // from the *same* constant, so "Regenerate" returned the same
                // paragraph every time.
                "seed": seed,
                "world_id": body.world_id,
                "personality_id": body.personality_id,
            }))
            .into_response()
        }
        Err(e) => prose::refusal(&e),
    }
}

/// `POST /v1/generate/description/stream`
///
/// The same generation as [`post_describe`], delivered as it happens: one
/// `token` line per decoded fragment, and one terminal `done` carrying the whole
/// description and its seed. See [`crate::ndjson`] for the wire format and why
/// it is not SSE.
///
/// **Why this exists next to the non-streaming route rather than replacing it.**
/// They are not the same operation to a caller. A page rendering a description
/// for a person to read wants it as it arrives, because forty tokens a second is
/// about reading speed and the alternative is eight seconds of spinner. Anything
/// generating a description as a *step* — a script seeding a cast, a test — wants
/// one request and one answer, and would have to reassemble a stream to get it.
pub async fn post_describe_stream(
    State(s): State<Arc<Authored>>,
    Json(body): Json<DescribeBody>,
) -> Response {
    // Validated before the stream opens, so a bad ask is a status a caller can
    // branch on rather than an `error` line inside a 200.
    if body.world_id.trim().is_empty() {
        return ndjson::refuse(
            StatusCode::BAD_REQUEST,
            "no_world",
            "a description is written against a world's setting — name one",
        );
    }
    let world = match s.worlds.read().await.get(&body.world_id) {
        Some(w) => w.body.clone(),
        None => {
            return ndjson::refuse(StatusCode::NOT_FOUND, "world_not_found", &body.world_id);
        }
    };
    let personality = if body.personality_id.trim().is_empty() {
        None
    } else {
        match s.personalities.read().await.get(&body.personality_id) {
            // With its id, because a personality document carries no `name` and
            // the id is what [`personality_label`] falls back to. Same shape the
            // listings hand the console.
            Some(p) => Some(crate::api::with_id("personality_id", &p.id, &p.body)),
            None => {
                return ndjson::refuse(
                    StatusCode::NOT_FOUND,
                    "personality_not_found",
                    &body.personality_id,
                );
            }
        }
    };

    let seed = resolve_seed(body.seed);
    let brief = brief_for(&world, personality.as_ref(), seed, &body.name);
    let request = prose::Request {
        system: format!("{VOICE}\n\n{}", brief.context),
        prompt: brief.task,
        max_tokens: MAX_TOKENS,
        temperature: Some(TEMPERATURE),
        seed: Some(seed),
        // Free prose — there is no fixed set of answers to constrain it to.
        choices: None,
    };

    // Unbounded, because the fragments are pushed from the decode's blocking
    // thread, which must not wait on the HTTP client. The bound on how much can
    // accumulate is `MAX_TOKENS` lines, which is the real limit.
    let (tx, rx) = unbounded_channel::<Value>();
    let on_fragment = {
        let tx = tx.clone();
        // A send that fails means the reader hung up. The decode runs on to its
        // end regardless — it is one sequence in a wave, and stopping it early
        // would save nothing worth the bookkeeping.
        move |text: &str| {
            let _ = tx.send(json!({ "event": "token", "text": text }));
        }
    };

    let world_id = body.world_id.clone();
    let personality_id = body.personality_id.clone();
    tokio::spawn(async move {
        let line = match prose::run_streamed(&s, request, on_fragment).await {
            Ok(prose::Answer { text, tokens, seed }) => {
                let description = text.trim();
                if description.is_empty() {
                    ndjson::error_line("empty_draft", "the model produced no description", false)
                } else {
                    // The terminal line carries the whole description, not the
                    // last fragment: a consumer replaces what it streamed with
                    // this, which is what makes the preview's resynchronisation
                    // invisible.
                    json!({
                        "event": "done",
                        "description": description,
                        "tokens": tokens,
                        "seed": seed,
                        "world_id": world_id,
                        "personality_id": personality_id,
                    })
                }
            }
            Err(e) => ndjson::error_line(e.code(), &e.to_string(), e.retry()),
        };
        let _ = tx.send(line);
    });

    ndjson::stream(rx)
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
            "excludes": ["sexual", "intimate"],
        })
    }

    /// **Shaped like a document in the mind, which has no `name`.**
    ///
    /// The fixture used to carry one, and that is the only reason the missing
    /// field went unnoticed: every prompt in production read "this personality"
    /// while every test read "Commander".
    fn personality() -> serde_json::Value {
        json!({
            "personality_id": "commander",
            "anchor": "You are the Commander.\nPosition is read before people are.\nline3\nline4\n\
                       line5\nline6\nline7 SHOULD NOT APPEAR\nline8",
        })
    }

    /// **The setting is the point.** A description written without it is a
    /// person from nowhere, and it goes into the character's permanent identity
    /// section in every prompt it will ever think under.
    /// A fixed seed, where the test is about something other than the seed.
    const S: u64 = 4;

    #[test]
    fn the_worlds_setting_is_the_context() {
        let b = brief_for(&world(), None, S, "");
        assert!(b.context.contains("Battle Cities"));
        assert!(
            b.context
                .contains("uploaded\n                        consciousness")
                || b.context.contains("uploaded consciousness")
        );
        assert!(b.task.contains("Battle Cities"));
    }

    /// **A world's exclusions do not reach the model, deliberately.**
    ///
    /// `excludes` is a retrieval filter over the mind's section categories, not
    /// a style guide; restating it in the prompt put a list of registers in
    /// front of a model that was not going to write them, which is how a thing
    /// gets into the frame rather than out of it. The register a description
    /// belongs in comes from the setting prose and from `VOICE`.
    #[test]
    fn the_worlds_exclusions_are_not_restated_as_instructions() {
        let b = brief_for(&world(), None, S, "");
        assert!(!b.context.contains("EXCLUDES"), "{}", b.context);
        assert!(!b.context.contains("sexual"), "{}", b.context);
        assert!(!b.context.contains("intimate"), "{}", b.context);
    }

    /// The personality is context when there is one, and its absence is not an
    /// error: the create page lets an author describe a character before
    /// settling on one.
    #[test]
    fn a_personality_is_optional_context() {
        let with = brief_for(&world(), Some(&personality()), S, "");
        assert!(with.context.contains("Commander"));
        assert!(with.task.contains("Commander"));

        let without = brief_for(&world(), None, S, "");
        assert!(!without.context.contains("Commander"));
        assert!(without.task.contains("Invent one person"));
    }

    /// **A personality is named from its id, because it has no name.**
    ///
    /// The bug this pins made every prompt in the daemon say "this personality"
    /// while the tests said "Commander" — the fixture carried a field no
    /// document in the mind has.
    #[test]
    fn a_personality_is_named_from_its_id_when_it_has_no_name_field() {
        assert_eq!(
            personality_label(&json!({ "personality_id": "the-dapper-lich" })),
            "The Dapper Lich"
        );
        // An authored `name` still wins, for a document that grows one.
        assert_eq!(
            personality_label(&json!({ "personality_id": "keeper", "name": "The Keeper" })),
            "The Keeper"
        );
        // Neither is not a crash, and not an empty heading.
        assert_eq!(personality_label(&json!({})), "this personality");

        let b = brief_for(&world(), Some(&personality()), S, "");
        assert!(b.context.contains("Commander"), "{}", b.context);
        assert!(
            !b.context.contains("this personality"),
            "the heading fell back to the placeholder: {}",
            b.context
        );
    }

    /// **A station is for when there is no personality, not a rival to one.**
    ///
    /// Stated in the task — the turn the model answers — "Make them someone who
    /// feeds people" beat an anchor sitting in the system turn, and the station
    /// is drawn from the seed with no knowledge of which personality was picked.
    /// That is how an anxious dragon came back a carpenter.
    #[test]
    fn a_station_is_only_dictated_when_no_personality_was_chosen() {
        // With a personality: no station, and the personality is named instead.
        for seed in 0..40u64 {
            let b = brief_for(&world(), Some(&personality()), seed, "");
            assert!(
                !b.task.contains("Make them"),
                "a station overrode the personality: {}",
                b.task
            );
            assert!(b.task.contains("cast as Commander"), "{}", b.task);
        }

        // Without one, the station is the only thing decorrelating the person,
        // so it must still be there.
        let plain = brief_for(&world(), None, S, "");
        assert!(plain.task.contains("Make them"), "{}", plain.task);
    }

    /// **An ordinary anchor reaches the model whole.**
    ///
    /// The budget replaced a six-line trim that cut most real anchors short —
    /// every anchor in the corpus but one is under 350 characters, so the
    /// interesting case is that they now arrive complete.
    #[test]
    fn an_ordinary_anchor_is_not_truncated() {
        let b = brief_for(&world(), Some(&personality()), S, "");
        assert!(b.context.contains("Position is read before people are."));
        assert!(
            b.context.contains("SHOULD NOT APPEAR"),
            "a short anchor was cut: {}",
            b.context
        );
    }

    /// **A long one is still bounded, and cut on a line.**
    ///
    /// An anchor is unbounded authored text, and a cut mid-sentence reads as a
    /// transmission error the model then completes from something the author
    /// never wrote.
    #[test]
    fn a_long_anchor_is_bounded_and_never_cut_mid_line() {
        let long: String = (0..400)
            .map(|i| format!("line {i} of a very long anchor indeed"))
            .collect::<Vec<_>>()
            .join("\n");
        let out = anchor_extract(&long);
        assert!(out.len() <= ANCHOR_BUDGET_CHARS, "{} chars", out.len());
        assert!(out.starts_with("line 0 of"));
        // Every line kept is a whole line of the original.
        for line in out.lines() {
            assert!(long.lines().any(|l| l == line), "cut mid-line: {line:?}");
        }
    }

    /// A single line past the whole budget still yields something rather than
    /// nothing — an anchor written as one long paragraph is not an error.
    #[test]
    fn one_enormous_line_still_briefs() {
        let one = "x".repeat(ANCHOR_BUDGET_CHARS * 3);
        let out = anchor_extract(&one);
        assert!(!out.is_empty());
        assert!(out.chars().count() <= ANCHOR_BUDGET_CHARS);
    }

    /// Context and task are separate turns, for the reason `lifegen::narrate`
    /// found the hard way: given both in the turn it answers, the model
    /// continues the context instead — headings and all.
    #[test]
    fn the_context_and_the_task_are_separate() {
        let b = brief_for(&world(), Some(&personality()), S, "");
        assert!(b.context.contains("THE WORLD"));
        assert!(!b.task.contains("THE WORLD"));
        assert!(b.task.starts_with("Invent one person"));
    }

    /// The request this route builds must be one the engine accepts — assembled
    /// exactly as the route assembles it.
    #[test]
    fn the_request_it_builds_is_servable() {
        let b = brief_for(&world(), None, S, "");
        let r = prose::Request {
            system: format!("{VOICE}\n\n{}", b.context),
            prompt: b.task,
            max_tokens: MAX_TOKENS,
            temperature: Some(TEMPERATURE),
            seed: None,
            choices: None,
        };
        assert!(r.check().is_ok(), "{:?}", r.check());
    }

    /// A world with no `setting` still produces a usable brief — the field is
    /// optional in the schema, and refusing here would break every world that
    /// has not written one.
    #[test]
    fn a_world_without_a_setting_still_briefs() {
        let bare = json!({ "id": "sandbox", "name": "Sandbox" });
        let b = brief_for(&bare, None, S, "");
        assert!(b.context.contains("Sandbox"));
        assert!(b.task.contains("Sandbox"));
    }

    /// **The name-clustering fix.** A name is decoded first, with no generated
    /// context in front of it, so its distribution is whatever the fixed prompt
    /// implies — and it is sharply peaked. Ten unseeded drafts of one world came
    /// back as Eira, Eira, Eira, Aria, Aria, Ella, Talia, Levi, Zara, Lei: ten
    /// different seeds drawing from the same few first tokens. The variety has
    /// to change the prompt, not re-roll against it.
    #[test]
    fn the_brief_varies_with_the_seed() {
        let tasks: HashSet<String> = (0..200)
            .map(|seed| brief_for(&world(), None, seed, "").task)
            .collect();
        assert!(
            tasks.len() > 100,
            "200 seeds produced only {} distinct briefs — the model is being asked the same \
             question every time, and only the sampling differs",
            tasks.len()
        );
    }

    /// **A named subject is described, not re-invented.** The create form names
    /// the character first, and a description that invented its own name left
    /// the form's name field and the prose disagreeing about who this is.
    #[test]
    fn a_given_name_becomes_the_subject() {
        let b = brief_for(&world(), None, S, "Ursula Ved");
        assert!(b.task.contains("Ursula Ved"), "the name is not in the task");
        assert!(
            !b.task.contains("Invent one person"),
            "it was still asked to invent somebody"
        );
        assert!(
            b.task.contains("do not rename"),
            "nothing stops it renaming the subject"
        );
        // The initial exists only to decorrelate a name this prompt would
        // otherwise invent. Naming a letter that contradicts the given name is
        // worse than saying nothing.
        assert!(
            !b.task.contains("begins with the letter"),
            "an initial was named alongside a name already fixed"
        );
    }

    /// A name is the subject; the world is still the context.
    #[test]
    fn a_given_name_does_not_displace_the_world() {
        let b = brief_for(&world(), None, S, "Ursula Ved");
        assert!(b.context.contains("Battle Cities"));
        assert!(b.context.contains("THE WORLD"));
        assert!(
            !b.context.contains("Ursula Ved"),
            "the name leaked into the context"
        );
    }

    /// Whitespace is not a name: a form field the author cleared must fall back
    /// to inventing one rather than describing a person called "   ".
    #[test]
    fn a_blank_name_still_invents() {
        let b = brief_for(&world(), None, S, "   ");
        assert!(b.task.starts_with("Invent one person"));
    }

    /// Every initial in the table is actually reachable. A draw that could only
    /// ever land on a few of them would narrow the prior right back down.
    #[test]
    fn every_initial_is_reachable() {
        let mut seen: HashSet<char> = HashSet::new();
        for seed in 0..5_000u64 {
            let task = brief_for(&world(), None, seed, "").task;
            let at =
                task.find("begins with the letter ").unwrap() + "begins with the letter ".len();
            seen.insert(task[at..].chars().next().unwrap());
        }
        assert_eq!(seen.len(), INITIALS.len(), "some initials never come up");
    }

    /// **Reproducibility survives the fix.** The seed now decides the prompt as
    /// well as the sampling, so a seed that rebuilt a *different* brief would
    /// reproduce neither — which is worse than not reporting a seed at all.
    #[test]
    fn one_seed_rebuilds_one_brief() {
        for seed in [0u64, 1, 99, u64::MAX] {
            let a = brief_for(&world(), Some(&personality()), seed, "");
            let b = brief_for(&world(), Some(&personality()), seed, "");
            assert_eq!(a.task, b.task);
            assert_eq!(a.context, b.context);
        }
    }

    /// The varying part is the task, not the context: the world's setting and
    /// its exclusions are what the description must be consistent with, and a
    /// seed must not be able to reword either.
    #[test]
    fn the_seed_never_touches_the_world() {
        let base = brief_for(&world(), None, 0, "").context;
        for seed in 1..50u64 {
            assert_eq!(brief_for(&world(), None, seed, "").context, base);
        }
    }

    #[test]
    fn an_unknown_field_is_refused() {
        assert!(serde_json::from_str::<DescribeBody>(r#"{"prompt":"x"}"#).is_err());
        let ok: DescribeBody =
            serde_json::from_str(r#"{"world_id":"w","personality_id":"p"}"#).unwrap();
        assert_eq!(ok.world_id, "w");
    }
}
