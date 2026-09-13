//! Writing one stratum of a life in the narrator's voice.
//!
//! # How this differs from the ladder
//!
//! [`super::generate::run_phase`] runs the whole ladder on the **main engine**:
//! it primes one prefix and forks it once per node, so five hundred months
//! share one copy of the story's K/V and batch into the same waves. That is the
//! right shape for bulk, and nothing here replaces it.
//!
//! This is the other case — **one node, on demand, in a narrator's voice.** An
//! operator reading a month that came out flat wants that month rewritten, now,
//! and wants it to read like prose rather than like a character deciding what
//! to do. So a single node is written through [`crate::prose`]: a throwaway
//! conversation on the resident model under the narrator's voice, with no
//! character's identity or acts in front of it — one prompt, one answer, no
//! fan-out.
//!
//! The two write through the same door. A node this produces is recorded with
//! [`Content::generated`], not `edit` — it is a generation, so a later ladder
//! run may still supersede it, and only a human's edit is sticky.

use std::sync::Arc;

use axum::extract::{Path as UrlPath, State};
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use axum::Json;
use serde::Deserialize;
use serde_json::json;

use super::plan::{NodeId, Plan};
use crate::api::Authored;
use crate::prose;

/// The voice, and the shape of the answer.
///
/// Stated per request rather than left to the default voice because this is a
/// *life document*, not free narration: it has to stay inside what the strata
/// above it already committed to, and a narrator that invents a sibling in
/// March contradicts every other month of that year — which are written by
/// separate forks that cannot see each other.
const VOICE: &str = "You are writing one stratum of a character's life history. Write close, \
     concrete prose in the past tense. Stay strictly inside what you are told: do not invent \
     people, places or events that the context above does not contain, because the rest of this \
     life is being written separately and cannot see what you add. Do not summarise, do not \
     address the reader, and do not explain what you are doing. Write the document itself.";

/// Tokens one node gets.
///
/// A month is a page. Generous enough that a good answer is never cut off, and
/// bounded so one rewrite cannot run on for pages.
const MAX_TOKENS: u32 = 900;

#[derive(Debug, Default, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct NarrateBody {
    /// Rewrite a node a human edited. Off by default: an edit is sticky
    /// everywhere else in this module, and a "rewrite this" button that
    /// silently discarded an operator's own prose would be the one destructive
    /// thing in the life editor.
    #[serde(default)]
    pub force: bool,
    /// Extra direction for this one node — "make it colder", "she is ill
    /// throughout". Not stored: it shapes this draft and nothing else.
    #[serde(default)]
    pub note: String,
    #[serde(default)]
    pub seed: Option<u64>,
    #[serde(default)]
    pub temperature: Option<f32>,
}

/// The prompt a node becomes: what it is, and everything above it.
///
/// Public to the crate and separate from the route because it is the whole of
/// the decision here — which context a stratum is written against — and it is
/// testable without a card or a checkpoint.
///
/// **Ancestors only, never siblings.** A month is given its year and the arc;
/// it is not given the other eleven months, because in the ladder those are
/// written concurrently by forks that cannot see each other, and a prose route
/// that read them would produce a month that only makes sense against a
/// particular generation order.
/// A narration request, split the way the dialect wants it.
///
/// **The context is the system turn and the task is the user turn**, and that
/// split is the whole reason this is a struct. With both in one user message
/// the model read the context block as a document in progress and *continued*
/// it — the first real year came back opening with `THE CHARACTER / Hess — a
/// quartermaster, of Nanyang…`, the headings echoed verbatim, before any prose.
/// A system turn is background it writes *from*; a user turn is the thing it
/// answers.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Narration {
    /// What is already true: the character, the arc, the strata above this one.
    pub context: String,
    /// The one thing to write.
    pub task: String,
}

pub fn prompt_for(plan: &Plan, id: NodeId, note: &str, voice: &str) -> Narration {
    let mut s = String::new();
    let seed = &plan.seed;
    s.push_str(&format!(
        "THE CHARACTER\n{} — {}, of {}. Born {}, through {}.\n",
        seed.display, seed.role, seed.place, seed.born, seed.through
    ));
    // **How they sound, not only what happened to them.** These documents are
    // injected into the character's own prefix, so a memory in a neutral
    // literary register is a few thousand tokens teaching the model to answer
    // in somebody else's words. See `prompt::voice_of`.
    if !voice.trim().is_empty() {
        s.push_str(&format!("\nHOW THIS PERSON WRITES\n{}\n", voice.trim()));
    }
    if !plan.story.content.text.trim().is_empty() {
        s.push_str(&format!("\nTHE ARC\n{}\n", plan.story.content.text.trim()));
    }
    if !plan.story.cast.is_empty() {
        s.push_str("\nTHE PEOPLE IN THIS LIFE\n");
        for c in &plan.story.cast {
            s.push_str(&format!("- {} — {}\n", c.display, c.what));
        }
    }

    // The ancestors, coarsest first, so the model reads the life narrowing.
    match id {
        NodeId::Story => {}
        NodeId::Year { year } => {
            if let Some(b) = plan.story.beat(year) {
                s.push_str(&format!(
                    "\nWHAT THIS YEAR IS FOR\n{} — {}\n",
                    b.title, b.premise
                ));
            }
        }
        NodeId::Month { year, month } => {
            push_ancestor(
                &mut s,
                plan,
                NodeId::Year { year },
                &format!("THE YEAR {year}"),
            );
            let _ = month;
        }
        NodeId::Day { year, month, day } => {
            push_ancestor(
                &mut s,
                plan,
                NodeId::Year { year },
                &format!("THE YEAR {year}"),
            );
            push_ancestor(
                &mut s,
                plan,
                NodeId::Month { year, month },
                &format!("THE MONTH {year}-{month:02}"),
            );
            let _ = day;
        }
    }

    // **Each rung says how much time it covers, and says it against the rung
    // below.** "In full" was the whole instruction a year got, and a model reads
    // that as "write it richly" rather than "write all of it" — so a year of
    // Yen's life came back as one morning: woke before dawn, practised in the
    // courtyard, paused as the sun rose. Excellent prose, and a day.
    //
    // The day's task already said "One day, closely" and produced days
    // correctly, which is the tell: the rung that stated its span got its span.
    // The contrast is explicit now because a year and a day are the same
    // *instruction* otherwise, and the only thing separating them is a number
    // the model has no reason to read as a duration.
    let mut task = match id {
        NodeId::Story => "Write the shape of this whole life as continuous prose — the arc of \
                          the whole span, not any one part of it."
            .to_string(),
        NodeId::Year { year } => format!(
            "Write the year {year} of this life — the WHOLE of it, not a scene from it.{}\n\n\
             A year is a span. Move through it: what was underway when it began, what changed \
             across it, what recurred, where it had got to by the end. Some of it in summary, \
             the way a year is remembered rather than relived.\n\n\
             Do NOT write a single day, a single morning, or one continuous episode. A day is \
             what the days below this are for, and one written here takes the place of the year.",
            // **The beat, in the task and not only in the context.** It is
            // already stated above as "WHAT THIS YEAR IS FOR", and the draft
            // ignored it: asked for 2792 — "you fight alongside Commander
            // Kaelor, who falls" — the model wrote a year of sparring in the
            // courtyard from the character sheet, with a man who had been dead
            // seven years by then.
            //
            // Third time this file has learned the same thing. "Prose only" and
            // the second person both had to move into the task for the same
            // reason: the context is long and vivid, the task is the last thing
            // read, and what the task does not name does not survive it.
            match plan.story.beat(year) {
                None => String::new(),
                Some(b) => format!(
                    "\n\nThis year is the one the outline calls \"{}\": {}. That is what the \
                     year is about — write it, rather than around it.",
                    b.title,
                    b.premise.trim_end_matches('.')
                ),
            }
        ),
        NodeId::Month { year, month } => format!(
            "Write the month {year}-{month:02} of this life — the whole month, not a day in \
             it.\n\n\
             Expand only the part of the year that falls inside this month, and move across the \
             weeks of it. A single episode belongs to a day, not here."
        ),
        NodeId::Day { year, month, day } => format!(
            "Write the day {year}-{month:02}-{day:02} — a day that became a memory. One day, \
             closely."
        ),
    };
    // **Prose only, said in the turn being answered.** The same instruction in
    // the system turn was not enough: given the context and the task together,
    // the model continued the context — headings and all — and the year came
    // back with `THE CHARACTER / Hess — a quartermaster…` pasted on the front.
    // The person is restated here for the same reason "prose only" is: `VOICE`
    // says it in the system turn and the drafts came back in third person
    // anyway — "Yen woke before dawn… she stretched" — because the context
    // immediately above the task is full of the character's name. The last
    // instruction read wins, so the last instruction says it.
    task.push_str(
        "\n\nWrite only the document's prose, in the SECOND person — \"You waited\", never the \
         character's name and never \"she\" or \"he\" as the subject. Do not repeat the headings \
         above, do not restate the character, and do not write a preamble.",
    );
    if !note.trim().is_empty() {
        // Last, so it is the most recent instruction the model read — and
        // labelled, so it cannot be mistaken for part of the life's canon.
        task.push_str(&format!(
            "\n\nDIRECTION FOR THIS DRAFT ONLY\n{}",
            note.trim()
        ));
    }
    Narration {
        context: s.trim_end().to_string(),
        task,
    }
}

/// A written ancestor's prose, if it has any.
fn push_ancestor(s: &mut String, plan: &Plan, id: NodeId, heading: &str) {
    let Some(c) = plan.content(id) else { return };
    if c.text.trim().is_empty() {
        return;
    }
    s.push_str(&format!("\n{heading}\n{}\n", c.text.trim()));
}

/// `POST /v1/life/:who/node/:key/narrate`
///
/// Generates one node through [`crate::prose`] and writes it into the plan and
/// onto disk, answering with the prose so the console can show it without a
/// second read.
/// **The body is required, and `{}` is the ordinary one.**
///
/// It was `Option<Json<NarrateBody>>`, so a bodyless POST would work — and that
/// quietly defeated `deny_unknown_fields`: axum's `Option` extractor turns a
/// *parse failure* into `None`, so a caller sending `{"prompt": "..."}` got the
/// defaults and a 200. Asking for something this route deliberately does not
/// offer has to be refused, not silently reinterpreted as asking for nothing.
pub async fn post_narrate(
    State(s): State<Arc<Authored>>,
    UrlPath((who, key)): UrlPath<(String, String)>,
    Json(body): Json<NarrateBody>,
) -> Response {
    let mind = match super::routes::resolve(&s, &who).await {
        Ok(m) => m,
        Err(r) => return *r,
    };
    let mut plan = match super::routes::open(&mind, &who) {
        Ok(p) => p,
        Err(r) => return *r,
    };
    let Some(id) = super::routes::node_id(&key) else {
        return fail(
            StatusCode::BAD_REQUEST,
            "bad_node",
            "a node is `story`, `YYYY`, `YYYY-MM` or `YYYY-MM-DD`",
        );
    };
    let Some(existing) = plan.content(id) else {
        return fail(StatusCode::NOT_FOUND, "no_such_node", &key);
    };
    if existing.edited && !body.force {
        return fail(
            StatusCode::CONFLICT,
            "node_edited",
            "somebody wrote this stratum by hand, and an edit is sticky — send `force: true` to \
             write over it",
        );
    }

    // Built before the decode and from a plan this route owns, so nothing else
    // can move the context underneath the prompt while it runs.
    let narration = prompt_for(
        &plan,
        id,
        &body.note,
        &crate::lifegen::routes::voice_for(&s, &who).await,
    );
    let request = prose::Request {
        // The voice and everything already true go in the system turn; the one
        // thing to write goes in the user turn. See [`Narration`] for what
        // putting both in the user turn produced.
        system: format!("{VOICE}\n\n{}", narration.context),
        prompt: narration.task,
        max_tokens: MAX_TOKENS,
        temperature: body.temperature,
        seed: body.seed,
        // Narration is prose, not a decision — nothing to constrain it to.
        choices: None,
    };

    let (text, tokens, seed) = match prose::run(&s, request).await {
        Ok(a) => (a.text, a.tokens, a.seed),
        Err(e) => return prose::refusal(&e),
    };

    let trimmed = text.trim();
    if trimmed.is_empty() {
        // **Not written.** A node that generated nothing must keep whatever it
        // had: `document::sync` drops a file whose node no longer reads as
        // generated, so writing an empty answer would delete a good document to
        // record a failed draft.
        return fail(
            StatusCode::BAD_GATEWAY,
            "empty_draft",
            "the narrator produced no prose, so the stratum was left as it was",
        );
    }

    // `generated`, not `edit`: this is a generation, so a later ladder run may
    // supersede it. Only a human's edit is sticky.
    let title = existing.title.clone();
    let title = if title.trim().is_empty() {
        default_title(id)
    } else {
        title
    };
    let Some(c) = plan.content_mut(id) else {
        return fail(StatusCode::NOT_FOUND, "no_such_node", &key);
    };
    c.generated(title, trimmed.to_string());
    let stale = plan.mark_stale_below(id);

    match super::routes::commit(&mind, &plan) {
        Ok(written) => Json(json!({
            "node": key,
            "text": trimmed,
            "tokens": tokens,
            // The draw that produced this stratum, so a narration an author
            // liked can be asked for again.
            "seed": seed,
            "stale_below": stale,
            "written": written,
        }))
        .into_response(),
        Err(r) => *r,
    }
}

/// What a node is called when it has never had a title.
///
/// The title becomes the document's filename, so an empty one is not cosmetic —
/// `document::sync` would write a file named after nothing.
fn default_title(id: NodeId) -> String {
    match id {
        NodeId::Story => "The Life".into(),
        NodeId::Year { year } => format!("{year}"),
        NodeId::Month { year, month } => format!("{year}-{month:02}"),
        NodeId::Day { year, month, day } => format!("{year}-{month:02}-{day:02}"),
    }
}

fn fail(status: StatusCode, error: &str, detail: &str) -> Response {
    (status, Json(json!({ "error": error, "detail": detail }))).into_response()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lifegen::seed::{check, Cadence, Seed};

    fn plan() -> Plan {
        let mut p = Plan::new(
            &check(&Seed {
                who: "cindy".into(),
                display: "Cindy Tan".into(),
                born: "1998-09-14".into(),
                through: "1999-03-02".into(),
                place: "Nanyang".into(),
                role: "a clerk".into(),
                cadence: Cadence::Even,
                facts: Vec::new(),
                world: Vec::new(),
                cast: Vec::new(),
                eras: Vec::new(),
            })
            .unwrap(),
        );
        p.story
            .content
            .generated("Arc".into(), "She grew up beside the river.".into());
        p.content_mut(NodeId::Year { year: 1998 })
            .unwrap()
            .generated("Nineteen".into(), "The year the granary burned.".into());
        p
    }

    /// The character and the arc are always present: a stratum written without
    /// them is a paragraph about nobody.
    #[test]
    fn every_prompt_carries_the_character_and_the_arc() {
        for id in [
            NodeId::Story,
            NodeId::Year { year: 1998 },
            NodeId::Month {
                year: 1998,
                month: 10,
            },
        ] {
            let p = prompt_for(&plan(), id, "", "");
            assert!(p.context.contains("Cindy Tan"), "{id:?} lost the character");
            assert!(
                p.context.contains("She grew up beside the river."),
                "{id:?} lost the arc"
            );
        }
    }

    /// **What is already true goes in the context; what to write goes in the
    /// task.** With both in one user turn the model continued the context
    /// rather than answering it — the first real year came back with
    /// `THE CHARACTER / Hess — a quartermaster…` echoed verbatim before any
    /// prose. The split is what a system turn is for.
    #[test]
    fn the_context_and_the_task_are_separate_turns() {
        let p = prompt_for(&plan(), NodeId::Year { year: 1998 }, "", "");
        assert!(
            p.context.contains("THE CHARACTER") && !p.task.contains("THE CHARACTER"),
            "the headings leaked into the turn the model answers"
        );
        assert!(
            p.task.contains("Write the year 1998"),
            "the task is not in the task"
        );
        assert!(
            !p.context.contains("Write the year 1998"),
            "the instruction leaked into the background"
        );
    }

    /// The task says "prose only" in the turn being answered, not only in the
    /// system turn — the system turn alone did not stop the echo.
    #[test]
    fn the_task_forbids_a_preamble_where_the_model_will_read_it() {
        let p = prompt_for(&plan(), NodeId::Story, "", "");
        assert!(p.task.contains("Do not repeat the headings"));
        assert!(p.task.contains("do not write a preamble"));
    }

    /// **Ancestors, never siblings.** In the ladder the other months of a year
    /// are written concurrently by forks that cannot see each other, so a month
    /// written against its siblings only makes sense against one particular
    /// generation order — and would contradict them the moment any is redone.
    #[test]
    fn a_month_is_written_against_its_year_and_not_its_siblings() {
        let mut p = plan();
        p.content_mut(NodeId::Month {
            year: 1998,
            month: 11,
        })
        .unwrap()
        .generated("Nov".into(), "SIBLING PROSE".into());

        let prompt = prompt_for(
            &p,
            NodeId::Month {
                year: 1998,
                month: 10,
            },
            "",
            "",
        );
        assert!(
            prompt.context.contains("The year the granary burned."),
            "lost its year"
        );
        assert!(
            !prompt.context.contains("SIBLING PROSE"),
            "a sibling month reached the prompt"
        );
    }

    /// A day gets both its month and its year — the month for what happened
    /// around it, the year for why it mattered.
    #[test]
    fn a_day_is_written_against_its_month_and_its_year() {
        let mut p = plan();
        p.ensure_day(1998, 10, 4).unwrap();
        p.content_mut(NodeId::Month {
            year: 1998,
            month: 10,
        })
        .unwrap()
        .generated("Oct".into(), "OCTOBER PROSE".into());

        let prompt = prompt_for(
            &p,
            NodeId::Day {
                year: 1998,
                month: 10,
                day: 4,
            },
            "",
            "",
        );
        assert!(prompt.context.contains("OCTOBER PROSE"));
        assert!(prompt.context.contains("The year the granary burned."));
    }

    /// An ancestor with no prose yet is skipped rather than pasted in as an
    /// empty heading — a heading with nothing under it reads to the model as a
    /// year in which nothing happened.
    #[test]
    fn an_unwritten_ancestor_is_left_out_entirely() {
        let mut p = plan();
        p.content_mut(NodeId::Year { year: 1998 })
            .unwrap()
            .generated(String::new(), String::new());
        let prompt = prompt_for(
            &p,
            NodeId::Month {
                year: 1998,
                month: 10,
            },
            "",
            "",
        );
        assert!(
            !prompt.context.contains("THE YEAR 1998"),
            "an empty year was pasted in"
        );
    }

    /// **The per-draft direction is labelled and last.** Unlabelled, a note
    /// like "she is ill throughout" reads as part of the life's canon and the
    /// model writes it into the record as established fact.
    #[test]
    fn a_direction_is_labelled_as_this_draft_only_and_comes_last() {
        let p = prompt_for(&plan(), NodeId::Year { year: 1998 }, "make it colder", "");
        let at = p.task.find("make it colder").expect("the note is missing");
        assert!(p.task[..at].contains("DIRECTION FOR THIS DRAFT ONLY"));
        assert!(
            at > p.task.find("Write the year").unwrap(),
            "the note came before the task it modifies"
        );
        assert!(
            !p.context.contains("make it colder"),
            "a per-draft direction reached the background, where it reads as canon"
        );
    }

    /// No note is no section — an empty heading is one more thing for the model
    /// to interpret.
    #[test]
    fn no_direction_leaves_no_heading() {
        let p = prompt_for(&plan(), NodeId::Story, "   ", "");
        assert!(!p.task.contains("DIRECTION"));
        assert!(!p.context.contains("DIRECTION"));
    }

    /// The request this route builds must be one the engine accepts — assembled
    /// exactly as the route assembles it, or the check is of something else.
    #[test]
    fn the_request_it_builds_is_servable() {
        let n = prompt_for(&plan(), NodeId::Year { year: 1998 }, "", "");
        let r = prose::Request {
            system: format!("{VOICE}\n\n{}", n.context),
            prompt: n.task,
            max_tokens: MAX_TOKENS,
            temperature: None,
            seed: None,
            choices: None,
        };
        assert!(r.check().is_ok(), "{:?}", r.check());
    }

    /// A title becomes a filename, so a node that never had one cannot be
    /// written with an empty one.
    #[test]
    fn a_node_with_no_title_gets_one_that_can_be_a_filename() {
        assert_eq!(default_title(NodeId::Year { year: 1998 }), "1998");
        assert_eq!(
            default_title(NodeId::Month {
                year: 1998,
                month: 9
            }),
            "1998-09"
        );
        assert!(!default_title(NodeId::Story).is_empty());
    }

    #[test]
    fn the_body_is_optional_and_defaults_to_not_forcing() {
        let b: NarrateBody = serde_json::from_str("{}").unwrap();
        assert!(!b.force);
        assert!(b.note.is_empty());
        assert!(serde_json::from_str::<NarrateBody>(r#"{"prompt":"x"}"#).is_err());
    }
}
