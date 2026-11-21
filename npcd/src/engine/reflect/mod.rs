//! One character stopped to think, and the dream brief that comes out of it,
//! produced in a throwaway conversation that is never kept.
//!
//! A reflection asks two questions in one conversation, and the first answer goes
//! back to the character before any dream is asked for. The brief is then checked,
//! retried once if it is malformed, and reviewed by repair passes that revise the
//! same dream rather than writing a new one. A pass that does not parse, or that
//! comes out worse, is discarded. The conversation is tombstoned however the run
//! ends, so nothing it produced reaches a later gather.
//!
//! - `answer`: reads the answer out of its call and builds the grammar and exchange.
//! - `brief`: the structural checks that decide whether a dream brief is usable.
//! - `reflection`: the result type and the check on the line that crosses back.
//! - `steer`: the domain rotation and the axis sample that steer the assumption.
//!
//! - `tests_answer`, `tests_brief`, `tests_reflection`: tests for the answer
//!   reader, the brief checks and the reflection line.

use std::sync::Arc;

use candle_conversation::stencil::{
    compile_think_tree, compile_tool_call_tree, Param as CallParam, ParamType, StencilTree,
    ThinkMode, ThinkSteerEnvelope, ToolCallEnvelope, ToolSpec, TurnGrammar,
};
use candle_conversation::throwaway::Throwaway;
use candle_conversation::{ConversationEngine, Sequence, SequenceConfig, TurnOptions};
use journal_core::json_repair::escape_control_in_strings;
use serde::Serialize;

use crate::engine::dreams;
use crate::engine::identity;
use crate::engine::mind::Projected;
use crate::engine::prompt::{self, Persona, Stance};
use crate::engine::schema::{ReflectionTurns, AXES_SLOT, DOMAIN_SLOT};
use crate::engine::tools::{self, Availability, Mode, Param, Plane, Tool};

/// How many turns of its own the reflection carries.
///
/// **Sized for the longest run, not the shortest.** The reflection, the loosed
/// turn, the dream and four repair passes is seven — but every one of those can
/// be refused up to [`MAX_REFUSALS`] times, so the worst case is around twenty
/// and a typical run is ten to thirteen.
///
/// It was 8, which a ten-turn run overran, and the symptom was subtle enough to
/// be worth writing down: the *axis* decayed. Held perfectly across five turns —
/// `The work is finished when it is finished` — it became `The work is` and then
/// `The` once the head of the conversation fell out of the window and the model
/// could no longer see what the field had been for. Nothing failed; the field
/// just quietly emptied from the front.
const CONTEXT_WINDOW_TURNS: usize = 24;

/// Runs reflections against one engine, under the turns the mind authors.
///
/// Carries no state of its own beyond those: a reflection is a whole
/// conversation's lifetime, and there is nothing to hold between two of them by
/// design.
pub struct Reflect<'t> {
    engine: Arc<ConversationEngine>,
    base_config: SequenceConfig,
    /// **Borrowed from the mind, never copied into here.** A default kept
    /// alongside is a second wording of the same prompt, and the two diverge —
    /// see [`ReflectionTurns`].
    turns: &'t ReflectionTurns,
    /// The mind's schema, when the daemon has one.
    ///
    /// **A reflection opens against the same sections a live conversation
    /// does.** Built by hand instead, the two prompts drift apart section by
    /// section and nothing says so — which is what happened: the reflection's
    /// frame named none of the schema's sections and carried none of its
    /// collections, so the anchor, the beliefs, the relationships and the mood
    /// were all authored, installed, and invisible to it.
    ///
    /// `None` for a daemon with no mind, which has no schema to open against.
    projected: Option<&'t Projected>,
}

impl<'t> Reflect<'t> {
    pub fn new(
        engine: Arc<ConversationEngine>,
        base_config: SequenceConfig,
        turns: &'t ReflectionTurns,
    ) -> Self {
        Self {
            engine,
            base_config,
            turns,
            projected: None,
        }
    }

    /// Open against the mind's schema rather than a rendered copy of it.
    pub fn under(mut self, projected: Option<&'t Projected>) -> Self {
        self.projected = projected;
        self
    }

    /// Stop a character, ask it two things, and throw the conversation away.
    ///
    /// `sampled_axes` must be a **sample** of the character's dream corpus and
    /// never the whole of it. Shown every axis it had used, a generator stopped
    /// inventing and recombined — returning an assumption that was three
    /// existing ones crossed, with a brief whose closing line restated one of
    /// them. Shown a random eight of the same set it found an axis nowhere on
    /// the list. The failure is anchoring, not exhaustion, and it gets worse as
    /// the corpus grows, so this is not an optimisation.
    #[allow(clippy::too_many_arguments)]
    pub async fn run(
        &self,
        npc_id: u64,
        persona: &Persona<'_>,
        mode: Mode,
        situation: &str,
        inner_thoughts: &str,
        feeling: &str,
        domain: &str,
        sampled_axes: &[String],
        // **Handed the line that crosses back the moment it exists** — after
        // the first question, before any dream is asked for. What lets the act
        // answer the character in the time one question takes rather than the
        // time the whole review does; the dream is written after it, on the
        // same thread, with nobody waiting on it.
        //
        // **And it answers whether to go on.** `false` ends the conversation
        // at the answer — no brief asked for and no dream: what a reflect gets
        // while its character already has a dream being written. See
        // `Runtime::claim_dream`.
        on_reflection: &mut (dyn FnMut(&str) -> bool + Send),
    ) -> anyhow::Result<Reflection> {
        let started = std::time::Instant::now();

        let mut cfg = self.base_config.clone();
        cfg.context_window_turns = CONTEXT_WINDOW_TURNS;

        // **The vocabulary is what it is asked for, not what it may do.** None
        // of the world's acts is ever shown here — a reflection offered one will
        // eventually call it, and an act emitted from a conversation the world
        // is not reading is an act the character believes it performed and did
        // not. What it reads is `reflection` and `dream`, and the answer
        // grammars below are compiled from the same two entries, so the shape
        // the prompt describes is literally the shape the tree emits. See
        // [`ASKED`].
        let env = ToolCallEnvelope::for_dialect(&self.base_config.dialect);
        let asked: Vec<&Tool> = ASKED.iter().collect();
        // The schema's own prompt when there is one, so this conversation opens
        // on the same sections a live one does. The rendered frame is what a
        // daemon with no mind gets, and nothing else.
        let system = match self.projected {
            Some(p) => p.prompt.clone(),
            None => prompt::build_for(Stance::Reflecting, persona, mode, &asked, &env),
        };

        let (mut sequence, timeline) = {
            let engine = &self.engine;
            let seq = match self.projected {
                // Its own dreams, gathered deep — §4: a reflection is the same
                // retrieval as an acting turn, run further. See `dreams`.
                Some(p) => engine.new_conversation_with_projection(
                    &system,
                    dreams::scoped(&p.builder, npc_id, dreams::IN_REFLECTION),
                    p.layer,
                    p.group,
                    cfg,
                )?,
                None => engine.new_conversation(&system, cfg)?,
            };
            let tl = seq.timeline_id();
            // Before the first turn seals. After it, the turn is already on the
            // cold path and this retracts nothing.
            engine.mark_timeline_transient(tl);
            (seq, tl)
        };
        // Tombstoned as well as transient, however this run ends. Transient
        // keeps it off disk; the tombstone is what keeps it out of every later
        // gather and out of `find_conversations_by_metadata`, so nothing can
        // resume or surface it. A drop guard rather than a tail call, because
        // a caller that drops this future mid-decode runs nothing else.
        let _retired = Throwaway::new(Arc::clone(&self.engine), timeline, "reflection");

        // **What the model actually read, not what was handed in.**
        //
        // Under the projection the string passed to `new_conversation_*` is only
        // the prelude — everything before the first collection. The anchor, the
        // world and the building are collection members the projection prefills
        // when a turn pins them, so a response reporting the handed-in string
        // reports a fraction of the frame and looks complete. That cost a
        // diagnosis: a dream arrived furnished with a coffee cup and a woman in a
        // grey coat, and there was no way to tell from the record whether the
        // vault had reached the prompt at all.
        let read = sequence.system_prompt().to_string();

        // **The think half of the acting grammar, and nothing else.** A
        // reflection has no acts, so an action loop would hold it to a shape it
        // is not producing — but it still decodes `<think>`, and with no stencil
        // on that token the block is unbounded. See [`think_off`].
        let grammar = think_off(&self.engine, &self.base_config);

        // **The stencil only preempts the block the turn OPENS with.**
        // `ThinkMode::Off` compiles to "inject `</think>`, then End", and past
        // `End` the tree constrains nothing — so a `<think>` the model decodes
        // *inside its answer* is outside the grammar entirely. Measured: a brief
        // came back as a full `Thinking Process:` deliberation which ran long
        // enough to rehearse its own answer in label form, and the parser
        // harvested the rehearsal.
        //
        // The sampling backstop is the half that covers that, and the inherited
        // one is the wrong size for this: `apply_think_mode` gives `Off` a
        // ~220/300-token *dialogue* budget, which caps a runaway rather than
        // preventing a block. A reflection wants no reasoning at all, so any
        // segment is force-closed after a single token. That is the same
        // collapse the summariser gets, reached by a different route — it gets
        // it for having a tiny answer budget, and a reflection's is large.
        let sampling = self
            .base_config
            .sampling
            .clone()
            .with_graceful_segment_close_after(0)
            .with_force_segment_close_after(1);

        // **The repetition control is the cast's, and it is not adjusted here.**
        //
        // The base sampling carries presence and any `cross_turn_penalty`
        // from the checkpoint, and the schema does not set
        // `free_tool_calls_from_penalties` — so all of it is live inside the
        // stencilled `brief` span, which is exactly where it is wanted: the
        // argument there *is* the prose. A reflection is the same model writing
        // under the same tuning as the character it belongs to, and a second
        // opinion about either from this file would be running it outside the
        // settings the numbers were measured under.
        //
        // Two overrides used to sit here — `eos_boost` zeroed for every writing
        // turn, `cross_turn_penalty` zeroed for the repair passes — and both are
        // gone. The first was added against a truncation that turned out to be a
        // hand-rolled prose span exiting on EOS or budget, fixed since by the
        // tool-call parameter span's close-marker terminator; what it left behind
        // was a decode with no EOS pressure at all, which is the shape an empty
        // reflection has.

        // One grammar per answer kind. Both fall back to the think stencil if
        // they will not compile, which leaves the old prose-and-parse behaviour
        // rather than no reflection at all.
        let dreaming = call_tree(&self.engine, &env, &DREAM).or_else(|| grammar.clone());
        let thinking = call_tree(&self.engine, &env, &REFLECTION).or_else(|| grammar.clone());
        // **Opening on the schema is not the same as reading it.**
        //
        // `Projection::prelude` stops at the first collection, so the anchor, the
        // world and the building reach a turn only when that turn pins them.
        // Acting turns do; this one did not, and the result was a dream set in an
        // office — phones, a coffee machine, grey carpet, a window that opens —
        // none of which exists in a sealed vault. The brief was well-formed and
        // about the wrong world, because the frame no longer said which world.
        //
        // Pinned once and reused on every turn: what a character *is* does not
        // change between being asked for a reflection and being asked for a dream.
        let mut selection = self
            .projected
            .map(|p| {
                p.identities.selection_for(
                    npc_id,
                    persona.personality,
                    persona.world_id,
                    persona.building,
                    identity::Deliberation::default(),
                )
            })
            .unwrap_or_default();
        // **The schema's `reflecting` branch, on every turn.** Unselected the
        // stance falls to `acting`, whose grounding forbids inventing a person
        // — in the conversation that is about to be asked for a dream, which
        // requires it. `reflecting` rules on neither, and each question carries
        // its own rule; see the module doc.
        selection.select(prompt::STANCE_SELECTOR, Stance::Reflecting.id());
        // **Each turn reads the one answer it is asked for.** The two calls sit
        // in the schema's `tools` collection beside the world's acts, and a turn
        // names its member the way an acting turn names what it can do: the
        // reflection's question shows `reflection`, every dream turn shows
        // `dream`, and the loosed turn — which calls nothing — shows nothing,
        // so the acting frame gated on the list goes with it.
        let asking = |tool: &Tool| {
            let mut s = selection.clone();
            tools::show(&mut s, [tool.name]);
            s
        };
        let for_reflection = asking(&REFLECTION);
        let for_dream = asking(&DREAM);

        let writing = || TurnOptions {
            turn_grammar: dreaming.clone().map(TurnGrammar::of),
            sampling: Some(sampling.clone()),
            selection: for_dream.clone(),
            ..Default::default()
        };
        let exchange = Exchange {
            response_open: self.base_config.dialect.tool_response_open,
            response_close: self.base_config.dialect.tool_response_close,
        };

        // **One exchange: ask in prose, answer in a call, accept or refuse.**
        //
        // The refusal is the part that matters. Before this, a short brief was
        // detected out of band and answered with a generic re-ask; now the
        // verdict goes back inside the wrapper the model reads verdicts in, and
        // it says in words what was wrong. Capped, because an uncapped argument
        // inside a blocking act can spend a character's afternoon on one dream.
        #[allow(clippy::too_many_arguments)]
        async fn ask_until_valid(
            sequence: &mut Sequence,
            exchange: &Exchange<'_>,
            ask: String,
            opts: TurnOptions,
            field_name: &'static str,
            min: usize,
            max: usize,
            what: &'static str,
            // The brief this turn is revising, when it is a
            // revision. `None` for the one that writes it.
            revising: Option<&str>,
            // A check on the whole call beyond the field's
            // length and drift. The reflection has one: its
            // two invariants used to be computed at the end
            // and *reported*, which is a fault nobody acts on
            // — measured, a two-sentence line came back, was
            // flagged, and went to the character anyway. The
            // dream's is for the field that is *not* being
            // measured, which is why it sees the whole call.
            also: Option<fn(&str) -> Option<String>>,
            tokens: &mut Vec<usize>,
            trail: &mut Vec<String>,
        ) -> anyhow::Result<String> {
            let mut turn = sequence
                .send_turn_with_options_async(&ask, opts.clone())
                .await?;
            for _ in 0..MAX_REFUSALS {
                tokens.push(turn.stats.tokens_generated);
                trail.push(turn.text.trim().to_string());
                let value = field(&turn.text, field_name).unwrap_or_default();
                let why = refusal(words(&value), min, max, what)
                    .or_else(|| {
                        also.and_then(|f| f(&turn.text))
                            .map(|w| format!("failed: {w}. Write it again."))
                    })
                    .or_else(|| {
                        // A pass that replaced the dream rather than adjusting it is
                        // refused on the same footing as one that came back the wrong
                        // length: both are the call not doing what was asked, and
                        // both are things the model can fix on being told.
                        revising
                            .filter(|before| rewritten(before, &value))
                            .map(|_| {
                                format!(
                                "failed: that is a different {what}, not a revision of the one \
                                 you were given. Keep its room, its objects and the thing that \
                                 is untrue in it, and change only what was asked."
                            )
                            })
                    });
                let Some(why) = why else {
                    return Ok(turn.text);
                };
                turn = sequence
                    .send_turn_with_options_async(&exchange.respond(&why), opts.clone())
                    .await?;
            }
            tokens.push(turn.stats.tokens_generated);
            trail.push(turn.text.trim().to_string());
            Ok(turn.text)
        }

        let mut tokens: Vec<usize> = Vec::new();
        let mut trail: Vec<String> = Vec::new();

        // ── the reflection ────────────────────────────────────────────────
        //
        // Asked in prose, answered as a call, refused until it is one sentence.
        // **Prose, not labelled fields.** This was three `label: value` lines
        // sitting directly above a grammar that says "now emit
        // `<parameter=said>`", and the model did the obvious thing: it filled the
        // field from the nearest value. One run came back with `said` holding the
        // caller's own `on_your_mind` verbatim, then a second call holding the
        // single word `uneasy`. It had not reflected; it had completed a form.
        //
        // Said as sentences there is no field to copy — the state is context and
        // the question is the only thing asking for an answer.
        // The em-dash rather than a verb, because the thought is the character's
        // own words and arrives however it arrives. `you were thinking {x}` ran
        // straight into "That the roster has had a name on it…" and read as a
        // sentence with a capital letter dropped into the middle of it.
        //
        // The situation goes in as the world wrote it. It is the world's own
        // percept — "You are in the muster hall…" and who is there — which is
        // already a sentence addressed to the reader; wrapping it in "You are
        // {}." made "You are You are in…".
        let mut opening = String::new();
        if !situation.trim().is_empty() {
            opening.push_str(situation.trim());
            opening.push_str("\n\n");
        }
        if !inner_thoughts.trim().is_empty() {
            opening.push_str(&format!(
                "A moment ago, going through your head — {}\n\n",
                inner_thoughts.trim()
            ));
        }
        if !feeling.trim().is_empty() {
            opening.push_str(&format!("What you feel is {}.\n\n", feeling.trim()));
        }
        opening.push_str(&self.turns.question_one);
        let thought_opts = TurnOptions {
            turn_grammar: thinking.clone().map(TurnGrammar::of),
            sampling: Some(sampling.clone()),
            selection: for_reflection,
            ..Default::default()
        };
        let first = ask_until_valid(
            &mut sequence,
            &exchange,
            opening,
            thought_opts,
            "said",
            1,
            REFLECTION_MAX_WORDS,
            "reflection",
            None,
            Some(reflection_call_fault),
            &mut tokens,
            &mut trail,
        )
        .await?;
        // Its one field, read the way every other answer here is read. The
        // markup-stripping salvage is for an answer that did not come back as a
        // call at all — run over a JSON block it would hand the character the
        // whole call object as its own thought.
        //
        // **And handed over now.** It is the only thing the character is
        // waiting on, and nothing after this point changes it.
        let crossing_back = field(&first, "said").unwrap_or_else(|| plain_prose(&first));
        if !on_reflection(&crossing_back) {
            // Retired exactly as a finished one is — the guard above.
            drop(sequence);
            return Ok(Reflection {
                npc_id,
                situation: situation.trim().to_string(),
                feeling: feeling.trim().to_string(),
                domain: domain.to_string(),
                sampled_axes: sampled_axes.to_vec(),
                reflection: crossing_back.clone(),
                assumption: None,
                brief: None,
                raw: String::new(),
                original_raw: String::new(),
                retried: false,
                fault: None,
                retry_raw: None,
                retry_fault: None,
                repairs: Vec::new(),
                reflection_fault: reflection_fault(&crossing_back),
                transcript: trail,
                tokens,
                timeline: timeline.raw(),
                ms: started.elapsed().as_millis() as u64,
                system_prompt: read,
            });
        }

        // **Then it is let go.** The call was the disciplined form; this turn is
        // unstencilled, so what follows is the character thinking in its own
        // voice with nothing holding the shape. It is not what crosses back —
        // that is the one sentence above — but it is what the dream is written
        // out of, and a dream seeded from a sentence is thinner than one seeded
        // from a thought.
        let loosed = sequence
            .send_turn_with_options_async(
                &exchange.respond("ok"),
                TurnOptions {
                    sampling: Some(sampling.clone()),
                    selection: selection.clone(),
                    ..Default::default()
                },
            )
            .await?;
        tokens.push(loosed.stats.tokens_generated);
        trail.push(loosed.text.trim().to_string());

        // ── the dream ─────────────────────────────────────────────────────
        //
        // The authored turn keeps its slots — that is how the mind decides where
        // the rotated domain and the diversity steer appear in the ask.
        let instruction = self
            .turns
            .question_two
            .replace(DOMAIN_SLOT, domain)
            .replace(AXES_SLOT, &steer(sampled_axes));

        let second_raw = ask_until_valid(
            &mut sequence,
            &exchange,
            instruction,
            writing(),
            "brief",
            BRIEF_MIN_WORDS,
            BRIEF_MAX_WORDS,
            "dream",
            None,
            Some(dream_call_fault),
            &mut tokens,
            &mut trail,
        )
        .await?;
        let original_raw = second_raw.trim().to_string();
        let mut raw = second_raw;
        let mut parsed = Brief::from_call(&raw);
        let fault = parsed.fault(sampled_axes);
        let mut retry_raw = None;
        let mut retry_fault = None;
        if fault.is_some() {
            let third = sequence
                .send_turn_with_options_async(&self.turns.retry, writing())
                .await?;
            // Kept like every other decode, so `transcript` and `tokens` stay one
            // entry per turn. It used to be counted and not kept: a retry that was
            // taken then appeared nowhere in the response, and every transcript
            // entry after it read against the wrong token count.
            tokens.push(third.stats.tokens_generated);
            trail.push(third.text.trim().to_string());
            let again = third.text;
            let reparsed = Brief::from_call(&again);
            let refault = reparsed.fault(sampled_axes);
            // Take the retry when it is clean, or when it produced a brief and
            // the first attempt did not. Otherwise keep the first: a second
            // attempt that is worse is a real outcome, and preferring it because
            // it came last would make the retry a downgrade.
            if refault.is_none() || reparsed.beats(&parsed) {
                raw = again;
                parsed = reparsed;
            } else {
                // Reported rather than dropped. A retry that failed is the only
                // evidence of *why* the retry is not working, and a response that
                // says "retried: true" and shows nothing of it leaves the next
                // question — is the retry turn wrong, or is the brief unfixable —
                // answerable only by guessing.
                retry_raw = Some(again.trim().to_string());
                retry_fault = refault;
            }
        }
        let retried = fault.is_some();

        // ── the repair pass ────────────────────────────────────────────────
        //
        // **Unconditional, and that is the point.** The retry above answers a
        // brief that came out malformed. These answer a brief that is perfectly
        // well-formed and is not a dream, which over 27 live samples was the
        // majority case — every structural check passing while the three rules
        // did not, most often because the strangeness had spread past the one
        // hole and taken the ordinary day with it. No cheap check sees that, so
        // it is repaired rather than detected.
        //
        // Each pass rewrites the whole brief and re-emits both parameters, so it is
        // parsed and checked exactly like the first. A pass whose answer does not
        // parse, or which faults when its predecessor did not, is **discarded and
        // the previous brief stands** — a repair that makes a brief worse is a
        // real outcome, and taking it because it came last is how a review turns
        // into a downgrade. Every pass is recorded either way.
        let mut repairs = Vec::with_capacity(self.turns.repair.len());
        // **Not over a brief that will never be dreamt.** A brief whose
        // assumption is still not one short line after the retry is refused at
        // the dream ([`Reflection::dream`]), so reviewing it is decode spent on
        // nothing — and it is the most expensive case there is: measured, a
        // thirty-sentence assumption re-emitted through every pass and every
        // refusal round, eighteen turns and 11,510 tokens in 162 seconds for a
        // reflection that ended with no dream.
        let repairable = parsed
            .assumption
            .as_deref()
            .is_some_and(|a| assumption_fault(a).is_none());
        let questions = if repairable {
            &self.turns.repair[..]
        } else {
            &[]
        };
        for (n, question) in questions.iter().enumerate() {
            // **The dream in front of it, not recalled.** A pass that works from
            // memory is working from a conversation several turns deep that the
            // window will eventually trim, and it drifts — one measured run went
            // from a warm stone on a desk to a mouth made of teeth in three
            // passes, each answering what it was asked and none of them revising
            // the same dream. Shown the text and the axis verbatim, the pass has
            // something to edit rather than something to remember.
            //
            // Editorial guidance stays a prose turn rather than a refusal: a
            // `<tool_response>` that says *failed* about a call that succeeded
            // teaches the model that success is arbitrary. Length and drift are
            // refusals, because both are the call not doing what was asked.
            let standing = parsed.brief.clone().unwrap_or_default();
            let axis = parsed.assumption.clone().unwrap_or_default();
            let ask = format!(
                "The dream as it stands:\n\n{standing}\n\nThe assumption it suspends, which does \
                 not change: {axis}\n\n{question}"
            );
            let answer = ask_until_valid(
                &mut sequence,
                &exchange,
                ask,
                writing(),
                "brief",
                BRIEF_MIN_WORDS,
                BRIEF_MAX_WORDS,
                "dream",
                Some(standing.as_str()),
                Some(dream_call_fault),
                &mut tokens,
                &mut trail,
            )
            .await?;
            let candidate = Brief::from_call(&answer);
            let cfault = candidate.fault(sampled_axes);
            let took = cfault.is_none() && candidate.brief.is_some();
            repairs.push(Repair {
                pass: n + 1,
                took,
                fault: cfault,
                raw: answer.trim().to_string(),
            });
            if took {
                raw = answer;
                parsed = candidate;
            }
        }

        // ── what crosses back ──────────────────────────────────────────────
        //
        // **The reflection, and it was asked first.**
        //
        // Not for convenience — the order is what keeps the dream out of the
        // action stream. The only line the character ever sees is written before
        // any dream exists, so there is nothing in it to leak. That safety comes
        // from the sequence rather than from asking the model not to repeat
        // itself, which is the difference between a property and a hope.
        //
        // Its length is held by the refusal loop rather than by a token ceiling:
        // one sentence, under [`REFLECTION_MAX_WORDS`], said back in words the
        // model can act on. Handed over above, as soon as it was answered.

        drop(sequence);

        Ok(Reflection {
            npc_id,
            situation: situation.trim().to_string(),
            feeling: feeling.trim().to_string(),
            domain: domain.to_string(),
            sampled_axes: sampled_axes.to_vec(),
            reflection: crossing_back.clone(),
            assumption: parsed.assumption,
            brief: parsed.brief,
            raw: raw.trim().to_string(),
            original_raw,
            retried,
            fault,
            retry_raw,
            retry_fault,
            repairs,
            reflection_fault: reflection_fault(&crossing_back),
            transcript: trail,
            tokens,
            timeline: timeline.raw(),
            ms: started.elapsed().as_millis() as u64,
            system_prompt: read,
        })
    }
}

mod answer;
mod brief;
mod reflection;
mod steer;
#[cfg(test)]
mod tests_answer;
#[cfg(test)]
mod tests_brief;
#[cfg(test)]
mod tests_reflection;

pub use answer::*;
use brief::*;
pub use reflection::*;
pub use steer::*;
