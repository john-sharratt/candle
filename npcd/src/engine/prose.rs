//! One piece of prose, decoded on the resident model in a conversation that is
//! then thrown away.
//!
//! The shape [`super::dreams::dream`] established: a fresh conversation under the
//! job's own system prompt, marked transient so none of it reaches the cold tier,
//! one turn, then tombstoned. Nothing about any character is in it — no world, no
//! personality, no acts, no history — so the model writes the prose it was asked
//! for rather than answering as whoever it was last playing. [`crate::prose`] is
//! the async front the routes call.
//!
//! It opens on its own reserved layer and group, closed — see
//! [`prose_projection`] for why both matter.
//!
//! # Reasoning is suppressed
//!
//! A name, a description or a verdict wants no deliberation, and a `<think>`
//! opened here is the model planning the answer in the answer's place. Free prose
//! decodes under [`think_off`] with the segment closed after one token — the
//! reflection's and the dream's mechanism.
//!
//! # An answer can be a call
//!
//! [`decode_call`] holds the turn to one of a set of calls — the dialect's own
//! call envelope, every argument closing on its terminator — and returns the call
//! as written, for a caller that reads its fields. The command table's mission
//! generator answers this way (`engine::mission_gen`).
//!
//! # A fixed set of answers is a grammar, not a request
//!
//! `choices` compiles to a stencil whose root branches once per arm, so an answer
//! outside the set is unreachable rather than unlikely. The dialect's closed think
//! block is prefilled ahead of it — the acting turn's opening — so the grammar
//! governs the answer itself and the first token the model chooses is already one
//! of the arms.

use std::sync::Arc;
use std::time::{Duration, Instant};

use candle_conversation::projection::{
    Builder, DecodePriority, GroupId, LayerId, Reserved, SectionId,
};
use candle_conversation::stencil::{
    compile_action_loop, compile_think_tree, compile_tool_call_tree, StencilTree,
    StencilTreeBuilder, ThinkMode, ThinkSteerEnvelope, ToolCallEnvelope, ToolSpec,
};
use candle_conversation::throwaway::{HeldFrame, Throwaway};
use candle_conversation::{
    ConversationEngine, Sequence, SequenceConfig, TokenDecoder, TurnEvent, TurnOptions,
    TurnResponse,
};

use crate::engine::reflect::{plain_prose, think_off};
use crate::prose::{Answer, Request, DEFAULT_SYSTEM};

/// How a job's system prompt is held on the engine.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Frame {
    /// Sealed for the job and retired once it lets go — see
    /// [`ConversationEngine::transient_prompt_section`]. For a voice that may
    /// never be asked again: a caller whose every prompt differs leaves nothing
    /// behind.
    Transient,
    /// Sealed once and kept in the substrate — see
    /// [`ConversationEngine::plain_prompt_section`]. For a **fixed** voice asked
    /// on every request: the next job reads the sealed frame instead of
    /// prefilling it again. A gate in front of another job asks this way; its
    /// questions never change, and the person behind it waits for every
    /// prefill. One frame per distinct text, so only a fixed voice may take it.
    Kept,
}

/// Decode `request` with `seed` on `band`, its system prompt held as `frame`,
/// handing each new fragment of text to `on_fragment` as it lands.
///
/// Awaits the decode; the routes in [`crate::prose`] call it directly, on the
/// characters' [`DecodePriority::Normal`]; a judgement an operation waits on
/// takes [`WORK_BAND`].
pub async fn decode(
    engine: &Arc<ConversationEngine>,
    base: &SequenceConfig,
    request: &Request,
    band: DecodePriority,
    frame: Frame,
    seed: u64,
    on_fragment: &mut (dyn FnMut(&str) + Send),
) -> anyhow::Result<Answer> {
    let system = match request.system.trim() {
        "" => DEFAULT_SYSTEM,
        _ => request.system.as_str(),
    };

    // The grammar and the opening, built before the conversation so a choice set
    // the vocabulary cannot compile costs nothing to refuse.
    let (turn_grammar, prefill) = match request.choices.as_deref() {
        Some(arms) if !arms.is_empty() => {
            let tree = choice_tree(engine, arms)?;
            (
                Some(Arc::new(tree)),
                Some(base.dialect.no_think_block.to_string()),
            )
        }
        _ => (think_off(engine, base), None),
    };
    let held = Held {
        turn_grammar,
        prefill,
        max_tokens: request.max_tokens as usize,
        temperature: request.temperature,
        think: None,
        band,
        frame,
    };
    let (text, tokens) = framed(
        engine,
        base,
        system,
        &request.prompt,
        held,
        seed,
        on_fragment,
    )
    .await?;
    Ok(Answer {
        text: plain_prose(&text),
        tokens,
        seed,
    })
}

/// One answer to ask for as a call — see [`decode_call`].
pub struct CallAsk<'a> {
    /// The voice.
    pub system: &'a str,
    /// What is asked.
    pub prompt: &'a str,
    /// The calls the answer may be — one of them, and nothing else.
    pub calls: &'a [ToolSpec],
    pub max_tokens: usize,
    /// `None` keeps the checkpoint's own.
    pub temperature: Option<f32>,
    pub seed: u64,
    /// How much the answer is thought through before the call is written.
    /// [`ThinkMode::Off`] opens straight into the call.
    pub think: ThinkMode,
    /// The band the job decodes at — see [`WORK_BAND`].
    pub band: DecodePriority,
}

/// The band of the work an operation waits on: the table's proposals and
/// readings, and a Maker's composing of the document.
///
/// **Above the characters' own turns, which wait on it.** At the characters'
/// band, a table call on WW waited 31 to 52 minutes for its first token behind
/// four Makers' turns, dreams and journals — its prefill throttled under their
/// decodes — and then decoded in under a minute. While it waited every Maker
/// drew routines, because the work the call would have handed them did not
/// exist yet. The engine's other prose — a description, a name, a life — stays
/// at the characters' band.
pub const WORK_BAND: DecodePriority = DecodePriority::High;

/// Decode an answer held to one of `ask.calls`, and return the call as written.
///
/// **The answer is the call, not prose about it.** Every argument closes on its
/// own terminator — the reflection's mechanism (`reflect::call_tree`) — so what
/// comes back is a call the caller reads with
/// [`crate::engine::journal::tools::arguments`].
///
/// **A thinking model is given somewhere to think.** Opened straight into the
/// call, it thought in the call's first free field instead: a reading's `notes`
/// ran "I will output the answer. I am done thinking. I will output…" to the
/// token cap, and its `checked` the same, because a field has no `</think>` to
/// end on. With `ask.think` set the turn opens a reasoning block the stencil
/// closes on its budget, and the call follows it as one grammar; what comes back
/// is the call alone.
pub async fn decode_call(
    engine: &Arc<ConversationEngine>,
    base: &SequenceConfig,
    ask: &CallAsk<'_>,
) -> anyhow::Result<String> {
    let whole = decode_call_thought(engine, base, ask).await?;
    Ok(past_thinking(&whole).to_string())
}

/// [`decode_call`]'s turn as written, its reasoning block with it — what a
/// caller weighing how the call was arrived at reads.
pub async fn decode_call_thought(
    engine: &Arc<ConversationEngine>,
    base: &SequenceConfig,
    ask: &CallAsk<'_>,
) -> anyhow::Result<String> {
    // **No one field may take the whole answer.** One that runs on is closed
    // where it stands with room left for the rest of the call, which the
    // caller's own checks then judge; unbounded, it ran to the turn's cap and
    // the call around it was lost.
    let env = ToolCallEnvelope::for_dialect(&base.dialect)
        .with_max_string_tokens((ask.max_tokens * 3 / 4) as u32);
    let (spec, prefill) = match ask.think {
        ThinkMode::Off => (compile_tool_call_tree(ask.calls, &env), None),
        mode => {
            let steer = think_steer(engine, base)?;
            let prelude = compile_think_tree(mode, &steer);
            let close_turn = ToolCallEnvelope::turn_close(&base.dialect);
            (
                compile_action_loop(ask.calls, &env, 1, &close_turn, Some(&prelude)),
                Some("<think>".to_string()),
            )
        }
    };
    let spec = spec.map_err(|e| anyhow::anyhow!("the answer grammar would not build: {e:#}"))?;
    let tree = engine.compile_stencil(&spec)?;
    // **The thinking is budgeted on top of the answer, not out of it.** Given
    // the answer's 2400 tokens for both, a generation that thought at length
    // came back as a call cut off before it closed — "that was not a call" — or
    // written in a hurry with a field left empty.
    let max_tokens = match ask.think {
        ThinkMode::Off => ask.max_tokens,
        mode => mode.eos_budget(ask.max_tokens as i32).2 as usize,
    };
    let held = Held {
        turn_grammar: Some(Arc::new(tree)),
        prefill,
        max_tokens,
        temperature: ask.temperature,
        think: (ask.think != ThinkMode::Off).then_some((ask.think, ask.max_tokens)),
        band: ask.band,
        frame: Frame::Transient,
    };
    let (text, _) = framed(
        engine,
        base,
        ask.system,
        ask.prompt,
        held,
        ask.seed,
        &mut |_: &str| {},
    )
    .await?;
    Ok(text)
}

/// The token ids a reasoning block is steered by, from this checkpoint.
fn think_steer(
    engine: &Arc<ConversationEngine>,
    base: &SequenceConfig,
) -> anyhow::Result<ThinkSteerEnvelope> {
    let e = engine;
    let tok = e.tokenizer();
    let (Some(think_open), Some(think_close)) =
        (tok.token_to_id("<think>"), tok.token_to_id("</think>"))
    else {
        anyhow::bail!("this checkpoint's tokenizer has no single <think> token to steer by");
    };
    Ok(ThinkSteerEnvelope {
        think_open,
        think_close,
        eos: tok
            .token_to_id(base.dialect.assistant_end.trim_end())
            .unwrap_or(0),
        // Spliced onto the call grammar, so the join is a node edge already.
        after_close: "",
    })
}

/// What a turn wrote after its reasoning block closed — all of it when it had
/// none. A thought may hold a brace, and the call is read from the first one.
fn past_thinking(text: &str) -> &str {
    match text.rfind("</think>") {
        Some(at) => text[at + "</think>".len()..].trim_start(),
        None => text,
    }
}

/// What one job's turn is held to.
struct Held {
    turn_grammar: Option<Arc<StencilTree>>,
    prefill: Option<String>,
    max_tokens: usize,
    temperature: Option<f32>,
    /// The reasoning block's dial and the answer's own budget after it, when
    /// the turn opens one: its close budget is programmed into the sampling the
    /// way an acting turn's is.
    think: Option<(ThinkMode, usize)>,
    /// The band the turn decodes at.
    band: DecodePriority,
    /// How its system prompt is held.
    frame: Frame,
}

/// One turn under `system`, in a conversation opened for it and then thrown
/// away. Returns the text past any prefilled opening, and the tokens decoded.
async fn framed(
    engine: &Arc<ConversationEngine>,
    base: &SequenceConfig,
    system: &str,
    prompt: &str,
    held: Held,
    seed: u64,
    on_fragment: &mut (dyn FnMut(&str) + Send),
) -> anyhow::Result<(String, u32)> {
    let mut cfg = base.clone();
    // One turn with nothing before it: there is no history to carry.
    cfg.context_window_turns = 0;
    let Held {
        turn_grammar,
        prefill,
        max_tokens,
        temperature,
        think,
        band,
        frame,
    } = held;

    // **The opening runs off the async task.** It waits on the scheduler for a
    // slot, which answers between waves — a wave's length under load — and it
    // waits synchronously. On the task it parked a runtime worker for that
    // long, and stalled everything polled beside this future: the MCP stream's
    // progress ticker shares its task, so an image call went a minute without
    // a notification and its client gave it up.
    let (mut job, decoder) = {
        let engine = Arc::clone(engine);
        let system = system.to_string();
        let formatted = base.dialect.format_system_prompt(&system);
        tokio::task::spawn_blocking(move || -> anyhow::Result<_> {
            let (id, frame) = match frame {
                Frame::Transient => {
                    let held = HeldFrame::take(Arc::clone(&engine), &system)?;
                    (held.id(), Some(held))
                }
                Frame::Kept => (engine.plain_prompt_section(&system)?, None),
            };
            let (builder, layer, group) = prose_projection(&system, id, band);
            let sequence =
                engine.new_conversation_with_projection(&formatted, builder, layer, group, cfg)?;
            let timeline = sequence.timeline_id();
            engine.mark_timeline_transient(timeline);
            let job = ProseJob {
                _retired: Throwaway::new(Arc::clone(&engine), timeline, "prose"),
                sequence,
                _frame: frame,
            };
            Ok((job, engine.token_decoder()))
        })
        .await
        .map_err(|e| anyhow::anyhow!("the prose opening did not finish: {e}"))??
    };

    let mut sampling = match think {
        // The block is closed on its dial's budget, as an acting turn's is.
        Some((mode, answer)) => base
            .sampling
            .clone()
            .with_seed(seed)
            .with_think_mode(mode, answer),
        None => base
            .sampling
            .clone()
            .with_seed(seed)
            .with_graceful_segment_close_after(0)
            .with_force_segment_close_after(1),
    };
    if let Some(t) = temperature {
        sampling.temperature = t;
    }
    let options = TurnOptions {
        max_tokens: Some(max_tokens),
        sampling: Some(sampling),
        turn_grammar,
        assistant_prefill: prefill.clone(),
        ..Default::default()
    };

    let decoded = run_turn(&mut job.sequence, prompt, options, &decoder, on_fragment).await;
    drop(job);

    let r = decoded?;
    // A prefilled block may come back at the head of the text. It was the
    // opening's scaffolding, not the answer.
    let text = match &prefill {
        Some(p) => r.text.strip_prefix(p.as_str()).unwrap_or(&r.text),
        None => r.text.as_str(),
    };
    Ok((text.to_string(), r.stats.tokens_generated as u32))
}

/// A prose job's conversation and what it holds, let go in the order the
/// fields are declared on every path out — the job done, failed, or its future
/// dropped mid-decode: the slot, then the timeline it wrote — retired whatever
/// happened, since nothing of a throwaway job is worth keeping — then its
/// transient frame, which the slot was reading. A kept frame is never
/// released.
struct ProseJob {
    sequence: Sequence,
    _retired: Throwaway<Arc<ConversationEngine>>,
    _frame: Option<HeldFrame<Arc<ConversationEngine>>>,
}

/// Submit the turn, stream its fragments, and seal it.
async fn run_turn(
    sequence: &mut Sequence,
    prompt: &str,
    options: TurnOptions,
    decoder: &TokenDecoder,
    on_fragment: &mut (dyn FnMut(&str) + Send),
) -> anyhow::Result<TurnResponse> {
    use tokio_stream::StreamExt;

    let handle = sequence.submit_turn_with_options(prompt, options)?;
    let seq = sequence.id();
    tracing::info!(
        %seq,
        prompt_words = prompt.split_whitespace().count(),
        "prose turn submitted"
    );
    let mut ids: Vec<u32> = Vec::new();
    let mut preview = Preview::default();
    let mut response = None;
    // **How long a prose turn waited, and how long it ran** — and, above, that
    // it was submitted at all. The table's generations and readings are prose
    // turns, and the loop waits on each: one that took eighty minutes said
    // nothing at all while it did.
    let submitted = Instant::now();
    let mut first_token: Option<Duration> = None;
    {
        let mut events = std::pin::pin!(handle.stream_async());
        while let Some(event) = events.next().await {
            match event {
                TurnEvent::Token(id) => {
                    first_token.get_or_insert_with(|| submitted.elapsed());
                    ids.push(id);
                    if let Some(delta) = preview.advance(visible(&decoder.decode(&ids))) {
                        on_fragment(&delta);
                    }
                }
                TurnEvent::Done(r) => {
                    response = Some(r);
                    break;
                }
                TurnEvent::Error(e) => anyhow::bail!("the prose turn failed: {e}"),
                _ => {}
            }
        }
    }
    let Some(r) = response else {
        anyhow::bail!("the scheduler went away before the prose finished");
    };
    tracing::info!(
        %seq,
        first_token_ms = first_token.map(|d| d.as_millis() as u64),
        total_ms = submitted.elapsed().as_millis() as u64,
        tokens = ids.len(),
        "prose turn done"
    );
    sequence.finish_turn(handle, &r)?;
    Ok(r)
}

/// The projection a prose job runs under: its own reserved layer and group, that
/// group closed ([`Builder::for_closed_reserved`]), and `frame` holding its
/// system prompt.
///
/// **`frame` comes from the engine, resolved from the prompt's own text.** A
/// section the substrate already holds is reused by id without its text being
/// compared, so every job opened under one fixed frame id read whichever voice
/// had sealed there first. The first job after a start is the create form's
/// name, so every description after it was written under "reply with the name
/// and nothing else" — and came back as a name. The frame is transient or kept
/// as the job asks — see [`Frame`].
///
/// **Reserved, so a prose job never shares ids with the mind.** A plain-prompt
/// schema's layer and group are otherwise fixed at 1, which is the mind's own
/// first layer and group — the job's turn would be written into the mind's
/// schema space.
///
/// **Closed, so one job never reads another's turn.** Every prose job writes to
/// the same reserved group, and a group with no tags admits every turn in it.
/// Tagged with something no turn carries, it admits none of them, and a job
/// reads its own system prompt and question and nothing else.
///
/// `band` is the job's: the characters' own, [`DecodePriority::Normal`] —
/// below zend's dialogue — or [`WORK_BAND`] for the work an operation waits on.
fn prose_projection(
    system: &str,
    frame: SectionId,
    band: DecodePriority,
) -> (Builder, LayerId, GroupId) {
    (
        Builder::for_closed_reserved(system, Reserved::Prose, frame, band),
        LayerId::reserved(Reserved::Prose),
        GroupId::reserved(Reserved::Prose),
    )
}

/// The grammar for a fixed set of answers: one branch per arm, and the turn ends
/// after it.
///
/// The arms are checked, and one that is not a single token is logged: a branch
/// masks the sampler to the first token of each arm and then forces the rest, so
/// a multi-token arm commits the whole word on the strength of its first token.
/// The cost is silent — the answer still parses, it is just no longer the model's
/// after the first token.
fn choice_tree(engine: &ConversationEngine, arms: &[String]) -> anyhow::Result<StencilTree> {
    let tokenizer = engine.tokenizer();
    for arm in arms {
        let n = tokenizer
            .encode(arm.as_str(), false)
            .map(|e| e.len())
            .unwrap_or(0);
        if n != 1 {
            tracing::warn!(
                arm = %arm,
                tokens = n,
                "a choice arm is not a single token — the walk commits to it on the first and \
                 forces the rest"
            );
        }
    }
    let pairs: Vec<(&str, &str)> = arms.iter().map(|a| (a.as_str(), "done")).collect();
    let spec = StencilTreeBuilder::new("choice")
        .root("pick")
        .branch("pick", &pairs)
        .end("done")
        .build()
        .map_err(|e| anyhow::anyhow!("building the choice grammar: {e}"))?;
    Ok(engine.compile_stencil(&spec)?)
}

/// What a watcher should see of `decoded` so far.
///
/// The reasoning close the think-off grammar injects arrives at the head of the
/// text; it is the grammar's scaffolding rather than prose, and [`plain_prose`]
/// strips it from the final answer, so the preview does not show it either.
fn visible(decoded: &str) -> &str {
    let t = decoded.trim_start();
    t.strip_prefix("</think>").map_or(t, str::trim_start)
}

/// What a watcher has been shown so far, so the next step can send only what is
/// new.
///
/// **Why the whole sequence is re-decoded every step.** A token is not a
/// character: detokenising one in isolation loses the spacing rule that depends on
/// its neighbour, and a multi-byte character can span two tokens, so
/// `decode(&[one])` yields a replacement character where the pair yields a letter.
/// The only reliable fragment is the difference between decoding the whole
/// sequence and decoding it one token shorter. That is quadratic in the token
/// count, which sounds worse than it is: a description is ~60 tokens, and each
/// decode is microseconds against a forward pass.
///
/// The comparison is a **common prefix**, not a strict one. Tokenizer cleanup can
/// revise a character already shown — `" don "` + `"'t"` becomes `" don't"`,
/// dropping a space that was already sent — and a strict-prefix check would treat
/// that as a break and go silent for the rest of the generation. Instead the
/// preview resynchronises: the sent text can end up a character or two from the
/// final, which is why the finished text is what the caller stores and the
/// fragments are only what it shows while waiting.
#[derive(Default)]
struct Preview {
    sent: String,
}

impl Preview {
    /// The part of `full` a watcher has not seen, or `None` when there is none.
    fn advance(&mut self, full: &str) -> Option<String> {
        let common = common_prefix_len(&self.sent, full);
        let delta = &full[common..];
        self.sent.truncate(common);
        if delta.is_empty() {
            // The step produced no new visible text — a token that only affected
            // cleanup. Nothing to send, and the state still follows what was
            // actually decoded.
            return None;
        }
        self.sent.push_str(delta);
        Some(delta.to_string())
    }
}

/// The length in bytes of the longest shared prefix, always on a character
/// boundary — slicing a string at a byte that splits a character panics, and a
/// multi-byte character is exactly where two decodes are most likely to differ.
fn common_prefix_len(a: &str, b: &str) -> usize {
    let mut n = 0;
    for (x, y) in a.chars().zip(b.chars()) {
        if x != y {
            break;
        }
        n += x.len_utf8();
    }
    n
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_conversation::projection::SystemPromptItem;

    /// **A prose job reads its own prompt** — on the reserved band, and framed by
    /// the section it was given rather than one every job shares. A shared frame
    /// is how a description came back as a name: it was written under the naming
    /// voice that had sealed there first.
    #[test]
    fn a_prose_projection_is_reserved_and_framed_by_its_own_section() {
        let frame = SectionId::new(7_001);
        let (builder, layer, group) =
            prose_projection("You name characters.", frame, DecodePriority::Normal);
        assert!(
            layer.is_reserved(),
            "a prose job must not share the mind's id space"
        );
        assert_eq!(layer, LayerId::reserved(Reserved::Prose));
        assert_eq!(group, GroupId::reserved(Reserved::Prose));
        let sealed: Vec<(SectionId, &str)> = builder
            .schema()
            .system_prompt
            .items
            .iter()
            .filter_map(|item| match item {
                SystemPromptItem::Section(s) => Some((s.id, s.content.as_str())),
                _ => None,
            })
            .collect();
        assert_eq!(sealed, vec![(frame, "You name characters.")]);
    }

    /// Each step sends only what is new, and a step with nothing new sends
    /// nothing.
    #[test]
    fn a_preview_sends_only_what_is_new() {
        let mut p = Preview::default();
        assert_eq!(p.advance("The"), Some("The".into()));
        assert_eq!(p.advance("The yard"), Some(" yard".into()));
        assert_eq!(p.advance("The yard"), None);
    }

    /// **Tokenizer cleanup resynchronises the preview rather than silencing
    /// it.** The space sent before `'t` arrived is revised away; the preview
    /// carries on from the shared prefix instead of treating the revision as a
    /// break.
    #[test]
    fn a_revised_character_resynchronises_the_preview() {
        let mut p = Preview::default();
        p.advance("I don ");
        assert_eq!(p.advance("I don't"), Some("'t".into()));
        assert_eq!(p.advance("I don't know"), Some(" know".into()));
    }

    /// **The call is read past the thinking**, which may hold a brace of its
    /// own; an answer with no block is all call.
    #[test]
    fn a_call_is_read_past_its_thinking() {
        let call = "<tool_call>\n{\"name\": \"reading\", \"arguments\": {}}\n</tool_call>";
        assert_eq!(
            past_thinking(&format!(
                "\nThe {{draft}} is set in 2805.\n</think>\n\n{call}"
            )),
            call
        );
        assert_eq!(past_thinking(call), call);
    }

    /// The shared prefix is measured in whole characters.
    #[test]
    fn the_common_prefix_never_splits_a_character() {
        assert_eq!(common_prefix_len("café", "cafè"), 3);
        assert_eq!(common_prefix_len("日本", "日本語"), "日本".len());
    }

    /// The reasoning close the grammar injects is scaffolding, not prose to
    /// show.
    #[test]
    fn a_leading_reasoning_close_is_not_shown() {
        assert_eq!(visible("</think>\n\nThe yard"), "The yard");
        assert_eq!(visible("\nThe yard"), "The yard");
        assert_eq!(visible("The yard"), "The yard");
    }
}
