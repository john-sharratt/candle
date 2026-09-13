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
//! # A fixed set of answers is a grammar, not a request
//!
//! `choices` compiles to a stencil whose root branches once per arm, so an answer
//! outside the set is unreachable rather than unlikely. The dialect's closed think
//! block is prefilled ahead of it — the acting turn's opening — so the grammar
//! governs the answer itself and the first token the model chooses is already one
//! of the arms.

use std::sync::{Arc, Mutex};

use candle_conversation::projection::{Builder, GroupId, LayerId, Reserved, SectionId};
use candle_conversation::stencil::{StencilTree, StencilTreeBuilder};
use candle_conversation::{
    ConversationEngine, Sequence, SequenceConfig, TokenDecoder, TurnEvent, TurnOptions,
    TurnResponse,
};

use crate::engine::reflect::{plain_prose, think_off};
use crate::prose::{Answer, Request, DEFAULT_SYSTEM};

/// Decode `request` with `seed`, handing each new fragment of text to
/// `on_fragment` as it lands.
///
/// Blocks for the length of the decode; [`crate::prose::run_streamed`] runs it
/// off the async pool.
pub fn decode(
    engine: &Arc<Mutex<ConversationEngine>>,
    base: &SequenceConfig,
    request: &Request,
    seed: u64,
    on_fragment: &mut dyn FnMut(&str),
) -> anyhow::Result<Answer> {
    let mut cfg = base.clone();
    // One turn with nothing before it: there is no history to carry.
    cfg.context_window_turns = 0;
    let system = match request.system.trim() {
        "" => DEFAULT_SYSTEM,
        _ => request.system.as_str(),
    };

    // The grammar and the opening, built before the conversation so a choice set
    // the vocabulary cannot compile costs nothing to refuse.
    let (turn_grammar, prefill) = match request.choices.as_deref() {
        Some(arms) if !arms.is_empty() => {
            let tree = choice_tree(&engine.lock().unwrap(), arms)?;
            (
                Some(Arc::new(tree)),
                Some(base.dialect.no_think_block.to_string()),
            )
        }
        _ => (think_off(engine, base), None),
    };

    let formatted = base.dialect.format_system_prompt(system);
    // Taken outside the engine lock below, so that when it is let go — at the
    // end of this function or on an early return — that lock is not held.
    let frame = HeldFrame {
        engine: Arc::clone(engine),
        id: engine.lock().unwrap().transient_prompt_section(system)?,
    };
    let (mut sequence, timeline, decoder) = {
        let e = engine.lock().unwrap();
        let (builder, layer, group) = prose_projection(system, frame.id)?;
        let seq = e.new_conversation_with_projection(&formatted, builder, layer, group, cfg)?;
        let tl = seq.timeline_id();
        e.mark_timeline_transient(tl);
        (seq, tl, e.token_decoder())
    };

    let mut sampling = base
        .sampling
        .clone()
        .with_seed(seed)
        .with_graceful_segment_close_after(0)
        .with_force_segment_close_after(1);
    if let Some(t) = request.temperature {
        sampling.temperature = t;
    }
    let options = TurnOptions {
        max_tokens: Some(request.max_tokens as usize),
        sampling: Some(sampling),
        turn_grammar,
        assistant_prefill: prefill.clone(),
        ..Default::default()
    };

    let decoded = run_turn(
        &mut sequence,
        &request.prompt,
        options,
        &decoder,
        on_fragment,
    );

    // Retired whatever happened: a job that failed half-way leaves nothing worth
    // keeping either.
    if let Ok(e) = engine.lock() {
        if let Err(err) = e.tombstone_timeline(timeline) {
            tracing::warn!("prose conversation {timeline} could not be retired: {err:?}");
        }
    }
    drop(sequence);

    let r = decoded?;
    // A prefilled block may come back at the head of the text. It was the
    // opening's scaffolding, not the answer.
    let text = match &prefill {
        Some(p) => r.text.strip_prefix(p.as_str()).unwrap_or(&r.text),
        None => r.text.as_str(),
    };
    Ok(Answer {
        text: plain_prose(text),
        tokens: r.stats.tokens_generated as u32,
        seed,
    })
}

/// A transient frame this job holds, released when the job is done with it.
///
/// On every path out, because a frame never released is a section never
/// retired. Declared ahead of the job's sequence, so the sequence — and the slot
/// reading the frame — is dropped first. See
/// [`ConversationEngine::transient_prompt_section`].
struct HeldFrame {
    engine: Arc<Mutex<ConversationEngine>>,
    id: SectionId,
}

impl Drop for HeldFrame {
    fn drop(&mut self) {
        if let Ok(e) = self.engine.lock() {
            e.release_prompt_section(self.id);
        }
    }
}

/// Submit the turn, stream its fragments, and seal it.
fn run_turn(
    sequence: &mut Sequence,
    prompt: &str,
    options: TurnOptions,
    decoder: &TokenDecoder,
    on_fragment: &mut dyn FnMut(&str),
) -> anyhow::Result<TurnResponse> {
    let handle = sequence.submit_turn_with_options(prompt, options)?;
    let mut ids: Vec<u32> = Vec::new();
    let mut preview = Preview::default();
    let mut response = None;
    for event in handle.stream() {
        match event {
            TurnEvent::Token(id) => {
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
    let Some(r) = response else {
        anyhow::bail!("the scheduler went away before the prose finished");
    };
    sequence.finish_turn(handle, &r)?;
    Ok(r)
}

/// The turn group of a plain-prompt schema — the one group every prose job's
/// conversation is written to.
const PROSE_GROUP: &str = "primary_conversation";

/// The tag no turn carries, which is what a closed group admits.
const CLOSED: &str = "prose:closed";

/// The projection a prose job runs under: its own reserved layer and group, that
/// group closed, and `frame` holding its system prompt.
///
/// **`frame` comes from the engine, resolved from the prompt's own text.** A
/// section the substrate already holds is reused by id without its text being
/// compared, so every job opened under one fixed frame id read whichever voice
/// had sealed there first. The first job after a start is the create form's
/// name, so every description after it was written under "reply with the name
/// and nothing else" — and came back as a name. The frame is transient, retired
/// once the job lets go of it — see
/// [`ConversationEngine::transient_prompt_section`].
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
fn prose_projection(system: &str, frame: SectionId) -> anyhow::Result<(Builder, LayerId, GroupId)> {
    let mut builder = Builder::for_plain_prompt_reserved(system, Reserved::Prose, frame);
    builder
        .set_group_tags(PROSE_GROUP, vec![CLOSED.to_string()])
        .map_err(|e| anyhow::anyhow!("closing the prose group: {e}"))?;
    Ok((
        builder,
        LayerId::reserved(Reserved::Prose),
        GroupId::reserved(Reserved::Prose),
    ))
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
            prose_projection("You name characters.", frame).expect("the group exists to be closed");
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
