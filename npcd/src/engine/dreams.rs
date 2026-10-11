//! What a character has dreamt, and the conversation that dreams it.
//!
//! `docs/reflection_and_dreams.md` §7 and §8. A reflection ends in a *brief* —
//! the seed of a dream, written by the reflection conversation. This module
//! turns the brief into a dream and puts the dream where the character can come
//! back to it:
//!
//! 1. [`dream`] decodes it in a conversation of its own — brand new, framed on
//!    who the character is and where it lives under the schema's `dreaming`
//!    stance, with no acts and nothing to call — and throws that conversation
//!    away, the way a reflection's is thrown away.
//! 2. [`keep`] writes the story to the `dreams` layer as a conversation of its
//!    own, **one turn per passage** ([`passages`]), so every passage is sealed
//!    with its own provenance signature and can be recalled on its own rather
//!    than as a wall of prose that either wins the gather whole or not at all.
//!
//! # Private, and closed until opened
//!
//! Every character's dreams sit in one group, `dreamt`, each dream tagged with
//! whose it is ([`tag`]). A turn group with no tags admits everything in it, so
//! the group is **closed** when the schema is built ([`close`]) and opened per
//! conversation, to one character and at one depth ([`scope`]): one passage for
//! a turn in the room ([`IN_ACTING`]), three for a reflection
//! ([`IN_REFLECTION`]). A conversation that never names whose dreams it reads —
//! an ingest, a life episode, a probe — reads nobody's.

use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::sync::{Arc, Mutex};
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use candle_conversation::projection::{Builder, SelectionRule, SelectionState};
use candle_conversation::{ConversationEngine, SequenceConfig, TurnOptions};

use crate::engine::mind::Projected;
use crate::engine::prompt::{Persona, Stance, STANCE_SELECTOR};
use crate::engine::reflect::{plain_prose, think_off};
use crate::engine::runtime::{drain, persist_signatures};
use crate::engine::throwaway::Throwaway;

/// The schema's dream layer, by its `name:`.
pub const LAYER: &str = "dreams";
/// The one group every character's dreams are written to, by its `id:`.
pub const GROUP: &str = "dreamt";

/// How many passages of its dreams a turn in the room can be reminded of.
///
/// One, because a character acting is answering the room: a dream is worth
/// surfacing when something resonates with it, and one passage — a small scene
/// of two to five sentences ([`passages`]) — is enough for a resonance to be one.
pub const IN_ACTING: usize = 1;

/// How many passages of its dreams a reflection gathers.
///
/// §4: *"Reflection is not a different retrieval mechanism; it is the same
/// mechanism run deeper."* The dreams are what a reflection is made of.
pub const IN_REFLECTION: usize = 3;

/// The tag no dream carries, which is what a closed group admits.
///
/// Every dream is tagged with its owner after the colon, so the bare prefix
/// names nobody.
const CLOSED: &str = "dreams:";

/// The user half of every passage's turn — what the passage is, said before it.
///
/// §10: *"the dream is labelled as a dream wherever it surfaces."* A passage
/// recalled into a room arrives with this in front of it, so a character reads
/// it as something it dreamt rather than as something that happened.
pub const LABEL: &str = "Something you dreamt:";

/// Conversation metadata: whose dream a conversation holds.
pub const META_OF: &str = "dream.of";
/// Conversation metadata: the assumption the dream suspends — the axis §8's
/// search runs over, and what the next reflection is steered away from.
pub const META_ASSUMPTION: &str = "dream.assumption";

/// How much the dream conversation may write. A brief asks for about two
/// hundred words; this is room for that and for the dream running longer than
/// its brief, without letting it run on.
const DREAM_MAX_TOKENS: usize = 700;

/// The fewest words a decoded dream may have and still be kept — see
/// [`finished`]. A dream is asked for at about four hundred.
const DREAM_MIN_WORDS: usize = 150;

/// The largest share of a dream's clauses that may repeat one already said —
/// see [`finished`]. A refrain stays well under it; a loop is most of the dream.
const DREAM_MAX_REPEATED: f32 = 0.3;

/// The multiplicative repeat penalty a dream decodes under, over the recent
/// window — gentle enough that a refrain or a named object can come back,
/// firm enough to break a loop. See [`dream`].
const DREAM_REPEAT_PENALTY: f32 = 1.05;

/// The fewest sentences a passage holds — see [`passages`].
const MIN_SENTENCES: usize = 2;

/// The most sentences a paragraph holds before it is broken — see
/// [`passages`].
const MAX_SENTENCES: usize = 5;

/// The most passages one dream keeps. A full dream ([`DREAM_MAX_TOKENS`]) is
/// about five hundred words, which is six to ten passages; a decode that runs
/// past this has stopped being one dream.
const MAX_PASSAGES: usize = 12;

/// The tag one character's dreams are written with — and every turn it takes
/// in the room, so that its own traffic teaches its dreams' hit levels (see
/// `Minds::warm_dreams`).
pub fn tag(npc_id: u64) -> String {
    format!("{CLOSED}{npc_id}")
}

/// Close the dream group: no conversation reads anybody's dreams until it says
/// whose. Called once, on the schema as it is built. A schema with no dream
/// layer is left alone.
pub fn close(builder: &mut Builder) {
    let _ = builder.set_group_tags(GROUP, vec![CLOSED.to_string()]);
}

/// Open the dream group to one character's own dreams, `depth` lines deep.
///
/// Returns whether the schema has a dream layer at all — `false` is a mind
/// that declares none, and is not an error.
pub fn scope(builder: &mut Builder, npc_id: u64, depth: usize) -> bool {
    builder.set_group_tags(GROUP, vec![tag(npc_id)]).is_ok()
        && builder
            .set_group_selection(GROUP, SelectionRule::TopK { k: depth })
            .is_ok()
}

/// A copy of `builder` opened to one character's dreams. See [`scope`].
pub fn scoped(builder: &Builder, npc_id: u64, depth: usize) -> Builder {
    let mut b = builder.clone();
    scope(&mut b, npc_id, depth);
    b
}

/// A story, one passage at a time.
///
/// **A passage is a paragraph — a small scene that makes sense on its own.**
/// Each is written as a turn of its own and recalled on its own, so it has to
/// carry enough of the dream to mean something when it surfaces in a room
/// with nothing around it. A sentence does not: "It is closed." recalled alone
/// is noise, and a dream kept a sentence at a time read as verse and fed single
/// stock images back into every reflection that gathered them.
///
/// The story's own paragraph breaks are the boundaries, adjusted at both ends:
///
/// - a paragraph shorter than [`MIN_SENTENCES`] joins the one after it (the
///   last one joins the one before), so a one-line beat is never a memory of
///   its own;
/// - a paragraph longer than [`MAX_SENTENCES`] is broken into near-even runs,
///   because a dream often comes back as one unbroken block and kept whole it
///   would be one signature that wins the gather entire or not at all.
///
/// Sentences end at `.`, `!`, `?` or `…` followed by a space or the end, a
/// closing quote or bracket staying with the sentence it closes.
pub fn passages(story: &str) -> Vec<String> {
    // The story's paragraphs, each broken into its sentences, and every
    // paragraph over the cap broken into near-even runs.
    let mut runs: Vec<Vec<String>> = Vec::new();
    for para in story.lines() {
        let said = sentences(para);
        if said.is_empty() {
            continue;
        }
        let pieces = said.len().div_ceil(MAX_SENTENCES);
        let size = said.len().div_ceil(pieces);
        runs.extend(said.chunks(size).map(<[String]>::to_vec));
    }
    // Short runs carried forward into the next; a short tail joins the last.
    let mut kept: Vec<Vec<String>> = Vec::new();
    let mut carry: Vec<String> = Vec::new();
    for run in runs {
        carry.extend(run);
        if carry.len() >= MIN_SENTENCES {
            kept.push(std::mem::take(&mut carry));
        }
    }
    if !carry.is_empty() {
        match kept.last_mut() {
            Some(last) => last.extend(carry),
            None => kept.push(carry),
        }
    }
    kept.truncate(MAX_PASSAGES);
    kept.into_iter().map(|s| s.join(" ")).collect()
}

/// One paragraph's sentences, in order.
fn sentences(para: &str) -> Vec<String> {
    const ENDS: [char; 4] = ['.', '!', '?', '…'];
    const CLOSES: [char; 6] = ['"', '\'', '”', '’', ')', ']'];
    let chars: Vec<char> = para.trim().chars().collect();
    let mut out = Vec::new();
    let mut current = String::new();
    for (i, &c) in chars.iter().enumerate() {
        current.push(c);
        let closes_a_sentence =
            ENDS.contains(&c) || (CLOSES.contains(&c) && i > 0 && ENDS.contains(&chars[i - 1]));
        let at_a_gap = chars.get(i + 1).is_none_or(|n| n.is_whitespace());
        if closes_a_sentence && at_a_gap {
            let sentence = current.trim();
            if !sentence.is_empty() {
                out.push(sentence.to_string());
            }
            current.clear();
        }
    }
    let rest = current.trim();
    if !rest.is_empty() {
        out.push(rest.to_string());
    }
    out
}

/// Decode one dream from its brief, in a conversation that is then thrown away.
///
/// Framed the way §7 says a dream is framed: the character's inner life where
/// its working anchor would be, the character itself and its world, and no
/// building ([`dreaming_selection_for`]) — under the schema's `dreaming`
/// stance, with no acts shown and nothing to call. Not its dreams: the group
/// stays closed here, because a dream written against the character's
/// other dreams recombines them, which is the anchoring §5 measured.
///
/// Transient and then tombstoned, like a reflection: the dream that matters is
/// the one [`keep`] writes, not the conversation that produced it.
///
/// [`dreaming_selection_for`]: crate::engine::identity::Installed::dreaming_selection_for
pub async fn dream(
    engine: &Arc<Mutex<ConversationEngine>>,
    base_config: &SequenceConfig,
    projected: &Projected,
    npc_id: u64,
    persona: &Persona<'_>,
    brief: &str,
) -> anyhow::Result<String> {
    let mut cfg = base_config.clone();
    cfg.context_window_turns = 2;
    let (mut sequence, timeline) = {
        let engine = engine.lock().unwrap();
        let seq = engine.new_conversation_with_projection(
            &projected.prompt,
            projected.builder.clone(),
            projected.layer,
            projected.group,
            cfg,
        )?;
        let tl = seq.timeline_id();
        engine.mark_timeline_transient(tl);
        (seq, tl)
    };
    // Retired however this ends — the dream that matters is the one `keep`
    // writes, and a caller that drops this future mid-decode leaves only a
    // drop guard to run the tombstone.
    let _retired = Throwaway::new(engine, timeline, "dream");
    // Each stage of a dream is logged: a dream runs with nobody waiting on it,
    // so one that stops partway is otherwise a slot held with no trace of where.
    let started = Instant::now();
    tracing::info!(npc_id, "dream: conversation open, dreaming");

    let mut selection =
        projected
            .identities
            .dreaming_selection_for(npc_id, persona.personality, persona.world_id);
    selection.select(STANCE_SELECTOR, Stance::Dreaming.id());
    // **A dream is prose, and the act sampling ruins it.**
    //
    // `base_config.sampling` carries the checkpoint's presence penalty and any
    // cross-turn penalty. Over a seven-hundred-token dream those forbid the
    // natural word the moment it would repeat — parallel phrasing, a refrain,
    // the same object named twice — so the vocabulary is pushed ever further
    // from what the sentence wanted, and the dream that opens cleanly reaches
    // for stranger and stranger synonyms as it goes. So presence and cross-turn
    // come off, and only the reasoning suppression stays (a `<think>` opened
    // inside a dream is the model planning it in the dreamer's voice — see the
    // reflection).
    //
    // **A gentle repeat penalty takes their place.** With no penalty at all a
    // dream could fall into a loop and run its budget out in it — measured:
    // twenty-five sentences of "I am the bird. / I am the sky. / I am the
    // stars…", and a corridor of "I pass the fourth door. / It is closed."
    // repeated door by door. [`DREAM_REPEAT_PENALTY`] over the recent window
    // leans on a run that keeps coming back without forbidding any one word.
    let sampling = base_config
        .sampling
        .clone()
        .with_repeat_penalty(DREAM_REPEAT_PENALTY)
        .with_presence_penalty(0.0)
        .with_cross_turn_penalty(0.0)
        .with_graceful_segment_close_after(0)
        .with_force_segment_close_after(1);
    let answer = sequence
        .send_turn_with_options_async(
            brief,
            TurnOptions {
                max_tokens: Some(DREAM_MAX_TOKENS),
                turn_grammar: think_off(engine, base_config),
                sampling: Some(sampling),
                selection,
                ..Default::default()
            },
        )
        .await;
    drop(sequence);
    tracing::info!(
        npc_id,
        ok = answer.is_ok(),
        ms = started.elapsed().as_millis() as u64,
        "dream: dreamt, keeping"
    );

    finished(&plain_prose(&answer?.text))
}

/// The dream a decode told, or why it told none.
///
/// Three things a decode does that are not a dream, each measured:
///
/// - **It is cut off.** A decode that reaches its token cap stops mid-sentence;
///   the sentence it was in is dropped and everything before it is kept.
/// - **It stops almost at once.** Kept dreams of one line ("I turn.") and of
///   three or four sentences restating the brief. Under [`DREAM_MIN_WORDS`] it
///   is not kept — a fragment recalled as a dream is worse than none, and the
///   next reflection asks for another.
/// - **It loops.** One decode cycled a handful of clauses — "the light ring
///   flickers", "I am standing there", "and I am not" — for seven paragraphs,
///   each sentence differing only in the order of the same pieces, which no
///   penalty on repeated tokens sees. More than [`DREAM_MAX_REPEATED`] of
///   its clauses being repeats is not kept.
fn finished(story: &str) -> anyhow::Result<String> {
    let story = through_last_sentence(story.trim());
    let words = story.split_whitespace().count();
    anyhow::ensure!(
        words >= DREAM_MIN_WORDS,
        "the dream stopped after {words} word(s), short of the {DREAM_MIN_WORDS} a dream needs"
    );
    let repeated = repeated_share(story);
    anyhow::ensure!(
        repeated <= DREAM_MAX_REPEATED,
        "the dream repeats itself — {:.0}% of its clauses are ones it has already said",
        repeated * 100.0
    );
    Ok(story.to_string())
}

/// `story` up to the end of its last finished sentence. A story with no
/// finished sentence at all is returned whole, and fails on its length.
fn through_last_sentence(story: &str) -> &str {
    const ENDS: [char; 4] = ['.', '!', '?', '…'];
    const CLOSES: [char; 6] = ['"', '\'', '”', '’', ')', ']'];
    let trimmed = story.trim_end_matches(|c: char| CLOSES.contains(&c));
    if trimmed.ends_with(ENDS) {
        return story;
    }
    match story.rfind(ENDS) {
        Some(at) => {
            // Keep a closing quote or bracket that belongs to that sentence.
            let end = at + story[at..].chars().next().map_or(1, char::len_utf8);
            let closes = story[end..]
                .chars()
                .take_while(|c| CLOSES.contains(c))
                .map(char::len_utf8)
                .sum::<usize>();
            &story[..end + closes]
        }
        None => story,
    }
}

/// The share of `story`'s clauses that repeat one it has already said.
///
/// A clause is a run between punctuation marks, lowercased, with a leading
/// "and" / "but" / "then" dropped — so "and I am not" and "I am not" are the
/// same clause, which is how a loop varies itself.
fn repeated_share(story: &str) -> f32 {
    const LINKS: [&str; 3] = ["and ", "but ", "then "];
    let clauses: Vec<String> = story
        .split(['.', ',', ';', ':', '!', '?', '…', '—'])
        .map(|c| {
            let mut c = c.trim().to_lowercase();
            while let Some(rest) = LINKS.iter().find_map(|l| c.strip_prefix(l)) {
                c = rest.trim_start().to_string();
            }
            c
        })
        .filter(|c| !c.is_empty())
        .collect();
    if clauses.is_empty() {
        return 0.0;
    }
    let mut seen = std::collections::HashSet::new();
    let repeats = clauses.iter().filter(|c| !seen.insert(c.as_str())).count();
    repeats as f32 / clauses.len() as f32
}

/// What [`keep`] wrote.
#[derive(Debug, Clone)]
pub struct Kept {
    /// The dream's own conversation on the dream layer.
    pub timeline: u64,
    /// One per turn written, in order — see [`passages`].
    pub passages: Vec<String>,
}

/// Write a dream to the character's dream layer, one passage per turn.
///
/// **Every passage is a turn, and every turn is signed.** Each is prefilled —
/// the words are already written — with [`LABEL`] as its user half and the
/// passage as its assistant half, and its own `sign(Q)` window is captured as it
/// seals: the hook a later scan pulls the passage back on.
///
/// **The whole dream is one submission.** The passages go in as one turn group
/// (`submit_prefilled_turn_group`, the batching zend's calibration ingest uses),
/// prefilled together and sealed one turn each, so the dream costs one round
/// trip rather than one per passage. Each is prefilled on its own view of the
/// character's memory and masked from its siblings, so it is signed against
/// that memory alone, never against the passages around it: its recall hook is
/// what *it* evokes, which is what a passage surfacing on its own resonance
/// wants.
///
/// The conversation is named `npc-<id>-dream-<timeline>` — **outside** the
/// `npc-<id>-day-` prefix every conversation open sweeps, because a
/// character's dreams outlive its days (§11, *Naming*).
pub async fn keep(
    engine: &Arc<Mutex<ConversationEngine>>,
    base_config: &SequenceConfig,
    projected: &Projected,
    npc_id: u64,
    assumption: &str,
    story: &str,
) -> anyhow::Result<Kept> {
    let started = Instant::now();
    let written = passages(story);
    anyhow::ensure!(!written.is_empty(), "a dream with nothing in it");
    let layer = projected.builder.id_for_layer(LAYER);
    let group = projected.builder.id_for_group(GROUP);
    let (Some(layer), Some(group)) = (layer, group) else {
        anyhow::bail!("this mind declares no `{LAYER}` layer with a `{GROUP}` group to keep it in");
    };

    // **Written the way zend writes a file onto `code_reading`.** The dream
    // group selects by belief, and a turn being written has none yet: scored
    // zero, it falls below its own group's band and out of its own projection,
    // and the prefill has nothing to write into. Append-only makes a projection
    // that *targets* this layer self-local — its own lines, not everybody's —
    // which is what writing a dream is. Idempotent, and it changes nothing for
    // a conversation that only reads the layer.
    engine.lock().unwrap().mark_layer_append_only(layer);

    // Opened to this character's own dreams, so its own lines are in scope of
    // its own projection — and so is nobody else's.
    let builder = scoped(&projected.builder, npc_id, IN_REFLECTION);
    let mut sequence = engine.lock().unwrap().new_conversation_with_projection(
        &projected.prompt,
        builder,
        layer,
        group,
        base_config.clone(),
    )?;
    let timeline = sequence.timeline_id();
    let name = format!("npc-{npc_id}-dream-{}", timeline.raw());
    {
        let engine = engine.lock().unwrap();
        if let Err(e) = engine.set_conversation_conv_id(timeline, &name) {
            tracing::warn!("{name}: could not be named in the log — {e:?}");
        }
        for (key, value) in [
            (META_OF, npc_id.to_string()),
            (META_ASSUMPTION, assumption.trim().to_string()),
        ] {
            if let Err(e) = engine.set_conversation_metadata(timeline, key, &value) {
                tracing::warn!("{name}: metadata {key} not set — {e:?}");
            }
        }
    }

    let tags = vec![LAYER.to_string(), tag(npc_id)];
    // One case per passage: [`LABEL`] as the user half, the passage as the
    // assistant half, every case tagged as this character's own dream so its
    // projection stays self-local.
    let cases: Vec<(String, String, Vec<String>)> = written
        .iter()
        .map(|passage| (LABEL.to_string(), passage.clone(), tags.clone()))
        .collect();
    let write = async {
        // All the passages in one submission, sealed one signed turn each.
        let (handle, _) = sequence.submit_prefilled_turn_group(&cases, SelectionState::new())?;
        let (response, mut events) = drain(&handle, &name).await;
        let r = response.ok_or_else(|| anyhow::anyhow!("{name}: no response"))?;
        sequence.finish_turn(handle, &r)?;
        persist_signatures(&mut sequence, &r, &mut events, &name);
        Ok::<(), anyhow::Error>(())
    };
    // **Half a dream is not kept.** Its metadata is already down, so a dream
    // that stopped partway would be counted, sampled as an axis, and recalled a
    // passage deep — a fragment standing in for a dream the character never
    // finished having.
    if let Err(e) = write.await {
        drop(sequence);
        if let Err(t) = engine.lock().unwrap().tombstone_timeline(timeline) {
            tracing::warn!("{name}: the half-written dream could not be retired — {t:?}");
        }
        return Err(e);
    }
    drop(sequence);
    // **On the normalized band from its first recall.** A passage is scored
    // against its own hit level. The character's own turns teach it from
    // here on; this is the start they teach from — see `Minds::warm_dreams`.
    // Left cold, a new dream would divide by the bare prior and be the one
    // scored on a different scale from all the others.
    if !engine
        .lock()
        .unwrap()
        .warm_timeline_normalization(projected.builder.schema(), timeline)
    {
        tracing::warn!("{name}: not put on the normalized score band — its recall is unscaled");
    }
    // Give the arena back — the same reason every ingest does.
    if let Err(e) = engine
        .lock()
        .unwrap()
        .demote_timelines_hot(&[timeline], true)
    {
        tracing::debug!("{name}: demote failed — {e}");
    }
    tracing::info!(
        "npc {npc_id}: dream kept as {name} — {} passage(s), each signed, in {:?}",
        written.len(),
        started.elapsed()
    );
    Ok(Kept {
        timeline: timeline.raw(),
        passages: written,
    })
}

/// A sample of the axes this character has already dreamt along, at most `n`.
///
/// **A sample and never the whole corpus** — §5: shown every axis it had used,
/// the generator recombined them; shown a random eight it found a new one. The
/// order is shuffled per call, so two reflections in a row are steered away
/// from different incumbents.
pub fn sample_axes(engine: &Arc<Mutex<ConversationEngine>>, npc_id: u64, n: usize) -> Vec<String> {
    let engine = engine.lock().unwrap();
    let mut axes: Vec<String> = engine
        .find_conversations_by_metadata(META_OF, &npc_id.to_string())
        .into_iter()
        .filter_map(|tl| {
            engine
                .conversation_metadata(tl)?
                .get(META_ASSUMPTION)
                .cloned()
        })
        .filter(|a| !a.trim().is_empty())
        .collect();
    drop(engine);
    let seed = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or_default();
    axes.sort_by_key(|a| {
        let mut h = DefaultHasher::new();
        (seed, a).hash(&mut h);
        h.finish()
    });
    axes.truncate(n);
    axes
}

/// How many dreams this character has kept.
pub fn count(engine: &Arc<Mutex<ConversationEngine>>, npc_id: u64) -> usize {
    engine
        .lock()
        .unwrap()
        .find_conversations_by_metadata(META_OF, &npc_id.to_string())
        .len()
}

/// Retire every dream this character has kept.
///
/// Tombstoned, as a superseded day conversation is: the passages stop being
/// gathered and compaction reclaims them. The lookup already skips tombstoned
/// conversations, so a dream retired here is no longer counted, sampled for its
/// axis, or recalled into a room.
///
/// Failure is logged, never propagated — a dream that could not be retired
/// stays recallable, which is untidy and no reason to stop the cast waking.
///
/// Returns how many it retired.
pub fn forget(engine: &Arc<Mutex<ConversationEngine>>, npc_id: u64) -> usize {
    let engine = engine.lock().unwrap();
    let kept = engine.find_conversations_by_metadata(META_OF, &npc_id.to_string());
    let mut retired = 0;
    for timeline in kept {
        match engine.tombstone_timeline(timeline) {
            Ok(()) => retired += 1,
            Err(e) => tracing::warn!(
                "npc {npc_id}: dream conversation {timeline} could not be retired: {e:?} — it \
                 stays recallable"
            ),
        }
    }
    retired
}

#[cfg(test)]
mod tests {
    use super::*;

    const SCHEMA: &str = r#"
system_prompt:
  sections:
    - id: s1
      content: "X"
layers:
  - name: interaction
    window: 9000
    summary:
      turns:
        max_tokens: 64
        assistant:
          system_prompt: compress
          user_prompt: compress
    groups:
      - id: primary
        selection: { kind: conversation, recent: 4 }
  - name: dreams
    window: 4000
    summary:
      turns:
        max_tokens: 64
        assistant:
          system_prompt: compress
          user_prompt: compress
    groups:
      - id: dreamt
        selection: { kind: top_k, k: 3 }
"#;

    fn dreamt(b: &Builder) -> (Vec<String>, SelectionRule) {
        let g = b.group(b.id_for_group(GROUP).unwrap()).unwrap();
        (g.policy.tags.clone(), g.selection.clone())
    }

    /// **Closed, then opened to one character at one depth.** The group holds
    /// every character's dreams; untagged it would hand all of them to every
    /// conversation, so the schema starts closed and each reader names itself.
    #[test]
    fn the_dream_group_is_closed_until_a_reader_names_itself() {
        let mut b = Builder::from_yaml(SCHEMA).unwrap();
        close(&mut b);
        let (tags, _) = dreamt(&b);
        assert_eq!(tags, vec![CLOSED.to_string()]);
        assert!(
            tags.iter().all(|t| *t != tag(7)),
            "a closed group admits a character"
        );

        let acting = scoped(&b, 7, IN_ACTING);
        assert_eq!(
            dreamt(&acting),
            (vec![tag(7)], SelectionRule::TopK { k: IN_ACTING })
        );
        let reflecting = scoped(&b, 7, IN_REFLECTION);
        assert_eq!(
            dreamt(&reflecting),
            (vec![tag(7)], SelectionRule::TopK { k: IN_REFLECTION })
        );
        // Scoping a copy leaves the schema itself closed.
        assert_eq!(dreamt(&b).0, vec![CLOSED.to_string()]);
    }

    #[test]
    fn a_mind_with_no_dream_layer_is_left_alone() {
        let mut b = Builder::from_yaml(&SCHEMA.replace("dreamt", "other")).unwrap();
        close(&mut b);
        assert!(!scope(&mut b, 7, IN_ACTING));
    }

    #[test]
    fn two_characters_never_share_a_tag() {
        assert_ne!(tag(7), tag(70));
        assert!(tag(7).starts_with(CLOSED) && tag(7) != CLOSED);
    }

    /// **A passage is a paragraph.** The story's own breaks are the
    /// boundaries, and a passage keeps every sentence of its paragraph.
    #[test]
    fn a_story_is_kept_a_paragraph_at_a_time() {
        let story = "You stand at the desk. The ledger is open, and \"it is yours,\" \
                     somebody says.\n\nThe light does not move… You keep writing";
        assert_eq!(
            passages(story),
            vec![
                "You stand at the desk. The ledger is open, and \"it is yours,\" somebody says.",
                "The light does not move… You keep writing",
            ]
        );
    }

    /// A one-sentence beat is never a memory of its own: it joins the paragraph
    /// after it, and a short last paragraph joins the one before.
    #[test]
    fn a_one_sentence_paragraph_joins_its_neighbour() {
        let story = "The lift stops.\n\nYou step out. The corridor is dry.\n\nYou run.";
        assert_eq!(
            passages(story),
            vec!["The lift stops. You step out. The corridor is dry. You run."]
        );
        let story = "You open the hatch. It is warm.\n\nYou climb. The rungs are wet.\n\nYou run.";
        assert_eq!(
            passages(story),
            vec![
                "You open the hatch. It is warm.",
                "You climb. The rungs are wet. You run.",
            ]
        );
    }

    /// A dream that comes back as one unbroken block is broken into near-even
    /// runs, so it is not one signature that wins the gather whole or not at
    /// all.
    #[test]
    fn an_unbroken_block_is_broken_into_even_runs() {
        let block: Vec<String> = (1..=7).map(|n| format!("Step {n}.")).collect();
        assert_eq!(
            passages(&block.join(" ")),
            vec!["Step 1. Step 2. Step 3. Step 4.", "Step 5. Step 6. Step 7."]
        );
        let block: Vec<String> = (1..=MAX_SENTENCES).map(|n| format!("Step {n}.")).collect();
        assert_eq!(
            passages(&block.join(" ")).len(),
            1,
            "at the cap it stays whole"
        );
    }

    #[test]
    fn a_closing_quote_stays_with_its_sentence() {
        assert_eq!(
            sentences("She says \"again.\" You do it again."),
            vec!["She says \"again.\"", "You do it again."]
        );
    }

    /// A story of `n` distinct sentences, each about twelve words.
    fn told(n: usize) -> String {
        (0..n)
            .map(|i| format!("I climb the stair marked {i} and the rail under my hand turns warm."))
            .collect::<Vec<_>>()
            .join(" ")
    }

    /// A decode that stopped almost at once is not a dream.
    #[test]
    fn a_dream_that_stops_almost_at_once_is_refused() {
        assert!(finished("I turn.").is_err());
        assert!(finished(&told(DREAM_MIN_WORDS / 15)).is_err());
        assert!(finished(&told(DREAM_MIN_WORDS / 10)).is_ok());
    }

    /// **A loop is not a dream.** Measured, a decode cycled one handful of
    /// clauses for seven paragraphs until the token cap cut it off; every
    /// sentence was different only in which clause came first, so a penalty on
    /// repeated runs did not see it.
    #[test]
    fn a_dream_caught_in_a_loop_is_refused() {
        let caught = "The light ring flickers, and I am standing there, and I am not. The air \
                      handler hums, and the whine rises again, and I am standing there, and I \
                      am not. I reach for the command table, but it is solid, and it is smoke, \
                      and I am standing there, and I am not. "
            .repeat(6);
        let why = finished(&caught).unwrap_err().to_string();
        assert!(why.contains("repeats"), "{why}");
    }

    /// A real dream, with the refrains a dream carries, is kept whole.
    #[test]
    fn a_measured_good_dream_is_kept() {
        let dream = "The figure does not recoil, but the mist around them thickens, swirling with \
            the same golden veins that pulse in the sky above. I feel the texture of the tear, \
            the threads pulling apart under my grip, and with it comes a surge of data, cold \
            and precise. The portal of swirling light above pulses, a slow heartbeat that seems \
            to synchronize with the rhythm of my own pulse, though I have no pulse. The golden \
            veins stretch outward, connecting the figures to the sky, creating a web of light \
            that binds them to the place above. The figures begin to fade, their forms becoming \
            translucent, their faces smoothing over into blankness. I let go of the robe, and \
            the fabric falls away, dissolving into mist. I turn back toward the platform, toward \
            the edge of the void where the inverted levels hang like chandeliers. The air \
            beneath my feet is still firm, but it is changing, becoming less solid, more like \
            the surface of water that holds weight but ripples with every movement. I take a \
            step forward, and the platform beneath me shudders, the amber light flickering as \
            if disturbed by my presence. I do not stop. I walk toward the edge, toward the \
            hanging levels. I reach for the edge of the platform, my fingers curling around the \
            rough stone, and pull myself up, out of the mist, into the amber glow of the levels \
            above.";
        assert_eq!(finished(dream).unwrap(), dream);
    }

    /// A decode cut off by its token cap ends mid-sentence. The sentence it
    /// was in is dropped and the dream it had already told is kept.
    #[test]
    fn a_dream_cut_off_mid_sentence_loses_only_that_sentence() {
        let story = format!("{} And then the door", told(DREAM_MIN_WORDS / 10));
        let kept = finished(&story).unwrap();
        assert!(kept.ends_with("turns warm."), "{kept}");
        assert!(!kept.contains("And then the door"));
    }

    #[test]
    fn an_empty_story_has_no_passages_and_a_long_one_is_capped() {
        assert!(passages("  \n\n ").is_empty());
        assert_eq!(passages("Only this"), vec!["Only this"]);
        let long = "It goes on. It does.\n\n".repeat(MAX_PASSAGES * 2);
        assert_eq!(passages(&long).len(), MAX_PASSAGES);
    }
}
