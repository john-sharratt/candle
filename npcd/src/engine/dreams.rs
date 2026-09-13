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
//!    own, **one turn per line**, so every line is sealed with its own
//!    provenance signature and can be recalled on its own rather than as a wall
//!    of prose that either wins the gather whole or not at all.
//!
//! # Private, and closed until opened
//!
//! Every character's dreams sit in one group, `dreamt`, each dream tagged with
//! whose it is ([`tag`]). A turn group with no tags admits everything in it, so
//! the group is **closed** when the schema is built ([`close`]) and opened per
//! conversation, to one character and at one depth ([`scope`]): three lines for
//! a turn in the room ([`IN_ACTING`]), eight for a reflection
//! ([`IN_REFLECTION`]). A conversation that never names whose dreams it reads —
//! an ingest, a life episode, a probe — reads nobody's.

use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::sync::{Arc, Mutex};
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use candle_conversation::projection::{Builder, SelectionRule, SelectionState};
use candle_conversation::{ConversationEngine, SequenceConfig, TurnOptions};

use crate::engine::identity;
use crate::engine::mind::Projected;
use crate::engine::prompt::{Persona, Stance, STANCE_SELECTOR};
use crate::engine::reflect::{plain_prose, think_off};
use crate::engine::runtime::{drain, persist_signatures, PROJECTION_MARKER};
use crate::engine::throwaway::Throwaway;

/// The schema's dream layer, by its `name:`.
pub const LAYER: &str = "dreams";
/// The one group every character's dreams are written to, by its `id:`.
pub const GROUP: &str = "dreamt";

/// How many lines of its dreams a turn in the room can be reminded of.
///
/// Few, because a character acting is answering the room: a dream is worth
/// surfacing when something resonates with it, and three lines is enough for a
/// resonance to be one.
pub const IN_ACTING: usize = 3;

/// How many lines of its dreams a reflection gathers.
///
/// §4: *"Reflection is not a different retrieval mechanism; it is the same
/// mechanism run deeper."* The dreams are what a reflection is made of.
pub const IN_REFLECTION: usize = 8;

/// The tag no dream carries, which is what a closed group admits.
///
/// Every dream is tagged with its owner after the colon, so the bare prefix
/// names nobody.
const CLOSED: &str = "dreams:";

/// The user half of every line's turn — what the line is, said before it.
///
/// §10: *"the dream is labelled as a dream wherever it surfaces."* A line
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

/// The most lines one dream keeps. A decode that runs past this has stopped
/// being one dream.
const MAX_LINES: usize = 48;

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

/// A story, one line at a time.
///
/// A line is a sentence: a paragraph is broken after every `.`, `!`, `?` or
/// `…` that is followed by a space or the end — a closing quote or bracket
/// staying with the sentence it closes. A story of one long paragraph is how a
/// dream usually comes back, so breaking only on newlines would keep it as one
/// line and one signature, which is the thing this exists not to do.
pub fn lines(story: &str) -> Vec<String> {
    const ENDS: [char; 4] = ['.', '!', '?', '…'];
    const CLOSES: [char; 6] = ['"', '\'', '”', '’', ')', ']'];
    let mut out = Vec::new();
    for para in story.lines() {
        let chars: Vec<char> = para.trim().chars().collect();
        let mut current = String::new();
        for (i, &c) in chars.iter().enumerate() {
            current.push(c);
            let closes_a_sentence =
                ENDS.contains(&c) || (CLOSES.contains(&c) && i > 0 && ENDS.contains(&chars[i - 1]));
            let at_a_gap = chars.get(i + 1).is_none_or(|n| n.is_whitespace());
            if closes_a_sentence && at_a_gap {
                let line = current.trim();
                if !line.is_empty() {
                    out.push(line.to_string());
                }
                current.clear();
            }
        }
        let rest = current.trim();
        if !rest.is_empty() {
            out.push(rest.to_string());
        }
    }
    out.truncate(MAX_LINES);
    out
}

/// Decode one dream from its brief, in a conversation that is then thrown away.
///
/// Framed the way §7 says a dream is framed: who the character is and where it
/// lives — the same identity members an acting turn pins — under the schema's
/// `dreaming` stance, with no acts shown and nothing to call. Not its dreams:
/// the group stays closed here, because a dream written against the character's
/// other dreams recombines them, which is the anchoring §5 measured.
///
/// Transient and then tombstoned, like a reflection: the dream that matters is
/// the one [`keep`] writes, not the conversation that produced it.
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

    let mut selection = projected.identities.selection_for(
        npc_id,
        persona.personality,
        persona.world_id,
        persona.building,
        identity::Deliberation::default(),
    );
    selection.select(STANCE_SELECTOR, Stance::Dreaming.id());
    // The reflection's reasoning suppression, for the same reason: a dream
    // wants no deliberation at all, and a `<think>` opened inside it would be
    // the model planning the dream in the voice of the dreamer.
    let sampling = base_config
        .sampling
        .clone()
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

    let story = plain_prose(&answer?.text);
    anyhow::ensure!(!story.trim().is_empty(), "the dream came back empty");
    Ok(story)
}

/// What [`keep`] wrote.
#[derive(Debug, Clone)]
pub struct Kept {
    /// The dream's own conversation on the dream layer.
    pub timeline: u64,
    /// One per turn written, in order.
    pub lines: Vec<String>,
}

/// Write a dream to the character's dream layer, one line per turn.
///
/// **Every line is a turn, and every turn is signed.** Each is prefilled — the
/// words are already written — with [`LABEL`] as its user half, and its
/// projection is persisted as it seals: what the gather selected out of the
/// substrate at that line, which is the hook a later scan pulls the line back
/// on. The lines go in order into one conversation, so each is sealed after the
/// ones before it and its signature is taken with them in view.
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
    let written = lines(story);
    anyhow::ensure!(!written.is_empty(), "a dream with no lines in it");
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

    let mut cfg = base_config.clone();
    // Every earlier line in view of the next one: the dream is one piece, and
    // a line signed without the lines before it is a line out of context.
    cfg.context_window_turns = MAX_LINES;
    // Opened to this character's own dreams, so its own lines are in scope of
    // its own projection — and so is nobody else's.
    let builder = scoped(&projected.builder, npc_id, IN_REFLECTION);
    let mut sequence = engine.lock().unwrap().new_conversation_with_projection(
        &projected.prompt,
        builder,
        layer,
        group,
        cfg,
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
    let write = async {
        for (i, line) in written.iter().enumerate() {
            let addr = format!("{name}#{i}");
            let handle = sequence.submit_prefilled_turn(
                LABEL,
                line,
                PROJECTION_MARKER,
                SelectionState::new(),
                tags.clone(),
            )?;
            let (response, mut events) = drain(&handle, &addr).await;
            let r = response.ok_or_else(|| anyhow::anyhow!("{addr}: no response"))?;
            sequence.finish_turn(handle, &r)?;
            persist_signatures(&mut sequence, &r, &mut events, &addr);
        }
        Ok::<(), anyhow::Error>(())
    };
    // **Half a dream is not kept.** Its metadata is already down, so a dream
    // that stopped partway would be counted, sampled as an axis, and recalled a
    // few lines deep — a fragment standing in for a dream the character never
    // finished having.
    if let Err(e) = write.await {
        drop(sequence);
        if let Err(t) = engine.lock().unwrap().tombstone_timeline(timeline) {
            tracing::warn!("{name}: the half-written dream could not be retired — {t:?}");
        }
        return Err(e);
    }
    drop(sequence);
    // **On the normalized band from its first recall.** A line is scored
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
        "npc {npc_id}: dream kept as {name} — {} line(s), each signed, in {:?}",
        written.len(),
        started.elapsed()
    );
    Ok(Kept {
        timeline: timeline.raw(),
        lines: written,
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
/// Tombstoned, as a superseded day conversation is: the lines stop being
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

    /// **A line is a sentence.** A dream usually comes back as one paragraph,
    /// and breaking it only at newlines would keep it as one line with one
    /// signature — the thing a line per turn exists not to do.
    #[test]
    fn a_story_is_broken_into_its_sentences() {
        let story = "You stand at the desk. The ledger is open, and \"it is yours,\" \
                     somebody says. Is it?\n\nThe light does not move… You keep writing";
        assert_eq!(
            lines(story),
            vec![
                "You stand at the desk.",
                "The ledger is open, and \"it is yours,\" somebody says.",
                "Is it?",
                "The light does not move…",
                "You keep writing",
            ]
        );
    }

    #[test]
    fn a_closing_quote_stays_with_its_sentence() {
        assert_eq!(
            lines("She says \"again.\" You do it again."),
            vec!["She says \"again.\"", "You do it again."]
        );
    }

    #[test]
    fn an_empty_story_has_no_lines_and_a_long_one_is_capped() {
        assert!(lines("  \n\n ").is_empty());
        let long = "It goes on. ".repeat(MAX_LINES * 2);
        assert_eq!(lines(&long).len(), MAX_LINES);
    }
}
