//! Builds the part of a character's turn that answers its last acts and
//! narrates what has since happened.
//!
//! The chat template returns a call's result in the user half of the next turn,
//! so the answers are written first, one block per call in the order the
//! character made them, and the world's narration follows as plain prose. A
//! refusal is returned like any other answer, since it is what separates a
//! failed act from one that did nothing. Each answer is bounded in characters
//! so no single result can grow a turn past what its card can hold. The window
//! of earlier turns is not repeated here, because the sequence already carries
//! it. The per-layer recall counts report what each shared layer contributed to
//! a turn's projection.

use super::*;

impl Minds {
    /// Every layer of the mind's projection, with how much of each this
    /// character can read. `None` before the projection is loaded.
    pub fn layer_counts(&self, npc_id: u64) -> Option<Vec<layers::LayerCount>> {
        let (builder, live) = self.as_character(npc_id)?;
        Some(layers::counts(&self.engine, &builder, live, npc_id))
    }

    /// The newest `limit` conversations of one layer this character can read.
    /// `None` for a layer the projection does not declare, or before it is
    /// loaded.
    pub fn layer_page(&self, npc_id: u64, layer: &str, limit: usize) -> Option<layers::Page> {
        let (builder, live) = self.as_character(npc_id)?;
        layers::page(&self.engine, &builder, live, npc_id, layer, limit)
    }

    /// Narrate the character's **own** act, for the tool response it reads back
    /// and the feed the GUI shows — in place of the bare confirmation the world
    /// gives an intent-carrying act ("You tell Pax."). Only the acts
    /// [`tools::narrates`] names come through here; anything with a real, fixed
    /// reply keeps it.
    ///
    /// Runs on the character's own narrator thread, so what it did is voiced in
    /// the same running narrative as the scene around it. `None` when the act
    /// carries no intent to render, or the character has no live conversation.
    pub async fn narrate_act(
        &self,
        npc_id: u64,
        persona: &Persona<'_>,
        act: &act::Act,
        departing: bool,
    ) -> anyhow::Result<Option<String>> {
        let Some(line) = own_act_line(act, departing) else {
            return Ok(None);
        };
        // A character the daemon no longer holds narrates nothing.
        if !self.live.pin().contains_key(&npc_id) {
            return Ok(None);
        }
        // Whoever the act is aimed at, so the narration has them present.
        let near: Vec<String> = ["to", "on"]
            .iter()
            .filter_map(|k| act.args.get(*k).and_then(|v| v.as_str()))
            .map(str::to_string)
            .collect();
        let turn = narrator::build_act_turn(
            persona.name,
            &narrator::sketch(persona.identity, persona.manner),
            &near,
            &line,
        );
        let narration = narration::render(&turn);
        Ok((!narration.is_empty()).then_some(narration))
    }
}

/// What the character reads this tick: the answers to what it did, then what
/// has happened since.
///
/// # The shape, and why it is this shape
///
/// A character calls acts and the world answers them. The template that answer
/// arrives in is not ours to choose — Qwen and Hermes both return a result in
/// the **user** half of the next turn, wrapped in `<tool_response>`, one block
/// per call, and the model is trained to read that wrapper as "this is what
/// came back" rather than as something a person said.
///
/// So the turn is built in two parts, in this order:
///
/// ```text
/// <tool_response>
/// You moved to the sorting room.
/// </tool_response>
///
/// The air near the door is noticeably fresher than the air by the wall.
/// ```
///
/// The results come **first** because they answer the turn before, and the
/// world's own events follow as ordinary prose. Reversing them would put the
/// world's voice between a call and its answer, which is the one arrangement
/// every tool-calling protocol forbids.
///
/// Before this, results were never returned at all: measured over 23 turns of
/// three characters, the substrate held 23 calls and zero responses. A
/// character acted and the next thing it read was the weather. Every one of
/// those 23 acts was `reflect` — with no act ever visibly causing anything,
/// there was nothing to prefer about acting over thinking.
///
/// # The world half is now curated prose
///
/// The events are no longer rendered one-per-line here — they are decoded into a
/// single third-person passage by the character's narrator thread (see
/// [`crate::engine::narrator`]) and handed in as `narration`. This function only
/// assembles the turn: the answers as `<tool_response>` blocks, then that
/// passage. When there is nothing to narrate, `narration` is empty and the turn
/// is the answers alone.
///
/// The window is still not pasted in: the sequence carries its own bounded tail
/// (`context_window_turns`) and the substrate carries the rest, so repeating it
/// here would teach the model that everything happens twice.
/// The one-line description of a character's own act for the narrator, or `None`
/// when the act carries no intent to render.
///
/// The actor is the focal character, so it opens with `you —` and
/// [`narration::render`] turns that into "You tell Pax that …". The verb
/// names the act and its target; the intent is the substance to voice.
pub(super) fn own_act_line(act: &act::Act, departing: bool) -> Option<String> {
    let arg = |k: &str| {
        act.args
            .get(k)
            .and_then(|v| v.as_str())
            .map(str::trim)
            .filter(|s| !s.is_empty())
    };
    let intent = arg("intent").or_else(|| arg("about"))?;
    let to = arg("to").or_else(|| arg("on"));
    let verb = match act.tool {
        "tell" => format!("tell {}", to.unwrap_or("them")),
        "ask" => format!("ask {}", to.unwrap_or("them")),
        "whisper" => format!("whisper to {}", to.unwrap_or("them")),
        "shout" => "shout to the room".to_string(),
        "gesture" => match to {
            Some(t) => format!("gesture to {t}"),
            None => "gesture".to_string(),
        },
        "act" => format!("act on {}", to.unwrap_or("yourself")),
        other => other.to_string(),
    };
    // A line called after somebody leaving carries a marker the narrator turns
    // into the parting frame ("As X walks away, you tell them …"). It is a
    // prefix on the relation so the verb and target parse unchanged after it.
    let verb = if departing {
        format!("(leaving) {verb}")
    } else {
        verb
    };
    Some(format!("you — {verb} — meaning: \"{intent}\""))
}

/// What a turn's projection brought in from each layer beneath its own `own`
/// layer, dreams aside (they are reported on their own): per layer, how many
/// turns were selected, the best score among them, and the timeline that
/// scored it.
pub(super) fn recalled_by_layer(turns: &[SelectedTurn], own: &str) -> BTreeMap<String, Recalled> {
    let mut out: BTreeMap<String, Recalled> = BTreeMap::new();
    for t in turns
        .iter()
        .filter(|t| t.selected && t.layer != own && t.layer != dreams::LAYER)
    {
        let e = out.entry(t.layer.clone()).or_insert(Recalled {
            count: 0,
            best: f32::MIN,
            top: None,
        });
        e.count += 1;
        if t.score > e.best {
            e.best = t.score;
            e.top = t.timeline;
        }
    }
    out
}

/// What one layer brought into a turn — see [`recalled_by_layer`].
#[derive(Debug, PartialEq)]
pub(super) struct Recalled {
    pub(super) count: usize,
    pub(super) best: f32,
    /// The timeline of the best-scoring turn.
    pub(super) top: Option<u64>,
}

/// The longest one answer a character is handed, in characters: above a page
/// of any document read whole, and a fraction of what a turn can hold. One
/// table's refusal of a story pasted into itself ran to 412,000 characters, was
/// handed back on every turn after, and grew one Maker's turn to 2.2 GB of KV
/// on its card; what any one answer may carry is bounded here, whatever made
/// it.
pub(super) const ANSWER_MAX_CHARS: usize = 40_000;

/// `answer` as it is handed over: whole within [`ANSWER_MAX_CHARS`], cut at
/// a character boundary past it, saying how much was left out.
pub(super) fn bounded(answer: &str) -> String {
    let answer = answer.trim();
    match answer.char_indices().nth(ANSWER_MAX_CHARS) {
        None => answer.to_string(),
        Some((cut, _)) => {
            let left = answer[cut..].chars().count();
            format!(
                "{}\n[cut: {left} more characters of this answer are not shown]",
                &answer[..cut]
            )
        }
    }
}

pub(super) fn compose_narrated(answers: &[String], narration: &str) -> String {
    let mut s = String::new();
    for a in answers {
        // The block is newline-delimited inside the tags because an outcome is
        // a sentence, not a JSON scalar — the same allowance zend's
        // `format_tool_responses` makes for an already-rendered result.
        s.push_str("<tool_response>\n");
        s.push_str(&bounded(a));
        s.push_str("\n</tool_response>\n");
    }
    // One blank line between the answers and the narration, so the two are
    // visibly different kinds of thing rather than one run-on block.
    if !answers.is_empty() && !narration.is_empty() {
        s.push('\n');
    }
    // The narration carries the compass — a refused document's last refusal
    // among it — so it is bounded as an answer is.
    match narration.char_indices().nth(ANSWER_MAX_CHARS) {
        None => s.push_str(narration),
        Some(_) => s.push_str(&bounded(narration)),
    }
    s
}

/// Render a window turn for a transcript view.
pub fn render_turn(speaker: Speaker, text: &str) -> String {
    match speaker {
        Speaker::World => text.to_string(),
        Speaker::Npc => format!("→ {text}"),
    }
}
