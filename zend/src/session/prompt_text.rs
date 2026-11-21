//! Serves the text behind a conversation's prompt for the projection panel:
//! its sections, its turns and the framing around them.
//!
//! The panel shows what a conversation would be given now, so this text is
//! resolved on demand rather than stored in the projection event. Section text
//! comes from the schema, the conversation's own journal sections, and a tool
//! summary rebuilt from the catalog for its tools mode, so the panel matches
//! what startup sealed. Retiring the persisted system prompt makes the next
//! start prefill it afresh under the current compression thresholds, while the
//! running daemon keeps serving its resident copies.

use super::*;

impl ZendSession {
    /// `GET /v1/conversations/{id}/prompt-tokens`: the tokens the conversation
    /// would be given if it decoded now, piece by piece. `Some(None)` when it
    /// is not open on this member, or does not reproject; `None` while the model
    /// is not loaded. Waits for a round in flight to end.
    pub async fn prompt_dump(&self, conv_id: &str) -> Option<Option<PromptDump>> {
        let state = self.inference.load_full()?;
        let held = state.conversations.get(conv_id);
        let Some(conversation) = held else {
            return Some(None);
        };
        let cs = conversation.lock().await;
        Some(cs.conv.prompt_dump())
    }

    /// Retire the persisted copy of every section of the dialogue system
    /// prompt — the schema's sections (prelude, tool catalog, outro, every
    /// section-tree variant) and each tools mode's catalog summary — so the next
    /// start prefills them fresh under the current compression thresholds.
    ///
    /// Backs `DELETE /v1/substrate/zend/system-prompt`. The running daemon keeps
    /// serving from its resident copies
    /// ([`candle_conversation::Conversation::retire_section_generation`]); the
    /// rebuild happens on restart. Ingest frames are not touched: a sealed
    /// ingest turn was computed against its frame as stored.
    ///
    /// `None` when the model is not loaded.
    pub fn retire_system_prompt(&self) -> Option<candle::Result<SystemPromptRetired>> {
        let state = self.inference.load_full()?;
        let conv = { state.engine.conversation() };
        let summaries = ToolMode::ALL
            .into_iter()
            .filter_map(tool_summary_section)
            .map(SectionId::reserved)
            .chain(crate::tool_availability::line_sections().map(|(id, _)| id));
        let mut streams = HashSet::new();
        let mut unregistered = 0usize;
        for id in state
            .refresh_builder
            .schema()
            .system_prompt
            .all_section_ids()
            .chain(summaries)
        {
            match conv.section_stream(id) {
                Some(stream) => {
                    streams.insert(stream);
                }
                None => unregistered += 1,
            }
        }
        for &stream in &streams {
            if let Err(e) = conv.retire_section_generation(stream, "system prompt rebuild") {
                return Some(Err(e));
            }
        }
        tracing::info!(
            retired = streams.len(),
            unregistered,
            "system prompt sections retired; a restart prefills them fresh",
        );
        Some(Ok(SystemPromptRetired {
            retired_streams: streams.len(),
            unregistered_sections: unregistered,
        }))
    }

    /// `(section name, content)` for every system-prompt section in the schema,
    /// the journal sections this conversation carries, and the runtime tool
    /// summary matching its tools mode. Backs the projection panel's expandable
    /// section text — resolved on demand, never stored in the projection event.
    /// The schema is workspace-wide; `conv_id` selects the journal and which
    /// tool summary (restricted vs comprehensive) to serve.
    pub fn section_content(&self, conv_id: &str) -> Option<Vec<(String, String)>> {
        let state = self.inference.load_full()?;
        let mut out = {
            let base = &state.base_conv;
            base.section_contents()
        };
        // A conversation's journal sections are its own, submitted to its
        // timeline rather than declared by the schema: served as the round
        // carries them (`journal_sections`).
        let timeline = conversation_timeline(conv_id);
        let journal = state
            .journals
            .of(conv_id, || journal_record(&state, timeline));
        let (_, sections, _) = journal_sections(&state, conv_id, &journal);
        out.extend(sections.into_iter().map(|s| (s.name, s.text)));
        // The `tools` collection's summary section is generated at runtime (not a
        // schema section), so its text isn't in `section_contents`. Serve the
        // summary matching this conversation's tools mode — the tools that mode
        // offers, and none in None (no tools are projected) — under the key the
        // projection event uses (`<collection> summary`) so the panel expands the
        // right list.
        let mode = state
            .tool_modes
            .lock()
            .unwrap()
            .get(conv_id)
            .copied()
            .unwrap_or(ToolMode::Comprehensive);
        if mode == ToolMode::None {
            return Some(out);
        }
        // The tools-collection summary is assembled deterministically from the
        // catalog (the same text the startup seals); rebuild the mode-appropriate
        // one for the panel rather than reading a cache.
        let text = crate::tool_summary::tool_summary_for_mode(mode);
        if !text.is_empty() {
            out.push(("tools summary".to_string(), text));
        }
        Some(out)
    }

    /// The target (dialogue) layer's name, for the panel's conversation prefix.
    pub fn target_layer_name(&self) -> Option<String> {
        let state = self.inference.load_full()?;
        let base = &state.base_conv;
        Some(base.target_layer_name())
    }

    /// The `(user, assistant)` halves of a projected turn `index` in `timeline`,
    /// read from the substrate. Backs the projection panel's memory-tier bodies.
    /// `timeline` is the turn's resolved identity from `SelectedTurn::timeline` —
    /// never a group→timeline guess (the shared substrate holds many
    /// conversations under one group).
    pub fn resolve_turn_text(
        &self,
        timeline: candle_conversation::projection::TimelineId,
        index: u32,
    ) -> Option<(String, String)> {
        let state = self.inference.load_full()?;
        let base = &state.base_conv;
        base.resolve_turn_text(timeline, index)
    }

    /// The entire turn `index` in `timeline` as one continuous string (the full
    /// sealed token range, decoded verbatim). Backs the projection panel's turn
    /// cards — emit the whole turn rather than the split user/assistant halves.
    pub fn resolve_turn_full_text(
        &self,
        timeline: candle_conversation::projection::TimelineId,
        index: u32,
    ) -> Option<String> {
        let state = self.inference.load_full()?;
        let base = &state.base_conv;
        base.resolve_turn_full_text(timeline, index)
    }

    /// The turn's segment-vector `TurnLayout` — the complete K/V description
    /// (glue / user / thinking / assistant, real vs ethereal). Backs the panel's
    /// exact-segment rendering. `timeline` is the turn's resolved identity.
    pub fn turn_layout(
        &self,
        timeline: candle_conversation::projection::TimelineId,
        index: u32,
    ) -> Option<candle_conversation::turn_layout::TurnLayout> {
        let state = self.inference.load_full()?;
        let base = &state.base_conv;
        base.turn_layout(timeline, index)
    }

    /// The dialect's framing markers — the glue wrapped around the system prompt
    /// and each turn. The projection panel renders these between sections/turns.
    pub fn glue_markers(&self) -> Option<GlueMarkers> {
        let state = self.inference.load_full()?;
        let base = &state.base_conv;
        Some(base.glue_markers())
    }
}
