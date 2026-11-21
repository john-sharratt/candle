//! Answers the substrate views' routes by reading the live substrate through a
//! cloned handle, without rebuilding it from the log.
//!
//! The views answer once the substrate has replayed, which can be before the
//! model is up. Overview, layer, timeline and selection reads are structural
//! snapshots of the shared substrate. The projection query prefills its text
//! under the base conversation's system prompt and scores it without writing
//! anything. It ranks on the raw belief before clamping to the normalized band,
//! because a lock-on sits far above that band and clamping first would make the
//! order of the leaders arbitrary.

use super::*;

impl ZendSession {
    /// What the substrate views read: the substrate and the target layer — the
    /// loaded engine's, or while zend is still loading the substrate the load
    /// published once its redo log replayed, with no target layer built yet.
    /// `None` before the replay has finished.
    pub(super) fn substrate_for_views(&self) -> Option<(Conversation, String)> {
        if let Some(state) = self.inference.load_full() {
            return Some((state.substrate.clone(), state.target_layer.clone()));
        }
        let conv = self.loading_substrate.get().cloned()?;
        Some((conv, String::new()))
    }

    /// `GET /v1/substrate` — a structural snapshot of the live substrate for the
    /// viewer: on-disk size, live counts, and the projection hierarchy (layers →
    /// conversations). Reads the shared `Substrate` through a cloned
    /// `Conversation` handle (one version of it, held for the walk), never a
    /// rebuild from the log.
    /// Answers while zend is still loading, once its substrate has replayed.
    pub fn substrate_overview(&self) -> Option<SubstrateOverview> {
        let (conv, target_layer) = self.substrate_for_views()?;
        // The builder parsed once at session construction: no lock held, and no
        // per-request re-parse.
        Some(overview(
            &conv,
            self.projection_builder.schema(),
            target_layer,
            &self.config.workspace.root().join(SUBSTRATE_DIR),
        ))
    }

    /// Flip a projection layer's runtime kill switch (in-memory, non-persistent).
    /// Returns the layer's NEW disabled state, or `None` if `name` isn't a
    /// declared layer. The exclusion takes effect on the next (re)projection; a
    /// restart clears every toggle. Diagnostic only — for isolating which layer's
    /// projected content is breaking coherence.
    pub fn toggle_projection_layer(&self, name: &str) -> Option<bool> {
        let known = self
            .projection_builder
            .schema()
            .layers
            .iter()
            .any(|l| l.name == name);
        if !known {
            return None;
        }
        let state = self.inference.load_full()?;
        let conv = { state.engine.conversation() };
        Some(conv.toggle_layer(name))
    }

    /// `GET /v1/substrate/zend/layer/{name}` — the conversations (timelines) that
    /// target one layer. Fetched when the layer card expands. `None` before the
    /// substrate has replayed or when the layer name is unknown.
    pub fn substrate_layer(&self, name: &str) -> Option<LayerConversations> {
        let schema = self.projection_builder.schema();
        // Resolve the layer's groups before touching the substrate.
        let groups = schema
            .layers
            .iter()
            .find(|l| l.name == name)
            .map(|l| l.groups.clone())?;
        let (conv, _) = self.substrate_for_views()?;
        let read = conv.read();
        Some(layer_conversations(&read, &groups, unit_of))
    }

    /// `GET /v1/substrate/zend/system-prompt` — the target (dialogue) layer's framing:
    /// its ordered item labels, its always-emit sections with authored text +
    /// materialized token cost, and the live tool-catalog size (the catalog
    /// itself loads from `substrate_tools`). Fetched when the system-prompt card
    /// expands.
    pub fn substrate_system_prompt(&self) -> Option<SystemPromptView> {
        let (conv, target_layer) = self.substrate_for_views()?;
        let read = conv.read();
        Some(system_prompt(
            &read,
            self.projection_builder.schema(),
            target_layer,
            crate::tool_def::all().len(),
        ))
    }

    /// zend's entries in `GET /v1/tools` — the live tool catalog projected into the
    /// dialogue layer's `tools` collection. Sourced from the bundled tool
    /// definitions (`src/prompts/tools/`, embedded); the schema YAML declares the
    /// collection empty and tools install at daemon start. Fetched when the tools
    /// collection expands. The bundled definitions alone: it answers while zend
    /// is still loading.
    pub fn substrate_tools(&self) -> Option<ToolsView> {
        Some(ToolsView {
            tools: crate::tool_def::all()
                .iter()
                .map(|d| ToolView {
                    name: d.name.clone(),
                    description: d.description.clone(),
                    high_risk: d.high_risk,
                    parameters: d.parameters.clone(),
                })
                .collect(),
        })
    }

    /// The conversation `conv_id` as [`Self::substrate_timeline`] reads it:
    /// every turn decoded. `None` before the substrate has replayed, or when it
    /// has no turns.
    pub fn conversation_detail(&self, conv_id: &str) -> Option<TimelineDetail> {
        self.substrate_timeline(conversation_timeline(conv_id).raw())
    }

    /// `GET /v1/substrate/zend/timeline/{tl}` — one conversation's full summary forest,
    /// every turn decoded. `tl` is the raw timeline id. `None` before the
    /// substrate has replayed or when no such timeline exists.
    pub fn substrate_timeline(&self, tl_raw: u64) -> Option<TimelineDetail> {
        let (conv, _) = self.substrate_for_views()?;
        let read = conv.read();
        timeline(&read, self.projection_builder.schema(), tl_raw)
    }

    /// `GET /v1/substrate/zend/timeline/{tl}/selection` — the most recent
    /// score-density selection diagnostic recorded for this timeline. `None`
    /// when the model isn't loaded, `tl` isn't a real timeline, or no
    /// projection has run for it yet (or it used the rule-based path, which
    /// records no diagnostic). In-memory only — last-write-wins per timeline,
    /// nothing here survives a restart. See
    /// `ConversationEngine::last_selection_diagnostics`.
    pub fn substrate_selection(&self, tl_raw: u64) -> Option<SelectionView> {
        let state = self.inference.load_full()?;
        let tl = projection::TimelineId::from_raw(tl_raw)?;
        let diag = state.engine.last_selection_diagnostics(tl)?;
        Some(SelectionView {
            selected: diag
                .selected_nodes
                .iter()
                .zip(diag.origins.iter())
                .map(|(n, o)| SelectedNodeView {
                    node_id: n.0,
                    origin: *o,
                    effective_score: diag.effective_scores.get(n).copied(),
                })
                .collect(),
            pending_count: diag.pending_count,
            selected_tokens: diag.selected_tokens,
            budget: diag.budget,
        })
    }

    /// `POST /v1/substrate/zend/project` — project a typed query against the substrate.
    /// Prefills `text` under the base conversation's system prompt (no seal, no
    /// belief mutation), then resolves the selected turns + scored section members
    /// into score-sorted tiles. `None` until the model is loaded.
    pub fn substrate_project(&self, text: &str, decode: usize) -> Option<ProjectView> {
        let state = self.inference.load_full()?;
        if text.trim().is_empty() {
            return Some(ProjectView {
                query_tokens: 0,
                tiles: Vec::new(),
                probe: String::new(),
                question: 0,
            });
        }

        // Project the query against the substrate as a warm ephemeral turn: the
        // probe materializes the full system prompt, captures the query's wide-Q,
        // and scores it — writing NOTHING to the substrate. Snapshot the handles
        // under a brief `base_conv` lock, then run the GPU-bound probe UNLOCKED so
        // the fork source isn't held across a full turn round-trip.
        let ctx = { state.base_conv.probe_ctx() };
        let Probed {
            event,
            sigs: probe,
            question,
        } = match ctx.probe(text, decode) {
            Ok(r) => r,
            Err(e) => {
                tracing::warn!("substrate_project probe failed: {e:#}");
                return Some(ProjectView {
                    query_tokens: 0,
                    tiles: Vec::new(),
                    probe: String::new(),
                    question: 0,
                });
            }
        };
        let query_tokens = probe.len();

        // Resolve tile labels + bodies from the substrate (read lock only).
        let conv = { state.engine.conversation() };
        let read = conv.read();
        let s = &*read;

        let mut tiles: Vec<ProjectTile> = Vec::new();

        // Selected conversation turns / summary nodes.
        for t in &event.selection.turns {
            let Some(tl_raw) = t.timeline else {
                continue; // the live user message has no source timeline
            };
            let kind = match t.kind {
                TurnKind::Normal => "normal",
                TurnKind::SummaryOfTurns => "sot",
                TurnKind::SummaryOfSummaries => "sos",
            };
            let (label, body) = match projection::TimelineId::from_raw(tl_raw) {
                Some(tl) => {
                    let title = s
                        .label_of(tl)
                        .filter(|l| !l.is_empty())
                        .or_else(|| s.conv_id_of(tl).filter(|c| !c.is_empty()))
                        .map(|x| x.to_string())
                        .or_else(|| {
                            s.custom_of(tl)
                                .and_then(|m| m.get("path"))
                                .filter(|p| !p.is_empty())
                                .cloned()
                        })
                        .unwrap_or_else(|| format!("turn #{}", t.index));
                    let idx = TurnIndex(t.index);
                    let u = s.user_text_of(tl, idx);
                    let a = s.assistant_text_of(tl, idx);
                    let body = if u.trim().is_empty() {
                        a
                    } else if a.trim().is_empty() {
                        u
                    } else {
                        format!("{u}\n\n{a}")
                    };
                    (title, body)
                }
                None => (format!("turn #{}", t.index), String::new()),
            };
            tiles.push(ProjectTile {
                kind,
                score: t.score,
                selected: t.selected,
                layer: t.layer.clone(),
                group: t.group.clone(),
                label,
                tokens: t.tokens,
                timeline: Some(tl_raw.to_string()),
                index: Some(t.index),
                text: body,
            });
        }

        // Scored section members (the collection tools). Text comes from the
        // bundled tool definitions (the schema declares the collection empty). Cut
        // the noise floor: keep only members the query selected or gave a positive
        // belief score.
        let tool_desc: HashMap<&str, &str> = crate::tool_def::all()
            .iter()
            .map(|d| (d.name.as_str(), d.description.as_str()))
            .collect();
        for item in &event.selection.system {
            if let SystemItem::Collection { name, sections, .. } = item {
                for sec in sections {
                    if !sec.selected && sec.score <= 0.0 {
                        continue;
                    }
                    tiles.push(ProjectTile {
                        kind: "section",
                        score: sec.score,
                        selected: sec.selected,
                        layer: String::new(),
                        group: name.clone(),
                        label: sec.name.clone(),
                        tokens: sec.tokens,
                        timeline: None,
                        index: None,
                        text: tool_desc
                            .get(sec.name.as_str())
                            .map(|d| d.to_string())
                            .unwrap_or_default(),
                    });
                }
            }
        }

        // Rank on the RAW belief, then bound what is rendered. Order first is
        // load-bearing: a lock-on rides far above the 0-1000 normalized band, so
        // clamping before the sort ties the leaders and the readout's ordering —
        // the thing the margin is actually read for — becomes arbitrary.
        tiles.sort_by(|a, b| b.score.total_cmp(&a.score));
        for tile in &mut tiles {
            tile.cap_score();
        }

        // Diagnostic: if `scored` is 0 while `query_tokens` > 0, the probe Q was
        // captured but didn't discriminate against the gallery (cold/partial-warm
        // context) — selection fell back to the default fill.
        let max_score = tiles.iter().map(|t| t.score).fold(0.0f32, f32::max);
        tracing::info!(
            query_tokens,
            tiles = tiles.len(),
            scored = tiles
                .iter()
                .filter(|t| t.score.abs() > f32::EPSILON)
                .count(),
            max_score,
            "substrate_project"
        );

        Some(ProjectView {
            query_tokens,
            tiles,
            probe: STANDARD.encode(encode_wide_sigs(&probe)),
            question,
        })
    }
}
