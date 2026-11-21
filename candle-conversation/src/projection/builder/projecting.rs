//! The entry points that run a projection for a target, with its mode and
//! selection chosen by the caller.
//!
//! `project` runs the steady-state decode mode. Prefill mode scores collections
//! with a calibrated profile, which gives a useful initial guess before any decode
//! probe exists. A selection overrides the section tree's dials for one call, and
//! an empty selection reproduces the authored defaults. A diagnostic run can replay
//! recorded projections in place of choosing them.

use super::*;

impl Builder {
    // ── Projection ────────────────────────────────────────────────────────────

    /// Run the full projection pipeline for the given `target`.
    ///
    /// Uses [`ProjectionMode::Decode`] — the steady-state mode for continuous
    /// reprojection during decode.  Call [`Self::project_with_mode`] with
    /// [`ProjectionMode::Prefill`] for the pre-decode initial-guess pass.
    ///
    /// See [`super::project`] for the 12-step pipeline. Pure given a valid
    /// schema and a working resolver — never errors.
    pub fn project<R: ContentResolver>(
        &self,
        target: ProjectionTarget,
        resolver: &R,
    ) -> Projection {
        run(
            &self.schema,
            target,
            resolver,
            ProjectionMode::Decode,
            &SelectionState::default(),
        )
    }

    /// Projection with an explicit section-tree [`SelectionState`] — the entry
    /// the runtime uses to choose selector options (e.g. the thinking-effort /
    /// response-length dials).  An empty state reproduces the authored defaults.
    pub fn project_with_selection<R: ContentResolver>(
        &self,
        target: ProjectionTarget,
        resolver: &R,
        mode: ProjectionMode,
        selection: &SelectionState,
    ) -> Projection {
        run(&self.schema, target, resolver, mode, selection)
    }

    /// Run the projection pipeline for an explicit [`ProjectionMode`].
    ///
    /// `Prefill` mode scores collections with the calibrated prefill profile
    /// (`Max` formula, semantic depth, no score threshold) so a useful
    /// initial section guess can be made before any decode probe runs.
    pub fn project_with_mode<R: ContentResolver>(
        &self,
        target: ProjectionTarget,
        resolver: &R,
        mode: ProjectionMode,
        selection: &SelectionState,
    ) -> Projection {
        run(&self.schema, target, resolver, mode, selection)
    }

    /// Variant of [`Self::project_with_mode`] that delivers score-density
    /// [`SelectionDiagnostics`] to a caller-supplied sink.  The scheduler
    /// uses this to ferry the diagnostic into the substrate's
    /// last-selection side-channel without polluting [`Projection`] with
    /// a test-only field.  When the projection used the rule-based path
    /// (no summary tree for the target timeline), the sink is never
    /// invoked.
    pub fn project_with_mode_and_sink<R: ContentResolver>(
        &self,
        target: ProjectionTarget,
        resolver: &R,
        mode: ProjectionMode,
        selection: &SelectionState,
        prior: &PriorBelief,
        decode_pos: Option<usize>,
        sink: &mut dyn FnMut(SelectionDiagnostics),
    ) -> Projection {
        let choice = self
            .script
            .as_ref()
            .and_then(|s| s.choice_at(decode_pos.unwrap_or(0)));
        let Some(choice) = choice else {
            return run_with_sink(
                &self.schema,
                target,
                resolver,
                mode,
                selection,
                prior,
                decode_pos,
                sink,
            );
        };
        let mut selection = selection.clone();
        selection.script_members(Arc::clone(&choice.members));
        for (selector, option) in &choice.dials {
            selection.select(selector.as_str(), option.as_str());
        }
        let mut projection = run_with_sink(
            &self.schema,
            target,
            resolver,
            mode,
            &selection,
            prior,
            decode_pos,
            sink,
        );
        // Faces as a live projection places them (`project.rs`): none when the
        // target is an ingest unit reading itself.
        let faces = !resolver.target_is_ingest_self();
        choice.place_turns(
            &mut projection,
            &|key| resolver.turn_token_count(key) > 0,
            &|key| faces.then(|| resolver.face_of(key)).flatten(),
        );
        projection
    }

    /// This builder, playing `script` back in place of choosing its projections.
    pub fn with_script(&self, script: ProjectionScript) -> Self {
        Self {
            script: Some(Arc::new(script)),
            ..self.clone()
        }
    }

    /// The script this builder plays back.
    pub fn script(&self) -> Option<&ProjectionScript> {
        self.script.as_deref()
    }

    /// Token-window budget configured for `layer`.  Used by the
    /// scheduler to populate `SelectionDiagnostics.budget`.  Returns
    /// `None` when the layer id isn't in the schema.
    pub fn schema_layer_window(&self, layer: LayerId) -> Option<usize> {
        self.schema
            .layers
            .iter()
            .find(|l| l.id == layer)
            .map(|l| l.window)
    }
}
