//! Runs a character's reflections and dreams as throwaway conversations that
//! leave its live one alone.
//!
//! Each reflection and each dream decodes in a timeline of its own, so a
//! character mid-turn is not waited on and nothing it shares is touched. Both
//! open against the mind's schema, so their prompts are built by the same code
//! as a live turn and cannot drift from it section by section. Dream lines are
//! scored against a learned hit level, which starts from a cold-start value and
//! is lifted by the traffic that reads them, so a line that matches every turn
//! is discounted. A dream is due twelve hours after the last one kept, or when
//! the character has never dreamt. Reflection is unavailable when the mind
//! declares no reflection turns, and that is said once at load.

use super::*;

impl Minds {
    /// Install the reflection turns the mind authors.
    ///
    /// `None` leaves reflection unavailable and says so once at load, rather than
    /// at the first request — a deployment that has lost the block should find out
    /// from its own startup log and not from a character that will not reflect.
    pub fn asking(mut self, turns: Option<ReflectionTurns>) -> Self {
        match &turns {
            Some(_) => tracing::info!(
                "reflection armed — two authored turns into a transient conversation that is \
                 tombstoned when they are answered"
            ),
            None => tracing::warn!(
                "reflection unavailable — the mind declares no usable `reflection` block, so \
                 nothing will generate a dream brief"
            ),
        }
        self.reflection = turns;
        self
    }

    /// Put every dream already kept on the normalized score band.
    ///
    /// A recalled line is scored against its own **hit level** — the score it
    /// reaches when it is the answer — so the dream group's gate means what the
    /// same numbers mean on `repo_map`. The levels are learned from the traffic
    /// that reads them: every turn a character takes carries its own dreams tag,
    /// so its seal teaches its own dreams, and a line that matches every turn
    /// ends up with a high level and is discounted.
    ///
    /// **This is the cold start, not the level.** A dream is probed by its own
    /// lines, and the vote is a sum over a probe's strongest tokens — a line is
    /// one sentence where a turn probes with its whole tail, so self-match lands
    /// well below what the room reaches and live traffic lifts it within a few
    /// turns. It still beats the bare prior, which multiplies the raw vote rather
    /// than normalizing it: recalled lines scored anywhere from four thousand to
    /// a million on it.
    ///
    /// Dreams kept after this are warmed as they are written — see
    /// [`dreams::keep`].
    pub(super) fn warm_dreams(&self, projected: &Projected) {
        let Some(group) = projected.builder.id_for_group(dreams::GROUP) else {
            return;
        };
        let warmed = self
            .engine
            .warm_group_normalization(projected.builder.schema(), group);
        tracing::info!("dreams: {warmed} dream(s) put on the normalized score band");
    }

    /// Retire every dream this character has kept — see [`dreams::forget`].
    ///
    /// Returns how many it retired.
    pub fn forget_dreams(&self, npc_id: u64) -> usize {
        dreams::forget(&self.engine, npc_id)
    }

    /// Stop a character and run one reflection — see [`crate::engine::reflect`].
    ///
    /// Goes through here rather than being built by the caller because the
    /// engine handle and the checkpoint's own [`SequenceConfig`] live here, and
    /// a reflection assembled from a second opinion about either would run the
    /// model outside the settings it was tuned under.
    ///
    /// **Touches none of this character's live state.** It does not open, read
    /// or disturb the conversation in [`Self::live`]: a reflection is its own
    /// throwaway timeline, and a character mid-decode is not waited on because
    /// there is nothing here they share.
    #[allow(clippy::too_many_arguments)]
    pub async fn reflect(
        &self,
        npc_id: u64,
        persona: &Persona<'_>,
        mode: Mode,
        situation: &str,
        inner_thoughts: &str,
        feeling: &str,
        domain: &str,
        sampled_axes: &[String],
        on_reflection: &mut (dyn FnMut(&str) -> bool + Send),
    ) -> anyhow::Result<reflect::Reflection> {
        // The mind's schema, so a reflection opens against the same sections a
        // live conversation does. Without it the two prompts are built by
        // different code and drift apart section by section — which is exactly
        // what had happened: the reflection's frame named none of the schema's
        // sections and carried none of its collections.
        let projected = self.projected();
        let turns = self.reflection.as_ref().ok_or_else(|| {
            anyhow::anyhow!(
                "this mind declares no `reflection` block, so there are no turns to send — \
                 author `reflection.question_one`, `.question_two` and `.question_two_retry` \
                 in projection.yaml"
            )
        })?;
        reflect::Reflect::new(Arc::clone(&self.engine), self.base_config.clone(), turns)
            .under(projected.as_deref())
            .run(
                npc_id,
                persona,
                mode,
                situation,
                inner_thoughts,
                feeling,
                domain,
                sampled_axes,
                on_reflection,
            )
            .await
    }

    /// Dream a brief and keep the dream — see [`crate::engine::dreams`].
    ///
    /// Two conversations, one after the other, neither of them this
    /// character's live one: the dream is decoded in a throwaway conversation
    /// and written, a line to a turn, into one of its own on the dream layer.
    /// Nothing here touches [`Self::live`], so a character mid-decode is not
    /// waited on.
    pub async fn dream(
        &self,
        npc_id: u64,
        persona: &Persona<'_>,
        brief: &str,
        assumption: &str,
    ) -> anyhow::Result<dreams::Kept> {
        let p = self
            .projected()
            .ok_or_else(|| anyhow::anyhow!("no schema, so no dream layer to keep a dream in"))?;
        let story =
            dreams::dream(&self.engine, &self.base_config, &p, npc_id, persona, brief).await?;
        dreams::keep(
            &self.engine,
            &self.base_config,
            &p,
            npc_id,
            assumption,
            &story,
        )
        .await
    }

    /// A sample of the axes this character has dreamt along — see
    /// [`dreams::sample_axes`].
    pub fn dreamt_axes(&self, npc_id: u64, n: usize) -> Vec<String> {
        dreams::sample_axes(&self.engine, npc_id, n)
    }

    /// How many dreams this character has kept.
    pub fn dreams_kept(&self, npc_id: u64) -> usize {
        dreams::count(&self.engine, npc_id)
    }

    /// Whether this character's next dream is due: twelve hours since its last
    /// kept one ([`dreams::DREAM_EVERY_MS`]), or it has never dreamt.
    pub fn dream_due(&self, npc_id: u64, now_ms: u64) -> bool {
        dreams::due(dreams::last_dreamt_ms(&self.engine, npc_id), now_ms)
    }

    /// Whether this daemon can reflect at all: a mind that authors the
    /// questions, and a schema to ask them under.
    pub fn can_reflect(&self) -> bool {
        self.reflection.is_some() && self.projection.load().is_some()
    }
}
