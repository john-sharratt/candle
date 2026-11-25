//! A trial turn, from the character it is put to: its persona, where it stands
//! and its day's conversation for the character's own frame, or the given
//! frame alone, and then the decode (`docs/npc_trials.md`).
//!
//! The chapter is read from the sim and not turned: turning one closes it in
//! the journal, which is the character's own turn's to do. A trial changes
//! nothing a later turn would read.

use std::time::Instant;

use super::Runtime;
use crate::clock;
use crate::engine::mind::TrialDecode;
use crate::engine::trial::{read_call, Trial, TrialAnswer, TrialRefused};

impl Runtime {
    /// Decode `trial` on `npc_id`, which the route has found suspended.
    /// `received` is when the route took it, which the answer's `ms` counts
    /// from.
    pub async fn trial(
        &self,
        npc_id: u64,
        trial: &Trial,
        received: Instant,
    ) -> Result<TrialAnswer, TrialRefused> {
        let minds = self
            .minds
            .read()
            .unwrap()
            .clone()
            .ok_or(TrialRefused::NotLoaded)?;
        let decoded: TrialDecode = match &trial.frame {
            Some(frame) => minds.trial_given(frame, trial).await?,
            None => {
                let persona = self
                    .persona_of_async(npc_id)
                    .await
                    .ok_or_else(|| anyhow::anyhow!("npc {npc_id} has no persona"))?;
                let within = self.within(npc_id);
                let day = clock::day_of(self.world_ms(npc_id));
                let chapter = self.chapter_now(npc_id);
                minds
                    .trial_own(
                        npc_id,
                        &persona.as_persona(),
                        persona.mode,
                        (day, chapter),
                        &within,
                        trial,
                    )
                    .await?
            }
        };
        Ok(TrialAnswer {
            call: read_call(&decoded.raw, &trial.tools),
            raw: decoded.raw,
            prompt_tokens: decoded.prompt_tokens,
            tokens: decoded.tokens,
            ms: received.elapsed().as_millis(),
        })
    }

    /// The chapter `npc_id` stands in — how many missions it has closed — read
    /// without closing one. `0` for a character with no body.
    fn chapter_now(&self, npc_id: u64) -> u64 {
        self.body_of(npc_id).map_or(0, |(hosted, body)| {
            hosted.sim(|s| s.missions.chapter(&body))
        })
    }
}
