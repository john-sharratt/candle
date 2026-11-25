//! A trial turn's decode (`docs/npc_trials.md`): on the character's own frame,
//! a fork of its day's conversation, opened for it as the character's own turn
//! would open it; on a given frame, a conversation opened for the one trial and
//! thrown away.
//!
//! Both are held alike, so a prompt measured on a given frame measures what it
//! will do as the mind's own: one call of the trial's tools under the same
//! grammar (`Minds::answer_tree`), compiled before anything else is touched;
//! the same opening (the dialect's closed think block and `<tool_call>`); and
//! the character's own think-off sampling at the trial's temperature and seed.
//! What comes back is the text past the opening, on both. Nothing of either is
//! written back: the fork is never sealed and the given frame's timeline is
//! tombstoned on every path out.
//!
//! **Every trial on the own frame reads one context.** The selection is
//! prepared once for the conversation — the mission, journal and channel
//! sections brought in line, waiting for journal entries still sealing — under
//! its lock, kept against the conversation's timeline, and handed to every
//! trial's fork. Each fork is taken on the conversation its selection was
//! prepared on.
//!
//! **A suspended character holds its conversation only while it is tried.**
//! Suspension lets a character's conversation go so it holds nothing on the
//! card (`Minds::let_go`); a trial opens it again, and the last trial in flight
//! lets it go once more as it ends.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

use candle_conversation::projection::DecodePriority;
use candle_conversation::stencil::{StencilTree, TurnGrammar};
use candle_conversation::{AskScope, SamplingConfig, TurnOptions};

use super::think::SEAL_WAIT;
use super::Minds;
use crate::engine::identity::Deliberation;
use crate::engine::prompt::Persona;
use crate::engine::prose::{decode_framed, FramedTurn};
use crate::engine::tools::{Mode, Within};
use crate::engine::trial::{GivenFrame, Trial, TrialRefused};

/// What a trial's decode came back with, before its call is read.
#[derive(Clone, Debug)]
pub struct TrialDecode {
    /// The decode past its opening.
    pub raw: String,
    /// The context the turn was read in: what it was projected onto and its
    /// own prompt.
    pub prompt_tokens: usize,
    /// Tokens decoded.
    pub tokens: usize,
}

/// One trial in flight on a character; the last to end lets its conversation
/// go.
struct InFlight<'a> {
    minds: &'a Minds,
    npc_id: u64,
    count: Arc<AtomicUsize>,
}

impl<'a> InFlight<'a> {
    fn begin(minds: &'a Minds, npc_id: u64) -> InFlight<'a> {
        let count = Arc::clone(
            minds
                .trials_in_flight
                .pin()
                .get_or_insert_with(npc_id, || Arc::new(AtomicUsize::new(0))),
        );
        count.fetch_add(1, Ordering::SeqCst);
        InFlight {
            minds,
            npc_id,
            count,
        }
    }
}

impl Drop for InFlight<'_> {
    fn drop(&mut self) {
        if self.count.fetch_sub(1, Ordering::SeqCst) == 1 {
            self.minds.trial_selections.pin().remove(&self.npc_id);
            self.minds.let_go(self.npc_id);
        }
    }
}

impl Minds {
    /// The grammar a trial's answer is held to, or why its tools make none.
    fn trial_grammar(&self, trial: &Trial) -> Result<Arc<StencilTree>, TrialRefused> {
        self.answer_tree(&trial.tools)
            .map_err(|e| TrialRefused::BadTools(format!("{e:#}")))
    }

    /// Decode `trial` on a fork of `npc_id`'s conversation for `(day, chapter)`,
    /// opening it — or rolling it over, as the character's own turn would —
    /// first when the character does not hold it.
    // Who, as what, in what mode, on what day, from where, asked what: every
    // trial on the own frame supplies all of them, as a turn does.
    #[allow(clippy::too_many_arguments)]
    pub async fn trial_own(
        &self,
        npc_id: u64,
        persona: &Persona<'_>,
        mode: Mode,
        (day, chapter): (u64, u64),
        within: &Within,
        trial: &Trial,
    ) -> Result<TrialDecode, TrialRefused> {
        let tree = self.trial_grammar(trial)?;
        let _in_flight = InFlight::begin(self, npc_id);
        let (resident, _) = self
            .resident_for(npc_id, day, chapter, persona, mode)
            .await?;
        let selection = {
            let mut live = resident.conversation.lock().await;
            let timeline = live.sequence.timeline_id();
            let kept = self
                .trial_selections
                .pin()
                .get(&npc_id)
                .filter(|(at, _)| *at == timeline)
                .map(|(_, s)| s.clone());
            match kept {
                Some(s) => s,
                None => {
                    let s = self
                        .prepare_waiting(&mut live, npc_id, persona, within, Some(SEAL_WAIT))
                        .await;
                    self.trial_selections
                        .pin()
                        .insert(npc_id, (timeline, s.clone()));
                    s
                }
            }
        };
        let opening = self.opening(Deliberation::None, true);
        let options = TurnOptions {
            assistant_prefill: opening.clone(),
            turn_grammar: Some(TurnGrammar::of(tree)),
            sampling: Some(self.trial_sampling(trial)),
            max_tokens: Some(trial.max_tokens),
            selection,
            ..Default::default()
        };
        let response = resident
            .asker
            .ask(
                &trial.turn,
                DecodePriority::Normal,
                AskScope::Whole,
                options,
            )
            .await
            .map_err(|e| TrialRefused::Failed(e.into()))?;
        let raw = match &opening {
            Some(o) => response
                .text
                .strip_prefix(o.as_str())
                .unwrap_or(&response.text),
            None => response.text.as_str(),
        }
        .to_string();
        Ok(TrialDecode {
            raw,
            prompt_tokens: response.stats.prefill_token_count,
            tokens: response.stats.tokens_generated,
        })
    }

    /// Decode `trial` under `frame`, in a conversation opened for it alone.
    pub async fn trial_given(
        &self,
        frame: &GivenFrame,
        trial: &Trial,
    ) -> Result<TrialDecode, TrialRefused> {
        let turn = FramedTurn {
            system: &frame.system,
            history: &frame.history,
            prompt: &trial.turn,
            grammar: self.trial_grammar(trial)?,
            prefill: self.opening(Deliberation::None, true),
            sampling: self.trial_sampling(trial),
            max_tokens: trial.max_tokens,
            band: DecodePriority::Normal,
        };
        let (raw, stats) = decode_framed(&self.engine, &self.base_config, &turn).await?;
        Ok(TrialDecode {
            raw,
            prompt_tokens: stats.prefill_token_count,
            tokens: stats.tokens_generated,
        })
    }

    /// The character's own think-off row, at the trial's temperature and seed.
    fn trial_sampling(&self, trial: &Trial) -> SamplingConfig {
        let mut sampling = self.sampling_for(Deliberation::None).with_seed(trial.seed);
        sampling.temperature = trial.temperature;
        sampling
    }
}
