//! Reflections: a character asked two questions, and the dream that may follow.
//!
//! A reflection stops a character and asks it questions in a conversation that is
//! then discarded. It runs serially behind its caller, since reflecting is what a
//! character does once it has chosen to stand still. The answer reaches the
//! character as soon as its first question is answered, and any dream it grows
//! into is written afterward on a task of its own. A character holds one dream
//! slot at a time, released on drop, so a dream that fails cannot leave the
//! character unable to dream again.

use super::*;

/// How many of a character's existing dream axes a reflection is steered away
/// from — `docs/reflection_and_dreams.md` §5: shown every axis it had used the
/// generator recombined them; shown a random eight it found a new one.
pub(super) const SAMPLED_AXES: usize = 8;

/// One character's dream slot, taken — see [`Runtime::claim_dream`]. Handed
/// back when dropped, so a dream that errors or panics part-way cannot leave
/// the character unable to dream again.
pub(super) struct DreamSlot<'a> {
    pub(super) rt: &'a Runtime,
    pub(super) npc_id: u64,
}

impl Drop for DreamSlot<'_> {
    fn drop(&mut self) {
        self.rt.dreaming.lock().unwrap().remove(&self.npc_id);
    }
}

impl Runtime {
    /// Stop one character, ask it two things, and throw the conversation away.
    ///
    /// The whole of it runs **serially**, behind the caller. That looks like a
    /// violation of the engine's standing rule that a fast clock must never wait
    /// on a slow one, and is not: reflecting is what a character does when it
    /// has already chosen to stand still, so the wait costs nothing it had not
    /// spent. The dream this produces a brief for is the part that stays
    /// asynchronous.
    ///
    /// `domain` rotates when the caller does not name one, so the behaviour
    /// space fills evenly instead of being sampled wherever the model prefers —
    /// see [`reflect::DOMAINS`]. `sampled_axes` is a **sample** of the
    /// character's existing dream corpus and must never be all of it.
    ///
    /// The dream is written after this returns, on a thread of its own — the
    /// caller gets the reflection, and the dream lands in the corpus whenever
    /// it lands. See [`Self::dream_now`].
    pub async fn reflect(
        self: &Arc<Self>,
        npc_id: u64,
        situation: Option<&str>,
        inner_thoughts: &str,
        feeling: &str,
        domain: Option<&str>,
        sampled_axes: &[String],
    ) -> anyhow::Result<reflect::Reflection> {
        let r = self
            .reflect_with(
                npc_id,
                situation,
                inner_thoughts,
                feeling,
                domain,
                sampled_axes,
                &mut |_| true,
            )
            .await?;
        match r.dream() {
            Ok((brief, assumption)) => {
                let (brief, assumption) = (brief.to_string(), assumption.to_string());
                let rt = Arc::clone(self);
                match &self.spawner {
                    Some(handle) => {
                        handle.spawn(
                            async move { rt.dream_if_free(npc_id, &brief, &assumption).await },
                        );
                    }
                    None => tracing::warn!(
                        "npc {npc_id}: the dream could not be started — no async runtime"
                    ),
                }
            }
            Err(why) => tracing::warn!("npc {npc_id}: no dream from this reflection — {why}"),
        }
        Ok(r)
    }

    /// One reflection, handing `on_reflection` the line that crosses back the
    /// moment the first question is answered. See [`reflect::Reflect::run`].
    // Each is a separate axis of one reflection — whose, where it says it is,
    // what it is thinking and feeling, which cell, steered from what, and who
    // hears the answer first — and every caller supplies all of them.
    #[allow(clippy::too_many_arguments)]
    pub(super) async fn reflect_with(
        &self,
        npc_id: u64,
        situation: Option<&str>,
        inner_thoughts: &str,
        feeling: &str,
        domain: Option<&str>,
        sampled_axes: &[String],
        on_reflection: &mut (dyn FnMut(&str) -> bool + Send),
    ) -> anyhow::Result<reflect::Reflection> {
        if self.scheduler.is_held(npc_id) {
            anyhow::bail!(
                "npc {npc_id} is suspended or at rest, and reflects on nothing until it is let go"
            );
        }
        let who = self
            .persona_of_async(npc_id)
            .await
            .ok_or_else(|| anyhow::anyhow!("no such character, or it has no persona"))?;

        // Failing that, where the character is in the world's own words: the
        // percept of the room its body stands in and who is there with it. The persona's own
        // `situation` is empty for every authored character — it is filled per
        // tick from a world *delta*, and there is no delta here — so a
        // reflection taken from it opened on nothing, and a character asked to
        // dream about its day dreamed about an office.
        //
        // **The character's own account first.** The act passes it — where the
        // character says it is and what is going on — and the reflection is
        // framed on that: `docs/reflection_and_dreams.md` §3. The percept is the
        // fallback for a caller that gives none, never a replacement for one
        // that did.
        let situation = situation
            .map(str::trim)
            .filter(|s| !s.is_empty())
            .map(str::to_string)
            .or_else(|| self.situation_of(npc_id))
            .unwrap_or_else(|| who.situation.clone());

        let domain = domain.map(str::to_string).unwrap_or_else(|| {
            let n = self.reflect_domain.fetch_add(1, Ordering::Relaxed);
            reflect::DOMAINS[n % reflect::DOMAINS.len()].to_string()
        });

        // Cloned out rather than read through the guard: this runs for the
        // length of several decodes, on a thread that reads `minds` again to
        // hand the answer back.
        let minds = self
            .minds
            .read()
            .unwrap()
            .clone()
            .ok_or_else(|| anyhow::anyhow!("no engine — the daemon is still loading"))?;
        minds
            .reflect(
                npc_id,
                &who.as_persona(),
                who.mode,
                &situation,
                inner_thoughts,
                feeling,
                &domain,
                sampled_axes,
                on_reflection,
            )
            .await
    }

    /// Dream a brief and keep the dream. Logged, never returned: nobody is
    /// waiting on it, and a dream that did not land costs a dream.
    pub(super) async fn dream_now(&self, npc_id: u64, brief: &str, assumption: &str) {
        if self.scheduler.is_held(npc_id) {
            tracing::info!("npc {npc_id}: suspended or at rest, so this brief is not dreamt");
            return;
        }
        let started = Instant::now();
        tracing::info!("npc {npc_id}: dream begun");
        let Some(minds) = self.minds.read().unwrap().clone() else {
            return;
        };
        let Some(who) = self.persona_of_async(npc_id).await else {
            return;
        };
        match minds
            .dream(npc_id, &who.as_persona(), brief, assumption)
            .await
        {
            Ok(kept) => tracing::info!(
                "npc {npc_id}: dreamt, {} passage(s) kept in {:?} — {} dream(s) now:\n{}",
                kept.passages.len(),
                started.elapsed(),
                minds.dreams_kept(npc_id),
                kept.passages.join("\n\n"),
            ),
            Err(e) => tracing::warn!("npc {npc_id}: the dream was not kept — {e:#}"),
        }
    }

    /// Dream a supplied brief once, on demand, and **return** it — the one-off
    /// counterpart to [`Self::dream_now`], which logs and discards because
    /// nobody is waiting. Here a caller hands the premise and sees what the
    /// dreamer makes of it (the dream API). It is kept, like any dream, so it
    /// also lands on the character's dream layer.
    pub async fn dream_once(
        &self,
        npc_id: u64,
        brief: &str,
        assumption: &str,
    ) -> anyhow::Result<crate::engine::dreams::Kept> {
        if self.scheduler.is_held(npc_id) {
            anyhow::bail!(
                "npc {npc_id} is suspended or at rest, and dreams nothing until it is let go"
            );
        }
        let minds = self
            .minds
            .read()
            .unwrap()
            .clone()
            .ok_or_else(|| anyhow::anyhow!("no engine — the daemon is still loading"))?;
        let who = self
            .persona_of_async(npc_id)
            .await
            .ok_or_else(|| anyhow::anyhow!("no such character, or it has no persona"))?;
        minds
            .dream(npc_id, &who.as_persona(), brief, assumption)
            .await
    }

    /// Claim this character's one reflection slot, if a reflection can run.
    ///
    /// `false` only when this daemon cannot reflect at all — no mind authoring
    /// the questions. Never for being soon after another: the character is held
    /// until its answer lands, so it cannot ask twice at once, and the work that
    /// is one at a time is the dream — see [`Self::claim_dream`].
    pub(super) fn can_reflect(&self) -> bool {
        self.minds
            .read()
            .unwrap()
            .as_ref()
            .is_some_and(|m| m.can_reflect())
    }

    /// Whether a reflection this character took now could go on to a dream:
    /// its next dream is due and none is being written. What `reflect` is
    /// offered on — see [`Self::within`].
    pub(super) fn dream_available(&self, npc_id: u64) -> bool {
        let due = self
            .minds
            .read()
            .unwrap()
            .as_ref()
            .is_some_and(|m| m.dream_due(npc_id, crate::api::now_ms()));
        due && !self.dreaming.lock().unwrap().contains(&npc_id)
    }

    /// Take this character's one dream slot, if it is free. Held until the
    /// returned guard drops. See [`Self::dreaming`].
    pub(super) fn claim_dream(&self, npc_id: u64) -> Option<DreamSlot<'_>> {
        // **The lock is released before any guard exists.** A guard dropped
        // while this lock is held deadlocks, because dropping one takes the
        // lock to hand the slot back — and `then_some` builds its value
        // eagerly, so on a taken slot it built a guard and dropped it on the
        // spot, inside the lock, and the thread waited on itself for ever. It
        // would also have handed back the slot it had just been refused.
        let free = self.dreaming.lock().unwrap().insert(npc_id);
        free.then(|| DreamSlot { rt: self, npc_id })
    }

    /// Dream a brief if this character is not already dreaming, and say so if
    /// it is. For the route, whose reflection has already asked for the brief.
    pub(super) async fn dream_if_free(&self, npc_id: u64, brief: &str, assumption: &str) {
        match self.claim_dream(npc_id) {
            Some(_slot) => self.dream_now(npc_id, brief, assumption).await,
            None => tracing::info!(
                "npc {npc_id}: a dream is already being written, so this brief is not dreamt"
            ),
        }
    }

    /// Answer a `reflect` with a reflection, as a task of its own.
    ///
    /// Returns the channel the reflection's first line arrives on. The
    /// character's own task **awaits it** before taking another turn — that
    /// await is the hold, expressed as code order: the room cannot give the
    /// character a turn while its task is not looping, and everything that
    /// arrives meanwhile queues in its inbox. A channel that closes without a
    /// line — the task could not start, or the reflection died before
    /// answering — means the world's own line (already in the owed slot)
    /// stands; see [`Self::settle_reflection`].
    ///
    /// The reflection task carries on behind the answer with nobody waiting:
    /// the loosed turn, the brief, and — if this character's dream slot is
    /// free — the dream.
    pub fn spawn_reflection(
        self: &Arc<Self>,
        npc_id: u64,
        situation: String,
        inner_thoughts: String,
        feeling: String,
    ) -> oneshot::Receiver<String> {
        let (answer_tx, answer_rx) = oneshot::channel();
        match &self.spawner {
            Some(handle) => {
                let rt = Arc::clone(self);
                handle.spawn(async move {
                    rt.reflection_task(npc_id, situation, inner_thoughts, feeling, answer_tx)
                        .await;
                });
            }
            None => {
                tracing::warn!(
                    "npc {npc_id}: the reflection could not be started — no async runtime"
                );
                // `answer_tx` drops here; the closed channel is the caller's
                // signal to fall back to the world's own line.
            }
        }
        answer_rx
    }

    /// The reflection task's whole life: ask, send the answer, then dream.
    /// See [`Self::spawn_reflection`].
    pub(super) async fn reflection_task(
        self: Arc<Self>,
        npc_id: u64,
        situation: String,
        inner_thoughts: String,
        feeling: String,
        answer_tx: oneshot::Sender<String>,
    ) {
        let started = Instant::now();
        let axes = {
            let Some(minds) = self.minds.read().unwrap().clone() else {
                // `answer_tx` drops: the character falls back to the world's line.
                return;
            };
            minds.dreamt_axes(npc_id, SAMPLED_AXES)
        };
        let mut answer_tx = Some(answer_tx);
        let mut slot: Option<DreamSlot<'_>> = None;
        // A dream follows a reflection once a twelve-hour period at most
        // (`dreams::DREAM_EVERY_MS`); the reflection answers every time.
        let dream_due = self
            .minds
            .read()
            .unwrap()
            .as_ref()
            .is_some_and(|m| m.dream_due(npc_id, crate::api::now_ms()));
        let result = self
            .reflect_with(
                npc_id,
                Some(&situation),
                &inner_thoughts,
                &feeling,
                None,
                &axes,
                // Send the character its answer, then decide whether this
                // reflection goes on to a dream: only if the character's dream
                // slot is free, and then it is held through the dream and
                // handed back when this task ends.
                &mut |line| {
                    if let Some(tx) = answer_tx.take() {
                        let _ = tx.send(line.to_string());
                        tracing::info!(
                            "npc {npc_id}: reflect answered in {:?}, before any dream — {line}",
                            started.elapsed()
                        );
                    }
                    slot = match dream_due {
                        true => self.claim_dream(npc_id),
                        false => None,
                    };
                    slot.is_some()
                },
            )
            .await;
        // A reflection that failed before it answered drops the channel here,
        // and the closed channel is the character's signal that nothing came —
        // [`body::NO_REFLECTION`], the line already in its slot, stands.
        drop(answer_tx.take());
        let r = match result {
            Ok(r) => r,
            Err(e) => {
                tracing::warn!("npc {npc_id}: reflection failed — {e:#}");
                return;
            }
        };
        // The turn count and each turn's tokens, because a slow reflection is
        // either too many tokens or too slow a token, and the two have different
        // fixes — the elapsed time alone cannot say which.
        tracing::info!(
            "npc {npc_id}: reflection done in {:?} — {} turn(s), {} token(s) {:?} ({} axis/axes \
             sampled){}",
            started.elapsed(),
            r.tokens.len(),
            r.tokens.iter().sum::<usize>(),
            r.tokens,
            axes.len(),
            r.fault
                .as_deref()
                .map(|f| format!(" — brief fault: {f}"))
                .unwrap_or_default()
        );
        if slot.is_none() {
            match dream_due {
                true => tracing::info!(
                    "npc {npc_id}: a dream is already being written, so this reflection stopped \
                     at its answer"
                ),
                false => tracing::info!(
                    "npc {npc_id}: dreamt within the last twelve hours, so this reflection \
                     stopped at its answer"
                ),
            }
            return;
        }
        match r.dream() {
            Ok((brief, assumption)) => self.dream_now(npc_id, brief, assumption).await,
            Err(why) => tracing::warn!("npc {npc_id}: no dream from this reflection — {why}"),
        }
    }

    /// Land a reflection's answer on the character: complete the act's row and
    /// hand back every answer the turn owed, the reflect's line among them.
    ///
    /// `line` is `None` when the reflection's channel closed without one — it
    /// never started, or died before answering — and the world's own line,
    /// already in the owed slot, is what the character reads.
    pub(super) async fn settle_reflection(
        &self,
        npc_id: u64,
        mut owed: Owed,
        line: Option<String>,
    ) {
        if let (Some(line), Some(answer)) = (line.as_ref(), owed.answers.get_mut(owed.slot)) {
            *answer = line.clone();
        }
        let came_back = owed.answers.get(owed.slot).cloned().unwrap_or_default();
        self.scheduler.amend_act(
            npc_id,
            &owed.row,
            format!("{} {LANDED} {came_back}", owed.row),
        );
        let minds = self.minds.read().unwrap().clone();
        if let Some(minds) = minds {
            minds.deliver_outcomes(npc_id, owed.answers).await;
        }
        // **And a turn to read it.** The tick that reflected has ended by
        // now, unscheduled — a reflect earns no follow-up of its own — so the
        // reflection and every answer it carried waited on the room.
        self.scheduler.think_again(npc_id);
    }

    /// Put `question` to a character and force an answer — see [`Minds::ask`].
    ///
    /// Waits for the character's own turn boundary, then reads it as it stands
    /// now: its persona, the room it is in, the mission it is carrying.
    pub async fn ask(
        &self,
        npc_id: u64,
        question: &str,
        choices: &[String],
    ) -> anyhow::Result<Answer> {
        if self.scheduler.is_held(npc_id) {
            anyhow::bail!(
                "npc {npc_id} is suspended or at rest, and answers nothing until it is let go"
            );
        }
        let minds = self
            .minds
            .read()
            .unwrap()
            .clone()
            .ok_or_else(|| anyhow::anyhow!("the engine is not loaded"))?;
        let persona = self
            .persona_of_async(npc_id)
            .await
            .ok_or_else(|| anyhow::anyhow!("npc {npc_id} has no persona"))?;
        let within = self.within(npc_id);
        minds
            .ask(npc_id, &persona.as_persona(), &within, question, choices)
            .await
    }
}
