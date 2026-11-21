//! Runtime-wide settings the daemon installs once at startup and reads after.
//!
//! Each setting is supplied after construction, because the registries that
//! provide it live in the authored state, and that state needs this runtime to
//! answer its own routes. Until a setting is installed, whatever depends on it is
//! refused or left empty rather than guessed at. The world clock is read per
//! character, so a character in a world paced differently is never handed another
//! world's instant.

use super::*;

impl Runtime {
    /// The agent client a device-bridge job is submitted through, when this
    /// member can reach one (`docs/zen_moon_mind.md` §15). `None` on a daemon
    /// that reaches no agent — a bridge act is then refused, not run here.
    pub fn agents(&self) -> Option<Agents> {
        self.agents.get().cloned()
    }

    /// Keep the agent client, so an NPC's effector verb and the operator route
    /// can hand a bridge job to an in-network agent.
    pub fn set_agents(&self, agents: Agents) {
        let _ = self.agents.set(agents);
    }

    /// The machines the house's agents run on, as last heard of — what a
    /// machine's terminal is listed and connected to from.
    pub fn machines(&self) -> Vec<Machine> {
        self.machines.read().unwrap().clone()
    }

    /// Keep `machines` as the machines last heard of, each with the checks it
    /// had listed when it was last connected to.
    pub(crate) fn set_machines(&self, mut machines: Vec<Machine>) {
        let mut kept = self.machines.write().unwrap();
        for m in &mut machines {
            if let Some(was) = kept.iter().find(|k| k.node == m.node) {
                if m.checks.is_empty() {
                    m.checks = was.checks.clone();
                }
            }
        }
        *kept = machines;
    }

    /// Supply the world-clock resolver. Called once at startup, before the
    /// driver runs.
    pub fn set_clock(&self, clock: WorldClock) {
        *self.clock.write().unwrap() = Some(clock);
    }

    /// Install the log of the guardian that is running.
    pub fn set_guardian_log(&self, log: Arc<GuardianLog>) {
        *self.guardian_log.write().unwrap() = Some(log);
    }

    /// The running guardian's log; `None` when there is no guardian.
    pub fn guardian_log(&self) -> Option<Arc<GuardianLog>> {
        self.guardian_log.read().unwrap().clone()
    }

    /// Supply the character-state resolver. Called once at startup, before the
    /// driver runs.
    pub fn set_persona_source(&self, src: PersonaSource) {
        *self.persona.write().unwrap() = Some(src);
    }

    /// Supply the registers a character may say it is in — the ids of whatever
    /// `<mind>/moods/` holds. Called once at startup, before the driver runs.
    ///
    /// The engine offers what it is given and has no opinion about the list:
    /// which registers a world's people have is that world's content, and a
    /// second copy compiled in here would be free to disagree with the mind.
    pub fn set_feelings(&self, feelings: Vec<String>) {
        *self.feelings.write().unwrap() = feelings;
    }

    /// The registers this daemon's mind holds. What the probe builds its
    /// grammar from, so a scenario runs against the same vocabulary a live
    /// character does.
    pub fn feelings(&self) -> Vec<String> {
        self.feelings.read().unwrap().clone()
    }

    /// Supply the sink that records how characters feel. Called once at
    /// startup, before the driver runs.
    pub fn set_mood_sink(&self, sink: MoodSink) {
        *self.mood_sink.write().unwrap() = Some(sink);
    }

    /// Supply the effector device's token store. Called once at startup, before
    /// the `local` router is built from it.
    pub fn set_tokens(&self, tokens: Arc<Tokens>) {
        *self.tokens.write().unwrap() = Some(tokens);
    }

    /// The token store, for a caller building the `local` router's auth layer.
    pub fn tokens(&self) -> Option<Arc<Tokens>> {
        self.tokens.read().unwrap().clone()
    }

    /// What time it is in this character's world. The resolver reads a
    /// lock-free snapshot of the worlds' clocks, so any thread may ask — the
    /// tick driver and a character task alike.
    ///
    /// Zero before a clock is installed — a window between construction and
    /// startup that the driver is not running in, so nothing reads it.
    pub fn world_ms(&self, npc_id: u64) -> u64 {
        let g = self.clock.read().unwrap();
        g.as_ref().map_or(0, |f| f(npc_id))
    }
}
