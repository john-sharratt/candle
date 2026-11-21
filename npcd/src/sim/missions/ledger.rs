//! Records every target the mission generator has found, so the same work is not
//! set twice and settled work waits until it changes.
//!
//! A target is held while a mission for it is being written, so two generations
//! do not choose it at once, and it returns to its earlier state if no mission
//! comes of it. A target that is in hand, done, or found to need nothing is not
//! worked again until what it is about changes. A target reported stuck is worked
//! again, but only until it has been reported stuck the limit number of times on
//! the same text.

use super::*;

/// What has become of a piece of work the generator found.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Settled {
    /// A mission for it waits at the table.
    Pooled,
    /// Somebody has taken it up.
    Carried,
    /// Reported done.
    Done,
    /// Reported as not done.
    Stuck,
    /// The generator looked and found nothing to do.
    Nothing,
}

/// One target in the ledger.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Ledgered {
    pub state: Settled,
    /// What the target held when it was last settled — see
    /// `engine::mission_gen::target::Target::fingerprint`.
    pub fingerprint: u64,
    /// Which generator found it.
    pub generator: String,
    /// The mission's brief, or why there was none.
    pub note: String,
    /// How many times a mission for it has been reported stuck while it held
    /// this fingerprint.
    #[serde(default)]
    pub stuck: u32,
    /// The document its mission writes, when it writes one.
    #[serde(default)]
    pub writes: Option<String>,
}

/// How many stuck reports a target takes before it is left until it changes.
pub(super) const STUCK_LIMIT: u32 = 2;

impl Ledgered {
    /// Whether a target with `fingerprint` is not to be worked now.
    ///
    /// In hand is always blocked. Done, or found to need nothing, is blocked
    /// until what it is about changes — a life gains an event, a document is
    /// edited. Stuck is worked again, until it has been stuck
    /// [`STUCK_LIMIT`] times on the same text.
    pub fn blocks(&self, fingerprint: u64) -> bool {
        match self.state {
            Settled::Pooled | Settled::Carried => true,
            Settled::Done | Settled::Nothing => self.fingerprint == fingerprint,
            Settled::Stuck => self.fingerprint == fingerprint && self.stuck >= STUCK_LIMIT,
        }
    }
}

impl Missions {
    /// Hold `key` while a mission is being written for it, so a second
    /// generation running at the same time does not choose it too. `false`
    /// when it is already blocked. Released by [`Self::release`] if no mission
    /// comes of it; settled by [`Self::offer`] or [`Self::decline`] if one does.
    pub fn reserve(&mut self, key: &str, generator: &str, fingerprint: u64) -> bool {
        if self.blocks(key, fingerprint) {
            return false;
        }
        let stuck = self
            .targets
            .get(key)
            .filter(|e| e.fingerprint == fingerprint)
            .map_or(0, |e| e.stuck);
        self.targets.insert(
            key.to_string(),
            Ledgered {
                state: Settled::Pooled,
                fingerprint,
                generator: generator.to_string(),
                note: "a mission is being written for it".to_string(),
                stuck,
                writes: None,
            },
        );
        true
    }

    /// Let go of a target [`Self::reserve`] held, when no mission came of it.
    ///
    /// **A target that was stuck goes back to stuck, count and all.** Forgotten
    /// instead, a target whose generation kept failing after the reservation
    /// would never reach [`STUCK_LIMIT`] however often it was reported stuck.
    pub fn release(&mut self, key: &str) {
        if self.pool.iter().any(|m| target_of(m) == Some(key)) {
            return;
        }
        let Some(entry) = self.targets.get_mut(key) else {
            return;
        };
        if entry.state != Settled::Pooled || entry.writes.is_some() {
            return;
        }
        if entry.stuck > 0 {
            entry.state = Settled::Stuck;
            entry.note = "released: no mission came of it".to_string();
        } else {
            self.targets.remove(key);
        }
    }

    /// Record that a generator looked at a target and found nothing to do.
    pub fn decline(&mut self, key: &str, generator: &str, fingerprint: u64, why: &str) {
        self.targets.insert(
            key.to_string(),
            Ledgered {
                state: Settled::Nothing,
                fingerprint,
                generator: generator.to_string(),
                note: why.to_string(),
                stuck: 0,
                writes: None,
            },
        );
    }

    /// Whether the target `key`, holding `fingerprint`, is not to be worked now.
    pub fn blocks(&self, key: &str, fingerprint: u64) -> bool {
        self.targets.get(key).is_some_and(|e| e.blocks(fingerprint))
    }

    /// Every target the generator has found, and what became of it.
    pub fn targets(&self) -> &BTreeMap<String, Ledgered> {
        &self.targets
    }

    /// Forget every target that is not in somebody's hands — done, stuck and
    /// found-to-need-nothing alike — so all of it can be found again. Returns
    /// how many were forgotten.
    pub fn forget_settled(&mut self) -> usize {
        let before = self.targets.len();
        self.targets
            .retain(|_, e| matches!(e.state, Settled::Pooled | Settled::Carried));
        before - self.targets.len()
    }

    /// Settle the ledger entry of the target a mission that is not an
    /// operation's is about.
    pub(super) fn settle(&mut self, mission: &Mission, outcome: Outcome, notes: &str) {
        let key = target_of(mission).map(str::to_string);
        self.settle_target(key, outcome, notes);
    }

    pub(super) fn settle_target(&mut self, key: Option<String>, outcome: Outcome, notes: &str) {
        if let Some(entry) = key.and_then(|k| self.targets.get_mut(&k)) {
            match outcome {
                Outcome::Pass => entry.state = Settled::Done,
                Outcome::Fail => {
                    entry.state = Settled::Stuck;
                    entry.stuck += 1;
                }
            }
            entry.note = notes.to_string();
        }
    }
}
