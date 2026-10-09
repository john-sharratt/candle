//! Operations: the objectives the command table holds, each carried out as a
//! chain of missions by different Makers.
//!
//! **The table holds the objective; a mission is one stage of it.** A Maker
//! drafts the document the operation is for. The table reads the draft against
//! the record. A second Maker — never the one who drafted it — reviews it with
//! that reading in hand, mends what is wrong in it, and passes the operation,
//! or rejects it. A life event or story the review passes is then checked
//! against the main storyline — the eras — by a third Maker, who reads the
//! storyline around it and accepts it, mends it, or rejects it. A rejected
//! draft leaves the record, so a world's lore is only what three Makers and the
//! table have read.
//!
//! An operation is kept after it finishes, so what was tried, by whom, and why
//! it passed or failed can be read and edited from the operations tab.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

use super::operation_names::name_for;
use crate::engine::mission::Stage;

/// Where an operation stands.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Phase {
    /// Its draft mission waits at the table or is being carried.
    Drafting,
    /// The draft is written; the table reads it before a review is set.
    Reading,
    /// Its review mission waits at the table or is being carried.
    Reviewing,
    /// Passed on review; its check against the main storyline is set next.
    Reviewed,
    /// Its canon check waits at the table or is being carried.
    Checking,
    /// Every stage passed: the document stands.
    Succeeded,
    /// Rejected, or given up on; the reason is in `why`.
    Failed,
    /// Called off by an operator, or its mission was replaced.
    Cancelled,
}

impl Phase {
    /// Whether the operation is over.
    pub fn finished(self) -> bool {
        matches!(self, Phase::Succeeded | Phase::Failed | Phase::Cancelled)
    }
}

/// One stage carried out, as it was reported.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Entry {
    pub stage: Stage,
    /// Who carried it — a body id, or `table` for the table's own reading.
    pub by: String,
    /// `done`, `stuck`, `passed`, `rejected`, `read` or `cancelled`.
    pub outcome: String,
    pub notes: String,
}

/// One objective and what has become of it.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Operation {
    pub id: u64,
    /// What it is called — "Operation Iron Lantern".
    pub name: String,
    /// What it is for, in a line.
    pub objective: String,
    /// The generator that found the work.
    pub generator: String,
    /// The ledger key of the target it works on.
    pub target: String,
    /// The mind path of the document it produces.
    pub document: String,
    pub phase: Phase,
    /// Who drafted it, once somebody has taken the draft up.
    #[serde(default)]
    pub writer: Option<String>,
    /// Who reviewed it, once somebody has taken the review up.
    #[serde(default)]
    pub reviewer: Option<String>,
    /// Who checked it against the main storyline, once somebody has.
    #[serde(default)]
    pub checker: Option<String>,
    /// The table's reading of the draft, as the reviewer is given it.
    #[serde(default)]
    pub reading: Option<String>,
    /// Every stage carried out, in order.
    #[serde(default)]
    pub log: Vec<Entry>,
    /// Why it failed or was called off.
    #[serde(default)]
    pub why: Option<String>,
    /// How many reviews were reported stuck.
    #[serde(default)]
    pub reviews_stuck: u32,
    /// How many canon checks were reported stuck.
    #[serde(default)]
    pub checks_stuck: u32,
    /// Whether a failed operation's draft has been retired from memory.
    #[serde(default)]
    pub retired: bool,
    /// Whether the table's latest reading found it sound.
    #[serde(default)]
    pub sound: bool,
    /// How many times the table has read it.
    #[serde(default)]
    pub readings: u32,
    /// Whether a review has passed it — after which the table's reading of the
    /// mended text is what decides it.
    #[serde(default)]
    pub review_passed: bool,
    /// The brief the draft was written to — what it was to tell, which every
    /// later stage answers to as well.
    #[serde(default)]
    pub brief: String,
    /// What its document said when it opened, for one that works on a document
    /// the record already held — put back if the operation fails, so a
    /// rejected correction does not stand in the record. `None` for a draft.
    #[serde(default)]
    pub before: Option<String>,
}

/// How many stuck reviews an operation takes before it is failed.
pub const REVIEW_STUCK_LIMIT: u32 = 2;

/// How many times the table reads an operation's document before, still
/// finding faults in it, the operation fails.
pub const READINGS_LIMIT: u32 = 3;

/// What the table's reading of an operation's document leads to.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AfterReading {
    /// A review is set, with the reading in hand.
    Review,
    /// A review passed it and the table now finds it sound: it goes on to its
    /// check against the storyline, or stands.
    Stands,
    /// The table has read it [`READINGS_LIMIT`] times and still finds faults.
    Failed,
}

/// Whether a document is lore to be checked against the main storyline — a
/// life event or a story. A correction is checked by what it agrees with
/// already, and other documents are the storyline itself.
pub fn needs_canon(document: &str) -> bool {
    document.starts_with("layers/life/") || document.starts_with("layers/stories/")
}

impl Operation {
    fn log(&mut self, stage: Stage, by: &str, outcome: &str, notes: &str) {
        self.log.push(Entry {
            stage,
            by: by.to_string(),
            outcome: outcome.to_string(),
            notes: notes.to_string(),
        });
    }

    /// Whether its document leaves the record when it fails: a life event or a
    /// story — a draft, whether this operation drafted it or it was put
    /// through review by hand — but never a document a correction (`pair:`)
    /// works on, nor any other document put through review.
    ///
    /// **Only a draft leaves the record when it fails.** A failed correction
    /// moved the era it was correcting out of the record and retired it from
    /// memory: canon lost because one attempt to mend it did not succeed.
    pub fn leaves_on_failure(&self) -> bool {
        !self.target.starts_with("pair:") && needs_canon(&self.document)
    }

    /// Finish it, failed, for `why`.
    pub fn fail(&mut self, why: &str) {
        self.phase = Phase::Failed;
        self.why = Some(why.to_string());
    }
}

/// Every operation a world's table has held, by id.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Operations {
    /// The id the next operation takes.
    next: u64,
    ops: BTreeMap<u64, Operation>,
}

impl Operations {
    /// Open an operation for `target`, drafting `document`. Returns its id.
    pub fn open(&mut self, generator: &str, target: &str, objective: &str, document: &str) -> u64 {
        self.next += 1;
        let id = self.next;
        let name = name_for(id, &|n| self.ops.values().any(|o| o.name == n));
        self.ops.insert(
            id,
            Operation {
                id,
                name,
                objective: objective.to_string(),
                generator: generator.to_string(),
                target: target.to_string(),
                document: document.to_string(),
                phase: Phase::Drafting,
                writer: None,
                reviewer: None,
                reading: None,
                log: Vec::new(),
                why: None,
                reviews_stuck: 0,
                checker: None,
                checks_stuck: 0,
                retired: false,
                sound: false,
                readings: 0,
                review_passed: false,
                brief: String::new(),
                before: None,
            },
        );
        id
    }

    /// Keep `text` as what operation `id`'s document said when it opened.
    pub fn kept_before(&mut self, id: u64, text: &str) {
        if let Some(op) = self.ops.get_mut(&id) {
            op.before = Some(text.to_string());
        }
    }

    /// Keep `brief` as the one operation `id`'s draft was written to.
    pub fn briefed(&mut self, id: u64, brief: &str) {
        if let Some(op) = self.ops.get_mut(&id) {
            op.brief = brief.to_string();
        }
    }

    /// The failed operations whose documents have not yet been settled — set
    /// aside in the record and, for a draft, retired from memory — however
    /// they failed: rejected, read and found wanting past the limit, or stuck.
    pub fn to_retire(&self) -> Vec<Operation> {
        self.ops
            .values()
            .filter(|o| o.phase == Phase::Failed && !o.retired)
            .cloned()
            .collect()
    }

    /// Whether an operation opened after `op` on the same document has
    /// succeeded — whose accepted text putting `op`'s document back would
    /// overwrite.
    pub fn succeeded_after(&self, op: &Operation) -> bool {
        self.ops
            .values()
            .any(|o| o.id > op.id && o.document == op.document && o.phase == Phase::Succeeded)
    }

    /// Put a reviewing operation back to be read by the table. `false` when it
    /// is not reviewing.
    pub fn back_to_reading(&mut self, id: u64) -> bool {
        match self.ops.get_mut(&id) {
            Some(o) if o.phase == Phase::Reviewing => {
                o.phase = Phase::Reading;
                o.reviewer = None;
                true
            }
            _ => false,
        }
    }

    /// Record that operation `id`'s draft has been retired from memory.
    pub fn mark_retired(&mut self, id: u64) {
        if let Some(o) = self.ops.get_mut(&id) {
            o.retired = true;
        }
    }

    pub fn get(&self, id: u64) -> Option<&Operation> {
        self.ops.get(&id)
    }

    pub fn get_mut(&mut self, id: u64) -> Option<&mut Operation> {
        self.ops.get_mut(&id)
    }

    /// Every operation, newest first.
    pub fn all(&self) -> impl Iterator<Item = &Operation> {
        self.ops.values().rev()
    }

    /// The operations whose drafts wait for the table's reading, oldest first.
    pub fn awaiting_reading(&self) -> Vec<u64> {
        self.ops
            .values()
            .filter(|o| o.phase == Phase::Reading)
            .map(|o| o.id)
            .collect()
    }

    /// `body` took up `stage` of operation `id`.
    pub fn taken(&mut self, id: u64, stage: Stage, body: &str) {
        if let Some(o) = self.ops.get_mut(&id) {
            match stage {
                Stage::Draft => o.writer = Some(body.to_string()),
                Stage::Review => o.reviewer = Some(body.to_string()),
                Stage::Canon => o.checker = Some(body.to_string()),
            }
        }
    }

    /// Whether `body` may take up the next stage of operation `id`.
    ///
    /// **Nobody checks their own work, or checks it twice.** A Maker who has
    /// carried any stage of an operation takes no other stage of it: the draft
    /// never comes back to be reviewed by its writer, and a review reported
    /// stuck goes to somebody new rather than back to who could not do it.
    pub fn may_take(&self, id: u64, body: &str) -> bool {
        let Some(o) = self.ops.get(&id) else {
            return true;
        };
        o.writer.as_deref() != Some(body)
            && o.reviewer.as_deref() != Some(body)
            && o.checker.as_deref() != Some(body)
            && !o.log.iter().any(|e| e.by == body)
    }

    /// The draft was reported done: the table reads it next.
    pub fn drafted(&mut self, id: u64, body: &str, notes: &str) {
        if let Some(o) = self.ops.get_mut(&id) {
            o.log(Stage::Draft, body, "done", notes);
            o.phase = Phase::Reading;
        }
    }

    /// The draft was reported stuck: the operation fails.
    pub fn draft_stuck(&mut self, id: u64, body: &str, why: &str) {
        if let Some(o) = self.ops.get_mut(&id) {
            o.log(Stage::Draft, body, "stuck", why);
            o.fail(&format!("the draft could not be written: {why}"));
        }
    }

    /// The table read the document — `sound` or not — and what follows.
    ///
    /// **The table has the last word on what a review mended.** The table reads
    /// with the whole of what the work answers to in front of it — the
    /// subject's anchor, its other events, the era — and a review is one Maker
    /// mending in a few turns. A review used to pass straight on to the canon
    /// check whatever the table had found: the table failed a Zen given hands
    /// and a chair, the reviewer "mended" it, and it stood. Now a document a
    /// review passed against a reading that was not sound is read again: sound,
    /// it goes on; still faulted, it goes to another review; read
    /// [`READINGS_LIMIT`] times and still faulted, the operation fails with the
    /// table's own words.
    pub fn table_read(&mut self, id: u64, sound: bool, reading: &str) -> AfterReading {
        let Some(o) = self.ops.get_mut(&id) else {
            return AfterReading::Failed;
        };
        o.readings += 1;
        o.sound = sound;
        o.log(Stage::Review, "table", "read", reading);
        if o.review_passed {
            if sound {
                o.phase = match needs_canon(&o.document) {
                    true => Phase::Reviewed,
                    false => Phase::Succeeded,
                };
                return AfterReading::Stands;
            }
            if o.readings >= READINGS_LIMIT {
                let n = o.readings;
                o.fail(&format!(
                    "the table read it {n} times, mended between, and still finds it wanting: \
                     {reading}"
                ));
                return AfterReading::Failed;
            }
        }
        o.reading = Some(reading.to_string());
        o.phase = Phase::Reviewing;
        o.reviewer = None;
        AfterReading::Review
    }

    /// The review passed it.
    ///
    /// Against a reading that was not sound, the mended text goes back to the
    /// table to be read again (see [`Self::table_read`]). Otherwise a life event
    /// or a story goes on to be checked against the main storyline
    /// ([`needs_canon`]), and anything else stands now. Answers whether the
    /// operation succeeded.
    pub fn passed(&mut self, id: u64, body: &str, notes: &str) -> bool {
        let Some(o) = self.ops.get_mut(&id) else {
            return false;
        };
        o.log(Stage::Review, body, "passed", notes);
        o.review_passed = true;
        if !o.sound {
            o.phase = Phase::Reading;
            o.reviewer = None;
            return false;
        }
        o.phase = match needs_canon(&o.document) {
            true => Phase::Reviewed,
            false => Phase::Succeeded,
        };
        o.phase == Phase::Succeeded
    }

    /// The operations whose canon check is owed, oldest first.
    pub fn awaiting_canon(&self) -> Vec<u64> {
        self.ops
            .values()
            .filter(|o| o.phase == Phase::Reviewed)
            .map(|o| o.id)
            .collect()
    }

    /// Send a succeeded life event or story to be checked against the main
    /// storyline — lore that stood before the check existed, or after the
    /// storyline changed. `false` when it is not succeeded lore.
    pub fn recheck(&mut self, id: u64) -> bool {
        match self.ops.get_mut(&id) {
            Some(o) if o.phase == Phase::Succeeded && needs_canon(&o.document) => {
                o.phase = Phase::Reviewed;
                o.checks_stuck = 0;
                true
            }
            _ => false,
        }
    }

    /// Its canon check is on the table.
    pub fn checking(&mut self, id: u64) {
        if let Some(o) = self.ops.get_mut(&id) {
            o.phase = Phase::Checking;
            o.checker = None;
        }
    }

    /// The canon check accepted it: the document stands.
    pub fn canon_passed(&mut self, id: u64, body: &str, notes: &str) {
        if let Some(o) = self.ops.get_mut(&id) {
            o.log(Stage::Canon, body, "passed", notes);
            o.phase = Phase::Succeeded;
        }
    }

    /// The canon check was reported stuck: it is set again for somebody else —
    /// `false` once that has happened [`REVIEW_STUCK_LIMIT`] times, and the
    /// operation has failed.
    pub fn canon_stuck(&mut self, id: u64, body: &str, why: &str) -> bool {
        let Some(o) = self.ops.get_mut(&id) else {
            return false;
        };
        o.log(Stage::Canon, body, "stuck", why);
        o.checks_stuck += 1;
        if o.checks_stuck >= REVIEW_STUCK_LIMIT {
            o.fail(&format!(
                "no check against the storyline could be carried out: {why}"
            ));
            return false;
        }
        o.phase = Phase::Reviewed;
        o.checker = None;
        true
    }

    /// The review or the canon check (`stage`) rejected it.
    pub fn rejected(&mut self, id: u64, stage: Stage, body: &str, why: &str) {
        if let Some(o) = self.ops.get_mut(&id) {
            o.log(stage, body, "rejected", why);
            let at = match stage {
                Stage::Canon => "against the main storyline",
                _ => "on review",
            };
            o.fail(&format!("rejected {at}: {why}"));
        }
    }

    /// The review was reported stuck. The draft goes back to be read and
    /// reviewed again by somebody else — `false` once that has happened
    /// [`REVIEW_STUCK_LIMIT`] times, and the operation has failed.
    pub fn review_stuck(&mut self, id: u64, body: &str, why: &str) -> bool {
        let Some(o) = self.ops.get_mut(&id) else {
            return false;
        };
        o.log(Stage::Review, body, "stuck", why);
        o.reviews_stuck += 1;
        if o.reviews_stuck >= REVIEW_STUCK_LIMIT {
            o.fail(&format!("no review could be carried out: {why}"));
            return false;
        }
        o.phase = Phase::Reading;
        o.reviewer = None;
        true
    }

    /// Call it off, for `why`. `false` when it was already over.
    pub fn cancel(&mut self, id: u64, why: &str) -> bool {
        match self.ops.get_mut(&id) {
            Some(o) if !o.phase.finished() => {
                o.phase = Phase::Cancelled;
                o.why = Some(why.to_string());
                true
            }
            _ => false,
        }
    }

    /// Rename it. Refused when another operation already has the name.
    pub fn rename(&mut self, id: u64, name: &str) -> Result<(), String> {
        let name = name.trim();
        if name.is_empty() {
            return Err("an operation needs a name".into());
        }
        if self.ops.values().any(|o| o.id != id && o.name == name) {
            return Err(format!("another operation is already called {name}"));
        }
        let o = self.ops.get_mut(&id).ok_or("no such operation")?;
        o.name = name.to_string();
        Ok(())
    }

    /// Restate what it is for.
    pub fn set_objective(&mut self, id: u64, objective: &str) -> Result<(), String> {
        let objective = objective.trim();
        if objective.is_empty() {
            return Err("an operation needs an objective".into());
        }
        let o = self.ops.get_mut(&id).ok_or("no such operation")?;
        o.objective = objective.to_string();
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn one() -> (Operations, u64) {
        let mut ops = Operations::default();
        let id = ops.open(
            "life-event",
            "life:creed",
            "Creed's life, 2950: The Silence Between Orders",
            "layers/life/creed/2950 The Silence Between Orders.md",
        );
        (ops, id)
    }

    /// **Draft, reading, review, passed** — and the review never goes to the
    /// Maker who drafted it.
    #[test]
    fn an_operation_runs_draft_reading_review_and_passes() {
        let (mut ops, id) = one();
        assert_eq!(ops.get(id).unwrap().phase, Phase::Drafting);
        assert_eq!(ops.get(id).unwrap().name, "Operation Iron Lantern");
        ops.taken(id, Stage::Draft, "wren");
        ops.drafted(id, "wren", "written");
        assert_eq!(ops.awaiting_reading(), vec![id]);
        ops.table_read(id, true, "sound; two sentences repeat");
        assert_eq!(ops.get(id).unwrap().phase, Phase::Reviewing);
        assert!(!ops.may_take(id, "wren"), "not its own draft");
        assert!(ops.may_take(id, "pax"));
        ops.taken(id, Stage::Review, "pax");
        assert!(
            !ops.passed(id, "pax", "mended the repeat"),
            "a life goes to canon"
        );
        assert_eq!(ops.get(id).unwrap().phase, Phase::Reviewed);
        assert_eq!(ops.awaiting_canon(), vec![id]);
        ops.checking(id);
        assert!(!ops.may_take(id, "wren"));
        assert!(!ops.may_take(id, "pax"));
        assert!(ops.may_take(id, "bram"), "a third Maker checks it");
        ops.taken(id, Stage::Canon, "bram");
        ops.canon_passed(id, "bram", "agrees with the Tower Age");
        let o = ops.get(id).unwrap();
        assert_eq!(o.phase, Phase::Succeeded);
        assert_eq!(o.reviewer.as_deref(), Some("pax"));
        assert_eq!(o.checker.as_deref(), Some("bram"));
        let outcomes: Vec<&str> = o.log.iter().map(|e| e.outcome.as_str()).collect();
        assert_eq!(outcomes, ["done", "read", "passed", "passed"]);
    }

    /// **The table has the last word on what a review mended.** A draft the
    /// table found wanting and a review passed goes back to the table: still
    /// wanting, another review; sound, on to its check against the storyline;
    /// read the limit and still wanting, failed in the table's words. A Zen
    /// given hands and a chair, which the table failed, stood because the review
    /// passed it straight on.
    #[test]
    fn what_a_review_passed_against_a_failing_reading_is_read_again() {
        let (mut ops, id) = one();
        ops.drafted(id, "wren", "written");
        assert_eq!(
            ops.table_read(id, false, "failing: Zen has hands"),
            AfterReading::Review
        );
        ops.taken(id, Stage::Review, "pax");
        assert!(!ops.passed(id, "pax", "mended"));
        assert_eq!(ops.get(id).unwrap().phase, Phase::Reading, "read again");
        assert!(ops.awaiting_canon().is_empty(), "not on to canon");

        // Still wanting: another review, by somebody new.
        assert_eq!(
            ops.table_read(id, false, "failing: still hands"),
            AfterReading::Review
        );
        assert!(!ops.may_take(id, "pax"));
        ops.taken(id, Stage::Review, "bram");
        ops.passed(id, "bram", "mended again");
        // Sound now: on to its check against the storyline.
        assert_eq!(ops.table_read(id, true, "sound"), AfterReading::Stands);
        assert_eq!(ops.awaiting_canon(), vec![id]);

        // And one never made sound fails, in the table's words.
        let (mut ops, id) = one();
        ops.drafted(id, "wren", "written");
        for (n, reviewer) in ["pax", "bram"].iter().enumerate() {
            assert_eq!(
                ops.table_read(id, false, "failing"),
                AfterReading::Review,
                "reading {n}"
            );
            ops.taken(id, Stage::Review, reviewer);
            ops.passed(id, reviewer, "mended");
        }
        assert_eq!(
            ops.table_read(id, false, "failing: a body it does not have"),
            AfterReading::Failed
        );
        let o = ops.get(id).unwrap();
        assert_eq!(o.phase, Phase::Failed);
        assert!(o
            .why
            .as_deref()
            .unwrap()
            .ends_with("a body it does not have"));
    }

    /// A sound reading and a pass go straight on: nothing is read twice that
    /// the table had no fault with.
    #[test]
    fn a_sound_reading_passed_goes_straight_on() {
        let (mut ops, id) = one();
        ops.drafted(id, "wren", "written");
        assert_eq!(ops.table_read(id, true, "sound"), AfterReading::Review);
        ops.taken(id, Stage::Review, "pax");
        ops.passed(id, "pax", "read it");
        assert_eq!(ops.awaiting_canon(), vec![id]);
    }

    /// **Only lore is checked against the storyline**: a correction to an era
    /// stands once it is reviewed. A stuck check goes to somebody new, and
    /// twice stuck fails; a check can reject.
    #[test]
    fn a_canon_check_is_for_lore_and_can_be_stuck_or_reject() {
        let mut ops = Operations::default();
        let fix = ops.open(
            "contradiction",
            "pair:a|b",
            "Correct an era",
            "layers/eras/a.md",
        );
        ops.drafted(fix, "wren", "changed");
        ops.table_read(fix, true, "r");
        assert!(ops.passed(fix, "pax", "agrees now"));
        assert_eq!(ops.get(fix).unwrap().phase, Phase::Succeeded);

        let (mut ops, id) = one();
        ops.drafted(id, "wren", "written");
        ops.table_read(id, true, "r");
        ops.passed(id, "pax", "ok");
        ops.checking(id);
        assert!(ops.canon_stuck(id, "bram", "no desk"));
        assert_eq!(ops.awaiting_canon(), vec![id]);
        ops.checking(id);
        ops.rejected(id, Stage::Canon, "yen", "it has the war won by the Houses");
        let o = ops.get(id).unwrap();
        assert_eq!(o.phase, Phase::Failed);
        assert_eq!(
            o.why.as_deref(),
            Some("rejected against the main storyline: it has the war won by the Houses")
        );
        assert!(needs_canon("layers/stories/x.md") && !needs_canon("layers/eras/x.md"));

        // Lore that stood before the check can be sent to it; a correction
        // cannot.
        let (mut ops, id) = one();
        ops.drafted(id, "wren", "written");
        ops.table_read(id, true, "r");
        ops.passed(id, "pax", "ok");
        ops.checking(id);
        ops.canon_passed(id, "bram", "agrees");
        assert!(ops.recheck(id));
        assert_eq!(ops.awaiting_canon(), vec![id]);
        assert!(!ops.recheck(id), "already on its way");
        let fix = ops.open("contradiction", "pair:a|b", "fix", "layers/eras/a.md");
        assert!(!ops.recheck(fix));
    }

    #[test]
    fn a_rejection_or_too_many_stuck_reviews_fail_it() {
        let (mut ops, id) = one();
        ops.drafted(id, "wren", "written");
        ops.table_read(id, true, "r");
        ops.rejected(
            id,
            Stage::Review,
            "pax",
            "it is set in the wrong era throughout",
        );
        let o = ops.get(id).unwrap();
        assert_eq!(o.phase, Phase::Failed);
        assert_eq!(
            o.why.as_deref(),
            Some("rejected on review: it is set in the wrong era throughout")
        );
        // Its draft is owed retirement from memory, once.
        let settle: Vec<(u64, String)> = ops
            .to_retire()
            .into_iter()
            .map(|o| (o.id, o.document))
            .collect();
        assert_eq!(
            settle,
            vec![(
                id,
                "layers/life/creed/2950 The Silence Between Orders.md".to_string()
            )]
        );
        ops.mark_retired(id);
        assert!(ops.to_retire().is_empty());

        // A failed correction is settled too — its era put back, not moved.
        let fix = ops.open("contradiction", "pair:a|b", "fix", "layers/eras/a.md");
        ops.get_mut(fix).unwrap().fail("rejected on review");
        assert!(!ops.get(fix).unwrap().leaves_on_failure());
        assert_eq!(ops.to_retire().len(), 1);
        // A later correction of the same era that succeeded is not overwritten.
        let later = ops.open("contradiction", "pair:a|c", "fix", "layers/eras/a.md");
        ops.get_mut(later).unwrap().phase = Phase::Succeeded;
        assert!(ops.succeeded_after(ops.get(fix).unwrap()));
        assert!(!ops.succeeded_after(ops.get(later).unwrap()));
        let by_hand = ops.open("operator", "doc:layers/eras/a.md", "x", "layers/eras/a.md");
        assert!(!ops.get(by_hand).unwrap().leaves_on_failure());
        let story = ops.open(
            "operator",
            "doc:layers/stories/s.md",
            "x",
            "layers/stories/s.md",
        );
        assert!(ops.get(story).unwrap().leaves_on_failure());

        let (mut ops, id) = one();
        ops.drafted(id, "wren", "written");
        ops.table_read(id, true, "r");
        ops.taken(id, Stage::Review, "pax");
        assert!(ops.review_stuck(id, "pax", "no desk"), "read again");
        assert_eq!(ops.get(id).unwrap().phase, Phase::Reading);
        assert!(
            !ops.may_take(id, "pax"),
            "not back to the one who was stuck"
        );
        assert!(!ops.may_take(id, "wren"), "nor to the writer");
        assert!(ops.may_take(id, "bram"));
        ops.table_read(id, true, "r");
        assert!(!ops.review_stuck(id, "bram", "no desk"));
        assert_eq!(ops.get(id).unwrap().phase, Phase::Failed);
    }

    #[test]
    fn names_are_unique_and_edits_are_checked() {
        let (mut ops, a) = one();
        let b = ops.open("untold", "era:x", "a story", "layers/stories/x.md");
        assert_ne!(ops.get(a).unwrap().name, ops.get(b).unwrap().name);
        let taken = ops.get(a).unwrap().name.clone();
        assert!(ops.rename(b, &taken).is_err());
        assert!(ops.rename(b, " ").is_err());
        assert!(ops.rename(b, "Operation Quiet Harbour").is_ok());
        assert!(ops.set_objective(b, "").is_err());
        assert!(ops.cancel(b, "not wanted"));
        assert!(!ops.cancel(b, "again"), "already over");
        let newest: Vec<u64> = ops.all().map(|o| o.id).collect();
        assert_eq!(newest, vec![b, a]);
    }
}
