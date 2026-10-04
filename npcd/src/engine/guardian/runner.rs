//! The guardian itself: on a timer, read every character, ask what the
//! modules want asked, and press on the ones that are unwell.

use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use std::time::{Duration, Instant};

use tokio::time::{interval, MissedTickBehavior};

use crate::engine::guardian::ladder::{Ladder, NpcState, Resolution, Rung};
use crate::engine::guardian::log::GuardianLog;
use crate::engine::guardian::module::Module;
use crate::engine::guardian::nudge;
use crate::engine::guardian::probe::Probe;
use crate::engine::guardian::view::{Concern, MissionView, NpcView, Reply, Verdict};

type Mark = (String, usize, usize);

/// How many of a character's latest acts a module is shown.
const RECENT_ACTS: usize = 12;

/// What the guardian remembers of one character between scans.
#[derive(Default)]
struct Tracked {
    ladder: NpcState,
    mark: Option<(Mark, Duration)>,
}

pub struct Guardian {
    pub(super) modules: Vec<Box<dyn Module>>,
    pub(super) ladder: Ladder,
    pub(super) scan_every: Duration,
    pub(super) log: Arc<GuardianLog>,
    tracked: HashMap<u64, Tracked>,
}

impl Guardian {
    pub(super) fn new(
        modules: Vec<Box<dyn Module>>,
        ladder: Ladder,
        scan_every: Duration,
        log: Arc<GuardianLog>,
    ) -> Self {
        Self {
            modules,
            ladder,
            scan_every,
            log,
            tracked: HashMap::new(),
        }
    }

    /// What the guardian has done, for an operator.
    pub fn log(&self) -> Arc<GuardianLog> {
        self.log.clone()
    }

    /// Scan the cast every `scan_every` until the engine stops.
    pub async fn run<P: Probe + 'static>(mut self, probe: Arc<P>) {
        let start = Instant::now();
        let mut timer = interval(self.scan_every);
        timer.set_missed_tick_behavior(MissedTickBehavior::Delay);
        loop {
            timer.tick().await;
            if probe.stopping() {
                return;
            }
            if !probe.ready() {
                continue;
            }
            self.scan(&*probe, start.elapsed()).await;
        }
    }

    /// One pass over the cast at time `now`.
    pub async fn scan<P: Probe>(&mut self, probe: &P, now: Duration) {
        let npcs = probe.npcs();
        let present: HashSet<u64> = npcs.iter().copied().collect();
        self.tracked.retain(|id, _| present.contains(id));
        for npc_id in npcs {
            self.scan_one(probe, npc_id, now).await;
        }
    }

    async fn scan_one<P: Probe>(&mut self, probe: &P, npc_id: u64, now: Duration) {
        let mission = probe.mission(npc_id);
        let tracked = self.tracked.entry(npc_id).or_default();
        let since_progress = since_progress(tracked, mission.as_ref(), now);
        if self.ladder.settling(&tracked.ladder, now) {
            return;
        }
        let view = NpcView {
            npc_id,
            mission,
            since_progress,
            stations: probe.stations(npc_id),
            recent_acts: probe.recent_acts(npc_id, RECENT_ACTS),
            journal: probe.journal(npc_id),
        };

        let mut concern = None;
        for module in &self.modules {
            let reply = match module.question(&view) {
                Some(question) => match probe.ask(npc_id, &question).await {
                    Ok(reply) => Some(reply),
                    Err(e) => {
                        self.log
                            .push(npc_id, "failed", format!("{}: ask: {e}", module.name()));
                        continue;
                    }
                },
                None => None,
            };
            let answer = reply.as_ref().map(|r| r.answer.as_str());
            match module.judge(&view, answer) {
                Verdict::Healthy => {}
                Verdict::Concern(c) => {
                    self.log.push(
                        npc_id,
                        "check",
                        format!("{}: {}", module.name(), c.as_str()),
                    );
                    concern.get_or_insert(c);
                }
                Verdict::TickStep(step, outcome) => {
                    if probe.tick_step(npc_id, &step, outcome) {
                        self.log
                            .push(npc_id, "tick", format!("{}: {step}", outcome.as_str()));
                    }
                }
                Verdict::Journal { worth } => {
                    let reply = reply.unwrap_or_else(|| Reply {
                        answer: String::new(),
                        reason: String::new(),
                        ms: 0,
                    });
                    if probe.decide_journal(npc_id, worth, &reply) {
                        let kind = if worth { "journal" } else { "journal_skip" };
                        self.log.push(npc_id, kind, reply.reason);
                    }
                }
            }
        }

        let tracked = self.tracked.entry(npc_id).or_default();
        let advance = self
            .ladder
            .advance(&mut tracked.ladder, now, concern.is_some());
        match advance.resolution {
            Some(Resolution::Held) => self.log.push(npc_id, "held", "well again"),
            Some(Resolution::Failed) => self.log.push(npc_id, "failed", "still unwell"),
            None => {}
        }
        if let (Some(rung), Some(concern)) = (advance.act, concern) {
            self.apply(probe, rung, concern, &view).await;
        }
    }

    async fn apply<P: Probe>(&self, probe: &P, rung: Rung, concern: Concern, view: &NpcView) {
        let npc_id = view.npc_id;
        let Some(text) = nudge::text(rung, concern, view) else {
            self.log.push(npc_id, rung.as_str(), concern.as_str());
            return;
        };
        let sent = match rung {
            Rung::Refresh => probe.rouse(npc_id, text.clone()).await,
            _ => probe.think(npc_id, text.clone()).await,
        };
        let detail = if sent {
            text
        } else {
            format!("not delivered: {text}")
        };
        self.log.push(npc_id, rung.as_str(), detail);
    }
}

/// How long the mission's progress mark has stood still, updating the mark.
fn since_progress(tracked: &mut Tracked, mission: Option<&MissionView>, now: Duration) -> Duration {
    let Some(mission) = mission else {
        tracked.mark = None;
        return Duration::ZERO;
    };
    let mark = mission.progress_mark();
    match &tracked.mark {
        Some((known, at)) if *known == mark => now.saturating_sub(*at),
        _ => {
            tracked.mark = Some((mark, now));
            Duration::ZERO
        }
    }
}

#[cfg(test)]
mod tests {
    use std::future::Future;
    use std::sync::Mutex;

    use super::*;
    use crate::engine::guardian::builder::GuardianBuilder;
    use crate::engine::guardian::modules::{Drift, Journal, Looping, StepTracker};
    use crate::engine::guardian::view::{Question, Station, Step};
    use crate::engine::journal::state::Waiting;
    use crate::engine::mission::StepOutcome;

    /// A cast of one whose answers and mission are scripted, recording what the
    /// guardian does to it.
    #[derive(Default)]
    struct Cast {
        stations: Mutex<Vec<Station>>,
        mission: Mutex<Option<MissionView>>,
        answers: Mutex<HashMap<String, String>>,
        thoughts: Mutex<Vec<String>>,
        rousings: Mutex<Vec<String>>,
        ticked: Mutex<Vec<(String, StepOutcome)>>,
        acts: Mutex<Vec<String>>,
        waiting: Mutex<Option<Waiting>>,
        decided: Mutex<Vec<(bool, String)>>,
    }

    impl Cast {
        fn carrying(steps: &[&str]) -> Self {
            let cast = Cast::default();
            *cast.mission.lock().unwrap() = Some(MissionView {
                prompt: "Find the ledger.".into(),
                steps: steps
                    .iter()
                    .map(|s| Step {
                        text: s.to_string(),
                        done: false,
                        reports: false,
                    })
                    .collect(),
                standing: "What has been asked of you: Find the ledger.".into(),
            });
            cast
        }

        fn answers(&self, containing: &str, answer: &str) {
            self.answers
                .lock()
                .unwrap()
                .insert(containing.to_string(), answer.to_string());
        }
    }

    impl Probe for Cast {
        fn npcs(&self) -> Vec<u64> {
            vec![7]
        }
        fn ready(&self) -> bool {
            true
        }
        fn stopping(&self) -> bool {
            false
        }
        fn mission(&self, _: u64) -> Option<MissionView> {
            self.mission.lock().unwrap().clone()
        }
        fn stations(&self, _: u64) -> Vec<Station> {
            self.stations.lock().unwrap().clone()
        }
        fn recent_acts(&self, _: u64, limit: usize) -> Vec<String> {
            let acts = self.acts.lock().unwrap();
            let skip = acts.len().saturating_sub(limit);
            acts[skip..].to_vec()
        }
        fn journal(&self, _: u64) -> Option<Waiting> {
            self.waiting.lock().unwrap().clone()
        }
        fn ask(
            &self,
            _: u64,
            question: &Question,
        ) -> impl Future<Output = anyhow::Result<Reply>> + Send {
            let answer = self
                .answers
                .lock()
                .unwrap()
                .iter()
                .find(|(key, _)| question.text.contains(key.as_str()))
                .map(|(_, a)| a.clone());
            async move {
                answer
                    .map(|answer| Reply {
                        answer,
                        reason: "because".into(),
                        ms: 5,
                    })
                    .ok_or_else(|| anyhow::anyhow!("no scripted answer"))
            }
        }
        fn decide_journal(&self, _: u64, worth: bool, reply: &Reply) -> bool {
            self.decided
                .lock()
                .unwrap()
                .push((worth, reply.reason.clone()));
            true
        }
        fn think(&self, _: u64, text: String) -> impl Future<Output = bool> + Send {
            self.thoughts.lock().unwrap().push(text);
            async { true }
        }
        fn rouse(&self, _: u64, text: String) -> impl Future<Output = bool> + Send {
            self.rousings.lock().unwrap().push(text);
            async { true }
        }
        fn tick_step(&self, _: u64, step: &str, outcome: StepOutcome) -> bool {
            self.ticked
                .lock()
                .unwrap()
                .push((step.to_string(), outcome));
            true
        }
    }

    fn secs(n: u64) -> Duration {
        Duration::from_secs(n)
    }

    fn guardian(modules: Vec<Box<dyn Module>>, rungs: Vec<Rung>) -> Guardian {
        let mut b = GuardianBuilder::new()
            .scan_every(secs(10))
            .cooldown(secs(120))
            .settle(secs(30))
            .escalation(rungs);
        for m in modules {
            b = b.module(m);
        }
        b.build().unwrap()
    }

    fn kinds(g: &Guardian) -> Vec<&'static str> {
        g.log().records().iter().map(|r| r.kind).collect()
    }

    #[tokio::test]
    async fn a_character_off_its_errand_is_nudged_with_the_errand() {
        let cast = Cast::carrying(&["read the ledger"]);
        cast.answers("Find the ledger.", "something else");
        let mut g = guardian(vec![Box::new(Drift)], vec![Rung::Nudge, Rung::Flag]);
        g.scan(&cast, secs(0)).await;
        let thoughts = cast.thoughts.lock().unwrap();
        assert_eq!(thoughts.len(), 1);
        assert!(thoughts[0].contains("Find the ledger."), "{}", thoughts[0]);
        assert_eq!(kinds(&g), vec!["check", "nudge"]);
    }

    fn console() -> Station {
        Station {
            name: "bridge console".into(),
            address: "http://local/tower/bridge~0".into(),
            verbs: vec!["command_tower".into()],
        }
    }

    #[tokio::test]
    async fn a_nudge_names_the_station_to_invoke() {
        let cast = Cast::carrying(&["command the tower"]);
        cast.answers("Find the ledger.", "something else");
        *cast.stations.lock().unwrap() = vec![console()];
        let mut g = guardian(vec![Box::new(Drift)], vec![Rung::Nudge, Rung::Flag]);
        g.scan(&cast, secs(0)).await;
        let thoughts = cast.thoughts.lock().unwrap();
        assert_eq!(thoughts.len(), 1);
        assert!(
            thoughts[0].contains("`invoke` http://local/tower/bridge~0/command_tower"),
            "{}",
            thoughts[0]
        );
        assert_eq!(kinds(&g), vec!["check", "nudge"]);
    }

    #[tokio::test]
    async fn a_character_on_its_errand_is_left_alone() {
        let cast = Cast::carrying(&["read the ledger"]);
        cast.answers("Find the ledger.", "part of what I was asked");
        let mut g = guardian(vec![Box::new(Drift)], vec![Rung::Nudge]);
        g.scan(&cast, secs(0)).await;
        assert!(cast.thoughts.lock().unwrap().is_empty());
        assert!(kinds(&g).is_empty());
    }

    #[tokio::test]
    async fn a_character_with_no_mission_is_not_asked_about_one() {
        let cast = Cast::default();
        let mut g = guardian(vec![Box::new(Drift)], vec![Rung::Nudge]);
        g.scan(&cast, secs(0)).await;
        assert!(cast.thoughts.lock().unwrap().is_empty());
        assert!(kinds(&g).is_empty());
    }

    #[tokio::test]
    async fn a_step_is_ticked_only_after_it_is_confirmed_twice() {
        let cast = Cast::carrying(&["read the ledger"]);
        cast.answers("read the ledger", "yes, it is done");
        cast.acts.lock().unwrap().push("scan — the ledger".into());
        let mut g = guardian(vec![Box::new(StepTracker::new(2))], vec![Rung::Nudge]);
        g.scan(&cast, secs(0)).await;
        assert!(cast.ticked.lock().unwrap().is_empty());
        g.scan(&cast, secs(10)).await;
        assert_eq!(
            *cast.ticked.lock().unwrap(),
            vec![("read the ledger".to_string(), StepOutcome::Achieved)]
        );
        assert_eq!(kinds(&g), vec!["tick"]);
    }

    #[tokio::test]
    async fn a_step_the_character_could_not_do_is_signed_off_as_thwarted() {
        let cast = Cast::carrying(&["read the ledger"]);
        cast.answers("read the ledger", "I tried and could not do it");
        cast.acts.lock().unwrap().push("scan — the ledger".into());
        let mut g = guardian(vec![Box::new(StepTracker::new(1))], vec![Rung::Nudge]);
        g.scan(&cast, secs(0)).await;
        assert_eq!(
            *cast.ticked.lock().unwrap(),
            vec![("read the ledger".to_string(), StepOutcome::Thwarted)]
        );
    }

    #[tokio::test]
    async fn a_character_repeating_one_act_is_nudged_toward_saying_what_stopped_it() {
        let cast = Cast::carrying(&["read the ledger"]);
        cast.acts
            .lock()
            .unwrap()
            .extend(vec!["scan — the shelf".to_string(); 5]);
        let mut g = guardian(vec![Box::new(Looping)], vec![Rung::Nudge, Rung::Flag]);
        g.scan(&cast, secs(0)).await;
        g.scan(&cast, secs(10)).await;
        let thoughts = cast.thoughts.lock().unwrap();
        assert_eq!(thoughts.len(), 1);
        assert!(thoughts[0].contains("`report_stuck`"), "{}", thoughts[0]);
        assert_eq!(kinds(&g), vec!["check", "nudge"]);
    }

    #[tokio::test]
    async fn a_character_varying_its_acts_is_not_asked_anything() {
        let cast = Cast::carrying(&["read the ledger"]);
        cast.acts.lock().unwrap().extend(
            ["scan — the shelf", "move_to — the hall", "scan — the desk"].map(String::from),
        );
        let mut g = guardian(vec![Box::new(Looping)], vec![Rung::Nudge]);
        g.scan(&cast, secs(0)).await;
        assert!(kinds(&g).is_empty());
    }

    #[tokio::test]
    async fn a_nudge_is_not_followed_by_another_while_it_settles() {
        let cast = Cast::carrying(&["read the ledger"]);
        cast.answers("Find the ledger.", "something else");
        let mut g = guardian(vec![Box::new(Drift)], vec![Rung::Nudge, Rung::Flag]);
        g.scan(&cast, secs(0)).await;
        g.scan(&cast, secs(10)).await;
        g.scan(&cast, secs(20)).await;
        assert_eq!(cast.thoughts.lock().unwrap().len(), 1);
    }

    #[tokio::test]
    async fn a_nudge_that_worked_is_recorded_as_held() {
        let cast = Cast::carrying(&["read the ledger"]);
        cast.answers("Find the ledger.", "something else");
        let mut g = guardian(vec![Box::new(Drift)], vec![Rung::Nudge, Rung::Flag]);
        g.scan(&cast, secs(0)).await;
        cast.answers("Find the ledger.", "part of what I was asked");
        g.scan(&cast, secs(40)).await;
        assert_eq!(kinds(&g), vec!["check", "nudge", "held"]);
    }

    #[tokio::test]
    async fn a_nudge_that_failed_takes_the_next_rung_after_the_cooldown() {
        let cast = Cast::carrying(&["read the ledger"]);
        cast.answers("Find the ledger.", "something else");
        let mut g = guardian(
            vec![Box::new(Drift)],
            vec![Rung::Nudge, Rung::Refresh, Rung::Flag],
        );
        g.scan(&cast, secs(0)).await;
        g.scan(&cast, secs(40)).await;
        assert_eq!(cast.rousings.lock().unwrap().len(), 0, "cooldown holds");
        g.scan(&cast, secs(130)).await;
        assert_eq!(
            *cast.rousings.lock().unwrap(),
            vec!["What has been asked of you: Find the ledger."]
        );
        assert!(kinds(&g).contains(&"failed"));
    }

    #[tokio::test]
    async fn the_last_rung_is_a_flag_that_says_nothing_to_the_character() {
        let cast = Cast::carrying(&["read the ledger"]);
        cast.answers("Find the ledger.", "something else");
        let mut g = guardian(vec![Box::new(Drift)], vec![Rung::Flag]);
        g.scan(&cast, secs(0)).await;
        assert!(cast.thoughts.lock().unwrap().is_empty());
        assert!(cast.rousings.lock().unwrap().is_empty());
        assert_eq!(kinds(&g), vec!["check", "flag"]);
    }

    fn a_stretch() -> Waiting {
        Waiting {
            turns: 16,
            from_ms: 0,
            to_ms: 60_000,
            empty: false,
        }
    }

    #[tokio::test]
    async fn a_stretch_the_character_says_needs_an_entry_is_sent_to_be_written() {
        let cast = Cast::default();
        *cast.waiting.lock().unwrap() = Some(a_stretch());
        cast.answers("not in your journal yet", "yes");
        let mut g = guardian(vec![Box::new(Journal)], vec![Rung::Nudge]);
        g.scan(&cast, secs(0)).await;
        assert_eq!(
            *cast.decided.lock().unwrap(),
            vec![(true, "because".to_string())]
        );
        assert_eq!(kinds(&g), vec!["journal"]);
        assert!(cast.thoughts.lock().unwrap().is_empty());
    }

    #[tokio::test]
    async fn a_stretch_with_no_entry_to_weigh_it_against_is_written_without_asking() {
        let cast = Cast::default();
        *cast.waiting.lock().unwrap() = Some(Waiting {
            empty: true,
            ..a_stretch()
        });
        let mut g = guardian(vec![Box::new(Journal)], vec![Rung::Nudge]);
        g.scan(&cast, secs(0)).await;
        assert_eq!(*cast.decided.lock().unwrap(), vec![(true, String::new())]);
        assert_eq!(kinds(&g), vec!["journal"]);
    }

    #[tokio::test]
    async fn a_stretch_the_character_says_does_not_is_closed_with_its_reason() {
        let cast = Cast::default();
        *cast.waiting.lock().unwrap() = Some(a_stretch());
        cast.answers("not in your journal yet", "no");
        let mut g = guardian(vec![Box::new(Journal)], vec![Rung::Nudge]);
        g.scan(&cast, secs(0)).await;
        assert_eq!(
            *cast.decided.lock().unwrap(),
            vec![(false, "because".to_string())]
        );
        assert_eq!(kinds(&g), vec!["journal_skip"]);
    }

    #[tokio::test]
    async fn a_journal_that_covers_everything_is_not_asked_about() {
        let cast = Cast::default();
        let mut g = guardian(vec![Box::new(Journal)], vec![Rung::Nudge]);
        g.scan(&cast, secs(0)).await;
        assert!(cast.decided.lock().unwrap().is_empty());
        assert!(kinds(&g).is_empty());
    }

    #[tokio::test]
    async fn a_question_that_cannot_be_answered_is_logged_and_does_not_press() {
        let cast = Cast::carrying(&["read the ledger"]);
        let mut g = guardian(vec![Box::new(Drift)], vec![Rung::Nudge]);
        g.scan(&cast, secs(0)).await;
        assert!(cast.thoughts.lock().unwrap().is_empty());
        assert_eq!(kinds(&g), vec!["failed"]);
    }
}
