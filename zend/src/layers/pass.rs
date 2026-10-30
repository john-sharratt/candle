//! One pass over one layer built on another, run by the branch ingest after
//! the layers below it (`crate::branch_ingest::BranchIngest::pass`).
//!
//! The pass walks the layers bottom up and carries what stands in each
//! ([`Ground`]) to the next, so a layer is planned against exactly what its
//! layer below holds current this pass — after that layer read what it had
//! queued. A layer whose units below are still being read again has nothing
//! of them to stand on and reads nothing of them; one above a layer that ran
//! out of time is planned again from the bottom next pass.

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use candle_conversation::projection::{GroupId, TimelineId};
use candle_conversation::{ConversationEngine, Sequence};

use super::days::{closed_days, day_of, DaySource};
use super::expiry::now_ms;
use super::keys::{termed, unit_key};
use super::plan::{plan_holding, Below, Done, Held, LiveFiles, Plan, Queued};
use super::run::{committed, ingest_units, EdgeFiles, UnitJob};
use super::spec::{LayerSpec, UnitKind};
use crate::calibrate::CalibrateSpec;
use crate::ingest_report::IngestReport;
use crate::refresh_ctx::RefreshContext;

/// What the pass knows so far, carried from each layer to the one above.
#[derive(Debug, Default)]
pub struct Ground {
    /// Every file version a branch holds: path → `(key, branches)`.
    pub live: LiveFiles,
    /// The `code_reading` keys of the files within the depth bound — the
    /// files whose units this pass reads; one past it is kept, never read.
    pub ingest: HashSet<String>,
    /// What stands in each layer planned so far, by layer name.
    pub standing: HashMap<String, Vec<Below>>,
    /// The repositories held back this pass: nothing looked at their
    /// branches, so no unit on them or depending on them is judged.
    pub held_back: Vec<String>,
    /// Each repository's default branch — its first in walk order.
    pub defaults: HashMap<String, String>,
    /// The units read this pass that did not commit, by job key: tried again
    /// next pass, not again in this one.
    pub tried: HashSet<String>,
}

impl Ground {
    /// Whether `path` (workspace-relative) is in a repository held back.
    fn is_held_back(&self, path: &str) -> bool {
        let repo = crate::repo_path::split(path).0;
        self.held_back.iter().any(|r| r == repo)
    }
}

/// One layer for [`pass_layer`].
pub struct LayerRun<'a> {
    pub spec: &'a LayerSpec,
    pub base: &'a Sequence,
    /// How the layer's units are calibrated as they commit, for a layer that
    /// declares a `calibrate:` block.
    pub calibrate: Option<&'a CalibrateSpec>,
    /// Each lower layer of units whose group is scoped by branch, by layer
    /// name: what a unit of this layer may retrieve from it is what stands
    /// there on a branch the unit's own file is on.
    pub sees: &'a [(String, GroupId)],
}

/// Which of a layer's queued units a [`pass_layer`] reads.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Reach {
    /// Those on their repository's default branch, and those of no branch (a
    /// day). The pass climbs every layer with this first, so the default
    /// branch is analysed all the way up before any other branch's units are
    /// read in any layer.
    Default,
    /// Every unit queued.
    All,
}

impl Reach {
    fn takes(self, q: &Queued, subject: &str, ground: &Ground) -> bool {
        match self {
            Reach::All => true,
            Reach::Default => {
                if q.branches.is_empty() {
                    return true;
                }
                let repo = crate::repo_path::split(subject).0;
                match ground.defaults.get(repo) {
                    Some(default) => q.branches.contains(default),
                    None => q
                        .branches
                        .iter()
                        .any(|b| crate::branch_ingest::walk::is_default_branch(b)),
                }
            }
        }
    }
}

/// Plan the layer, tombstone what is stale, read what is queued within
/// `reach`, and leave what stands in `ground` for the layer above. Answers
/// whether the layer changed.
pub fn pass_layer(
    ctx: &RefreshContext<'_>,
    run: &LayerRun<'_>,
    ground: &mut Ground,
    reach: Reach,
) -> anyhow::Result<bool> {
    layer(ctx, run, ground, Some(reach))
}

/// Plan the layer and tombstone what is stale, reading nothing: what a pass
/// does to every layer of units before the layers below read anything, so a
/// unit made stale by work below goes the moment that work is found — the
/// layer has no unit for the file while the layers under it are read again,
/// rather than one describing the version being replaced.
pub fn sweep_layer(ctx: &RefreshContext<'_>, run: &LayerRun<'_>, ground: &mut Ground) -> bool {
    layer(ctx, run, ground, None).unwrap_or(false)
}

fn layer(
    ctx: &RefreshContext<'_>,
    run: &LayerRun<'_>,
    ground: &mut Ground,
    read: Option<Reach>,
) -> anyhow::Result<bool> {
    let spec = run.spec;
    let days = match (&spec.after, spec.unit) {
        (None, UnitKind::Day) => dialogue_days(ctx),
        _ => Vec::new(),
    };
    let below: Vec<Below> = match &spec.after {
        Some(after) => match ground.standing.get(after) {
            Some(below) => below.clone(),
            // The layer below was not planned this pass — it is disabled, or
            // the pass stopped short of it. Nothing is known of what stands
            // there, so nothing here is judged against it: an empty layer
            // below would read as every unit here gone.
            None => {
                tracing::info!(
                    target: "zend::layers",
                    layer = %spec.layer,
                    after = %after,
                    "the layer below was not planned this pass: this one is left as it is",
                );
                return Ok(false);
            }
        },
        None => days.iter().map(day_below).collect(),
    };
    let first = settle(ctx.engine, spec, &below, ground);
    let mut changed = !first.tombstone.is_empty();
    ground
        .standing
        .insert(spec.layer.clone(), standing(&below, &first.held));
    let Some(reach) = read else {
        return Ok(changed);
    };
    if spec.edges && !ground.held_back.is_empty() {
        // An answer naming a file of a repository held back could not be
        // keyed by it: the unit would stand without that edge for good.
        tracing::info!(
            target: "zend::layers",
            layer = %spec.layer,
            held_back = ?ground.held_back,
            "a repository is held back this pass: the layer's units are read next pass",
        );
        return Ok(changed);
    }
    // A layer reads until nothing within reach is queued in it before the
    // layer above reads anything: a unit it reads can leave branches its
    // dependencies are not current on, and each of those queues a unit of its
    // own. Each unit is tried once a pass — one that does not commit is read
    // again next pass — and a pass asked for stops it.
    let now = now_ms();
    let within = |p: &Plan, ground: &Ground| -> Vec<Queued> {
        p.queued
            .iter()
            .filter(|q| {
                let b = &below[q.below];
                reach.takes(q, &b.subject, ground)
                    && !ground.tried.contains(&job_key(spec, b, q, now))
            })
            .cloned()
            .collect()
    };
    let mut queued = within(&first, ground);
    let mut report: Option<IngestReport> = None;
    while !queued.is_empty() {
        let Some((next, round)) = read_queued(ctx, run, ground, &below, &days, &queued, now)?
        else {
            break;
        };
        changed = true;
        match report.as_mut() {
            Some(r) => r.merge(round),
            None => report = Some(round),
        }
        for q in queued.iter().filter(|q| next.queued.contains(q)) {
            ground.tried.insert(job_key(spec, &below[q.below], q, now));
        }
        queued = within(&next, ground);
    }
    if let Some(report) = report {
        crate::ingest_report::publish(&spec.layer, report);
    }
    Ok(changed)
}

/// The key a unit is leased under. A layer that names its dependencies can
/// build several units on one unit below, one per set of branches whose
/// dependencies differ: each is leased apart by its branches. It commits
/// under a key of what it depended on (`run::read_one`). A unit of a layer
/// that expires is leased for the term it is read in at `now`
/// ([`Expiry::term`](super::expiry::Expiry::term)), so its replacement is
/// never taken for the unit whose time is up.
fn job_key(spec: &LayerSpec, b: &Below, q: &Queued, now: u64) -> String {
    let built_on = if spec.edges {
        format!("{}\n{}", b.key, q.branches.join(","))
    } else {
        b.key.clone()
    };
    unit_key(&spec.layer, &termed(&built_on, term_of(spec, b, now)))
}

/// The term a unit on `b` read at `now` stands for, when its layer expires.
fn term_of(spec: &LayerSpec, b: &Below, now: u64) -> Option<u64> {
    spec.expiry.map(|e| e.term(&b.subject, now))
}

/// Read `queued` and plan the layer again, leaving what stands in `ground`.
/// `None` when a pass was asked for first and nothing was read.
fn read_queued(
    ctx: &RefreshContext<'_>,
    run: &LayerRun<'_>,
    ground: &mut Ground,
    below: &[Below],
    days: &[DaySource],
    queued: &[Queued],
    now: u64,
) -> anyhow::Result<Option<(Plan, IngestReport)>> {
    let spec = run.spec;
    if crate::ingest_worker::preempted() || crate::ingest_cancel::cancelled() {
        tracing::info!(
            target: "zend::layers",
            layer = %spec.layer,
            queued = queued.len(),
            "a pass was asked for: the layers below are read first, this one next pass",
        );
        return Ok(None);
    }
    let by_day: HashMap<&str, &DaySource> = days.iter().map(|d| (d.key.as_str(), d)).collect();
    let jobs: Vec<UnitJob> = queued
        .iter()
        .map(|q| {
            let b = &below[q.below];
            let prompt = match spec.unit {
                UnitKind::File => spec.file_prompt(&b.subject),
                UnitKind::Day => {
                    let transcript = match by_day.get(b.key.as_str()) {
                        Some(day) => transcript(ctx.engine, day),
                        None => String::new(),
                    };
                    spec.day_prompt(&b.subject, &transcript)
                }
            };
            UnitJob {
                key: job_key(spec, b, q, now),
                term: term_of(spec, b, now),
                subject: b.subject.clone(),
                after: b.key.clone(),
                parent: b.timeline,
                file_key: b.file_key.clone(),
                branches: q.branches.clone(),
                at: b.at.clone(),
                prompt,
                scopes: scopes(run.sees, ground, &q.branches, &b.subject),
            }
        })
        .collect();
    let known: Vec<&str> = ground.live.keys().map(String::as_str).collect();
    let files = EdgeFiles {
        live: &ground.live,
        known: &known,
    };
    let report = ingest_units(ctx, spec, &jobs, run.base, Some(&files), run.calibrate)?;
    // What this layer read stands now too, for the layer above.
    let after = settle(ctx.engine, spec, below, ground);
    ground
        .standing
        .insert(spec.layer.clone(), standing(below, &after.held));
    Ok(Some((after, report)))
}

/// Plan the layer against `below` and tombstone what the plan lets go.
fn settle(engine: &ConversationEngine, spec: &LayerSpec, below: &[Below], ground: &Ground) -> Plan {
    let (done, broken) = committed(engine, &spec.layer);
    let in_scope = |b: &Below| match spec.unit {
        UnitKind::File => b
            .file_key
            .as_ref()
            .is_some_and(|k| ground.ingest.contains(k)),
        UnitKind::Day => true,
    };
    let now = now_ms();
    let expired = |d: &Done| {
        spec.expiry
            .is_some_and(|e| e.expired(&d.subject, d.built_ms, now))
    };
    let p = plan_holding(
        below,
        &done,
        &ground.live,
        &in_scope,
        &super::keys::spec_digest(spec),
        &|path| ground.is_held_back(path),
        &expired,
    );
    // A committed unit whose chain stopped part-way is stale too.
    let mut gone: Vec<TimelineId> = p.tombstone.clone();
    gone.extend(broken.iter().copied().filter(|tl| is_keyed(engine, *tl)));
    if !gone.is_empty() {
        let e = engine;
        for tl in &gone {
            if let Err(err) = e.tombstone_timeline(*tl) {
                tracing::warn!(
                    target: "zend::layers",
                    layer = %spec.layer,
                    timeline = tl.raw(),
                    "tombstone of a stale unit failed: {err:#}",
                );
            }
        }
    }
    tracing::info!(
        target: "zend::layers",
        layer = %spec.layer,
        below = below.len(),
        committed = done.len(),
        standing = p.held.len(),
        stale = gone.len(),
        queued = p.queued.len(),
        "planned the layer against the layer below",
    );
    p
}

/// Whether `timeline` carries a content key — committed, so a stopped chain
/// and not a unit another member is reading.
fn is_keyed(engine: &ConversationEngine, timeline: TimelineId) -> bool {
    engine
        .conversation_metadata(timeline)
        .is_some_and(|m| m.contains_key(super::keys::CONTENT_KEY))
}

/// What stands in the layer, as the layer above builds on it: each held
/// unit, keyed as it was committed, of the file or day of the unit below.
fn standing(below: &[Below], held: &[Held]) -> Vec<Below> {
    held.iter()
        .map(|h| {
            let b = &below[h.below];
            Below {
                key: h.key.clone(),
                subject: b.subject.clone(),
                timeline: Some(h.timeline),
                file_key: b.file_key.clone(),
                branches: if h.branches.is_empty() {
                    b.branches.clone()
                } else {
                    h.branches.clone()
                },
                at: b.at.clone(),
            }
        })
        .collect()
}

/// What a unit on `branches` may retrieve from each scoped layer below: what
/// stands there on one of its branches, or all of it for a unit on none.
fn scopes(
    sees: &[(String, GroupId)],
    ground: &Ground,
    branches: &[String],
    subject: &str,
) -> Vec<(GroupId, Arc<HashSet<TimelineId>>)> {
    let repo = crate::repo_path::split(subject).0;
    // A branch name means something only inside its repository: a unit draws
    // on its own repository's units on its branches, and on another's what
    // stands on that repository's default branch. A unit of no branch (a day)
    // draws on every repository's default branch, not on every branch's copy.
    let visible = |b: &Below| {
        if b.branches.is_empty() {
            return true; // of no branch itself: a day, or a repository held back
        }
        let theirs = crate::repo_path::split(&b.subject).0;
        if !branches.is_empty() && theirs == repo {
            return b.branches.iter().any(|x| branches.contains(x));
        }
        match ground.defaults.get(theirs) {
            Some(default) => b.branches.contains(default),
            None => b
                .branches
                .iter()
                .any(|x| crate::branch_ingest::walk::is_default_branch(x)),
        }
    };
    sees.iter()
        .map(|(layer, group)| {
            let allowed: HashSet<TimelineId> = ground
                .standing
                .get(layer)
                .into_iter()
                .flatten()
                .filter(|b| visible(b))
                // Its own file's units are its lineage already: offered again,
                // one large one took its group's whole window from the units of
                // the other files it was there to draw on.
                .filter(|b| b.subject != subject)
                .filter_map(|b| b.timeline)
                .collect();
            (*group, Arc::new(allowed))
        })
        .collect()
}

/// A day of dialogue as the unit below a day layer's.
fn day_below(day: &DaySource) -> Below {
    Below {
        key: day.key.clone(),
        subject: day.day.clone(),
        timeline: None,
        file_key: None,
        branches: Vec::new(),
        at: None,
    }
}

/// Every closed day of the live dialogue.
fn dialogue_days(ctx: &RefreshContext<'_>) -> Vec<DaySource> {
    let Some((group, titler)) = ctx.dialogue else {
        return Vec::new();
    };
    let e = ctx.engine;
    let convs: Vec<_> = e
        .group_conversations(group)
        .into_iter()
        .filter(|tl| Some(*tl) != titler && e.conversation_conv_id(*tl).is_some())
        // A turn stamped with no wall time (a legacy stamp) is of no day:
        // taken at face value it would make a 1970-01-01 of its own.
        .map(|tl| {
            let mut turns = e.timeline_turn_wall_ms(tl);
            turns.retain(|(_, ms)| *ms > 0);
            (tl, turns)
        })
        .collect();
    closed_days(&convs, &day_of(now_ms()))
}

/// The most of a whole day's transcript a unit is shown, in characters — a
/// day's dialogue can run to far more than one context; what is past this is
/// named, not shown.
const DAY_CHARS: usize = 48_000;

/// The least of one turn's half the transcript shows when the day is too long
/// to share [`DAY_CHARS`] out further.
const MIN_HALF_CHARS: usize = 400;

/// The day's dialogue as its unit reads it: each conversation by its title,
/// each turn's question and answer, the answer's thinking set aside.
fn transcript(engine: &ConversationEngine, day: &DaySource) -> String {
    let mut convs = Vec::new();
    for (tl, turns) in &day.turns {
        let texts = engine.conversation_texts(*tl);
        let title = engine
            .conversation_label_of(*tl)
            .or_else(|| engine.conversation_conv_id(*tl))
            .unwrap_or_else(|| format!("conversation {}", tl.raw()));
        let halves = turns
            .iter()
            .filter_map(|t| texts.get(t.0 as usize))
            .map(|(user, assistant)| {
                let answer = match assistant.rfind("</think>") {
                    Some(i) => &assistant[i + "</think>".len()..],
                    None => assistant.as_str(),
                };
                (user.trim().to_string(), answer.trim().to_string())
            })
            .collect::<Vec<_>>();
        convs.push((title, halves));
    }
    render(&convs)
}

/// The conversations rendered within [`DAY_CHARS`] bytes: every turn's half
/// shown up to one shared cap, so a short half leaves its room to the long
/// ones. A day too long for even the floor shows its turns while they fit and
/// names the rest.
fn render(convs: &[(String, Vec<(String, String)>)]) -> String {
    // The framing — titles, labels, cut marks — is set aside first.
    let lens = convs
        .iter()
        .flat_map(|(_, h)| h.iter())
        .flat_map(|(u, a)| [u.len(), a.len()])
        .collect::<Vec<_>>();
    let framing = convs
        .iter()
        .map(|(t, h)| t.len() + 5 + h.len() * (20 + 2 * '…'.len_utf8()))
        .sum::<usize>();
    let cap = half_cap(&lens, DAY_CHARS.saturating_sub(framing)).max(MIN_HALF_CHARS);
    let mut out = String::new();
    let mut left = 0usize;
    for (n, (title, halves)) in convs.iter().enumerate() {
        // Each conversation may take an even share of what is left, so the
        // first of a long day does not crowd out the rest; what it leaves
        // passes on to those after it.
        let share = DAY_CHARS.saturating_sub(out.len()) / (convs.len() - n);
        let limit = out.len() + share;
        let head = format!("## {title}\n");
        if out.len() + head.len() > limit {
            left += halves.len();
            continue;
        }
        out.push_str(&head);
        for (user, answer) in halves {
            let turn = format!(
                "User: {}\nAssistant: {}\n",
                clip(user, cap),
                clip(answer, cap)
            );
            if out.len() + turn.len() > limit {
                left += 1;
                continue;
            }
            out.push_str(&turn);
        }
        out.push('\n');
    }
    if left > 0 {
        out.push_str(&format!("[{left} more turn(s) that day are not shown]\n"));
    }
    out
}

/// The largest cap with every length cut to it summing within `budget`.
fn half_cap(lens: &[usize], budget: usize) -> usize {
    let fits = |c: usize| lens.iter().map(|&l| l.min(c)).sum::<usize>() <= budget;
    let (mut lo, mut hi) = (0, lens.iter().copied().max().unwrap_or(0));
    if fits(hi) {
        return hi;
    }
    while lo < hi {
        let mid = (lo + hi).div_ceil(2);
        if fits(mid) {
            lo = mid;
        } else {
            hi = mid - 1;
        }
    }
    lo
}

/// `s` cut to at most `max` bytes, on a character boundary.
fn clip(s: &str, max: usize) -> String {
    if s.len() <= max {
        return s.to_string();
    }
    let mut end = max;
    while !s.is_char_boundary(end) {
        end -= 1;
    }
    format!("{}…", &s[..end])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_default_reach_takes_the_default_branch_and_what_is_of_no_branch() {
        let q = |branches: &[&str]| Queued {
            below: 0,
            branches: branches.iter().map(|b| b.to_string()).collect(),
        };
        let ground = Ground {
            defaults: HashMap::from([
                ("a".to_string(), "main".to_string()),
                ("b".to_string(), "trunk".to_string()),
            ]),
            ..Ground::default()
        };
        let takes = |reach: Reach, subject: &str, branches: &[&str]| {
            reach.takes(&q(branches), subject, &ground)
        };
        assert!(takes(Reach::Default, "a/x.rs", &["main", "topic"]));
        assert!(takes(Reach::Default, "a/x.rs", &[]));
        assert!(!takes(Reach::Default, "a/x.rs", &["topic", "other"]));
        // Each repository's own default: `trunk` here, and a `main` that is
        // not its default is not.
        assert!(takes(Reach::Default, "b/x.rs", &["trunk"]));
        assert!(!takes(Reach::Default, "b/x.rs", &["main"]));
        // A repository with no branches known falls back to the names.
        assert!(takes(Reach::Default, "c/x.rs", &["master"]));
        assert!(takes(Reach::All, "a/x.rs", &["topic"]));
    }

    #[test]
    fn a_short_half_leaves_its_room_to_the_long_ones() {
        assert_eq!(half_cap(&[10, 100, 1000], 2000), 1000);
        assert_eq!(half_cap(&[10, 100, 1000], 500), 390);
        assert_eq!(half_cap(&[], 500), 0);
    }

    #[test]
    fn a_long_day_shares_its_transcript_out() {
        let long = "x".repeat(40_000);
        let convs = vec![(
            "c".to_string(),
            vec![("q".to_string(), long.clone()), ("q".to_string(), long)],
        )];
        let out = render(&convs);
        assert!(out.len() <= DAY_CHARS + 64, "{}", out.len());
        assert_eq!(out.matches("Assistant: ").count(), 2);
    }

    /// Text of many bytes a character is cut by bytes, so it fits as ASCII
    /// does; and a day too long for the floor keeps every conversation's
    /// first turns rather than dropping a conversation whole.
    #[test]
    fn a_wide_or_long_day_keeps_every_conversation() {
        let wide = "漢".repeat(20_000);
        let convs = (0..3)
            .map(|i| (format!("c{i}"), vec![("q".to_string(), wide.clone()); 40]))
            .collect::<Vec<_>>();
        let out = render(&convs);
        assert!(out.len() <= DAY_CHARS + 64, "{}", out.len());
        for i in 0..3 {
            assert!(out.contains(&format!("## c{i}\n")), "c{i} was dropped");
        }
        assert!(out.contains("more turn(s) that day are not shown"));
        assert_eq!(clip("漢字", 4), "漢…");
    }

    fn tl(n: u64) -> TimelineId {
        TimelineId::from_raw(n).unwrap()
    }

    fn below(n: u64, branches: &[&str]) -> Below {
        Below {
            key: format!("k{n}"),
            subject: format!("r/s{n}"),
            timeline: Some(tl(n)),
            file_key: Some(format!("f{n}")),
            branches: branches.iter().map(|b| b.to_string()).collect(),
            at: None,
        }
    }

    /// A unit may retrieve, from a scoped layer below, only what stands on a
    /// branch it is on.
    #[test]
    fn a_unit_retrieves_what_stands_on_its_branches() {
        let mut ground = Ground::default();
        ground.standing.insert(
            "static_analysis".into(),
            vec![
                below(1, &["main"]),
                below(2, &["topic"]),
                below(3, &["main", "topic"]),
            ],
        );
        let group = candle_conversation::projection::GroupId::for_test(7);
        let sees = [("static_analysis".to_string(), group)];
        let got = scopes(&sees, &ground, &["topic".to_string()], "r/elsewhere");
        assert_eq!(got.len(), 1);
        let mut allowed: Vec<u64> = got[0].1.iter().map(|t| t.raw()).collect();
        allowed.sort();
        assert_eq!(allowed, [2, 3]);
        // A unit of no branch (a day) sees the default branch's.
        let day = scopes(&sees, &ground, &[], "r/elsewhere");
        let mut on_main: Vec<u64> = day[0].1.iter().map(|t| t.raw()).collect();
        on_main.sort();
        assert_eq!(on_main, [1, 3]);
    }

    /// A unit is never offered its own file's units below: they are its
    /// lineage already, and offered again they took the room the other files'
    /// units were drawn into.
    #[test]
    fn a_unit_is_not_offered_its_own_files_units() {
        let mut ground = Ground::default();
        ground.standing.insert(
            "static_analysis".into(),
            vec![below(1, &["main"]), below(2, &["main"])],
        );
        let group = candle_conversation::projection::GroupId::for_test(7);
        let sees = [("static_analysis".to_string(), group)];
        let got = scopes(&sees, &ground, &["main".to_string()], "r/s1");
        let allowed: Vec<u64> = got[0].1.iter().map(|t| t.raw()).collect();
        assert_eq!(allowed, [2]);
    }

    /// A branch name is matched inside its own repository only: another
    /// repository is drawn on at its default branch.
    #[test]
    fn another_repository_is_seen_at_its_default_branch() {
        let mut ground = Ground::default();
        ground.defaults.insert("o".into(), "trunk".into());
        let other = |n: u64, branches: &[&str]| {
            let mut b = below(n, branches);
            b.subject = format!("o/s{n}");
            b
        };
        ground.standing.insert(
            "static_analysis".into(),
            vec![
                other(1, &["topic"]),
                other(2, &["trunk"]),
                below(3, &["topic"]),
            ],
        );
        let group = candle_conversation::projection::GroupId::for_test(7);
        let sees = [("static_analysis".to_string(), group)];
        let got = scopes(&sees, &ground, &["topic".to_string()], "r/x");
        let mut allowed: Vec<u64> = got[0].1.iter().map(|t| t.raw()).collect();
        allowed.sort();
        assert_eq!(allowed, [2, 3]);
    }

    #[test]
    fn a_clip_keeps_whole_characters() {
        assert_eq!(clip("héllo", 2), "h…");
        assert_eq!(clip("héllo", 3), "hé…");
        assert_eq!(clip("hi", 5), "hi");
    }
}
