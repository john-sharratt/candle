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
use std::sync::{Arc, Mutex};

use candle_conversation::projection::{GroupId, TimelineId};
use candle_conversation::{ConversationEngine, Sequence};

use super::days::{closed_days, day_of, DaySource};
use super::keys::unit_key;
use super::plan::{plan, Below, Held, LiveFiles, Plan};
use super::run::{committed, ingest_units, EdgeFiles, UnitJob};
use super::spec::{LayerSpec, UnitKind};
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
}

/// One layer for [`pass_layer`].
pub struct LayerRun<'a> {
    pub spec: &'a LayerSpec,
    pub base: &'a Mutex<Sequence>,
    /// Each lower layer of units whose group is scoped by branch, by layer
    /// name: what a unit of this layer may retrieve from it is what stands
    /// there on a branch the unit's own file is on.
    pub sees: &'a [(String, GroupId)],
}

/// Plan the layer, tombstone what is stale, read what is queued, and leave
/// what stands in `ground` for the layer above. Answers whether the layer
/// changed.
pub fn pass_layer(
    ctx: &RefreshContext<'_>,
    run: &LayerRun<'_>,
    ground: &mut Ground,
) -> anyhow::Result<bool> {
    layer(ctx, run, ground, true)
}

/// Plan the layer and tombstone what is stale, reading nothing: what a pass
/// does to every layer of units before the layers below read anything, so a
/// unit made stale by work below goes the moment that work is found — the
/// layer has no unit for the file while the layers under it are read again,
/// rather than one describing the version being replaced.
pub fn sweep_layer(ctx: &RefreshContext<'_>, run: &LayerRun<'_>, ground: &mut Ground) -> bool {
    layer(ctx, run, ground, false).unwrap_or(false)
}

fn layer(
    ctx: &RefreshContext<'_>,
    run: &LayerRun<'_>,
    ground: &mut Ground,
    read: bool,
) -> anyhow::Result<bool> {
    let spec = run.spec;
    let days = match (&spec.after, spec.unit) {
        (None, UnitKind::Day) => dialogue_days(ctx),
        _ => Vec::new(),
    };
    let below: Vec<Below> = match &spec.after {
        Some(after) => ground.standing.get(after).cloned().unwrap_or_default(),
        None => days.iter().map(day_below).collect(),
    };
    let first = settle(ctx.engine, spec, &below, ground);
    let mut changed = !first.tombstone.is_empty();
    ground
        .standing
        .insert(spec.layer.clone(), standing(&below, &first.held));
    if !read || first.queued.is_empty() {
        return Ok(changed);
    }
    if crate::ingest_worker::preempted() || crate::ingest_cancel::cancelled() {
        tracing::info!(
            target: "zend::layers",
            layer = %spec.layer,
            queued = first.queued.len(),
            "a pass was asked for: the layers below are read first, this one next pass",
        );
        return Ok(changed);
    }
    let by_day: HashMap<&str, &DaySource> = days.iter().map(|d| (d.key.as_str(), d)).collect();
    let jobs: Vec<UnitJob> = first
        .queued
        .iter()
        .map(|q| {
            let b = &below[q.below];
            let prompt = match spec.unit {
                UnitKind::File => spec.file_prompt(&b.subject),
                UnitKind::Day => {
                    let transcript = match by_day.get(b.key.as_str()) {
                        Some(day) => transcript(&ctx.engine.lock().unwrap(), day),
                        None => String::new(),
                    };
                    spec.day_prompt(&b.subject, &transcript)
                }
            };
            // A layer that names its dependencies can build several units on
            // one unit below, one per set of branches whose dependencies
            // differ: each is leased apart by its branches. It commits under a
            // key of what it depended on (`run::read_one`).
            let key = if spec.edges {
                unit_key(&spec.layer, &format!("{}\n{}", b.key, q.branches.join(",")))
            } else {
                unit_key(&spec.layer, &b.key)
            };
            UnitJob {
                key,
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
    ingest_units(ctx, spec, &jobs, run.base, Some(&files))?;
    changed = true;
    // What this layer read stands now too, for the layer above.
    let after = settle(ctx.engine, spec, &below, ground);
    ground
        .standing
        .insert(spec.layer.clone(), standing(&below, &after.held));
    Ok(changed)
}

/// Plan the layer against `below` and tombstone what the plan lets go.
fn settle(
    engine: &Mutex<ConversationEngine>,
    spec: &LayerSpec,
    below: &[Below],
    ground: &Ground,
) -> Plan {
    let (done, broken) = committed(engine, &spec.layer);
    let in_scope = |b: &Below| match spec.unit {
        UnitKind::File => b
            .file_key
            .as_ref()
            .is_some_and(|k| ground.ingest.contains(k)),
        UnitKind::Day => true,
    };
    let p = plan(
        below,
        &done,
        &ground.live,
        &in_scope,
        &super::keys::spec_digest(spec),
    );
    // A committed unit whose chain stopped part-way is stale too.
    let mut gone: Vec<TimelineId> = p.tombstone.clone();
    gone.extend(broken.iter().copied().filter(|tl| is_keyed(engine, *tl)));
    if !gone.is_empty() {
        let e = engine.lock().unwrap();
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
fn is_keyed(engine: &Mutex<ConversationEngine>, timeline: TimelineId) -> bool {
    engine
        .lock()
        .unwrap()
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
    sees.iter()
        .map(|(layer, group)| {
            let allowed: HashSet<TimelineId> = ground
                .standing
                .get(layer)
                .into_iter()
                .flatten()
                .filter(|b| branches.is_empty() || b.branches.iter().any(|x| branches.contains(x)))
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
    let e = ctx.engine.lock().unwrap();
    let convs: Vec<_> = e
        .group_conversations(group)
        .into_iter()
        .filter(|tl| Some(*tl) != titler && e.conversation_conv_id(*tl).is_some())
        .map(|tl| (tl, e.timeline_turn_wall_ms(tl)))
        .collect();
    drop(e);
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0);
    closed_days(&convs, &day_of(now))
}

/// The most of one turn's half a day's transcript shows, in characters.
const TURN_CHARS: usize = 1_500;

/// The most of a whole day's transcript a unit is shown, in characters — a
/// day's dialogue can run to far more than one context; what is past this is
/// named, not shown.
const DAY_CHARS: usize = 48_000;

/// The day's dialogue as its unit reads it: each conversation by its title,
/// each turn's question and answer, the answer's thinking set aside.
fn transcript(engine: &ConversationEngine, day: &DaySource) -> String {
    let mut out = String::new();
    let mut left = 0usize;
    for (tl, turns) in &day.turns {
        let texts = engine.conversation_texts(*tl);
        let title = engine
            .conversation_label_of(*tl)
            .or_else(|| engine.conversation_conv_id(*tl))
            .unwrap_or_else(|| format!("conversation {}", tl.raw()));
        let mut block = format!("## {title}\n");
        for t in turns {
            let Some((user, assistant)) = texts.get(t.0 as usize) else {
                continue;
            };
            let answer = match assistant.rfind("</think>") {
                Some(i) => &assistant[i + "</think>".len()..],
                None => assistant.as_str(),
            };
            block.push_str(&format!(
                "User: {}\nAssistant: {}\n",
                clip(user.trim(), TURN_CHARS),
                clip(answer.trim(), TURN_CHARS)
            ));
        }
        if out.len() + block.len() > DAY_CHARS {
            left += 1;
            continue;
        }
        out.push_str(&block);
        out.push('\n');
    }
    if left > 0 {
        out.push_str(&format!(
            "[{left} more conversation(s) that day are not shown]\n"
        ));
    }
    out
}

/// `s` cut to at most `max` characters, on a character boundary.
fn clip(s: &str, max: usize) -> String {
    match s.char_indices().nth(max) {
        Some((i, _)) => format!("{}…", &s[..i]),
        None => s.to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tl(n: u64) -> TimelineId {
        TimelineId::from_raw(n).unwrap()
    }

    fn below(n: u64, branches: &[&str]) -> Below {
        Below {
            key: format!("k{n}"),
            subject: format!("s{n}"),
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
        let got = scopes(&sees, &ground, &["topic".to_string()], "elsewhere");
        assert_eq!(got.len(), 1);
        let mut allowed: Vec<u64> = got[0].1.iter().map(|t| t.raw()).collect();
        allowed.sort();
        assert_eq!(allowed, [2, 3]);
        let all = scopes(&sees, &ground, &[], "elsewhere");
        assert_eq!(all[0].1.len(), 3);
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
        let got = scopes(&sees, &ground, &["main".to_string()], "s1");
        let allowed: Vec<u64> = got[0].1.iter().map(|t| t.raw()).collect();
        assert_eq!(allowed, [2]);
    }

    #[test]
    fn a_clip_keeps_whole_characters() {
        assert_eq!(clip("héllo", 2), "hé…");
        assert_eq!(clip("hi", 5), "hi");
    }
}
