//! Selection-replay TDD suite — engine-less projection scoring over probes and
//! galleries captured from a real conversation (`selection_replay_data/`).
//!
//! The fixture is the live tour conversation of 2026-08-02 (see
//! `selection_replay_data/README.md`): per dialogue turn the raw wide-Q probe
//! sigs the daemon actually used and its recorded `ProjectionEvent` sequence,
//! plus the sig galleries scoring ran against — the turns the projections
//! selected (the observed junk), the `models/builder.rs` file conversation
//! (the miss), the repo_map structure clusters, and the tool corpus.
//! `ideal_projections.json` states what a correct projection must do per turn.
//!
//! The whole fixture loads ONCE per process (`fixture()`); every test shares
//! the decoded sigs and events.
//!
//! Three tiers, all always-on:
//! - **Baseline**: every recorded projection point replays through the shipped
//!   additive scorer against a checked-in golden digest — the pre-design
//!   state, totally characterized; any additive-scorer change diffs per point.
//!   (The recorded winner sets themselves are NOT the assertion: measured ~7%
//!   of them re-rank near the top under instantaneous raw scoring — recorded
//!   selection is mostly belief/hysteresis inertia; see the baseline test's
//!   doc-comment and README.)
//! - **Guards**: tool selection already works; it must stay working through
//!   every scoring change.
//! - **Design acceptance**: the formerly-red TDD targets of
//!   `docs/provenance_adaptive_projection.md`, asserted through the code-reading
//!   scopes group's scan as the daemon runs it (a file one case, the top-8
//!   square-rooted vote law, each file read among its own kind, question/tail
//!   max-fusion), and the attention mass on the ungated additive sum.

use std::collections::{BTreeMap, HashMap};
use std::path::{Path, PathBuf};
use std::sync::{Arc, OnceLock};

use candle_conversation::normalization::{kind_z_band, path_kind};
use candle_conversation::projection::{attention_mass, decode_events, ProjectionEvent};
use candle_conversation::provenance::{
    decode_wide_sigs, score_slots, score_slots_grouped, score_slots_law, FusionMode, VoteLaw,
    WideQSig,
};
use serde_json::Value;

/// Production probe window: `reproject_max_probe_tokens` (config.rs default).
const MAX_PROBE_TOKENS: usize = 256;

/// The tour conversation's dialogue timeline (`ideal_projections.json`).
const TOUR_TL: u64 = 10454652321042114835;

// ── Fixture (loaded once) ─────────────────────────────────────────────────────

struct TurnRef {
    timeline: u64,
    index: u64,
    tags: Vec<String>,
}

struct Fixture {
    /// Decoded sig windows for every exported turn.
    sigs: HashMap<(u64, u64), Arc<Vec<WideQSig>>>,
    /// Recorded projection events per dialogue turn.
    events: BTreeMap<(u64, u64), Vec<ProjectionEvent>>,
    /// Every turn any recorded projection selected (the observed record).
    candidates: Vec<TurnRef>,
    /// Target galleries by manifest group name.
    targets: BTreeMap<String, Vec<TurnRef>>,
}

fn export_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/selection_replay_data/export")
}

fn parse_refs(v: &Value) -> Vec<TurnRef> {
    v.as_array()
        .map(|a| {
            a.iter()
                .filter(|t| t["has_sig"].as_bool() == Some(true))
                .filter_map(|t| {
                    Some(TurnRef {
                        timeline: t["timeline"].as_str()?.parse().ok()?,
                        index: t["index"].as_u64()?,
                        tags: t["tags"]
                            .as_array()
                            .map(|a| {
                                a.iter()
                                    .filter_map(|v| v.as_str().map(str::to_string))
                                    .collect()
                            })
                            .unwrap_or_default(),
                    })
                })
                .collect()
        })
        .unwrap_or_default()
}

/// The fixture data is local-only (gitignored — 657 files / ~46 MiB; see
/// `selection_replay_data/.gitignore`). When it is absent the tests SKIP with
/// a message instead of failing, so a fresh clone stays green; regenerate with
/// `substrate_inspect export-replay` per `selection_replay_data/README.md`.
fn fixture() -> Option<&'static Fixture> {
    static FIXTURE: OnceLock<Option<Fixture>> = OnceLock::new();
    FIXTURE
        .get_or_init(|| {
            let dir = export_dir();
            if !dir.join("manifest.json").is_file() {
                eprintln!(
                    "SKIP: selection_replay_data/export/ is absent (local-only fixture) — \
                     regenerate with `substrate_inspect export-replay` \
                     (see selection_replay_data/README.md)"
                );
                return None;
            }
            Some(load_fixture(&dir))
        })
        .as_ref()
}

fn load_fixture(dir: &Path) -> Fixture {
    {
        let raw = std::fs::read(dir.join("manifest.json")).expect("manifest.json reads");
        let manifest: Value = serde_json::from_slice(&raw).expect("manifest parses");

        let mut sigs = HashMap::new();
        for entry in std::fs::read_dir(dir.join("sigs")).expect("sigs dir") {
            let path = entry.expect("dir entry").path();
            let Some(stem) = path.file_stem().and_then(|s| s.to_str()) else {
                continue;
            };
            let Some((tl, idx)) = stem.split_once('_') else {
                continue;
            };
            let (Ok(tl), Ok(idx)) = (tl.parse::<u64>(), idx.parse::<u64>()) else {
                continue;
            };
            let raw = std::fs::read(&path).expect("sig blob reads");
            let decoded = decode_wide_sigs(&raw)
                .unwrap_or_else(|| panic!("sig blob {} fails to decode", path.display()));
            sigs.insert((tl, idx), Arc::new(decoded));
        }

        let mut events = BTreeMap::new();
        for d in manifest["dialogue_turns"]
            .as_array()
            .expect("dialogue_turns")
        {
            let (Some(tl), Some(idx)) = (
                d["timeline"].as_str().and_then(|s| s.parse::<u64>().ok()),
                d["index"].as_u64(),
            ) else {
                continue;
            };
            let Some(name) = d["events"].as_str() else {
                continue;
            };
            let raw = std::fs::read(dir.join("events").join(name)).expect("events file");
            events.insert((tl, idx), decode_events(&raw));
        }

        let candidates = parse_refs(&manifest["selected_candidates"]);
        let mut targets = BTreeMap::new();
        for (name, v) in manifest["targets"].as_object().expect("targets") {
            targets.insert(name.clone(), parse_refs(v));
        }
        Fixture {
            sigs,
            events,
            candidates,
            targets,
        }
    }
}

impl Fixture {
    fn sig(&self, tl: u64, idx: u64) -> &[WideQSig] {
        self.sigs
            .get(&(tl, idx))
            .unwrap_or_else(|| panic!("sig blob for {tl}_{idx} missing from fixture"))
    }

    /// The probe window at generated-token position `t` of a dialogue turn,
    /// reconstructed from the sealed turn's full sig sequence and the recorded
    /// event cadence: the last event's `start_token` is decode-end, so the
    /// user-prefix length is `len − generated_total`, and the probe at `t` is
    /// the trailing window ending at `user_len + t` — the same trailing-window
    /// slice the live reprojection gathered (query-head chunks approximated
    /// away).
    fn probe_at(&self, tl: u64, idx: u64, t: u32) -> Vec<WideQSig> {
        let all = self.sig(tl, idx);
        let generated_total = self.events[&(tl, idx)]
            .iter()
            .map(|e| e.start_token)
            .max()
            .unwrap_or(0) as usize;
        let user_len = all.len().saturating_sub(generated_total);
        let end = (user_len + t as usize).min(all.len()).max(1);
        all[end.saturating_sub(MAX_PROBE_TOKENS)..end].to_vec()
    }

    /// Trailing probe of the whole turn (its final reprojection's view).
    fn probe(&self, tl: u64, idx: u64) -> Vec<WideQSig> {
        let all = self.sig(tl, idx);
        all[all.len().saturating_sub(MAX_PROBE_TOKENS)..].to_vec()
    }

    fn code_candidates(&self, exclude_path: Option<&str>) -> Vec<&TurnRef> {
        self.candidates
            .iter()
            .filter(|c| c.tags.first().map(String::as_str) == Some("code"))
            .filter(|c| {
                let path = c.tags.get(1).map(String::as_str).unwrap_or_default();
                !exclude_path.is_some_and(|ex| path.contains(ex))
            })
            .collect()
    }
}

/// Score a probe against named slots, each backed by 1+ gallery windows.
fn rank(probe: &[WideQSig], slots: &[(String, Vec<&[WideQSig]>)]) -> Vec<(String, f32)> {
    let mut windows: Vec<&[WideQSig]> = Vec::new();
    let mut slot_of: Vec<usize> = Vec::new();
    for (i, (_, wins)) in slots.iter().enumerate() {
        for w in wins {
            windows.push(w);
            slot_of.push(i);
        }
    }
    let scores = score_slots(probe, &windows, &slot_of, slots.len());
    let mut out: Vec<(String, f32)> = slots.iter().map(|(n, _)| n.clone()).zip(scores).collect();
    out.sort_by(|a, b| b.1.total_cmp(&a.1));
    out
}

fn slot_key(tl: u64, idx: u64) -> String {
    format!("{tl}#{idx}")
}

// ── Baseline: the current state, totally characterized (always on) ───────────

/// Replay EVERY recorded projection point of every captured conversation
/// through the raw scorer and compare an exact digest (per-point top-3 slot +
/// rounded score) against the checked-in golden
/// (`selection_replay_data/baseline_golden.json`).
///
/// This is a characterization test: it pins the CURRENT scorer's complete
/// behavior over the captured conversation in one fixture load, so any scoring
/// change shows up as a precise per-point diff. When a change is intentional
/// (a design-doc concept landing), regenerate the golden with
/// `SELECTION_REPLAY_REGEN=1 cargo test --test selection_replay baseline` and
/// review the diff like source.
///
/// Deliberately NOT asserted: equality with the daemon's recorded winner sets.
/// Measured at capture time, only ~7% of recorded winners re-rank near the top
/// under instantaneous raw scoring — the recorded selections are dominated by
/// belief accumulation, hit-level normalization, and hysteresis retention
/// (selection inertia), which is itself part of what
/// `docs/provenance_adaptive_projection.md` addresses. The recorded events in
/// the fixture remain the as-observed reference; this golden pins the replay's
/// raw-signal layer.
#[test]
fn baseline_every_recorded_projection_point_replays() {
    let Some(f) = fixture() else { return };
    let pool: Vec<(String, Vec<&[WideQSig]>)> = f
        .candidates
        .iter()
        .map(|c| {
            (
                slot_key(c.timeline, c.index),
                vec![f.sig(c.timeline, c.index)],
            )
        })
        .collect();

    let mut digest: Vec<Value> = Vec::new();
    for ((tl, idx), events) in &f.events {
        for ev in events {
            let ranked = rank(&f.probe_at(*tl, *idx, ev.start_token), &pool);
            let top: Vec<Value> = ranked
                .iter()
                .take(3)
                .map(|(k, s)| serde_json::json!([k, format!("{:.1}", s)]))
                .collect();
            digest.push(serde_json::json!({
                "tl": tl.to_string(),
                "turn": idx,
                "t": ev.start_token,
                "top": top,
            }));
        }
    }
    assert!(
        digest.len() >= 300,
        "expected the fixture to carry ≥300 projection points, found {}",
        digest.len()
    );

    let golden_path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/selection_replay_data/baseline_golden.json");
    let computed = serde_json::to_string_pretty(&digest).expect("digest serializes");
    if std::env::var("SELECTION_REPLAY_REGEN").is_ok_and(|v| !v.is_empty() && v != "0") {
        std::fs::write(&golden_path, &computed).expect("write golden");
        eprintln!(
            "baseline golden REGENERATED: {} points → {}",
            digest.len(),
            golden_path.display()
        );
        return;
    }
    let golden = std::fs::read_to_string(&golden_path).expect(
        "baseline_golden.json missing — bootstrap with \
         SELECTION_REPLAY_REGEN=1 cargo test --test selection_replay baseline",
    );
    let golden: Vec<Value> = serde_json::from_str(&golden).expect("golden parses");
    let mut diffs = 0usize;
    for (i, (a, b)) in golden.iter().zip(digest.iter()).enumerate() {
        if a != b && diffs < 5 {
            eprintln!("baseline diff at point {i}:\n  golden:   {a}\n  computed: {b}");
            diffs += 1;
        } else if a != b {
            diffs += 1;
        }
    }
    assert!(
        diffs == 0 && golden.len() == digest.len(),
        "baseline digest diverged from golden at {diffs} of {} points \
         (+{} count drift) — if this change is intentional, regenerate with \
         SELECTION_REPLAY_REGEN=1 and review the diff",
        digest.len(),
        (golden.len() as i64 - digest.len() as i64).abs(),
    );
}

// ── Guards: tool selection works today and must keep working (always on) ─────

fn tool_slots(f: &'static Fixture) -> Vec<(String, Vec<&'static [WideQSig]>)> {
    let mut by_tool: BTreeMap<String, Vec<&[WideQSig]>> = BTreeMap::new();
    for t in &f.targets["tool"] {
        let Some(tool) = t.tags.iter().find(|g| *g != "tool") else {
            continue;
        };
        by_tool
            .entry(tool.clone())
            .or_default()
            .push(f.sig(t.timeline, t.index));
    }
    by_tool.into_iter().collect()
}

#[test]
fn datetime_probe_ranks_datetime_top1() {
    let Some(f) = fixture() else { return };
    let ranked = rank(&f.probe(TOUR_TL, 1), &tool_slots(f));
    assert_eq!(
        ranked[0].0,
        "datetime",
        "the 'what time is it?' probe must rank datetime first; got {:?}",
        &ranked[..5.min(ranked.len())]
    );
}

#[test]
fn calculator_probe_ranks_calculator_top1() {
    let Some(f) = fixture() else { return };
    let ranked = rank(&f.probe(TOUR_TL, 3), &tool_slots(f));
    assert_eq!(
        ranked[0].0,
        "calculator",
        "the sqrt probe must rank calculator first; got {:?}",
        &ranked[..5.min(ranked.len())]
    );
}

// ── Design acceptance — the production code-reading scan ─────────────────────
//
// Formerly the RED TDD targets of `docs/provenance_adaptive_projection.md`,
// asserted through the code-reading scopes group's scan as the daemon runs it
// (`zend/src/prompts/projection.yaml`, the `scopes` group;
// `projection/resolver/belief_files_kinds.rs`): each file one case, every
// window of it in that case (`cases: file`); additive fusion; each probe token
// voting for its best 8 files with every kept vote square-rooted
// (`vote_top_k: 8`, `vote_power: 0.5`); each file read as a z-score among the
// scoring files of its own kind on the probe (`normalize: kind_z`); and the
// pinned question's scan max-fused with the tail's (`question_pin`; the
// question window is the head-64 approximation, see F11). Any change to that
// chain that breaks these breaks the design's headline behavior.

/// The vote law the code-reading scopes group scans with.
const SCOPES_LAW: VoteLaw = VoteLaw {
    top_k: 8,
    power: 0.5,
};

/// One case of a competition: its name, its kind (the extension `kind_z`
/// reads it among), and every gallery window it holds.
type Case = (String, String, Vec<&'static [WideQSig]>);

/// The production code-reading scan of `cases` for dialogue turn `(tl, idx)`:
/// the tail and the question window each scanned under [`SCOPES_LAW`], each
/// read as its kind's z-score on the band, and the larger of the two per case.
fn production_pipeline_scores(f: &'static Fixture, cases: &[Case], tl: u64, idx: u64) -> Vec<f32> {
    let mut windows: Vec<&[WideQSig]> = Vec::new();
    let mut case_of: Vec<usize> = Vec::new();
    for (i, (_, _, wins)) in cases.iter().enumerate() {
        for w in wins {
            windows.push(w);
            case_of.push(i);
        }
    }
    let kinds: Vec<&str> = cases.iter().map(|(_, k, _)| k.as_str()).collect();
    let scan = |probe: &[WideQSig]| -> Vec<f32> {
        let raw = score_slots_law(probe, &windows, &case_of, cases.len(), &[], SCOPES_LAW);
        kind_z_band(&raw, &kinds)
    };
    let all = f.sig(tl, idx);
    let tail = scan(&f.probe(tl, idx));
    let question = scan(&all[..64.min(all.len())]);
    tail.iter().zip(&question).map(|(t, q)| t.max(*q)).collect()
}

/// The code candidates the recorded projections selected, each file one case
/// holding every exchange of it — `except` a path left out.
fn file_cases(f: &'static Fixture, except: Option<&str>) -> Vec<Case> {
    let mut by_path: BTreeMap<String, Vec<&'static [WideQSig]>> = BTreeMap::new();
    for c in f.code_candidates(except) {
        let path = c.tags.get(1).cloned().unwrap_or_default();
        by_path
            .entry(path)
            .or_default()
            .push(f.sig(c.timeline, c.index));
    }
    by_path
        .into_iter()
        .map(|(path, wins)| {
            let kind = path_kind(&path).to_string();
            (path, kind, wins)
        })
        .collect()
}

fn ranked_by(cases: &[Case], scores: &[f32]) -> Vec<(String, f32)> {
    let mut out: Vec<(String, f32)> = cases
        .iter()
        .map(|(n, _, _)| n.clone())
        .zip(scores.iter().copied())
        .collect();
    out.sort_by(|a, b| b.1.total_cmp(&a.1));
    out
}

/// The ModelBuilder question must retrieve `models/builder.rs` over every junk
/// scope the recorded projections actually selected instead of it.
#[test]
fn modelbuilder_probe_ranks_builder_rs_over_observed_junk() {
    let Some(f) = fixture() else { return };
    let builder: Vec<&'static [WideQSig]> = f.targets["models/builder.rs"]
        .iter()
        .map(|t| f.sig(t.timeline, t.index))
        .collect();
    let mut cases: Vec<Case> = vec![("models/builder.rs".into(), "rs".into(), builder)];
    cases.extend(file_cases(f, Some("models/builder.rs")));
    let ranked = ranked_by(&cases, &production_pipeline_scores(f, &cases, TOUR_TL, 5));
    let rank = ranked
        .iter()
        .position(|(l, _)| l == "models/builder.rs")
        .map(|p| p + 1)
        .unwrap_or(usize::MAX);
    // The selection-level criterion: the scopes group places what it ranks
    // highest, so builder.rs inside the top 3 is projected (live on 2026-08-02
    // it was absent entirely: "no codebase context").
    assert!(
        rank <= 3,
        "models/builder.rs must land inside the production selection; got rank {rank}: {:?}",
        &ranked[..5.min(ranked.len())]
    );
}

/// The tour question must put repository structure over raw file scopes.
#[test]
fn tour_probe_ranks_structure_over_scopes() {
    let Some(f) = fixture() else { return };
    let mut cases: Vec<Case> = f.targets["repo_map"]
        .iter()
        .map(|t| {
            (
                format!(
                    "cluster:{}#{}",
                    t.tags.get(1).map(String::as_str).unwrap_or("?"),
                    t.index
                ),
                String::new(),
                vec![f.sig(t.timeline, t.index)],
            )
        })
        .collect();
    cases.extend(file_cases(f, None));
    let ranked = ranked_by(&cases, &production_pipeline_scores(f, &cases, TOUR_TL, 0));
    assert!(
        ranked[0].0.starts_with("cluster:"),
        "a codebase-tour probe must rank repository structure over any single \
         file scope under the production pipeline; got {:?}",
        &ranked[..5.min(ranked.len())]
    );
}

/// The ungated additive group sum of the tail scan of `(tl, idx)` over
/// `slots`: what the attention mass (Concept B) keys on.
fn mass_base(
    f: &'static Fixture,
    slots: &[(String, Vec<&[WideQSig]>)],
    tl: u64,
    idx: u64,
) -> Vec<f32> {
    let mut windows: Vec<&[WideQSig]> = Vec::new();
    let mut slot_of: Vec<usize> = Vec::new();
    for (i, (_, wins)) in slots.iter().enumerate() {
        for w in wins {
            windows.push(w);
            slot_of.push(i);
        }
    }
    let grouped = score_slots_grouped(&f.probe(tl, idx), &windows, &slot_of, slots.len(), &[]);
    if grouped.is_empty() {
        vec![0.0; slots.len()]
    } else {
        FusionMode::Additive.fuse(&grouped)
    }
}

/// A pure history question must carry (relatively) no code attention: the same
/// scope gallery that lights up for a code question must stay quiet for it —
/// the Concept B mass contrast, computed with the production `attention_mass`.
#[test]
fn recall_probe_code_mass_collapses_relative_to_code_probe() {
    let Some(f) = fixture() else { return };
    let slots: Vec<(String, Vec<&[WideQSig]>)> = f
        .code_candidates(None)
        .iter()
        .map(|c| {
            (
                slot_key(c.timeline, c.index),
                vec![f.sig(c.timeline, c.index)],
            )
        })
        .collect();
    // Mass keys on the UNGATED additive tail scan with the shipped constants
    // (k = 1, ρ = 2 — the top-1-share² concentration).
    let mass = |idx: u64| -> f32 {
        attention_mass(&mass_base(f, &slots, TOUR_TL, idx), 0.0, f32::MAX, 1, 2.0)
    };
    let recall = mass(6);
    let code = mass(5);
    // The measured contrast on the ungated top-1-share² formula (results doc
    // §25): the raw sum is INVERTED (recall 5591 > code 3470); concentration
    // flips it to ~0.65× — assert the direction with margin, which is what the
    // flexbox gain modulates on.
    assert!(
        recall < 0.75 * code,
        "history-question code mass ({recall:.1}) must sit clearly below a \
         code question's ({code:.1}) — adaptive budgets key on this contrast"
    );
}
