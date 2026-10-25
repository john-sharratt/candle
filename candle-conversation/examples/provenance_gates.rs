//! The provenance gates over a real substrate, offline
//! (`docs/provenance_scoped_signatures.md` §3.2–§3.3, §4).
//!
//! Opens a workspace's substrate **read-only** — beside a running daemon, which
//! keeps writing it — and runs on the CPU, so a GPU serving decode is left
//! alone:
//!
//! - **Gate 2, capture coverage.** Every `code_reading` turn's signature tokens
//!   against its tokens: a turn whose signature falls short cannot be windowed.
//! - **Gate 3, retrieval.** Probes cut by the projection's own boundary rule
//!   and window shape (`provenance::boundaries`, `provenance::probe_windows`),
//!   from two sources that are not the gallery's own tokens:
//!   - *leave-turn-out*: a file's last turn's last window, scored with that
//!     turn removed from the gallery;
//!   - *analysis units*: the last window of a `dependency_analysis`,
//!     `code_review` or `security_review` unit's own turns, which must find the
//!     file the unit forked from.
//!
//!   A third source is the control: a window of a file's own pages, nothing
//!   left out, which every rule must rank first.
//!
//!   Each probe ranks every `code_reading` file by late fusion's votes
//!   (`provenance::window_score`): summed per file, which is today's scorer,
//!   and tallied per window under each file aggregate. Reported per rule:
//!   recall at 1, 5 and 10,
//!   the mean reciprocal rank, and the rank correlation between a file's size
//!   and how often it is a false top-10 hit, which must sit near zero.
//!
//! Run, from the repository root:
//!   cargo run -p candle-conversation --release --example provenance_gates -- \
//!     <workspace dir> [probes per source]

use std::cmp::Ordering;
use std::collections::{BTreeMap, HashMap};
use std::iter::repeat_n;
use std::ops::Range;
use std::path::PathBuf;
use std::time::Instant;

use candle_conversation::persistence::resume::decode_token_ids;
use candle_conversation::persistence::streams::StreamDecl;
use candle_conversation::persistence::SubstratePersistence;
use candle_conversation::projection::TimelineId;
use candle_conversation::provenance::boundaries::BoundaryRule;
use candle_conversation::provenance::probe_windows::{probe_windows, ProbeWindow};
use candle_conversation::provenance::window_score::{
    file_scores, late_fusion_votes, window_tally, FileAggregate,
};
use candle_conversation::provenance::{decode_wide_sigs, WideQSig};
use candle_conversation::substrate::Substrate;
use candle_conversation::{
    REPROJECT_EVERY_N_TOKENS, REPROJECT_MAX_PROBE_TOKENS, REPROJECT_TRIGGER_TEXTS,
};
use tokenizers::Tokenizer;

/// The `code_reading` layer's persisted id (`zend/src/prompts/projection.yaml`).
const CODE_READING: u32 = 2;
/// The analysis layers that fork from a file's reading.
const ANALYSIS: [u32; 3] = [4, 11, 12];
const CHUNK: usize = 32;

/// One turn as the harness reads it.
struct Turn {
    tokens: Vec<u32>,
    sigs: Vec<WideQSig>,
}

/// One file's turns, in order.
struct File {
    timeline: u64,
    turns: Vec<Turn>,
}

/// The gallery: every file's covered turns laid end to end, windowed.
struct Gallery {
    tokens: Vec<WideQSig>,
    /// Each file's whole turns, as `(file, range)`: today's cases.
    turns: Vec<(usize, usize, Range<usize>)>,
    windows: Vec<Vec<Range<usize>>>,
    window_file: Vec<usize>,
    window_turn: Vec<(usize, usize)>,
    identity: Vec<bool>,
    file_tokens: Vec<usize>,
}

fn main() -> anyhow::Result<()> {
    let mut args = std::env::args().skip(1);
    let dir = PathBuf::from(args.next().unwrap_or_else(|| "..".into()));
    let per_source: usize = args.next().map(|a| a.parse()).transpose()?.unwrap_or(40);

    let t = Instant::now();
    let mut sub = Substrate::new();
    let mut sp = SubstratePersistence::open_in_with_substrate_read_only(&dir, &mut sub)?;
    println!(
        "opened {} read-only in {:.1}s",
        dir.display(),
        t.elapsed().as_secs_f64()
    );
    let tokenizer = match sp.read_tokenizer_bytes()? {
        Some(bytes) => Tokenizer::from_bytes(bytes).map_err(|e| anyhow::anyhow!("{e}"))?,
        None => anyhow::bail!("the substrate holds no tokenizer"),
    };
    let triggers: Vec<String> = REPROJECT_TRIGGER_TEXTS.map(String::from).to_vec();
    let rule = BoundaryRule::from_tokenizer(&tokenizer, &triggers, REPROJECT_EVERY_N_TOKENS);
    println!("boundary rule: {rule:?}");

    // Every turn of the two kinds of conversation, by timeline. A turn's tokens
    // are not held in RAM by an open: each is read from its `Tokens` record.
    let mut found = Vec::new();
    for (sid, e) in sub.all_streams() {
        let Some(StreamDecl::Turn(d)) = e.decl.as_ref() else {
            continue;
        };
        if d.layer_id != CODE_READING && !ANALYSIS.contains(&d.layer_id) {
            continue;
        }
        let sigs = e
            .wide_q_sigs
            .as_ref()
            .and_then(|b| decode_wide_sigs(b))
            .unwrap_or_default();
        found.push((d.layer_id, d.timeline_id, d.turn_index, sid, sigs));
    }
    let mut by_layer: HashMap<u32, BTreeMap<u64, BTreeMap<u32, Turn>>> = HashMap::new();
    for (layer, timeline, turn_index, sid, sigs) in found {
        let tokens = match sp.read_tokens(&sub, sid)? {
            Some(bytes) => decode_token_ids(&bytes)?,
            None => Vec::new(),
        };
        by_layer
            .entry(layer)
            .or_default()
            .entry(timeline)
            .or_default()
            .insert(turn_index, Turn { tokens, sigs });
    }

    let files: Vec<File> = by_layer
        .remove(&CODE_READING)
        .unwrap_or_default()
        .into_iter()
        .map(|(timeline, turns)| File {
            timeline,
            turns: turns.into_values().collect(),
        })
        .collect();
    coverage(&files);
    if files.len() < 2 {
        println!("fewer than two code_reading files — nothing to rank yet");
        return Ok(());
    }

    let gallery = build_gallery(&files, &rule);
    println!(
        "gallery: {} files, {} tokens, {} windows",
        files.len(),
        gallery.tokens.len(),
        gallery.windows.len()
    );

    // Probes: (source, target file, the probe's signatures, the turn left out).
    let file_of: HashMap<u64, usize> = files
        .iter()
        .enumerate()
        .map(|(i, f)| (f.timeline, i))
        .collect();
    let mut probes: Vec<(&str, usize, Vec<WideQSig>, Option<(usize, usize)>)> = Vec::new();
    let stride = (files.len() / per_source).max(1);
    for (fi, f) in files.iter().enumerate().step_by(stride).take(per_source) {
        let held: Vec<usize> = (0..f.turns.len())
            .filter(|&i| covered(&f.turns[i]))
            .collect();
        if held.len() < 2 {
            continue;
        }
        // The control: a window of the file's own pages, nothing left out —
        // every rule must find it. Then the closing summary's last window, its
        // turn left out; and the request's, its turn left out — a question
        // about the file.
        let pages = &f.turns[held[held.len() - 1]];
        if let Some(w) =
            probe_windows(&pages.tokens, &rule, REPROJECT_MAX_PROBE_TOKENS, CHUNK).get(3)
        {
            probes.push(("self (control)", fi, gather(&pages.sigs, w), None));
        }
        for (source, ti) in [
            ("summary-to-rest", held[held.len() - 1]),
            ("request-to-rest", held[0]),
        ] {
            let turn = &f.turns[ti];
            if let Some(w) =
                probe_windows(&turn.tokens, &rule, REPROJECT_MAX_PROBE_TOKENS, CHUNK).last()
            {
                probes.push((source, fi, gather(&turn.sigs, w), Some((fi, ti))));
            }
        }
    }
    let mut units = 0usize;
    for layer in ANALYSIS {
        for (timeline, turns) in by_layer.remove(&layer).unwrap_or_default() {
            if units >= per_source {
                break;
            }
            let parent = TimelineId::from_raw(timeline).and_then(|tl| sub.parent_timeline(tl));
            let Some(&fi) = parent.and_then(|p| file_of.get(&p.raw())) else {
                continue;
            };
            let Some(turn) = turns.values().rev().find(|t| covered(t)) else {
                continue;
            };
            if let Some(w) =
                probe_windows(&turn.tokens, &rule, REPROJECT_MAX_PROBE_TOKENS, CHUNK).last()
            {
                probes.push(("analysis-unit", fi, gather(&turn.sigs, w), None));
                units += 1;
            }
        }
    }
    println!("probes: {}", probes.len());

    let rules: Vec<String> = ["late-fusion (today)".to_string()]
        .into_iter()
        .chain(
            FileAggregate::ALL
                .iter()
                .map(|a| format!("window votes / {}", a.name())),
        )
        .collect();
    // [source][rule] → ranks of the target; [rule] → false top-10 hits per file.
    let mut ranks: BTreeMap<(&str, &str), Vec<usize>> = BTreeMap::new();
    let mut false_hits: HashMap<&str, Vec<usize>> = HashMap::new();
    let t = Instant::now();
    for (source, target, probe, left_out) in &probes {
        let mut per_rule: Vec<Vec<f32>> = Vec::new();
        // The left-out turn's tokens belong to no file.
        let token_file: Vec<usize> = gallery
            .turns
            .iter()
            .flat_map(|(f, t, r)| {
                let file = if Some((*f, *t)) == *left_out {
                    usize::MAX
                } else {
                    *f
                };
                repeat_n(file, r.len())
            })
            .collect();
        let votes = late_fusion_votes(probe, &gallery.tokens, &token_file, files.len(), &[]);
        // Today: every vote summed per file.
        let mut today = vec![0.0f32; files.len()];
        for v in &votes {
            today[v.file] += v.weight;
        }
        per_rule.push(today);
        let tally = window_tally(&votes, &gallery.windows, gallery.tokens.len());
        let live: Vec<bool> = gallery
            .window_turn
            .iter()
            .map(|wt| Some(*wt) != *left_out)
            .collect();
        for a in FileAggregate::ALL {
            per_rule.push(file_scores(
                &tally,
                &gallery.window_file,
                &gallery.identity,
                &live,
                files.len(),
                a,
            ));
        }
        for (rule_name, s) in rules.iter().map(String::as_str).zip(per_rule) {
            let mut order: Vec<usize> = (0..s.len()).collect();
            order.sort_by(|&a, &b| s[b].total_cmp(&s[a]));
            let rank = order.iter().position(|&f| f == *target).unwrap_or(s.len()) + 1;
            ranks.entry((source, rule_name)).or_default().push(rank);
            let hits = false_hits
                .entry(rule_name)
                .or_insert_with(|| vec![0; files.len()]);
            for &f in order.iter().take(10).filter(|&&f| f != *target) {
                hits[f] += 1;
            }
        }
    }
    println!(
        "scored {} probes in {:.1}s\n",
        probes.len(),
        t.elapsed().as_secs_f64()
    );

    println!(
        "{:<16} {:<22} {:>5} {:>6} {:>6} {:>6} {:>6}",
        "source", "rule", "n", "R@1", "R@5", "R@10", "MRR"
    );
    for ((source, rule_name), r) in &ranks {
        let n = r.len() as f32;
        let at = |k: usize| r.iter().filter(|&&x| x <= k).count() as f32 / n;
        let mrr = r.iter().map(|&x| 1.0 / x as f32).sum::<f32>() / n;
        println!(
            "{source:<16} {rule_name:<22} {:>5} {:>6.2} {:>6.2} {:>6.2} {:>6.3}",
            r.len(),
            at(1),
            at(5),
            at(10),
            mrr
        );
    }
    println!();
    for rule_name in &rules {
        if let Some(hits) = false_hits.get(rule_name.as_str()) {
            println!(
                "size bias {rule_name:<22} spearman(file tokens, false top-10 hits) = {:+.3}",
                spearman(&gallery.file_tokens, hits)
            );
        }
    }
    Ok(())
}

/// Whether a turn's signature covers its tokens one to one.
fn covered(t: &Turn) -> bool {
    !t.tokens.is_empty() && t.sigs.len() == t.tokens.len()
}

/// Gate 2: every turn's signature tokens against its tokens.
fn coverage(files: &[File]) {
    let (mut turns, mut exact, mut short, mut none, mut long) = (0, 0, 0, 0, 0);
    let mut examples = Vec::new();
    for f in files {
        for (i, t) in f.turns.iter().enumerate() {
            turns += 1;
            match t.sigs.len().cmp(&t.tokens.len()) {
                _ if t.sigs.is_empty() => none += 1,
                Ordering::Equal => exact += 1,
                Ordering::Less => short += 1,
                Ordering::Greater => long += 1,
            }
            if t.sigs.len() != t.tokens.len() && examples.len() < 8 {
                examples.push(format!(
                    "  {:016x} turn {i}: {} tokens, {} signatures",
                    f.timeline,
                    t.tokens.len(),
                    t.sigs.len()
                ));
            }
        }
    }
    println!(
        "gate 2 coverage: {} files, {turns} turns — {exact} exact, {short} short, {long} long, {none} without a signature",
        files.len()
    );
    for e in examples {
        println!("{e}");
    }
}

/// A probe window's signatures.
fn gather(sigs: &[WideQSig], w: &ProbeWindow) -> Vec<WideQSig> {
    w.ranges().flat_map(|r| sigs[r].iter().cloned()).collect()
}

fn build_gallery(files: &[File], rule: &BoundaryRule) -> Gallery {
    let mut g = Gallery {
        tokens: Vec::new(),
        turns: Vec::new(),
        windows: Vec::new(),
        window_file: Vec::new(),
        window_turn: Vec::new(),
        identity: Vec::new(),
        file_tokens: vec![0; files.len()],
    };
    for (fi, f) in files.iter().enumerate() {
        let covered_turns: Vec<usize> = (0..f.turns.len())
            .filter(|&i| covered(&f.turns[i]))
            .collect();
        let first = covered_turns.first().copied();
        let last = covered_turns.last().copied();
        for &ti in &covered_turns {
            let t = &f.turns[ti];
            let base = g.tokens.len();
            g.tokens.extend(t.sigs.iter().cloned());
            g.turns.push((fi, ti, base..base + t.sigs.len()));
            g.file_tokens[fi] += t.sigs.len();
            let windows = probe_windows(&t.tokens, rule, REPROJECT_MAX_PROBE_TOKENS, CHUNK);
            let n = windows.len();
            for (wi, w) in windows.iter().enumerate() {
                g.windows
                    .push(w.ranges().map(|r| base + r.start..base + r.end).collect());
                g.window_file.push(fi);
                g.window_turn.push((fi, ti));
                // The first turn's opening window holds the request; the last
                // turn's closing window the summary.
                g.identity
                    .push((Some(ti) == first && wi == 0) || (Some(ti) == last && wi + 1 == n));
            }
        }
    }
    g
}

/// Spearman's rank correlation of two equal-length series.
fn spearman(a: &[usize], b: &[usize]) -> f32 {
    let rank = |v: &[usize]| -> Vec<f32> {
        let mut idx: Vec<usize> = (0..v.len()).collect();
        idx.sort_by_key(|&i| v[i]);
        let mut r = vec![0.0f32; v.len()];
        let mut i = 0;
        while i < idx.len() {
            let mut j = i;
            while j + 1 < idx.len() && v[idx[j + 1]] == v[idx[i]] {
                j += 1;
            }
            let avg = (i + j) as f32 / 2.0;
            for k in i..=j {
                r[idx[k]] = avg;
            }
            i = j + 1;
        }
        r
    };
    let (ra, rb) = (rank(a), rank(b));
    let n = ra.len() as f32;
    let (ma, mb) = (ra.iter().sum::<f32>() / n, rb.iter().sum::<f32>() / n);
    let cov: f32 = ra.iter().zip(&rb).map(|(x, y)| (x - ma) * (y - mb)).sum();
    let va: f32 = ra.iter().map(|x| (x - ma).powi(2)).sum();
    let vb: f32 = rb.iter().map(|y| (y - mb).powi(2)).sum();
    if va == 0.0 || vb == 0.0 {
        0.0
    } else {
        cov / (va.sqrt() * vb.sqrt())
    }
}
