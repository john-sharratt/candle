//! Gate 3: retrieval of a file from probes that are not its own tokens
//! (`docs/provenance_scoped_signatures.md` §3.3).
//!
//! - **self (control)** — a window of a file's own pages, nothing left out;
//!   every rule must rank the file first.
//! - **summary-to-rest** — the last turn's last window, that turn left out.
//! - **request-to-rest** — the first turn's last window, that turn left out.
//! - **analysis-unit** — the last window of a `dependency_analysis`,
//!   `code_review` or `security_review` unit's own turns, which must find the
//!   file the unit forked from.

use std::collections::{BTreeMap, HashMap};
use std::time::Instant;

use candle_conversation::projection::TimelineId;
use candle_conversation::provenance::boundaries::BoundaryRule;
use candle_conversation::provenance::WideQSig;
use candle_conversation::substrate::Substrate;

use crate::corpus::{gather, ByLayer, File, Gallery, ANALYSIS};
use crate::ranking::{order, rule_names, score};
use crate::stats::{mrr, recall, spearman};

/// Probes taken from each analysis unit's answer, its last windows first.
const UNIT_WINDOWS: usize = 8;

/// `(source, target file, the probe's signatures, the turn left out)`.
pub type Probe = (&'static str, usize, Vec<WideQSig>, Option<(usize, usize)>);

pub fn run(
    sub: &Substrate,
    files: &[File],
    gallery: &Gallery,
    by_layer: &mut ByLayer,
    rule: &BoundaryRule,
    per_source: usize,
) {
    let probes = probes(sub, files, by_layer, rule, per_source);
    println!("probes: {}", probes.len());
    let rules = rule_names();
    let mut ranks: BTreeMap<(&str, &str), Vec<usize>> = BTreeMap::new();
    let mut false_hits: HashMap<&str, Vec<usize>> = HashMap::new();
    let t = Instant::now();
    for (source, target, probe, left_out) in &probes {
        let per_rule = score(probe, gallery, files.len(), *left_out);
        for (name, s) in rules.iter().map(String::as_str).zip(per_rule) {
            let order = order(&s);
            let rank = order.iter().position(|&f| f == *target).unwrap_or(s.len()) + 1;
            ranks.entry((source, name)).or_default().push(rank);
            let hits = false_hits
                .entry(name)
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
        "{:<16} {:<40} {:>5} {:>6} {:>6} {:>6} {:>6}",
        "source", "rule", "n", "R@1", "R@5", "R@10", "MRR"
    );
    for ((source, name), r) in &ranks {
        println!(
            "{source:<16} {name:<40} {:>5} {:>6.2} {:>6.2} {:>6.2} {:>6.3}",
            r.len(),
            recall(r, 1),
            recall(r, 5),
            recall(r, 10),
            mrr(r)
        );
    }
    println!();
    for name in &rules {
        if let Some(hits) = false_hits.get(name.as_str()) {
            println!(
                "size bias {name:<40} spearman(file tokens, false top-10 hits) = {:+.3}",
                spearman(&gallery.file_tokens, hits)
            );
        }
    }
}

pub fn probes(
    sub: &Substrate,
    files: &[File],
    by_layer: &mut ByLayer,
    rule: &BoundaryRule,
    per_source: usize,
) -> Vec<Probe> {
    let mut probes = Vec::new();
    let stride = (files.len() / per_source.max(1)).max(1);
    for (fi, f) in files.iter().enumerate().step_by(stride).take(per_source) {
        let held: Vec<usize> = (0..f.turns.len())
            .filter(|&i| f.turns[i].covered())
            .collect();
        if held.len() < 2 {
            continue;
        }
        let pages = &f.turns[held[held.len() - 1]];
        if let Some(w) = pages.windows(rule).get(3) {
            probes.push(("self (control)", fi, gather(&pages.sigs, w), None));
        }
        for (source, ti) in [
            ("summary-to-rest", held[held.len() - 1]),
            ("request-to-rest", held[0]),
        ] {
            let turn = &f.turns[ti];
            if let Some(w) = turn.windows(rule).last() {
                probes.push((source, fi, gather(&turn.sigs, w), Some((fi, ti))));
            }
        }
    }
    let file_of: HashMap<u64, usize> = files
        .iter()
        .enumerate()
        .map(|(i, f)| (f.timeline, i))
        .collect();
    let mut units = 0usize;
    for layer in ANALYSIS {
        let timelines = by_layer.remove(&layer).unwrap_or_default();
        let (mut parented, mut filed, mut held) = (0usize, 0usize, 0usize);
        let n_units = timelines.len();
        for (timeline, turns) in timelines {
            if units >= per_source {
                break;
            }
            let parent = TimelineId::from_raw(timeline).and_then(|tl| sub.parent_timeline(tl));
            parented += usize::from(parent.is_some());
            let Some(&fi) = parent.and_then(|p| file_of.get(&p.raw())) else {
                continue;
            };
            filed += 1;
            let Some(turn) = turns.values().rev().find(|t| t.covered()) else {
                continue;
            };
            held += 1;
            // Every window of the unit's answer is about its subject; its tool
            // rounds read other files, which a probe from them should find.
            let windows = turn.windows(rule);
            for w in windows.iter().rev().take(UNIT_WINDOWS) {
                probes.push(("analysis-unit", fi, gather(&turn.sigs, w), None));
            }
            units += 1;
        }
        println!(
            "layer {layer}: {n_units} units, {parented} with a parent, {filed} whose parent \
             is a file, {held} with a covered turn"
        );
    }
    probes
}
