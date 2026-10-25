//! The substrate as the gates read it: each conversation's turns, their tokens
//! and signatures, and the `code_reading` gallery windowed as a projection's
//! probes are cut.

use std::cmp::Ordering;
use std::collections::{BTreeMap, HashMap};
use std::ops::Range;

use candle_conversation::persistence::resume::decode_token_ids;
use candle_conversation::persistence::streams::StreamDecl;
use candle_conversation::persistence::SubstratePersistence;
use candle_conversation::projection::TimelineId;
use candle_conversation::provenance::boundaries::BoundaryRule;
use candle_conversation::provenance::probe_windows::{probe_windows, ProbeWindow};
use candle_conversation::provenance::window_score::FlatSigs;
use candle_conversation::provenance::{decode_wide_sigs, WideQSig};
use candle_conversation::substrate::Substrate;
use candle_conversation::turn_layout::TurnSegment;
use candle_conversation::REPROJECT_MAX_PROBE_TOKENS;
use tokenizers::Tokenizer;

/// The `code_reading` layer's persisted id (`zend/src/prompts/projection.yaml`).
pub const CODE_READING: u32 = 2;
/// The analysis layers that fork from a file's reading.
pub const ANALYSIS: [u32; 3] = [4, 11, 12];
/// The dialogue layer's persisted id.
pub const DIALOGUE: u32 = 10;
pub const CHUNK: usize = 32;

/// One turn as the gates read it.
pub struct Turn {
    /// Its tokens outside its reasoning: what its signature is captured over.
    /// The seal signs a turn's tokens but its `Thinking` spans, so a turn that
    /// reasoned signs fewer tokens than it holds.
    pub tokens: Vec<u32>,
    pub sigs: Vec<WideQSig>,
}

impl Turn {
    /// The turn whose `Tokens` record holds `all`, laid out by `segments`.
    pub fn new(all: Vec<u32>, sigs: Vec<WideQSig>, segments: &[TurnSegment]) -> Turn {
        let thinking: Vec<Range<usize>> = segments
            .iter()
            .filter(|s| matches!(s, TurnSegment::Thinking { .. }))
            .filter_map(|s| s.kv())
            .map(|kv| kv.range())
            .collect();
        let tokens = if thinking.is_empty() {
            all
        } else {
            all.into_iter()
                .enumerate()
                .filter(|(i, _)| !thinking.iter().any(|r| r.contains(i)))
                .map(|(_, t)| t)
                .collect()
        };
        Turn { tokens, sigs }
    }

    /// Whether its signature covers its signed tokens one to one.
    pub fn covered(&self) -> bool {
        !self.tokens.is_empty() && self.sigs.len() == self.tokens.len()
    }

    /// Its probe-shaped windows.
    pub fn windows(&self, rule: &BoundaryRule) -> Vec<ProbeWindow> {
        probe_windows(&self.tokens, rule, REPROJECT_MAX_PROBE_TOKENS, CHUNK)
    }
}

/// One `code_reading` file: its timeline, the workspace path its request
/// names, its turns in order, and its calibrations' questions.
pub struct File {
    pub timeline: u64,
    pub path: Option<String>,
    pub turns: Vec<Turn>,
    pub questions: Vec<Question>,
}

/// One calibration question: its calibration's timeline, its turn there, and
/// its signatures. A calibration is distilled provenance-only, so its tokens
/// are gone and only the signatures are read.
pub struct Question {
    pub timeline: u64,
    pub turn: u32,
    pub sigs: Vec<WideQSig>,
}

/// Every calibration's questions, keyed by the timeline it calibrates.
pub fn questions(sub: &Substrate) -> HashMap<u64, Vec<Question>> {
    let mut of: HashMap<u64, u64> = HashMap::new();
    for (unit, cals) in sub.calibrations() {
        for cal in cals {
            of.insert(cal.raw(), unit.raw());
        }
    }
    let mut out: HashMap<u64, Vec<Question>> = HashMap::new();
    for (_, e) in sub.all_streams() {
        let Some(StreamDecl::Turn(d)) = e.decl.as_ref() else {
            continue;
        };
        let Some(&unit) = of.get(&d.timeline_id) else {
            continue;
        };
        let sigs = e
            .wide_q_sigs
            .as_ref()
            .and_then(|b| decode_wide_sigs(b))
            .unwrap_or_default();
        if sigs.is_empty() {
            continue;
        }
        out.entry(unit).or_default().push(Question {
            timeline: d.timeline_id,
            turn: d.turn_index,
            sigs,
        });
    }
    for qs in out.values_mut() {
        qs.sort_by_key(|q| (q.timeline, q.turn));
    }
    out
}

/// Every turn of `layers`, by layer, then timeline, then turn index — each
/// turn's tokens read from its `Tokens` record, which an open does not hold.
pub type ByLayer = HashMap<u32, BTreeMap<u64, BTreeMap<u32, Turn>>>;

pub fn load(
    sub: &Substrate,
    sp: &mut SubstratePersistence,
    layers: &[u32],
) -> anyhow::Result<ByLayer> {
    let mut found = Vec::new();
    for (sid, e) in sub.all_streams() {
        let Some(StreamDecl::Turn(d)) = e.decl.as_ref() else {
            continue;
        };
        if !layers.contains(&d.layer_id) {
            continue;
        }
        // A tombstoned conversation or turn votes in no live scan: a file's
        // superseded reading, a calibration's draft.
        let live = TimelineId::from_raw(d.timeline_id)
            .is_some_and(|tl| !sub.is_tombstoned(tl) && !sub.is_turn_tombstoned(tl, d.turn_index));
        if !live {
            continue;
        }
        let sigs = e
            .wide_q_sigs
            .as_ref()
            .and_then(|b| decode_wide_sigs(b))
            .unwrap_or_default();
        found.push((
            d.layer_id,
            d.timeline_id,
            d.turn_index,
            sid,
            sigs,
            d.segments.clone(),
        ));
    }
    let mut by_layer = ByLayer::new();
    for (layer, timeline, turn_index, sid, sigs, segments) in found {
        let tokens = match sp.read_tokens(sub, sid)? {
            Some(bytes) => decode_token_ids(&bytes)?,
            None => Vec::new(),
        };
        by_layer
            .entry(layer)
            .or_default()
            .entry(timeline)
            .or_default()
            .insert(turn_index, Turn::new(tokens, sigs, &segments));
    }
    Ok(by_layer)
}

/// The `code_reading` files of `by_layer`, each named by the path its opening
/// request reads, with its calibrations' questions from `questions`.
pub fn files(
    by_layer: &mut ByLayer,
    tokenizer: &Tokenizer,
    mut questions: HashMap<u64, Vec<Question>>,
) -> Vec<File> {
    by_layer
        .remove(&CODE_READING)
        .unwrap_or_default()
        .into_iter()
        .map(|(timeline, turns)| {
            let turns: Vec<Turn> = turns.into_values().collect();
            let path = turns
                .first()
                .and_then(|t| tokenizer.decode(&t.tokens, false).ok())
                .and_then(|text| requested_path(&text));
            File {
                timeline,
                path,
                turns,
                questions: questions.remove(&timeline).unwrap_or_default(),
            }
        })
        .collect()
}

/// The workspace path a `code_reading` request names: "Read the entire
/// contents of `<path>` in the `<repo>` repository" (`zend::code_read`).
pub fn requested_path(text: &str) -> Option<String> {
    let rest = text.split("Read the entire contents of `").nth(1)?;
    let (path, rest) = rest.split_once("` in the `")?;
    let (repo, _) = rest.split_once("` repository")?;
    Some(format!("{repo}/{path}"))
}

/// Gate 2: every turn's signature tokens against its tokens.
pub fn coverage(files: &[File]) {
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
        "gate 2 coverage: {} files, {turns} turns — {exact} exact, {short} short, \
         {long} long, {none} without a signature",
        files.len()
    );
    for e in examples {
        println!("{e}");
    }
}

/// A probe window's signatures.
pub fn gather(sigs: &[WideQSig], w: &ProbeWindow) -> Vec<WideQSig> {
    w.ranges().flat_map(|r| sigs[r].iter().cloned()).collect()
}

/// The gallery: every file's covered turns laid end to end, windowed.
pub struct Gallery {
    pub flat: FlatSigs,
    /// Each covered turn, as `(file, turn, range)`.
    pub turns: Vec<(usize, usize, Range<usize>)>,
    pub windows: Vec<Vec<Range<usize>>>,
    pub window_file: Vec<usize>,
    pub window_turn: Vec<(usize, usize)>,
    pub identity: Vec<bool>,
    pub file_tokens: Vec<usize>,
}

impl Gallery {
    /// `None` when no covered turn holds a signature, or they are not all one
    /// width.
    pub fn build(files: &[File], rule: &BoundaryRule) -> Option<Gallery> {
        let mut tokens: Vec<WideQSig> = Vec::new();
        let mut turns = Vec::new();
        let mut windows = Vec::new();
        let mut window_file = Vec::new();
        let mut window_turn = Vec::new();
        let mut identity = Vec::new();
        let mut file_tokens = vec![0; files.len()];
        for (fi, f) in files.iter().enumerate() {
            let covered: Vec<usize> = (0..f.turns.len())
                .filter(|&i| f.turns[i].covered())
                .collect();
            let first = covered.first().copied();
            let last = covered.last().copied();
            for &ti in &covered {
                let t = &f.turns[ti];
                let base = tokens.len();
                tokens.extend(t.sigs.iter().cloned());
                turns.push((fi, ti, base..base + t.sigs.len()));
                file_tokens[fi] += t.sigs.len();
                let cut = t.windows(rule);
                let n = cut.len();
                for (wi, w) in cut.iter().enumerate() {
                    windows.push(w.ranges().map(|r| base + r.start..base + r.end).collect());
                    window_file.push(fi);
                    window_turn.push((fi, ti));
                    // The first turn's opening window holds the request; the
                    // last turn's closing window the summary.
                    identity
                        .push((Some(ti) == first && wi == 0) || (Some(ti) == last && wi + 1 == n));
                }
            }
        }
        Some(Gallery {
            flat: FlatSigs::new(&tokens)?,
            turns,
            windows,
            window_file,
            window_turn,
            identity,
            file_tokens,
        })
    }
}
