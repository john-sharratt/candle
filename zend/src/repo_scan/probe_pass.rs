//! Running the probe layer: generating a folder's candidate questions, and
//! ingesting the admitted ones as decoded turns.
//!
//! The pure half of this layer lives in [`crate::repo_scan::probe`] — symbols,
//! frequencies, filters, planning — and is testable without a GPU. This file is
//! the half that touches the engine, and it makes two structural choices worth
//! stating plainly, because both were arrived at by elimination.
//!
//! # Generation runs in its own conversation
//!
//! The folder's conversation already holds the listing, the anchor excerpt and
//! the summary, so it looks like the natural place to ask for questions. It is
//! not, for two reasons that compound:
//!
//! * The generator's answer is a **list of forty-eight questions**, and asked
//!   there it seals into the folder's own timeline as retrievable content. A
//!   scan would then match that block — the densest, most question-shaped thing
//!   the folder owns — instead of any individual probe, and the folder's
//!   strongest signal would be a numbered list that answers nothing.
//! * [`Sequence::fork`] does not inherit the parent's turns; it projects the
//!   system prompt and nothing else. So there is no way to borrow that context
//!   for a branch and throw the branch away.
//!
//! Hence a fresh conversation carrying the compact evidence block
//! ([`probe::render::evidence`]) — a few hundred tokens against the folder
//! chain's ~1,500 — tombstoned the moment its candidates are parsed.
//!
//! # Probes are decoded on their own conversation, not the folder's
//!
//! Appending the probe turns to the folder's own conversation looks cheapest —
//! its evidence is already resident, so each probe would cost only its marginal
//! tokens. It is wrong, and the reason is the system prompt.
//!
//! The folder conversation is created with the **summariser** framing:
//! `persona: summarize` ("reply with ONLY a concise summary — at most two
//! sentences"), `response_length: terse`, and `thinking_effort: off`. A probe
//! answered under it produces two clipped sentences and **no `<think>` block at
//! all** — and the think block is the point. A provenance scan runs continuously
//! *during* decode, so when a live conversation is mid-reasoning about paging,
//! the Q hitting the corpus is reasoning-shaped rather than question-shaped. A
//! corpus of terse answers is indexed in the wrong shape for most of the moments
//! it is actually scanned.
//!
//! So each folder gets a second conversation framed to *answer*
//! ([`ANSWER_BRANCH`]), prefilled with one turn: the folder's request and its
//! decoded summary. That is deliberately not the full evidence chain — it is
//! what the projection would inject at query time, so a probe is answered under
//! the same conditions a real query creates, at ~200 tokens per folder instead
//! of ~1,500.
//!
//! Both conversations are tagged with the directory, and the tag is the join: a
//! scan hitting a probe resolves to the same folder as one hitting its summary.
//! The two artifacts stay cleanly separated — the summary is the payload, the
//! probes are the index over it.
//!
//! The remaining price is that probe *n* answers with probes 1..n−1 in its
//! context. [`probe::plan`] interleaves the registers precisely so that cost
//! falls evenly across all four rather than landing on whichever register came
//! last — and the per-register held-out scores in [`probe::harness`] are what
//! will say whether it cost anything at all.

use std::collections::BTreeMap;

use candle_conversation::{SelectionState, Sequence, TurnOptions};

use crate::repo_scan::dir_unit::DirUnit;
use crate::repo_scan::probe::idf::TermIndex;
use crate::repo_scan::probe::{plan, render, Probe, ProbeSet};

/// Decode budget for one generation turn: reasoning cap **plus** answer budget.
///
/// The reasoning block is not competed with, it is **excluded from the answer's
/// budget** — the two are sized separately and added, so a long think can never
/// starve the questions. It is then stripped before parsing
/// ([`probe::render::parse_register`]), so nothing the model deliberated reaches
/// the corpus.
///
/// Suppression was tried four ways first and none of them holds on this path:
///
/// 1. `thinking_effort: off` — a sentence in the prompt ("Answer directly,
///    without deliberating first"). The model overrode it routinely.
/// 2. The schema's `no_think` toggle — it does emit the marker, but a
///    statically-assembled prompt walks the schema in declaration order and the
///    toggle sits inside the section tree, so the marker landed mid-body where
///    the soft switch is not read.
/// 3. `/no_think` prepended to the conversation's system-prompt text. That text
///    is only the conversation's recorded prompt; the model reads the
///    projection of the shared schema, so the switch never reached it.
/// 4. `/no_think` on the user turn, the form Qwen reads from the latest message.
///
/// Splitting generation to one turn per register did not help either: a
/// twelve-question ask still drew a 12,752-character block, so the cause is not
/// task size. Budgeting for it is the fix that works.
const GENERATION_MAX_TOKENS: usize = GENERATION_THINK_CAP + GENERATION_ANSWER_BUDGET;

/// Runaway cutoff for the generator's reasoning block.
///
/// The block cannot be suppressed on this path (see below), so it is **budgeted
/// for instead of competed with**: the decode cap is this plus
/// [`GENERATION_ANSWER_BUDGET`], so reasoning never eats the questions. Sized
/// above every block measured — 2,183 / 5,342 / 6,363 / 6,735 / 12,752 / 15,021 /
/// 15,140 characters, the largest ~3,800 tokens — with headroom.
///
/// It is a cap, not an allowance. A block that runs past it is degenerate, and
/// the decode stops there rather than spending unbounded tokens on one folder;
/// the unterminated `<think>` then yields no questions, which is the honest
/// outcome and is reported as its own failure mode.
const GENERATION_THINK_CAP: usize = 4608;

/// Decode budget for the questions themselves — twelve short lines, with room
/// for a generator that numbers them verbosely.
const GENERATION_ANSWER_BUDGET: usize = 900;

/// Probe questions submitted in one stuffed prefill.
///
/// A stuffed group is ONE forward, so its cost is the sum of its cases — and the
/// wave's transient tier has to hold the activations for all of them at once.
/// A hand-authored folder can carry seventy-six questions, and submitting those
/// together made the tier ask for **5,939 MiB** with fourteen directories in
/// flight, which the partition refused; ten directories were lost to it.
///
/// Twenty-four is the calibration precedent — a tool's question set is grouped
/// the same way, at ~1.5k tokens of lossless K/V, for exactly this reason. It
/// also matches this layer's own per-directory budget, so an unauthored folder
/// is a single group and only a richly-authored one pays a second forward.
const PROBES_PER_GROUP: usize = 24;

/// The section-tree branch a probe-generation conversation frames on.
///
/// Mirrors `repo_scan::SUMMARIZE_BRANCH` in construction and differs in exactly
/// the way the task does: the persona writes questions rather than summaries,
/// and the length dial must not cap a forty-eight-item list.
pub const GENERATE_BRANCH: &[(&str, &str)] = &[
    // `summarize` pins "reply with ONLY a concise summary … at most two
    // sentences", which is a direct contradiction of the request and loses the
    // folder's whole probe set; `assistant` wraps the list in conversation the
    // parser then has to guess through.
    ("persona", "question_writer"),
    // `terse` and `standard` both cap length well under forty-eight questions.
    ("response_length", "comprehensive"),
    // There is nothing to deliberate about: the evidence is supplied and the
    // output shape is dictated. Thinking here spends the decode budget that the
    // last register needs.
    //
    // This dial is NOT sufficient on its own — it is a sentence in the prompt
    // ("Answer directly, without deliberating first"), and the model overrides it
    // routinely. The mechanism is `/no_think` on the user turn (see [`generate`]);
    // the dial stays because it also selects the matching sampling parameters.
    //
    // Measured with the dial alone: generations of 6,363 and 6,735 characters,
    // every one of them reasoning about how to answer, none reaching a single
    // question. Both directories lost their whole probe set.
    ("thinking_effort", "off"),
    // The worked FOLDER examples teach summarising, which is the one thing this
    // conversation must not do.
    ("summarize_examples", "absent"),
];

/// The branch a probe conversation frames on.
///
/// Probe turns are PREFILLED, never decoded, so the dials here do not steer a
/// generation — they only decide what the system prompt says while the questions
/// are laid down, which is the context their signatures are captured against.
/// Kept minimal and non-conversational for that reason.
pub const ANSWER_BRANCH: &[(&str, &str)] = &[
    ("persona", "assistant"),
    ("thinking_effort", "off"),
    ("response_length", "terse"),
    ("summarize_examples", "absent"),
];

/// A branch as the [`SelectionState`] a turn projects under: every
/// `(node, option)` pair selected, every other selector left to the target
/// layer's dials and then the schema default.
///
/// This is how a branch reaches the model. The conversation's system-prompt
/// text is only its recorded prompt; each turn's context is the projection of
/// the shared schema under the turn's selection, so a turn submitted without
/// its branch runs on the dialogue assistant's defaults.
pub fn branch_state(branch: &[(&str, &str)]) -> SelectionState {
    let mut sel = SelectionState::new();
    for (node, option) in branch {
        sel.select(*node, *option);
    }
    sel
}

/// Per-directory probe accounting, folded into the pass report.
#[derive(Debug, Clone, Default)]
pub struct ProbeStats {
    pub dirs_generated: usize,
    pub dirs_failed: usize,
    pub probes_ingested: usize,
    pub held_out: usize,
    pub rejections: BTreeMap<String, usize>,
    /// Directories with no distinctive term at all — they can carry systemic
    /// probes only, and that population is worth watching: if it grows, the
    /// symbol extractor has regressed rather than the repository having changed.
    pub dirs_without_seeds: usize,
    /// Directories that filled every register.
    ///
    /// The total probe count alone cannot distinguish "most folders are complete"
    /// from "half the folders got twice as many as they should" — and it is the
    /// per-directory floor, not the sum, that decides whether a query about a
    /// given folder has anything to hit.
    pub dirs_complete: usize,
    /// Probes admitted per register, so a starved register is visible without
    /// waiting for the retrieval eval. The registers fail differently — the
    /// specific three starve on the rarity gate, the systemic one on nothing at
    /// all — so a single admitted count hides which lever moved.
    pub by_register: BTreeMap<String, usize>,
}

impl ProbeStats {
    pub fn merge(&mut self, set: &ProbeSet, had_seeds: bool) {
        self.dirs_generated += 1;
        self.probes_ingested += set.probes.len();
        self.held_out += set.held_out.len();
        if !had_seeds {
            self.dirs_without_seeds += 1;
        }
        if set.is_complete() {
            self.dirs_complete += 1;
        }
        for probe in &set.probes {
            *self
                .by_register
                .entry(probe.register.id().to_string())
                .or_insert(0) += 1;
        }
        for (tag, n) in set.rejection_tally() {
            *self.rejections.entry(tag.to_string()).or_insert(0) += n;
        }
    }
}

/// Ask a throwaway conversation for this folder's candidate questions.
///
/// `summary` is the folder's own decoded description; `seeds` the distinctive
/// terms. The conversation is created, decoded once, and left for the caller to
/// tombstone — it must never persist, and it must never be tagged into the
/// `repo_map` layer.
pub fn generate(
    conv: &mut Sequence,
    unit: &DirUnit,
    summary: &str,
    seeds: &[&str],
) -> anyhow::Result<Vec<(crate::repo_scan::probe::Register, Vec<String>)>> {
    {
        let mut prompt = render::evidence(unit, summary, seeds);
        prompt.push('\n');
        prompt.push_str(&render::instruction(unit, seeds));
        // The soft switch, on the USER turn — the one form of it this path can
        // actually reach.
        //
        // Qwen reads `/think` and `/no_think` from the latest user message, and that
        // is the only lever left after three failures: the `thinking_effort` dial is
        // advisory prose, the schema's `no_think` toggle lands mid-prompt where the
        // switch is not read, and the conversation's system-prompt text is only its
        // recorded prompt, which the model never reads.
        //
        // Left unsuppressed the reasoning is not merely wasteful, it is unbounded:
        // measured blocks of 5,342 and 15,102 characters, the latter past even a
        // 3,500-token budget, so the folder decoded no question at all.
        prompt.push_str("\n/no_think\n");

        let options = TurnOptions {
            max_tokens: Some(GENERATION_MAX_TOKENS),
            // The question writer's branch: the turn's projection is the prompt
            // this conversation runs on.
            selection: branch_state(GENERATE_BRANCH),
            ..Default::default()
        };
        let handle = conv
            .submit_turn_with_options(&prompt, options)
            .map_err(|e| anyhow::anyhow!("probe generation submit: {e}"))?;
        let response = handle
            .wait()
            .map_err(|e| anyhow::anyhow!("probe generation decode: {e}"))?;
        let text = response.text.clone();
        conv.finish_turn(handle, &response)
            .map_err(|e| anyhow::anyhow!("probe generation finish: {e}"))?;
        let parsed = render::parse(&text);
        if parsed.iter().all(|(_, q)| q.is_empty()) {
            // Separate the two failures, because they have different fixes: a
            // reasoning block that never closed means the cap was hit and should
            // move, while a closed block with no questions after it means the
            // model answered the wrong thing and the PROMPT should move.
            let ran_away = text.contains("<think>") && !text.contains("</think>");
            tracing::warn!(
                target: "zend::repo_scan::probe",
                dir = %unit.dir,
                chars = text.len(),
                think_ran_away = ran_away,
                think_cap = GENERATION_THINK_CAP,
                head = %text.chars().take(300).collect::<String>().replace('\n', " ⏎ "),
                "probe generation parsed to NOTHING",
            );
        }
        Ok(parsed)
    }
}

/// The folder's decoded summary — the last assistant turn on its conversation.
///
/// Read back rather than threaded through the chain call, because the chain
/// reports tokens rather than text and the summary is wanted only here.
pub fn last_summary(conv: &Sequence) -> String {
    conv.turns()
        .iter()
        .rev()
        .find(|t| t.role == candle_conversation::Role::Assistant)
        .map(|t| t.text.clone())
        .unwrap_or_default()
}

/// Build a probe set from questions already written in `.substrate.yaml`.
///
/// Authored questions go through the SAME admission as generated ones — a
/// hand-written question can still name a path, or be an instruction rather than
/// a question, and the filters are what keep those out of the corpus. What
/// changes is the budget: an authored file is a deliberate act, so every
/// admissible question in it is used rather than the first six per register.
/// Someone who wrote forty questions for a folder meant all forty.
pub fn admit_authored(
    unit: &DirUnit,
    index: &TermIndex,
    meta: &crate::repo_scan::metadata::FolderMetadata,
) -> ProbeSet {
    let candidates: Vec<(crate::repo_scan::probe::Register, Vec<String>)> =
        crate::repo_scan::probe::Register::ALL
            .into_iter()
            .map(|r| (r, meta.register(r)))
            .collect();
    plan::build_authored(&unit.dir, unit, index, &candidates)
}

/// Build a directory's probe set from generated candidates.
pub fn admit(
    unit: &DirUnit,
    index: &TermIndex,
    candidates: &[(crate::repo_scan::probe::Register, Vec<String>)],
) -> ProbeSet {
    plan::build(&unit.dir, unit, index, candidates)
}

/// Seed a probe conversation with the folder's summary — the context a live
/// query would have when this folder is retrieved.
///
/// One prefilled turn, no decode. The request half is the same wording the
/// summary chain used, so the pair reads as a real exchange rather than a bare
/// assertion the model has no reason to trust.
pub fn seed_context(conv: &mut Sequence, unit: &DirUnit, summary: &str) -> anyhow::Result<usize> {
    if summary.trim().is_empty() {
        return Ok(0);
    }
    // Every turn of this conversation — this seed and the probe groups after it
    // — projects under the answering branch.
    conv.set_selection(branch_state(ANSWER_BRANCH));
    conv.insert_turn_staged(
        &crate::repo_scan::render::render_request(unit),
        summary,
        vec!["repo_map".to_string(), unit.dir.clone()],
    )
    .map_err(|e| anyhow::anyhow!("probe context seed: {e}"))
}

/// Prefill a directory's probes as ONE stuffed group — no decode.
///
/// This is the shape the tool catalog uses, and the tool catalog is the part of
/// this system that already works: `CalibCase::Questions` puts a tool's whole
/// question set through `submit_prefilled_turn_group` in a single forward, with
/// an empty assistant half, and tool routing reaches 100% top-3 on it.
///
/// It replaces a design that decoded an answer per probe. That was chosen so
/// each probe would carry a `<think>` block — a reasoning-shaped signature for a
/// scan that runs mid-decode — and the cost of it was the whole problem:
///
/// | | decode tokens per folder | full workspace |
/// |---|--:|--:|
/// | one decoded answer per probe | ~11,000 | ~7.7 h at 140 tok/s |
/// | one stuffed prefill | ~0 | **~6 min** |
///
/// The measured justification for giving up the think blocks is that the tool
/// layer never had them either. What retrieval matches is the QUESTION, and the
/// question's signature is captured identically whether the turn was prefilled
/// or decoded — `submit_prefilled_turn_group` seals each carved case as a
/// complete turn, so the wide `sign(Q)` capture is the same one a decode would
/// have produced.
///
/// Returns the number of probe turns sealed.
pub fn ingest(conv: &mut Sequence, probes: &[Probe], dir: &str, pad_token: u32) -> usize {
    let mut sealed = 0usize;
    for group in probes.chunks(PROBES_PER_GROUP) {
        if candle_conversation::ingest_cancelled() {
            break;
        }
        let cases: Vec<(String, Vec<String>)> = group
            .iter()
            .map(|p| (p.text.clone(), p.tags(dir)))
            .collect();
        let submitted =
            conv.submit_prefilled_turn_group(&cases, branch_state(ANSWER_BRANCH), pad_token);
        let (handle, _indices) = match submitted {
            Ok(pair) => pair,
            Err(e) => {
                tracing::warn!(
                    target: "zend::repo_scan::probe",
                    dir, n = group.len(),
                    "probe group prefill failed; folder keeps what already landed: {e}",
                );
                break;
            }
        };
        let response = match handle.wait() {
            Ok(r) => r,
            Err(e) => {
                tracing::warn!(
                    target: "zend::repo_scan::probe",
                    dir, n = group.len(), "probe group prefill did not complete: {e}",
                );
                break;
            }
        };
        if let Err(e) = conv.finish_turn(handle, &response) {
            tracing::warn!(
                target: "zend::repo_scan::probe",
                dir, "probe group finish failed: {e}",
            );
            break;
        }
        sealed += group.len();
    }
    sealed
}

/// Where the held-out queries land, relative to the workspace.
///
/// Under `.zend/`, which [`crate::repo_scan::walk_workspace`] excludes from the
/// walk — so the holdout file can never become a file the repo map describes,
/// and its questions can never be ingested by the very pass they exist to score.
pub const HOLDOUT_FILE: &str = ".zend/probe_holdout.json";

/// One scored-query record: the question, the directory that should answer it,
/// and which register it came from.
#[derive(serde::Serialize, serde::Deserialize, Debug, Clone, PartialEq, Eq)]
pub struct HoldoutQuery {
    pub query: String,
    pub dir: String,
    pub register: String,
    /// Whether this query's own signature is in the corpus.
    ///
    /// Held-out queries are `false` and are the quality measure. Ingested probes
    /// are written too, marked `true`, because scoring them is a useful
    /// *necessary* condition — a probe that cannot retrieve its own folder even
    /// with its own signature resident is broken beyond argument — but the two
    /// populations must never be pooled, so the distinction is recorded rather
    /// than left to the reader.
    pub resident: bool,
}

/// Write every directory's held-out queries, plus its ingested probes marked
/// resident, as one JSON file for the retrieval harness.
///
/// **Merges with what is already on disk**, replacing only the directories this
/// pass produced. A pass sees just the directories it processed — a resume-cache
/// hit contributes nothing — so a plain overwrite silently destroys every
/// earlier pass's queries. That is not hypothetical: an interrupted run followed
/// by a restart reduced a populated holdout file to `[]`, and the probes it
/// described were unrecoverable because their directories now hit the cache and
/// would never be regenerated.
pub fn write_holdout(workspace: &std::path::Path, sets: &[ProbeSet]) -> anyhow::Result<usize> {
    let fresh: std::collections::HashSet<&str> = sets.iter().map(|s| s.dir.as_str()).collect();
    let mut rows: Vec<HoldoutQuery> = read_holdout(workspace)
        .unwrap_or_default()
        .into_iter()
        .filter(|q| !fresh.contains(q.dir.as_str()))
        .collect();
    for set in sets {
        for probe in &set.held_out {
            rows.push(HoldoutQuery {
                query: probe.text.clone(),
                dir: set.dir.clone(),
                register: probe.register.id().to_string(),
                resident: false,
            });
        }
        for probe in &set.probes {
            rows.push(HoldoutQuery {
                query: probe.text.clone(),
                dir: set.dir.clone(),
                register: probe.register.id().to_string(),
                resident: true,
            });
        }
    }
    // Deterministic order so two runs over an unchanged tree produce comparable
    // files and a diff between them means something.
    rows.sort_by(|a, b| {
        a.dir
            .cmp(&b.dir)
            .then_with(|| a.resident.cmp(&b.resident))
            .then_with(|| a.register.cmp(&b.register))
            .then_with(|| a.query.cmp(&b.query))
    });
    let path = workspace.join(HOLDOUT_FILE);
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    std::fs::write(&path, serde_json::to_vec_pretty(&rows)?)?;
    Ok(rows.len())
}

/// Read a holdout file back.
pub fn read_holdout(workspace: &std::path::Path) -> anyhow::Result<Vec<HoldoutQuery>> {
    let path = workspace.join(HOLDOUT_FILE);
    let body = std::fs::read_to_string(&path)
        .map_err(|e| anyhow::anyhow!("reading {}: {e}", path.display()))?;
    Ok(serde_json::from_str(&body)?)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::repo_scan::probe::filters::Rejection;
    use crate::repo_scan::probe::Register;

    /// The reasoning block is EXCLUDED from the answer's budget rather than
    /// competing with it: the cap is a think allowance plus an answer allowance,
    /// so a long think can shorten nothing but itself.
    ///
    /// Only the generator decodes now — probes are prefilled — so this is the
    /// one budget left that has to carry a reasoning block.
    #[test]
    fn the_generation_budget_covers_thinking_on_top_of_the_answer() {
        assert_eq!(
            GENERATION_MAX_TOKENS,
            GENERATION_THINK_CAP + GENERATION_ANSWER_BUDGET,
        );
    }

    /// The generator's cap must clear every reasoning block actually observed —
    /// the largest was 15,140 characters, roughly 3,800 tokens — or the decode
    /// stops mid-thought and the register yields nothing.
    #[test]
    fn the_generation_think_cap_clears_the_largest_observed_block() {
        // ~4 chars per token; the largest block actually seen was 15,140 chars.
        const LARGEST_OBSERVED_TOKENS: usize = 15_140 / 4;
        const _: () = assert!(GENERATION_THINK_CAP > LARGEST_OBSERVED_TOKENS);
    }

    /// Probes are PREFILLED, never decoded — one stuffed group per directory,
    /// the same call the tool calibration uses. Decoding an answer per probe cost
    /// ~11,000 tokens per folder and ~7.7 hours over the workspace; this costs a
    /// single forward. Asserted on the call itself, because the whole throughput
    /// argument rests on it.
    #[test]
    fn probes_are_prefilled_as_one_group_never_decoded() {
        let src = include_str!("probe_pass.rs");
        let ingest = src
            .split("pub fn ingest(")
            .nth(1)
            .expect("the ingest fn")
            .split("\n}")
            .next()
            .expect("its body");
        assert!(
            ingest.contains("submit_prefilled_turn_group"),
            "probe ingest must use the stuffed-group prefill",
        );
        assert!(
            !ingest.contains("max_tokens"),
            "probe ingest must not decode — no token budget belongs here",
        );
    }

    #[test]
    fn the_generation_branch_selects_the_question_writer_persona() {
        let persona = GENERATE_BRANCH
            .iter()
            .find(|(node, _)| *node == "persona")
            .map(|(_, option)| *option);
        assert_eq!(persona, Some("question_writer"));
    }

    /// A length dial that caps the answer truncates the LAST register written,
    /// which is always the systemic one — the register this design most depends
    /// on. Assert the branch cannot silently acquire a short dial.
    #[test]
    fn the_generation_branch_does_not_cap_length() {
        let length = GENERATE_BRANCH
            .iter()
            .find(|(node, _)| *node == "response_length")
            .map(|(_, option)| *option);
        assert_eq!(length, Some("comprehensive"));
    }

    /// The folder examples teach summarising, which is precisely what this
    /// conversation must not produce.
    #[test]
    fn the_generation_branch_drops_the_summarize_examples() {
        assert!(GENERATE_BRANCH.contains(&("summarize_examples", "absent")));
    }

    /// Every pair of a branch lands in the selection its turns submit with. A
    /// pair left out falls back to the layer dial and then the schema default —
    /// for `persona`, the dialogue assistant.
    #[test]
    fn a_branch_selects_every_one_of_its_options() {
        for branch in [GENERATE_BRANCH, ANSWER_BRANCH] {
            let sel = branch_state(branch);
            for (node, option) in branch {
                assert_eq!(sel.get(node), Some(*option), "`{node}` not selected");
            }
        }
    }

    /// The holdout file must land somewhere the walk cannot see. If it were
    /// walked, its questions would become a file the repo map describes — and
    /// the corpus would then contain the exact queries used to score it.
    #[test]
    fn the_holdout_file_is_written_where_the_walk_excludes_it() {
        assert!(HOLDOUT_FILE.starts_with(".zend/"), "{HOLDOUT_FILE}");
    }

    #[test]
    fn holdout_round_trips_with_residency_preserved() {
        let d = tempfile::tempdir().unwrap();
        let sets = vec![ProbeSet {
            dir: "a/".to_string(),
            probes: vec![Probe {
                text: "What is a ChunkGid?".to_string(),
                register: Register::Conceptual,
            }],
            held_out: vec![Probe {
                text: "What does GidPool hand out?".to_string(),
                register: Register::Conceptual,
            }],
            rejected: Vec::new(),
        }];
        let n = write_holdout(d.path(), &sets).unwrap();
        assert_eq!(n, 2);

        let back = read_holdout(d.path()).unwrap();
        let held: Vec<&HoldoutQuery> = back.iter().filter(|q| !q.resident).collect();
        let resident: Vec<&HoldoutQuery> = back.iter().filter(|q| q.resident).collect();
        assert_eq!(held.len(), 1);
        assert_eq!(held[0].query, "What does GidPool hand out?");
        assert_eq!(held[0].dir, "a/");
        assert_eq!(resident.len(), 1);
        assert_eq!(resident[0].query, "What is a ChunkGid?");
    }

    /// A later pass must not destroy an earlier one's queries. A pass only sees
    /// the directories it processed — everything else resume-cache hits — so an
    /// overwrite silently empties the file, and the lost queries cannot be
    /// regenerated because their directories are now cached.
    #[test]
    fn a_later_pass_merges_rather_than_replacing_earlier_directories() {
        let d = tempfile::tempdir().unwrap();
        let first = vec![ProbeSet {
            dir: "a/".to_string(),
            held_out: vec![Probe {
                text: "Where does the arena come from?".to_string(),
                register: Register::Locational,
            }],
            ..Default::default()
        }];
        write_holdout(d.path(), &first).unwrap();

        let second = vec![ProbeSet {
            dir: "b/".to_string(),
            held_out: vec![Probe {
                text: "What seals a chunk?".to_string(),
                register: Register::Mechanistic,
            }],
            ..Default::default()
        }];
        write_holdout(d.path(), &second).unwrap();

        let back = read_holdout(d.path()).unwrap();
        let dirs: std::collections::BTreeSet<&str> = back.iter().map(|q| q.dir.as_str()).collect();
        assert_eq!(dirs.len(), 2, "{back:?}");
        assert!(dirs.contains("a/"), "the earlier pass survived: {dirs:?}");
    }

    /// …and a re-run of the SAME directory replaces its rows rather than
    /// duplicating them, or a folder ingested twice votes twice in the score.
    #[test]
    fn re_running_a_directory_replaces_its_rows() {
        let d = tempfile::tempdir().unwrap();
        let make = |text: &str| {
            vec![ProbeSet {
                dir: "a/".to_string(),
                held_out: vec![Probe {
                    text: text.to_string(),
                    register: Register::Locational,
                }],
                ..Default::default()
            }]
        };
        write_holdout(d.path(), &make("First question?")).unwrap();
        write_holdout(d.path(), &make("Second question?")).unwrap();
        let back = read_holdout(d.path()).unwrap();
        assert_eq!(back.len(), 1, "{back:?}");
        assert_eq!(back[0].query, "Second question?");
    }

    /// Two runs over an unchanged tree must produce comparable files, or a diff
    /// between them is unreadable.
    #[test]
    fn holdout_rows_are_written_in_a_deterministic_order() {
        let d = tempfile::tempdir().unwrap();
        let sets = vec![
            ProbeSet {
                dir: "z/".to_string(),
                held_out: vec![Probe {
                    text: "Zed?".to_string(),
                    register: Register::Systemic,
                }],
                ..Default::default()
            },
            ProbeSet {
                dir: "a/".to_string(),
                held_out: vec![Probe {
                    text: "Aye?".to_string(),
                    register: Register::Locational,
                }],
                ..Default::default()
            },
        ];
        write_holdout(d.path(), &sets).unwrap();
        let first = read_holdout(d.path()).unwrap();
        write_holdout(d.path(), &sets).unwrap();
        assert_eq!(first, read_holdout(d.path()).unwrap());
        assert_eq!(first[0].dir, "a/", "sorted by directory");
    }

    #[test]
    fn stats_accumulate_across_directories() {
        let mut stats = ProbeStats::default();
        let set = ProbeSet {
            dir: "a/".to_string(),
            probes: vec![Probe {
                text: "What is a ChunkGid?".to_string(),
                register: Register::Conceptual,
            }],
            held_out: vec![Probe {
                text: "What does GidPool allocate?".to_string(),
                register: Register::Conceptual,
            }],
            rejected: vec![("bad".to_string(), Rejection::NotAQuestion)],
        };
        stats.merge(&set, true);
        stats.merge(&set, false);
        assert_eq!(stats.dirs_generated, 2);
        assert_eq!(stats.probes_ingested, 2);
        assert_eq!(stats.held_out, 2);
        assert_eq!(stats.dirs_without_seeds, 1);
        assert_eq!(stats.rejections.get("not_a_question"), Some(&2));
    }
}
