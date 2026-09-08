//! Assembling a directory's [`ProbeSet`] from generated candidates.
//!
//! The split between what is ingested and what is held back is decided here,
//! and it is the reason the retrieval oracle can be trusted. Each register is
//! admitted up to `PROBES_PER_REGISTER + HELD_OUT_PER_REGISTER`; the first six
//! survivors become probes, the next three become test queries that are never
//! ingested. Both populations passed the same gates, so the held-out set is not
//! a set of rejects — it is the same quality of question, differing only in
//! having lost a coin toss for a slot.

use super::filters::{self, Rejection};
use super::idf::TermIndex;
use super::{Probe, ProbeSet, Register, PROBES_PER_REGISTER};
use crate::repo_scan::dir_unit::DirUnit;

/// Admissible candidates per register kept back as test queries.
///
/// Three is enough for a per-register held-out score to mean something across a
/// workspace (415 directories × 3 = ~1,200 trials per register) while leaving
/// the generator's twelve candidates comfortably able to fill six probe slots
/// after rejections.
pub const HELD_OUT_PER_REGISTER: usize = 3;

/// Interleave probes across registers rather than emitting register by register.
///
/// The probes are ingested as consecutive decoded turns on the folder's own
/// conversation, so a later probe answers with the earlier ones in its context.
/// Emitting all six locational probes, then all six mechanistic, and so on would
/// load that accumulation entirely onto whichever register came last — and the
/// last register would be [`Register::Systemic`], the one whose coverage this
/// design most depends on and can least afford to degrade.
///
/// Round-robin spreads the same accumulation evenly, so if it costs anything the
/// per-register scores in [`super::harness`] fall together and say so, instead
/// of one register quietly absorbing all of it.
fn interleave(mut by_register: Vec<Vec<Probe>>) -> Vec<Probe> {
    let mut out = Vec::new();
    let mut round = 0;
    loop {
        let mut emitted = false;
        for register in by_register.iter_mut() {
            if let Some(probe) = register.get(round) {
                out.push(probe.clone());
                emitted = true;
            }
        }
        if !emitted {
            break;
        }
        round += 1;
    }
    by_register.clear();
    out
}

/// Build a directory's probe set from the generator's parsed output.
///
/// `candidates` is what [`super::render::parse`] returned: a register and its
/// questions in the order the model wrote them. Order is respected — a
/// generator's earlier answers are its more confident ones — so admission takes
/// the first survivors rather than re-ranking.
pub fn build(
    dir: &str,
    unit: &DirUnit,
    index: &TermIndex,
    candidates: &[(Register, Vec<String>)],
) -> ProbeSet {
    let keep = PROBES_PER_REGISTER + HELD_OUT_PER_REGISTER;
    let mut per_register: Vec<Vec<Probe>> = Vec::new();
    let mut held_out: Vec<Probe> = Vec::new();
    let mut rejected: Vec<(String, Rejection)> = Vec::new();

    for register in Register::ALL {
        let texts = candidates
            .iter()
            .find(|(r, _)| *r == register)
            .map(|(_, q)| q.as_slice())
            .unwrap_or(&[]);
        let (admitted, refused) = filters::admit_register(texts, register, unit, index, keep);
        rejected.extend(refused);
        let mut admitted = admitted;
        let spare = admitted.split_off(admitted.len().min(PROBES_PER_REGISTER));
        held_out.extend(spare);
        per_register.push(admitted);
    }

    ProbeSet {
        dir: dir.to_string(),
        probes: interleave(per_register),
        held_out,
        rejected,
    }
}

/// Build a probe set from AUTHORED questions — every admissible one, not the
/// first six per register.
///
/// The generated path caps each register because the model is asked for twelve
/// and its later answers are its weaker ones. An authored file has no such
/// gradient: someone who wrote forty questions for a folder meant all forty, and
/// truncating them to six would silently discard the best material this layer
/// can have. Admission still applies in full — a hand-written question can name
/// a path or fail to be a question — and the last few per register are still
/// held out, so the retrieval oracle keeps working.
pub fn build_authored(
    dir: &str,
    unit: &DirUnit,
    index: &TermIndex,
    candidates: &[(Register, Vec<String>)],
) -> ProbeSet {
    let mut per_register: Vec<Vec<Probe>> = Vec::new();
    let mut held_out: Vec<Probe> = Vec::new();
    let mut rejected: Vec<(String, Rejection)> = Vec::new();

    for register in Register::ALL {
        let texts = candidates
            .iter()
            .find(|(r, _)| *r == register)
            .map(|(_, q)| q.as_slice())
            .unwrap_or(&[]);
        // No cap: `keep` is the whole list.
        let (admitted, refused) =
            filters::admit_register(texts, register, unit, index, texts.len().max(1));
        rejected.extend(refused);
        let mut admitted = admitted;
        // Hold back the tail, so a hand-authored folder still contributes to the
        // held-out measure — but never below the count that makes a register
        // complete. Only the surplus above `PROBES_PER_REGISTER` is available to
        // hold, a quarter of the register at a time, capped at
        // `HELD_OUT_PER_REGISTER`: a folder with forty questions spares three, a
        // folder with eight spares two and still ingests six.
        //
        // The threshold clamp is the part that matters to a person. The file's
        // own header names `PROBES_PER_REGISTER` questions as what a complete
        // register holds; someone who writes exactly that many has written what
        // was asked, and a flat quarter withheld one of them — leaving five in
        // the corpus, the folder short of the bar it had just met, and nothing
        // anywhere saying why.
        let spare = admitted.len().saturating_sub(PROBES_PER_REGISTER);
        let hold = (admitted.len() / 4).min(spare).min(HELD_OUT_PER_REGISTER);
        let keep = admitted.len().saturating_sub(hold);
        held_out.extend(admitted.split_off(keep));
        per_register.push(admitted);
    }

    ProbeSet {
        dir: dir.to_string(),
        probes: interleave(per_register),
        held_out,
        rejected,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::repo_scan::dir_unit::build_units;
    use crate::repo_scan::types::{FileEntry, Language, RepoMap};

    /// A workspace with one folder whose symbols are all distinctive, so the
    /// rarity gate never fires and the test measures the split alone.
    fn fixture() -> (DirUnit, TermIndex) {
        let syms: Vec<String> = (0..40).map(|i| format!("widget_{i:02}_thing")).collect();
        let mut map = RepoMap {
            files: vec![FileEntry {
                path: "a/x.rs".to_string(),
                line_count: 1,
                language: Language::Rust,
                size_bytes: 1,
                module_hint: None,
            }],
            ..Default::default()
        };
        map.symbols.insert("a/x.rs".to_string(), syms);
        let d = tempfile::tempdir().unwrap();
        let units = build_units(&map, d.path());
        let index = TermIndex::build(&units, &map);
        (units.into_iter().next().unwrap(), index)
    }

    /// Distinct questions, each naming its own rare term so nothing is refused
    /// for vocabulary and nothing collapses as a duplicate.
    fn candidates_for(register: Register, n: usize, offset: usize) -> (Register, Vec<String>) {
        let qs = (0..n)
            .map(|i| {
                format!(
                    "Where does widget_{:02}_thing get initialised during startup?",
                    i + offset
                )
            })
            .collect();
        (register, qs)
    }

    /// Genuinely distinct vocabulary-free questions.
    ///
    /// They have to differ in subject, not just in a number: the duplicate rule
    /// treats two systemic questions sharing their wording as one probe, which
    /// is correct, so a fixture that varies only a digit would measure the
    /// duplicate filter rather than the probe/held-out split.
    fn systemic(n: usize) -> (Register, Vec<String>) {
        const QUESTIONS: &[&str] = &[
            "How does the project keep long conversations from filling up memory?",
            "What happens when a request arrives while a model is still loading?",
            "Where would I look to understand how work gets spread over a GPU?",
            "How are old results reused instead of being computed twice?",
            "What stops two parts of the program writing to the same place at once?",
            "How does a developer add support for a new kind of hardware?",
            "What decides which pieces stay resident and which get dropped?",
            "How is a failure in one unit prevented from stopping everything else?",
            "What runs first when the program starts up?",
            "How does configuration reach the parts that need it?",
            "Where are numbers measured and reported for later comparison?",
            "What guarantees a restart picks up where the last run stopped?",
        ];
        (
            Register::Systemic,
            QUESTIONS.iter().take(n).map(|s| s.to_string()).collect(),
        )
    }

    #[test]
    fn the_first_six_per_register_are_probes_and_the_next_three_are_held_out() {
        let (unit, index) = fixture();
        let candidates = vec![
            candidates_for(Register::Locational, 12, 0),
            candidates_for(Register::Mechanistic, 12, 10),
            candidates_for(Register::Conceptual, 12, 20),
            systemic(12),
        ];
        let set = build("a/", &unit, &index, &candidates);

        assert!(set.is_complete(), "{:?}", set.probes.len());
        assert_eq!(set.probes.len(), 24);
        assert_eq!(set.held_out.len(), HELD_OUT_PER_REGISTER * 4);
        for register in Register::ALL {
            assert_eq!(set.in_register(register).count(), PROBES_PER_REGISTER);
            assert_eq!(
                set.held_out
                    .iter()
                    .filter(|p| p.register == register)
                    .count(),
                HELD_OUT_PER_REGISTER,
            );
        }
    }

    /// The held-out queries must be genuinely absent from the ingested set, or
    /// the oracle is scoring questions the corpus has already seen.
    #[test]
    fn held_out_queries_never_appear_among_the_probes() {
        let (unit, index) = fixture();
        let candidates = vec![
            candidates_for(Register::Locational, 12, 0),
            candidates_for(Register::Mechanistic, 12, 10),
            candidates_for(Register::Conceptual, 12, 20),
            systemic(12),
        ];
        let set = build("a/", &unit, &index, &candidates);
        for held in &set.held_out {
            assert!(
                !set.probes.iter().any(|p| p.text == held.text),
                "held-out query leaked into the corpus: {}",
                held.text,
            );
        }
    }

    /// Round-robin, so context accumulation across the folder's decoded probe
    /// turns cannot land entirely on the register that happens to be emitted
    /// last.
    #[test]
    fn probes_are_interleaved_across_registers() {
        let (unit, index) = fixture();
        let candidates = vec![
            candidates_for(Register::Locational, 12, 0),
            candidates_for(Register::Mechanistic, 12, 10),
            candidates_for(Register::Conceptual, 12, 20),
            systemic(12),
        ];
        let set = build("a/", &unit, &index, &candidates);
        let first_four: Vec<Register> = set.probes.iter().take(4).map(|p| p.register).collect();
        assert_eq!(first_four, Register::ALL.to_vec(), "{first_four:?}");
    }

    /// A thin generator run yields fewer probes rather than an error, and takes
    /// its held-out set only from what is genuinely spare.
    #[test]
    fn a_short_candidate_list_fills_probes_before_held_out() {
        let (unit, index) = fixture();
        let candidates = vec![candidates_for(Register::Locational, 4, 0)];
        let set = build("a/", &unit, &index, &candidates);
        assert_eq!(set.in_register(Register::Locational).count(), 4);
        assert!(set.held_out.is_empty(), "{:?}", set.held_out);
        assert!(!set.is_complete());
    }

    /// Rejections are carried, not swallowed — a folder that produced nothing
    /// usable has to be able to say why.
    #[test]
    fn rejections_are_reported_with_their_reason() {
        let (unit, index) = fixture();
        let candidates = vec![(
            Register::Locational,
            vec![
                "What does this module do?".to_string(),
                "too short?".to_string(),
                "Where is the completely unknown vocabulary initialised at boot?".to_string(),
            ],
        )];
        let set = build("a/", &unit, &index, &candidates);
        assert!(set.probes.is_empty(), "{:?}", set.probes);
        let tally = set.rejection_tally();
        assert_eq!(tally.get("deixis"), Some(&1), "{tally:?}");
        assert_eq!(tally.get("too_short"), Some(&1), "{tally:?}");
        assert_eq!(tally.get("no_distinctive_term"), Some(&1), "{tally:?}");
    }

    /// **Writing exactly what the file asks for gets exactly that.** The
    /// metadata header names `PROBES_PER_REGISTER` questions as a complete
    /// register; a flat quarter held back withheld one of them, leaving the
    /// folder five in the corpus and short of the bar it had just met.
    #[test]
    fn a_hand_written_register_at_the_threshold_holds_back_nothing() {
        let (unit, index) = fixture();
        let candidates = vec![candidates_for(Register::Locational, PROBES_PER_REGISTER, 0)];
        let set = build_authored("a/", &unit, &index, &candidates);
        assert_eq!(
            set.in_register(Register::Locational).count(),
            PROBES_PER_REGISTER,
            "every hand-written question must reach the corpus",
        );
        assert!(set.held_out.is_empty(), "{:?}", set.held_out);
        assert!(set.is_complete() || set.in_register(Register::Systemic).count() == 0);
    }

    /// Above the threshold the surplus is what gets held, still a quarter at a
    /// time and still capped — and the register keeps its full complement.
    #[test]
    fn a_larger_hand_written_register_holds_back_only_its_surplus() {
        let (unit, index) = fixture();
        for (written, want_held) in [(8usize, 2usize), (12, 3), (40, 3)] {
            let candidates = vec![candidates_for(Register::Locational, written, 0)];
            let set = build_authored("a/", &unit, &index, &candidates);
            assert_eq!(set.held_out.len(), want_held, "{written} written");
            assert_eq!(
                set.in_register(Register::Locational).count(),
                written - want_held,
            );
            assert!(
                set.in_register(Register::Locational).count() >= PROBES_PER_REGISTER,
                "{written} written must never fall under the threshold",
            );
        }
    }

    #[test]
    fn a_register_the_generator_omitted_entirely_is_simply_empty() {
        let (unit, index) = fixture();
        let set = build("a/", &unit, &index, &[systemic(12)]);
        assert_eq!(
            set.in_register(Register::Systemic).count(),
            PROBES_PER_REGISTER
        );
        assert_eq!(set.in_register(Register::Locational).count(), 0);
    }
}
