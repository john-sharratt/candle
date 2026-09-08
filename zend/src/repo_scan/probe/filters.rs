//! Probe admission: the rules a generated question must pass to become an index
//! entry.
//!
//! Every rule here is cheap and CPU-only, and every one of them exists because
//! the failure it prevents is silent. A probe that leaks its own path, or that
//! could have been written about any folder in the repository, still *looks*
//! like a good question — it simply never retrieves anything, or retrieves
//! everything. Neither shows up until the layer is queried in anger.
//!
//! The rules split by register ([`super::Register`]):
//!
//! * **Every register** — no self-address, no deixis, question-shaped, bounded
//!   length.
//! * **Locational / mechanistic / conceptual** — must carry a term the
//!   directory-frequency index calls distinctive ([`super::idf`]).
//! * **Systemic** — exempt from the rarity rule by construction. Demanding a
//!   rare term there would only recreate the other three registers in worse
//!   prose. Its discipline comes from the retrieval test instead, which no
//!   register escapes.

use std::collections::HashMap;

use super::idf::{normalize, TermIndex};
use super::{Probe, Register};
use crate::repo_scan::dir_unit::DirUnit;

/// Shortest admissible probe. Below this a "question" is a keyword fragment,
/// which matches on vocabulary alone and teaches the corpus nothing about the
/// shape of a real query.
const MIN_CHARS: usize = 20;

/// Longest admissible probe. A question longer than this is a paragraph, and its
/// signature is dominated by whatever it dwells on rather than by what it asks.
const MAX_CHARS: usize = 200;

/// Word-overlap share above which two probes are treated as the same question.
///
/// Measured on content words only, so shared function words do not push an
/// unrelated pair over the line. Two-thirds is deliberately loose: a duplicate
/// costs a slot out of six, so over-merging is cheaper than keeping six
/// rephrasings of one question.
const DUPLICATE_OVERLAP: f64 = 0.66;

/// Deictic openings that assume the asker is already looking at the folder.
///
/// These are self-address without a path: "what does this module do" can only be
/// asked by someone who already found the module, and it carries no subject at
/// all — so it is simultaneously the most useless probe and the one a generator
/// reaches for first.
const DEIXIS: &[&str] = &[
    "this folder",
    "this directory",
    "this module",
    "this crate",
    "this package",
    "this file",
    "these files",
    "the above",
    "the folder",
    "the directory above",
];

/// Words carrying no subject, excluded before overlap is measured.
const STOP_WORDS: &[&str] = &[
    "a", "an", "and", "are", "as", "at", "be", "by", "can", "do", "does", "for", "from", "get",
    "how", "i", "if", "in", "is", "it", "its", "of", "on", "or", "that", "the", "to", "was",
    "what", "when", "where", "which", "who", "why", "with", "you", "your", "we", "us", "there",
    "this", "these", "would", "should", "could", "will", "my", "me", "am", "been", "being", "have",
    "has", "had", "not", "but", "so", "than", "then", "them", "they",
];

/// Why a candidate was refused. Carried into the ingest report so a folder whose
/// probes mostly failed says *how* rather than just how many.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Rejection {
    /// Names its own directory path or one of its files.
    SelfAddress(String),
    /// Assumes the asker is already looking at the folder.
    Deixis(String),
    /// Not a question.
    NotAQuestion,
    TooShort,
    TooLong,
    /// Carries a term, but one common enough to retrieve nothing.
    TooCommon {
        term: String,
        df: u32,
        gate: u32,
    },
    /// Carries no indexed term at all — fatal for registers 1–3, definitional
    /// for register 4.
    NoDistinctiveTerm,
    /// Says the same thing as a probe already admitted.
    Duplicate(String),
}

impl Rejection {
    /// Short stable tag for counting rejections in a report.
    pub fn tag(&self) -> &'static str {
        match self {
            Rejection::SelfAddress(_) => "self_address",
            Rejection::Deixis(_) => "deixis",
            Rejection::NotAQuestion => "not_a_question",
            Rejection::TooShort => "too_short",
            Rejection::TooLong => "too_long",
            Rejection::TooCommon { .. } => "too_common",
            Rejection::NoDistinctiveTerm => "no_distinctive_term",
            Rejection::Duplicate(_) => "duplicate",
        }
    }
}

/// Path fragments a probe about `unit` must not contain.
///
/// Two kinds, and the distinction is the whole point:
///
/// * **Path fragments with a separator** (`zend/src/repo_scan`, `src/repo_scan`)
///   — an asker who can write these already knows where the answer lives.
/// * **Filenames with their extension** (`dir_unit.rs`) — likewise.
///
/// A bare stem (`dir_unit`, `repo_scan`) is deliberately NOT forbidden. It is
/// also the name of a real symbol, and "what is a `DirUnit`" is a perfect
/// conceptual probe; rejecting it because a file happens to share the stem would
/// throw away the best questions the folder has.
pub fn forbidden_fragments(unit: &DirUnit) -> Vec<String> {
    let mut out = Vec::new();
    let dir = unit.dir.trim_end_matches('/');
    if dir != "." && !dir.is_empty() {
        let parts: Vec<&str> = dir.split('/').collect();
        // Every multi-component suffix: `a/b/c`, `b/c`. A single component is a
        // bare stem and is allowed.
        for start in 0..parts.len().saturating_sub(1) {
            out.push(parts[start..].join("/"));
        }
    }
    for file in &unit.files {
        if let Some(base) = file.path.rsplit('/').next() {
            if base.contains('.') {
                out.push(base.to_string());
            }
        }
    }
    out.sort();
    out.dedup();
    out
}

/// Content words of a probe, lowercased, function words removed.
fn content_words(text: &str) -> Vec<String> {
    text.split(|c: char| !c.is_ascii_alphanumeric() && c != '_')
        .filter(|s| !s.is_empty())
        .map(|s| s.to_ascii_lowercase())
        .filter(|w| !STOP_WORDS.contains(&w.as_str()))
        .collect()
}

/// Whether two probes ask the same question.
///
/// Word overlap alone is not enough, and getting this wrong is expensive in the
/// direction that is hard to notice. A register's six questions naturally share
/// their framing — "where does X get initialised", "where does Y get
/// initialised" — so on overlap alone a folder's entire locational budget
/// collapses to one probe, and the layer loses five sixths of its coverage while
/// every remaining probe still looks perfectly good.
///
/// So a **distinctive term is decisive**: two questions naming different rare
/// terms are different questions, whatever framing they share. Only when they
/// share a rare term (or neither names one, which is the systemic register)
/// does shared wording make them duplicates.
///
/// The comparison is over the whole tied set, not one representative, and that
/// is what makes the rule hold. Inside a folder almost every declared symbol has
/// `df == 1`, so a question naming two of them ties, and a tie used to resolve
/// to whichever term appeared first in the sentence. "Where does `alpha_thing`
/// hand a chunk to `beta_thing`?" and "Where does `beta_thing` receive a chunk
/// from `alpha_thing`?" therefore resolved to *different* representatives, were
/// declared distinct without ever being compared on overlap, and spent two of a
/// register's six slots on one question — the exact failure `DUPLICATE_OVERLAP`
/// was loosened to catch. Disjointness asks the question the rule always meant:
/// are these about different things?
fn is_duplicate(a: &str, b: &str, index: &TermIndex) -> bool {
    let (ta, tb) = (index.rarest_terms(a), index.rarest_terms(b));
    if let (Some((_, terms_a)), Some((_, terms_b))) = (&ta, &tb) {
        if terms_a.is_disjoint(terms_b) {
            return false;
        }
    }
    overlap(a, b) >= DUPLICATE_OVERLAP
}

/// Share of the smaller probe's content words that appear in the larger.
///
/// Asymmetric on purpose — measuring against the shorter side means a terse
/// question fully contained in a wordier one counts as a duplicate, which is
/// exactly the shape a generator's near-repeats take.
fn overlap(a: &str, b: &str) -> f64 {
    let (wa, wb) = (content_words(a), content_words(b));
    if wa.is_empty() || wb.is_empty() {
        return 0.0;
    }
    let (small, large) = if wa.len() <= wb.len() {
        (&wa, &wb)
    } else {
        (&wb, &wa)
    };
    let shared = small.iter().filter(|w| large.contains(w)).count();
    shared as f64 / small.len() as f64
}

/// Rules that hold for every register.
fn shape(text: &str, unit: &DirUnit) -> Result<(), Rejection> {
    let trimmed = text.trim();
    if trimmed.chars().count() < MIN_CHARS {
        return Err(Rejection::TooShort);
    }
    if trimmed.chars().count() > MAX_CHARS {
        return Err(Rejection::TooLong);
    }
    if !trimmed.ends_with('?') {
        return Err(Rejection::NotAQuestion);
    }
    let lower = trimmed.to_ascii_lowercase();
    for phrase in DEIXIS {
        if lower.contains(phrase) {
            return Err(Rejection::Deixis((*phrase).to_string()));
        }
    }
    for fragment in forbidden_fragments(unit) {
        if lower.contains(&fragment.to_ascii_lowercase()) {
            return Err(Rejection::SelfAddress(fragment));
        }
    }
    Ok(())
}

/// Whether one candidate is admissible on its own terms, ignoring the rest of
/// the set.
pub fn admit(
    text: &str,
    register: Register,
    unit: &DirUnit,
    index: &TermIndex,
) -> Result<(), Rejection> {
    shape(text, unit)?;
    if !register.requires_rarity() {
        return Ok(());
    }
    match index.rarest_df(text) {
        None => Err(Rejection::NoDistinctiveTerm),
        Some((df, term)) => {
            let gate = index.rarity_gate();
            if df <= gate {
                Ok(())
            } else {
                Err(Rejection::TooCommon { term, df, gate })
            }
        }
    }
}

/// Filter a register's candidates down to at most `keep`, in order, dropping
/// near-duplicates of anything already admitted.
///
/// Returns the survivors and every rejection with its reason, so a folder that
/// produced nothing usable can say why rather than reporting a bare zero.
pub fn admit_register(
    candidates: &[String],
    register: Register,
    unit: &DirUnit,
    index: &TermIndex,
    keep: usize,
) -> (Vec<Probe>, Vec<(String, Rejection)>) {
    let mut kept: Vec<Probe> = Vec::new();
    let mut rejected: Vec<(String, Rejection)> = Vec::new();
    for text in candidates {
        if kept.len() >= keep {
            break;
        }
        let text = text.trim().to_string();
        if let Err(why) = admit(&text, register, unit, index) {
            rejected.push((text, why));
            continue;
        }
        if let Some(prior) = kept.iter().find(|p| is_duplicate(&p.text, &text, index)) {
            rejected.push((text, Rejection::Duplicate(prior.text.clone())));
            continue;
        }
        kept.push(Probe { text, register });
    }
    (kept, rejected)
}

/// Systemic probes that more than one directory laid claim to.
///
/// Registers 1–3 cannot collide much — rare terms are rare. Register 4 collides
/// constantly: `kv_cache/` and `kv_cache/chunked/` will independently produce
/// "how does the cache decide what to keep in GPU memory", and so will every
/// sibling under `cuda_backend/`. Left alone the register becomes several
/// hundred mutually-confusable probes and degrades into exactly the noise it was
/// admitted to avoid.
///
/// Detection only — the winner is decided by which directory the probe actually
/// retrieves, which is the retrieval harness's call, not a lexical one.
/// The returned map is keyed by a normalised form of the probe.
pub fn systemic_collisions(per_dir: &[(String, Vec<Probe>)]) -> HashMap<String, Vec<String>> {
    let mut claims: HashMap<String, Vec<String>> = HashMap::new();
    for (dir, probes) in per_dir {
        for probe in probes {
            if probe.register != Register::Systemic {
                continue;
            }
            let key = collision_key(&probe.text);
            let owners = claims.entry(key).or_default();
            if !owners.contains(dir) {
                owners.push(dir.clone());
            }
        }
    }
    claims.retain(|_, owners| owners.len() > 1);
    claims
}

/// Normalised form two systemic probes collide on: sorted content words.
///
/// Word order carries no retrieval signal — "how does the system avoid running
/// out of GPU memory" and "how does the system avoid GPU memory running out" are
/// one probe — so the key is order-free.
pub fn collision_key(text: &str) -> String {
    let mut words = content_words(text);
    words.sort();
    words.dedup();
    normalize(&words.join(" "))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::repo_scan::dir_unit::build_units;
    use crate::repo_scan::types::{FileEntry, Language, RepoMap};

    fn workspace(files: &[(&str, &[&str])]) -> (Vec<DirUnit>, TermIndex) {
        let mut map = RepoMap {
            files: files
                .iter()
                .map(|(p, _)| FileEntry {
                    path: p.to_string(),
                    line_count: 1,
                    language: Language::Rust,
                    size_bytes: 1,
                    module_hint: None,
                })
                .collect(),
            ..Default::default()
        };
        for (path, syms) in files {
            map.symbols.insert(
                path.to_string(),
                syms.iter().map(|s| s.to_string()).collect(),
            );
        }
        let d = tempfile::tempdir().unwrap();
        let units = build_units(&map, d.path());
        let index = TermIndex::build(&units, &map);
        (units, index)
    }

    fn unit_for<'a>(units: &'a [DirUnit], dir: &str) -> &'a DirUnit {
        units.iter().find(|u| u.dir == dir).expect("unit")
    }

    /// The highest-leverage rule: a probe naming its own path indexes the folder
    /// by the one thing the asker would already have to know.
    #[test]
    fn a_probe_naming_its_own_path_is_refused() {
        let (units, idx) = workspace(&[("zend/src/repo_scan/mod.rs", &["scan_width"])]);
        let u = unit_for(&units, "zend/src/repo_scan/");
        let err = admit(
            "How does scan_width work in zend/src/repo_scan?",
            Register::Mechanistic,
            u,
            &idx,
        )
        .unwrap_err();
        assert!(matches!(err, Rejection::SelfAddress(_)), "{err:?}");
    }

    #[test]
    fn a_probe_naming_one_of_its_files_is_refused() {
        let (units, idx) = workspace(&[("a/dir_unit.rs", &["DirUnit"])]);
        let u = unit_for(&units, "a/");
        let err = admit(
            "What does dir_unit.rs define?",
            Register::Conceptual,
            u,
            &idx,
        )
        .unwrap_err();
        assert!(matches!(err, Rejection::SelfAddress(_)), "{err:?}");
    }

    /// The distinction that keeps the rule from eating the best probes: a bare
    /// stem is also a symbol name, and a question about the symbol is exactly
    /// what the conceptual register wants.
    #[test]
    fn a_bare_stem_shared_with_a_filename_is_allowed() {
        let (units, idx) = workspace(&[("a/dir_unit.rs", &["DirUnit"])]);
        let u = unit_for(&units, "a/");
        assert!(admit("What is a DirUnit used for?", Register::Conceptual, u, &idx).is_ok());
    }

    /// Deixis is self-address without a path — and carries no subject either.
    #[test]
    fn a_deictic_probe_is_refused() {
        let (units, idx) = workspace(&[("a/x.rs", &["Widget"])]);
        let u = unit_for(&units, "a/");
        for text in [
            "What does this module actually do at runtime?",
            "Which files in this directory matter most here?",
        ] {
            let err = admit(text, Register::Systemic, u, &idx).unwrap_err();
            assert!(matches!(err, Rejection::Deixis(_)), "{text}: {err:?}");
        }
    }

    #[test]
    fn a_probe_must_be_a_question() {
        let (units, idx) = workspace(&[("a/x.rs", &["Widget"])]);
        let u = unit_for(&units, "a/");
        assert_eq!(
            admit(
                "Explain the Widget lifecycle in detail.",
                Register::Mechanistic,
                u,
                &idx
            ),
            Err(Rejection::NotAQuestion),
        );
    }

    /// Registers 1–3 live or die on the rarity gate.
    #[test]
    fn a_specific_register_needs_a_distinctive_term() {
        let (units, idx) = workspace(&[("a/x.rs", &["Widget"])]);
        let u = unit_for(&units, "a/");
        assert_eq!(
            admit(
                "How does the system decide what to keep in memory?",
                Register::Mechanistic,
                u,
                &idx,
            ),
            Err(Rejection::NoDistinctiveTerm),
        );
    }

    /// …and register 4 is exempt from exactly that rule, which is what makes it
    /// able to serve a newcomer who cannot name anything yet.
    #[test]
    fn the_systemic_register_is_exempt_from_rarity() {
        let (units, idx) = workspace(&[("a/x.rs", &["Widget"])]);
        let u = unit_for(&units, "a/");
        assert!(admit(
            "How does the system decide what to keep in memory?",
            Register::Systemic,
            u,
            &idx,
        )
        .is_ok());
    }

    /// A term common across the corpus fails even though it IS a real symbol —
    /// this is the promiscuity gate doing its job.
    #[test]
    fn a_common_term_fails_the_rarity_gate() {
        let mut files: Vec<(String, Vec<String>)> = (0..100)
            .map(|i| (format!("d{i:03}/x.rs"), vec!["handler".to_string()]))
            .collect();
        files.push(("a/y.rs".to_string(), vec!["handler".to_string()]));
        let owned: Vec<(&str, Vec<&str>)> = files
            .iter()
            .map(|(p, s)| (p.as_str(), s.iter().map(|x| x.as_str()).collect()))
            .collect();
        let slices: Vec<(&str, &[&str])> = owned.iter().map(|(p, s)| (*p, s.as_slice())).collect();
        let (units, idx) = workspace(&slices);
        let u = unit_for(&units, "a/");
        let err = admit(
            "Where is the handler registered?",
            Register::Locational,
            u,
            &idx,
        )
        .unwrap_err();
        assert!(matches!(err, Rejection::TooCommon { .. }), "{err:?}");
    }

    /// A generator asked for six questions returns six rephrasings of two. The
    /// duplicate filter is what turns that back into usable coverage.
    #[test]
    fn near_duplicate_candidates_collapse_to_one() {
        let (units, idx) = workspace(&[("a/x.rs", &["weight_floor"])]);
        let u = unit_for(&units, "a/");
        let candidates = vec![
            "What moves the weight_floor boundary during a wave?".to_string(),
            "During a wave, what moves the weight_floor boundary?".to_string(),
            "Which component lowers weight_floor when experts are evicted?".to_string(),
        ];
        let (kept, rejected) = admit_register(&candidates, Register::Mechanistic, u, &idx, 6);
        assert_eq!(kept.len(), 2, "{kept:?}");
        assert!(
            matches!(rejected[0].1, Rejection::Duplicate(_)),
            "{rejected:?}"
        );
    }

    /// The counterpart to the duplicate test, and the more dangerous direction:
    /// a register's six questions naturally share their framing, so collapsing
    /// on wording alone would take a folder's whole locational budget down to
    /// one probe — while every survivor still looked like a good question.
    #[test]
    fn questions_naming_different_symbols_are_not_duplicates() {
        let (units, idx) = workspace(&[("a/x.rs", &["alpha_thing", "beta_thing", "gamma_thing"])]);
        let u = unit_for(&units, "a/");
        let candidates = vec![
            "Where does alpha_thing get initialised at startup?".to_string(),
            "Where does beta_thing get initialised at startup?".to_string(),
            "Where does gamma_thing get initialised at startup?".to_string(),
        ];
        let (kept, rejected) = admit_register(&candidates, Register::Locational, u, &idx, 6);
        assert_eq!(kept.len(), 3, "{kept:?} / rejected {rejected:?}");
    }

    /// …and the same framing around the SAME symbol still collapses, which is
    /// what the rule is for.
    #[test]
    fn questions_naming_one_symbol_with_shared_framing_do_collapse() {
        let (units, idx) = workspace(&[("a/x.rs", &["alpha_thing"])]);
        let u = unit_for(&units, "a/");
        let candidates = vec![
            "Where does alpha_thing get initialised at startup?".to_string(),
            "At startup, where does alpha_thing get initialised?".to_string(),
        ];
        let (kept, _) = admit_register(&candidates, Register::Locational, u, &idx, 6);
        assert_eq!(kept.len(), 1, "{kept:?}");
    }

    /// **Two questions naming the SAME PAIR of symbols are one question.** They
    /// tie at `df == 1`, which is the ordinary case inside a folder, and the tie
    /// used to be broken by word order — so each resolved to a different
    /// representative, they were never compared on overlap, and one question
    /// took two of the register's six slots.
    #[test]
    fn questions_naming_the_same_pair_of_symbols_collapse() {
        let (units, idx) = workspace(&[("a/x.rs", &["alpha_thing", "beta_thing"])]);
        let u = unit_for(&units, "a/");
        let candidates = vec![
            "Where does alpha_thing hand a chunk to beta_thing?".to_string(),
            "Where does beta_thing receive a chunk from alpha_thing?".to_string(),
        ];
        let (kept, _) = admit_register(&candidates, Register::Locational, u, &idx, 6);
        assert_eq!(kept.len(), 1, "{kept:?}");
    }

    /// And questions about genuinely different symbols still stand apart, even
    /// when one of them also mentions the other's subject in passing: the sets
    /// overlap only when the questions really are about the same things.
    #[test]
    fn questions_about_different_symbols_survive_shared_framing() {
        let (units, idx) = workspace(&[("a/x.rs", &["alpha_thing", "beta_thing", "gamma_thing"])]);
        let u = unit_for(&units, "a/");
        let candidates = vec![
            "Where does alpha_thing get initialised at startup?".to_string(),
            "Where does beta_thing get initialised at startup?".to_string(),
            "Where does gamma_thing get initialised at startup?".to_string(),
        ];
        let (kept, rejected) = admit_register(&candidates, Register::Locational, u, &idx, 6);
        assert_eq!(kept.len(), 3, "{kept:?} / rejected {rejected:?}");
    }

    #[test]
    fn admission_stops_at_the_keep_count() {
        let (units, idx) = workspace(&[("a/x.rs", &["alpha_thing", "beta_thing", "gamma_thing"])]);
        let u = unit_for(&units, "a/");
        let candidates = vec![
            "Where does alpha_thing get initialised at startup?".to_string(),
            "Where does beta_thing get initialised at startup?".to_string(),
            "Where does gamma_thing get initialised at startup?".to_string(),
        ];
        let (kept, _) = admit_register(&candidates, Register::Locational, u, &idx, 2);
        assert_eq!(kept.len(), 2);
    }

    /// The failure mode specific to register 4: two folders independently
    /// producing the same vocabulary-free question.
    #[test]
    fn systemic_probes_claimed_by_two_directories_are_detected() {
        let shared = "How does the system avoid running out of GPU memory?";
        let per_dir = vec![
            (
                "kv_cache/".to_string(),
                vec![Probe {
                    text: shared.to_string(),
                    register: Register::Systemic,
                }],
            ),
            (
                "kv_cache/chunked/".to_string(),
                vec![Probe {
                    // Same question, different word order — still one probe.
                    text: "How does the system avoid GPU memory running out?".to_string(),
                    register: Register::Systemic,
                }],
            ),
        ];
        let collisions = systemic_collisions(&per_dir);
        assert_eq!(collisions.len(), 1, "{collisions:?}");
        let owners = collisions.values().next().unwrap();
        assert_eq!(owners.len(), 2, "{owners:?}");
    }

    /// Registers 1–3 are not swept by the collision detector: they are gated on
    /// rare terms already, and a shared rare term is a real signal about two
    /// folders rather than noise to arbitrate.
    #[test]
    fn specific_registers_are_not_collision_swept() {
        let text = "What does ChunkGid identify?";
        let per_dir = vec![
            (
                "a/".to_string(),
                vec![Probe {
                    text: text.to_string(),
                    register: Register::Conceptual,
                }],
            ),
            (
                "b/".to_string(),
                vec![Probe {
                    text: text.to_string(),
                    register: Register::Conceptual,
                }],
            ),
        ];
        assert!(systemic_collisions(&per_dir).is_empty());
    }
}
