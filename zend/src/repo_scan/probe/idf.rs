//! Directory-frequency index over declared symbols — the specificity engine.
//!
//! One **document per directory**, its terms the symbols declared by the files
//! directly inside it ([`super::symbols`]). A term occurring in one directory is
//! a scalpel; one occurring in two hundred is noise. That single statistic does
//! three jobs:
//!
//! 1. **Seeds generation** — [`TermIndex::distinctive`] hands the model the
//!    terms that characterise a folder, so specificity is an *input* to probe
//!    writing rather than something hoped for afterwards.
//! 2. **Gates registers 1–3** — [`TermIndex::rarest_df`] finds a probe's rarest
//!    term, and [`TermIndex::rarity_gate`] says whether it is rare enough. A
//!    question whose most distinctive word occurs in half the repository cannot
//!    retrieve anything.
//! 3. **Identifies register 4** — a probe containing no indexed term at all is
//!    vocabulary-free by construction, which is exactly what the systemic
//!    register is for. The absence is the signal, so it is not a failure here.
//!
//! **Terms are matched across word boundaries.** A symbol is written
//! `region_pool` in code and "region pool" in a question, so both index keys and
//! probe lookups normalise to lowercase alphanumerics with separators dropped,
//! and a lookup joins runs of up to [`MAX_PHRASE_WORDS`] adjacent words before
//! testing. Without that every probe phrased in English would read as
//! vocabulary-free and the rarity gate would never fire.

use std::collections::{BTreeSet, HashMap};

use super::symbols;
use crate::repo_scan::dir_unit::DirUnit;
use crate::repo_scan::types::RepoMap;

/// Adjacent words joined when looking a phrase up. `region pool` reaches
/// `region_pool` at two; `binary directional provenance` reaches
/// `BinaryDirectionalProvenance` at three. Beyond that the joins are noise —
/// no identifier in practice spells out four English words.
const MAX_PHRASE_WORDS: usize = 3;

/// Directory share above which a term is not distinctive, in percent.
///
/// Two percent of a 415-directory workspace is eight directories: a term at
/// that frequency still narrows retrieval to a handful of candidates, which the
/// belief scan can rank, while one at fifty directories cannot narrow anything.
/// Expressed as a share rather than a count so the gate holds on a workspace of
/// any size.
const RARITY_MAX_DF_PCT: usize = 2;

/// Floor under [`RARITY_MAX_DF_PCT`]. On a small workspace two percent rounds to
/// zero, which would reject every probe including perfect ones.
const RARITY_MIN_DF: u32 = 3;

/// Seed-ranking weight for a single-word identifier with no internal boundary.
///
/// Held well below one so compound names win every contested slot, but non-zero
/// so a folder whose whole vocabulary is single words (`marlin`, `flash`,
/// `cute`) still gets seeds instead of falling back to systemic probes alone.
const STRUCTURELESS_WEIGHT: f64 = 0.3;

/// Normalised lookup key: lowercase ASCII alphanumerics, everything else
/// dropped. Collapses `region_pool`, `RegionPool` and `region pool` onto one
/// key so a question written in English reaches a symbol written in code.
pub fn normalize(s: &str) -> String {
    s.chars()
        .filter(|c| c.is_ascii_alphanumeric())
        .flat_map(|c| c.to_lowercase())
        .collect()
}

/// Stable order-scrambling key for breaking score ties (FNV-1a).
///
/// Deterministic across runs and machines — it must be, because the seed list
/// feeds a content hash — while bearing no relation to spelling, which is the
/// whole point.
fn tie_break(key: &str) -> u64 {
    let mut hash: u64 = 0xcbf2_9ce4_8422_2325;
    for byte in key.as_bytes() {
        hash ^= *byte as u64;
        hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
    }
    hash
}

/// Words of a text, in order, for phrase assembly.
fn words(text: &str) -> Vec<&str> {
    text.split(|c: char| !c.is_ascii_alphanumeric())
        .filter(|s| !s.is_empty())
        .collect()
}

/// Directory-frequency index over the whole workspace.
#[derive(Debug, Default)]
pub struct TermIndex {
    /// Directories declaring each key.
    df: HashMap<String, u32>,
    /// Per-directory term counts, keyed by `DirUnit::dir`.
    per_dir: HashMap<String, HashMap<String, u32>>,
    /// The form to show a reader for each key — the symbol as it is actually
    /// written in the source, which is what a seeded question should use.
    display: HashMap<String, String>,
    n_dirs: usize,
}

impl TermIndex {
    /// Build the index from the walked units and the walk's symbol side-table.
    ///
    /// Terms come from files **directly inside** a directory, not its subtree: a
    /// unit describes its own files, and crediting a parent with everything
    /// below it would make every ancestor look like the owner of its
    /// descendants' vocabulary — with the workspace root owning all of it.
    pub fn build(units: &[DirUnit], map: &RepoMap) -> Self {
        let mut index = Self {
            n_dirs: units.len(),
            ..Default::default()
        };
        for unit in units {
            let mut counts: HashMap<String, u32> = HashMap::new();
            for file in &unit.files {
                let Some(syms) = map.symbols.get(&file.path) else {
                    continue;
                };
                for sym in syms {
                    let key = normalize(sym);
                    if key.is_empty() {
                        continue;
                    }
                    *counts.entry(key.clone()).or_insert(0) += 1;
                    index.display.entry(key).or_insert_with(|| sym.clone());
                }
            }
            for key in counts.keys() {
                *index.df.entry(key.clone()).or_insert(0) += 1;
            }
            index.per_dir.insert(unit.dir.clone(), counts);
        }
        index
    }

    /// Directories in the index — the document count behind every frequency.
    pub fn n_dirs(&self) -> usize {
        self.n_dirs
    }

    /// Directories declaring `key`; zero when the term is unknown.
    pub fn df(&self, key: &str) -> u32 {
        self.df.get(key).copied().unwrap_or(0)
    }

    /// Highest document frequency a term may carry and still count as
    /// distinctive.
    pub fn rarity_gate(&self) -> u32 {
        ((self.n_dirs * RARITY_MAX_DF_PCT) / 100).max(RARITY_MIN_DF as usize) as u32
    }

    /// The `k` terms that most characterise `dir`, as they are written in the
    /// source, most distinctive first.
    ///
    /// Ranked by `tf × ln(N / df)` — the standard weighting, which is what keeps
    /// a folder's *central* rare concept ahead of a rare name mentioned once. A
    /// term already above the rarity gate is dropped outright: it cannot pass
    /// the filter later, so seeding a question with it would only waste a slot.
    pub fn distinctive(&self, dir: &str, k: usize) -> Vec<&str> {
        let Some(counts) = self.per_dir.get(dir) else {
            return Vec::new();
        };
        let gate = self.rarity_gate();
        let n = self.n_dirs.max(1) as f64;
        let mut scored: Vec<(f64, &str, &str)> = counts
            .iter()
            .filter_map(|(key, tf)| {
                let df = self.df(key);
                if df == 0 || df > gate {
                    return None;
                }
                let shown = self.display.get(key)?.as_str();
                // Compound identifiers are domain vocabulary; a bare lowercase
                // word usually is not. `region_pool` and `ChunkGid` are what a
                // question can be built around, while `marlin` or `access` may
                // be either — so structure is a weighting, not a gate, and a
                // genuinely distinctive single word still surfaces when the
                // folder has nothing better.
                let structure = if symbols::is_structureless(shown) {
                    STRUCTURELESS_WEIGHT
                } else {
                    1.0
                };
                let score = *tf as f64 * (n / df as f64).ln() * structure;
                Some((score, key.as_str(), shown))
            })
            .collect();
        // Score descending, then by a stable HASH of the key — never by the key
        // itself.
        //
        // Ties are the common case, not the exception: most terms occur once in
        // one directory, so `tf × ln(N/df)` is identical across a long tail and
        // the tie-break decides what the generator actually sees. An alphabetical
        // tie-break then hands over the front of the alphabet and nothing else.
        // Measured on a live run, all 28 seeds for one folder were
        // `add_collection`, `add_section`, `adopt_turn`, `adaptive`, `aggregate`,
        // `all_section_ids`, `AnchorConfig`, `AnchorMember`, `any_layer_disabled`,
        // `allocation`, `BranchPrefixIds` — every probe for that directory was
        // about a term starting with `a`.
        //
        // Hashing keeps the order deterministic (the probe content hash depends
        // on it) while spreading the tail across the folder's whole vocabulary.
        scored.sort_by(|a, b| {
            b.0.partial_cmp(&a.0)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| tie_break(a.1).cmp(&tie_break(b.1)))
        });
        scored
            .into_iter()
            .take(k)
            .map(|(_, _, shown)| shown)
            .collect()
    }

    /// The lowest document frequency among the indexed terms `text` mentions,
    /// with **every** term that carried it.
    ///
    /// `None` means the text mentions no indexed term at all — the systemic
    /// register's defining property, not an error.
    ///
    /// The set matters because ties are the normal case, not the exception:
    /// inside one folder nearly every declared symbol has `df == 1`, so a
    /// question naming two of them has two equally rare terms and picking one is
    /// picking arbitrarily. Returning the whole tied set lets a caller ask
    /// whether two questions are *about* the same things rather than whether an
    /// arbitrary pick happened to agree — see [`super::filters`].
    pub fn rarest_terms(&self, text: &str) -> Option<(u32, BTreeSet<String>)> {
        let words = words(text);
        let mut best: Option<(u32, BTreeSet<String>)> = None;
        for start in 0..words.len() {
            let end = (start + MAX_PHRASE_WORDS).min(words.len());
            for stop in start + 1..=end {
                let key = normalize(&words[start..stop].join(""));
                let df = self.df(&key);
                if df == 0 {
                    continue;
                }
                let shown = self
                    .display
                    .get(&key)
                    .cloned()
                    .unwrap_or_else(|| key.clone());
                match &mut best {
                    Some((b, set)) if df == *b => {
                        set.insert(shown);
                    }
                    Some((b, _)) if df > *b => {}
                    _ => best = Some((df, BTreeSet::from([shown]))),
                }
            }
        }
        best
    }

    /// The lowest document frequency among the indexed terms `text` mentions,
    /// with one of the terms that carried it.
    ///
    /// Only the frequency is well-defined when several terms tie; callers that
    /// care which terms those are want [`Self::rarest_terms`].
    pub fn rarest_df(&self, text: &str) -> Option<(u32, String)> {
        self.rarest_terms(text)
            .and_then(|(df, set)| set.into_iter().next().map(|t| (df, t)))
    }

    /// Whether `text` carries a term distinctive enough to retrieve on.
    pub fn is_specific(&self, text: &str) -> bool {
        self.rarest_df(text)
            .is_some_and(|(df, _)| df <= self.rarity_gate())
    }
}

/// Extract every walked file's declared symbols into the map's side-table.
///
/// Called by the walk, on bytes it has already read, so this adds a scan rather
/// than a second pass over the corpus.
pub fn record_symbols(
    map: &mut RepoMap,
    path: &str,
    body: &str,
    language: crate::repo_scan::types::Language,
) {
    let syms = symbols::extract(body, language);
    if !syms.is_empty() {
        map.symbols.insert(path.to_string(), syms);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::repo_scan::dir_unit::build_units;
    use crate::repo_scan::types::{FileEntry, Language, RepoMap};

    fn entry(path: &str) -> FileEntry {
        FileEntry {
            path: path.to_string(),
            line_count: 1,
            language: Language::Rust,
            size_bytes: 1,
            module_hint: None,
        }
    }

    /// A workspace of `(path, [symbols])`, walked into units plus an index.
    fn index_of(files: &[(&str, &[&str])]) -> (Vec<DirUnit>, TermIndex) {
        let mut map = RepoMap {
            files: files.iter().map(|(p, _)| entry(p)).collect(),
            ..Default::default()
        };
        for (path, syms) in files {
            map.symbols.insert(
                path.to_string(),
                syms.iter().map(|s| s.to_string()).collect(),
            );
        }
        let dir = tempfile::tempdir().unwrap();
        let units = build_units(&map, dir.path());
        let index = TermIndex::build(&units, &map);
        (units, index)
    }

    #[test]
    fn document_frequency_counts_directories_not_files() {
        let (_, idx) = index_of(&[
            ("a/one.rs", &["Widget"]),
            ("a/two.rs", &["Widget"]),
            ("b/three.rs", &["Widget"]),
        ]);
        // Three files, two directories.
        assert_eq!(idx.df("widget"), 2);
    }

    /// The normalisation is what lets a question written in English reach a
    /// symbol written in code. Without it every probe would look
    /// vocabulary-free.
    #[test]
    fn a_phrase_in_english_reaches_a_snake_case_symbol() {
        let (_, idx) = index_of(&[("a/x.rs", &["region_pool"])]);
        assert_eq!(
            idx.rarest_df("how does the region pool work"),
            Some((1, "region_pool".to_string()))
        );
    }

    #[test]
    fn a_phrase_reaches_a_camel_case_symbol() {
        let (_, idx) = index_of(&[("a/x.rs", &["ChunkGid"])]);
        assert!(idx.rarest_df("what is a chunk gid").is_some());
        assert!(idx.rarest_df("what is a ChunkGid").is_some());
    }

    /// Three words is the ceiling; a four-word join is noise, and admitting it
    /// would let arbitrary English fragments collide with identifiers.
    #[test]
    fn phrase_joins_stop_at_three_words() {
        let (_, idx) = index_of(&[("a/x.rs", &["one_two_three_four"])]);
        assert_eq!(idx.rarest_df("one two three four"), None);
        let (_, idx3) = index_of(&[("a/x.rs", &["one_two_three"])]);
        assert!(idx3.rarest_df("one two three").is_some());
    }

    /// A probe carrying no indexed term is the systemic register's defining
    /// property — reported as absence, never as a failure.
    #[test]
    fn a_vocabulary_free_probe_reports_no_term() {
        let (_, idx) = index_of(&[("a/x.rs", &["ChunkGid"])]);
        assert_eq!(
            idx.rarest_df("how does the system avoid running out of memory"),
            None
        );
        assert!(!idx.is_specific("how does the system avoid running out of memory"));
    }

    /// The gate is what makes a promiscuous term unusable. `new` in every
    /// directory must fail; a name in one must pass.
    #[test]
    fn the_rarity_gate_rejects_a_promiscuous_term_and_admits_a_rare_one() {
        let mut files: Vec<(String, Vec<String>)> = (0..100)
            .map(|i| (format!("d{i:03}/x.rs"), vec!["new_thing".to_string()]))
            .collect();
        files.push(("special/y.rs".to_string(), vec!["ChunkGid".to_string()]));
        let refs: Vec<(&str, Vec<&str>)> = files
            .iter()
            .map(|(p, s)| (p.as_str(), s.iter().map(|x| x.as_str()).collect()))
            .collect();
        let as_slices: Vec<(&str, &[&str])> =
            refs.iter().map(|(p, s)| (*p, s.as_slice())).collect();
        let (_, idx) = index_of(&as_slices);

        assert_eq!(idx.n_dirs(), 101);
        assert!(
            !idx.is_specific("where is new_thing"),
            "df=100 must fail the gate"
        );
        assert!(idx.is_specific("what is a ChunkGid"), "df=1 must pass");
    }

    /// Seeds must be usable: a term already above the gate can never pass the
    /// filter, so offering it to the generator would waste a question slot.
    #[test]
    fn seeds_never_include_a_term_that_would_fail_the_gate() {
        let mut files: Vec<(String, Vec<String>)> = (0..100)
            .map(|i| (format!("d{i:03}/x.rs"), vec!["common".to_string()]))
            .collect();
        files[0].1.push("rare_local_thing".to_string());
        let refs: Vec<(&str, Vec<&str>)> = files
            .iter()
            .map(|(p, s)| (p.as_str(), s.iter().map(|x| x.as_str()).collect()))
            .collect();
        let as_slices: Vec<(&str, &[&str])> =
            refs.iter().map(|(p, s)| (*p, s.as_slice())).collect();
        let (_, idx) = index_of(&as_slices);

        let seeds = idx.distinctive("d000/", 10);
        assert!(seeds.contains(&"rare_local_thing"), "{seeds:?}");
        assert!(!seeds.contains(&"common"), "{seeds:?}");
    }

    /// The seed list feeds a content hash, so ties must break deterministically
    /// rather than by hash-map order.
    #[test]
    fn seed_ordering_is_deterministic_across_builds() {
        let files: &[(&str, &[&str])] = &[(
            "a/x.rs",
            &["alpha_one", "beta_two", "gamma_three", "delta_four"],
        )];
        let (_, a) = index_of(files);
        let (_, b) = index_of(files);
        assert_eq!(a.distinctive("a/", 4), b.distinctive("a/", 4));
    }

    /// …and it must NOT break alphabetically. Ties are the common case — most
    /// terms occur once in one directory — so an alphabetical tie-break hands the
    /// generator the front of the alphabet and nothing else. Observed live: all
    /// 28 seeds for one folder began with `a`.
    #[test]
    fn seed_ties_do_not_break_alphabetically() {
        // Twenty-six single-occurrence terms, one per letter: every score is
        // identical, so the tie-break alone decides the order.
        let owned: Vec<String> = ('a'..='z')
            .map(|c| format!("{c}{c}{c}_distinct_term"))
            .collect();
        let refs: Vec<&str> = owned.iter().map(|s| s.as_str()).collect();
        let (_, idx) = index_of(&[("a/x.rs", refs.as_slice())]);

        let top: Vec<&str> = idx.distinctive("a/", 8);
        let alphabetical: Vec<&str> = {
            let mut all = refs.clone();
            all.sort();
            all.into_iter().take(8).collect()
        };
        assert_eq!(top.len(), 8);
        assert_ne!(top, alphabetical, "the tie-break is still alphabetical");
    }

    /// A parent must not inherit its children's vocabulary, or every ancestor
    /// looks like the owner of everything beneath it — and the workspace root
    /// owns the entire repository, which is the promiscuity this index exists
    /// to prevent.
    #[test]
    fn a_parent_does_not_absorb_its_subtree_terms() {
        let (_, idx) = index_of(&[
            ("a/top.rs", &["ParentThing"]),
            ("a/b/deep.rs", &["ChildThing"]),
        ]);
        let parent = idx.distinctive("a/", 10);
        assert!(parent.contains(&"ParentThing"), "{parent:?}");
        assert!(!parent.contains(&"ChildThing"), "{parent:?}");
    }

    /// The rarest term is the one that decides the gate — a probe mixing a
    /// common word with a rare one is specific.
    #[test]
    fn the_rarest_term_decides() {
        let mut files: Vec<(String, Vec<String>)> = (0..50)
            .map(|i| (format!("d{i:03}/x.rs"), vec!["handler".to_string()]))
            .collect();
        files[0].1.push("weight_floor".to_string());
        let refs: Vec<(&str, Vec<&str>)> = files
            .iter()
            .map(|(p, s)| (p.as_str(), s.iter().map(|x| x.as_str()).collect()))
            .collect();
        let as_slices: Vec<(&str, &[&str])> =
            refs.iter().map(|(p, s)| (*p, s.as_slice())).collect();
        let (_, idx) = index_of(&as_slices);

        let (df, term) = idx
            .rarest_df("which handler moves the weight floor")
            .unwrap();
        assert_eq!(df, 1);
        assert_eq!(term, "weight_floor");
    }
}
