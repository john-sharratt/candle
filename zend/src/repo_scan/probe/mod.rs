//! Folder probes — the `repo_map` layer's **retrieval surface**.
//!
//! A folder is not retrievable because of what it *is*. It is retrievable
//! because of the questions it is the answer to. The layer's decoded
//! two-sentence summary describes the folder to a reader who has already found
//! it; nothing about it is shaped like the query that should have found it in
//! the first place, and a BDP scan comparing `sign(Q_decode)` against stored `K`
//! is being asked to bridge that genre gap on every lookup.
//!
//! So each directory carries two artifacts with different jobs:
//!
//! * the **summary** — the payload injected when the folder is retrieved, and
//! * a set of **probes** — question-shaped turns whose only job is to be hit by
//!   a scan and resolve to this folder. Distilled provenance-only: signatures
//!   resident, text discarded.
//!
//! That is the same shape the tool catalog already uses, and for the same
//! reason: matching a query against *examples of queries* beats matching it
//! against a definition. Tool selection went to 100% top-3 / 97% top-1 on
//! exactly this change.
//!
//! # The four registers
//!
//! Real queries arrive in distinct shapes, and a generator asked simply for
//! "questions" returns six of one. The budget is therefore spent explicitly:
//!
//! | register | shape | rarity gate | dedup |
//! |---|---|---|---|
//! | [`Register::Locational`] | where is X / which file handles Y | required | within folder |
//! | [`Register::Mechanistic`] | how does Z work / what happens when | required | within folder |
//! | [`Register::Conceptual`] | what is a `ChunkGid` / what does BDP mean | required | within folder |
//! | [`Register::Systemic`] | subject-bearing, vocabulary-free | **exempt** | **global** |
//!
//! The fourth register is the one that matters most, and the one a
//! specificity-only design would have missed. A probe requiring a rare symbol
//! only ever fires for someone who has already seen that symbol — and that
//! person does not need a repo map. The people who need one cannot name
//! anything yet, and their questions are the ones the layer must answer:
//! *"give me a tour of the codebase"* is the opening query of every quality
//! battery and has no distinctive term in it at all.
//!
//! What keeps register 4 from becoming noise is not rarity but a rule about
//! *what it carries*: **subject-bearing but vocabulary-free**. "What does this
//! module do" carries no subject and matches every folder equally — it is not a
//! probe, it is a constant. "How does the system avoid running out of GPU memory
//! during a long conversation?" carries the subject in plain language and
//! discriminates. [`filters`] enforces the first half; the retrieval harness
//! enforces the second.
//!
//! # Admission
//!
//! Two gates and one oracle, in that order:
//!
//! 1. [`filters::admit`] — shape, self-address, deixis, and (registers 1–3) the
//!    rarity gate from [`idf::TermIndex`].
//! 2. [`filters::systemic_collisions`] — register 4 only, across all folders.
//! 3. **Leave-one-out retrieval** ([`crate::repo_scan::probe::harness`]) — every
//!    probe, every register, must rank its own folder first against the full
//!    corpus. A probe that cannot retrieve its own folder is useless; one that
//!    retrieves somebody else's is actively harmful. No register is exempt.

pub mod filters;
pub mod harness;
pub mod idf;
pub mod plan;
pub mod render;
pub mod symbols;

/// Probes kept per register, per directory. Four registers, so 24 per folder.
pub const PROBES_PER_REGISTER: usize = 6;

/// Candidates requested per register before filtering.
///
/// Generate wide, keep the best — the "top 6" is a selection, not a quota to
/// fill. Doubling the ask costs almost nothing (questions are short, and all
/// four registers come out of one decode) while the expensive artifact, the
/// think→response chain, is only ever built for a probe that has already passed
/// every gate.
pub const CANDIDATES_PER_REGISTER: usize = 12;

/// Total probes a fully-admitted directory carries.
pub const PROBES_PER_DIR: usize = PROBES_PER_REGISTER * 4;

/// The query shape a probe is written in.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Register {
    /// Where something lives. The layer's primary job — routing attention to a
    /// folder — so these are the probes that most directly do the work.
    Locational,
    /// How something behaves. The most common shape a real question takes.
    Mechanistic,
    /// What a name means. Vocabulary: a newcomer meeting a term in code needs to
    /// find where it lives, and this is the register that answers them.
    Conceptual,
    /// The subject in plain language, with none of the folder's jargon.
    Systemic,
}

impl Register {
    /// Every register, in generation order.
    pub const ALL: [Register; 4] = [
        Register::Locational,
        Register::Mechanistic,
        Register::Conceptual,
        Register::Systemic,
    ];

    /// Stable identifier, used as the generation heading, the persisted tag, and
    /// the report key. One spelling for all three so a log line, a metadata tag,
    /// and a parsed heading can never disagree.
    pub fn id(self) -> &'static str {
        match self {
            Register::Locational => "locational",
            Register::Mechanistic => "mechanistic",
            Register::Conceptual => "conceptual",
            Register::Systemic => "systemic",
        }
    }

    /// Parse an id back, for reading a generated block's heading.
    pub fn from_id(s: &str) -> Option<Self> {
        Register::ALL
            .into_iter()
            .find(|r| r.id().eq_ignore_ascii_case(s.trim()))
    }

    /// Whether a probe in this register must carry a term the
    /// directory-frequency index calls distinctive.
    ///
    /// False for [`Register::Systemic`] alone, and that exemption is the point
    /// of the register rather than a concession: requiring a rare term there
    /// would recreate the other three in worse prose, and lose the newcomer
    /// queries that no specific probe can serve.
    pub fn requires_rarity(self) -> bool {
        self != Register::Systemic
    }

    /// The instruction shown to the generator for this register.
    pub fn brief(self) -> &'static str {
        match self {
            Register::Locational => {
                "Questions asking WHERE something lives — which part of the codebase implements \
                 a named thing, or handles a named job. Each must name at least one of the \
                 distinctive terms listed above."
            }
            Register::Mechanistic => {
                "Questions asking HOW something works, or what happens when it runs. Each must \
                 name at least one of the distinctive terms listed above."
            }
            Register::Conceptual => {
                "Questions asking WHAT A NAME MEANS — the definition, purpose or role of a type, \
                 constant or concept. Each must name at least one of the distinctive terms \
                 listed above."
            }
            Register::Systemic => {
                "Questions a NEWCOMER would ask, who does not know any of this project's \
                 vocabulary yet. Use NONE of the distinctive terms above and no code \
                 identifiers at all — plain English only.\n\
                 Each question MUST NAME THE SUBJECT in ordinary words — the kind of thing \
                 this folder deals with, taken from the description above and said the way \
                 someone outside the project would say it. \"How does the system stay \
                 consistent?\" is worthless: it fits every folder ever written. \"How does the \
                 assistant decide which parts of a huge codebase to show the model?\" is \
                 useful, because only one area of the system could answer it. If your question \
                 would still make sense pasted under a different folder, it is wrong."
            }
        }
    }
}

/// One admitted probe.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Probe {
    pub text: String,
    pub register: Register,
}

impl Probe {
    /// Gather-scope tags for this probe's turn: the layer, the directory, and
    /// the register.
    ///
    /// The directory tag is what makes a scan hit resolve to a folder — it is
    /// the same key [`crate::repo_scan`] tags a folder's summary conversation
    /// with, so probe and payload are joined by the tag rather than by any
    /// separate table.
    pub fn tags(&self, dir: &str) -> Vec<String> {
        vec![
            "repo_map".to_string(),
            dir.to_string(),
            format!("probe:{}", self.register.id()),
        ]
    }
}

/// A directory's admitted probe set, with the accounting behind it.
#[derive(Debug, Clone, Default)]
pub struct ProbeSet {
    pub dir: String,
    pub probes: Vec<Probe>,
    /// Admissible candidates that lost on slots alone, kept as **test queries**
    /// and never ingested.
    ///
    /// This is what makes the retrieval oracle honest, and it is free: scoring a
    /// probe that is already in the corpus measures self-consistency, because
    /// its own signature is resident and self-matches. A question the corpus has
    /// never seen measures retrieval. Generating twelve per register and keeping
    /// six leaves exactly that population behind — well-formed, gate-passing,
    /// simply out of slots — so the held-out set costs one extra parse and no
    /// extra decode. See [`harness`].
    pub held_out: Vec<Probe>,
    /// Candidates refused, with the reason — so a folder that produced nothing
    /// usable reports *how* rather than a bare zero.
    pub rejected: Vec<(String, filters::Rejection)>,
}

impl ProbeSet {
    /// Probes admitted in one register.
    pub fn in_register(&self, register: Register) -> impl Iterator<Item = &Probe> {
        self.probes.iter().filter(move |p| p.register == register)
    }

    /// Rejection counts by reason tag, for the report.
    pub fn rejection_tally(&self) -> std::collections::BTreeMap<&'static str, usize> {
        let mut out = std::collections::BTreeMap::new();
        for (_, why) in &self.rejected {
            *out.entry(why.tag()).or_insert(0) += 1;
        }
        out
    }

    /// Whether every register reached its full complement.
    ///
    /// `>=`, matching [`super::metadata::FolderMetadata::is_complete`]. A
    /// register is not capped at [`PROBES_PER_REGISTER`] — `plan::build_authored`
    /// deliberately admits everything a folder's file offers, and a folder whose
    /// file carries a pass's six probes plus its held-out three comes back with
    /// seven — so `==` read every over-full register as incomplete and
    /// `ProbeStats::dirs_complete` reported zero across the workspace from the
    /// second run onward, which is precisely the reading that counter exists to
    /// distinguish from "half of them got twice as many".
    pub fn is_complete(&self) -> bool {
        Register::ALL
            .into_iter()
            .all(|r| self.in_register(r).count() >= PROBES_PER_REGISTER)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn register_ids_round_trip() {
        for r in Register::ALL {
            assert_eq!(Register::from_id(r.id()), Some(r));
        }
        assert_eq!(Register::from_id("LOCATIONAL"), Some(Register::Locational));
        assert_eq!(Register::from_id("nonsense"), None);
    }

    /// The exemption is the design, not an oversight — assert it directly so a
    /// later tidy-up cannot quietly make register 4 specific and delete the
    /// newcomer queries.
    #[test]
    fn only_the_systemic_register_is_exempt_from_rarity() {
        assert!(Register::Locational.requires_rarity());
        assert!(Register::Mechanistic.requires_rarity());
        assert!(Register::Conceptual.requires_rarity());
        assert!(!Register::Systemic.requires_rarity());
    }

    /// The systemic brief must forbid identifiers outright. Without that clause
    /// the generator writes specific questions in all four registers and the
    /// newcomer coverage silently disappears.
    #[test]
    fn the_systemic_brief_forbids_identifiers() {
        let brief = Register::Systemic.brief();
        assert!(brief.contains("plain English"), "{brief}");
        assert!(brief.contains("NONE of the distinctive terms"), "{brief}");
    }

    /// Forbidding vocabulary is only half the register. Told only to avoid
    /// identifiers, the generator writes questions that fit every folder ever
    /// written — "How does the system ensure consistency between cached metadata
    /// and live sources?" — which is the promiscuous attractor this whole design
    /// exists to avoid. The brief must also DEMAND the subject.
    #[test]
    fn the_systemic_brief_demands_the_subject_in_plain_words() {
        let brief = Register::Systemic.brief();
        assert!(brief.contains("MUST NAME THE SUBJECT"), "{brief}");
        assert!(
            brief.contains("pasted under a different folder"),
            "the brief must state the disqualifying test: {brief}",
        );
    }

    #[test]
    fn a_probes_tags_carry_layer_directory_and_register() {
        let p = Probe {
            text: "What is a ChunkGid?".to_string(),
            register: Register::Conceptual,
        };
        assert_eq!(
            p.tags("candle-nn/src/kv_cache/"),
            vec![
                "repo_map".to_string(),
                "candle-nn/src/kv_cache/".to_string(),
                "probe:conceptual".to_string(),
            ],
        );
    }

    #[test]
    fn completeness_needs_every_register_full() {
        let mut set = ProbeSet {
            dir: "a/".to_string(),
            ..Default::default()
        };
        for r in Register::ALL {
            for i in 0..PROBES_PER_REGISTER {
                set.probes.push(Probe {
                    text: format!("q{i}?"),
                    register: r,
                });
            }
        }
        assert!(set.is_complete());
        set.probes.pop();
        assert!(!set.is_complete());
    }

    /// Generate-wide-then-select is the design, not an accident of two
    /// constants: admission must have real competition to choose from, and the
    /// surplus is also where [`ProbeSet::held_out`] comes from. Halving the
    /// candidate count would silently take both away.
    #[test]
    fn the_budget_is_twenty_four_per_directory_with_room_to_select() {
        assert_eq!(PROBES_PER_DIR, 24);
        const _: () = assert!(CANDIDATES_PER_REGISTER >= PROBES_PER_REGISTER * 2);
    }
}
