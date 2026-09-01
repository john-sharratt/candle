//! What an operator hands the generator to start a life.
//!
//! # The seed is the reusable thing, not the life
//!
//! It is tempting to treat a finished life as the template — it is the big
//! artifact, and copying it is one file operation. It is also how every guard
//! in the city ends up with the same grandmother. Two characters from one seed
//! must get two different lives; that is what the seed being the reusable half
//! means, and it is why nothing generated is stored here.
//!
//! # Cadence, and the reason it is an input
//!
//! Left alone, an authored life bends toward one arc: humble origin, formative
//! loss, mentor, betrayal, resolve. It is a good arc. It is not fifty good
//! arcs, and a city whose whole cast shares it reads as one person wearing
//! different names.
//!
//! The failure does not announce itself in any single life — each one is fine.
//! It shows up across the cast, at which point every life has been generated
//! and the fix is to do all of it again. So the lever exists from the start,
//! it is cheap ([`Cadence`] is one enum on a form), and it acts on the artifact
//! that decides the shape before a word of prose has been paid for.

use serde::{Deserialize, Serialize};

use super::calendar::{age_on, months_in_year, years, BadDate, Date};
use super::plan::is_safe_id;

/// How a life's defining days are distributed across it.
///
/// Not how *much* happens — every month of every year is written whatever the
/// cadence — but where the days that became memories cluster.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Cadence {
    /// Few defining days, widely spaced. A life that happened to them quietly.
    Quiet,
    /// Defining days spread evenly. No single period owns the character.
    #[default]
    Even,
    /// Front-loaded: the person was formed young and has been living with it.
    Early,
    /// Back-loaded: a long ordinary stretch, then everything at once.
    Late,
    /// Long flat runs broken by tight clusters — a life of episodes rather than
    /// a gradient.
    Punctuated,
}

impl Cadence {
    pub const ALL: &'static [Cadence] = &[
        Cadence::Quiet,
        Cadence::Even,
        Cadence::Early,
        Cadence::Late,
        Cadence::Punctuated,
    ];

    /// The instruction the year phase is given. Written as direction to a
    /// writer, because that is what it is — the prompt says this verbatim.
    pub fn instruction(self) -> &'static str {
        match self {
            Cadence::Quiet => {
                "This is a quiet life. Most years hold no day worth remembering on its own. \
                 Mark defining days sparingly and let long stretches pass without one."
            }
            Cadence::Even => {
                "Spread the defining days across the whole life. No single period should own \
                 this character."
            }
            Cadence::Early => {
                "This person was formed young. Concentrate the defining days in the first \
                 third of the life; the later years live with what those years did."
            }
            Cadence::Late => {
                "Most of this life was ordinary. Keep the early and middle years light on \
                 defining days, and let the last third carry them."
            }
            Cadence::Punctuated => {
                "This life came in episodes. Leave long flat runs with no defining day at \
                 all, broken by tight clusters where several fall close together."
            }
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            Cadence::Quiet => "quiet",
            Cadence::Even => "even",
            Cadence::Early => "formed early",
            Cadence::Late => "formed late",
            Cadence::Punctuated => "punctuated",
        }
    }
}

/// What a mind was doing across one stretch of its existence.
///
/// # A life is not one continuous thing
///
/// The ladder was built for a person: born, lived, done. It does not survive contact with a
/// mind that was written in a simulation, archived through a war it never saw, woken for two
/// years of training, and shelved again for three centuries. Those are four different kinds of
/// time and only two of them produce memories.
///
/// So the timeline is a sequence of spans, each declaring what kind of time it was. The
/// generator writes the ones that were lived and **writes nothing at all** for the ones that
/// were not — which is the point: a dormant span is a gap the character genuinely has, not a
/// stretch the author forgot.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EraKind {
    /// Ordinary conscious existence. Written.
    #[default]
    Lived,
    /// Developed in a lab, a simulation, a training substrate — conscious, but inside a world
    /// somebody else was running. Written, and the prompts say which it was: a mind formed
    /// here learned rules that may not hold outside.
    Developed,
    /// Archived, in stasis, powered down. **Nothing is written.**
    ///
    /// Not "nothing interesting happened" — nothing happened, and the character has no memory
    /// of the span at all. That absence is a fact about them and the generator must not paper
    /// over it with prose.
    Dormant,
}

impl EraKind {
    pub const ALL: &'static [EraKind] = &[EraKind::Lived, EraKind::Developed, EraKind::Dormant];

    /// Does this span produce documents?
    pub fn is_written(self) -> bool {
        !matches!(self, EraKind::Dormant)
    }

    pub fn label(self) -> &'static str {
        match self {
            EraKind::Lived => "lived",
            EraKind::Developed => "developed",
            EraKind::Dormant => "dormant",
        }
    }

    /// How the prompts describe this span to the writer.
    pub fn instruction(self) -> &'static str {
        match self {
            EraKind::Lived => "They were awake and in the world for this.",
            EraKind::Developed => {
                "They were being developed inside a simulation for this — conscious, but in a \
                 world with rules somebody else set, which they may not know were rules."
            }
            EraKind::Dormant => {
                "They were archived for this. Nothing happened to them and they remember none \
                 of it."
            }
        }
    }
}

/// The finest stratum written densely across a span.
///
/// **Months do not survive a three-century span.** Every month of a 325-year vigil is 3,900
/// documents for one character, and there are seventy-four of them. A span's grain is how much
/// resolution that stretch of life actually earns: the two years of the classes deserve every
/// month, the long watch does not.
///
/// Days are selective under either — a day is written when it became a memory, which is a
/// property of the day and not of the span.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Grain {
    /// Years only. For long stretches where a year is already a fine enough unit.
    Years,
    /// Years and every month inside them.
    #[default]
    Months,
}

impl Grain {
    pub const ALL: &'static [Grain] = &[Grain::Years, Grain::Months];

    pub fn label(self) -> &'static str {
        match self {
            Grain::Years => "years",
            Grain::Months => "years and months",
        }
    }
}

/// One span of a life.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Era {
    /// `YYYY-MM-DD`, inclusive.
    pub from: String,
    /// `YYYY-MM-DD`, inclusive.
    pub to: String,
    #[serde(default)]
    pub kind: EraKind,
    #[serde(default)]
    pub grain: Grain,
    /// What this span was, in the author's words. Reaches the prompts.
    #[serde(default)]
    pub what: String,
}

/// Something that happened in the world, which this life is lived against.
///
/// **Authored before the life, and shared by every life in the world.** If
/// Cindy's history says the granary burned on 2001-04-02, so must Hess's. The
/// only way two independently generated lives agree about a shared event is for
/// neither of them to have invented it.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct WorldEvent {
    /// `YYYY`, `YYYY-MM` or `YYYY-MM-DD` — world events are known to whatever
    /// precision the world knows them to, and a war does not have a day.
    pub date: String,
    pub what: String,
}

/// Somebody who is in this character's life from the start.
///
/// Seeded rather than generated when the person already exists — another NPC,
/// or a figure the world document names. [`Self::npc_id`] is what makes the
/// relationship the life forms point at a real character rather than at a slug
/// nothing resolves.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct SeedCast {
    pub entity_id: String,
    pub display: String,
    /// Who they are to this character, in one line.
    pub what: String,
    /// The NPC this entity *is*, when it is one.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub npc_id: Option<u64>,
}

/// The whole input to generating a life.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Seed {
    /// The personality id whose life this is — the directory name under
    /// `layers/life/`.
    pub who: String,
    /// Their name, as the prose will use it.
    pub display: String,
    /// `YYYY-MM-DD`.
    pub born: String,
    /// `YYYY-MM-DD` — where the **authored** life stops and the lived one
    /// begins. Normally the world clock's start date.
    pub through: String,
    /// Where this life was lived.
    pub place: String,
    /// What they are, or became.
    pub role: String,
    #[serde(default)]
    pub cadence: Cadence,
    /// Things that must be true of this character, in the author's words.
    #[serde(default)]
    pub facts: Vec<String>,
    /// The world this life is lived against.
    #[serde(default)]
    pub world: Vec<WorldEvent>,
    /// People who are already in the world when this life is generated.
    #[serde(default)]
    pub cast: Vec<SeedCast>,
    /// How this existence divides into spans — see [`Era`].
    ///
    /// Empty means one ordinary lived span from `born` to `through`, which is what a person
    /// is. [`check`] normalises it to exactly that, so nothing downstream ever has to ask
    /// whether a life has eras: it always does.
    #[serde(default)]
    pub eras: Vec<Era>,
}

/// Why a seed cannot be generated from.
#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
#[serde(tag = "problem", rename_all = "snake_case")]
pub enum BadSeed {
    /// A field that cannot be empty is.
    Missing {
        field: &'static str,
    },
    /// A date that is not one.
    Date {
        field: &'static str,
        message: String,
    },
    /// The life ends before it starts.
    Backwards {
        born: String,
        through: String,
    },
    /// Two cast members share an id, so a relationship formed against it would
    /// be ambiguous.
    DuplicateCast {
        entity_id: String,
    },
    /// `who` becomes a directory and a filename, and this one must not.
    UnsafeId {
        who: String,
    },
    /// An era's own dates are unusable.
    EraDate {
        index: usize,
        message: String,
    },
    /// An era ends before it starts.
    EraBackwards {
        index: usize,
        from: String,
        to: String,
    },
    /// The spans do not tile the life: a hole, or an overlap.
    ///
    /// **Refused rather than patched.** A hole means some stretch of the existence is
    /// unaccounted for, and the whole reason spans exist is that "nothing is written here" has
    /// to be a statement somebody made rather than a stretch nobody noticed.
    EraGap {
        after: String,
        before: String,
    },
    EraOverlap {
        at: String,
    },
    /// The spans do not start at `born`, or do not reach `through`.
    EraBounds {
        message: String,
    },
    /// More documents than anyone meant to ask for.
    TooMuchWritten {
        nodes: usize,
    },
}

impl BadSeed {
    pub fn message(&self) -> String {
        match self {
            BadSeed::Missing { field } => format!("`{field}` is required"),
            BadSeed::Date { field, message } => format!("`{field}`: {message}"),
            BadSeed::Backwards { born, through } => {
                format!("the life ends before it starts — born {born}, through {through}")
            }
            BadSeed::DuplicateCast { entity_id } => {
                format!("two cast members are `{entity_id}` — a relationship could mean either")
            }
            BadSeed::UnsafeId { who } => format!(
                "`{who}` cannot be a character id — lowercase letters, digits, `-` and `_` only"
            ),
            BadSeed::EraDate { index, message } => format!("span {}: {message}", index + 1),
            BadSeed::EraBackwards { index, from, to } => {
                format!("span {} ends before it starts — {from} to {to}", index + 1)
            }
            BadSeed::EraGap { after, before } => format!(
                "nothing accounts for {after} to {before}. Every stretch needs a span, even an \
                 empty one: a dormant span says the character was archived, and a hole says \
                 somebody forgot"
            ),
            BadSeed::EraOverlap { at } => {
                format!("two spans both cover {at} — a life is in one state at a time")
            }
            BadSeed::EraBounds { message } => message.clone(),
            BadSeed::TooMuchWritten { nodes } => format!(
                "this seed asks for {nodes} documents. Check the dates, or coarsen a long \
                 span's grain to years"
            ),
        }
    }
}

/// The most documents one seed may ask the generator for.
///
/// **This replaced a maximum lifespan, and the change is not cosmetic.** A
/// ceiling on years was a statement about mortality, and the minds this engine
/// hosts are not mortal: a construct commissioned before the war is six
/// centuries old, and a mind written in a simulation is older than the war it
/// slept through. Refusing them was refusing the setting.
///
/// What the old guard was *for* survives — a typo'd century must not become a
/// wave queue nothing cancels in time — so the bound moved to the thing that
/// actually costs: how much gets written. A dormant span is free however long it
/// runs, and a long span that earns only its years is cheap. Three centuries at
/// month grain is neither, and this is what says so.
/// Sized against what a legitimate character actually asks for. A three-century vigil at year
/// grain is 331; the two years of a training course at month grain is 26; the longest
/// defensible month-grain span is a few thousand. A mistyped millennium at month grain is
/// twelve thousand, and that is the gap this sits in.
const MAX_NODES: usize = 6_000;

/// A seed that has been checked, with its dates already parsed.
///
/// Every consumer past this point takes one of these rather than a [`Seed`], so
/// the date parsing and the range checks happen exactly once — a generator that
/// re-parses is a generator that can disagree with the validator.
#[derive(Clone, Debug)]
pub struct Checked {
    pub seed: Seed,
    pub born: Date,
    pub through: Date,
    /// The spans, normalised: sorted, tiling `born..through` with no hole and no overlap, and
    /// never empty — a seed that declared none gets one `Lived` span covering everything.
    ///
    /// Downstream never asks whether a life has spans. It always does, so there is one shape
    /// to handle rather than two.
    pub eras: Vec<CheckedEra>,
}

/// A span with its dates parsed.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CheckedEra {
    pub from: Date,
    pub to: Date,
    pub kind: EraKind,
    pub grain: Grain,
    pub what: String,
}

impl CheckedEra {
    /// The years this span touches, inclusive.
    pub fn years(&self) -> Vec<i32> {
        years(self.from, self.to)
    }

    /// Documents this span will produce, before any day is marked: its years, plus their
    /// months when the grain asks for them.
    pub fn nodes(&self) -> usize {
        if !self.kind.is_written() {
            return 0;
        }
        let ys = self.years();
        match self.grain {
            Grain::Years => ys.len(),
            Grain::Months => {
                ys.len()
                    + ys.iter()
                        .map(|y| months_in_year(*y, self.from, self.to).len())
                        .sum::<usize>()
            }
        }
    }
}

impl Checked {
    /// Every year the life is **written** in, in order and without repeats.
    ///
    /// Dormant spans contribute nothing, so a mind archived for three centuries has no years
    /// for them — which is the whole point: there is no document to write and no year to put
    /// one in.
    pub fn years(&self) -> Vec<i32> {
        let mut out = Vec::new();
        for e in self.eras.iter().filter(|e| e.kind.is_written()) {
            for y in e.years() {
                if !out.contains(&y) {
                    out.push(y);
                }
            }
        }
        out.sort_unstable();
        out
    }

    /// How old they are where the authored life stops — counted from birth in calendar years,
    /// dormancy included. A mind shelved for three centuries is three centuries older.
    pub fn age(&self) -> i32 {
        age_on(self.born, self.through)
    }

    /// Total documents the skeleton will hold.
    pub fn nodes(&self) -> usize {
        self.eras.iter().map(CheckedEra::nodes).sum()
    }
}

/// Check a seed, reporting **everything** wrong with it.
///
/// All the problems, not the first: an operator fixing a form wants the whole
/// list, and a validator that stops at the first fault turns one correction
/// into five round trips.
pub fn check(seed: &Seed) -> Result<Checked, Vec<BadSeed>> {
    let mut bad = Vec::new();

    for (field, value) in [
        ("who", &seed.who),
        ("display", &seed.display),
        ("place", &seed.place),
        ("role", &seed.role),
    ] {
        if value.trim().is_empty() {
            bad.push(BadSeed::Missing { field });
        }
    }
    // `who` becomes a directory under `layers/life/` and a filename under
    // `.lifegen/`, so it is checked here rather than trusted from wherever the
    // seed arrived.
    if !seed.who.trim().is_empty() && !is_safe_id(&seed.who) {
        bad.push(BadSeed::UnsafeId {
            who: seed.who.clone(),
        });
    }

    let mut date = |field: &'static str, raw: &str| match Date::parse(raw) {
        Ok(d) => Some(d),
        Err(e) => {
            bad.push(BadSeed::Date {
                field,
                message: match &e {
                    BadDate::Shape(_) | BadDate::NoSuchDay(_) => e.message(),
                },
            });
            None
        }
    };
    let born = date("born", &seed.born);
    let through = date("through", &seed.through);

    if let (Some(b), Some(t)) = (born, through) {
        if t < b {
            bad.push(BadSeed::Backwards {
                born: seed.born.clone(),
                through: seed.through.clone(),
            });
        }
    }

    // The spans, parsed and ordered. Checked against each other only when every one of them
    // is individually a date range — comparing a span that failed to parse against its
    // neighbour reports a hole that is really a typo.
    let mut eras: Vec<CheckedEra> = Vec::new();
    let mut era_dates_ok = true;
    for (i, e) in seed.eras.iter().enumerate() {
        let mut one = |raw: &str| match Date::parse(raw) {
            Ok(d) => Some(d),
            Err(err) => {
                bad.push(BadSeed::EraDate {
                    index: i,
                    message: err.message(),
                });
                None
            }
        };
        let (from, to) = (one(&e.from), one(&e.to));
        let (Some(from), Some(to)) = (from, to) else {
            era_dates_ok = false;
            continue;
        };
        if to < from {
            bad.push(BadSeed::EraBackwards {
                index: i,
                from: e.from.clone(),
                to: e.to.clone(),
            });
            era_dates_ok = false;
            continue;
        }
        eras.push(CheckedEra {
            from,
            to,
            kind: e.kind,
            grain: e.grain,
            what: e.what.clone(),
        });
    }
    eras.sort_by_key(|e| e.from);

    if era_dates_ok {
        if eras.is_empty() {
            // A life that declared no spans is one ordinary lived stretch, which is what a
            // person is. Normalised here so nothing downstream has to handle the absence.
            if let (Some(from), Some(to)) = (born, through) {
                eras.push(CheckedEra {
                    from,
                    to,
                    kind: EraKind::Lived,
                    grain: Grain::Months,
                    what: String::new(),
                });
            }
        } else {
            for w in eras.windows(2) {
                let (a, b) = (&w[0], &w[1]);
                if b.from <= a.to {
                    bad.push(BadSeed::EraOverlap {
                        at: b.from.to_string(),
                    });
                } else if !a.to.is_day_before(b.from) {
                    bad.push(BadSeed::EraGap {
                        after: a.to.to_string(),
                        before: b.from.to_string(),
                    });
                }
            }
            if let (Some(b), Some(t)) = (born, through) {
                let (first, last) = (&eras[0], &eras[eras.len() - 1]);
                if first.from != b {
                    bad.push(BadSeed::EraBounds {
                        message: format!(
                            "the first span starts {} but they begin {b} — every stretch of an \
                             existence needs a span",
                            first.from
                        ),
                    });
                }
                if last.to != t {
                    bad.push(BadSeed::EraBounds {
                        message: format!(
                            "the last span ends {} but the authored life runs to {t}",
                            last.to
                        ),
                    });
                }
            }
        }
        let nodes: usize = eras.iter().map(CheckedEra::nodes).sum();
        if nodes > MAX_NODES {
            bad.push(BadSeed::TooMuchWritten { nodes });
        }
    }

    let mut seen: Vec<&str> = Vec::new();
    for c in &seed.cast {
        if seen.contains(&c.entity_id.as_str()) {
            bad.push(BadSeed::DuplicateCast {
                entity_id: c.entity_id.clone(),
            });
        }
        seen.push(&c.entity_id);
    }

    match (bad.is_empty(), born, through) {
        (true, Some(born), Some(through)) => Ok(Checked {
            seed: seed.clone(),
            born,
            through,
            eras,
        }),
        _ => Err(bad),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn seed() -> Seed {
        Seed {
            who: "cindy-tan".into(),
            display: "Cindy Tan".into(),
            born: "1980-03-11".into(),
            through: "2024-01-01".into(),
            place: "Nanyang".into(),
            role: "a records clerk".into(),
            cadence: Cadence::Even,
            facts: vec!["She never learned to swim.".into()],
            world: vec![WorldEvent {
                date: "2001-04-02".into(),
                what: "The east granary burned.".into(),
            }],
            cast: vec![SeedCast {
                entity_id: "prof-lim".into(),
                display: "Professor Lim".into(),
                what: "Taught her to question assumptions.".into(),
                npc_id: Some(7),
            }],
            eras: Vec::new(),
        }
    }

    #[test]
    fn a_complete_seed_checks_out_with_its_dates_parsed() {
        let c = check(&seed()).unwrap();
        assert_eq!(c.born.year, 1980);
        assert_eq!(c.through.year, 2024);
        assert_eq!(c.age(), 43);
        assert_eq!(c.years().len(), 45);
    }

    /// **Every problem, not the first.** An operator fixing a form wants the
    /// whole list; stopping at the first fault turns one correction into five
    /// round trips.
    #[test]
    fn checking_reports_every_problem_at_once() {
        let mut s = seed();
        s.who = String::new();
        s.role = "  ".into();
        s.born = "1980-3-11".into();
        let bad = check(&s).unwrap_err();
        assert!(bad.contains(&BadSeed::Missing { field: "who" }));
        assert!(bad.contains(&BadSeed::Missing { field: "role" }));
        assert!(bad
            .iter()
            .any(|b| matches!(b, BadSeed::Date { field: "born", .. })));
        assert_eq!(bad.len(), 3, "{bad:?}");
    }

    #[test]
    fn a_life_that_ends_before_it_starts_is_refused() {
        let mut s = seed();
        s.through = "1970-01-01".into();
        let bad = check(&s).unwrap_err();
        assert!(matches!(bad[0], BadSeed::Backwards { .. }));
        assert!(bad[0].message().contains("ends before it starts"));
    }

    /// **The guard is on the work, not on the lifespan** — and the difference is the whole
    /// point of the change. A mistyped millennium at month grain is twelve thousand documents
    /// and is refused; the *same span* asking only for its years is nine hundred and is
    /// allowed, because the minds this engine hosts really are that old.
    #[test]
    fn a_long_span_is_refused_for_what_it_writes_not_for_being_long() {
        let mut s = seed();
        s.born = "1080-03-11".into();

        let bad = check(&s).unwrap_err();
        assert!(
            bad.iter()
                .any(|b| matches!(b, BadSeed::TooMuchWritten { .. })),
            "{bad:?}"
        );
        assert!(bad[0].message().contains("coarsen"), "{}", bad[0].message());

        // Coarsened to years, the very same nine centuries are fine.
        s.eras = vec![Era {
            from: s.born.clone(),
            to: s.through.clone(),
            kind: EraKind::Lived,
            grain: Grain::Years,
            what: String::new(),
        }];
        let ok = check(&s).expect("years grain over the same span");
        assert_eq!(ok.nodes(), ok.years().len());
        assert!(ok.nodes() > 900, "{} nodes", ok.nodes());
    }

    /// **A dormant span costs nothing, however long it runs.** Keeper's three centuries of
    /// hibernation must not count against a budget meant to catch runaway work.
    #[test]
    fn a_dormant_span_is_free() {
        let mut s = seed();
        s.born = "2461-01-01".into();
        s.through = "3087-01-01".into();
        s.eras = vec![
            Era {
                from: "2461-01-01".into(),
                to: "2792-12-31".into(),
                kind: EraKind::Lived,
                grain: Grain::Years,
                what: "The long watch.".into(),
            },
            Era {
                from: "2793-01-01".into(),
                to: "3087-01-01".into(),
                kind: EraKind::Dormant,
                grain: Grain::Years,
                what: "Hibernation.".into(),
            },
        ];
        let c = check(&s).expect("a life with a gap in it");
        // 332 written years, and not one document for the 295 dormant ones.
        assert_eq!(c.nodes(), 332);
        assert_eq!(c.years().len(), 332);
        assert!(
            !c.years().contains(&3000),
            "a dormant year has no documents"
        );
        // The character is still six centuries old — dormancy is time, just not memory.
        assert_eq!(c.age(), 626);
    }

    /// **A hole is refused, not filled.** The reason spans exist at all is that "nothing is
    /// written here" has to be something an author said. A stretch nobody accounted for is
    /// indistinguishable from a stretch somebody forgot, and the generator would quietly write
    /// neither.
    #[test]
    fn spans_must_tile_the_whole_existence() {
        let mut s = seed();
        s.born = "2000-01-01".into();
        s.through = "2010-12-31".into();
        let span = |from: &str, to: &str| Era {
            from: from.into(),
            to: to.into(),
            kind: EraKind::Lived,
            grain: Grain::Years,
            what: String::new(),
        };

        // A hole between two spans.
        s.eras = vec![
            span("2000-01-01", "2004-12-31"),
            span("2006-01-01", "2010-12-31"),
        ];
        let bad = check(&s).unwrap_err();
        assert!(
            bad.iter().any(|b| matches!(b, BadSeed::EraGap { .. })),
            "{bad:?}"
        );
        assert!(bad[0].message().contains("somebody forgot"));

        // Overlapping spans — a life is in one state at a time.
        s.eras = vec![
            span("2000-01-01", "2006-12-31"),
            span("2005-01-01", "2010-12-31"),
        ];
        let bad = check(&s).unwrap_err();
        assert!(
            bad.iter().any(|b| matches!(b, BadSeed::EraOverlap { .. })),
            "{bad:?}"
        );

        // Not reaching the end.
        s.eras = vec![span("2000-01-01", "2008-12-31")];
        let bad = check(&s).unwrap_err();
        assert!(
            bad.iter().any(|b| matches!(b, BadSeed::EraBounds { .. })),
            "{bad:?}"
        );

        // Abutting exactly — the day after is not a hole.
        s.eras = vec![
            span("2000-01-01", "2004-12-31"),
            span("2005-01-01", "2010-12-31"),
        ];
        assert!(check(&s).is_ok());
    }

    /// A life that declares no spans is one ordinary lived stretch — normalised here so
    /// nothing downstream ever has to handle the absence.
    #[test]
    fn a_life_with_no_spans_declared_becomes_one_lived_span() {
        let c = check(&seed()).unwrap();
        assert_eq!(c.eras.len(), 1);
        assert_eq!(c.eras[0].kind, EraKind::Lived);
        assert_eq!(c.eras[0].from, c.born);
        assert_eq!(c.eras[0].to, c.through);
    }

    /// Every kind and grain must say something the prompts can use, or a span that reads as
    /// distinct in the console is indistinguishable to the writer.
    #[test]
    fn every_kind_and_grain_carries_its_own_words() {
        for k in EraKind::ALL {
            assert!(!k.label().is_empty());
            assert!(!k.instruction().is_empty(), "{k:?}");
        }
        assert!(EraKind::Lived.is_written() && EraKind::Developed.is_written());
        assert!(!EraKind::Dormant.is_written());
        for g in Grain::ALL {
            assert!(!g.label().is_empty());
        }
    }

    /// A duplicated id makes every relationship formed against it ambiguous,
    /// and the ambiguity would only surface as a character with two versions of
    /// one person.
    #[test]
    fn two_cast_members_may_not_share_an_id() {
        let mut s = seed();
        let dup = s.cast[0].clone();
        s.cast.push(dup);
        let bad = check(&s).unwrap_err();
        assert_eq!(
            bad[0],
            BadSeed::DuplicateCast {
                entity_id: "prof-lim".into()
            }
        );
    }

    /// **`who` becomes a directory and a filename.** It arrives from a URL, so
    /// an unchecked one is a path traversal.
    #[test]
    fn a_character_id_that_could_escape_the_mind_directory_is_refused() {
        for who in [
            "../../etc/passwd",
            "..",
            "a/b",
            "a\\b",
            "C:windows",
            "Cindy",
            "cindy tan",
            "cindy.tan",
            "café",
            &"x".repeat(65),
        ] {
            let mut s = seed();
            s.who = who.into();
            let bad = check(&s).unwrap_err();
            assert!(
                bad.iter().any(|b| matches!(b, BadSeed::UnsafeId { .. })),
                "{who:?} was accepted as a path component"
            );
        }
        // And the shapes real personality ids take are all fine.
        for who in ["cindy-tan", "hess", "guard_07", "a"] {
            let mut s = seed();
            s.who = who.into();
            assert!(check(&s).is_ok(), "{who} was refused");
        }
    }

    /// A date that is shaped right but names no real day must be caught here,
    /// not discovered as a filename that sorts into a day that never happened.
    #[test]
    fn an_impossible_day_is_refused() {
        let mut s = seed();
        s.born = "1981-02-29".into();
        let bad = check(&s).unwrap_err();
        assert!(bad[0].message().contains("not a day the calendar has"));
    }

    /// Every cadence must give the year phase something to act on, or the lever
    /// against a city of identical arcs is a dropdown that does nothing.
    #[test]
    fn every_cadence_carries_an_instruction_and_a_label() {
        for c in Cadence::ALL {
            assert!(!c.instruction().is_empty(), "{c:?} has no instruction");
            assert!(!c.label().is_empty(), "{c:?} has no label");
        }
        assert_eq!(Cadence::default(), Cadence::Even);
    }

    /// The seed round-trips through JSON: it is written to disk beside the plan
    /// and sent to the console, and a field lost in either direction is an
    /// operator's work lost.
    #[test]
    fn a_seed_round_trips_through_json() {
        let s = seed();
        let back: Seed = serde_json::from_str(&serde_json::to_string(&s).unwrap()).unwrap();
        assert_eq!(s, back);
    }

    /// The optional halves default rather than failing, so a hand-written seed
    /// need only carry what it actually says.
    #[test]
    fn the_optional_fields_may_be_omitted() {
        let s: Seed = serde_json::from_str(
            r#"{"who":"a","display":"A","born":"1990-01-01","through":"2020-01-01",
                "place":"p","role":"r"}"#,
        )
        .unwrap();
        assert_eq!(s.cadence, Cadence::Even);
        assert!(s.facts.is_empty() && s.world.is_empty() && s.cast.is_empty());
        assert!(check(&s).is_ok());
    }
}
