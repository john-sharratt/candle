//! The life under construction: a tree the operator edits and the generator
//! fills.
//!
//! # Why a plan exists at all, rather than generating straight to documents
//!
//! The documents in `layers/life/<who>/` are the product. The plan is the thing
//! you can still change your mind about. It holds what has been generated, what
//! a human has touched, and what a change upstream has invalidated — none of
//! which a directory of markdown can express.
//!
//! It is **not ingestible**: it is JSON, and
//! [`crate::engine::watcher::is_ingestible`] admits only `.md`, `.yaml` and
//! `.yml`. It also lives outside `layers/` entirely, under `.lifegen/`, because
//! a plan sitting in a life directory would be walked by
//! [`crate::engine::life::episodes`] and reported as an undated document once
//! per load, forever.
//!
//! # Staleness, and why a hand edit outranks a regeneration
//!
//! Edit a year and its months no longer follow from it. Three things could
//! happen, and only one is trustworthy:
//!
//! - Leave them, now contradicting their parent, with nothing to say so.
//! - Regenerate the subtree, discarding whatever the operator wrote in it.
//! - **Mark the subtree stale and let the operator decide** — which is this.
//!
//! A hand edit sets [`Content::edited`], and a regeneration skips an edited
//! node unless told otherwise. The reason is not sentiment about authorship: a
//! tool that eats your work when you fix a typo two levels up is one you stop
//! using for anything you care about, and the whole design rests on a human
//! reviewing what the model wrote.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use super::calendar::{months_in_year, Date};
use super::consequence::Consequence;
use super::seed::{Checked, Era, Grain, Seed};

/// One generated body of text, and what has happened to it since.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Content {
    /// What this stratum is called. Becomes the document's title, so it is part
    /// of the filename and not decoration.
    #[serde(default)]
    pub title: String,
    /// The prose. Empty means never generated — there is no separate flag,
    /// because a generated-but-empty document is not a state worth having.
    #[serde(default)]
    pub text: String,
    /// **A human wrote or changed this.** Sticky: a regeneration of an ancestor
    /// leaves it alone.
    #[serde(default)]
    pub edited: bool,
    /// An ancestor changed after this was generated, so it no longer follows
    /// from what is above it.
    #[serde(default)]
    pub stale: bool,
}

impl Content {
    pub fn is_generated(&self) -> bool {
        !self.text.trim().is_empty()
    }

    /// Does this node want the generator?
    ///
    /// Ungenerated, or stale — but never when a human has touched it. That last
    /// clause is the sticky rule, in the one place it is decided.
    pub fn wants_generation(&self) -> bool {
        !self.edited && (!self.is_generated() || self.stale)
    }

    /// Record a human edit: the text is theirs now, and it is no longer stale
    /// with respect to anything.
    pub fn edit(&mut self, title: String, text: String) {
        self.title = title;
        self.text = text;
        self.edited = true;
        self.stale = false;
    }

    /// Record a generation. Clears staleness; does **not** set `edited`.
    pub fn generated(&mut self, title: String, text: String) {
        self.title = title;
        self.text = text;
        self.stale = false;
    }
}

/// Somebody in this character's life.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct CastMember {
    pub entity_id: String,
    pub display: String,
    /// Who they are to this character, in one line.
    #[serde(default)]
    pub what: String,
    /// The NPC this entity is, when it is one — so a relationship the life
    /// forms points at a real character rather than at a slug that resolves to
    /// nothing.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub npc_id: Option<u64>,
    /// Carried from the seed rather than invented by the story phase.
    #[serde(default)]
    pub from_seed: bool,
}

/// One year as the story phase allocated it: a name and a premise, before any
/// of it has been written out.
///
/// **This is the "assign" half of assign-above-expand-below.** The story decides
/// what each year of the life is *for*; the year phase expands that into prose
/// without adding events of its own. Because the allocation is made once, in one
/// decode that sees the whole life, the years can then be written in parallel
/// without any of them needing to know what the others said.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct YearBeat {
    pub year: i32,
    pub title: String,
    /// What this year is for, in a line or two.
    pub premise: String,
}

/// The whole-life arc. Undated, and therefore not a life document.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Story {
    #[serde(flatten)]
    pub content: Content,
    /// **Every person the life may refer to, fixed here and inherited by every
    /// stratum below.** No lower phase invents a person; that is what stops one
    /// professor becoming four slugs across fifteen episodes.
    #[serde(default)]
    pub cast: Vec<CastMember>,
    /// The year-by-year allocation of the arc. One entry per year of the life.
    #[serde(default)]
    pub outline: Vec<YearBeat>,
}

impl Story {
    /// The premise assigned to one year, if the outline reached it.
    pub fn beat(&self, year: i32) -> Option<&YearBeat> {
        self.outline.iter().find(|b| b.year == year)
    }
}

/// One day that became a memory.
///
/// Not `Eq`, and nothing below it can be: a consequence's arguments are
/// `serde_json::Value`, which carries `f64`. Equality on a dial is approximate
/// by nature, so the derive stops at `PartialEq` rather than pretending.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct Day {
    pub day: u32,
    #[serde(flatten)]
    pub content: Content,
    /// What this day is required to produce — authored by the operator, never
    /// by the model. See [`super::consequence`].
    #[serde(default)]
    pub consequences: Vec<Consequence>,
}

/// One month. Always written; never skipped for being uneventful.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct Month {
    pub month: u32,
    #[serde(flatten)]
    pub content: Content,
    #[serde(default)]
    pub days: Vec<Day>,
}

/// One year. Always written.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct Year {
    pub year: i32,
    #[serde(flatten)]
    pub content: Content,
    #[serde(default)]
    pub months: Vec<Month>,
}

/// A life being built.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Plan {
    pub seed: Seed,
    #[serde(default)]
    pub story: Story,
    #[serde(default)]
    pub years: Vec<Year>,
}

/// Which stratum a phase writes.
///
/// The ladder, in execution order. Each rung's prompt is built from the rungs
/// above it, so the order is a dependency and not a preference.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Phase {
    Story,
    Years,
    Months,
    Days,
}

impl Phase {
    pub const ALL: &'static [Phase] = &[Phase::Story, Phase::Years, Phase::Months, Phase::Days];

    /// The line the progress overlay renders.
    pub fn label(self) -> &'static str {
        match self {
            Phase::Story => "Writing the life story",
            Phase::Years => "Laying out the years",
            Phase::Months => "Writing the months",
            Phase::Days => "Writing the defining days",
        }
    }

    /// What this phase's counter counts.
    pub fn unit(self) -> &'static str {
        match self {
            Phase::Story => "story",
            Phase::Years => "years",
            Phase::Months => "months",
            Phase::Days => "days",
        }
    }
}

/// One node in the tree, addressed.
///
/// Ordered so a sort puts a life in reading order, which the console's timeline
/// and the generator's fan-out both want.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(tag = "node", rename_all = "snake_case")]
pub enum NodeId {
    Story,
    Year { year: i32 },
    Month { year: i32, month: u32 },
    Day { year: i32, month: u32, day: u32 },
}

impl NodeId {
    pub fn phase(self) -> Phase {
        match self {
            NodeId::Story => Phase::Story,
            NodeId::Year { .. } => Phase::Years,
            NodeId::Month { .. } => Phase::Months,
            NodeId::Day { .. } => Phase::Days,
        }
    }

    /// The date key a life document made from this node is named with. `None`
    /// for the story, which is undated and therefore not a life document.
    pub fn date_key(self) -> Option<String> {
        match self {
            NodeId::Story => None,
            NodeId::Year { year } => Some(format!("{year:04}")),
            NodeId::Month { year, month } => Some(format!("{year:04}-{month:02}")),
            NodeId::Day { year, month, day } => Some(format!("{year:04}-{month:02}-{day:02}")),
        }
    }
}

impl Plan {
    /// A skeleton for a checked seed: every year and every month of the life
    /// laid out empty, and no days at all.
    ///
    /// **The coarse strata are laid out from the calendar, not from the model.**
    /// How many years a life has is arithmetic, and asking a model to enumerate
    /// them is inviting it to skip 2003. Days are the opposite — which days
    /// mattered is the one thing only the writing can decide — so none are
    /// created here.
    pub fn new(checked: &Checked) -> Plan {
        // **Laid out span by span, and only the written ones.** A dormant stretch produces no
        // years at all — not empty years, none — because there is nothing to write and a
        // character with an empty 1998 reads as a character whose 1998 failed rather than one
        // who was not there for it.
        //
        // Months follow the span's own grain: three centuries of vigil earns its years, the
        // two years of a training course earns every month.
        let mut years: Vec<Year> = Vec::new();
        for era in checked.eras.iter().filter(|e| e.kind.is_written()) {
            for year in era.years() {
                if years.iter().any(|y| y.year == year) {
                    continue;
                }
                let months = match era.grain {
                    Grain::Years => Vec::new(),
                    Grain::Months => months_in_year(year, era.from, era.to)
                        .into_iter()
                        .map(|month| Month {
                            month,
                            content: Content::default(),
                            days: Vec::new(),
                        })
                        .collect(),
                };
                years.push(Year {
                    year,
                    content: Content::default(),
                    months,
                });
            }
        }
        years.sort_by_key(|y| y.year);

        // The normalised spans are written back onto the seed, so a plan on disk always states
        // them explicitly — including the single lived span a plain human life implies. One
        // shape for the console, the prompts and the operator to read.
        let mut seed = checked.seed.clone();
        seed.eras = checked
            .eras
            .iter()
            .map(|e| Era {
                from: e.from.to_string(),
                to: e.to.to_string(),
                kind: e.kind,
                grain: e.grain,
                what: e.what.clone(),
            })
            .collect();

        Plan {
            seed,
            story: Story::default(),
            years,
        }
    }

    /// The span a year falls in, and whether anything is written there.
    pub fn era_for(&self, year: i32) -> Option<&Era> {
        self.seed.eras.iter().find(|e| {
            let (from, to) = (e.from.get(..4), e.to.get(..4));
            match (
                from.and_then(|s| s.parse::<i32>().ok()),
                to.and_then(|s| s.parse::<i32>().ok()),
            ) {
                (Some(a), Some(b)) => a <= year && year <= b,
                _ => false,
            }
        })
    }

    pub fn year(&self, year: i32) -> Option<&Year> {
        self.years.iter().find(|y| y.year == year)
    }

    pub fn year_mut(&mut self, year: i32) -> Option<&mut Year> {
        self.years.iter_mut().find(|y| y.year == year)
    }

    pub fn month(&self, year: i32, month: u32) -> Option<&Month> {
        self.year(year)?.months.iter().find(|m| m.month == month)
    }

    pub fn month_mut(&mut self, year: i32, month: u32) -> Option<&mut Month> {
        self.year_mut(year)?
            .months
            .iter_mut()
            .find(|m| m.month == month)
    }

    pub fn day_mut(&mut self, year: i32, month: u32, day: u32) -> Option<&mut Day> {
        self.month_mut(year, month)?
            .days
            .iter_mut()
            .find(|d| d.day == day)
    }

    pub fn content_mut(&mut self, id: NodeId) -> Option<&mut Content> {
        match id {
            NodeId::Story => Some(&mut self.story.content),
            NodeId::Year { year } => self.year_mut(year).map(|y| &mut y.content),
            NodeId::Month { year, month } => self.month_mut(year, month).map(|m| &mut m.content),
            NodeId::Day { year, month, day } => {
                self.day_mut(year, month, day).map(|d| &mut d.content)
            }
        }
    }

    /// Add a day to a month, or return the one already there.
    ///
    /// Days are inserted in date order so the month's list reads chronologically
    /// without a sort at every use.
    pub fn ensure_day(&mut self, year: i32, month: u32, day: u32) -> Option<&mut Day> {
        let m = self.month_mut(year, month)?;
        match m.days.iter().position(|d| d.day == day) {
            Some(i) => Some(&mut m.days[i]),
            None => {
                let at = m.days.partition_point(|d| d.day < day);
                m.days.insert(
                    at,
                    Day {
                        day,
                        ..Day::default()
                    },
                );
                Some(&mut m.days[at])
            }
        }
    }

    /// Everything strictly below `id`, marked stale.
    ///
    /// Called when a node is edited or regenerated. Returns how many nodes were
    /// marked, so the console can say "12 months and 4 days no longer follow
    /// from this" rather than silently changing colour.
    pub fn mark_stale_below(&mut self, id: NodeId) -> usize {
        let mut n = 0;
        let mut mark = |c: &mut Content| {
            // An ungenerated node is not stale, it is simply not written yet —
            // and an edited one is the operator's, so staleness (which invites
            // a regeneration) would be a lie about what is going to happen.
            if c.is_generated() && !c.edited && !c.stale {
                c.stale = true;
                n += 1;
            }
        };
        let (from_year, from_month) = match id {
            NodeId::Story => (None, None),
            NodeId::Year { year } => (Some(year), None),
            NodeId::Month { year, month } => (Some(year), Some((year, month))),
            // A day has nothing below it.
            NodeId::Day { .. } => return 0,
        };
        for y in &mut self.years {
            if from_year.is_some_and(|fy| y.year != fy) {
                continue;
            }
            if from_year.is_none() {
                mark(&mut y.content);
            }
            for m in &mut y.months {
                if let Some((_, fm)) = from_month {
                    if m.month != fm {
                        continue;
                    }
                } else {
                    mark(&mut m.content);
                }
                for d in &mut m.days {
                    mark(&mut d.content);
                }
            }
        }
        n
    }

    /// Which nodes of a phase want generating, in reading order.
    pub fn pending(&self, phase: Phase) -> Vec<NodeId> {
        let mut out = Vec::new();
        match phase {
            Phase::Story => {
                if self.story.content.wants_generation() {
                    out.push(NodeId::Story);
                }
            }
            Phase::Years => out.extend(
                self.years
                    .iter()
                    .filter(|y| y.content.wants_generation())
                    .map(|y| NodeId::Year { year: y.year }),
            ),
            Phase::Months => {
                for y in &self.years {
                    out.extend(
                        y.months
                            .iter()
                            .filter(|m| m.content.wants_generation())
                            .map(|m| NodeId::Month {
                                year: y.year,
                                month: m.month,
                            }),
                    );
                }
            }
            Phase::Days => {
                for y in &self.years {
                    for m in &y.months {
                        out.extend(m.days.iter().filter(|d| d.content.wants_generation()).map(
                            |d| NodeId::Day {
                                year: y.year,
                                month: m.month,
                                day: d.day,
                            },
                        ));
                    }
                }
            }
        }
        out
    }

    /// Every day of the life in order, with its consequences — what
    /// [`super::consequence::check_ordered`] walks.
    pub fn ordered_consequences(&self) -> Vec<(NodeId, &[Consequence])> {
        let mut out = Vec::new();
        for y in &self.years {
            for m in &y.months {
                for d in &m.days {
                    out.push((
                        NodeId::Day {
                            year: y.year,
                            month: m.month,
                            day: d.day,
                        },
                        d.consequences.as_slice(),
                    ));
                }
            }
        }
        out
    }

    /// Entities that exist without any day forming them — the seed's cast.
    pub fn seeded_entities(&self) -> Vec<String> {
        self.seed
            .cast
            .iter()
            .map(|c| c.entity_id.clone())
            .chain(
                self.story
                    .cast
                    .iter()
                    .filter(|c| c.from_seed)
                    .map(|c| c.entity_id.clone()),
            )
            .collect()
    }

    /// How many nodes of each phase are generated, for the console's summary.
    pub fn counts(&self) -> BTreeMap<&'static str, (usize, usize)> {
        let mut out = BTreeMap::new();
        out.insert("story", (usize::from(self.story.content.is_generated()), 1));
        out.insert(
            "years",
            (
                self.years
                    .iter()
                    .filter(|y| y.content.is_generated())
                    .count(),
                self.years.len(),
            ),
        );
        let months: Vec<&Month> = self.years.iter().flat_map(|y| y.months.iter()).collect();
        out.insert(
            "months",
            (
                months.iter().filter(|m| m.content.is_generated()).count(),
                months.len(),
            ),
        );
        let days: Vec<&Day> = months.iter().flat_map(|m| m.days.iter()).collect();
        out.insert(
            "days",
            (
                days.iter().filter(|d| d.content.is_generated()).count(),
                days.len(),
            ),
        );
        out
    }

    /// **The identity of the shared prefix every phase forks from.**
    ///
    /// A fork is only valid against the prompt it was planned for. Edit the
    /// story after a phase primed its prefix and the primed prompt still holds
    /// the old arc — every node generated against it would be built on a parent
    /// that no longer exists, plausibly and with nothing to say so. A job
    /// carries the hash it primed at and refuses to fan out against a different
    /// one; see [`super::generate`].
    ///
    /// # Why this covers the story and nothing below it
    ///
    /// Only what actually reaches the *shared prefix* is hashed, and the shared
    /// prefix is the same for every phase: the seed, the arc, the cast and the
    /// year outline. A node's own parent text — the year a month expands, the
    /// month a day expands — travels in that fork's **turn**, which is built
    /// from the plan at fan-out time and is therefore never stale.
    ///
    /// That is what keeps an edit local. Correcting one year invalidates its own
    /// months through [`Self::mark_stale_below`] and leaves every other month in
    /// the life alone, where a coarser hash would force the whole phase to
    /// regenerate over an edit to one line.
    pub fn prefix_hash(&self) -> String {
        let mut h = Sha256::new();
        h.update(serde_json::to_vec(&self.seed).unwrap_or_default());
        h.update(self.story.content.text.as_bytes());
        h.update(serde_json::to_vec(&self.story.cast).unwrap_or_default());
        h.update(serde_json::to_vec(&self.story.outline).unwrap_or_default());
        format!("{:x}", h.finalize())
    }
}

/// Is this a character id that may be turned into a path?
///
/// **`who` becomes a directory name and a filename.** It arrives from a URL, so
/// an unchecked one is a path traversal: `../../etc/passwd` would have this
/// module reading and writing wherever it pointed. The API also resolves `who`
/// against the personality registry, which is the stronger check — but the
/// stronger check lives in a different file, and a path built here must be safe
/// on its own terms rather than because of what some caller happened to do
/// first.
///
/// Deliberately narrow: lowercase ASCII, digits, hyphen and underscore. Every
/// authored personality id is already of that shape, so nothing legitimate is
/// refused, and everything that makes a path interesting — separators, `.`,
/// drive letters, NUL, non-ASCII that normalises unpredictably — is gone.
pub fn is_safe_id(who: &str) -> bool {
    !who.is_empty()
        && who.len() <= 64
        && who
            .bytes()
            .all(|b| b.is_ascii_lowercase() || b.is_ascii_digit() || b == b'-' || b == b'_')
}

/// Where a plan is stored: `<mind>/.lifegen/<who>.json`.
///
/// A dot-directory outside `layers/`, for two independent reasons — the walker
/// admits only `.md`/`.yaml`, so JSON is invisible to it anyway, and a file
/// inside a life directory would be reported by
/// [`crate::engine::life::episodes`] as an undated document on every load.
///
/// `None` for an id that must not become a path; see [`is_safe_id`].
pub fn path(mind: &Path, who: &str) -> Option<PathBuf> {
    is_safe_id(who).then(|| mind.join(".lifegen").join(format!("{who}.json")))
}

fn unsafe_id(who: &str) -> anyhow::Error {
    anyhow::anyhow!("`{who}` is not a character id this daemon will build a path from")
}

/// Read a plan, or `None` when there is not one yet.
pub fn load(mind: &Path, who: &str) -> anyhow::Result<Option<Plan>> {
    let p = path(mind, who).ok_or_else(|| unsafe_id(who))?;
    match std::fs::read(&p) {
        Ok(bytes) => Ok(Some(serde_json::from_slice(&bytes)?)),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(None),
        Err(e) => Err(e.into()),
    }
}

/// Write a plan, atomically.
///
/// Through a temporary and a rename, because the alternative is a truncated
/// plan when the daemon dies mid-write — and a plan is the only record of what
/// an operator edited by hand. The documents can be regenerated; their
/// corrections cannot.
pub fn save(mind: &Path, plan: &Plan) -> anyhow::Result<()> {
    let who = &plan.seed.who;
    let p = path(mind, who).ok_or_else(|| unsafe_id(who))?;
    if let Some(dir) = p.parent() {
        std::fs::create_dir_all(dir)?;
    }
    let tmp = p.with_extension("json.tmp");
    std::fs::write(&tmp, serde_json::to_vec_pretty(plan)?)?;
    std::fs::rename(&tmp, &p)?;
    Ok(())
}

/// The date a node falls on, as a real calendar date — used to check a day
/// actually exists in its month before a document is named for it.
pub fn day_date(year: i32, month: u32, day: u32) -> Option<Date> {
    Date::parse(&format!("{year:04}-{month:02}-{day:02}")).ok()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lifegen::seed::{check, Cadence, Seed};

    fn seed(born: &str, through: &str) -> Seed {
        Seed {
            who: "cindy-tan".into(),
            display: "Cindy Tan".into(),
            born: born.into(),
            through: through.into(),
            place: "Nanyang".into(),
            role: "a records clerk".into(),
            cadence: Cadence::Even,
            facts: Vec::new(),
            world: Vec::new(),
            cast: Vec::new(),
            eras: Vec::new(),
        }
    }

    fn plan() -> Plan {
        Plan::new(&check(&seed("1998-09-14", "2001-03-02")).unwrap())
    }

    /// **The skeleton comes from the calendar, not from a model.** How many
    /// years a life has is arithmetic; asking a model to enumerate them invites
    /// it to skip one.
    #[test]
    fn a_new_plan_lays_out_every_year_and_month_clipped_to_the_life() {
        let p = plan();
        assert_eq!(
            p.years.iter().map(|y| y.year).collect::<Vec<_>>(),
            vec![1998, 1999, 2000, 2001]
        );
        assert_eq!(
            p.year(1998)
                .unwrap()
                .months
                .iter()
                .map(|m| m.month)
                .collect::<Vec<_>>(),
            vec![9, 10, 11, 12],
            "the birth year starts at the birth month"
        );
        assert_eq!(p.year(1999).unwrap().months.len(), 12);
        assert_eq!(
            p.year(2001)
                .unwrap()
                .months
                .iter()
                .map(|m| m.month)
                .collect::<Vec<_>>(),
            vec![1, 2, 3],
            "the final year stops where the authored life does"
        );
    }

    /// **A dormant span produces no years at all** — not empty ones. A character with an empty
    /// 3000 reads as a character whose 3000 failed to generate; a character with no 3000 reads
    /// as one who was not there for it, which is the truth.
    #[test]
    fn a_dormant_span_lays_out_nothing_and_a_coarse_one_lays_out_no_months() {
        use crate::lifegen::seed::{Era, EraKind, Grain};
        let mut s = seed("2461-01-01", "3087-01-01");
        s.eras = vec![
            Era {
                from: "2461-01-01".into(),
                to: "2463-12-31".into(),
                kind: EraKind::Lived,
                grain: Grain::Months,
                what: "Commissioned.".into(),
            },
            Era {
                from: "2464-01-01".into(),
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
        let p = Plan::new(&check(&s).unwrap());

        // Nothing at all for the 295 dormant years.
        assert!(
            p.year(3000).is_none(),
            "a dormant year exists as a document"
        );
        assert!(p.year(2793).is_none());
        assert_eq!(p.years.last().unwrap().year, 2792);

        // The month-grain span has its months; the year-grain span has none.
        assert_eq!(p.year(2461).unwrap().months.len(), 12);
        assert!(
            p.year(2500).unwrap().months.is_empty(),
            "year grain wrote months"
        );
        assert_eq!(p.years.len(), 3 + 329);

        // And the normalised spans are on the saved plan, so the console and the prompts read
        // one shape rather than inferring the simple case.
        assert_eq!(p.seed.eras.len(), 3);
        assert_eq!(p.era_for(3000).unwrap().kind, EraKind::Dormant);
        assert_eq!(p.era_for(2500).unwrap().grain, Grain::Years);
    }

    /// **No days are laid out.** Which days mattered is the one thing only the
    /// writing can decide.
    #[test]
    fn a_new_plan_has_no_days() {
        assert!(plan()
            .years
            .iter()
            .flat_map(|y| &y.months)
            .all(|m| m.days.is_empty()));
    }

    #[test]
    fn every_ungenerated_node_of_a_phase_is_pending() {
        let p = plan();
        assert_eq!(p.pending(Phase::Story), vec![NodeId::Story]);
        assert_eq!(p.pending(Phase::Years).len(), 4);
        assert_eq!(p.pending(Phase::Months).len(), 4 + 12 + 12 + 3);
        assert!(p.pending(Phase::Days).is_empty(), "no days exist yet");
    }

    #[test]
    fn a_generated_node_stops_being_pending() {
        let mut p = plan();
        p.content_mut(NodeId::Year { year: 1999 })
            .unwrap()
            .generated("Nineteen".into(), "It rained.".into());
        assert_eq!(
            p.pending(Phase::Years),
            vec![
                NodeId::Year { year: 1998 },
                NodeId::Year { year: 2000 },
                NodeId::Year { year: 2001 }
            ]
        );
    }

    /// **The sticky rule, in the one place it is decided.** A tool that eats
    /// your work when you fix a typo two levels up is one you stop using.
    #[test]
    fn an_edited_node_is_never_pending_even_when_stale() {
        let mut c = Content::default();
        c.edit("Mine".into(), "I wrote this.".into());
        assert!(!c.wants_generation());
        c.stale = true;
        assert!(!c.wants_generation(), "an edit outranks staleness");
    }

    #[test]
    fn a_stale_generated_node_is_pending_again() {
        let mut c = Content::default();
        c.generated("t".into(), "text".into());
        assert!(!c.wants_generation());
        c.stale = true;
        assert!(c.wants_generation());
    }

    /// Editing a year invalidates its own months and days — and nothing in any
    /// other year.
    #[test]
    fn editing_a_year_marks_only_its_own_subtree_stale() {
        let mut p = plan();
        for y in [1998, 1999] {
            p.content_mut(NodeId::Year { year: y })
                .unwrap()
                .generated("y".into(), "t".into());
            for m in p
                .year(y)
                .unwrap()
                .months
                .iter()
                .map(|m| m.month)
                .collect::<Vec<_>>()
            {
                p.content_mut(NodeId::Month { year: y, month: m })
                    .unwrap()
                    .generated("m".into(), "t".into());
            }
        }
        let n = p.mark_stale_below(NodeId::Year { year: 1998 });
        assert_eq!(n, 4, "1998 has four months in this life");
        assert!(p.month(1998, 9).unwrap().content.stale);
        assert!(
            !p.month(1999, 1).unwrap().content.stale,
            "another year is untouched"
        );
        assert!(
            !p.year(1998).unwrap().content.stale,
            "the node itself is not its own descendant"
        );
    }

    /// An edited descendant survives an ancestor's change — it is not marked
    /// stale, because staleness invites a regeneration that would discard it.
    #[test]
    fn marking_stale_skips_edited_descendants() {
        let mut p = plan();
        p.content_mut(NodeId::Month {
            year: 1998,
            month: 9,
        })
        .unwrap()
        .edit("Mine".into(), "I wrote this.".into());
        p.content_mut(NodeId::Month {
            year: 1998,
            month: 10,
        })
        .unwrap()
        .generated("m".into(), "t".into());
        assert_eq!(p.mark_stale_below(NodeId::Year { year: 1998 }), 1);
        assert!(!p.month(1998, 9).unwrap().content.stale);
        assert!(p.month(1998, 10).unwrap().content.stale);
    }

    /// An ungenerated node is not stale — it is simply not written yet, and
    /// colouring it as invalidated would be a lie about what is going to happen.
    #[test]
    fn marking_stale_skips_nodes_that_were_never_generated() {
        let mut p = plan();
        assert_eq!(p.mark_stale_below(NodeId::Story), 0);
    }

    #[test]
    fn editing_the_story_marks_the_whole_tree_stale() {
        let mut p = plan();
        p.content_mut(NodeId::Year { year: 1998 })
            .unwrap()
            .generated("y".into(), "t".into());
        p.content_mut(NodeId::Month {
            year: 2000,
            month: 5,
        })
        .unwrap()
        .generated("m".into(), "t".into());
        assert_eq!(p.mark_stale_below(NodeId::Story), 2);
    }

    /// A day has nothing below it, so marking is a no-op rather than a walk.
    #[test]
    fn a_day_has_no_subtree() {
        let mut p = plan();
        p.ensure_day(1998, 9, 14).unwrap();
        assert_eq!(
            p.mark_stale_below(NodeId::Day {
                year: 1998,
                month: 9,
                day: 14
            }),
            0
        );
    }

    #[test]
    fn days_are_inserted_in_date_order_and_never_duplicated() {
        let mut p = plan();
        for d in [30, 14, 21, 14] {
            p.ensure_day(1998, 9, d).unwrap();
        }
        assert_eq!(
            p.month(1998, 9)
                .unwrap()
                .days
                .iter()
                .map(|d| d.day)
                .collect::<Vec<_>>(),
            vec![14, 21, 30]
        );
    }

    #[test]
    fn a_node_id_names_the_date_its_document_is_written_at() {
        assert_eq!(NodeId::Story.date_key(), None);
        assert_eq!(NodeId::Year { year: 1998 }.date_key().unwrap(), "1998");
        assert_eq!(
            NodeId::Month {
                year: 1998,
                month: 9
            }
            .date_key()
            .unwrap(),
            "1998-09"
        );
        assert_eq!(
            NodeId::Day {
                year: 1998,
                month: 9,
                day: 4
            }
            .date_key()
            .unwrap(),
            "1998-09-04"
        );
    }

    /// **The hash a fork is checked against.** Everything in the shared prefix
    /// invalidates it: the seed, the arc, the cast and the year outline.
    #[test]
    fn everything_in_the_shared_prefix_changes_its_hash() {
        let mut p = plan();
        let start = p.prefix_hash();

        p.story
            .content
            .edit("Arc".into(), "A different life.".into());
        let after_story = p.prefix_hash();
        assert_ne!(start, after_story, "the arc is in the prefix");

        p.story.cast.push(CastMember {
            entity_id: "lim".into(),
            display: "Professor Lim".into(),
            what: String::new(),
            npc_id: None,
            from_seed: false,
        });
        let after_cast = p.prefix_hash();
        assert_ne!(after_story, after_cast, "the cast is in the prefix");

        p.story.outline.push(YearBeat {
            year: 1999,
            title: "The Long Year".into(),
            premise: "Nothing arrives.".into(),
        });
        let after_outline = p.prefix_hash();
        assert_ne!(after_cast, after_outline, "the outline is in the prefix");

        p.seed.place = "Somewhere else".into();
        assert_ne!(after_outline, p.prefix_hash(), "the seed is in the prefix");
    }

    /// **An edit below the story must stay local.** A year's own text travels
    /// in its months' *turns*, not in the shared prefix, so correcting one year
    /// invalidates that year's months and leaves every other month in the life
    /// alone. A coarser hash would force a whole phase to regenerate over one
    /// corrected line.
    #[test]
    fn editing_a_year_leaves_the_shared_prefix_intact_and_marks_only_its_own_months() {
        let mut p = plan();
        for y in [1998, 1999] {
            for m in p
                .year(y)
                .unwrap()
                .months
                .iter()
                .map(|m| m.month)
                .collect::<Vec<_>>()
            {
                p.content_mut(NodeId::Month { year: y, month: m })
                    .unwrap()
                    .generated("m".into(), "t".into());
            }
        }
        let before = p.prefix_hash();
        p.content_mut(NodeId::Year { year: 1999 })
            .unwrap()
            .edit("Nineteen".into(), "Everything changed.".into());
        p.mark_stale_below(NodeId::Year { year: 1999 });

        assert_eq!(
            before,
            p.prefix_hash(),
            "a year's prose is not in the prefix"
        );
        assert!(p.month(1999, 1).unwrap().content.stale);
        assert!(
            !p.month(1998, 9).unwrap().content.stale,
            "another year is untouched"
        );
    }

    /// The story's beat for a year is what the year phase expands.
    #[test]
    fn the_outline_is_looked_up_by_year() {
        let mut p = plan();
        p.story.outline.push(YearBeat {
            year: 2000,
            title: "The Fire".into(),
            premise: "The granary goes.".into(),
        });
        assert_eq!(p.story.beat(2000).unwrap().title, "The Fire");
        assert!(p.story.beat(1998).is_none());
    }

    #[test]
    fn counts_report_generated_over_total_per_phase() {
        let mut p = plan();
        p.content_mut(NodeId::Year { year: 1998 })
            .unwrap()
            .generated("y".into(), "t".into());
        let c = p.counts();
        assert_eq!(c["story"], (0, 1));
        assert_eq!(c["years"], (1, 4));
        assert_eq!(c["months"], (0, 31));
        assert_eq!(c["days"], (0, 0));
    }

    #[test]
    fn a_plan_round_trips_through_json() {
        let mut p = plan();
        p.ensure_day(1998, 9, 14)
            .unwrap()
            .content
            .generated("D".into(), "text".into());
        p.story.cast.push(CastMember {
            entity_id: "lim".into(),
            display: "Professor Lim".into(),
            what: "taught her".into(),
            npc_id: Some(3),
            from_seed: false,
        });
        let back: Plan = serde_json::from_str(&serde_json::to_string(&p).unwrap()).unwrap();
        assert_eq!(p, back);
    }

    #[test]
    fn a_plan_saves_and_loads_from_a_dot_directory_outside_the_layers() {
        let mind = std::env::temp_dir().join(format!("npcd-plan-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&mind);
        std::fs::create_dir_all(&mind).unwrap();

        assert!(
            load(&mind, "cindy-tan").unwrap().is_none(),
            "absent is ordinary"
        );
        let p = plan();
        save(&mind, &p).unwrap();
        assert_eq!(load(&mind, "cindy-tan").unwrap().unwrap(), p);

        // Outside `layers/`, and not a file the ingest walker would ever admit.
        let stored = path(&mind, "cindy-tan").unwrap();
        assert!(stored.starts_with(mind.join(".lifegen")));
        assert!(!crate::engine::watcher::is_ingestible(&stored));
        let _ = std::fs::remove_dir_all(&mind);
    }

    #[test]
    fn ordered_consequences_walk_the_life_in_date_order() {
        let mut p = plan();
        p.ensure_day(2000, 5, 2).unwrap();
        p.ensure_day(1998, 9, 14).unwrap();
        p.ensure_day(1998, 12, 1).unwrap();
        assert_eq!(
            p.ordered_consequences()
                .iter()
                .map(|(id, _)| *id)
                .collect::<Vec<_>>(),
            vec![
                NodeId::Day {
                    year: 1998,
                    month: 9,
                    day: 14
                },
                NodeId::Day {
                    year: 1998,
                    month: 12,
                    day: 1
                },
                NodeId::Day {
                    year: 2000,
                    month: 5,
                    day: 2
                },
            ]
        );
    }

    /// Every phase must name itself, or the progress overlay renders a blank
    /// line where a phase should be.
    #[test]
    fn every_phase_has_a_label_and_a_unit() {
        for f in Phase::ALL {
            assert!(!f.label().is_empty());
            assert!(!f.unit().is_empty());
        }
        assert_eq!(Phase::ALL.len(), 4, "a phase was added without a decision");
        // The ladder's order is a dependency, so it must be the sort order too.
        let mut sorted = Phase::ALL.to_vec();
        sorted.sort();
        assert_eq!(sorted, Phase::ALL);
    }
}
