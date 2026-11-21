//! One group of a layer's turns, and the rules that decide which of them a
//! projection may select.
//!
//! A group carries three gates that answer different questions, so they stay
//! separate: `score_threshold` gates selection on fresh per-turn scores, a
//! declared policy band gates the accumulated belief those scores feed, and an
//! absent value falls through to the next source instead of becoming a default
//! that admits everything. A drawing group takes the units another group owns
//! under its paths, and a unit is placed through one group or the other, never
//! both. A working-set unit fades with age: a file name that opens with a
//! `YYYY-MM-DD-HHMM` stamp has its score scaled by how old it is, and a name with
//! no stamp is not scaled.

use super::*;

/// Schema for one group within a layer.
#[derive(Debug, Clone)]
pub struct GroupSchema {
    /// Crate-assigned id, **globally unique** across all layers in the schema.
    pub id: GroupId,
    /// Original YAML `id:` string.
    pub name: String,
    /// Which turns survive into the projection.
    pub selection: SelectionRule,
    /// Turns whose score is below this threshold are invisible to selection.
    /// `None` when the YAML omits it — distinct from an explicit `0.0`, because
    /// this value doubles as the belief gate in [`Self::belief_config`]: a group
    /// that declares no threshold must fall through to its `policy`'s band
    /// rather than have it overwritten by a default that admits everything.
    pub score_threshold: Option<f32>,
    /// Selection policy (belief-update + budget + tag scope) for this group,
    /// resolved from the group's `policy:` or inherited from the enclosing layer.
    pub policy: SelectionPolicy,
    /// Whether the score band in `policy` was **explicitly declared** (a
    /// `min_score:` in the group's or its layer's `policy:` block) rather than
    /// inherited from the schema default. Only a declared band may stand in for
    /// an absent `score_threshold` in [`Self::belief_config`] — the default
    /// preset is a tool-scope policy whose 800 bar would silently gate every
    /// turn group that simply omitted both fields.
    pub policy_band_declared: bool,
    pub budget: Budget,
    /// Concept B: mass-driven member-budget extension. `None` = static budget.
    pub budget_adaptive: Option<MemberBudgetAdaptive>,
    /// Concept C: a hit drags its timeline neighbors into contention. `None` =
    /// no locality.
    pub locality: Option<LocalityConfig>,
    /// Concept D: a timeline member that rides along whenever any of its
    /// exchanges is selected (e.g. `first` = the file-header exchange). `None`
    /// = no anchor.
    pub anchor: Option<AnchorConfig>,
    /// The most conversations a mid-decode reprojection may bring into this
    /// group's selection (`docs/projection_swap_cap.md`); the rest wait for the
    /// next. Belief-driven groups only — the working set fades by its own
    /// momentum, and section collections (the tool catalog) are not capped.
    /// Never zero: the smallest cap still admits one whole conversation.
    pub max_swaps: NonZeroUsize,
    /// The units of another group of this layer that this group places
    /// instead (`docs/guidance_layer.md` §1). `None` for a group that places
    /// only its own.
    pub draws: Option<Draws>,
    /// How a working-set group's units fade with age (`docs/guidance_layer.md`
    /// §4). `None` for a group whose units do not age.
    pub recency: Option<Recency>,
}

/// What a drawing group takes from the group that owns the units: the ones
/// under its paths. A unit is placed through one group or the other, never
/// both. The owner holds every unit as it always has; which of them the
/// drawing group takes is decided per target by its retrieval scope.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Draws {
    /// The group whose conversations hold the units.
    pub from: GroupId,
    /// Workspace-relative path prefixes — `candle/mcp/src/guides/`,
    /// `journals/` — a unit's file must lie under one of.
    pub paths: Vec<String>,
    /// File names never drawn wherever they lie under `paths` — the hub
    /// files (`README.md`, `group.yaml`) a folder carries beside its entries,
    /// which match every question and answer none. They stay with the owner.
    pub except: Vec<String>,
}

/// How a working-set group's units fade as they age (`docs/guidance_layer.md`
/// §4): a unit whose file name opens with a `YYYY-MM-DD-HHMM` UTC stamp has
/// its fresh score scaled by `0.5^(age_days / half_life_days)` before it feeds
/// momentum. A unit whose name carries no stamp is not scaled.
#[derive(Debug, Clone, PartialEq)]
pub struct Recency {
    pub half_life_days: f32,
    /// The unit metadata that holds its path.
    pub dated_by: String,
}

pub(super) const DAY_MS: f64 = 86_400_000.0;

impl Recency {
    /// The factor a unit at `path` is scaled by at `now_ms`: `1` for a name
    /// with no stamp, and for one stamped after `now_ms`.
    pub fn factor(&self, path: &str, now_ms: u64) -> f32 {
        let name = path.rsplit('/').next().unwrap_or(path);
        let Some(stamped) = stamp_ms(name) else {
            return 1.0;
        };
        let age_days = now_ms.saturating_sub(stamped) as f64 / DAY_MS;
        0.5f64.powf(age_days / f64::from(self.half_life_days)) as f32
    }
}

/// The UTC instant a `YYYY-MM-DD-HHMM` stamp opening `name` names, in Unix
/// milliseconds; `None` for a name that does not open with one.
pub(super) fn stamp_ms(name: &str) -> Option<u64> {
    let s = name.get(..15)?;
    let b = s.as_bytes();
    if [4, 7, 10].iter().any(|&i| b[i] != b'-') {
        return None;
    }
    let num = |r: Range<usize>| -> Option<i64> {
        let t = &s[r];
        t.bytes()
            .all(|c| c.is_ascii_digit())
            .then(|| t.parse().ok())?
    };
    let (y, m, d) = (num(0..4)?, num(5..7)?, num(8..10)?);
    let (hh, mm) = (num(11..13)?, num(13..15)?);
    if !(1..=12).contains(&m) || !(1..=31).contains(&d) || hh > 23 || mm > 59 {
        return None;
    }
    let days = days_from_civil(y, m, d);
    let ms = ((days * 24 + hh) * 60 + mm) * 60_000;
    u64::try_from(ms).ok()
}

/// Days from 1970-01-01 to the proleptic Gregorian date `y-m-d`.
pub(super) fn days_from_civil(y: i64, m: i64, d: i64) -> i64 {
    let y = if m <= 2 { y - 1 } else { y };
    let era = y.div_euclid(400);
    let yoe = y - era * 400;
    let mp = (m + 9) % 12;
    let doy = (153 * mp + 2) / 5 + d - 1;
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    era * 146_097 + doe - 719_468
}

impl Draws {
    /// Whether the file at workspace-relative `path` is one this group draws.
    pub fn takes(&self, path: &str) -> bool {
        let name = path.rsplit('/').next().unwrap_or(path);
        self.paths
            .iter()
            .any(|prefix| path.starts_with(prefix.as_str()))
            && !self.except.iter().any(|hub| hub == name)
    }
}

impl GroupSchema {
    /// Whether this group's turns are selected by the provenance belief
    /// mechanism (RelLeak scores + budget) rather than pure recency.
    /// `Sequence` is intrinsically a recency rule (recent-N inviolate); every
    /// other rule ranks by score and so becomes belief-driven once turns carry
    /// a fresh wide-Q score.
    pub fn is_belief_driven(&self) -> bool {
        !matches!(
            self.selection,
            SelectionRule::Sequence { .. } | SelectionRule::WorkingSet { .. }
        )
    }

    /// Whether this group is filled from the projection target's working set.
    ///
    /// Not belief-driven — its selection is the working set's, not the RelLeak
    /// band's — but still **scanned**: the belief scan's fresh per-file scores
    /// are what the working set's momentum is built from.
    pub fn is_working_set(&self) -> bool {
        matches!(self.selection, SelectionRule::WorkingSet { .. })
    }

    /// Whether the belief scan scores this group's turns.
    pub fn is_scanned(&self) -> bool {
        self.is_belief_driven() || self.is_working_set()
    }

    /// The belief [`PolicyConfig`] for a belief-driven group: the group's policy
    /// (β leak rate, inherited) with the **budget** taken from the selection
    /// *rule* — `top_k(k)` → at most `k`, `single` → 1, `always_visible` → all
    /// `n_candidates` — so the belief path's surviving set matches what
    /// `apply_selection` would rank, just carried across reprojections.
    /// `n_candidates` bounds the unbounded `always_visible` case.
    ///
    /// The score gates resolve in this order:
    ///
    /// 1. **`score_threshold: 0.0`** — an explicit zero is a deliberate "no
    ///    gate", not an absence, and disables the gate even against a declared
    ///    band.
    /// 2. **A declared `policy:` band** — the belief-specific knob, so it
    ///    governs the belief gate.
    /// 3. **A declared `score_threshold`** — the gate for the vast majority of
    ///    groups, which declare a threshold and no band.
    /// 4. **Neither** — no gate. An *inherited* default band must not become
    ///    one; it is a tool-scope preset rather than this group's intent.
    ///
    /// # Why a band outranks a threshold
    ///
    /// The two gate different things. `score_threshold` gates SELECTION on
    /// fresh per-turn scores — it is what stops a zero-evidence group (fork or
    /// startup, before any probe) from seating arbitrary turns on a score tie.
    /// A declared band gates ACCUMULATED belief, which is what this config
    /// feeds.
    ///
    /// `repo_map/structure` is the only group in the bundled schema that
    /// declares both, and it declares them for exactly that reason — its
    /// `score_threshold: 1.0` comment says "the live belief loop is untouched
    /// (its own min/evict of 250 governs there)". When the threshold outranked
    /// the band, adding that line silently dropped the group's belief gate from
    /// 250 to 1.0 — the entire accumulated-evidence range — which
    /// `bundled_schema_belief_gates_resolve_from_the_right_source` has caught
    /// since `4e762163` introduced it.
    pub fn belief_config(&self, n_candidates: usize) -> PolicyConfig {
        let mut cfg = self.policy.config;
        cfg.budget_min = 0;
        cfg.budget_max = match &self.selection {
            SelectionRule::TopK { k } | SelectionRule::Offered { k, .. } => *k,
            SelectionRule::Single => 1,
            SelectionRule::AlwaysVisible => n_candidates.max(1),
            // Named/Sequence aren't belief-driven, and a working-set group is
            // sized by its target's budget rather than a member count; fall back
            // to the policy budget.
            SelectionRule::Named { .. }
            | SelectionRule::Sequence { .. }
            | SelectionRule::WorkingSet { .. } => self.policy.config.budget_max,
        };
        match (self.score_threshold, self.policy_band_declared) {
            // An explicit zero disables the gate outright, band or no band.
            (Some(0.0), _) => {
                cfg.min_score = 0.0;
                cfg.evict_score = 0.0;
            }
            // A declared band is the belief gate, with or without a threshold.
            (_, true) => {}
            // No band: the group's threshold is its gate.
            (Some(threshold), false) => {
                cfg.min_score = threshold;
                cfg.evict_score = threshold;
            }
            // Neither declared: no gate. (An inherited default band must not
            // become one; it is a tool-scope preset, not this group's intent.)
            (None, false) => {
                cfg.min_score = 0.0;
                cfg.evict_score = 0.0;
            }
        }
        cfg
    }
}

/// Flexbox-style token budget descriptor.
///
/// All percentages are **of the parent's resolved token budget** at
/// projection time, not of the global window. A min_percent of 30 on a group
/// inside a layer that received 1000 tokens means "at least 300 tokens for
/// this group."
///
/// # Reconciliation rules
///
/// - **`priority`** — relative weight when distributing the parent's budget.
///   Higher = larger share of remainder. Must be > 0.
/// - **`min_percent`** — floor. The flexbox distributor reserves at least
///   this fraction (subject to dynamic-shortfall proportional shrink if
///   sibling mins exceed 100%).
/// - **`max_percent`** — ceiling. Once a node hits its max, it's saturated
///   and excess budget redistributes to unsaturated siblings.
#[derive(Debug, Clone)]
pub struct Budget {
    /// Relative weight for proportional allocation. Must be > 0.
    pub priority: f32,
    /// Floor as a percent of parent budget (0–100).
    pub min_percent: Option<f32>,
    /// Ceiling as a percent of parent budget (0–100).
    pub max_percent: Option<f32>,
    /// Concept B: mass-modulated priority within the declared rails. `None` =
    /// static budget (today's behaviour).
    pub adaptive: Option<BudgetAdaptive>,
}

impl Default for Budget {
    fn default() -> Self {
        Self {
            priority: 50.0,
            min_percent: None,
            max_percent: None,
            adaptive: None,
        }
    }
}

/// The score an inherited lineage turn is never ranked below — one
/// full-strength hit, the top of the normalized band a reprojection scores on.
pub const LINEAGE_BASE: f32 = 1000.0;
