//! Diagnostics: a structured budget snapshot that renders as a table and that
//! unit tests assert against — the same view a human debugs with.

use super::{AllocClass, Criticality, VramGovernor};

/// One per-class row of the budget table (loose reserved tally).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BudgetRow {
    pub class: AllocClass,
    pub reserved: u64,
}

/// A structured snapshot of the governor's budget state. Returned by
/// [`VramGovernor::budget_table`] for assertions and rendered by
/// [`VramGovernor::log_budget`].
#[derive(Clone, Debug)]
pub struct BudgetTable {
    /// Balloon-measured resident capacity `C` (0 = not yet measured).
    pub capacity_c: u64,
    /// Total device VRAM (from the last measurement).
    pub total: u64,
    /// Live headroom (the source of truth).
    pub headroom: u64,
    /// Which backend produced the reading.
    pub source: super::ProbeKind,
    /// Loose per-class reserved tallies.
    pub rows: [BudgetRow; AllocClass::COUNT],
    /// KV floor (`3 GiB + 15% × (C − Weights)`).
    pub kv_floor: u64,
    /// The five ladder trip points, indexed by [`Criticality`].
    pub thresholds: [u64; 5],
    /// Reversibly-evictable KV (up to `Moderate`) — the forecast input.
    pub evictable_reversible: u64,
    /// The last relief episode `(deepest tier, bytes freed)`.
    pub last_relief: Option<(Criticality, u64)>,
}

impl BudgetTable {
    /// The reserved tally for a class.
    pub fn reserved(&self, class: AllocClass) -> u64 {
        self.rows[class.idx()].reserved
    }
}

fn mib(b: u64) -> u64 {
    b / (1024 * 1024)
}

impl VramGovernor {
    /// A structured snapshot of the current budget state.
    pub fn budget_table(&self) -> BudgetTable {
        let reading = self.measure_or_default();
        let rows = [
            BudgetRow {
                class: AllocClass::Weights,
                reserved: self.class_reserved(AllocClass::Weights),
            },
            BudgetRow {
                class: AllocClass::Expert,
                reserved: self.class_reserved(AllocClass::Expert),
            },
            BudgetRow {
                class: AllocClass::Scratch,
                reserved: self.class_reserved(AllocClass::Scratch),
            },
            BudgetRow {
                class: AllocClass::Kv,
                reserved: self.class_reserved(AllocClass::Kv),
            },
        ];
        let thresholds = [
            self.tier_threshold(Criticality::Trivial),
            self.tier_threshold(Criticality::Cheap),
            self.tier_threshold(Criticality::Moderate),
            self.tier_threshold(Criticality::Costly),
            self.tier_threshold(Criticality::Critical),
        ];
        BudgetTable {
            capacity_c: self.capacity(),
            total: reading.total,
            headroom: reading.headroom,
            source: reading.source,
            rows,
            kv_floor: self.kv_floor(),
            thresholds,
            evictable_reversible: self.evictable_estimate(Criticality::Moderate),
            last_relief: self.last_relief(),
        }
    }

    /// Render the budget table into a multi-line string (for logs / tests).
    pub fn render_budget(&self, whence: &str) -> String {
        let t = self.budget_table();
        let mut s = String::new();
        s.push_str(&format!(
            "vram budget [{whence}] source={:?} C={}MiB total={}MiB headroom={}MiB kv_floor={}MiB\n",
            t.source,
            mib(t.capacity_c),
            mib(t.total),
            mib(t.headroom),
            mib(t.kv_floor),
        ));
        for row in &t.rows {
            s.push_str(&format!(
                "  {:<8?} reserved={}MiB\n",
                row.class,
                mib(row.reserved)
            ));
        }
        s.push_str(&format!(
            "  thresholds MiB: Trivial={} Cheap={} Moderate={} Costly={} Critical={}\n",
            mib(t.thresholds[0]),
            mib(t.thresholds[1]),
            mib(t.thresholds[2]),
            mib(t.thresholds[3]),
            mib(t.thresholds[4]),
        ));
        s.push_str(&format!(
            "  evictable(≤Moderate)={}MiB last_relief={:?}",
            mib(t.evictable_reversible),
            t.last_relief.map(|(tier, b)| (tier, mib(b)))
        ));
        s
    }

    /// Log the budget table at INFO on the `candle_core::vram` target.
    pub fn log_budget(&self, whence: &str) {
        tracing::info!(target: "candle_core::vram", "{}", self.render_budget(whence));
    }
}
