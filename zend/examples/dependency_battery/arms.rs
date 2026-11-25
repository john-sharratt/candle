//! The battery's arms (`docs/unbounded_agents.md` §9.12, *Arms*). The seven
//! memory arms are each a setting of the turn's three retrieval dials, sent as
//! the chat request's `seed`, `projection` and `reprojection` fields. The
//! baselines project nothing: R1 is handed the files a BM25 ranking of the
//! repository retrieves for the question (`rag/`), and T1 may call the
//! daemon's tools to read the repository itself. P1 is handed, the same way
//! and under the same budget, the files provenance selection placed when C1
//! asked the question (`selected.rs`): the two differ only in who chose them.

use serde::Serialize;
use serde_json::{json, Value};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize)]
pub enum Arm {
    A0,
    A1,
    D1,
    B0,
    B1,
    C0,
    C1,
    R1,
    P1,
    T1,
}

/// Every arm, in the order the paper tables them.
pub const ARMS: [Arm; 10] = [
    Arm::A0,
    Arm::A1,
    Arm::D1,
    Arm::B0,
    Arm::B1,
    Arm::C0,
    Arm::C1,
    Arm::R1,
    Arm::P1,
    Arm::T1,
];

/// One contrast: `(name, arm, against)`, tested as `arm − against`.
pub type Contrast = (&'static str, Arm, Arm);

/// The five contrasts the paper pre-registers: its primary family, Holm-
/// corrected across the five.
pub const CONTRASTS: [Contrast; 5] = [
    ("The memory", Arm::C1, Arm::A1),
    ("Reasoning-time reprojection", Arm::C1, Arm::B1),
    ("Continuous against retrieve-once", Arm::C1, Arm::D1),
    (
        "Reasoning-time reprojection without the seed",
        Arm::C0,
        Arm::B0,
    ),
    ("The seed files", Arm::C1, Arm::C0),
];

/// The memory against the two baselines, and retrieval against the seed files
/// alone for reference: the secondary family, Holm-corrected across the
/// contrasts of it the answers can test.
pub const SECONDARY: [Contrast; 4] = [
    (
        "Provenance memory against retrieval-augmented generation",
        Arm::C1,
        Arm::R1,
    ),
    (
        "Provenance selection against BM25 retrieval, both read as text",
        Arm::P1,
        Arm::R1,
    ),
    ("Provenance memory against tool use", Arm::C1, Arm::T1),
    (
        "Retrieval-augmented generation against the seed files",
        Arm::R1,
        Arm::A1,
    ),
];

impl Arm {
    pub fn id(self) -> &'static str {
        match self {
            Arm::A0 => "a0",
            Arm::A1 => "a1",
            Arm::D1 => "d1",
            Arm::B0 => "b0",
            Arm::B1 => "b1",
            Arm::C0 => "c0",
            Arm::C1 => "c1",
            Arm::R1 => "r1",
            Arm::P1 => "p1",
            Arm::T1 => "t1",
        }
    }

    pub fn of(id: &str) -> Option<Self> {
        ARMS.into_iter().find(|a| a.id() == id)
    }

    /// Whether the arm's conversation opens on its seed files.
    pub fn seed(self) -> bool {
        matches!(self, Arm::A1 | Arm::D1 | Arm::B1 | Arm::C1)
    }

    /// Whether the arm projects the memory at all.
    pub fn projects(self) -> bool {
        !matches!(self, Arm::A0 | Arm::A1 | Arm::R1 | Arm::P1 | Arm::T1)
    }

    /// The tools mode the arm's turns run in, as the chat request's `tools`
    /// names it. T1 asks for `restricted`: the battery speaks as a `user`
    /// (`host/roles.yaml`), for whom `comprehensive` is served as
    /// `restricted` anyway (`zend::access`), and restricted holds the
    /// workspace's read and search tools.
    pub fn tools(self) -> &'static str {
        match self {
            Arm::T1 => "restricted",
            _ => "none",
        }
    }

    /// The retrieval dials the arm's requests carry. An arm that projects
    /// nothing names no reprojection: with nothing projected there is nothing
    /// to re-select.
    pub fn dials(self) -> Value {
        let reprojection = match self {
            Arm::A0 | Arm::A1 | Arm::R1 | Arm::P1 | Arm::T1 => None,
            Arm::D1 => Some("opening"),
            Arm::B0 | Arm::B1 => Some("held"),
            Arm::C0 | Arm::C1 => Some("full"),
        };
        let mut d = json!({
            "seed": self.seed(),
            "projection": if self.projects() { "full" } else { "minimal" },
        });
        if let Some(r) = reprojection {
            d["reprojection"] = json!(r);
        }
        d
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// **Each arm is the row the paper's table gives it.**
    #[test]
    fn each_arm_sets_the_dials_its_row_names() {
        let rows: Vec<(&str, Value, &str)> = ARMS
            .iter()
            .map(|a| (a.id(), a.dials(), a.tools()))
            .collect();
        assert_eq!(
            rows,
            vec![
                (
                    "a0",
                    json!({"seed": false, "projection": "minimal"}),
                    "none"
                ),
                ("a1", json!({"seed": true, "projection": "minimal"}), "none"),
                (
                    "d1",
                    json!({"seed": true, "projection": "full", "reprojection": "opening"}),
                    "none"
                ),
                (
                    "b0",
                    json!({"seed": false, "projection": "full", "reprojection": "held"}),
                    "none"
                ),
                (
                    "b1",
                    json!({"seed": true, "projection": "full", "reprojection": "held"}),
                    "none"
                ),
                (
                    "c0",
                    json!({"seed": false, "projection": "full", "reprojection": "full"}),
                    "none"
                ),
                (
                    "c1",
                    json!({"seed": true, "projection": "full", "reprojection": "full"}),
                    "none"
                ),
                (
                    "r1",
                    json!({"seed": false, "projection": "minimal"}),
                    "none"
                ),
                (
                    "p1",
                    json!({"seed": false, "projection": "minimal"}),
                    "none"
                ),
                (
                    "t1",
                    json!({"seed": false, "projection": "minimal"}),
                    "restricted"
                ),
            ]
        );
        assert_eq!(Arm::of("b1"), Some(Arm::B1));
        assert_eq!(Arm::of("t1"), Some(Arm::T1));
        assert_eq!(Arm::of("x"), None);
        assert!(!Arm::R1.projects() && !Arm::P1.projects() && !Arm::T1.projects());
    }

    /// **The primary family is the five the paper pre-registers**, and the
    /// secondary family sets the memory against the two baselines.
    #[test]
    fn the_families_hold_their_contrasts() {
        let primary: Vec<(Arm, Arm)> = CONTRASTS.iter().map(|&(_, a, b)| (a, b)).collect();
        assert_eq!(
            primary,
            vec![
                (Arm::C1, Arm::A1),
                (Arm::C1, Arm::B1),
                (Arm::C1, Arm::D1),
                (Arm::C0, Arm::B0),
                (Arm::C1, Arm::C0),
            ]
        );
        let secondary: Vec<(Arm, Arm)> = SECONDARY.iter().map(|&(_, a, b)| (a, b)).collect();
        assert_eq!(
            secondary,
            vec![
                (Arm::C1, Arm::R1),
                (Arm::P1, Arm::R1),
                (Arm::C1, Arm::T1),
                (Arm::R1, Arm::A1)
            ]
        );
    }
}
