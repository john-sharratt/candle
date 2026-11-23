//! The battery's seven arms (`docs/unbounded_agents.md` §9.12, *Arms*): each a
//! setting of the turn's three retrieval dials, sent as the chat request's
//! `seed`, `projection` and `reprojection` fields.

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
}

/// Every arm, in the order the paper tables them.
pub const ARMS: [Arm; 7] = [
    Arm::A0,
    Arm::A1,
    Arm::D1,
    Arm::B0,
    Arm::B1,
    Arm::C0,
    Arm::C1,
];

/// The five contrasts the paper tests, each `(name, arm, against)`.
pub const CONTRASTS: [(&str, Arm, Arm); 5] = [
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
        !matches!(self, Arm::A0 | Arm::A1)
    }

    /// The retrieval dials the arm's requests carry. An arm that projects
    /// nothing names no reprojection: with nothing projected there is nothing
    /// to re-select.
    pub fn dials(self) -> Value {
        let reprojection = match self {
            Arm::A0 | Arm::A1 => None,
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
        let rows: Vec<(&str, Value)> = ARMS.iter().map(|a| (a.id(), a.dials())).collect();
        assert_eq!(
            rows,
            vec![
                ("a0", json!({"seed": false, "projection": "minimal"})),
                ("a1", json!({"seed": true, "projection": "minimal"})),
                (
                    "d1",
                    json!({"seed": true, "projection": "full", "reprojection": "opening"})
                ),
                (
                    "b0",
                    json!({"seed": false, "projection": "full", "reprojection": "held"})
                ),
                (
                    "b1",
                    json!({"seed": true, "projection": "full", "reprojection": "held"})
                ),
                (
                    "c0",
                    json!({"seed": false, "projection": "full", "reprojection": "full"})
                ),
                (
                    "c1",
                    json!({"seed": true, "projection": "full", "reprojection": "full"})
                ),
            ]
        );
        assert_eq!(Arm::of("b1"), Some(Arm::B1));
        assert_eq!(Arm::of("x"), None);
    }
}
