//! The things in a room, and what they let you do.
//!
//! A node is somewhere to stand; a **part** is something standing in it. A
//! terminal, a wall board, a chair. Parts are defined once, in their own
//! files, and placed by reference — so *world history terminal* means the same
//! thing on every level of every building that has one, and gains a tool
//! everywhere at once when it gains one here.
//!
//! # Why the tools live here
//!
//! Most of what an NPC can do is a property of what is within reach, not of
//! who it is. Standing at a chronicle terminal is what makes rewriting an era
//! possible; walking away is what makes it impossible. Hanging [`Part::tools`]
//! on the part rather than on the NPC means the reachable tool surface is
//! computed from the map, changes as a body moves, and cannot be reasoned
//! about wrongly — a tool an NPC is not standing next to is not offered.
//!
//! # Two descriptions, for two readers
//!
//! [`Part::short`] is a clause for a level's prose: what somebody glancing
//! round the room would say about it. [`Part::long`] is **never** prose. It is
//! the provenance carried alongside the tools when they are offered — what
//! this thing is, what it does, what taking one commits you to — and it is
//! read at the moment the tools are, not while describing a level.
//!
//! Keeping them apart matters because they are wanted at different times and
//! at different lengths. A level that inlined every long description would be
//! unreadable; a tool offered without one would be unusable.

use serde::{Deserialize, Serialize};

/// What sort of thing a part is, which decides how a description counts it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum PartKind {
    /// Something worked at, and usually claimed while it is. Counted, and the
    /// count is what makes a level able to hold a crew.
    Station,
    /// Something read or consulted, fixed in place. Named rather than counted.
    Fixture,
    /// Somewhere to be that is not work. Counted.
    Seat,
}

/// One kind of thing that stands in a room.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Part {
    pub id: String,
    pub kind: PartKind,
    /// What it is called, singular and bare: "world history terminal".
    /// Articles and plurals are the description's business, not the file's.
    pub name: String,
    /// The plural, where adding an `s` would be wrong.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub plural: Option<String>,
    /// What taking one of these claims, as a noun phrase: "one era".
    ///
    /// On the part rather than on the room, because it is the terminal that
    /// holds an era — the room merely has terminals in it, and the same
    /// terminal claims the same thing wherever it is standing.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub binds: Option<String>,
    /// One clause for a level's prose. Authored, optional, carried verbatim.
    ///
    /// **It must name its own subject.** Several parts stand in one room and
    /// their clauses run together, so a `short` opening with a bare "it"
    /// silently attaches itself to whichever part happened to be listed first
    /// — *any page can be pulled from it. It runs the length of the room* is
    /// two things and reads as one.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub short: Option<String>,
    /// What this is and what it does, for the provenance carried with the
    /// tools. **Never** appears in a level description.
    pub long: String,
    /// The tools being within reach of this part makes available.
    #[serde(default)]
    pub tools: Vec<String>,
}

impl Part {
    /// The name, in the right number.
    pub fn count_name(&self, n: u32) -> String {
        if n == 1 {
            self.name.clone()
        } else {
            self.plural
                .clone()
                .unwrap_or_else(|| format!("{}s", self.name))
        }
    }
}

/// A part standing in a node, and how many of it.
///
/// Written either as `- seat` or as `- { part: seat, count: 12 }`. The bare
/// form is for the common case of one, which is most fixtures.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum Placement {
    Bare(String),
    Counted {
        part: String,
        #[serde(default = "one")]
        count: u32,
    },
}

fn one() -> u32 {
    1
}

impl Placement {
    pub fn part(&self) -> &str {
        match self {
            Placement::Bare(id) => id,
            Placement::Counted { part, .. } => part,
        }
    }

    pub fn count(&self) -> u32 {
        match self {
            Placement::Bare(_) => 1,
            Placement::Counted { count, .. } => *count,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn part(name: &str, plural: Option<&str>) -> Part {
        Part {
            id: "p".into(),
            kind: PartKind::Station,
            name: name.into(),
            plural: plural.map(String::from),
            binds: None,
            short: None,
            long: "l".into(),
            tools: vec![],
        }
    }

    #[test]
    fn a_name_pluralises_by_adding_an_s_unless_told_otherwise() {
        assert_eq!(part("terminal", None).count_name(1), "terminal");
        assert_eq!(part("terminal", None).count_name(6), "terminals");
        assert_eq!(part("bench", Some("benches")).count_name(2), "benches");
    }

    #[test]
    fn a_bare_placement_means_one_of_it() {
        let bare: Placement = serde_yaml::from_str("the-roster").unwrap();
        assert_eq!(bare.part(), "the-roster");
        assert_eq!(bare.count(), 1);
    }

    #[test]
    fn a_counted_placement_says_how_many() {
        let counted: Placement = serde_yaml::from_str("{ part: seat, count: 12 }").unwrap();
        assert_eq!(counted.part(), "seat");
        assert_eq!(counted.count(), 12);
    }

    #[test]
    fn a_counted_placement_without_a_count_is_still_one() {
        let counted: Placement = serde_yaml::from_str("{ part: seat }").unwrap();
        assert_eq!(counted.count(), 1);
    }
}
