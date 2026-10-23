//! One journal entry: what the writer took from a stretch of turns.
//!
//! The model writes claim text and says which turns each rests on; everything
//! else on a [`Claim`] — its [`Kind`], whether it [`perishes`](Claim::perishable)
//! — is derived by [`crate::verify`]. An entry is rendered to prose once, when it
//! is kept ([`Entry::render`]), because the sealed section it becomes cannot be
//! re-rendered when it is gathered: the time it carries is therefore absolute,
//! never "twenty minutes ago".

use serde::{Deserialize, Serialize};

use crate::style::Style;

/// How a claim came to be known, derived from what it cites.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Kind {
    /// A tool result, or an event the writer itself received.
    Observed,
    /// Somebody else said so.
    Heard,
    /// The writer's own act, recorded: a fact about what it did, not about the
    /// world.
    Did,
    /// Nothing the writer saw or heard supports it; it was concluded.
    Inferred,
}

impl Kind {
    /// The name it is stored and shown under.
    pub fn name(self) -> &'static str {
        match self {
            Kind::Observed => "observed",
            Kind::Heard => "heard",
            Kind::Did => "did",
            Kind::Inferred => "inferred",
        }
    }

    /// How strongly a cited turn of this kind supports a claim: a claim citing
    /// several takes the strongest.
    pub fn rank(self) -> u8 {
        match self {
            Kind::Observed => 3,
            Kind::Heard => 2,
            Kind::Did => 1,
            Kind::Inferred => 0,
        }
    }

    fn heading(self, style: &Style) -> &'static str {
        match self {
            Kind::Observed => style.saw,
            Kind::Heard => style.heard,
            Kind::Did => style.did,
            Kind::Inferred => style.inferred,
        }
    }
}

/// A claim the world can be asked about.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Typed {
    pub subject: String,
    pub attribute: String,
    pub value: String,
}

/// One thing the entry says is so.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Claim {
    pub text: String,
    /// The turn ids it rests on.
    pub cite: Vec<u64>,
    pub kind: Kind,
    /// Whether it can stop being true without the writer noticing.
    pub perishable: bool,
    pub typed: Option<Typed>,
    /// The world disagreed with the writer, and this is what the world said.
    pub corrected: bool,
}

/// Something the writer has not finished with.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Item {
    pub id: u64,
    pub text: String,
}

/// A stretch of turns, as the writer kept it.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Entry {
    pub id: u64,
    pub from_turn: u64,
    pub to_turn: u64,
    pub from_ms: u64,
    pub to_ms: u64,
    pub claims: Vec<Claim>,
    pub intend: Vec<String>,
    /// Items this entry opened.
    pub opened: Vec<Item>,
    /// Ids of open items this entry settled.
    pub resolved: Vec<u64>,
}

/// The order claims are grouped in when an entry is read back.
const KINDS: [Kind; 4] = [Kind::Observed, Kind::Heard, Kind::Did, Kind::Inferred];

impl Entry {
    /// The span the entry covers, as `style` writes it.
    pub fn span(&self, style: &Style) -> String {
        (style.span)(self.from_ms, self.to_ms)
    }

    /// The entry as the writer reads it back.
    pub fn render(&self, style: &Style) -> String {
        let mut out = self.span(style);
        let mut section = |title: &str, lines: Vec<String>| {
            if lines.is_empty() {
                return;
            }
            out.push('\n');
            out.push_str(title);
            for l in lines {
                out.push_str("\n- ");
                out.push_str(&l);
            }
        };
        for kind in KINDS {
            let lines: Vec<String> = self
                .claims
                .iter()
                .filter(|c| c.kind == kind)
                .map(|c| c.line(style))
                .collect();
            section(kind.heading(style), lines);
        }
        section(style.meaning_to, self.intend.clone());
        section(
            style.left_open,
            self.opened
                .iter()
                .map(|i| format!("#{} {}", i.id, i.text))
                .collect(),
        );
        section(
            style.settled,
            self.resolved.iter().map(|id| format!("#{id}")).collect(),
        );
        out
    }
}

impl Claim {
    fn line(&self, style: &Style) -> String {
        let mut s = self.text.trim().to_string();
        if self.corrected {
            s.push_str(style.corrected);
        }
        if self.perishable {
            s.push_str(style.recheck);
        }
        s
    }
}

#[cfg(test)]
pub(crate) mod testing {
    use super::*;

    pub fn claim(text: &str, kind: Kind) -> Claim {
        Claim {
            text: text.into(),
            cite: vec![1],
            kind,
            perishable: false,
            typed: None,
            corrected: false,
        }
    }

    pub fn entry(id: u64) -> Entry {
        Entry {
            id,
            from_turn: 1,
            to_turn: 9,
            from_ms: 100,
            to_ms: 200,
            claims: vec![],
            intend: vec![],
            opened: vec![],
            resolved: vec![],
        }
    }

    pub fn entry_to(to_turn: u64) -> Entry {
        Entry {
            id: 0,
            to_turn,
            ..entry(0)
        }
    }

    pub fn item(id: u64, text: &str) -> Item {
        Item {
            id,
            text: text.into(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::testing::{claim, entry};
    use super::*;
    use crate::style::testing::PLAIN;

    #[test]
    fn the_span_is_written_by_the_style() {
        assert_eq!(entry(1).span(&PLAIN), "[100–200]");
    }

    #[test]
    fn claims_are_grouped_by_how_they_came_to_be_known() {
        let mut e = entry(1);
        e.claims = vec![
            claim("Somebody is stealing power.", Kind::Inferred),
            claim("The tower holds 40 ore.", Kind::Observed),
            claim("I told Pax I was coming.", Kind::Did),
            claim("Pax says the east door is jammed.", Kind::Heard),
        ];
        assert_eq!(
            e.render(&PLAIN),
            "[100–200]\n\
             Saw:\n- The tower holds 40 ore.\n\
             Heard:\n- Pax says the east door is jammed.\n\
             Did:\n- I told Pax I was coming.\n\
             Concluded:\n- Somebody is stealing power."
        );
    }

    #[test]
    fn a_perishable_claim_asks_to_be_rechecked_and_a_corrected_one_says_so() {
        let mut e = entry(1);
        let mut c = claim("Fabricator 5 is running.", Kind::Observed);
        c.perishable = true;
        c.corrected = true;
        e.claims = vec![c];
        assert!(e
            .render(&PLAIN)
            .contains("- Fabricator 5 is running. (corrected) (re-check)"));
    }

    #[test]
    fn open_items_carry_their_ids_and_settled_ones_are_named_by_id() {
        let mut e = entry(2);
        e.intend = vec!["Ask Pax about the door.".into()];
        e.opened = vec![Item {
            id: 4,
            text: "Who holds the shield decision?".into(),
        }];
        e.resolved = vec![2, 3];
        let r = e.render(&PLAIN);
        assert!(r.contains("Meaning to:\n- Ask Pax about the door."));
        assert!(r.contains("Left open:\n- #4 Who holds the shield decision?"));
        assert!(r.ends_with("Settled:\n- #2\n- #3"));
    }

    #[test]
    fn an_empty_section_is_left_out() {
        assert_eq!(entry(1).render(&PLAIN), "[100–200]");
    }

    #[test]
    fn an_entry_survives_its_own_json() {
        let mut e = entry(7);
        e.claims = vec![claim("x", Kind::Heard)];
        let back: Entry = serde_json::from_str(&serde_json::to_string(&e).unwrap()).unwrap();
        assert_eq!(back, e);
    }

    #[test]
    fn the_strongest_support_ranks_highest() {
        let mut kinds = KINDS;
        kinds.sort_by_key(|k| std::cmp::Reverse(k.rank()));
        assert_eq!(kinds, KINDS);
    }
}
