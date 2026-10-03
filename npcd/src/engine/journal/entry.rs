//! One journal entry: what the character took from a span of its own window.
//!
//! The model writes claim text and says which turns each rests on; everything
//! else on a [`Claim`] — its [`Kind`], whether it [`perishes`](Claim::perishable)
//! — is derived by [`crate::engine::journal::verify`]. An entry is rendered to
//! prose once, at write time ([`Entry::render`]), because the sealed KV a journal
//! turn becomes cannot be re-rendered when it is gathered: the time it carries is
//! therefore absolute (`14 Jun 2187, 14:20`), never "twenty minutes ago".

use serde::{Deserialize, Serialize};

use crate::clock::{day_of, WorldTime};

/// How a claim came to be known, derived from what it cites.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Kind {
    /// A tool result or a world event the character itself received.
    Observed,
    /// Somebody else said so — a peer or the narrator.
    Heard,
    /// The character's own act, recorded: a fact about what it did, not about
    /// the world.
    Did,
    /// Nothing in the world supports it; the character concluded it.
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
}

/// A claim about the world the sim can be asked about.
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
    /// The window turn ids it rests on.
    pub cite: Vec<u64>,
    pub kind: Kind,
    /// Whether it can stop being true without the character noticing.
    pub perishable: bool,
    pub typed: Option<Typed>,
    /// The world disagreed with the character, and this is what the world said.
    pub corrected: bool,
}

/// Something the character has not finished with.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Item {
    pub id: u64,
    pub text: String,
}

/// A span of the window, as the character kept it.
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

impl Entry {
    /// The span the entry covers: `[14 Jun 2187, 14:20–14:41]`, or both ends in
    /// full when it crosses midnight.
    pub fn span(&self) -> String {
        let (a, b) = (WorldTime::of(self.from_ms), WorldTime::of(self.to_ms));
        if day_of(self.from_ms) == day_of(self.to_ms) {
            format!("[{}, {}–{}]", a.date(), a.clock(), b.clock())
        } else {
            format!("[{} – {}]", a.stamp(), b.stamp())
        }
    }

    /// The entry as the character reads it back.
    pub fn render(&self) -> String {
        let mut out = self.span();
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
        let of = |kind: Kind| -> Vec<String> {
            self.claims
                .iter()
                .filter(|c| c.kind == kind)
                .map(Claim::line)
                .collect()
        };
        section("Saw:", of(Kind::Observed));
        section("Heard:", of(Kind::Heard));
        section("Did:", of(Kind::Did));
        section("Concluded, not seen or heard:", of(Kind::Inferred));
        section("Meaning to:", self.intend.clone());
        section(
            "Left open:",
            self.opened
                .iter()
                .map(|i| format!("#{} {}", i.id, i.text))
                .collect(),
        );
        section(
            "Settled:",
            self.resolved.iter().map(|id| format!("#{id}")).collect(),
        );
        out
    }
}

impl Claim {
    fn line(&self) -> String {
        let mut s = self.text.trim().to_string();
        if self.corrected {
            s.push_str(" (the world read differently than I first wrote)");
        }
        if self.perishable {
            s.push_str(" (re-check)");
        }
        s
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::clock::DAY_MS;

    const H: u64 = 3_600_000;
    const M: u64 = 60_000;

    pub(crate) fn claim(text: &str, kind: Kind) -> Claim {
        Claim {
            text: text.into(),
            cite: vec![1],
            kind,
            perishable: false,
            typed: None,
            corrected: false,
        }
    }

    pub(crate) fn entry(id: u64) -> Entry {
        Entry {
            id,
            from_turn: 1,
            to_turn: 9,
            from_ms: DAY_MS * 3 + 14 * H + 20 * M,
            to_ms: DAY_MS * 3 + 14 * H + 41 * M,
            claims: vec![],
            intend: vec![],
            opened: vec![],
            resolved: vec![],
        }
    }

    #[test]
    fn a_span_inside_a_day_names_the_day_once() {
        assert_eq!(entry(1).span(), "[4 Jan 1970, 14:20–14:41]");
    }

    #[test]
    fn a_span_across_midnight_names_both_days() {
        let mut e = entry(1);
        e.from_ms = DAY_MS * 3 + 23 * H + 50 * M;
        e.to_ms = DAY_MS * 4 + 10 * M;
        assert_eq!(e.span(), "[4 Jan 1970, 23:50 – 5 Jan 1970, 00:10]");
    }

    #[test]
    fn claims_are_grouped_by_how_they_came_to_be_known() {
        let mut e = entry(1);
        e.claims = vec![
            claim("The tower holds 40 ore.", Kind::Observed),
            claim("Pax says the east door is jammed.", Kind::Heard),
            claim("I told Pax I was coming.", Kind::Did),
            claim("Somebody is stealing power.", Kind::Inferred),
        ];
        assert_eq!(
            e.render(),
            "[4 Jan 1970, 14:20–14:41]\n\
             Saw:\n- The tower holds 40 ore.\n\
             Heard:\n- Pax says the east door is jammed.\n\
             Did:\n- I told Pax I was coming.\n\
             Concluded, not seen or heard:\n- Somebody is stealing power."
        );
    }

    #[test]
    fn a_perishable_claim_asks_to_be_rechecked_and_a_corrected_one_says_so() {
        let mut e = entry(1);
        let mut c = claim("Fabricator 5 is running.", Kind::Observed);
        c.perishable = true;
        c.corrected = true;
        e.claims = vec![c];
        assert!(e.render().contains(
            "- Fabricator 5 is running. (the world read differently than I first wrote) (re-check)"
        ));
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
        let r = e.render();
        assert!(r.contains("Meaning to:\n- Ask Pax about the door."));
        assert!(r.contains("Left open:\n- #4 Who holds the shield decision?"));
        assert!(r.ends_with("Settled:\n- #2\n- #3"));
    }

    #[test]
    fn an_empty_section_is_left_out() {
        assert_eq!(entry(1).render(), "[4 Jan 1970, 14:20–14:41]");
    }

    #[test]
    fn an_entry_survives_its_own_json() {
        let mut e = entry(7);
        e.claims = vec![claim("x", Kind::Heard)];
        let back: Entry = serde_json::from_str(&serde_json::to_string(&e).unwrap()).unwrap();
        assert_eq!(back, e);
    }
}
