//! One journal entry: what the writer took from a stretch of turns.
//!
//! The model writes claim text and says which turns each rests on; everything
//! else on a [`Claim`] — its [`Kind`], whether it [`perishes`](Claim::perishable)
//! — is derived by [`crate::verify`]. An entry is rendered to prose once, when it
//! is kept ([`Entry::render`]), because the sealed section it becomes cannot be
//! re-rendered when it is gathered: the time it carries is therefore absolute,
//! never "twenty minutes ago".

use serde::{Deserialize, Serialize};

use crate::period::Period;
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
///
/// It reads two ways. In the prompt it is one line — its number and its
/// `summary` ([`Self::render`]), so the journal stays short however long it
/// gets. Asked for, it is everything: the summary, the cited claims, what is
/// meant next and what it opened and settled ([`Self::detail`]).
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Entry {
    pub id: u64,
    pub from_turn: u64,
    pub to_turn: u64,
    pub from_ms: u64,
    pub to_ms: u64,
    /// The task or step the stretch was, in one sentence: the entry as the
    /// prompt reads it.
    pub summary: String,
    pub claims: Vec<Claim>,
    pub intend: Vec<String>,
    /// Items this entry opened.
    pub opened: Vec<Item>,
    /// Ids of open items this entry settled.
    pub resolved: Vec<u64>,
    /// What it stands for: a stretch of turns, or a day, week, month or year
    /// of entries folded into one ([`crate::period`]).
    #[serde(default)]
    pub period: Period,
    /// A folded entry's detailed summary of what it folded: what the journal's
    /// tools read behind its line. Empty for a step, whose detail is its claims.
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub digest: String,
}

impl Entry {
    /// The span the entry covers, as `style` writes it.
    pub fn span(&self, style: &Style) -> String {
        (style.span)(self.from_ms, self.to_ms)
    }

    /// When the entry is read after `before` — the entry above it in the
    /// journal, `None` for the first — the line that marks the time moving on:
    /// the day and hour when the day changed, the hour alone when only the hour
    /// did, nothing when neither did. Coarse on purpose: a journal reads as one
    /// running list, dated where the date moves.
    pub fn when(&self, before: Option<&Entry>, style: &Style) -> Option<String> {
        let (day, hour) = ((style.day)(self.from_ms), (style.hour)(self.from_ms));
        match before.map(|b| ((style.day)(b.from_ms), (style.hour)(b.from_ms))) {
            Some((d, h)) if d == day && h == hour => None,
            Some((d, _)) if d == day => Some(hour),
            _ => Some(format!("{day}, {hour}")),
        }
    }

    /// The entry as the prompt reads it after `before` (see [`Self::when`]):
    /// the time line when the time moved on, then one line — its number and
    /// its summary.
    pub fn render(&self, before: Option<&Entry>, style: &Style) -> String {
        let line = format!("{} {} — {}", style.entry, self.id, self.summary.trim());
        match self.when(before, style) {
            Some(when) => format!("{when}\n{line}"),
            None => line,
        }
    }

    /// Everything the entry holds, as the journal's tools show it: its number,
    /// span and summary, then its claims in the order the turns they rest on
    /// happened, each under how it came to be known, then what is meant next
    /// and what it opened and settled.
    ///
    /// **In order, not grouped by kind.** Grouped, an entry read its results
    /// before the request that led to them, and a writer asked what happened
    /// told it in that order.
    pub fn detail(&self, style: &Style) -> String {
        let mut out = format!(
            "{} {} {}\n{}",
            style.entry,
            self.id,
            self.span(style),
            self.summary.trim()
        );
        if !self.digest.trim().is_empty() {
            out.push('\n');
            out.push_str(self.digest.trim());
        }
        let mut claims: Vec<&Claim> = self.claims.iter().collect();
        claims.sort_by_key(|c| c.cite.iter().min().copied().unwrap_or(u64::MAX));
        for c in claims {
            out.push_str(&format!("\n- {} {}", c.kind.heading(style), c.line(style)));
        }
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
    /// The claim as its line reads: its sentence, what the world read when it
    /// read the claim's fact differently, and whether it may have changed.
    fn line(&self, style: &Style) -> String {
        let mut s = self.text.trim().to_string();
        if let (true, Some(t)) = (self.corrected, &self.typed) {
            s.push_str(&format!(
                "{} {} {} as {})",
                style.corrected, t.subject, t.attribute, t.value
            ));
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
            summary: format!("Step {id}."),
            claims: vec![],
            intend: vec![],
            opened: vec![],
            resolved: vec![],
            period: Period::Step,
            digest: String::new(),
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

    /// The order claims of each kind are strongest first in.
    const KINDS: [Kind; 4] = [Kind::Observed, Kind::Heard, Kind::Did, Kind::Inferred];

    fn citing(text: &str, kind: Kind, cite: &[u64]) -> Claim {
        Claim {
            cite: cite.to_vec(),
            ..claim(text, kind)
        }
    }

    #[test]
    fn the_span_is_written_by_the_style() {
        assert_eq!(entry(1).span(&PLAIN), "[100–200]");
    }

    /// In the prompt an entry is one line, whatever it holds.
    #[test]
    fn the_prompt_reads_an_entry_as_its_number_and_summary() {
        let mut e = entry(3);
        e.summary = " Fixed the scan's race. ".into();
        e.claims = vec![citing("The tower holds 40 ore.", Kind::Observed, &[2])];
        e.intend = vec!["Ask Pax.".into()];
        assert_eq!(
            e.render(None, &PLAIN),
            "d0, h1\nEntry 3 — Fixed the scan's race."
        );
    }

    #[test]
    fn the_detail_reads_claims_in_the_order_they_happened_each_under_how_it_was_known() {
        let mut e = entry(3);
        e.claims = vec![
            citing("Somebody is stealing power.", Kind::Inferred, &[9]),
            citing("The tower holds 40 ore.", Kind::Observed, &[5, 2]),
            citing("I told Pax I was coming.", Kind::Did, &[7]),
            citing("Pax says the east door is jammed.", Kind::Heard, &[1]),
        ];
        assert_eq!(
            e.detail(&PLAIN),
            "Entry 3 [100–200]\n\
             Step 3.\n\
             - Heard: Pax says the east door is jammed.\n\
             - Saw: The tower holds 40 ore.\n\
             - Did: I told Pax I was coming.\n\
             - Concluded: Somebody is stealing power."
        );
    }

    fn at(id: u64, from_ms: u64) -> Entry {
        Entry {
            from_ms,
            to_ms: from_ms,
            ..entry(id)
        }
    }

    /// The journal is one running list, dated only where the date moves: the
    /// day and hour for the first entry and on a new day, the hour alone on a
    /// new hour, and nothing within the hour.
    #[test]
    fn an_entry_is_dated_only_where_the_day_or_hour_moved() {
        let (a, b, c, d) = (at(1, 1_100), at(2, 1_150), at(3, 1_250), at(4, 2_050));
        let heads: Vec<String> = [
            a.render(None, &PLAIN),
            b.render(Some(&a), &PLAIN),
            c.render(Some(&b), &PLAIN),
            d.render(Some(&c), &PLAIN),
        ]
        .into();
        assert_eq!(
            heads,
            [
                "d1, h11\nEntry 1 — Step 1.",
                "Entry 2 — Step 2.",
                "h12\nEntry 3 — Step 3.",
                "d2, h20\nEntry 4 — Step 4."
            ]
        );
    }

    #[test]
    fn a_perishable_claim_asks_to_be_rechecked_and_a_corrected_one_carries_the_reading() {
        let mut e = entry(1);
        let mut c = claim("Fabricator 5 is running.", Kind::Observed);
        c.perishable = true;
        c.corrected = true;
        c.typed = Some(Typed {
            subject: "fabricator 5".into(),
            attribute: "mode".into(),
            value: "idle".into(),
        });
        e.claims = vec![c];
        let r = e.detail(&PLAIN);
        assert!(
            r.contains(
                "- Saw: Fabricator 5 is running. (reads fabricator 5 mode as idle) (re-check)"
            ),
            "{r}"
        );
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
        let r = e.detail(&PLAIN);
        assert!(r.contains("Meaning to:\n- Ask Pax about the door."));
        assert!(r.contains("Left open:\n- #4 Who holds the shield decision?"));
        assert!(r.ends_with("Settled:\n- #2\n- #3"));
    }

    #[test]
    fn a_folded_entry_reads_its_digest_behind_its_line() {
        let mut e = entry(9);
        e.period = Period::Day;
        e.summary = "Fixed the scan race and shipped it.".into();
        e.digest = "Found the race in reclaim; bounded the loop; tests green.".into();
        assert_eq!(
            e.detail(&PLAIN),
            "Entry 9 [100–200]\nFixed the scan race and shipped it.\n\
             Found the race in reclaim; bounded the loop; tests green."
        );
        let json = serde_json::to_value(&e).unwrap();
        assert_eq!(json["period"], "day");
        let step = serde_json::to_value(entry(1)).unwrap();
        assert!(step.get("digest").is_none(), "a step stores no digest");
    }

    #[test]
    fn an_empty_section_is_left_out_of_the_detail() {
        assert_eq!(entry(1).detail(&PLAIN), "Entry 1 [100–200]\nStep 1.");
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
