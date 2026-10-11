//! The entry a character's journal gains when it hands a mission in — written
//! by the engine from the mission's own record, not asked of the character.
//!
//! **Each mission is a chapter**, and a character starts a new conversation with
//! each (`Minds::think`). The journal is what carries the work across, so the
//! close of a chapter is not left to whether the character thinks it worth an
//! entry, nor to how it feels about it: what it was asked, which document, how it
//! was reported and in what year it was worked are facts the mission holds, and
//! they are written as the character's own `Did:` lines. A stretch that went
//! badly — Makers once told each other for hours that there was nothing left to
//! report — is not what the next chapter opens on; the work is.

use super::entry::{Claim, Entry, Kind};
use crate::engine::mission::{Mission, Outcome};

/// The longest a line of the closing entry runs, in characters.
const LINE: usize = 220;

/// The entry closing `mission`, at world time `at_ms`, covering nothing of the
/// window past `covered_to` — the turns are the character's own entries' to
/// cover.
pub fn closing_entry(mission: &Mission, at_ms: u64, covered_to: u64) -> Entry {
    let did = |text: String| Claim {
        text,
        cite: Vec::new(),
        kind: Kind::Did,
        perishable: false,
        typed: None,
        corrected: false,
    };
    let ask = mission
        .mission_text()
        .lines()
        .find(|l| !l.trim().is_empty())
        .unwrap_or_default()
        .trim()
        .to_string();
    let mut claims = vec![did(format!("Handed in: {}", clip(&ask)))];
    if let Some(work) = &mission.work {
        claims.push(did(format!("The document was {}.", work.writes)));
    }
    claims.push(did(match &mission.report {
        Some(r) if r.outcome == Outcome::Pass => format!("Reported it done: {}", clip(&r.notes)),
        Some(r) => format!("Reported it not done: {}", clip(&r.notes)),
        None => "It was called off before it was reported.".to_string(),
    }));
    if let Some(year) = mission.year {
        claims.push(did(format!(
            "Worked in {year}, at a time machine; back in the present since."
        )));
    }
    Entry {
        id: 0,
        from_turn: covered_to,
        to_turn: covered_to,
        from_ms: at_ms,
        to_ms: at_ms,
        claims,
        intend: Vec::new(),
        opened: Vec::new(),
        resolved: Vec::new(),
    }
}

/// `s` on one line, at most [`LINE`] characters, cut at a word.
fn clip(s: &str) -> String {
    let flat = s.split_whitespace().collect::<Vec<_>>().join(" ");
    if flat.chars().count() <= LINE {
        return flat;
    }
    let cut: String = flat.chars().take(LINE).collect();
    match cut.rsplit_once(' ') {
        Some((head, _)) => format!("{head}…"),
        None => format!("{cut}…"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::mission::{Origin, Todo, Work};

    fn drafted() -> Mission {
        Mission::new(
            "Write Shooter's life: 3082-06-15, \"The Last Calibration\".\n\nWho Shooter is…",
            vec![Todo::new("the one step")],
            Origin::Generated {
                generator: "life-event".into(),
                target: "life:zenling-shooter".into(),
                operation: 3,
                step: "write".into(),
            },
        )
        .with_work(Work {
            writes: "layers/life/zenling-shooter/3082-06-15 The Last Calibration.md".into(),
            reads: Vec::new(),
            min_words: 250,
            edit_optional: false,
            anew: false,
            checks: Vec::new(),
            tools: Vec::new(),
        })
    }

    /// **The close of a chapter is the mission's record, told as the
    /// character's own acts** — what it was asked, the document, how it was
    /// reported, and the year it was worked in.
    #[test]
    fn a_handed_in_mission_closes_its_chapter_in_the_journal() {
        let mut m = drafted();
        m.travelled(3082);
        m.complete(
            Outcome::Pass,
            "Wrote the calibration, in Shooter's third person.",
            None,
        );
        let e = closing_entry(&m, 5_000, 41);
        let lines: Vec<&str> = e.claims.iter().map(|c| c.text.as_str()).collect();
        assert_eq!(
            lines,
            [
                "Handed in: Write Shooter's life: 3082-06-15, \"The Last Calibration\".",
                "The document was layers/life/zenling-shooter/3082-06-15 The Last Calibration.md.",
                "Reported it done: Wrote the calibration, in Shooter's third person.",
                "Worked in 3082, at a time machine; back in the present since.",
            ]
        );
        assert!(e
            .claims
            .iter()
            .all(|c| c.kind == Kind::Did && !c.perishable));
        assert_eq!((e.from_turn, e.to_turn), (41, 41), "it covers no turns");
    }

    #[test]
    fn a_mission_called_off_says_so() {
        let e = closing_entry(&drafted(), 0, 0);
        assert_eq!(
            e.claims.last().map(|c| c.text.as_str()),
            Some("It was called off before it was reported.")
        );
    }

    #[test]
    fn a_long_line_is_cut_at_a_word() {
        let long = "word ".repeat(100);
        let c = clip(&long);
        assert!(c.ends_with("word…"), "{c}");
        assert!(c.chars().count() <= LINE + 1);
    }
}
