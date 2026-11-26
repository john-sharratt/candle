//! What a person asked of a character on the phone, as the mind is held to it
//! (`Scheduler::ask`): their own words, and what the character means to do
//! about them.
//!
//! **The person's words, never the character's account of its answer.** The
//! held request was the conversation's whole journal entry, which leads with
//! what the character *said*: "Told Johnathan Sharratt that I am checking on
//! the Drones and will provide a more detailed report". Restated at the head of
//! every turn as what he had asked, it set Zen the task of being someone who
//! says it is checking — and saying so was doing it. Zen promised the report
//! twice in eight seconds, posted a Drone's "I'm checking the status board"
//! back as its own, and chose to rest eighty times a minute, never taking an
//! act toward the report. John had asked what the Drones were up to.

use journal_core::entry::Entry;

/// What the mind is held to after a conversation turn in which `party` said
/// `asked` — their lines, oldest first — and the kept `entry` meant to do
/// something about it. `None` when the entry means to do nothing, or nothing
/// was said to answer: there is nothing to hold the mind to.
pub fn held_request(party: &str, asked: &[&str], entry: &Entry) -> Option<String> {
    let asked: Vec<&str> = asked
        .iter()
        .map(|l| l.trim())
        .filter(|l| !l.is_empty())
        .collect();
    let meant: Vec<&str> = entry
        .intend
        .iter()
        .map(|l| l.trim())
        .filter(|l| !l.is_empty())
        .collect();
    if asked.is_empty() || meant.is_empty() {
        return None;
    }
    let quoted: Vec<String> = asked.iter().map(|l| format!("\"{l}\"")).collect();
    let mut out = format!(
        "{party} said: {}\nWhat you meant to do about it:",
        quoted.join(" ")
    );
    for m in meant {
        out.push_str("\n- ");
        out.push_str(m);
    }
    Some(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use journal_core::period::Period;

    fn entry(summary: &str, intend: &[&str]) -> Entry {
        Entry {
            id: 2,
            from_turn: 0,
            to_turn: 1,
            from_ms: 0,
            to_ms: 0,
            summary: summary.to_string(),
            claims: Vec::new(),
            intend: intend.iter().map(|s| s.to_string()).collect(),
            opened: Vec::new(),
            resolved: Vec::new(),
            period: Period::default(),
            digest: String::new(),
        }
    }

    /// **The request is the person's words and the intent, never the summary
    /// of what the character told them** — the live entry from the loop.
    #[test]
    fn the_request_is_what_was_asked_and_what_is_meant() {
        let e = entry(
            "Told Johnathan Sharratt that I am checking on the Drones and will provide a more \
             detailed report once I have specifics from each of them.",
            &["Ask each Drone what it is doing, and tell Johnathan what they say."],
        );
        assert_eq!(
            held_request("Johnathan", &["Excellent. What are the drones up too?"], &e).unwrap(),
            "Johnathan said: \"Excellent. What are the drones up too?\"\n\
             What you meant to do about it:\n\
             - Ask each Drone what it is doing, and tell Johnathan what they say."
        );
    }

    /// Several lines are quoted in order; blank ones are not lines.
    #[test]
    fn every_line_asked_is_quoted_in_order() {
        let e = entry("x", &["Read the panel.", "  "]);
        assert_eq!(
            held_request("Mira", &["Is the lift fixed?", " ", "And the lights?"], &e).unwrap(),
            "Mira said: \"Is the lift fixed?\" \"And the lights?\"\n\
             What you meant to do about it:\n\
             - Read the panel."
        );
    }

    /// Nothing meant, or nothing asked, holds the mind to nothing.
    #[test]
    fn nothing_meant_or_nothing_asked_holds_nothing() {
        assert_eq!(held_request("Mira", &["Hi"], &entry("x", &[])), None);
        assert_eq!(held_request("Mira", &[], &entry("x", &["Read it."])), None);
    }
}
