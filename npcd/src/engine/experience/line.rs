//! How each kind of experience reads in a bot's journal
//! (`docs/npc_conversations.md` §5.1): the words a later catch-up is given,
//! written so a reader who has never seen the world can follow them.

/// What the bot heard: `Mira, on the phone: the parcel is at the door.`
pub fn heard(from: &str, on: &str, text: &str) -> String {
    format!("{from}, {on}: {}", text.trim())
}

/// What the bot said: `To Mira, on the phone: I'll bring it in.`
pub fn said(to: &str, on: &str, text: &str) -> String {
    format!("To {to}, {on}: {}", text.trim())
}

/// What the bot promised: `Promised John the garden report, by Friday.`
pub fn promised(to: &str, what: &str, by: &str) -> String {
    format!("Promised {to} {}, by {}.", trim_stop(what), trim_stop(by))
}

/// What became of its mission: `John withdrew 'Tidy the lounge'.`
pub fn mission(by: &str, did: &str, title: &str) -> String {
    format!("{by} {did} '{}'.", trim_stop(title))
}

/// Where a line was said, as its journal names it: the phone for a direct
/// thread, the group by name otherwise.
pub fn on_thread(direct: bool, name: &str) -> String {
    match direct {
        true => "on the phone".to_string(),
        false => format!("on {name}"),
    }
}

fn trim_stop(s: &str) -> &str {
    s.trim().trim_end_matches('.')
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn each_line_reads_as_its_journal_writes_it() {
        assert_eq!(
            heard("Mira", &on_thread(true, "Mira"), " the parcel is at the door. "),
            "Mira, on the phone: the parcel is at the door."
        );
        assert_eq!(
            said("Mira", &on_thread(false, "#house"), "I'll bring it in."),
            "To Mira, on #house: I'll bring it in."
        );
        assert_eq!(
            promised("John", "the garden report.", "Friday"),
            "Promised John the garden report, by Friday."
        );
        assert_eq!(
            mission("John", "withdrew", "Tidy the lounge."),
            "John withdrew 'Tidy the lounge'."
        );
    }
}
