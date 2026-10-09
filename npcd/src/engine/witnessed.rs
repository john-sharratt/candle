//! A refusal the room can see.
//!
//! **A failure nobody else can see is one nobody else can help with.** A Maker
//! whose call to a time machine named no year was refused in a line only it
//! read; it told its colleagues it needed to know what had happened in 2950,
//! and they — who could not see the call, only what it made of the refusal —
//! told it what they guessed. When a Maker works something in the room and is
//! refused, the room sees it (`World::show`): who worked what, and what it said
//! back. A colleague can then say what was wrong, and the one refused can say
//! what actually happened rather than what it concluded.
//!
//! **Not a document's.** Work at a writing desk is refused for what the text
//! says — an edit that quoted something not there, a piece under its length —
//! and the room seeing that only confused it: a Maker took a colleague's
//! refused edit for its own and said its edits were being rejected when they
//! were landing, and another copied the refusal it overheard into its story
//! word for word. The bench's own acts ([`body::document_work`]) are refused
//! in private.

use crate::engine::act::Act;
use crate::engine::body::{self, document_work};
use crate::engine::tools::CATALOG;
use crate::world::Hosted;

/// The longest a refusal is quoted to the room, in characters.
const QUOTED: usize = 160;

/// Let the room `body` stands in see that something standing there refused
/// its `act`, with the first sentence of `answer`. Only an act at a part — a
/// station act, or an `invoke` of a device here — and never work on a
/// document; a refused walk or word is nobody else's to watch either, and
/// nothing is shown for them.
pub fn show(hosted: &Hosted, body: &str, act: &Act, answer: &str) {
    if document_work(act) {
        return;
    }
    let addressed = act
        .args
        .get("url")
        .and_then(|v| v.as_str())
        .and_then(part_of_url);
    let at_part = |id: &str| match body::is_device(act.tool) {
        true => addressed == Some(id),
        false => CATALOG
            .iter()
            .any(|t| t.name == act.tool && t.at.contains(&id)),
    };
    let thing = hosted.read(|w| {
        let node = w.actor(body).and_then(|a| w.node(&a.at))?;
        w.map()
            .parts_at(node)
            .find(|(part, _)| at_part(&part.id))
            .map(|(part, _)| part.name.clone())
    });
    let Some(thing) = thing else {
        return;
    };
    if let Err(why) = hosted.with(|w| w.show(body, None, refused(&thing, answer))) {
        tracing::debug!(npc = %body, "a refusal the room could not be shown: {why}");
    }
}

/// What the room sees when its `thing` refuses somebody, as the deed the
/// world's `show` takes — a verb phrase with no actor: "works the time
/// machine, and it refuses: …".
pub fn refused(thing: &str, answer: &str) -> String {
    let thing = ["a ", "an ", "the "]
        .iter()
        .find_map(|article| thing.strip_prefix(article))
        .unwrap_or(thing);
    format!(
        "works the {thing}, and it refuses: {}",
        first_sentence(answer)
    )
}

/// The part an `invoke` addressed, by id, from its address:
/// `http://local/time/time-machine~9/travel` → `time-machine`.
pub fn part_of_url(url: &str) -> Option<&str> {
    let path = url.split("://").nth(1).unwrap_or(url);
    let instance = path.split('/').nth(2)?;
    Some(instance.split('~').next().unwrap_or(instance)).filter(|p| !p.is_empty())
}

/// The first sentence of `s`, on one line, at most [`QUOTED`] characters.
fn first_sentence(s: &str) -> String {
    let flat = s.split_whitespace().collect::<Vec<_>>().join(" ");
    let end = flat
        .char_indices()
        .find(|&(i, c)| matches!(c, '.' | '!' | '?') && flat[i + c.len_utf8()..].starts_with(' '))
        .map(|(i, c)| i + c.len_utf8())
        .unwrap_or(flat.len());
    let sentence = &flat[..end];
    match sentence.chars().count() > QUOTED {
        true => format!("{}…", sentence.chars().take(QUOTED).collect::<String>()),
        false => sentence.to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_room_sees_who_was_refused_by_what_and_why() {
        assert_eq!(
            refused(
                "time machine",
                "Your call named no year — the machine needs the year written in it. Your \
                 mission asks you to work in 2950."
            ),
            "works the time machine, and it refuses: Your call named no year — the machine \
             needs the year written in it."
        );
    }

    /// A part named with its article is named once.
    #[test]
    fn a_part_named_with_an_article_is_not_doubled() {
        assert_eq!(
            refused("a writing desk", "Nothing is written there yet."),
            "works the writing desk, and it refuses: Nothing is written there yet."
        );
    }

    #[test]
    fn an_invoke_names_the_part_it_addressed() {
        assert_eq!(
            part_of_url("http://local/time/time-machine~9/travel"),
            Some("time-machine")
        );
        assert_eq!(
            part_of_url("http://local/command/order-table~1/report_done"),
            Some("order-table")
        );
        assert_eq!(part_of_url("http://local/"), None);
    }
}
