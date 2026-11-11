//! The lines a member has not written yet (`docs/npc_conversations.md` §5.2):
//! per bot, swapped out whole at each flush, so the beat path takes the lock
//! only to push.
//!
//! **Bounded two ways.** A bot's buffer holds at most [`MAX_EPISODES`], newest
//! kept, so a flush is always one op. And a bot writes at most [`DAY_LINES`] a
//! day: past that, the day's oldest `did` and `heard` lines go first, and a
//! line of those kinds with none of them left to give way is not kept.

use std::collections::BTreeMap;
use std::mem;
use std::sync::Mutex;

use journals::experience::ExperienceKind;
use journals::ops::episodes::{Lived, MAX_EPISODES};
use journals::path::JournalPath;

/// The most lines a bot writes in a day.
pub const DAY_LINES: usize = 200;

/// A day, as the cap counts them: days since the epoch, in UTC.
fn day_of(lived: &Lived) -> i64 {
    lived.at.secs().div_euclid(86_400)
}

/// Whether a line gives way first when a bot's day is full.
fn gives_way(kind: ExperienceKind) -> bool {
    matches!(kind, ExperienceKind::Did | ExperienceKind::Heard)
}

#[derive(Default)]
struct Held {
    /// Each bot's journal and its unwritten lines, oldest first.
    lines: BTreeMap<JournalPath, Vec<Lived>>,
    /// How many lines each bot has taken today: written and unwritten.
    days: BTreeMap<JournalPath, (i64, usize)>,
}

/// Every bot's unwritten lines on this member.
#[derive(Default)]
pub struct Buffer {
    held: Mutex<Held>,
}

impl Buffer {
    pub fn new() -> Buffer {
        Buffer::default()
    }

    /// Keep `lived` for `bot`'s next flush, within the bot's caps.
    pub fn push(&self, bot: &JournalPath, lived: Lived) {
        let mut guard = self.held.lock().unwrap_or_else(|e| e.into_inner());
        let held = &mut *guard;
        let day = day_of(&lived);
        let taken = match held.days.get(bot) {
            Some((d, n)) if *d == day => *n,
            _ => 0,
        };
        let lines = held.lines.entry(bot.clone()).or_default();
        if taken >= DAY_LINES {
            let yields = lines
                .iter()
                .position(|l| day_of(l) == day && gives_way(l.experience.kind));
            match yields {
                Some(i) => {
                    lines.remove(i);
                }
                None => return,
            }
        } else {
            held.days.insert(bot.clone(), (day, taken + 1));
        }
        lines.push(lived);
        if lines.len() > MAX_EPISODES {
            let over = lines.len() - MAX_EPISODES;
            lines.drain(..over);
        }
    }

    /// Everything unwritten, taken out to be written.
    pub fn take(&self) -> BTreeMap<JournalPath, Vec<Lived>> {
        let mut held = self.held.lock().unwrap_or_else(|e| e.into_inner());
        mem::take(&mut held.lines)
    }

    /// Put back a batch that could not be written for a passing reason,
    /// ahead of anything pushed since, within the buffer's cap.
    pub fn restore(&self, bot: &JournalPath, mut batch: Vec<Lived>) {
        let mut held = self.held.lock().unwrap_or_else(|e| e.into_inner());
        let lines = held.lines.entry(bot.clone()).or_default();
        batch.append(lines);
        if batch.len() > MAX_EPISODES {
            let over = batch.len() - MAX_EPISODES;
            batch.drain(..over);
        }
        *lines = batch;
    }

    /// `bot`'s unwritten lines, oldest first: what a catch-up on this member
    /// merges with what its journal already holds (§6.1).
    pub fn pending(&self, bot: &JournalPath) -> Vec<Lived> {
        let held = self.held.lock().unwrap_or_else(|e| e.into_inner());
        held.lines.get(bot).cloned().unwrap_or_default()
    }
}

#[cfg(test)]
mod tests {
    use journals::experience::Experience;
    use journals::path::Viewer;
    use journals::time::Stamp;

    use super::*;

    fn bot(name: &str) -> JournalPath {
        Viewer {
            group: None,
            here: None,
        }
        .parse(&format!("@moon/bots/{name}"))
        .unwrap()
    }

    fn at(secs: i64) -> Stamp {
        Stamp::parse("2026-11-10T00:00:00Z").unwrap().plus(secs)
    }

    fn line(secs: i64, kind: ExperienceKind, text: &str) -> Lived {
        Lived {
            at: at(secs),
            text: text.into(),
            experience: Experience::of(kind),
        }
    }

    fn texts(lines: &[Lived]) -> Vec<&str> {
        lines.iter().map(|l| l.text.as_str()).collect()
    }

    #[test]
    fn a_flush_takes_every_bot_s_lines_and_leaves_none() {
        let b = Buffer::new();
        b.push(&bot("zen"), line(1, ExperienceKind::Heard, "a"));
        b.push(&bot("wick"), line(2, ExperienceKind::Said, "b"));
        b.push(&bot("zen"), line(3, ExperienceKind::Did, "c"));
        let taken = b.take();
        assert_eq!(texts(&taken[&bot("zen")]), ["a", "c"]);
        assert_eq!(texts(&taken[&bot("wick")]), ["b"]);
        assert!(b.take().is_empty());
    }

    #[test]
    fn a_batch_put_back_goes_ahead_of_what_came_since() {
        let b = Buffer::new();
        b.push(&bot("zen"), line(1, ExperienceKind::Heard, "a"));
        let taken = b.take().remove(&bot("zen")).unwrap();
        b.push(&bot("zen"), line(2, ExperienceKind::Said, "b"));
        b.restore(&bot("zen"), taken);
        assert_eq!(texts(&b.pending(&bot("zen"))), ["a", "b"]);
    }

    #[test]
    fn a_bot_s_buffer_keeps_its_newest_lines() {
        let b = Buffer::new();
        let batch = (0..(MAX_EPISODES + 5))
            .map(|i| line(i as i64, ExperienceKind::Said, &i.to_string()))
            .collect();
        b.restore(&bot("zen"), batch);
        let held = b.pending(&bot("zen"));
        assert_eq!(held.len(), MAX_EPISODES);
        assert_eq!(held.first().unwrap().text, "5");
        assert_eq!(held.last().unwrap().text, (MAX_EPISODES + 4).to_string());
    }

    #[test]
    fn a_full_day_gives_up_its_oldest_did_and_heard_lines_first() {
        let b = Buffer::new();
        b.push(&bot("zen"), line(0, ExperienceKind::Said, "said-0"));
        b.push(&bot("zen"), line(1, ExperienceKind::Did, "did-1"));
        b.push(&bot("zen"), line(2, ExperienceKind::Heard, "heard-2"));
        for i in 3..DAY_LINES as i64 {
            b.push(&bot("zen"), line(i, ExperienceKind::Said, "said"));
        }
        // The day is full: a promise takes the oldest act's place, and the
        // next takes the heard line's.
        b.push(&bot("zen"), line(500, ExperienceKind::Promised, "promised"));
        b.push(
            &bot("zen"),
            line(501, ExperienceKind::Promised, "promised-2"),
        );
        let held = b.pending(&bot("zen"));
        assert_eq!(held.len(), DAY_LINES);
        assert_eq!(texts(&held[..2]), ["said-0", "said"]);
        assert_eq!(texts(&held[held.len() - 2..]), ["promised", "promised-2"]);
        // With nothing left to give way, a heard line is not kept.
        b.push(&bot("zen"), line(502, ExperienceKind::Heard, "heard-late"));
        assert_eq!(b.pending(&bot("zen")).last().unwrap().text, "promised-2");
        // A new day starts the count again.
        b.push(&bot("zen"), line(86_400, ExperienceKind::Heard, "tomorrow"));
        assert_eq!(b.pending(&bot("zen")).last().unwrap().text, "tomorrow");
    }
}
