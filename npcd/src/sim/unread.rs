//! The standing reminder of what waits unread on a character's groups and
//! channels.
//!
//! A line reaches a character's mind once, the moment after it arrives
//! (`environment::deliver_messages`). A character busy with something else
//! moved past it, so while anything waits unread it is told how much and
//! where — an [`EventKind::Unread`] that replaces the last one — whenever
//! that changes, and again every [`REMIND_EVERY_MS`] it stays the same.
//! Reading clears it.
//!
//! **Only groups and channels.** A direct thread is a conversation's: its
//! lines are answered by the conversation as they land (`engine::talk`), so
//! the mind is never reminded of them.
//!
//! [`EventKind::Unread`]: crate::engine::event::EventKind::Unread

use serde::{Deserialize, Serialize};

use super::phone::Threads;

/// How long the same unread lines wait before the character is reminded of
/// them again.
pub const REMIND_EVERY_MS: u64 = 5 * 60 * 1000;

/// What a member was last reminded of, and when.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Reminded {
    /// Each group with something unread, by the member's name for it, and
    /// how much.
    pub threads: Vec<(String, usize)>,
    pub at_ms: u64,
}

/// A reminder due now: what waits, and whether any of it was said to the
/// character by name, which makes it as urgent as a direct line.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Due {
    pub threads: Vec<(String, usize)>,
    pub to_me: bool,
}

impl Threads {
    /// What waits unread for `who` on its groups, thread by thread, in the
    /// order the threads are kept.
    pub fn unread_threads(&self, who: &str) -> Vec<(String, usize)> {
        self.groups_of(who)
            .into_iter()
            .filter_map(|t| {
                let n = t.unread_for(who);
                (n > 0).then(|| (t.as_named_to(who), n))
            })
            .collect()
    }

    /// Whether anything unread for `who` on its groups was said to it by
    /// name.
    fn unread_to(&self, who: &str) -> bool {
        self.groups_of(who).into_iter().any(|t| {
            t.unread(who)
                .into_iter()
                .any(|m| m.to.as_deref() == Some(who))
        })
    }

    /// The reminder due `who` at `now_ms`, recording it as given — or `None`
    /// — and how much waits unread for it in all.
    ///
    /// Due when something waits unread and it differs from what `who` was
    /// last reminded of, or [`REMIND_EVERY_MS`] has passed since. `told_now`
    /// says `who` was just handed new lines, which tell it all a reminder
    /// would.
    pub fn unread_due(&mut self, who: &str, now_ms: u64, told_now: bool) -> (Option<Due>, usize) {
        let threads = self.unread_threads(who);
        let unread = threads.iter().map(|(_, n)| n).sum();
        if threads.is_empty() {
            self.reminded.remove(who);
            return (None, unread);
        }
        let stale = self.reminded.get(who).is_none_or(|r| {
            r.threads != threads || now_ms.saturating_sub(r.at_ms) >= REMIND_EVERY_MS
        });
        if !stale {
            return (None, unread);
        }
        let to_me = self.unread_to(who);
        self.reminded.insert(
            who.to_string(),
            Reminded {
                threads: threads.clone(),
                at_ms: now_ms,
            },
        );
        ((!told_now).then_some(Due { threads, to_me }), unread)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sim::phone::{Said, Who};

    const T0: u64 = 1_000_000;

    fn who(id: &str, name: &str) -> Who {
        Who::new(id, name)
    }

    /// Zen on a server channel with two lines waiting, and a DM with one.
    fn waiting() -> (Threads, String) {
        let mut p = Threads::new();
        let general = p.open_standing("zenworld #general", &[who("npc-1", "Zen")]);
        p.join(&general, &who("u_kit", "Kit"));
        p.send_on(&general, "u_kit", Said::words("anyone?"))
            .unwrap();
        p.send_on(&general, "u_kit", Said::words("hello?")).unwrap();
        let dm = p.reach(&who("u_john", "John"), &who("npc-1", "Zen"));
        p.send_on(&dm, "u_john", Said::words("are you there?"))
            .unwrap();
        (p, general)
    }

    /// **Something unread is reminded once, and again only when it changes
    /// or the interval passes**; a direct line is never reminded.
    #[test]
    fn a_reminder_is_due_when_what_waits_changes_or_the_interval_passes() {
        let (mut p, general) = waiting();
        assert_eq!(
            p.unread_due("npc-1", T0, false),
            (
                Some(Due {
                    threads: vec![("zenworld #general".to_string(), 2)],
                    to_me: false,
                }),
                2
            ),
            "the DM is the conversation's, not the reminder's"
        );
        assert_eq!(p.unread_due("npc-1", T0 + 1_000, false), (None, 2));
        assert!(p
            .unread_due("npc-1", T0 + REMIND_EVERY_MS, false)
            .0
            .is_some());
        p.send_on(&general, "u_kit", Said::words("still?")).unwrap();
        assert_eq!(
            p.unread_due("npc-1", T0 + REMIND_EVERY_MS + 1, false)
                .0
                .map(|d| d.threads),
            Some(vec![("zenworld #general".to_string(), 3)])
        );
        assert_eq!(p.unread_due("u_kit", T0, false), (None, 0), "its own lines");
    }

    #[test]
    fn reading_clears_the_reminder() {
        let (mut p, _) = waiting();
        assert!(p.unread_due("npc-1", T0, false).0.is_some());
        p.read("npc-1", "zenworld #general").unwrap();
        assert_eq!(
            p.unread_due("npc-1", T0 + REMIND_EVERY_MS, false),
            (None, 0)
        );
        assert!(p.reminded.is_empty());
    }

    #[test]
    fn lines_just_handed_count_as_reminded() {
        let (mut p, _) = waiting();
        assert_eq!(p.unread_due("npc-1", T0, true), (None, 2));
        assert_eq!(p.unread_due("npc-1", T0 + 1_000, false), (None, 2));
        assert!(p
            .unread_due("npc-1", T0 + REMIND_EVERY_MS, false)
            .0
            .is_some());
    }

    /// **A group line said to the character by name makes it urgent.**
    #[test]
    fn a_group_line_said_to_the_character_is_said_to_it() {
        let (mut p, general) = waiting();
        let mut said = Said::words("Zen?");
        said.to = Some("npc-1".into());
        p.send_on(&general, "u_kit", said).unwrap();
        assert!(p.unread_due("npc-1", T0, false).0.is_some_and(|d| d.to_me));
    }
}
