//! The standing reminder of what waits unread on a character's handset.
//!
//! A message reaches a character's mind once, the moment after it arrives
//! (`environment::deliver_messages`). A character busy with something else
//! moved past it, and nothing told it again: a person's Discord message to Zen
//! sat unread while Zen worked a routine, and the count of it surfaced only in
//! a `scan` Zen had no reason to make. So while anything waits unread, the
//! character is told how much and where — an [`EventKind::Unread`] that
//! replaces the last one — whenever that changes, and again every
//! [`REMIND_EVERY_MS`] it stays the same. Reading clears it: nothing unread,
//! no reminder.
//!
//! [`EventKind::Unread`]: crate::engine::event::EventKind::Unread

use serde::{Deserialize, Serialize};

use super::phone::{Kind, Threads};

/// How long the same unread messages wait before the character is reminded
/// of them again.
pub const REMIND_EVERY_MS: u64 = 5 * 60 * 1000;

/// What a member was last reminded of, and when.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Reminded {
    /// Each thread with something unread, by the member's name for it, and how
    /// much.
    pub threads: Vec<(String, usize)>,
    pub at_ms: u64,
}

/// A reminder due now: what waits, and whether any of it was said to the
/// character itself — a direct thread, or a group line addressed to it —
/// which makes it as urgent as the messages were.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Due {
    pub threads: Vec<(String, usize)>,
    pub to_me: bool,
}

impl Threads {
    /// What waits unread for `who`, thread by thread, in the order the threads
    /// are kept.
    pub fn unread_threads(&self, who: &str) -> Vec<(String, usize)> {
        self.of(who)
            .into_iter()
            .filter_map(|t| {
                let n = t.unread_for(who);
                (n > 0).then(|| (t.as_named_to(who), n))
            })
            .collect()
    }

    /// Whether anything unread for `who` was said to it: on a direct thread,
    /// or a group line addressed to it by name.
    fn unread_to(&self, who: &str) -> bool {
        self.of(who).into_iter().any(|t| {
            t.messages
                .iter()
                .skip(t.arrived_at(who))
                .filter(|m| m.from != who && !m.seen_by.iter().any(|s| s == who))
                .any(|m| t.kind() == Kind::Direct || m.to.as_deref() == Some(who))
        })
    }

    /// The reminder due `who` at `now_ms`, recording it as given — or `None`
    /// — and how much waits unread for it in all, every thread together: the
    /// sum the reminder is judged from, so a caller needs no second scan of
    /// the threads for it.
    ///
    /// Due when something waits unread and it differs from what `who` was
    /// last reminded of, or [`REMIND_EVERY_MS`] has passed since. `told_now`
    /// says `who` was just handed new messages, which tell it all a reminder
    /// would: what waits is recorded as reminded, and nothing is due.
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

    const T0: u64 = 1_000_000;

    /// Zen on a server channel and a DM, each with something said to it.
    fn waiting() -> Threads {
        let mut p = Threads::new();
        let general = p.open_standing("zenworld #general", &["Zen".to_string()]);
        p.join(&general, "Kit (@kit)");
        p.send_on(&general, "Kit (@kit)", "anyone?", None).unwrap();
        p.send_on(&general, "Kit (@kit)", "hello?", None).unwrap();
        p
    }

    /// **Something unread is reminded once, and again only when it changes
    /// or the interval passes**; nothing unread, nothing due.
    #[test]
    fn a_reminder_is_due_when_what_waits_changes_or_the_interval_passes() {
        let mut p = waiting();
        assert_eq!(
            p.unread_due("Zen", T0, false),
            (
                Some(Due {
                    threads: vec![("zenworld #general".to_string(), 2)],
                    to_me: false,
                }),
                2
            )
        );
        assert_eq!(
            p.unread_due("Zen", T0 + 1_000, false),
            (None, 2),
            "said already, and still waiting"
        );
        assert!(p
            .unread_due("Zen", T0 + REMIND_EVERY_MS, false)
            .0
            .is_some_and(|d| d.threads == [("zenworld #general".to_string(), 2)]));

        p.reach("John", "Zen");
        p.send("John", "Zen", "are you there?").unwrap();
        assert_eq!(
            p.unread_due("Zen", T0 + REMIND_EVERY_MS + 1, false),
            (
                Some(Due {
                    threads: vec![
                        ("zenworld #general".to_string(), 2),
                        ("John".to_string(), 1),
                    ],
                    to_me: true,
                }),
                3
            ),
            "a new message changes it, and a direct one is said to it"
        );
        assert_eq!(
            p.unread_due("Zen", T0 + REMIND_EVERY_MS + 2, false).1,
            p.waiting_for("Zen")
        );

        assert_eq!(
            p.unread_due("Kit (@kit)", T0, false),
            (None, 0),
            "its own lines"
        );
        assert_eq!(p.unread_due("Nobody", T0, false), (None, 0));
    }

    /// **Reading clears it**: once every thread is read, nothing is due, and
    /// what arrives after is reminded afresh.
    #[test]
    fn reading_clears_the_reminder() {
        let mut p = waiting();
        assert!(p.unread_due("Zen", T0, false).0.is_some());
        p.read("Zen", "zenworld #general").unwrap();
        assert_eq!(p.unread_due("Zen", T0 + REMIND_EVERY_MS, false), (None, 0));
        assert!(p.reminded.is_empty());
        let general = p.of("Zen")[0].id.clone();
        p.send_on(&general, "Kit (@kit)", "still there?", None)
            .unwrap();
        assert_eq!(
            p.unread_due("Zen", T0 + REMIND_EVERY_MS + 1, false)
                .0
                .map(|d| d.threads),
            Some(vec![("zenworld #general".to_string(), 1)])
        );
    }

    /// **A character just handed its messages is not reminded of them**: it
    /// has been told, and the reminder waits for the interval from there.
    #[test]
    fn messages_just_handed_count_as_reminded() {
        let mut p = waiting();
        assert_eq!(p.unread_due("Zen", T0, true), (None, 2));
        assert_eq!(p.unread_due("Zen", T0 + 1_000, false), (None, 2));
        assert!(p.unread_due("Zen", T0 + REMIND_EVERY_MS, false).0.is_some());
    }

    /// **A group line said to the character by name makes it urgent.**
    #[test]
    fn a_group_line_said_to_the_character_is_said_to_it() {
        let mut p = waiting();
        let general = p.of("Zen")[0].id.clone();
        p.send_to(&general, "Kit (@kit)", "Zen?", None, Some("Zen"))
            .unwrap();
        assert!(p.unread_due("Zen", T0, false).0.is_some_and(|d| d.to_me));
    }
}
