//! What a held character reads once it is let go: everything that reached it
//! while it was suspended, bounded per source.
//!
//! A suspended character takes no turn, and nothing that reaches it is dropped
//! — so a character held for a day under a busy channel comes back to
//! thousands of arrivals, and the first turn after it resumes would read every
//! one. [`bound`] keeps the newest [`PER_SOURCE`] from each source and puts one
//! counted line where the rest were: "While you were suspended, 214 more
//! messages on #general." A source is what an arrival came from — one thread, one speaker,
//! one thing watched, the world around it — so a burst on one channel does not
//! crowd out the one line somebody said to it directly.
//!
//! An arrival that only restates a standing fact (a situation, an unread count,
//! a standing instruction — [`EventKind::replaces`]) is superseded by the next
//! of its band anyway, so only the newest of each band is kept, and the rest go
//! without a line: nothing in them is missing from the one kept.

use std::collections::{HashMap, HashSet};

use crate::engine::event::{Event, EventKind, Salience};

/// How many arrivals from one source the turn after a resume reads.
pub const PER_SOURCE: usize = 20;

/// Where an arrival came from, and what to call those of it left out.
struct Source {
    key: String,
    /// One left out, and several.
    one: String,
    many: String,
    /// Superseded by the next of its band, so kept only as the newest.
    standing: bool,
}

fn source(kind: &EventKind) -> Source {
    let named = |key: String, one: String, many: String| Source {
        key,
        one,
        many,
        standing: false,
    };
    if let Some(band) = kind.replaces() {
        return Source {
            key: format!("band:{band}"),
            one: String::new(),
            many: String::new(),
            standing: true,
        };
    }
    match kind {
        EventKind::Message { thread, .. } => named(
            format!("message:{thread}"),
            format!("message on {thread}"),
            format!("messages on {thread}"),
        ),
        // Held for the conversation, never the mind; named should one wait.
        EventKind::Talk { thread, .. } => named(
            format!("talk:{thread}"),
            format!("line on {thread}"),
            format!("lines on {thread}"),
        ),
        EventKind::Purpose { party, .. } => named(
            format!("purpose:{party}"),
            format!("thing meant for {party}"),
            format!("things meant for {party}"),
        ),
        EventKind::Concluded { party, .. } => named(
            format!("concluded:{party}"),
            format!("thing settled with {party}"),
            format!("things settled with {party}"),
        ),
        EventKind::Speech { speaker, .. } => named(
            format!("speech:{speaker}"),
            format!("thing {speaker} said"),
            format!("things {speaker} said"),
        ),
        EventKind::Entity { entity_id, .. } => named(
            format!("entity:{entity_id}"),
            format!("thing you noticed of {entity_id}"),
            format!("things you noticed of {entity_id}"),
        ),
        EventKind::Description { .. } => named(
            "description".into(),
            "thing that happened around you".into(),
            "things that happened around you".into(),
        ),
        EventKind::Announcement { .. } => named(
            "announcement".into(),
            "word sent across the world".into(),
            "words sent across the world".into(),
        ),
        EventKind::Operator { .. } => named(
            "operator".into(),
            "line from the operator".into(),
            "lines from the operator".into(),
        ),
        EventKind::Heartbeat => named(
            "heartbeat".into(),
            "quiet moment".into(),
            "quiet moments".into(),
        ),
        EventKind::Sleep { .. } => named("sleep".into(), "day's end".into(), "days' ends".into()),
        EventKind::Wake { .. } => named("wake".into(), "new day".into(), "new days".into()),
        // Every banded kind returned above.
        EventKind::Unread { .. }
        | EventKind::Situation { .. }
        | EventKind::Nudge { .. }
        | EventKind::MindControl { .. } => named(kind.tag().into(), String::new(), String::new()),
    }
}

/// The line that stands for `n` arrivals of `src` left out.
fn left_out(src: &Source, n: usize) -> String {
    let noun = if n == 1 { &src.one } else { &src.many };
    format!("While you were suspended, {n} more {noun}.")
}

/// `queue`, bounded per source: the newest `keep` of each in their places,
/// and where the first of the rest stood, one line counting them — as urgent
/// as the most urgent of them, at the moment the first arrived, numbered by
/// `seq`. The newest of each standing band only, and no line for the rest.
pub fn bound(queue: Vec<Event>, keep: usize, mut seq: impl FnMut() -> u64) -> Vec<Event> {
    let sources: Vec<Source> = queue.iter().map(|e| source(&e.kind)).collect();
    let mut total: HashMap<&str, usize> = HashMap::new();
    for src in &sources {
        *total.entry(src.key.as_str()).or_insert(0) += 1;
    }
    // Which are left out — the oldest of each source past its bound — and the
    // loudest of those, per source.
    let mut seen: HashMap<&str, usize> = HashMap::new();
    let mut loudest: HashMap<&str, Salience> = HashMap::new();
    let out_of: Vec<bool> = queue
        .iter()
        .zip(&sources)
        .map(|(e, src)| {
            let n = total[src.key.as_str()];
            let kept = if src.standing { 1 } else { keep };
            let i = seen.entry(src.key.as_str()).or_insert(0);
            *i += 1;
            let left = n > kept && *i <= n - kept;
            if left {
                let l = loudest.entry(src.key.as_str()).or_insert(e.salience);
                *l = l.max_of(e.salience);
            }
            left
        })
        .collect();
    // One line per source, where the first of those it counts stood.
    let mut told: HashSet<&str> = HashSet::new();
    let mut out = Vec::with_capacity(queue.len());
    for ((e, src), left) in queue.into_iter().zip(&sources).zip(out_of) {
        if !left {
            out.push(e);
            continue;
        }
        if src.standing || !told.insert(src.key.as_str()) {
            continue;
        }
        let n = total[src.key.as_str()] - keep;
        out.push(Event::new(
            seq(),
            e.at_ms,
            loudest[src.key.as_str()],
            EventKind::Description {
                text: left_out(src, n),
            },
        ));
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn message(thread: &str, unread: usize) -> EventKind {
        EventKind::Message {
            thread: thread.into(),
            from: "Kit".into(),
            unread,
        }
    }

    fn prose(events: &[Event]) -> Vec<String> {
        events.iter().map(Event::prose).collect()
    }

    /// **A source past its bound keeps its newest and counts the rest in one
    /// line where they began**; a source within it is untouched, and only the
    /// newest situation is kept, with no line for the others.
    #[test]
    fn each_source_keeps_its_newest_and_counts_the_rest() {
        let mut q = Vec::new();
        let mut at = 0;
        let mut push = |kind: EventKind, s: Salience| {
            at += 1;
            q.push(Event::new(at, at, s, kind));
        };
        for i in 1..=5 {
            let s = if i == 2 {
                Salience::URGENT
            } else {
                Salience::NORMAL
            };
            push(message("#general", i), s);
        }
        push(
            EventKind::Description {
                text: "A door slams.".into(),
            },
            Salience::NORMAL,
        );
        push(message("#general", 6), Salience::NORMAL);
        push(
            EventKind::Situation {
                text: "You are in the hall.".into(),
            },
            Salience::NORMAL,
        );
        push(
            EventKind::Situation {
                text: "You are in the yard.".into(),
            },
            Salience::NORMAL,
        );

        let mut next = 100;
        let out = bound(q, 2, || {
            next += 1;
            next
        });
        assert_eq!(
            prose(&out),
            vec![
                "While you were suspended, 4 more messages on #general.".to_string(),
                "Your phone shows a message from Kit on #general (5 unread messages in all). \
                 `read` #general to see what was said, and answer it there with `reply`, not \
                 to Kit alone."
                    .to_string(),
                "A door slams.".to_string(),
                "Your phone shows a message from Kit on #general (6 unread messages in all). \
                 `read` #general to see what was said, and answer it there with `reply`, not \
                 to Kit alone."
                    .to_string(),
                "You are in the yard.".to_string(),
            ]
        );
        let line = &out[0];
        assert_eq!(
            (line.seq, line.at_ms, line.salience),
            (101, 1, Salience::URGENT),
            "as urgent as the loudest left out, where the first of them stood"
        );
    }

    /// One left out reads in the singular, and a queue within every bound
    /// comes back as it went in.
    #[test]
    fn one_left_out_is_singular_and_a_short_queue_is_untouched() {
        let q: Vec<Event> = (1..=3)
            .map(|i| {
                Event::new(
                    i,
                    i,
                    Salience::NORMAL,
                    EventKind::Speech {
                        speaker: "Wren".into(),
                        text: format!("line {i}"),
                        to: Default::default(),
                    },
                )
            })
            .collect();
        let short = bound(q.clone(), 3, || 0);
        assert_eq!(
            short.iter().map(|e| e.seq).collect::<Vec<_>>(),
            vec![1, 2, 3]
        );
        let out = bound(q, 2, || 9);
        assert_eq!(
            out[0].prose(),
            "While you were suspended, 1 more thing Wren said."
        );
        assert_eq!(out.iter().map(|e| e.seq).collect::<Vec<_>>(), vec![9, 2, 3]);
    }
}
