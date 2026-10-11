//! What a world is doing, as pulses (`docs/replicated_substrate.md` §9.1): each
//! beat the authority sends what happened in the world since its last pulse —
//! a line spoken, a body arriving, the building stirring — and every member
//! following the world takes it into its own log, where its characters perceive
//! it. A pulse is never persisted and never sent twice: one missed costs only
//! that moment, and the next state record settles the world.

use npc_map::world::{Event, Tick};

/// A pulse's kind: the world's events since the last one.
pub const EVENTS: &str = "events";

/// The events in `log` after tick `after`, and the tick of the last one.
pub fn since(log: &[Event], after: Tick) -> (Vec<Event>, Tick) {
    let events: Vec<Event> = log.iter().filter(|e| e.at > after).cloned().collect();
    let last = events.last().map_or(after, |e| e.at);
    (events, last)
}

pub fn encode(events: &[Event]) -> Vec<u8> {
    serde_json::to_vec(events).expect("a world's events always encode")
}

pub fn decode(body: &[u8]) -> Result<Vec<Event>, serde_json::Error> {
    serde_json::from_slice(body)
}

#[cfg(test)]
mod tests {
    use npc_map::world::{Happening, Voice, Where};

    use super::*;

    fn said(at: Tick, words: &str) -> Event {
        Event {
            at,
            actor: "m1".into(),
            place: Where::new("vault-casting", "green-room"),
            what: Happening::Said {
                to: None,
                words: words.into(),
                voice: Voice::Said,
            },
        }
    }

    /// Only what came after the last pulse, and the tick to start from next.
    #[test]
    fn a_pulse_carries_what_happened_since_the_last() {
        let log = vec![said(3, "one"), said(5, "two"), said(8, "three")];
        let (events, last) = since(&log, 5);
        assert_eq!(events, vec![said(8, "three")]);
        assert_eq!(last, 8);
        assert_eq!(since(&log, 8), (Vec::new(), 8));
        assert_eq!(decode(&encode(&events)).unwrap(), events);
    }
}
