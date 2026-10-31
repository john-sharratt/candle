//! The last [`KEEP`] events of one bot's bridge, newest kept.

use std::collections::VecDeque;

use super::note::Note;

/// How many events a bot's trace keeps.
pub const KEEP: usize = 100;

/// A bot's last events, oldest first.
#[derive(Debug, Clone, Default)]
pub struct Ring {
    notes: VecDeque<Note>,
}

impl Ring {
    /// Keep `note`, letting the oldest go past [`KEEP`].
    pub fn push(&mut self, note: Note) {
        if self.notes.len() == KEEP {
            self.notes.pop_front();
        }
        self.notes.push_back(note);
    }

    /// What it holds, newest first.
    pub fn newest_first(&self) -> Vec<Note> {
        self.notes.iter().rev().cloned().collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::discord::trace::note::Direction;

    #[test]
    fn a_ring_keeps_the_newest_hundred_newest_first() {
        let mut ring = Ring::default();
        for at in 0..(KEEP as u64 + 5) {
            ring.push(Note::new(at, Direction::In, "dm", "applied"));
        }
        let kept: Vec<u64> = ring.newest_first().iter().map(|n| n.at_ms).collect();
        assert_eq!(kept.len(), KEEP);
        assert_eq!((kept[0], kept[KEEP - 1]), (KEEP as u64 + 4, 5));
    }
}
