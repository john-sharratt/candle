//! The whole record, for readers that are not bodies.
//!
//! [`crate::witness`] answers what one body could make out from where it was
//! standing: its own room, and what it can see into. That is the right answer
//! for an NPC and the wrong one for everything else. A dispatch board has to
//! show every level at once. A persister has to write down what happened in
//! rooms nobody was in. A replay has to reproduce all of it.
//!
//! So those readers come here instead, and this module makes exactly one
//! promise the other cannot: **no place filtering, ever.**
//!
//! # This must never be handed to an NPC
//!
//! Every social mechanism in the vault rests on a Maker not being able to read
//! a colleague's work off a wall two floors up. One call to [`Stream::drain`]
//! wired into a percept would undo the green room, the relations table and the
//! trip upstairs to the dispatch board, and it would do it silently — the
//! prose would still read correctly, and the building would simply stop
//! mattering. If an NPC needs to know something, it walks to where that
//! something is legible.
//!
//! # Pull with a cursor, not push with a callback
//!
//! "Push" is the natural word for a stream and the wrong shape here. The world
//! lives behind one lock; a callback fired mid-mutation would re-enter it, and
//! a queue per subscriber would drop events whenever one fell behind. A cursor
//! over an append-only log gives every reader its own pace, loses nothing when
//! one stalls for an hour, and rewinds for a replay — and since the log is
//! ordered by [`Tick`], reading from a cursor is a slice, not a scan.

use std::collections::BTreeMap;

use crate::world::{Event, Tick, World};

/// Named cursors over the record, each at its own position.
///
/// One `Stream` serves many readers — the board, the persister, a test — and
/// none of them can hold another one up.
#[derive(Debug, Clone, Default)]
pub struct Stream {
    readers: BTreeMap<String, Tick>,
}

impl Stream {
    pub fn new() -> Stream {
        Stream::default()
    }

    /// Where a reader has got to. A reader that has never read starts at the
    /// beginning of the record, because a persister that skipped everything
    /// before it was created would be a persister with a hole in it.
    pub fn cursor(&self, reader: &str) -> Tick {
        self.readers.get(reader).copied().unwrap_or(0)
    }

    /// What a reader has not seen yet, without advancing it.
    ///
    /// Unfiltered: every event in the world, whoever caused it and wherever it
    /// happened.
    pub fn peek<'a>(&self, world: &'a World, reader: &str) -> &'a [Event] {
        after(world.log(), self.cursor(reader))
    }

    /// What a reader has not seen yet, advancing it past them.
    pub fn drain<'a>(&mut self, world: &'a World, reader: &str) -> &'a [Event] {
        let fresh = after(world.log(), self.cursor(reader));
        let to = fresh.last().map(|e| e.at).unwrap_or_else(|| world.now());
        self.readers.insert(reader.to_string(), to);
        fresh
    }

    /// Move a reader to a position — forward to skip, back to replay.
    pub fn seek(&mut self, reader: &str, to: Tick) {
        self.readers.insert(reader.to_string(), to);
    }

    /// Start a reader at the present, so it sees only what happens next.
    pub fn catch_up(&mut self, world: &World, reader: &str) {
        self.seek(reader, world.now());
    }

    /// Every reader and where it has got to. What a health check wants: a
    /// cursor a long way behind the others is a consumer that has stalled.
    pub fn readers(&self) -> impl Iterator<Item = (&str, Tick)> {
        self.readers.iter().map(|(k, v)| (k.as_str(), *v))
    }
}

/// The tail of the log after `cursor`.
///
/// The log is appended in tick order and never reordered, so this is a slice
/// found by bisection rather than a filter over everything that ever happened
/// — which matters once a vault has been running for a week.
fn after(log: &[Event], cursor: Tick) -> &[Event] {
    let start = log.partition_point(|e| e.at <= cursor);
    &log[start..]
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::load::MapSet;
    use crate::world::{Where, World};

    fn vault() -> World {
        World::new(
            MapSet::load_dir(concat!(env!("CARGO_MANIFEST_DIR"), "/maps"))
                .expect("the vault must load"),
        )
    }

    #[test]
    fn a_slice_after_a_cursor_skips_everything_up_to_it() {
        let mut w = vault();
        w.enter("m1", "Maker-01", Where::new("vault-casting", "band-one"))
            .unwrap();
        let mark = w.now();
        w.take("m1", Some("cindy")).unwrap();

        assert_eq!(after(w.log(), 0).len(), 2);
        assert_eq!(after(w.log(), mark).len(), 1);
        assert_eq!(after(w.log(), w.now()).len(), 0);
    }

    #[test]
    fn a_reader_that_has_never_read_starts_at_the_beginning() {
        let mut w = vault();
        w.enter("m1", "Maker-01", Where::new("vault-casting", "band-one"))
            .unwrap();
        let s = Stream::new();
        assert_eq!(s.cursor("persister"), 0);
        assert_eq!(s.peek(&w, "persister").len(), 1);
    }
}
