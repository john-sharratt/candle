//! The section generations a `SectionTombstone` has ended
//! (`docs/replicated_substrate.md` §6.5).
//!
//! A section is content-addressed, so a rebuilt section lands on the stream id
//! the ended one had. A stamped tombstone names the **generation** it ends —
//! the writer and the `oseq` of the declaration that started it — and that
//! generation stays ended in any arrival order: its declaration is refused and
//! every record that carries it in its `epoch` is dropped, before or after the
//! tombstone. A later generation of the same section is untouched.
//!
//! A tombstone written before generations names none. It ends the unstamped
//! stream as the log stands at that point, and a later unstamped declaration
//! starts the stream again — the append-order rule every unstamped record
//! keeps.

use crate::persistence::record::SectionGeneration;
use crate::persistence::streams::StreamId;

use super::pcoll::POrdSet;

/// See the module documentation. Persistent: a clone is O(1).
#[derive(Debug, Default, Clone)]
pub struct EndedSections {
    generations: POrdSet<(StreamId, SectionGeneration)>,
    unstamped: POrdSet<StreamId>,
}

impl EndedSections {
    /// `generation` of section `stream_id` has ended.
    pub fn end(&mut self, stream_id: StreamId, generation: SectionGeneration) {
        self.generations.insert((stream_id, generation));
    }

    /// The unstamped stream `stream_id` has ended.
    pub fn end_unstamped(&mut self, stream_id: StreamId) {
        self.unstamped.insert(stream_id);
    }

    /// An unstamped declaration started `stream_id` again.
    pub fn lift_unstamped(&mut self, stream_id: StreamId) {
        self.unstamped.remove(&stream_id);
    }

    /// Whether `generation` of section `stream_id` has ended.
    pub fn is_ended(&self, stream_id: StreamId, generation: SectionGeneration) -> bool {
        self.generations.contains(&(stream_id, generation))
    }

    /// Whether any generation of `stream_id` has ended.
    pub fn any(&self, stream_id: StreamId) -> bool {
        self.unstamped.contains(&stream_id)
            || self
                .generations
                .range(
                    (stream_id, SectionGeneration { writer: 0, oseq: 0 })
                        ..=(
                            stream_id,
                            SectionGeneration {
                                writer: u64::MAX,
                                oseq: u64::MAX,
                            },
                        ),
                )
                .next()
                .is_some()
    }

    /// Every ended generation, in order.
    pub fn generations(&self) -> impl Iterator<Item = (StreamId, SectionGeneration)> + '_ {
        self.generations.iter().copied()
    }

    /// Every ended unstamped stream, in order.
    pub fn unstamped(&self) -> impl Iterator<Item = StreamId> + '_ {
        self.unstamped.iter().copied()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn g(writer: u64, oseq: u64) -> SectionGeneration {
        SectionGeneration { writer, oseq }
    }

    #[test]
    fn a_generation_ends_alone() {
        let mut e = EndedSections::default();
        e.end(StreamId(7), g(3, 10));
        assert!(e.is_ended(StreamId(7), g(3, 10)));
        assert!(!e.is_ended(StreamId(7), g(3, 11)), "a later generation");
        assert!(!e.is_ended(StreamId(7), g(4, 10)), "another writer's");
        assert!(e.any(StreamId(7)));
        assert!(!e.any(StreamId(8)));
        assert!(!e.any(StreamId(6)));
    }

    #[test]
    fn an_unstamped_stream_ends_until_declared_again() {
        let mut e = EndedSections::default();
        e.end_unstamped(StreamId(9));
        assert!(e.any(StreamId(9)));
        e.lift_unstamped(StreamId(9));
        assert!(!e.any(StreamId(9)));
        assert_eq!(e.unstamped().count(), 0);
    }
}
