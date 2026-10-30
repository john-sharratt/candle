//! The section generations this node is writing (`docs/replicated_substrate.md`
//! §6.5).
//!
//! A section is content-addressed, so a section rebuilt after a tombstone is
//! written under the stream id the ended one had. What tells the two apart is
//! the **generation**: the `oseq` of the declaration that started it, which
//! every record of the generation carries in its `epoch`.

use std::collections::{HashMap, HashSet};

use super::super::record::{RecordHeader, RecordType, SectionTombstonePayload};
use super::super::streams::declares_section;

/// Section stream id → the `oseq` of this node's declaration that started the
/// generation it is writing.
#[derive(Clone, Debug, Default)]
pub struct SectionGenerations {
    live: HashMap<u64, u64>,
}

/// What an open walk learns of this node's own sections: every section
/// declaration it wrote, and every one of its generations a tombstone ended.
/// Folded into the live generations once the walk is done, so the order the
/// walk met them in does not matter.
#[derive(Debug, Default)]
pub struct OwnSections {
    declared: HashMap<u64, u64>,
    ended: HashSet<(u64, u64)>,
}

impl OwnSections {
    /// Note one walked record, of node `node`'s substrate.
    pub fn observe(&mut self, node: u64, header: &RecordHeader, payload: &[u8]) {
        match header.record_type {
            RecordType::StreamDecl if header.origin == node && declares_section(payload) => {
                let held = self.declared.entry(header.stream_id).or_insert(0);
                if header.oseq > *held {
                    *held = header.oseq;
                }
            }
            RecordType::SectionTombstone => {
                if let Some(g) = SectionTombstonePayload::decode(payload)
                    .ok()
                    .and_then(|p| p.generation)
                {
                    if g.writer == node {
                        self.ended.insert((header.stream_id, g.oseq));
                    }
                }
            }
            _ => {}
        }
    }

    /// The generations this node is still writing: each section's latest
    /// declaration, unless a tombstone ended it.
    pub fn live(self) -> SectionGenerations {
        let live = self
            .declared
            .into_iter()
            .filter(|(sid, oseq)| !self.ended.contains(&(*sid, *oseq)))
            .collect();
        SectionGenerations { live }
    }
}

impl SectionGenerations {
    /// A declaration written as `oseq` started a generation of `stream_id`. A
    /// later declaration supersedes an earlier one.
    pub fn start(&mut self, stream_id: u64, oseq: u64) {
        let held = self.live.entry(stream_id).or_insert(oseq);
        if oseq > *held {
            *held = oseq;
        }
    }

    /// The generation `oseq` of `stream_id` has ended. A newer generation
    /// already started stays.
    pub fn end(&mut self, stream_id: u64, oseq: u64) {
        if self.live.get(&stream_id) == Some(&oseq) {
            self.live.remove(&stream_id);
        }
    }

    /// The generation of `stream_id` being written, if any.
    pub fn generation(&self, stream_id: u64) -> Option<u64> {
        self.live.get(&stream_id).copied()
    }

    /// Every section being written, with its generation.
    pub fn live(&self) -> impl Iterator<Item = (u64, u64)> + '_ {
        self.live.iter().map(|(s, o)| (*s, *o))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::persistence::content_hash::ContentHash;
    use crate::persistence::record::SectionGeneration;
    use crate::persistence::streams::{ContentAddress, SectionDecl, StreamDecl, TurnDecl};

    fn header(rt: RecordType, stream_id: u64, origin: u64, oseq: u64) -> RecordHeader {
        RecordHeader {
            record_type: rt,
            stream_id,
            origin,
            oseq,
            ..RecordHeader::NONE
        }
    }

    fn section() -> Vec<u8> {
        StreamDecl::PromptSection(SectionDecl {
            address: ContentAddress {
                prefix_hash: ContentHash { lo: 1, hi: 2 },
                section_hash: ContentHash { lo: 3, hi: 4 },
            },
            debug_name: "sys".into(),
        })
        .encode()
    }

    fn tombstone(writer: u64, oseq: u64) -> Vec<u8> {
        SectionTombstonePayload {
            generation: Some(SectionGeneration { writer, oseq }),
            reason: None,
        }
        .encode()
    }

    /// The walk folds to each section's latest own declaration not ended, in
    /// any order: another node's declarations, turn declarations and another
    /// node's ended generations do not count.
    #[test]
    fn an_open_walk_finds_the_generations_still_being_written() {
        let turn = StreamDecl::Turn(TurnDecl {
            timeline_id: 1,
            turn_index: 0,
            turn_id_day: 0,
            turn_id_seq: 1,
            role: 1,
            block_start: 0,
            block_end: 0,
            layer_id: 1,
            group_id: 1,
            anchored_prefix: Vec::new(),
            view: Vec::new(),
            segments: Vec::new(),
            tags: Vec::new(),
        })
        .encode();
        let mut walk = OwnSections::default();
        walk.observe(
            9,
            &header(RecordType::SectionTombstone, 40, 9, 12),
            &tombstone(9, 11),
        );
        walk.observe(9, &header(RecordType::StreamDecl, 40, 9, 20), &section());
        walk.observe(9, &header(RecordType::StreamDecl, 40, 9, 11), &section());
        walk.observe(9, &header(RecordType::StreamDecl, 41, 9, 30), &section());
        walk.observe(
            9,
            &header(RecordType::SectionTombstone, 41, 9, 31),
            &tombstone(9, 30),
        );
        walk.observe(9, &header(RecordType::StreamDecl, 42, 4, 5), &section());
        walk.observe(9, &header(RecordType::StreamDecl, 43, 9, 6), &turn);
        walk.observe(9, &header(RecordType::StreamDecl, 44, 9, 7), &section());
        walk.observe(
            9,
            &header(RecordType::SectionTombstone, 44, 4, 8),
            &tombstone(4, 7),
        );
        let live = walk.live();
        assert_eq!(live.generation(40), Some(20));
        assert_eq!(live.generation(41), None, "its one generation ended");
        assert_eq!(live.generation(42), None, "another node's section");
        assert_eq!(live.generation(43), None, "a turn");
        assert_eq!(
            live.generation(44),
            Some(7),
            "another node's generation ended"
        );
    }

    #[test]
    fn the_latest_declaration_is_the_live_generation() {
        let mut g = SectionGenerations::default();
        g.start(7, 10);
        g.start(7, 4);
        assert_eq!(g.generation(7), Some(10));
        g.end(7, 4);
        assert_eq!(g.generation(7), Some(10), "an older generation's end");
        g.end(7, 10);
        assert_eq!(g.generation(7), None);
        assert_eq!(g.generation(8), None);
    }
}
