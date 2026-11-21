//! Tests turn-level behaviour: a conversation's node, its lineage history, the
//! metadata it sees, and the index pages an adopted turn carries.
//!
//! A forked conversation opens with its ancestors' turns and sees no sibling's, and
//! an unforked one sees only its own. An adopted scope turn keeps its index page,
//! because a page left under the fork's stream would be unreachable once the fork
//! is tombstoned. A held page is read from the log once, and the in-memory lookup
//! finds only pages a caller holds.

use super::*;

#[cfg(test)]
pub(super) mod tests {
    use super::Conversation;

    use std::sync::Arc;

    use crate::persistence::content_hash::turn_stream_id;

    use crate::projection::{GroupId, LayerId, TimelineId};

    use crate::substrate::{Substrate, TurnPartWrite};

    use crate::token_buffer::TokenBuffer;

    use crate::turn::Role;

    use candle_nn::kv_cache::{ArenaLocation, SealedChunk, SealedSequence};

    pub(in super::super) fn tags(v: &[&str]) -> Vec<String> {
        v.iter().map(|s| s.to_string()).collect()
    }

    /// A conversation names its substrate's node and the last `oseq` it
    /// originated — what a cluster member says in `HELLO` and `UNCLAIM`.
    #[test]
    fn a_conversation_reports_its_node_and_last_oseq() {
        let conv = Conversation::ephemeral();
        assert_ne!(conv.node(), 0);
        let before = conv.last_oseq();
        let id = conv.minter.mint(0);
        assert_eq!((id.origin, id.oseq), (conv.node(), before + 1));
        assert_eq!(conv.last_oseq(), id.oseq);
    }

    /// **A forked conversation's "own history" spans its lineage.**
    ///
    /// This is the projection half of `Substrate::inherited_chain`: the target
    /// group is still masked to one conversation (a slot never sees a sibling
    /// chat), but that conversation now opens with the turns of whatever it was
    /// forked from, oldest first. It is what lets a dialogue start already
    /// holding the priming chain's documents without a single turn being
    /// copied onto it — and without the ancestor being pinned resident, since
    /// each key keeps its own timeline and is elevated from whatever tier it is
    /// on when selected.
    #[test]
    fn a_forked_conversation_opens_with_its_ancestors_turns() {
        use crate::projection::ids::{TurnIndex, TurnKey};
        use crate::projection::project::ProjectionTarget;
        use crate::substrate::ContentResolver;

        let conv = Conversation::ephemeral();
        let layer = LayerId::from_raw(1).expect("layer id");
        let group = GroupId::from_raw(1).expect("group id");
        let grandparent = TimelineId::from_raw(21).expect("timeline id");
        let parent = TimelineId::from_raw(22).expect("timeline id");
        let child = TimelineId::from_raw(23).expect("timeline id");
        for tl in [grandparent, parent, child] {
            conv.register_timeline(tl, layer, group);
        }
        conv.update(|w| {
            w.append_with_blocks(grandparent, 8, 0, 1);
            w.append_with_blocks(parent, 8, 1, 2);
            w.append_with_blocks(child, 8, 2, 3);
        });
        for (c, p) in [(child, parent), (parent, grandparent)] {
            conv.set_conversation_metadata(c, Substrate::FORKED_FROM_KEY, &p.raw().to_string())
                .expect("record lineage");
        }

        let read = conv.read_for(ProjectionTarget {
            layer,
            group,
            timeline: child,
        });
        assert_eq!(
            ContentResolver::group_turns(&read, group),
            vec![
                TurnKey::new(grandparent, TurnIndex(0)),
                TurnKey::new(parent, TurnIndex(0)),
                TurnKey::new(child, TurnIndex(0)),
            ],
            "oldest ancestor first, the target's own turns last",
        );
    }

    /// **A dialogue's projection sees the engine's trained window.** The
    /// targeted read every dialogue projects through answered `None` while
    /// the substrate held the window, so no slot was ever filled
    /// (`ContentResolver::trained_window`). Every reader reports what the
    /// substrate holds, before and after it is set.
    #[test]
    fn every_read_reports_the_substrates_trained_window() {
        use crate::projection::project::ProjectionTarget;
        use crate::substrate::ContentResolver;

        let conv = Conversation::ephemeral();
        let target = ProjectionTarget {
            layer: LayerId::from_raw(1).expect("layer id"),
            group: GroupId::from_raw(1).expect("group id"),
            timeline: TimelineId::from_raw(41).expect("timeline id"),
        };
        let windows = |conv: &Conversation| {
            let read = conv.read();
            (
                ContentResolver::trained_window(&*read),
                ContentResolver::trained_window(&read),
                ContentResolver::trained_window(&conv.read_for(target)),
            )
        };
        assert_eq!(windows(&conv), (None, None, None));
        conv.update(|s| s.set_trained_window(Some(262_144)));
        let set = Some(262_144);
        assert_eq!(windows(&conv), (set, set, set));
    }

    /// A conversation that never forked is unaffected — the masking that keeps
    /// one chat out of another's projection is exactly as it was.
    #[test]
    fn an_unforked_conversation_still_sees_only_its_own_turns() {
        use crate::projection::ids::{TurnIndex, TurnKey};
        use crate::projection::project::ProjectionTarget;
        use crate::substrate::ContentResolver;

        let conv = Conversation::ephemeral();
        let layer = LayerId::from_raw(1).expect("layer id");
        let group = GroupId::from_raw(1).expect("group id");
        let mine = TimelineId::from_raw(31).expect("timeline id");
        let sibling = TimelineId::from_raw(32).expect("timeline id");
        conv.register_timeline(mine, layer, group);
        conv.register_timeline(sibling, layer, group);
        conv.update(|w| {
            w.append_with_blocks(mine, 8, 0, 1);
            w.append_with_blocks(sibling, 8, 1, 2);
        });

        let read = conv.read_for(ProjectionTarget {
            layer,
            group,
            timeline: mine,
        });
        assert_eq!(
            ContentResolver::group_turns(&read, group),
            vec![TurnKey::new(mine, TurnIndex(0))],
            "a sibling conversation stays masked out",
        );
    }

    /// **A spliced turn keeps its index page.**
    ///
    /// `adopt_turn` moves a scope fork's sealed turn onto the file timeline by
    /// reference and re-keys it to a new stream. The turn's QSA index page is
    /// keyed by stream as well, and the fork is tombstoned straight after the
    /// splice — so a page left under the fork's stream is gone, and every
    /// projection that later borrows the turn past the identity threshold holds
    /// K/V its attention layers cannot select. That was every code_reading turn
    /// of an ingest: a greeting carried ~16k tokens no index covered, and the
    /// model answered an earlier question instead of the greeting.
    #[test]
    fn an_adopted_turn_carries_its_index_page_onto_the_new_stream() {
        let conv = Conversation::ephemeral();
        let layer = LayerId::from_raw(1).expect("layer id");
        let group = GroupId::from_raw(1).expect("group id");
        let fork = TimelineId::from_raw(7).expect("fork timeline id");
        let file = TimelineId::from_raw(8).expect("file timeline id");
        conv.register_timeline(fork, layer, group);
        conv.register_timeline(file, layer, group);
        conv.mark_timeline_splice_source(fork);

        let sealed = Arc::new(vec![SealedSequence {
            chunks: vec![SealedChunk::for_test(1000, 4)],
            token_count: 4,
            chunk_size: 32,
            location: ArenaLocation::Gpu,
        }]);
        let from = conv
            .record_turn(
                fork,
                Role::Assistant,
                TurnPartWrite {
                    token_ids: TokenBuffer::from(vec![11, 12, 13, 14]),
                    token_count: 4,
                    block_start: 0,
                    block_end: 1,
                    sealed_gpu: Some(sealed),
                    ..Default::default()
                },
                |seqs| Ok(seqs.to_vec()),
            )
            .expect("record the fork's turn");
        let page = vec![0xA5u8, 0x5A, 0x01, 0x02, 0x03];
        conv.enqueue_index_page(turn_stream_id(fork.raw(), from.0), page.clone());

        let adopted = conv
            .adopt_turn(
                fork,
                from,
                file,
                Role::Assistant,
                tags(&["code", "src/lib.rs"]),
            )
            .expect("adopt the fork's turn");

        assert_eq!(
            conv.turn_index_page(file, adopted)
                .expect("the adopted turn's page reads")
                .map(|p| p.bytes().to_vec()),
            Some(page.clone()),
            "the adopted turn must carry the fork's index page under its own stream"
        );
    }

    /// **A turn whose page is in the log projects as one whose page is held.**
    /// Once its write lands the substrate holds only where the page is
    /// ([`IndexPage::Stored`]); the page a projection reads back is the bytes
    /// the seal wrote, it decodes to the same pages, and adopting the turn
    /// carries those bytes onto the new stream — the path the index pages have
    /// failed to travel before, now with them read from disk.
    #[test]
    fn a_turn_whose_page_is_on_disk_projects_as_one_held() {
        let conv = Conversation::ephemeral();
        let layer = LayerId::from_raw(1).expect("layer id");
        let group = GroupId::from_raw(1).expect("group id");
        let fork = TimelineId::from_raw(17).expect("fork timeline id");
        let file = TimelineId::from_raw(18).expect("file timeline id");
        conv.register_timeline(fork, layer, group);
        conv.register_timeline(file, layer, group);
        conv.mark_timeline_splice_source(fork);
        let sealed = Arc::new(vec![SealedSequence {
            chunks: vec![SealedChunk::for_test(1000, 4)],
            token_count: 4,
            chunk_size: 32,
            location: ArenaLocation::Gpu,
        }]);
        let from = conv
            .record_turn(
                fork,
                Role::Assistant,
                TurnPartWrite {
                    token_ids: TokenBuffer::from(vec![21, 22, 23, 24]),
                    token_count: 4,
                    block_start: 0,
                    block_end: 1,
                    sealed_gpu: Some(sealed),
                    ..Default::default()
                },
                |seqs| Ok(seqs.to_vec()),
            )
            .expect("record the fork's turn");
        // A real page: two framed pages, so the decode has something to agree on.
        let page = crate::index_pages::encode(&[(2, vec![7u8; 96]), (2, vec![9u8; 96])]);
        conv.enqueue_index_page(turn_stream_id(fork.raw(), from.0), page.clone());
        conv.flush_writer();

        let at = conv
            .read()
            .index_page_at(fork, from)
            .expect("the turn has a page");
        assert!(
            at.stored().is_some(),
            "written, the page is held in the log, not RAM"
        );
        let read = conv
            .turn_index_page(fork, from)
            .expect("the page reads")
            .expect("the turn has a page");
        assert_eq!(
            read.bytes(),
            page.as_slice(),
            "the seal's bytes, read from disk"
        );
        assert_eq!(
            crate::sealed_turn::SealedPages::of(Some(read.bytes())),
            crate::sealed_turn::SealedPages::of(Some(&page)),
            "the same pages as the bytes held"
        );

        let adopted = conv
            .adopt_turn(fork, from, file, Role::Assistant, tags(&["code"]))
            .expect("adopt the fork's turn");
        conv.flush_writer();
        let carried = conv
            .turn_index_page(file, adopted)
            .expect("the adopted page reads")
            .expect("the adopted turn has a page");
        assert_eq!(
            carried.bytes(),
            page.as_slice(),
            "adoption carries the bytes read from disk"
        );
    }

    /// **A page a slot holds is not read off the log again.**
    ///
    /// A reprojection re-injects every turn behind the first piece that
    /// changed, and each one read its page from the log — 3,590 reads in a
    /// 56 s turn, 7.1 s of its rebuilds. While a caller holds the bytes a read
    /// produced, the same record hands back that very allocation; once the
    /// last holder lets go, the next read goes to the log; and a new page for
    /// the stream is read fresh, though the old one is still held.
    #[test]
    fn a_held_page_is_read_from_the_log_once() {
        let conv = Conversation::ephemeral();
        let layer = LayerId::from_raw(1).expect("layer id");
        let group = GroupId::from_raw(1).expect("group id");
        let tl = TimelineId::from_raw(27).expect("timeline id");
        conv.register_timeline(tl, layer, group);
        let sealed = Arc::new(vec![SealedSequence {
            chunks: vec![SealedChunk::for_test(1000, 4)],
            token_count: 4,
            chunk_size: 32,
            location: ArenaLocation::Gpu,
        }]);
        let idx = conv
            .record_turn(
                tl,
                Role::Assistant,
                TurnPartWrite {
                    token_ids: TokenBuffer::from(vec![31, 32, 33, 34]),
                    token_count: 4,
                    block_start: 0,
                    block_end: 1,
                    sealed_gpu: Some(sealed),
                    ..Default::default()
                },
                |seqs| Ok(seqs.to_vec()),
            )
            .expect("record the turn");
        let stream = turn_stream_id(tl.raw(), idx.0);
        let first = crate::index_pages::encode(&[(2, vec![7u8; 96])]);
        conv.enqueue_index_page(stream, first.clone());
        conv.flush_writer();
        assert!(
            conv.read()
                .index_page_at(tl, idx)
                .and_then(|p| p.stored())
                .is_some(),
            "the page is in the log, not RAM"
        );

        let held = conv
            .turn_index_page(tl, idx)
            .expect("the page reads")
            .expect("the turn has a page");
        let again = conv
            .turn_index_page(tl, idx)
            .expect("the page reads")
            .expect("the turn has a page");
        assert!(
            Arc::ptr_eq(&held, &again),
            "a held page is handed back, not read again"
        );
        assert_eq!(again.bytes(), first.as_slice());

        drop(again);
        drop(held);
        let fresh = conv
            .turn_index_page(tl, idx)
            .expect("the page reads")
            .expect("the turn has a page");
        assert_eq!(
            fresh.bytes(),
            first.as_slice(),
            "with no holder left, the page is read from the log again"
        );

        let second = crate::index_pages::encode(&[(2, vec![9u8; 96])]);
        conv.enqueue_index_page(stream, second.clone());
        conv.flush_writer();
        let newer = conv
            .turn_index_page(tl, idx)
            .expect("the page reads")
            .expect("the turn has a page");
        assert_eq!(
            newer.bytes(),
            second.as_slice(),
            "a new page for the stream is read, though the old one is still held"
        );
        assert!(!Arc::ptr_eq(&newer, &fresh));
    }

    /// **The in-memory lookup never reads the log.** A written page that no
    /// caller holds is not found; once a read holds it, the lookup hands back
    /// that very allocation; and when the holder lets go it is not found again.
    #[test]
    fn the_in_memory_lookup_finds_only_pages_held() {
        let conv = Conversation::ephemeral();
        let layer = LayerId::from_raw(1).expect("layer id");
        let group = GroupId::from_raw(1).expect("group id");
        let tl = TimelineId::from_raw(29).expect("timeline id");
        conv.register_timeline(tl, layer, group);
        let sealed = Arc::new(vec![SealedSequence {
            chunks: vec![SealedChunk::for_test(1000, 4)],
            token_count: 4,
            chunk_size: 32,
            location: ArenaLocation::Gpu,
        }]);
        let idx = conv
            .record_turn(
                tl,
                Role::Assistant,
                TurnPartWrite {
                    token_ids: TokenBuffer::from(vec![41, 42, 43, 44]),
                    token_count: 4,
                    block_start: 0,
                    block_end: 1,
                    sealed_gpu: Some(sealed),
                    ..Default::default()
                },
                |seqs| Ok(seqs.to_vec()),
            )
            .expect("record the turn");
        let page = crate::index_pages::encode(&[(2, vec![5u8; 96])]);
        conv.enqueue_index_page(turn_stream_id(tl.raw(), idx.0), page.clone());
        conv.flush_writer();

        assert!(
            conv.turn_index_page_in_memory(tl, idx).is_none(),
            "written and held by no one, the page is only in the log"
        );
        let held = conv
            .turn_index_page(tl, idx)
            .expect("the page reads")
            .expect("the turn has a page");
        let found = conv
            .turn_index_page_in_memory(tl, idx)
            .expect("a held page is in memory");
        assert!(Arc::ptr_eq(&held, &found), "the held allocation itself");
        assert_eq!(found.bytes(), page.as_slice());

        drop(found);
        drop(held);
        assert!(
            conv.turn_index_page_in_memory(tl, idx).is_none(),
            "with no holder left, the page is the log's alone"
        );
    }
}
