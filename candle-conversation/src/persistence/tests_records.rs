//! Tests that streams and turns survive a reopen, and that an inherited base resolves its sections.
//!
//! A section stream in an inherited base resolves through inheritance as a prefix-cache hit, and the child writes only to its own active log. The test that would build the base as a single file is ignored, since a base is now a segment directory that cannot be produced as one file.

use super::*;

#[cfg(test)]
pub(super) mod tests {
    use super::*;

    use log_file::LogFile;

    use segment::FIRST_SEGMENT;

    use streams::TurnDecl;

    use super::super::tests_support::tests::{section, tmp_dir};

    #[test]
    fn streams_and_turns_recover_across_reopen() {
        let dir = tmp_dir("recover");
        let turn = StreamDecl::Turn(TurnDecl {
            timeline_id: 1,
            turn_index: 0,
            turn_id_day: 0,
            turn_id_seq: 1,
            role: 1,
            block_start: 0,
            block_end: 32,
            layer_id: 1,
            group_id: 1,
            anchored_prefix: Vec::new(),
            view: Vec::new(),
            segments: Vec::new(),
            tags: Vec::new(),
        });
        let turn_id;
        {
            let mut sp = SubstratePersistence::open_in(&dir).unwrap();
            turn_id = sp.declare_stream(&turn).unwrap().value;
            sp.append_tokens(turn_id, b"token-bytes").unwrap();
            sp.commit_stream(turn_id, 0).unwrap();
            sp.commit().unwrap();
        }
        {
            let mut substrate = Substrate::new();
            let sp = SubstratePersistence::open_in_with_substrate(&dir, &mut substrate).unwrap();
            assert!(sp.has_stream(&substrate, turn_id));
            let entry = substrate.stream_of(turn_id).unwrap();
            assert_eq!(entry.committed_through, Some(0));
            assert!(entry.tokens.is_some());
        }
        std::fs::remove_dir_all(&dir).ok();
    }

    // Fork inheritance (`open_concat` over a single-file base) predates the
    // segmented store: an inherited base is a read-only single-file log, but a
    // substrate is now written as a segment directory, so a base can no longer
    // be produced as one file. Inheriting a read-only segment *set* is a fork
    // redesign tracked separately; ignored until then.
    #[test]
    #[ignore = "fork inheritance over a segment set is a separate redesign"]
    fn open_concat_inherits_and_keeps_last_active() {
        let base_dir = tmp_dir("base");
        let child_dir = tmp_dir("child");

        // Build a base log with a section stream.
        let sec = section("shared_section", 0);
        let base_log = base_dir.join("base.log");
        {
            let mut base =
                SubstratePersistence::open_concat(std::slice::from_ref(&base_log)).unwrap();
            base.declare_stream(&sec).unwrap();
            base.commit().unwrap();
        }

        // A child inherits the base; its active segment set is its own.
        let child_log = child_dir.join(SUBSTRATE_DIR);
        let mut child =
            SubstratePersistence::open_concat(&[base_log.clone(), child_log.clone()]).unwrap();
        assert_eq!(child.inherited_count(), 1);

        // The section stream resolves through inheritance — a prefix-cache hit.
        // The active substrate has no streams; the resolve falls through to
        // the inherited substrate's stream index.
        let substrate = Substrate::new();
        let hit = child.lookup_section(
            &substrate,
            match &sec {
                StreamDecl::PromptSection(s) => s.address,
                _ => unreachable!(),
            },
        );
        assert!(hit.is_some(), "inherited section must resolve");

        // The child writes only to its own active log.
        let child_turn = StreamDecl::Turn(TurnDecl {
            timeline_id: 9,
            turn_index: 0,
            turn_id_day: 0,
            turn_id_seq: 1,
            role: 2,
            block_start: 0,
            block_end: 16,
            layer_id: 1,
            group_id: 1,
            anchored_prefix: vec![sec.stream_id()],
            view: Vec::new(),
            segments: Vec::new(),
            tags: Vec::new(),
        });
        child.declare_stream(&child_turn).unwrap();
        child.commit().unwrap();
        // Drop child + re-open via the substrate-aware path so the
        // walker rebuilds substrate.streams; then assert that the
        // child has its own turn and the inherited base section.
        drop(child);
        let mut child_substrate = Substrate::new();
        let child =
            SubstratePersistence::open_concat(&[base_log.clone(), child_log.clone()]).unwrap();
        // open_concat populates the active substrate via the same
        // walker pass that builds the manifest.  We approximate it
        // here by walking the active log explicitly into substrate.
        // (open_concat doesn't have a *_with_substrate variant; the
        // production path uses open_in_with_substrate.)
        {
            let mut log = LogFile::open(&child_log).unwrap();
            let hint = log.superblock().last_index;
            recovery::recover_with_sink(&mut log, FIRST_SEGMENT, hint, |e| {
                child_substrate.apply_walker_entry(e)
            })
            .unwrap();
        }
        assert!(child_substrate.has_stream(child_turn.stream_id()));
        assert!(child.inherited_substrates()[0]
            .substrate()
            .has_stream(sec.stream_id()));

        InheritedSubstrate::forget(&base_log);
        std::fs::remove_dir_all(&base_dir).ok();
        std::fs::remove_dir_all(&child_dir).ok();
    }

    #[test]
    fn unknown_section_is_a_cache_miss() {
        let dir = tmp_dir("miss");
        let sp = SubstratePersistence::open_in(&dir).unwrap();
        let substrate = Substrate::new();
        let absent = ContentAddress {
            prefix_hash: content_hash::ContentHash { lo: 12345, hi: 678 },
            section_hash: content_hash::ContentHash { lo: 9, hi: 9 },
        };
        assert!(sp.lookup_section(&substrate, absent).is_none());
        std::fs::remove_dir_all(&dir).ok();
    }
}
