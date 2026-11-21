//! Tests that recurrent snapshots read correctly, including while compaction holds
//! the persistence owner.
//!
//! A retired tail reads as no snapshot. The index declining a snapshot means it is
//! never reported durable, since a slot released on that report would lose state
//! the record cannot give back. The admission read completes while a compaction
//! holds the owner, because the scheduler would otherwise wait on it for every
//! decode. An index that names a missing segment is an error, not a missing
//! snapshot.

use super::*;

#[cfg(test)]
pub(super) mod recurrent_snapshot_read_tests {
    use std::sync::mpsc;
    use std::time::Duration;

    use super::Conversation;
    use crate::persistence::content_hash::snapshot_stream_id;
    use crate::persistence::record::SnapshotPayload;
    use crate::persistence::segmented_log::segment_path;
    use crate::projection::TimelineId;

    /// A recognisable snapshot for `timeline`.
    fn payload(timeline: TimelineId, turn_index: u32) -> SnapshotPayload {
        SnapshotPayload {
            timeline_id: timeline.raw(),
            turn_index,
            schedule_hash: 0xC0FF_EE00,
            layers: Vec::new(),
            // Some state: a snapshot with neither layers nor aux is a retired
            // tail, which reads as none kept.
            aux: vec![turn_index as u8; 8],
        }
    }

    /// **A retired tail reads as no snapshot.** A timeline marked as never
    /// resumed supersedes its state with an empty record; the read must answer
    /// `None` — the ordinary "nothing kept" — rather than an empty state.
    #[test]
    fn a_retired_tail_reads_as_no_snapshot() {
        let conv = Conversation::ephemeral();
        let timeline = TimelineId::for_test(41);
        store(&conv, timeline, 0, true);
        assert!(conv.read_recurrent_snapshot(timeline).unwrap().is_some());

        let stream = snapshot_stream_id(timeline.raw());
        let loc = conv.persistence.call(|p| {
            let loc = p
                .write_snapshot(
                    stream,
                    &SnapshotPayload::retired(timeline.raw()).parts().slices(),
                )
                .expect("append");
            p.commit().expect("commit");
            loc
        });
        conv.update(|s| s.apply_snapshot_loc(stream, loc.value, loc.id));
        assert!(conv.read_recurrent_snapshot(timeline).unwrap().is_none());
    }

    /// Store a snapshot exactly as the writer does — append on the persistence
    /// owner, then publish the location in a substrate edit — and optionally
    /// seal the segment it landed in.
    fn store(conv: &Conversation, timeline: TimelineId, turn_index: u32, seal: bool) {
        let stream = snapshot_stream_id(timeline.raw());
        let loc = conv.persistence.call(|p| {
            let loc = p
                .write_snapshot(stream, &payload(timeline, turn_index).parts().slices())
                .expect("append");
            p.commit().expect("commit");
            if seal {
                p.seal_active().expect("seal");
            }
            loc
        });
        conv.update(|s| s.apply_snapshot_loc(stream, loc.value, loc.id));
    }

    /// Occupy the persistence owner inside a call, as a compaction's relocation
    /// phase does, until the returned sender is dropped or sent to.
    fn hold_owner(conv: &Conversation) -> (mpsc::Sender<()>, std::thread::JoinHandle<()>) {
        let (release, released) = mpsc::channel::<()>();
        let (entered, is_entered) = mpsc::channel::<()>();
        let owner = conv.persistence.clone();
        let holder = std::thread::spawn(move || {
            owner.call(move |_| {
                entered.send(()).expect("signal entry");
                let _ = released.recv();
            });
        });
        is_entered
            .recv_timeout(Duration::from_secs(10))
            .expect("the owner took the holding call");
        (release, holder)
    }

    /// **The admission read completes while a compaction occupies the
    /// persistence owner.** That hold is what stalled the scheduler: a mean of
    /// 16.5 s per maintenance op, measured, with every decode waiting behind the
    /// snapshot read at admission. The owner is held here for the whole read,
    /// as the relocation phase holds it, and the read runs against a deadline.
    #[test]
    fn a_sealed_snapshot_reads_while_persistence_is_held() {
        let conv = Conversation::ephemeral();
        let timeline = TimelineId::for_test(41);
        store(&conv, timeline, 3, true);

        let (release, holder) = hold_owner(&conv);
        let reader = conv.clone();
        let (tx, rx) = mpsc::channel();
        let worker = std::thread::spawn(move || {
            let _ = tx.send(reader.read_recurrent_snapshot(timeline));
        });
        let got = rx
            .recv_timeout(Duration::from_secs(10))
            .expect("the snapshot read finished while persistence was held")
            .expect("readable")
            .expect("present");
        worker.join().expect("worker");
        drop(release);
        holder.join().expect("holder");
        assert_eq!(got.turn_index, 3);
        assert_eq!(got.schedule_hash, 0xC0FF_EE00);
        assert_eq!(got.timeline_id, timeline.raw());
    }

    /// **A committed snapshot still in the active segment reads while the
    /// persistence owner is occupied, too.** The snapshot a seal just wrote is
    /// the one the next admission reads back, and it is nearly always still in
    /// the active segment: once committed it is as final as a sealed record, so
    /// the read does not go through the owner.
    #[test]
    fn a_committed_active_snapshot_reads_while_persistence_is_held() {
        let conv = Conversation::ephemeral();
        let timeline = TimelineId::for_test(42);
        store(&conv, timeline, 5, false);

        let (release, holder) = hold_owner(&conv);
        let reader = conv.clone();
        let (tx, rx) = mpsc::channel();
        let worker = std::thread::spawn(move || {
            let _ = tx.send(reader.read_recurrent_snapshot(timeline));
        });
        let got = rx
            .recv_timeout(Duration::from_secs(10))
            .expect("the active read finished while persistence was held")
            .expect("readable")
            .expect("present");
        worker.join().expect("worker");
        drop(release);
        holder.join().expect("holder");
        assert_eq!(got.turn_index, 5);
    }

    /// **Only a snapshot the index took is reported.** One written for a
    /// timeline already tombstoned is declined by the index — its stream is
    /// retired, so a read of the timeline's snapshot finds nothing — and a slot
    /// told it was durable would drop state its record cannot give back. The
    /// live timeline's snapshot, written beside it, is reported.
    #[test]
    fn a_snapshot_the_index_declined_is_not_reported() {
        let conv = Conversation::ephemeral();
        let retired = TimelineId::for_test(45);
        let live = TimelineId::for_test(46);
        conv.tombstone_timeline(retired).expect("tombstone");
        conv.enqueue_recurrent_snapshot(retired, payload(retired, 2));
        conv.enqueue_recurrent_snapshot(live, payload(live, 3));
        let deadline = std::time::Instant::now() + Duration::from_secs(10);
        let mut reported = Vec::new();
        while !reported.contains(&(live, 3)) {
            reported.extend(conv.drain_durable_snapshots());
            assert!(
                std::time::Instant::now() < deadline,
                "the writer never reported the live snapshot"
            );
            std::thread::sleep(Duration::from_millis(5));
        }
        assert_eq!(
            reported,
            vec![(live, 3)],
            "the retired one is never reported"
        );
        assert!(conv.read_recurrent_snapshot(retired).unwrap().is_none());
    }

    /// **The writer reports a snapshot once its record is durable, once.** A
    /// slot holding the state of a sealed turn may release it when this says
    /// so, because the read above then serves the record without the owner.
    /// The report is the timeline and turn the snapshot was taken at; a second
    /// drain has nothing new.
    #[test]
    fn a_written_snapshot_is_reported_once_it_is_durable() {
        let conv = Conversation::ephemeral();
        let timeline = TimelineId::for_test(44);
        conv.enqueue_recurrent_snapshot(timeline, payload(timeline, 6));
        let deadline = std::time::Instant::now() + Duration::from_secs(10);
        let reported = loop {
            let got = conv.drain_durable_snapshots();
            if !got.is_empty() {
                break got;
            }
            assert!(
                std::time::Instant::now() < deadline,
                "the writer never reported the snapshot durable"
            );
            std::thread::sleep(Duration::from_millis(5));
        };
        assert_eq!(reported, vec![(timeline, 6)]);
        assert!(conv.drain_durable_snapshots().is_empty(), "reported once");

        let (release, holder) = hold_owner(&conv);
        // Asked of the timeline, the turn is durable whatever the drain took,
        // and so is every earlier one; a later one is not.
        assert!(conv.snapshot_durable(timeline, 6));
        assert!(conv.snapshot_durable(timeline, 5));
        assert!(!conv.snapshot_durable(timeline, 7));
        assert!(!conv.snapshot_durable(TimelineId::for_test(99), 0));
        let reader = conv.clone();
        let (tx, rx) = mpsc::channel();
        let worker = std::thread::spawn(move || {
            let _ = tx.send(reader.read_recurrent_snapshot(timeline));
        });
        let got = rx
            .recv_timeout(Duration::from_secs(10))
            .expect("a reported snapshot reads without the owner")
            .expect("readable")
            .expect("present");
        worker.join().expect("worker");
        drop(release);
        holder.join().expect("holder");
        assert_eq!(got.turn_index, 6);
    }

    /// **An index naming a segment that is gone is still an error.** The
    /// direct read re-reads the location on a missing file, and when the
    /// index still names that file it hands over to the owner's read — which
    /// reports it rather than answering `None`. A missing snapshot that reads as
    /// "no snapshot" is the silent amnesia this path exists to prevent.
    #[test]
    fn a_snapshot_whose_segment_vanished_is_an_error_not_a_none() {
        let conv = Conversation::ephemeral();
        let timeline = TimelineId::for_test(43);
        store(&conv, timeline, 7, true);
        let loc = conv.recurrent_snapshot_loc(timeline).expect("indexed");
        let file = segment_path(conv.persistence.dir(), loc.segment);
        std::fs::remove_file(&file).expect("remove the segment");
        let err = conv
            .read_recurrent_snapshot(timeline)
            .expect_err("a vanished snapshot is not an absent one");
        assert!(
            err.to_string().contains("indexed but unreadable"),
            "the owner's read's own report: {err}"
        );
    }

    /// **A read retries only a location the index has moved away from.** A
    /// snapshot written again lands at a new location, and the one read before
    /// it is stale; the current location is not. This is what both read paths
    /// consult after a failure, so an unreadable record where the index still
    /// points is reported rather than chased.
    #[test]
    fn only_a_relocated_snapshot_counts_as_moved() {
        let conv = Conversation::ephemeral();
        let timeline = TimelineId::for_test(45);
        store(&conv, timeline, 1, true);
        let first = conv.recurrent_snapshot_loc(timeline).expect("indexed");
        store(&conv, timeline, 2, true);
        let second = conv.recurrent_snapshot_loc(timeline).expect("indexed");
        assert_ne!(first, second, "the test's own premise");
        assert_eq!(conv.moved_snapshot_loc(timeline, first), Some(second));
        assert_eq!(conv.moved_snapshot_loc(timeline, second), None);
        // And the read follows the index to the newer record.
        let got = conv
            .read_recurrent_snapshot(timeline)
            .expect("readable")
            .expect("present");
        assert_eq!(got.turn_index, 2);
    }

    /// No snapshot is an ordinary `None`, and asks nothing of either path.
    #[test]
    fn no_snapshot_is_none() {
        let conv = Conversation::ephemeral();
        assert!(conv
            .read_recurrent_snapshot(TimelineId::for_test(44))
            .expect("readable")
            .is_none());
    }
}
