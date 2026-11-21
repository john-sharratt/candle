//! Tests the status snapshot's detail and the wording of a waiting prefill.
//!
//! A detail sent while no reader is subscribed is still the next snapshot's.
//! The load reports each phase once as it begins, and a dropped send would
//! leave the loading screen with no name for it. A wait reads as what is ahead
//! and how long, with the part about what is ahead shown only when something is.

use super::*;

#[cfg(test)]
pub(super) mod queued_status_tests {
    use super::queued_status;

    /// What a waiting prefill shows: whole seconds, tokens in K past a
    /// thousand, the ahead part only when something is ahead.
    #[test]
    fn a_wait_reads_as_what_is_ahead_and_how_long() {
        assert_eq!(
            queued_status(18_400, 2, 4_900),
            "Waiting for the GPU — 2 turns (18K tokens) ahead, 4 s"
        );
        assert_eq!(
            queued_status(512, 1, 1_000),
            "Waiting for the GPU — 1 turn (512 tokens) ahead, 1 s"
        );
        assert_eq!(queued_status(0, 0, 3_200), "Waiting for the GPU — 3 s");
    }
}

#[cfg(test)]
pub(super) mod status_detail_tests {
    use super::ZendSession;
    use crate::config::DaemonConfig;
    use crate::workspace::single_repo;
    use tempfile::TempDir;

    /// **A detail sent while nobody is reading is the next snapshot's.** The
    /// load sends each phase's name once, as the phase begins; when the
    /// channel had no reader open, the send was refused and dropped, and every
    /// snapshot read an empty detail.
    #[test]
    fn a_detail_sent_between_snapshots_is_the_next_snapshots() {
        let tmp = TempDir::new().unwrap();
        let session = ZendSession::new(DaemonConfig::new(
            single_repo(tmp.path(), "project").unwrap(),
        ));
        session
            .status_tx
            .send("Reading experts into host memory".into())
            .unwrap();
        assert_eq!(
            session.status_snapshot().detail,
            "Reading experts into host memory"
        );
    }
}
