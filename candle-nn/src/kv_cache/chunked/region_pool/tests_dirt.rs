//! Tests for which region releases force a device-wide wait before reuse.
//!
//! A region's dirt records when its bytes last belonged to someone else and
//! which stream may still be touching them. Ground that any stream may still
//! read waits for every claim until a quiesce retires it, while ground only one
//! stream touches waits only for a claim on another stream. These tests pin that
//! a later epoch's tenant does not widen an earlier one's readers.

use super::*;

#[cfg(test)]
pub(super) mod dirt_tests {
    use super::{Dirt, ANY_STREAM};

    const COMPUTE: usize = 0x1000;
    const COPY: usize = 0x2000;

    #[test]
    fn pristine_ground_never_waits() {
        let d = Dirt::default();
        assert!(!d.needs_sync(1, COMPUTE));
        assert!(!d.needs_sync(1, COPY));
    }

    #[test]
    fn ground_any_stream_left_waits_for_every_claim_this_epoch_and_none_after() {
        let d = Dirt::any(4);
        assert!(d.needs_sync(4, COMPUTE));
        assert!(d.needs_sync(4, COPY));
        // A quiesce since retired its readers.
        assert!(!d.needs_sync(5, COMPUTE));
    }

    #[test]
    fn ground_one_stream_left_waits_only_for_a_claim_on_another_stream() {
        let d = Dirt::default().left(4, COMPUTE);
        assert_eq!((d.epoch, d.on), (4, COMPUTE));
        // Stream order retires the forward; another stream is not ordered
        // behind it.
        assert!(!d.needs_sync(4, COMPUTE));
        assert!(d.needs_sync(4, COPY));
        assert!(!d.needs_sync(5, COPY));
    }

    #[test]
    fn two_tenants_in_one_epoch_leave_any_streams_bytes() {
        // A KV arena released (any stream), then the tier over it (compute).
        let d = Dirt::any(4).left(4, COMPUTE);
        assert_eq!(d.on, ANY_STREAM);
        assert!(d.needs_sync(4, COMPUTE));
        // Two streams' tenants in one epoch.
        let d = Dirt::default().left(4, COMPUTE).left(4, COPY);
        assert_eq!(d.on, ANY_STREAM);
        // The same stream twice stays that stream's.
        let d = Dirt::default().left(4, COMPUTE).left(4, COMPUTE);
        assert_eq!(d.on, COMPUTE);
    }

    #[test]
    fn an_earlier_epochs_bytes_do_not_widen_a_later_tenant() {
        // Left by any stream before a quiesce, then by the tier after it: only
        // the tier's readers remain.
        let d = Dirt::any(3).left(4, COMPUTE);
        assert_eq!((d.epoch, d.on), (4, COMPUTE));
        assert!(!d.needs_sync(4, COMPUTE));
    }
}
