//! Fault mode for a full disk (§7.7), and for a build that failed
//! certification (§3.8). A member whose volume falls below the floor, or whose
//! write failed for lack of space, stays connected and says so:
//! it hands back its claims and leases, proposes nothing, stops applying
//! remote events and sending `ACK`s — but keeps voting through its vote
//! journal, keeps sending its own events and answering fetches, and frees what
//! it can by dropping dead segments, which needs no scratch space.
//!
//! It leaves once free space is back above twice the floor and a test append
//! and commit succeed — the hysteresis keeps it from flapping at the floor —
//! then catches up on what was relayed while it was faulted before it takes
//! work again.

use std::sync::Arc;

use super::catchup::Purpose;
use super::replicator::Replicator;
use super::store::Disk;
use crate::clock::now_ms;

/// The least free space below which a member is in fault mode.
pub const FLOOR_MIN: u64 = 2 << 30;

/// The fault floor of a volume of `total` bytes: 2 GiB, or 1% of the volume if
/// that is larger.
pub fn floor(total: u64) -> u64 {
    FLOOR_MIN.max(total / 100)
}

/// Why a member is in fault mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Cause {
    /// Its disk is full: it leaves once space is back (§7.7).
    Disk,
    /// Its build failed certification (§3.8): it leaves when the build is
    /// accepted after all, never by itself.
    Certification,
}

/// A member in fault mode.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Fault {
    pub cause: Cause,
    pub reason: String,
    pub since_ms: u64,
    pub claims_released: usize,
    /// Free space when it entered, against which what it freed is counted.
    pub free_at_entry: u64,
    pub freed: u64,
}

/// What a disk reading means for a member in `faulted` state: enter with the
/// reason, try to leave, or nothing.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Verdict {
    Enter(String),
    TryLeave,
    Stay,
}

/// Judge `disk`. Entering needs a failed write or free space under the floor;
/// leaving needs twice the floor (and then a test write, which the caller
/// makes).
pub fn judge(disk: Disk, faulted: bool) -> Verdict {
    let floor = floor(disk.total);
    if !faulted {
        if disk.write_failed {
            return Verdict::Enter("a write failed for lack of space".into());
        }
        if disk.free < floor {
            return Verdict::Enter(format!(
                "{} MiB free, under the floor of {} MiB",
                disk.free >> 20,
                floor >> 20
            ));
        }
        return Verdict::Stay;
    }
    if disk.free >= 2 * floor {
        Verdict::TryLeave
    } else {
        Verdict::Stay
    }
}

impl Replicator {
    /// Whether the member is in fault mode.
    pub fn faulted(&self) -> bool {
        self.state().fault.is_some()
    }

    /// The fault, while there is one.
    pub fn fault(&self) -> Option<Fault> {
        self.state().fault.clone()
    }

    /// Measure the volume and act on it: enter fault mode, free space while in
    /// it, or leave. Returns the reading.
    pub async fn check_disk(self: &Arc<Self>) -> Option<Disk> {
        let store = self.store.clone();
        let disk = tokio::task::spawn_blocking(move || store.disk())
            .await
            .ok()?;
        // A build that failed certification stays faulted whatever the disk
        // says: space coming back is no reason to take work.
        if self
            .fault()
            .is_some_and(|f| f.cause == Cause::Certification)
        {
            return Some(disk);
        }
        let faulted = self.faulted();
        match judge(disk, faulted) {
            Verdict::Enter(reason) => self.enter_fault(Cause::Disk, reason, disk).await,
            Verdict::Stay if faulted => self.free_space(disk).await,
            Verdict::TryLeave => {
                self.free_space(disk).await;
                let store = self.store.clone();
                let written = tokio::task::spawn_blocking(move || store.test_write())
                    .await
                    .unwrap_or(false);
                if written {
                    self.leave_fault();
                }
            }
            Verdict::Stay => {}
        }
        Some(disk)
    }

    async fn enter_fault(self: &Arc<Self>, cause: Cause, reason: String, disk: Disk) {
        let released = self.hand_back().await;
        {
            let mut s = self.state();
            s.fault = Some(Fault {
                cause,
                reason: reason.clone(),
                since_ms: now_ms(),
                claims_released: released,
                free_at_entry: disk.free,
                freed: 0,
            });
            for r in s.ready.iter_mut() {
                *r = false;
            }
            // Nothing is applied in fault mode, so a catch-up in progress
            // could only stall; leaving starts fresh ones.
            let running: Vec<u64> = s.catchups.keys().copied().collect();
            for id in running {
                s.drop_catchup(id);
            }
        }
        tracing::error!(node = self.node, "fault mode: {reason}");
        if let Some(m) = self.member() {
            m.capacity_changed();
        }
        self.coord_now();
    }

    /// Drop a dead segment if there is one, and count what has been freed.
    async fn free_space(self: &Arc<Self>, disk: Disk) {
        let store = self.store.clone();
        let _ = tokio::task::spawn_blocking(move || store.drop_dead_segment()).await;
        if let Some(f) = self.state().fault.as_mut() {
            f.freed = f.freed.max(disk.free.saturating_sub(f.free_at_entry));
        }
    }

    /// Its build failed certification (§3.8): fault mode, with the failing
    /// routes as the reason. It still votes and sends what it wrote; the
    /// gateways route nothing to it.
    pub async fn certification_failed(self: &Arc<Self>, reason: String) {
        if self
            .fault()
            .is_some_and(|f| f.cause == Cause::Certification)
        {
            return;
        }
        let store = self.store.clone();
        let disk = tokio::task::spawn_blocking(move || store.disk())
            .await
            .unwrap_or_default();
        self.enter_fault(
            Cause::Certification,
            format!("certification failed: {reason}"),
            disk,
        )
        .await;
    }

    /// Its build was accepted after all (`certify-anyway`, or rolled out):
    /// leave a certification fault, catching up first.
    pub fn certification_accepted(self: &Arc<Self>) {
        if self
            .fault()
            .is_some_and(|f| f.cause == Cause::Certification)
        {
            self.leave_fault();
        }
    }

    /// Space is back and a test write succeeded, or a failed build was
    /// accepted: catch up on what was relayed meanwhile, then take work again.
    fn leave_fault(self: &Arc<Self>) {
        let sockets: Vec<usize> = {
            let mut s = self.state();
            s.fault = None;
            s.admitted().into_iter().map(|(c, _)| c).collect()
        };
        tracing::warn!(node = self.node, "fault mode over: catching up");
        for c in sockets {
            self.start_catchup(c, Purpose::Admission, None);
        }
        if let Some(m) = self.member() {
            m.capacity_changed();
        }
        self.coord_now();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const GIB: u64 = 1 << 30;

    fn disk(free: u64, total: u64, write_failed: bool) -> Disk {
        Disk {
            free,
            total,
            write_failed,
        }
    }

    /// The floor is 2 GiB, or 1% of a volume larger than 200 GiB.
    #[test]
    fn the_floor_is_two_gib_or_one_percent() {
        assert_eq!(floor(100 * GIB), 2 * GIB);
        assert_eq!(floor(1000 * GIB), 10 * GIB);
    }

    /// A failed write or free space under the floor enters; leaving needs
    /// twice the floor, so free space hovering at the floor does not flap.
    #[test]
    fn entering_and_leaving_have_hysteresis() {
        assert!(matches!(
            judge(disk(GIB, 100 * GIB, false), false),
            Verdict::Enter(_)
        ));
        assert!(matches!(
            judge(disk(50 * GIB, 100 * GIB, true), false),
            Verdict::Enter(_)
        ));
        assert_eq!(judge(disk(3 * GIB, 100 * GIB, false), false), Verdict::Stay);
        assert_eq!(judge(disk(3 * GIB, 100 * GIB, false), true), Verdict::Stay);
        assert_eq!(
            judge(disk(4 * GIB, 100 * GIB, false), true),
            Verdict::TryLeave
        );
    }
}
