//! World pulses (§9.1): transients a world's authority broadcasts — the steps
//! of a walk, a line spoken — never persisted, never retransmitted. A pulse
//! goes out on one socket: every member joins every cluster, so one gateway
//! reaches them all, and a copy through a second would only play the
//! animation twice. A member that misses one misses only that animation.

use tokio::sync::broadcast;

use super::replicator::Replicator;
use crate::frame::{Frame, Pulse};

/// How many pulses a slow follower may fall behind by before it misses some.
pub(super) const PULSES_KEPT: usize = 4096;

impl Replicator {
    /// Every pulse that arrives from now on.
    pub fn pulses(&self) -> broadcast::Receiver<Pulse> {
        self.pulses.subscribe()
    }

    /// Broadcast `pulse` to the app's other members; whether a socket took it.
    pub fn pulse(&self, pulse: Pulse) -> bool {
        let socket = self.state().admitted().into_iter().next();
        socket.is_some_and(|(_, h)| h.ordered(Frame::Pulse(pulse)).is_ok())
    }

    /// A pulse arrived from another member.
    pub(super) fn pulsed(&self, pulse: Pulse) {
        let _ = self.pulses.send(pulse);
    }
}
