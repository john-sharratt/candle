//! Announcing what a member holds (§3.9): `BUILD_HAVE` for its running build,
//! each build it keeps and each it verified — so a peer that lacks one fetches
//! it from here — and, as a candidate, `BUILD_OFFER` for its own build. Sent on
//! every admitted socket when one is admitted and every
//! [`ANNOUNCE_EVERY`](super::ANNOUNCE_EVERY), so a gateway restarted or a member
//! arrived since hears them too.

use std::time::Instant;

use super::served::{grouped, kept, running, verified, HeldAs};
use super::view::kept_lines;
use super::{Upgrader, ANNOUNCE_EVERY};
use crate::frame::{BuildArtifacts, Frame};

impl Upgrader {
    /// What it serves, read from disk.
    pub(super) async fn scan(&self) {
        let setup = self.setup.clone();
        let parts = self.parts.clone();
        let (items, kept_now) = tokio::task::spawn_blocking(move || {
            let mut items = running(&setup).unwrap_or_else(|e| {
                tracing::error!("the running build's artifacts are unreadable: {e}");
                Vec::new()
            });
            items.extend(kept(&setup));
            items.extend(verified(&parts));
            (items, kept_lines(&setup.exe))
        })
        .await
        .unwrap_or_default();
        let mut s = self.state();
        s.served = items;
        s.kept = kept_now;
        s.scanned = true;
        s.announce_due = true;
    }

    /// Announce, when it is due.
    pub(super) async fn announce_tick(&self) {
        let (haves, offer) = {
            let mut s = self.state();
            let due = s.announce_due
                || s.last_announce
                    .is_none_or(|at| at.elapsed() >= ANNOUNCE_EVERY);
            if !due || !s.scanned {
                return;
            }
            s.announce_due = false;
            s.last_announce = Some(Instant::now());
            let held: Vec<_> = s
                .served
                .iter()
                .filter(|i| i.from != HeldAs::Offered)
                .cloned()
                .collect();
            let offered: Vec<_> = s
                .served
                .iter()
                .filter(|i| i.from == HeldAs::Offered)
                .cloned()
                .collect();
            let offer = (!offered.is_empty()).then(|| {
                let mut b = grouped(&offered).remove(0);
                // The UI goes with the binaries.
                for i in s.served.iter().filter(|i| {
                    i.from == HeldAs::Running && i.artifact.kind != self.setup.info.kind
                }) {
                    b.artifacts.push(i.artifact.clone());
                }
                b
            });
            (grouped(&held), offer)
        };
        for m in self.apps() {
            let candidate = self.state().candidates.contains_key(m.app());
            for c in m.admitted_sockets() {
                for b in &haves {
                    m.send_on(c, Frame::BuildHave(b.clone()));
                }
                if let Some(b) = offer.as_ref().filter(|_| candidate) {
                    m.send_on(c, Frame::BuildOffer(b.clone()));
                }
            }
        }
    }

    /// Announce `b` on every admitted socket at once: a download just
    /// verified.
    pub(super) fn announce_now(&self, b: &BuildArtifacts) {
        for m in self.apps() {
            for c in m.admitted_sockets() {
                m.send_on(c, Frame::BuildHave(b.clone()));
            }
        }
    }
}
