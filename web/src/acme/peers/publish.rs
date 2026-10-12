//! An order's answers published on every peer (§13.4 step 2): `ACME_CHALLENGE`
//! to each linked peer, and the CA is asked to validate only once each has
//! acknowledged with `ACME_ACK` — or has dropped its link, or [`ACK_WAIT`] has
//! passed, after which the order goes ahead on the gateways that did.

use std::collections::BTreeSet;
use std::time::Duration;

use cluster::clock::now_ms;
use cluster::frame::{AcmeChallenge, Frame};
use tokio::time::{sleep_until, Instant};

use super::AcmePeers;
use crate::acme::publish::Publish;

/// How long the orderer waits for its peers' acknowledgements.
pub const ACK_WAIT: Duration = Duration::from_secs(30);

/// How long a peer answers a challenge whose orderer never withdraws it.
pub const CHALLENGE_LIFE: Duration = Duration::from_secs(600);

/// `site`'s order, published on this gateway and its peers.
pub struct Peered<'a> {
    pub peers: &'a AcmePeers,
    pub site: &'a str,
}

impl Publish for Peered<'_> {
    async fn publish(&self, token: &str, answer: &str) {
        self.peers.publish(self.site, token, answer).await;
    }

    fn withdraw(&self, token: &str) {
        self.peers.withdraw(token);
    }
}

impl AcmePeers {
    /// Answer `token` here and on every linked peer, and wait for their
    /// acknowledgements.
    pub async fn publish(&self, site: &str, token: &str, answer: &str) {
        self.certs.challenges.set(token, answer);
        self.state().acks.insert(token.to_string(), BTreeSet::new());
        let sent = self.peering.broadcast(&Frame::AcmeChallenge(AcmeChallenge {
            site: site.to_string(),
            token: token.to_string(),
            answer: answer.to_string(),
            expires_ms: now_ms() + CHALLENGE_LIFE.as_millis() as u64,
        }));
        let deadline = Instant::now() + ACK_WAIT;
        loop {
            let acked = self.acked.notified();
            let linked: BTreeSet<u64> = self.peering.linked().into_iter().map(|(n, _)| n).collect();
            let waiting: Vec<u64> = {
                let s = self.state();
                let got = s.acks.get(token);
                sent.iter()
                    .filter(|n| linked.contains(n) && !got.is_some_and(|g| g.contains(n)))
                    .copied()
                    .collect()
            };
            if waiting.is_empty() {
                return;
            }
            tokio::select! {
                _ = acked => {}
                _ = sleep_until(deadline) => {
                    tracing::warn!(
                        site, token, waiting = ?waiting,
                        "acme: peers did not acknowledge the challenge — validating without them"
                    );
                    return;
                }
            }
        }
    }

    /// Stop answering `token`, here and on every peer.
    pub fn withdraw(&self, token: &str) {
        self.certs.challenges.clear(token);
        self.state().acks.remove(token);
        self.peering.broadcast(&Frame::AcmeClear {
            token: token.to_string(),
        });
    }
}
