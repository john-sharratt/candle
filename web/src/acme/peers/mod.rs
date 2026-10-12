//! This gateway's ACME shared with its peers (§13.4): what each peer frame
//! does here, and what the renewal loop asks of the peers — whether this
//! gateway orders a site, announcing an order and its end, publishing its
//! challenges, sharing what it was issued, its ledger and its account.
//!
//! | Module | |
//! |---|---|
//! | [`publish`] | challenges published on every peer and acknowledged |
//! | [`inventory`] | the certificates exchanged on connect; the later `not_after` wins |
//! | [`view`] | the Peers block's ACME columns |

pub mod inventory;
pub mod publish;
pub mod view;

use std::collections::{BTreeMap, BTreeSet};
use std::path::PathBuf;
use std::sync::{Arc, Mutex, MutexGuard};

use anyhow::Result;
use cluster::clock::now_ms;
use cluster::frame::{AcmeCert, CertLine, Frame};
use cluster::peer::{PeerHandler, Peering};
use instant_acme::AccountCredentials;
use tokio::sync::Notify;

use self::inventory::{line, newer_here, replaces};
use super::certs::Certs;
use super::ledger::Ledger;
use crate::acme::issue::Issued;

#[derive(Default)]
struct State {
    /// The peers that acknowledged each challenge this gateway published.
    acks: BTreeMap<String, BTreeSet<u64>>,
    /// Challenges peers published here, and when each lapses.
    foreign: BTreeMap<String, u64>,
    /// Sites a peer is ordering now, and which peer.
    ordering: BTreeMap<String, u64>,
    /// Each peer's inventory, as it last sent it.
    inventories: BTreeMap<u64, Vec<CertLine>>,
    /// The site this gateway is ordering now.
    mine: Option<String>,
}

pub struct AcmePeers {
    peering: Arc<Peering>,
    certs: Arc<Certs>,
    state: Mutex<State>,
    /// A peer acknowledged a challenge.
    acked: Notify,
    /// A peer left or finished an order: the renewal loop looks again.
    wake: Notify,
}

fn hex(node: u64) -> String {
    format!("{node:016x}")
}

impl AcmePeers {
    pub fn new(peering: Arc<Peering>, certs: Arc<Certs>) -> Arc<AcmePeers> {
        let peers = Arc::new(AcmePeers {
            peering,
            certs,
            state: Mutex::new(State::default()),
            acked: Notify::new(),
            wake: Notify::new(),
        });
        let handler: Arc<dyn PeerHandler> = peers.clone();
        peers.peering.set_handler(Arc::downgrade(&handler));
        peers
    }

    fn state(&self) -> MutexGuard<'_, State> {
        self.state.lock().unwrap_or_else(|e| e.into_inner())
    }

    fn ledger_path(&self) -> PathBuf {
        self.certs.store.ledger_path()
    }

    /// `text` about `subject` — a peer, or this gateway — in the log.
    fn note(&self, subject: u64, text: String) {
        tracing::info!(subject = hex(subject), "acme: {text}");
        self.peering.note(subject, format!("acme: {text}"));
    }

    /// Resolves when the renewal loop should look again before its timer.
    pub async fn woken(&self) {
        self.wake.notified().await;
    }

    /// Whether this gateway orders `site` now: it is the site's orderer among
    /// the gateways linked to it, and no peer has announced an order for it.
    pub fn orders(&self, site: &str) -> Result<(), String> {
        let orderer = self.peering.orderer(site);
        if orderer != self.peering.node() {
            return Err(format!("{} orders it", hex(orderer)));
        }
        match self.state().ordering.get(site) {
            Some(peer) => Err(format!("{} is ordering it now", hex(*peer))),
            None => Ok(()),
        }
    }

    /// An order for `site` begins here.
    pub fn ordering(&self, site: &str) {
        self.state().mine = Some(site.to_string());
        self.peering.broadcast(&Frame::AcmeOrdering {
            site: site.to_string(),
        });
    }

    /// The order for `site` has ended here.
    pub fn done(&self, site: &str) {
        self.state().mine = None;
        self.peering.broadcast(&Frame::AcmeDone {
            site: site.to_string(),
        });
    }

    /// `site`'s certificate as a frame, if this gateway holds one.
    fn cert_frame(&self, site: &str) -> Result<Option<Frame>> {
        let held = self.certs.held()?;
        let Some(s) = held.iter().find(|s| s.site == site) else {
            return Ok(None);
        };
        let (chain_pem, key_pem) = self.certs.store.read_pair(site)?;
        Ok(Some(Frame::AcmeCert(AcmeCert {
            site: site.to_string(),
            chain_pem,
            key_pem,
            not_after: s.not_after,
        })))
    }

    /// Send `frame`, carrying a key, to `peer` only over a verified link.
    fn send_secret(&self, peer: u64, frame: Frame) {
        if self.peering.secure(peer) {
            self.peering.send(peer, frame);
        } else {
            self.note(
                peer,
                "not over verified TLS: no certificate key or account is sent".into(),
            );
        }
    }

    /// `issued` for `site`, valid to `not_after`, installed here: every peer
    /// gets it.
    pub fn issued(&self, site: &str, issued: &Issued, not_after: u64) {
        let frame = Frame::AcmeCert(AcmeCert {
            site: site.to_string(),
            chain_pem: issued.chain_pem.clone(),
            key_pem: issued.key_pem.clone(),
            not_after,
        });
        for (peer, _) in self.peering.linked() {
            self.send_secret(peer, frame.clone());
        }
    }

    /// The ledger changed here: every peer merges it.
    pub fn ledger_changed(&self) -> Result<()> {
        let ledger = Ledger::load(&self.ledger_path())?;
        self.peering.broadcast(&Frame::AcmeLedger {
            ledger: serde_json::to_vec(&ledger)?,
        });
        Ok(())
    }

    /// The account was created here: every peer takes it.
    pub fn account_created(&self) -> Result<()> {
        if let Some(frame) = self.account_frame()? {
            for (peer, _) in self.peering.linked() {
                self.send_secret(peer, frame.clone());
            }
        }
        Ok(())
    }

    fn account_frame(&self) -> Result<Option<Frame>> {
        Ok(self
            .certs
            .store
            .load_account()?
            .map(|a| serde_json::to_vec(&a))
            .transpose()?
            .map(|account| Frame::AcmeAccount { account }))
    }

    fn inventory(&self) -> Result<Vec<CertLine>> {
        Ok(self.certs.held()?.iter().map(line).collect())
    }

    fn on_linked(&self, peer: u64, secure: bool) -> Result<()> {
        self.peering.send(
            peer,
            Frame::AcmeInventory {
                certs: self.inventory()?,
            },
        );
        self.peering.send(
            peer,
            Frame::AcmeLedger {
                ledger: serde_json::to_vec(&Ledger::load(&self.ledger_path())?)?,
            },
        );
        if secure {
            if let Some(frame) = self.account_frame()? {
                self.peering.send(peer, frame);
            }
        }
        // An order running here is announced to the newcomer too.
        if let Some(site) = self.state().mine.clone() {
            self.peering.send(peer, Frame::AcmeOrdering { site });
        }
        Ok(())
    }

    fn on_frame(&self, peer: u64, frame: Frame) -> Result<()> {
        let now = now_ms();
        let lapsed: Vec<String> = {
            let mut s = self.state();
            let lapsed: Vec<String> = s
                .foreign
                .iter()
                .filter(|(_, at)| **at <= now)
                .map(|(t, _)| t.clone())
                .collect();
            for t in &lapsed {
                s.foreign.remove(t);
            }
            lapsed
        };
        for t in lapsed {
            self.certs.challenges.clear(&t);
        }
        match frame {
            Frame::AcmeChallenge(c) => {
                self.certs.challenges.set(&c.token, &c.answer);
                self.state().foreign.insert(c.token.clone(), c.expires_ms);
                self.peering.send(peer, Frame::AcmeAck { token: c.token });
            }
            Frame::AcmeAck { token } => {
                if let Some(got) = self.state().acks.get_mut(&token) {
                    got.insert(peer);
                }
                self.acked.notify_waiters();
            }
            Frame::AcmeClear { token } => {
                self.state().foreign.remove(&token);
                self.certs.challenges.clear(&token);
            }
            Frame::AcmeCert(c) => {
                if !self.peering.secure(peer) {
                    self.note(
                        peer,
                        format!("refused {}'s certificate: not over verified TLS", c.site),
                    );
                    return Ok(());
                }
                let held = self
                    .certs
                    .held()?
                    .iter()
                    .find(|s| s.site == c.site)
                    .map(|s| s.not_after);
                if replaces(held, c.not_after) {
                    self.certs.install(&c.site, &c.chain_pem, &c.key_pem)?;
                    self.note(
                        peer,
                        format!(
                            "installed {}'s certificate, valid to {}",
                            c.site, c.not_after
                        ),
                    );
                }
            }
            Frame::AcmeLedger { ledger } => {
                let theirs: Ledger = serde_json::from_slice(&ledger)?;
                let mut mine = Ledger::default();
                mine.merge(&theirs);
                mine.save_merged(&self.ledger_path())?;
            }
            Frame::AcmeInventory { certs } => {
                let mine = self.inventory()?;
                for site in newer_here(&mine, &certs) {
                    if let Some(frame) = self.cert_frame(&site)? {
                        self.send_secret(peer, frame);
                    }
                }
                self.state().inventories.insert(peer, certs);
            }
            Frame::AcmeOrdering { site } => {
                self.state().ordering.insert(site, peer);
            }
            Frame::AcmeDone { site } => {
                let mut s = self.state();
                if s.ordering.get(&site) == Some(&peer) {
                    s.ordering.remove(&site);
                }
                drop(s);
                self.wake.notify_one();
            }
            Frame::AcmeAccount { account } => {
                if !self.peering.secure(peer) {
                    return Ok(());
                }
                if self.certs.store.load_account()?.is_none() {
                    let shared: AccountCredentials = serde_json::from_slice(&account)?;
                    self.certs.store.save_account(&shared)?;
                    self.note(peer, "took the shared ACME account".into());
                }
            }
            _ => {}
        }
        Ok(())
    }
}

impl PeerHandler for AcmePeers {
    fn linked(&self, peer: u64, secure: bool) {
        if let Err(e) = self.on_linked(peer, secure) {
            tracing::error!(peer = hex(peer), error = %format!("{e:#}"), "acme: a new peer was not told this gateway's state");
        }
    }

    fn frame(&self, peer: u64, frame: Frame) {
        if let Err(e) = self.on_frame(peer, frame) {
            tracing::error!(peer = hex(peer), error = %format!("{e:#}"), "acme: a peer's frame was not applied");
        }
    }

    fn unlinked(&self, peer: u64) {
        {
            let mut s = self.state();
            s.ordering.retain(|_, p| *p != peer);
            s.inventories.remove(&peer);
        }
        self.acked.notify_waiters();
        self.wake.notify_one();
    }
}
