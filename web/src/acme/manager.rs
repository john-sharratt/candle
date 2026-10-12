//! The renewal loop: on start and then twice a day, every site's certificate is
//! checked and, where one is due and the ledger allows it, ordered.
//!
//! A site's certificate is due when
//!
//! * there is none,
//! * the CA's renewal-information window has opened for it (the order then
//!   carries `replaces`, which Let's Encrypt exempts from its limits),
//! * a third of its lifetime or less is left — the floor that renews it even
//!   if the renewal window could not be read, or
//! * a name the site serves is missing from it **and** that name now reaches
//!   this gateway.
//!
//! Sites are ordered one at a time. Every order's outcome is written to the
//! ledger before the next step can fail, so no path through here — an error, a
//! crash, a restart — can place an order the ledger does not know about.
//!
//! With peers (§13.4), a due site is ordered only by its orderer among the
//! gateways linked now, and not while a peer announces an order for it; the
//! ledger is merged with the peers' and sent to them on every change, and a
//! new account is shared. A peer leaving, or finishing an order, wakes the
//! loop early — after at least [`MIN_SLEEP`] — since this gateway may now be
//! the orderer of a site that is due.

use std::sync::Arc;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use anyhow::{Context, Result};
use instant_acme::{Account, CertificateIdentifier, NewAccount};
use tokio::time::sleep;

use super::ca::LetsEncrypt;
use super::certs::Certs;
use super::challenges::Challenges;
use super::issue::IssueError;
use super::ledger::{Attempt, Ledger, Outcome, HOUR};
use super::names::Group;
use super::order::order_site;
use super::peers::AcmePeers;
use super::preflight::reachable;
use super::store::{Store, Stored};
use super::LETS_ENCRYPT;
use crate::config::Acme;
use crate::tls::Resolver;

/// How often every certificate is looked at when nothing asks sooner.
const CHECK_EVERY: u64 = 12 * HOUR;
/// The shortest sleep between passes, so a wait that has just ended cannot
/// spin the loop.
const MIN_SLEEP: u64 = 300;

pub fn unix_now() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

/// Why a site is being ordered for.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Due {
    Missing,
    Window,
    Floor,
    NewNames,
}

pub struct Manager {
    groups: Vec<Group>,
    certs: Arc<Certs>,
    peers: Option<Arc<AcmePeers>>,
}

impl Manager {
    pub fn new(
        acme: Acme,
        groups: Vec<Group>,
        challenges: Challenges,
        resolver: Arc<Resolver>,
    ) -> Self {
        // The ACME client builds its TLS configuration from the process-wide
        // provider, and rustls has none until one is installed. `ring`, like
        // every configuration this crate builds itself. Already installed is
        // fine: it can only have been installed as the same one.
        let _ = rustls::crypto::ring::default_provider().install_default();
        Self {
            groups,
            certs: Arc::new(Certs {
                store: Store::new(&acme.store),
                resolver,
                challenges,
            }),
            peers: None,
        }
    }

    /// What this gateway serves, for its peers to install into too.
    pub fn certs(&self) -> Arc<Certs> {
        self.certs.clone()
    }

    /// Order with `peers` (§13.4).
    pub fn with_peers(mut self, peers: Arc<AcmePeers>) -> Self {
        self.peers = Some(peers);
        self
    }

    /// Put what the store holds in front of the entrance. Called before the
    /// entrance binds, so a restart serves the certificates it already has
    /// without waiting on the CA.
    pub fn install_stored(&self) -> Result<()> {
        let certs = self.certs.held()?;
        self.certs.resolver.install(&certs);
        tracing::info!(
            names = ?self.certs.resolver.names(),
            "acme: certificates installed from the store"
        );
        Ok(())
    }

    /// Run until the process ends.
    pub async fn run(self) {
        loop {
            let now = unix_now();
            let next = match self.pass(now).await {
                Ok(next) => next,
                Err(e) => {
                    tracing::error!(error = %format!("{e:#}"), "acme: the renewal pass failed");
                    now + HOUR
                }
            };
            let secs = next.saturating_sub(unix_now()).max(MIN_SLEEP);
            tracing::info!(in_secs = secs, "acme: next certificate check");
            match &self.peers {
                Some(peers) => {
                    tokio::select! {
                        _ = sleep(Duration::from_secs(secs)) => {}
                        _ = peers.woken() => {
                            let rested = unix_now().saturating_sub(now);
                            let wait = MIN_SLEEP.saturating_sub(rested);
                            tracing::info!(in_secs = wait, "acme: a peer left or finished an order — checking again");
                            sleep(Duration::from_secs(wait)).await;
                        }
                    }
                }
                None => sleep(Duration::from_secs(secs)).await,
            }
        }
    }

    /// Write `ledger`, merged with what the file holds, and send it to the
    /// peers.
    fn save(&self, ledger: &mut Ledger) -> Result<()> {
        ledger.save_merged(&self.certs.store.ledger_path())?;
        if let Some(p) = &self.peers {
            p.ledger_changed()?;
        }
        Ok(())
    }

    /// One look at every site. Returns when the next look is wanted.
    async fn pass(&self, now: u64) -> Result<u64> {
        let store = &self.certs.store;
        let mut ledger = Ledger::load(&store.ledger_path())?;
        let stored = self.certs.held()?;
        let mut account = match store.load_account()? {
            Some(creds) => Some(
                Account::builder()?
                    .from_credentials(creds)
                    .await
                    .context("acme: restoring the account")?,
            ),
            None => None,
        };
        let mut next = now + CHECK_EVERY;

        // Every name is proved first, whether or not anything is due: the
        // log then always says which names would pass HTTP-01 right now, and
        // a broken route shows up before the order that would have spent a
        // failed validation on it. Only this gateway's own traffic — the CA is
        // not involved.
        let mut reach: Vec<Vec<String>> = Vec::with_capacity(self.groups.len());
        for group in &self.groups {
            let ok = reachable(&self.certs.challenges, &group.names).await;
            let unreachable: Vec<&String> =
                group.names.iter().filter(|n| !ok.contains(n)).collect();
            if unreachable.is_empty() {
                tracing::info!(site = %group.site, names = ?ok, "acme: preflight — every name reaches this gateway");
            } else {
                tracing::warn!(
                    site = %group.site, reachable = ?ok, ?unreachable,
                    "acme: preflight — some names do not reach this gateway"
                );
            }
            reach.push(ok);
        }

        for (group, names) in self.groups.iter().zip(reach) {
            let current = stored.iter().find(|s| s.site == group.site);
            let (due, replaces) = match self
                .due(account.as_ref(), current, group, now, &mut next)
                .await
            {
                Some(d) => d,
                None => continue,
            };

            if let Some(p) = &self.peers {
                if let Err(why) = p.orders(&group.site) {
                    tracing::info!(site = %group.site, ?due, %why, "acme: due, ordered by a peer");
                    continue;
                }
            }

            if let Err(wait) = ledger.may_order(&group.site, now) {
                tracing::info!(
                    site = %group.site, ?due, why = wait.why, until = wait.until,
                    "acme: due, held by the ledger"
                );
                next = next.min(wait.until);
                continue;
            }

            if names.is_empty() {
                tracing::warn!(
                    site = %group.site, ?due,
                    "acme: no name of this site reaches this gateway — nothing is ordered"
                );
                continue;
            }
            if due == Due::NewNames {
                let held = current.map(|c| c.names.as_slice()).unwrap_or(&[]);
                if names.iter().all(|n| held.contains(n)) {
                    // The names it lacks still do not reach here.
                    continue;
                }
            }

            if account.is_none() {
                if let Err(wait) = ledger.may_create_account(now) {
                    tracing::warn!(
                        why = wait.why,
                        until = wait.until,
                        "acme: no account, and none may be created yet"
                    );
                    next = next.min(wait.until);
                    break;
                }
                // Recorded before the request: a creation that fails after the
                // CA counted it must still count here.
                ledger.record_account(now);
                self.save(&mut ledger)?;
                let (acct, creds) = Account::builder()?
                    .create(
                        &NewAccount {
                            contact: &[],
                            terms_of_service_agreed: true,
                            only_return_existing: false,
                        },
                        LETS_ENCRYPT.to_owned(),
                        None,
                    )
                    .await
                    .context("acme: creating the account")?;
                store.save_account(&creds)?;
                tracing::info!("acme: account created");
                if let Some(p) = &self.peers {
                    p.account_created()?;
                }
                account = Some(acct);
            }
            let acct = account.as_ref().expect("set above");

            tracing::info!(site = %group.site, ?due, ?names, "acme: ordering");
            let ca = LetsEncrypt {
                account: acct,
                replaces,
            };
            let ordered =
                order_site(&ca, &self.certs, self.peers.as_deref(), &group.site, &names).await;
            let outcome = match &ordered.outcome {
                Ok(()) => Outcome::Issued,
                Err(IssueError::RateLimited(_)) => Outcome::RateLimited,
                Err(IssueError::Failed(_)) => Outcome::Failed,
            };
            ledger.record(Attempt {
                site: group.site.clone(),
                names: names.clone(),
                at: unix_now(),
                outcome,
            });
            self.save(&mut ledger)?;
            ordered.installed?;

            match ordered.outcome {
                Ok(()) => {
                    tracing::info!(site = %group.site, ?names, "acme: certificate issued and installed");
                }
                Err(IssueError::RateLimited(m)) => {
                    tracing::error!(site = %group.site, detail = %m, "acme: the CA reports a rate limit — every site backs off");
                    break;
                }
                Err(IssueError::Failed(m)) => {
                    tracing::warn!(site = %group.site, error = %m, "acme: the order failed");
                }
            }
        }
        Ok(next)
    }

    /// Whether `group` needs an order, why, and the certificate it replaces
    /// when the CA's renewal window is the reason.
    async fn due(
        &self,
        account: Option<&Account>,
        current: Option<&Stored>,
        group: &Group,
        now: u64,
        next: &mut u64,
    ) -> Option<(Due, Option<CertificateIdentifier<'static>>)> {
        let Some(cur) = current else {
            return Some((Due::Missing, None));
        };
        let id: Option<CertificateIdentifier<'static>> = cur
            .certified
            .cert
            .first()
            .and_then(|leaf| CertificateIdentifier::try_from(leaf).ok());

        if let (Some(acct), Some(id)) = (account, id.as_ref()) {
            match acct.renewal_info(id).await {
                Ok((info, retry)) => {
                    let start = info.suggested_window.start.unix_timestamp().max(0) as u64;
                    *next = (*next).min(now + retry.as_secs().max(HOUR));
                    if now >= start {
                        return Some((Due::Window, Some(id.clone())));
                    }
                    *next = (*next).min(start);
                }
                Err(e) => {
                    tracing::warn!(site = %group.site, error = %e, "acme: no renewal window — the expiry floor stands alone");
                }
            }
        }
        if cur.past_renewal_floor(now) {
            return Some((Due::Floor, id));
        }
        if group.names.iter().any(|n| !cur.names.contains(n)) {
            return Some((Due::NewNames, None));
        }
        None
    }
}
