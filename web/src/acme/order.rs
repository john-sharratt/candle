//! One site's order, start to finish, as §13.4 lays it out: announced to the
//! peers with `ACME_ORDERING`, its challenges published on every peer and
//! acknowledged before the CA validates, what is issued installed here and
//! sent to every peer, then `ACME_DONE`. Without peers it is the CA's order
//! alone. Whether to order at all is the caller's: the ledger's, and
//! [`AcmePeers::orders`].

use anyhow::Result;

use super::ca::Ca;
use super::certs::Certs;
use super::issue::IssueError;
use super::peers::publish::Peered;
use super::peers::AcmePeers;

/// How an order ended: the CA's answer, and whether what it issued was
/// installed. The ledger records the first whatever the second says.
pub struct Ordered {
    pub outcome: Result<(), IssueError>,
    pub installed: Result<()>,
}

pub async fn order_site<C: Ca>(
    ca: &C,
    certs: &Certs,
    peers: Option<&AcmePeers>,
    site: &str,
    names: &[String],
) -> Ordered {
    if let Some(p) = peers {
        p.ordering(site);
    }
    let result = match peers {
        Some(p) => ca.order(names, &Peered { peers: p, site }).await,
        None => ca.order(names, &certs.challenges).await,
    };
    let (outcome, installed) = match result {
        Ok(issued) => {
            let installed =
                certs
                    .install(site, &issued.chain_pem, &issued.key_pem)
                    .map(|not_after| {
                        if let Some(p) = peers {
                            p.issued(site, &issued, not_after);
                        }
                    });
            (Ok(()), installed)
        }
        Err(e) => (Err(e), Ok(())),
    };
    if let Some(p) = peers {
        p.done(site);
    }
    Ordered { outcome, installed }
}
