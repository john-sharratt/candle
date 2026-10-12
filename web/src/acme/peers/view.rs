//! The Peers block's ACME columns (§13.4, §12.2): each site's orderer and the
//! orders running, the challenges outstanding with the peers that have
//! acknowledged them, every gateway's certificate per site, and the shared
//! ledger.

use cluster::upgrade::sum::hex as hex_bytes;
use serde::Serialize;

use super::inventory::line;
use super::{hex, AcmePeers};
use crate::acme::ledger::Ledger;

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct SiteOrder {
    pub site: String,
    pub orderer: String,
    /// The gateway ordering it now, if one is.
    pub ordering: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct ChallengeView {
    pub token: String,
    pub acked_by: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct CertView {
    pub gateway: String,
    pub site: String,
    pub not_after: u64,
    pub digest: String,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct AcmeView {
    pub sites: Vec<SiteOrder>,
    pub challenges: Vec<ChallengeView>,
    pub certs: Vec<CertView>,
    pub ledger: Ledger,
}

impl AcmePeers {
    /// The view for `sites`, the site table's certificate groups.
    pub fn view(&self, sites: &[String]) -> AcmeView {
        let me = self.peering.node();
        let s = self.state();
        let mut certs: Vec<CertView> = self
            .certs
            .held()
            .unwrap_or_default()
            .iter()
            .map(|c| {
                let l = line(c);
                CertView {
                    gateway: hex(me),
                    site: l.site,
                    not_after: l.not_after,
                    digest: hex_bytes(&l.digest),
                }
            })
            .collect();
        for (peer, lines) in &s.inventories {
            certs.extend(lines.iter().map(|l| CertView {
                gateway: hex(*peer),
                site: l.site.clone(),
                not_after: l.not_after,
                digest: hex_bytes(&l.digest),
            }));
        }
        AcmeView {
            sites: sites
                .iter()
                .map(|site| SiteOrder {
                    site: site.clone(),
                    orderer: hex(self.peering.orderer(site)),
                    ordering: if s.mine.as_deref() == Some(site) {
                        Some(hex(me))
                    } else {
                        s.ordering.get(site).map(|p| hex(*p))
                    },
                })
                .collect(),
            challenges: s
                .acks
                .iter()
                .map(|(token, acked)| ChallengeView {
                    token: token.clone(),
                    acked_by: acked.iter().map(|n| hex(*n)).collect(),
                })
                .collect(),
            certs,
            ledger: Ledger::load(&self.ledger_path()).unwrap_or_default(),
        }
    }
}
