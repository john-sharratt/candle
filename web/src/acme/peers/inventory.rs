//! The certificate inventory two peers exchange on connect, and what each
//! sends the other from it: a certificate the peer lacks, or holds with an
//! earlier `not_after`. The later `not_after` wins, so the outcome is the same
//! whichever arrives first.

use cluster::frame::CertLine;
use cluster::peer::digest;

use crate::acme::store::Stored;

/// One certificate as an inventory line, its digest over the chain's DER.
pub fn line(s: &Stored) -> CertLine {
    let mut der = Vec::new();
    for c in &s.certified.cert {
        der.extend_from_slice(c.as_ref());
    }
    CertLine {
        site: s.site.clone(),
        not_after: s.not_after,
        digest: digest(&der),
    }
}

/// The sites whose certificate `mine` holds newer than `theirs`, or theirs
/// lacks.
pub fn newer_here(mine: &[CertLine], theirs: &[CertLine]) -> Vec<String> {
    mine.iter()
        .filter(|m| {
            theirs
                .iter()
                .find(|t| t.site == m.site)
                .is_none_or(|t| t.not_after < m.not_after)
        })
        .map(|m| m.site.clone())
        .collect()
}

/// Whether a certificate for a site, valid to `offered`, replaces the one held
/// valid to `held`.
pub fn replaces(held: Option<u64>, offered: u64) -> bool {
    held.is_none_or(|h| offered > h)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn l(site: &str, not_after: u64) -> CertLine {
        CertLine {
            site: site.into(),
            not_after,
            digest: [0; 32],
        }
    }

    /// A peer is sent what it lacks and what it holds older; not what it holds
    /// as new or newer.
    #[test]
    fn a_peer_is_sent_what_it_lacks_or_holds_older() {
        let mine = [l("tokera", 200), l("zend", 100), l("npcd", 100)];
        let theirs = [l("tokera", 100), l("zend", 100), l("other", 50)];
        assert_eq!(newer_here(&mine, &theirs), ["tokera", "npcd"]);
        assert!(newer_here(&theirs, &theirs).is_empty());
    }

    /// The later `not_after` wins; an equal one changes nothing.
    #[test]
    fn the_later_not_after_wins() {
        assert!(replaces(None, 1));
        assert!(replaces(Some(1), 2));
        assert!(!replaces(Some(2), 2));
        assert!(!replaces(Some(3), 2));
    }
}
