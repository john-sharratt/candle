//! The certificate authority an order goes to: Let's Encrypt through the
//! account, behind one call, so the order's steps around it — announcing it to
//! peers, installing and sharing what was issued — are the same whoever signs.

use std::future::Future;

use instant_acme::{Account, CertificateIdentifier};

use super::issue::{issue, IssueError, Issued};
use super::publish::Publish;

pub trait Ca: Sync {
    /// Order a certificate for `names`, its HTTP-01 answers published through
    /// `publish`.
    fn order<P: Publish>(
        &self,
        names: &[String],
        publish: &P,
    ) -> impl Future<Output = Result<Issued, IssueError>> + Send;
}

/// Let's Encrypt, as `account`. `replaces` names the certificate a renewal
/// replaces, when the CA's renewal window asked for it.
pub struct LetsEncrypt<'a> {
    pub account: &'a Account,
    pub replaces: Option<CertificateIdentifier<'static>>,
}

impl Ca for LetsEncrypt<'_> {
    async fn order<P: Publish>(&self, names: &[String], publish: &P) -> Result<Issued, IssueError> {
        issue(self.account, publish, names, self.replaces.clone()).await
    }
}
