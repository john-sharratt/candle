//! The frames between peer gateways (§13.4): `PEER_HELLO`, and the ACME state
//! active-active gateways share — challenges and their acknowledgements,
//! certificates, the order ledger, the inventory exchanged on connect, an
//! order's start and end, and the shared account.

use super::types::BuildId;
use crate::wire::{Reader, WireError, Writer};

/// A gateway's first frame on a peer link, after `ADMIT`: who it is, and the
/// digests of the two secrets peers must share for a session cookie one
/// signed to verify on the other. The secrets themselves never cross.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PeerHello {
    pub node: u64,
    pub build: BuildId,
    /// SHA-256 of the session key's bytes; of nothing, with no sign-in.
    pub session_key: [u8; 32],
    /// SHA-256 of the Google OAuth client id; of nothing, with no sign-in.
    pub oauth_client: [u8; 32],
}

impl PeerHello {
    pub fn write(&self, w: &mut Writer) {
        w.u64(self.node);
        self.build.write(w);
        w.fixed(&self.session_key).fixed(&self.oauth_client);
    }

    pub fn read(r: &mut Reader) -> Result<Self, WireError> {
        Ok(Self {
            node: r.u64()?,
            build: BuildId::read(r)?,
            session_key: r.fixed()?,
            oauth_client: r.fixed()?,
        })
    }
}

/// An HTTP-01 answer the orderer is about to ask the CA to check.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AcmeChallenge {
    pub site: String,
    pub token: String,
    pub answer: String,
    /// Wall-clock milliseconds past which a peer drops it unasked.
    pub expires_ms: u64,
}

impl AcmeChallenge {
    pub fn write(&self, w: &mut Writer) {
        w.str(&self.site)
            .str(&self.token)
            .str(&self.answer)
            .u64(self.expires_ms);
    }

    pub fn read(r: &mut Reader) -> Result<Self, WireError> {
        Ok(Self {
            site: r.str()?,
            token: r.str()?,
            answer: r.str()?,
            expires_ms: r.u64()?,
        })
    }
}

/// A certificate a gateway was issued, with its key.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AcmeCert {
    pub site: String,
    pub chain_pem: String,
    pub key_pem: String,
    /// Unix seconds.
    pub not_after: u64,
}

impl AcmeCert {
    pub fn write(&self, w: &mut Writer) {
        w.str(&self.site)
            .str(&self.chain_pem)
            .str(&self.key_pem)
            .u64(self.not_after);
    }

    pub fn read(r: &mut Reader) -> Result<Self, WireError> {
        Ok(Self {
            site: r.str()?,
            chain_pem: r.str()?,
            key_pem: r.str()?,
            not_after: r.u64()?,
        })
    }
}

/// One site's certificate in an inventory.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CertLine {
    pub site: String,
    /// Unix seconds.
    pub not_after: u64,
    /// SHA-256 of the chain's PEM.
    pub digest: [u8; 32],
}

impl CertLine {
    pub fn write(&self, w: &mut Writer) {
        w.str(&self.site).u64(self.not_after).fixed(&self.digest);
    }

    pub fn read(r: &mut Reader) -> Result<Self, WireError> {
        Ok(Self {
            site: r.str()?,
            not_after: r.u64()?,
            digest: r.fixed()?,
        })
    }
}
