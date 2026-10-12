//! `HELLO`, `CHALLENGE`, `PROVE` and `ADMIT` (§3.3).

use super::types::{BuildId, ClaimInfo, MemberInfo, ModelId, RepoAd};
use crate::wire::{Reader, WireError, Writer};

/// The member's opening frame.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Hello {
    pub proto: u16,
    pub app: String,
    pub node: u64,
    pub host: String,
    pub build: BuildId,
    /// The group rejected this build, by the certification verdicts in the
    /// member's own substrate: it never sets a floor and is never a candidate,
    /// even at a gateway that has not yet heard the verdict (§3.11).
    pub rejected: bool,
    /// The model its substrate is bound to; an agent sends none.
    pub model: Option<ModelId>,
    pub power: u32,
    /// The repositories it holds: a zend member's manifest, an agent's
    /// checkouts (§10.1, §11.2).
    pub repos: Vec<RepoAd>,
    pub now_ms: u64,
    pub nonce: [u8; 32],
}

impl Hello {
    pub fn write(&self, w: &mut Writer) {
        w.u16(self.proto)
            .str(&self.app)
            .u64(self.node)
            .str(&self.host);
        self.build.write(w);
        w.bool(self.rejected)
            .option(self.model.as_ref(), |w, m| m.write(w))
            .u32(self.power)
            .list(&self.repos, |w, r| r.write(w))
            .u64(self.now_ms)
            .fixed(&self.nonce);
    }

    pub fn read(r: &mut Reader) -> Result<Self, WireError> {
        Ok(Self {
            proto: r.u16()?,
            app: r.str()?,
            node: r.u64()?,
            host: r.str()?,
            build: BuildId::read(r)?,
            rejected: r.bool()?,
            model: r.option(ModelId::read)?,
            power: r.u32()?,
            repos: r.list(RepoAd::read)?,
            now_ms: r.u64()?,
            nonce: r.fixed()?,
        })
    }
}

/// The gateway's nonce and its proof.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Challenge {
    pub nonce: [u8; 32],
    pub proof: [u8; 32],
}

impl Challenge {
    pub fn write(&self, w: &mut Writer) {
        w.fixed(&self.nonce).fixed(&self.proof);
    }

    pub fn read(r: &mut Reader) -> Result<Self, WireError> {
        Ok(Self {
            nonce: r.fixed()?,
            proof: r.fixed()?,
        })
    }
}

/// The member's proof.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Prove {
    pub proof: [u8; 32],
}

impl Prove {
    pub fn write(&self, w: &mut Writer) {
        w.fixed(&self.proof);
    }

    pub fn read(r: &mut Reader) -> Result<Self, WireError> {
        Ok(Self { proof: r.fixed()? })
    }
}

/// The gateway's admission: the session, what a new member needs to route and
/// to resume (§3.3, §5.4, §7.5).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Admit {
    pub session: u64,
    pub members: Vec<MemberInfo>,
    pub claims: Vec<ClaimInfo>,
    /// The app's build floor: the `count` and `hash` admission compared.
    pub floor_count: u32,
    pub floor_hash: String,
    /// The gateway holds no dispatch table for this member and wants `ROUTES`.
    pub routes_wanted: bool,
    /// The highest `applied_through` any other member acknowledged for this
    /// member's own origin (§7.5).
    pub own_applied_through: u64,
}

impl Admit {
    pub fn write(&self, w: &mut Writer) {
        w.u64(self.session)
            .list(&self.members, |w, m| m.write(w))
            .list(&self.claims, |w, c| c.write(w))
            .u32(self.floor_count)
            .str(&self.floor_hash)
            .bool(self.routes_wanted)
            .u64(self.own_applied_through);
    }

    pub fn read(r: &mut Reader) -> Result<Self, WireError> {
        Ok(Self {
            session: r.u64()?,
            members: r.list(MemberInfo::read)?,
            claims: r.list(ClaimInfo::read)?,
            floor_count: r.u32()?,
            floor_hash: r.str()?,
            routes_wanted: r.bool()?,
            own_applied_through: r.u64()?,
        })
    }
}
