//! Field types more than one frame carries.

use crate::wire::{Reader, WireError, Writer};

/// A build's version (§3.5): `count` of `origin/main` commits HEAD contains,
/// `hash` of that merge base, and `local`, a digest of anything beyond it —
/// empty for a build exactly at the merge base.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct BuildId {
    pub count: u32,
    pub hash: String,
    pub local: String,
}

impl BuildId {
    pub fn write(&self, w: &mut Writer) {
        w.u32(self.count).str(&self.hash).str(&self.local);
    }

    pub fn read(r: &mut Reader) -> Result<Self, WireError> {
        Ok(Self {
            count: r.u32()?,
            hash: r.str()?,
            local: r.str()?,
        })
    }

    /// A build with commits or edits beyond `origin/main` (§3.10).
    pub fn is_local(&self) -> bool {
        !self.local.is_empty()
    }
}

/// The model a member's substrate is bound to (§3.6): the fields
/// `ModelKey::same_model` compares, and its description for a person.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ModelId {
    pub arch: String,
    pub weights: [u8; 32],
    pub tokenizer: [u8; 32],
    pub rope: String,
    pub describe: String,
}

impl ModelId {
    pub fn write(&self, w: &mut Writer) {
        w.str(&self.arch)
            .fixed(&self.weights)
            .fixed(&self.tokenizer)
            .str(&self.rope)
            .str(&self.describe);
    }

    pub fn read(r: &mut Reader) -> Result<Self, WireError> {
        Ok(Self {
            arch: r.str()?,
            weights: r.fixed()?,
            tokenizer: r.fixed()?,
            rope: r.str()?,
            describe: r.str()?,
        })
    }

    /// The same model: everything but the description agrees.
    pub fn same_model(&self, other: &ModelId) -> bool {
        self.arch == other.arch
            && self.weights == other.weights
            && self.tokenizer == other.tokenizer
            && self.rope == other.rope
    }
}

/// A member's state as `MEMBERS` and `CAPACITY` report it.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
pub enum MemberState {
    /// Admitted, still loading or catching up: routed nothing (§3.8, §7.2).
    #[default]
    Loading = 0,
    Ready = 1,
    /// Below the floor, taking the floor's build (§3.5).
    Upgrading = 2,
    /// Its disk is full (§7.7).
    Fault = 3,
    /// Handing its work back before it stops (§12.4).
    Draining = 4,
}

impl MemberState {
    pub fn write(self, w: &mut Writer) {
        w.u8(self as u8);
    }

    pub fn read(r: &mut Reader) -> Result<Self, WireError> {
        Ok(match r.u8()? {
            0 => Self::Loading,
            1 => Self::Ready,
            2 => Self::Upgrading,
            3 => Self::Fault,
            4 => Self::Draining,
            v => {
                return Err(WireError::Invalid {
                    what: r.what(),
                    value: v as u64,
                })
            }
        })
    }
}

/// One member as the gateway lists it (§4.1 `MEMBERS`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MemberInfo {
    pub node: u64,
    pub host: String,
    pub app: String,
    pub build: BuildId,
    pub weight: u32,
    pub state: MemberState,
    /// The member's own highest `oseq` (§7.1).
    pub high_water: u64,
}

impl MemberInfo {
    pub fn write(&self, w: &mut Writer) {
        w.u64(self.node).str(&self.host).str(&self.app);
        self.build.write(w);
        w.u32(self.weight);
        self.state.write(w);
        w.u64(self.high_water);
    }

    pub fn read(r: &mut Reader) -> Result<Self, WireError> {
        Ok(Self {
            node: r.u64()?,
            host: r.str()?,
            app: r.str()?,
            build: BuildId::read(r)?,
            weight: r.u32()?,
            state: MemberState::read(r)?,
            high_water: r.u64()?,
        })
    }
}

/// A repository a member holds, as its `HELLO` advertises it (§10.1, §11.2):
/// its name in the workspace manifest, its origin's URL, and whether the
/// member can push to that origin.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RepoAd {
    pub name: String,
    pub origin: String,
    pub push: bool,
}

impl RepoAd {
    pub fn write(&self, w: &mut Writer) {
        w.str(&self.name).str(&self.origin).bool(self.push);
    }

    pub fn read(r: &mut Reader) -> Result<Self, WireError> {
        Ok(Self {
            name: r.str()?,
            origin: r.str()?,
            push: r.bool()?,
        })
    }
}

/// A claim's epoch `(ms, node)` (§5.4): ordered by `ms`, then `node`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Epoch {
    pub ms: u64,
    pub node: u64,
}

impl Epoch {
    pub fn write(&self, w: &mut Writer) {
        w.u64(self.ms).u64(self.node);
    }

    pub fn read(r: &mut Reader) -> Result<Self, WireError> {
        Ok(Self {
            ms: r.u64()?,
            node: r.u64()?,
        })
    }
}

/// A claim the gateway holds for routing: who serves `key` (§5.4).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ClaimInfo {
    pub key: Vec<u8>,
    pub epoch: Epoch,
}

impl ClaimInfo {
    pub fn write(&self, w: &mut Writer) {
        w.blob(&self.key);
        self.epoch.write(w);
    }

    pub fn read(r: &mut Reader) -> Result<Self, WireError> {
        Ok(Self {
            key: r.blob_vec()?,
            epoch: Epoch::read(r)?,
        })
    }
}

/// A Paxos ballot (§5.9): ordered by `round`, then the proposer's `node`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
pub struct Ballot {
    pub round: u64,
    pub node: u64,
}

impl Ballot {
    pub fn write(&self, w: &mut Writer) {
        w.u64(self.round).u64(self.node);
    }

    pub fn read(r: &mut Reader) -> Result<Self, WireError> {
        Ok(Self {
            round: r.u64()?,
            node: r.u64()?,
        })
    }
}

/// One downloadable piece of a build (§3.9).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Artifact {
    pub kind: String,
    pub platform: String,
    pub size: u64,
    pub sha256: [u8; 32],
}

impl Artifact {
    pub fn write(&self, w: &mut Writer) {
        w.str(&self.kind)
            .str(&self.platform)
            .u64(self.size)
            .fixed(&self.sha256);
    }

    pub fn read(r: &mut Reader) -> Result<Self, WireError> {
        Ok(Self {
            kind: r.str()?,
            platform: r.str()?,
            size: r.u64()?,
            sha256: r.fixed()?,
        })
    }
}
