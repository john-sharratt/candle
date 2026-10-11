//! `MEMBERS`, `CAPACITY`, `PING`/`PONG`, `ROUTES`, `CLAIM`/`UNCLAIM` and
//! `COORD` (§4.1, §5.3, §5.4, §12.1).

use super::types::{Epoch, MemberInfo, MemberState};
use crate::wire::{Reader, WireError, Writer};

/// Every admitted member of the app, sent on each change and every 5 s.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Members {
    pub epoch: u64,
    pub members: Vec<MemberInfo>,
}

impl Members {
    pub fn write(&self, w: &mut Writer) {
        w.u64(self.epoch).list(&self.members, |w, m| m.write(w));
    }

    pub fn read(r: &mut Reader) -> Result<Self, WireError> {
        Ok(Self {
            epoch: r.u64()?,
            members: r.list(MemberInfo::read)?,
        })
    }
}

/// What a member can take on now (§5.3), every 1 s or on change.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Capacity {
    pub state: MemberState,
    /// Sustained aggregate decode rate, tokens per second.
    pub power: u32,
    pub idle: bool,
    pub slots_free: u32,
    pub decodes: u32,
    pub prefill_backlog: u32,
    pub vram_free_mib: u32,
    pub cores: u16,
    /// Host load average × 1000.
    pub load_milli: u32,
}

impl Capacity {
    pub fn write(&self, w: &mut Writer) {
        self.state.write(w);
        w.u32(self.power)
            .bool(self.idle)
            .u32(self.slots_free)
            .u32(self.decodes)
            .u32(self.prefill_backlog)
            .u32(self.vram_free_mib)
            .u16(self.cores)
            .u32(self.load_milli);
    }

    pub fn read(r: &mut Reader) -> Result<Self, WireError> {
        Ok(Self {
            state: MemberState::read(r)?,
            power: r.u32()?,
            idle: r.bool()?,
            slots_free: r.u32()?,
            decodes: r.u32()?,
            prefill_backlog: r.u32()?,
            vram_free_mib: r.u32()?,
            cores: r.u16()?,
            load_milli: r.u32()?,
        })
    }
}

/// The member's dispatch table (§5.2), as that section encodes it. Opaque
/// here.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Routes {
    pub table: Vec<u8>,
}

/// This member serves `key` from `epoch` on.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Claim {
    pub key: Vec<u8>,
    pub epoch: Epoch,
}

impl Claim {
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

/// The claim taken at `epoch` is let go; its holder's own `oseq` was
/// `through` when it let go (§5.4).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Unclaim {
    pub key: Vec<u8>,
    pub epoch: Epoch,
    pub through: u64,
}

impl Unclaim {
    pub fn write(&self, w: &mut Writer) {
        w.blob(&self.key);
        self.epoch.write(w);
        w.u64(self.through);
    }

    pub fn read(r: &mut Reader) -> Result<Self, WireError> {
        Ok(Self {
            key: r.blob_vec()?,
            epoch: Epoch::read(r)?,
            through: r.u64()?,
        })
    }
}

/// The member's coordination state for the status page (§12.1), as that
/// section encodes it. Opaque here.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Coord {
    pub state: Vec<u8>,
}
