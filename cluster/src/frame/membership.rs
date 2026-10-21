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

/// What a member can take on now (§5.3, §11.2), every 1 s or on change.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Capacity {
    pub state: MemberState,
    /// Sustained aggregate decode rate, tokens per second.
    pub power: u32,
    pub idle: bool,
    /// Requests it can start now: decode slots for a model member, job
    /// slots for an agent.
    pub slots_free: u32,
    pub decodes: u32,
    pub prefill_backlog: u32,
    pub vram_free_mib: u32,
    pub cores: u16,
    /// Host load average × 1000.
    pub load_milli: u32,
    /// The host's operating system, as `std::env::consts::OS` names it.
    pub os: String,
    pub mem_free_mib: u32,
    /// Jobs running on an agent.
    pub jobs: u32,
    /// The programs of the sandbox allow-list found on its `PATH`.
    pub toolchains: Vec<String>,
    /// An agent sharing its host with a model member (§11.5): its jobs run
    /// below normal priority, and it is chosen only when no dedicated agent
    /// can take a job.
    pub low_priority: bool,
    /// What one more turn would cost a model member now, from what its own
    /// scheduler has learned (`docs/cooperative_admission.md` §2.1): its
    /// prefill rate at the width it runs, tokens a second, and its decode
    /// step with one row more than it decodes, in microseconds. Both zero
    /// until it has learned from a prefill and a decode — and always for an
    /// agent.
    pub prefill_tok_s: u32,
    pub decode_step_next_us: u32,
    /// Its decode step with [`PLACEMENT_WIDE`] rows more than it decodes, in
    /// microseconds: how the step grows as a burst lands. A card that copies
    /// every routed expert grows steeply with width, and the one-more-row step
    /// alone priced a burst of 24 at the idle step. Zero when unlearned, and
    /// never below `decode_step_next_us` once learned — equal means flat.
    pub decode_step_wide_us: u32,
    /// Its prefill rate with [`PLACEMENT_WIDE`] rows more than it decodes,
    /// tokens a second: how prefill slows as a burst lands, since every placed
    /// turn decodes beside the prompts behind it. A burst of 24 charged at the
    /// idle rate went to a member advertising 1,771 tok/s that then drained
    /// 213–463. Zero when unlearned, and never above `prefill_tok_s` once
    /// learned — equal means flat.
    pub prefill_tok_s_wide: u32,
    /// The mean prompt, in tokens, of the turns its scheduler has begun, and the
    /// mean reply such a turn decodes before its next prompt: the sizes of the
    /// work actually arriving, which a placement prices a turn at in place of a
    /// fixed nominal. Zero until learned.
    pub prompt_tokens: u32,
    pub reply_tokens: u32,
    /// Tokens a decoding sequence commits a step on this card, in thousandths:
    /// above 1000 where it speculates. A reply of `reply_tokens` takes that many
    /// fewer steps, which the step time alone does not say. Zero until learned.
    pub tokens_per_step_milli: u32,
}

/// The rows beyond its decodes at which a member states its wide figures
/// (`Capacity::decode_step_wide_us`, `Capacity::prefill_tok_s_wide`).
pub const PLACEMENT_WIDE: u32 = 16;

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
            .u32(self.load_milli)
            .str(&self.os)
            .u32(self.mem_free_mib)
            .u32(self.jobs)
            .list(&self.toolchains, |w, t| {
                w.str(t);
            })
            .bool(self.low_priority)
            .u32(self.prefill_tok_s)
            .u32(self.decode_step_next_us)
            .u32(self.decode_step_wide_us)
            .u32(self.prefill_tok_s_wide)
            .u32(self.prompt_tokens)
            .u32(self.reply_tokens)
            .u32(self.tokens_per_step_milli);
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
            os: r.str()?,
            mem_free_mib: r.u32()?,
            jobs: r.u32()?,
            toolchains: r.list(|r| r.str())?,
            low_priority: r.bool()?,
            prefill_tok_s: r.u32()?,
            decode_step_next_us: r.u32()?,
            decode_step_wide_us: r.u32()?,
            prefill_tok_s_wide: r.u32()?,
            prompt_tokens: r.u32()?,
            reply_tokens: r.u32()?,
            tokens_per_step_milli: r.u32()?,
        })
    }

    /// Idle cores: `cores × (1 − load / cores)`, in thousandths (§11.3).
    pub fn idle_cores_milli(&self) -> i64 {
        i64::from(self.cores) * 1000 - i64::from(self.load_milli)
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

/// The claim taken at `epoch` is let go (§5.4). `through` is the `oseq` of the
/// key's own last state record in its holder's origin — the newest commit of
/// the conversation it names — so a member holding the key's records through
/// it may take the key's next turn, whatever else of the holder's it lacks.
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

/// Keys let go that this member now holds complete (`docs/cooperative_admission.md`
/// §2.2): every record of each one's conversation through its `UNCLAIM`'s
/// `through`, and every turn of it whole. Sent as each turns complete, and
/// whole again on every admission, until the key is next claimed.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Complete {
    pub keys: Vec<Vec<u8>>,
}

impl Complete {
    pub fn write(&self, w: &mut Writer) {
        w.list(&self.keys, |w, k| {
            w.blob(k);
        });
    }

    pub fn read(r: &mut Reader) -> Result<Self, WireError> {
        Ok(Self {
            keys: r.list(|r| r.blob_vec())?,
        })
    }
}

/// The member's coordination state for the status page (§12.1), as that
/// section encodes it. Opaque here.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Coord {
    pub state: Vec<u8>,
}
