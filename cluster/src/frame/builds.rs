//! `BUILD_OFFER`, `BUILD_HAVE`, `BUILD_FETCH`, `BUILD_CHUNK` and `VERDICT`
//! (§3.8, §3.9).

use super::types::{Artifact, BuildId};
use crate::wire::{Reader, WireError, Writer};

/// The most bytes one `BUILD_CHUNK` carries.
pub const MAX_BUILD_CHUNK: usize = 1 << 20;

/// A build and the artifacts its sender holds verified — `BUILD_OFFER` from a
/// candidate, `BUILD_HAVE` from a member offering what it fetched or kept.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BuildArtifacts {
    pub build: BuildId,
    pub artifacts: Vec<Artifact>,
}

impl BuildArtifacts {
    pub fn write(&self, w: &mut Writer) {
        self.build.write(w);
        w.list(&self.artifacts, |w, a| a.write(w));
    }

    pub fn read(r: &mut Reader) -> Result<Self, WireError> {
        Ok(Self {
            build: BuildId::read(r)?,
            artifacts: r.list(Artifact::read)?,
        })
    }
}

/// Ask the member the frame is relayed to for an artifact, from `offset`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BuildFetch {
    /// The artifact's SHA-256.
    pub artifact: [u8; 32],
    pub offset: u64,
    pub from_node: u64,
    pub reply_to: u64,
}

impl BuildFetch {
    pub fn write(&self, w: &mut Writer) {
        w.fixed(&self.artifact)
            .u64(self.offset)
            .u64(self.from_node)
            .u64(self.reply_to);
    }

    pub fn read(r: &mut Reader) -> Result<Self, WireError> {
        Ok(Self {
            artifact: r.fixed()?,
            offset: r.u64()?,
            from_node: r.u64()?,
            reply_to: r.u64()?,
        })
    }
}

/// Up to [`MAX_BUILD_CHUNK`] bytes of an artifact at `offset`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BuildChunk {
    pub artifact: [u8; 32],
    pub offset: u64,
    pub reply_to: u64,
    pub bytes: Vec<u8>,
}

impl BuildChunk {
    pub fn write(&self, w: &mut Writer) {
        w.fixed(&self.artifact)
            .u64(self.offset)
            .u64(self.reply_to)
            .blob(&self.bytes);
    }

    pub fn read(r: &mut Reader) -> Result<Self, WireError> {
        let artifact = r.fixed()?;
        let offset = r.u64()?;
        let reply_to = r.u64()?;
        let bytes = r.blob_vec()?;
        if bytes.len() > MAX_BUILD_CHUNK {
            return Err(WireError::TooLong {
                what: r.what(),
                len: bytes.len(),
                max: MAX_BUILD_CHUNK,
            });
        }
        Ok(Self {
            artifact,
            offset,
            reply_to,
            bytes,
        })
    }
}

/// A gateway's certification verdict on a candidate (§3.8).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VerdictResult {
    Pass = 0,
    Fail = 1,
    NoOpinion = 2,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Verdict {
    pub build: BuildId,
    pub result: VerdictResult,
    /// The figures it rests on, as §3.8 encodes them. Opaque here.
    pub figures: Vec<u8>,
}

impl Verdict {
    pub fn write(&self, w: &mut Writer) {
        self.build.write(w);
        w.u8(self.result as u8).blob(&self.figures);
    }

    pub fn read(r: &mut Reader) -> Result<Self, WireError> {
        let build = BuildId::read(r)?;
        let result = match r.u8()? {
            0 => VerdictResult::Pass,
            1 => VerdictResult::Fail,
            2 => VerdictResult::NoOpinion,
            v => {
                return Err(WireError::Invalid {
                    what: r.what(),
                    value: v as u64,
                })
            }
        };
        Ok(Self {
            build,
            result,
            figures: r.blob_vec()?,
        })
    }
}
