//! Paxos between members (§5.9): `PREPARE`, `PROMISE`, `NACK`, `PROPOSE`,
//! `ACCEPTED` and `RELEASE`. The gateway carries these and decodes nothing it
//! does not route by.

use super::types::Ballot;
use crate::wire::{Reader, WireError, Writer};

/// Phase 1: a proposer asks every acceptor to promise `ballot` for `key`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Prepare {
    pub key: Vec<u8>,
    pub ballot: Ballot,
}

impl Prepare {
    pub fn write(&self, w: &mut Writer) {
        w.blob(&self.key);
        self.ballot.write(w);
    }

    pub fn read(r: &mut Reader) -> Result<Self, WireError> {
        Ok(Self {
            key: r.blob_vec()?,
            ballot: Ballot::read(r)?,
        })
    }
}

/// An acceptance an acceptor holds: the ballot it accepted at and the value.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AcceptedValue {
    pub ballot: Ballot,
    pub value: Vec<u8>,
}

impl AcceptedValue {
    pub fn write(&self, w: &mut Writer) {
        self.ballot.write(w);
        w.blob(&self.value);
    }

    pub fn read(r: &mut Reader) -> Result<Self, WireError> {
        Ok(Self {
            ballot: Ballot::read(r)?,
            value: r.blob_vec()?,
        })
    }
}

/// An acceptor's promise, with what it has already accepted.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Promise {
    pub key: Vec<u8>,
    pub ballot: Ballot,
    pub accepted: Option<AcceptedValue>,
    pub reply_to: u64,
}

impl Promise {
    pub fn write(&self, w: &mut Writer) {
        w.blob(&self.key);
        self.ballot.write(w);
        w.option(self.accepted.as_ref(), |w, a| a.write(w))
            .u64(self.reply_to);
    }

    pub fn read(r: &mut Reader) -> Result<Self, WireError> {
        Ok(Self {
            key: r.blob_vec()?,
            ballot: Ballot::read(r)?,
            accepted: r.option(AcceptedValue::read)?,
            reply_to: r.u64()?,
        })
    }
}

/// An acceptor's refusal, naming the ballot it has promised.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Nack {
    pub key: Vec<u8>,
    pub promised: Ballot,
    pub reply_to: u64,
}

impl Nack {
    pub fn write(&self, w: &mut Writer) {
        w.blob(&self.key);
        self.promised.write(w);
        w.u64(self.reply_to);
    }

    pub fn read(r: &mut Reader) -> Result<Self, WireError> {
        Ok(Self {
            key: r.blob_vec()?,
            promised: Ballot::read(r)?,
            reply_to: r.u64()?,
        })
    }
}

/// Phase 2: a proposer asks every acceptor to accept `value` at `ballot`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Propose {
    pub key: Vec<u8>,
    pub ballot: Ballot,
    pub value: Vec<u8>,
}

impl Propose {
    pub fn write(&self, w: &mut Writer) {
        w.blob(&self.key);
        self.ballot.write(w);
        w.blob(&self.value);
    }

    pub fn read(r: &mut Reader) -> Result<Self, WireError> {
        Ok(Self {
            key: r.blob_vec()?,
            ballot: Ballot::read(r)?,
            value: r.blob_vec()?,
        })
    }
}

/// An acceptor accepted `ballot` for `key`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Accepted {
    pub key: Vec<u8>,
    pub ballot: Ballot,
    pub reply_to: u64,
}

impl Accepted {
    pub fn write(&self, w: &mut Writer) {
        w.blob(&self.key);
        self.ballot.write(w);
        w.u64(self.reply_to);
    }

    pub fn read(r: &mut Reader) -> Result<Self, WireError> {
        Ok(Self {
            key: r.blob_vec()?,
            ballot: Ballot::read(r)?,
            reply_to: r.u64()?,
        })
    }
}

/// A lease holder hands the lease taken at `ballot` back; its own `oseq` was
/// `through` (§5.9, §9.6).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Release {
    pub key: Vec<u8>,
    pub ballot: Ballot,
    pub through: u64,
}

impl Release {
    pub fn write(&self, w: &mut Writer) {
        w.blob(&self.key);
        self.ballot.write(w);
        w.u64(self.through);
    }

    pub fn read(r: &mut Reader) -> Result<Self, WireError> {
        Ok(Self {
            key: r.blob_vec()?,
            ballot: Ballot::read(r)?,
            through: r.u64()?,
        })
    }
}
