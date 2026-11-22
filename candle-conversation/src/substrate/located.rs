//! A stream's record bytes held in RAM, or the place in the log that holds
//! them, and the reader that brings stored bytes back.
//!
//! A turn's wide-Q signature window and its projection events are written once
//! and read rarely — a gallery upload, a host scan, a replay — but a long-lived
//! store holds hundreds of thousands of them. Held for the life of the process
//! they were 16 GB of heap on WhiteWorld (2026-11-22), paged straight back out.
//! So a value recovered from the log stays in the log ([`Located::Stored`]),
//! and one this process wrote is held only until its append returns
//! ([`Located::Held`]).
//!
//! A stored value is read through the substrate's [`PayloadSource`], which every
//! version of the substrate shares. The first read goes lock-free to the
//! durable copy at the location the substrate names, and is taken only when the
//! record there is the one asked for. Anything else — a record not yet durable,
//! or one maintenance moved since — is read by identity through the log's
//! owner, which finds a record wherever it now is.

use std::fmt;
use std::sync::{Arc, OnceLock, Weak};

use crate::persistence::identity::RecordIdentity;
use crate::persistence::manifest::RecordLoc;
use crate::persistence::owner::{on_owner_thread, PersistenceOwner};
use crate::persistence::record::RecordHeader;
use crate::persistence::record_reader::RecordReader;
use crate::persistence::{PersistenceError, Result};

/// A stream's record bytes, or where the log keeps them.
#[derive(Debug, Clone)]
pub enum Located {
    /// The bytes, as this process wrote them, until the append lands.
    Held(Arc<[u8]>),
    /// The record holding them, read when something uses it.
    Stored(RecordLoc),
}

impl Located {
    /// The held bytes, for a value this process still holds.
    pub fn held(&self) -> Option<&Arc<[u8]>> {
        match self {
            Self::Held(b) => Some(b),
            Self::Stored(_) => None,
        }
    }

    /// The record's location, for a value left in the log.
    pub fn stored(&self) -> Option<RecordLoc> {
        match self {
            Self::Stored(loc) => Some(*loc),
            Self::Held(_) => None,
        }
    }

    /// The value's length in bytes, held or stored.
    pub fn payload_len(&self) -> u64 {
        match self {
            Self::Held(b) => b.len() as u64,
            Self::Stored(loc) => loc.payload_len,
        }
    }
}

/// Reads a stored [`Located`] value back. One per store, shared by every
/// version of its substrate.
#[derive(Default)]
pub struct PayloadSource {
    /// Reads durable records without the log's owner — the store's
    /// `SealedReader`, set when the store opens, or a tool's own reader.
    reader: OnceLock<Arc<dyn RecordReader>>,
    /// The log's owner, set when it spawns: reads what `reader` cannot, by
    /// identity. Weak, because the owner holds the substrate this is part of.
    owner: OnceLock<Weak<PersistenceOwner>>,
}

impl fmt::Debug for PayloadSource {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("PayloadSource")
            .field("reader", &self.reader.get().is_some())
            .field("owner", &self.owner.get().is_some())
            .finish()
    }
}

impl PayloadSource {
    /// Read durable records through `reader`. The first reader set is kept.
    pub fn set_reader(&self, reader: Arc<dyn RecordReader>) {
        let _ = self.reader.set(reader);
    }

    /// Read what the reader cannot through `owner`. The first owner set is
    /// kept.
    pub fn set_owner(&self, owner: &Arc<PersistenceOwner>) {
        let _ = self.owner.set(Arc::downgrade(owner));
    }

    /// The payload of record `id`, stored at `loc`.
    ///
    /// The fallback waits on the log's owner, so it is refused on the owner's
    /// own thread, where a record the reader cannot reach is an error; and it
    /// must not run inside a substrate edit, which does no I/O.
    pub fn read(&self, id: RecordIdentity, loc: RecordLoc) -> Result<Vec<u8>> {
        if let Some(payload) = self.read_direct(id, &loc, None) {
            return Ok(payload);
        }
        self.read_by_owner(id, loc)
    }

    /// The first `n` bytes of record `id`'s payload, stored at `loc` — one
    /// sector where the record is durable where the substrate names it.
    pub fn read_head(&self, id: RecordIdentity, loc: RecordLoc, n: usize) -> Result<Vec<u8>> {
        if let Some(head) = self.read_direct(id, &loc, Some(n)) {
            return Ok(head);
        }
        let mut payload = self.read_by_owner(id, loc)?;
        payload.truncate(n);
        Ok(payload)
    }

    /// The read through `reader`, taken only when the record at `loc` is `id`.
    fn read_direct(
        &self,
        id: RecordIdentity,
        loc: &RecordLoc,
        head: Option<usize>,
    ) -> Option<Vec<u8>> {
        let (header, payload) = self.reader.get()?.read(loc, head)?.ok()?;
        names(&header, id).then_some(payload)
    }

    /// The read through the log's owner: by identity, wherever the record is
    /// now, or at `loc` for a record that carries none.
    fn read_by_owner(&self, id: RecordIdentity, loc: RecordLoc) -> Result<Vec<u8>> {
        if on_owner_thread() {
            return Err(PersistenceError::Corrupt(format!(
                "record {:x}/{} at {:?}+{} was read on the log owner's own thread, where a \
                 call to the owner would wait for itself",
                id.origin, id.oseq, loc.segment, loc.offset
            )));
        }
        let Some(owner) = self.owner.get().and_then(Weak::upgrade) else {
            return Err(PersistenceError::Corrupt(format!(
                "record {:x}/{} at {:?}+{} is not readable: no reader holds it",
                id.origin, id.oseq, loc.segment, loc.offset
            )));
        };
        owner.call(move |p| {
            if id.is_none() {
                return p.read_record_payload(&loc);
            }
            match p.read_by_identity(id.origin, id.oseq)? {
                Some(record) => Ok(record.payload),
                None => Err(PersistenceError::Corrupt(format!(
                    "record {:x}/{} is not held by this member",
                    id.origin, id.oseq
                ))),
            }
        })
    }
}

/// Whether `header` is record `id`'s. A record without an identity is taken at
/// its location: nothing else can name it.
fn names(header: &RecordHeader, id: RecordIdentity) -> bool {
    id.is_none() || header.identity() == id
}
