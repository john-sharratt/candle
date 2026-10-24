//! Substrate frames (§6.3, §7): `EVENT`, `RECEIPT`, `ACK`, `CATCHUP`,
//! `HEADERS`, `FETCH`, `DONE` and `HAVE`. A record inside `EVENT` is the
//! record's own encoded header line and payload; this crate never decodes it.

use applied_set::AppliedSet;

use super::packing::{read_applied, write_applied};
use crate::wire::{Reader, WireError, Writer};

/// The most bytes one `EVENT` fragment carries (§4.2).
pub const MAX_FRAGMENT: usize = 1 << 20;

/// One fragment of one record. A record of at most [`MAX_FRAGMENT`] bytes is
/// one fragment at offset zero.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Event {
    pub origin: u64,
    pub oseq: u64,
    /// The whole record's length.
    pub total: u32,
    /// Where this fragment starts in it.
    pub offset: u32,
    /// The requester a fetched record answers (`flags & ADDRESSED`).
    pub reply_to: Option<u64>,
    pub bytes: Vec<u8>,
}

impl Event {
    /// Written after the frame header; the `ADDRESSED` flag says whether
    /// `reply_to` is present.
    pub fn write(&self, w: &mut Writer) {
        w.u64(self.origin)
            .u64(self.oseq)
            .u32(self.total)
            .u32(self.offset);
        if let Some(reply_to) = self.reply_to {
            w.u64(reply_to);
        }
        w.fixed(&self.bytes);
    }

    pub fn read(r: &mut Reader, addressed: bool) -> Result<Self, WireError> {
        let what = r.what();
        let origin = r.u64()?;
        let oseq = r.u64()?;
        let total = r.u32()?;
        let offset = r.u32()?;
        let reply_to = if addressed { Some(r.u64()?) } else { None };
        let bytes = r.rest().to_vec();
        if origin == 0 || oseq == 0 {
            return Err(WireError::Invalid { what, value: 0 });
        }
        if bytes.is_empty() || bytes.len() > MAX_FRAGMENT {
            return Err(WireError::TooLong {
                what,
                len: bytes.len(),
                max: MAX_FRAGMENT,
            });
        }
        if offset as u64 + bytes.len() as u64 > total as u64 {
            return Err(WireError::Invalid {
                what,
                value: offset as u64 + bytes.len() as u64,
            });
        }
        Ok(Self {
            origin,
            oseq,
            total,
            offset,
            reply_to,
            bytes,
        })
    }

    /// Whether this fragment ends the record.
    pub fn is_last(&self) -> bool {
        self.offset as usize + self.bytes.len() == self.total as usize
    }
}

/// Gateway → member: every frame of this session through `through` has been
/// taken in and relayed (§4.3, §6.3).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Receipt {
    pub through: u64,
}

/// What this member has applied of each origin: `applied_through` per origin.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Ack {
    pub applied: Vec<(u64, u64)>,
}

impl Ack {
    pub fn write(&self, w: &mut Writer) {
        w.list(&self.applied, |w, (origin, through)| {
            w.u64(*origin).u64(*through);
        });
    }

    pub fn read(r: &mut Reader) -> Result<Self, WireError> {
        Ok(Self {
            applied: r.list(|r| Ok((r.u64()?, r.u64()?)))?,
        })
    }
}

/// A member asking what exists after what it holds (§7.2).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Catchup {
    pub request: u64,
    pub reply_to: u64,
    /// The member to ask; absent for the gateway to pick one round robin.
    pub to: Option<u64>,
    pub since: AppliedSet,
}

impl Catchup {
    pub fn write(&self, w: &mut Writer) {
        w.u64(self.request)
            .u64(self.reply_to)
            .option(self.to.as_ref(), |w, to| {
                w.u64(*to);
            });
        write_applied(w, &self.since);
    }

    pub fn read(r: &mut Reader) -> Result<Self, WireError> {
        Ok(Self {
            request: r.u64()?,
            reply_to: r.u64()?,
            to: r.option(|r| r.u64())?,
            since: read_applied(r)?,
        })
    }
}

/// One record a `HEADERS` lists: about forty bytes, never the payload.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RecordHeader {
    pub oseq: u64,
    /// The record's type tag.
    pub record_type: u8,
    pub stream_id: u64,
    pub chunk_index: u64,
    pub len: u32,
    pub digest: [u8; 16],
}

impl RecordHeader {
    pub fn write(&self, w: &mut Writer) {
        w.u64(self.oseq)
            .u8(self.record_type)
            .u64(self.stream_id)
            .u64(self.chunk_index)
            .u32(self.len)
            .fixed(&self.digest);
    }

    pub fn read(r: &mut Reader) -> Result<Self, WireError> {
        Ok(Self {
            oseq: r.u64()?,
            record_type: r.u8()?,
            stream_id: r.u64()?,
            chunk_index: r.u64()?,
            len: r.u32()?,
            digest: r.fixed()?,
        })
    }
}

/// An `oseq` the answerer knows no longer exists, with its deleted-set entry
/// when it was an expired tombstone (§6.8).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Gone {
    pub oseq: u64,
    pub deleted_entry: Option<Vec<u8>>,
}

impl Gone {
    pub fn write(&self, w: &mut Writer) {
        w.u64(self.oseq)
            .option(self.deleted_entry.as_ref(), |w, e| {
                w.blob(e);
            });
    }

    pub fn read(r: &mut Reader) -> Result<Self, WireError> {
        Ok(Self {
            oseq: r.u64()?,
            deleted_entry: r.option(|r| r.blob_vec())?,
        })
    }
}

/// What the answerer holds of `origin`'s range `from..=to`, a range it covers
/// completely (§7.2). `answerer` names it, the member a `FETCH` for these
/// records goes to.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Headers {
    pub request: u64,
    pub reply_to: u64,
    pub answerer: u64,
    pub origin: u64,
    pub from: u64,
    pub to: u64,
    pub gone: Vec<Gone>,
    pub records: Vec<RecordHeader>,
}

impl Headers {
    pub fn write(&self, w: &mut Writer) {
        w.u64(self.request)
            .u64(self.reply_to)
            .u64(self.answerer)
            .u64(self.origin)
            .u64(self.from)
            .u64(self.to)
            .list(&self.gone, |w, g| g.write(w))
            .list(&self.records, |w, h| h.write(w));
    }

    pub fn read(r: &mut Reader) -> Result<Self, WireError> {
        let h = Self {
            request: r.u64()?,
            reply_to: r.u64()?,
            answerer: r.u64()?,
            origin: r.u64()?,
            from: r.u64()?,
            to: r.u64()?,
            gone: r.list(Gone::read)?,
            records: r.list(RecordHeader::read)?,
        };
        let outside = |oseq: u64| oseq < h.from || oseq > h.to;
        if h.from == 0
            || h.from > h.to
            || h.gone.iter().any(|g| outside(g.oseq))
            || h.records.iter().any(|rec| outside(rec.oseq))
        {
            return Err(WireError::Invalid {
                what: r.what(),
                value: h.from,
            });
        }
        Ok(h)
    }
}

/// Ask one member for records themselves.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Fetch {
    pub request: u64,
    pub reply_to: u64,
    pub from_node: u64,
    pub origin: u64,
    pub oseqs: Vec<u64>,
}

impl Fetch {
    pub fn write(&self, w: &mut Writer) {
        w.u64(self.request)
            .u64(self.reply_to)
            .u64(self.from_node)
            .u64(self.origin)
            .list(&self.oseqs, |w, o| {
                w.u64(*o);
            });
    }

    pub fn read(r: &mut Reader) -> Result<Self, WireError> {
        Ok(Self {
            request: r.u64()?,
            reply_to: r.u64()?,
            from_node: r.u64()?,
            origin: r.u64()?,
            oseqs: r.list(|r| r.u64())?,
        })
    }
}

/// The last `HEADERS` of an answer, or the last record of a `FETCH`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Done {
    pub request: u64,
    pub reply_to: u64,
}

impl Done {
    pub fn write(&self, w: &mut Writer) {
        w.u64(self.request).u64(self.reply_to);
    }

    pub fn read(r: &mut Reader) -> Result<Self, WireError> {
        Ok(Self {
            request: r.u64()?,
            reply_to: r.u64()?,
        })
    }
}

/// Gateway → member: catch-up or fetch `request` will get no answer (§7.2) —
/// no member can answer it yet, or the member answering it is gone. The
/// requester learns so now, not from 30 s of silence, which a reply still
/// arriving over a slow link also gives: silence cannot stand for loss. A
/// frame dropped at the catch-up lane's cap needs no notice — the `DONE`
/// that ends every answer is never dropped, and a fetch that ends short asks
/// again.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Lost {
    pub request: u64,
    pub reply_to: u64,
}

impl Lost {
    pub fn write(&self, w: &mut Writer) {
        w.u64(self.request).u64(self.reply_to);
    }

    pub fn read(r: &mut Reader) -> Result<Self, WireError> {
        Ok(Self {
            request: r.u64()?,
            reply_to: r.u64()?,
        })
    }
}

/// A restarted member's offer of everything it holds (§7.5).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Have {
    pub request: u64,
    pub reply_to: u64,
    pub applied: AppliedSet,
}

impl Have {
    pub fn write(&self, w: &mut Writer) {
        w.u64(self.request).u64(self.reply_to);
        write_applied(w, &self.applied);
    }

    pub fn read(r: &mut Reader) -> Result<Self, WireError> {
        Ok(Self {
            request: r.u64()?,
            reply_to: r.u64()?,
            applied: read_applied(r)?,
        })
    }
}
