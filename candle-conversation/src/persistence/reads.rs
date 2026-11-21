//! Reads records back by identity or location, from wherever maintenance has moved them.
//!
//! A record this node wrote cut off comes back with the generation a rejoin fixed for it. A read of a record this member does not hold yields nothing rather than an error. Reads of sealed segments go through a reader that needs no lock on the persistence handle, so a read that finds its record sealed never waits on it. Lookups of streams, sections and tokens resolve across the active log and then every inherited log.

use super::*;

impl SubstratePersistence {
    /// The bytes of record `(origin, oseq)` — header line, payload and
    /// padding — from wherever maintenance has moved it, or `None` when this
    /// member does not hold it. What a member sends for an event it owes, or
    /// to answer a fetch (§6.2, §6.3, §7.2). A record this node wrote cut off
    /// comes back with the generation its rejoin fixed (§6.7).
    pub fn read_event(&mut self, origin: u64, oseq: u64) -> Result<Option<Vec<u8>>> {
        let Some(at) = self.locations.get(origin, oseq) else {
            return Ok(None);
        };
        self.flush_staged()?;
        let mut probe = vec![0u8; HEADER_PROBE_SIZE];
        self.segments.read_into(at.segment, at.offset, &mut probe)?;
        let (header, header_bytes) = decode_header(&probe)?;
        let id = header.identity();
        if (id.origin, id.oseq) != (origin, oseq) {
            return Err(PersistenceError::Corrupt(format!(
                "the location index names {:x}/{oseq} at {:?}+{} but the record there is \
                 {:x}/{}",
                origin, at.segment, at.offset, id.origin, id.oseq
            )));
        }
        let mut bytes = vec![0u8; padded_record_len(header_bytes - 1, header.payload_len)];
        self.segments.read_into(at.segment, at.offset, &mut bytes)?;
        let fixed = self.rebases.fix(id);
        if fixed == id {
            return Ok(Some(bytes));
        }
        let (header, payload, _) = decode_record(&bytes)?;
        Ok(Some(encode_record(&header.with_identity(fixed), payload)))
    }

    /// [`Self::read_event`], decoded and checked against its CRC.
    pub fn read_by_identity(&mut self, origin: u64, oseq: u64) -> Result<Option<Record>> {
        let Some(bytes) = self.read_event(origin, oseq)? else {
            return Ok(None);
        };
        let (header, payload, _) = decode_record(&bytes)?;
        verify_record_crc(&header, payload)?;
        Ok(Some(Record {
            header,
            payload: payload.to_vec(),
        }))
    }

    /// Read a stream's latest `Tokens` record payload — from the active log,
    /// else any inherited log. `None` if the stream has no `Tokens` record.
    pub fn read_tokens(
        &mut self,
        substrate: &Substrate,
        stream_id: StreamId,
    ) -> Result<Option<Vec<u8>>> {
        if let Some(loc) = substrate.stream_of(stream_id).and_then(|s| s.tokens) {
            let record = self
                .segments
                .read_record_at(loc.segment, loc.offset, loc.record_size)?;
            return Ok(Some(record.payload));
        }
        for inherited in &self.inherited {
            if let Some(loc) = inherited
                .substrate()
                .stream_of(stream_id)
                .and_then(|s| s.tokens)
            {
                let record = inherited.read_record(loc.offset, loc.record_size)?;
                return Ok(Some(record.payload));
            }
        }
        Ok(None)
    }

    /// A reader for this store's sealed segments that needs no lock on it —
    /// taken once, beside the handle, so a read that finds its record sealed
    /// never waits on this persistence's mutex. See [`sealed_reader`].
    pub fn sealed_reader(&self) -> SealedReader {
        self.segments.sealed_reader()
    }

    /// Where `(origin, oseq)` lies, from memory — for a read made without
    /// this persistence's lock (`SealedReader::read_raw`). `None` when it is
    /// not held here.
    pub fn locate(&self, origin: u64, oseq: u64) -> Option<(SegmentId, u64)> {
        self.locations
            .get(origin, oseq)
            .map(|at| (at.segment, at.offset))
    }

    /// Whether a record read off disk is already as it travels: no rebase
    /// can apply to a detached identity (`read_event`'s fix-up).
    pub fn no_rebases(&self) -> bool {
        self.rebases.is_empty()
    }

    /// Every live recurrent tail — a conversation's snapshot or a prompt
    /// branch's checkpoint — by its stream id, with where its record is.
    pub fn recurrent_tails(&self) -> Vec<(StreamId, RecordLoc)> {
        self.snapshot_locs
            .iter()
            .map(|(stream, loc)| (StreamId(*stream), *loc))
            .collect()
    }

    /// Read one record's payload back by location — the snapshot-restore
    /// read (the caller got `loc` from the substrate's snapshot index).
    pub fn read_record_payload(&mut self, loc: &RecordLoc) -> Result<Vec<u8>> {
        let record = self
            .segments
            .read_record_at(loc.segment, loc.offset, loc.record_size)?;
        Ok(record.payload)
    }

    /// Resolve a content-addressed prompt section across the active log and
    /// every inherited log (§13.5). `Some` is a prefix-cache hit.
    pub fn lookup_section(&self, substrate: &Substrate, addr: ContentAddress) -> Option<StreamRef> {
        let id = content_hash::section_stream_id(addr);
        if self.has_stream(substrate, id) {
            Some(StreamRef {
                stream_id: id,
                kind: StreamKind::PromptSection,
            })
        } else {
            None
        }
    }

    /// Whether `stream_id` is present in the active substrate or any
    /// inherited log.  The active source is read from `substrate.streams`
    /// (the in-RAM index built by the reload walk); inherited logs still
    /// carry their own manifest.
    pub fn has_stream(&self, substrate: &Substrate, stream_id: StreamId) -> bool {
        substrate.has_stream(stream_id) || self.inherited.iter().any(|i| i.has_stream(stream_id))
    }
}
