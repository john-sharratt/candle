//! Appends a record to the active redo-log segment and keeps the in-RAM manifest, header index and accounting in step with it.
//!
//! A record written by this substrate is stamped with its identity: this node as origin, the next origin sequence number, the generation and a clock reading. A copy of an existing record keeps the identity it has, so it is recognised as the record it copies. A chunk's golden is checked against the bytes about to be written, and a mismatch is logged while the bytes are still written, since a rebuild regenerates the store. A read-only handle refuses appends and stages nothing. Data records are digested into the header-index chain, and index records are not, because their `prev` links describe them.

use super::*;

impl SubstratePersistence {
    /// Append one record to the active segment, updating the in-RAM manifest.
    /// Returns where it landed and the identity it carries — callers that
    /// index the record (cold-load, chunk persistence) need the segment so the
    /// read routes back to the right file, the padded on-disk size for the
    /// bytes-on-disk footprint, and the identity the substrate keeps beside
    /// the value.
    ///
    /// A read-only handle refuses with [`PersistenceError::ReadOnly`] and
    /// stages nothing.
    ///
    /// **The record is this substrate's own**: it is stamped with a fresh
    /// identity — this node as origin, the next `oseq`, the current generation
    /// and a clock reading (`docs/replicated_substrate.md` §6.2) — except a
    /// `HeaderIndex`, which no origin writes. A copy of an existing record goes
    /// through [`Self::append_carried`] instead and keeps the identity it has.
    pub fn append_record(
        &mut self,
        record_type: RecordType,
        format: u8,
        stream_id: u64,
        chunk_index: u64,
        token_count: u64,
        golden: u32,
        payload: &[u8],
    ) -> Result<Appended> {
        if self.is_read_only() {
            return Err(PersistenceError::ReadOnly);
        }
        let identity = if record_type == RecordType::HeaderIndex {
            RecordIdentity::NONE
        } else {
            self.minter.mint_for(record_type, stream_id)
        };
        // Write-boundary validation of this member's own seal: recompute the
        // chunk's golden over the host bytes about to be written and compare
        // it with the golden the seal computed on the GPU, before the DtoH
        // copy. A mismatch means the copy or host handling corrupted the KV
        // bytes — a real byte-movement fault. Log it (a rebuild regenerates
        // the substrate; this proves the pipeline) and still write what we
        // have. A record carried with its identity already decided — another
        // member's, checked by its digest on arrival, or a rewrite of bytes
        // this store already holds — has no copy of its own to prove, and the
        // golden is checked again on the device when the chunk is loaded.
        if record_type == RecordType::Chunk && golden != 0 {
            match crate::persistence::record::recompute_chunk_golden(payload) {
                Ok(recomputed) if recomputed != golden => tracing::error!(
                    target: "candle_conversation::persistence::golden",
                    stream_id,
                    chunk_index,
                    golden,
                    recomputed,
                    "chunk golden mismatch before write — KV bytes changed between GPU seal and disk (DtoH/host corruption)"
                ),
                Ok(_) => {}
                Err(e) => tracing::error!(
                    target: "candle_conversation::persistence::golden",
                    stream_id,
                    chunk_index,
                    "chunk payload undecodable before write: {e}"
                ),
            }
        }
        self.append_carried(
            record_type,
            format,
            stream_id,
            chunk_index,
            token_count,
            golden,
            payload,
            identity,
        )
    }

    /// [`Self::append_record`] for a record whose identity is already decided:
    /// a resident value re-encoded from RAM by a rewrite, a singleton carried
    /// forward, or a runtime write whose identity was minted when the value
    /// changed in memory ([`Self::minter`]). The record keeps `identity`, so a
    /// copy is recognised everywhere as the record it copies and ranks where
    /// that record did.
    pub fn append_carried(
        &mut self,
        record_type: RecordType,
        format: u8,
        stream_id: u64,
        chunk_index: u64,
        token_count: u64,
        golden: u32,
        payload: &[u8],
        identity: RecordIdentity,
    ) -> Result<Appended> {
        if self.is_read_only() {
            return Err(PersistenceError::ReadOnly);
        }
        let header = RecordHeader {
            record_type,
            format,
            payload_len: payload.len() as u64,
            // `Chunk` records store the GPU-computed golden (Fletcher-32 over the
            // arena bytes, before the DtoH copy) here; `encode_record` keeps it.
            // Every other type ignores `golden` — `encode_record` fills `crc`
            // with crc32 over the payload.
            crc: golden,
            stream_id,
            chunk_index,
            token_count,
            ..RecordHeader::NONE
        }
        .with_identity(identity);
        let (written, bytes) = encode_record_written(&header, payload);
        let (segment, offset) = self.segments.stage(&bytes);
        let size = bytes.len() as u64;
        self.account_appended(&written, payload, segment, offset, size)?;
        Ok(Appended {
            segment,
            offset,
            size,
            identity,
        })
    }

    /// Append a recurrent-state record — a [`RecordType::Snapshot`] or a
    /// [`RecordType::BranchCheckpoint`] — from its payload's parts, written
    /// straight to the active segment rather than staged. The bookkeeping is
    /// [`Self::append_record`]'s.
    ///
    /// Only these two types come through here: they are the multi-megabyte
    /// records, and none of the append hooks reads their payload, so the parts
    /// never need joining. Any other type is refused.
    pub(super) fn append_recurrent_parts(
        &mut self,
        record_type: RecordType,
        stream_id: u64,
        parts: &[&[u8]],
    ) -> Result<Appended> {
        if self.is_read_only() {
            return Err(PersistenceError::ReadOnly);
        }
        assert!(
            matches!(
                record_type,
                RecordType::Snapshot | RecordType::BranchCheckpoint
            ),
            "append_recurrent_parts writes recurrent-state records only, not {record_type:?}"
        );
        let header = RecordHeader {
            record_type,
            format: 0,
            payload_len: parts.iter().map(|p| p.len() as u64).sum(),
            crc: 0,
            stream_id,
            chunk_index: 0,
            token_count: 0,
            ..RecordHeader::NONE
        }
        .with_identity(self.minter.mint(0));
        let head = record::record_head(&header, parts);
        let appended = self.segments.append_parts(&head.line, parts, head.pad);
        let (segment, offset) = self.watch_space(appended)?;
        let size = (head.line.len() + header.payload_len as usize + head.pad) as u64;
        // The hooks key these types by header alone, so the payload they are
        // handed is never read.
        self.account_appended(&head.header, &[], segment, offset, size)?;
        Ok(Appended {
            segment,
            offset,
            size,
            identity: header.identity(),
        })
    }

    /// Everything an append records once its bytes are placed: the relocation
    /// watch, dead-byte accounting, the singleton and snapshot locations, the
    /// manifest, the header-index chain and segment rotation.
    pub(super) fn account_appended(
        &mut self,
        header: &RecordHeader,
        payload: &[u8],
        segment: SegmentId,
        offset: u64,
        size: u64,
    ) -> Result<()> {
        let header = *header;
        if let Some(watch) = self.relocation_watch.as_mut() {
            watch.record(&header, payload);
        }
        self.facts.note_held(
            1u32.checked_shl(header.record_type.tag() as u32)
                .unwrap_or(0),
        );
        self.track_identity(&header, segment, offset);
        self.accounting.record(&header, payload, size);
        self.track_metadata_loc(&header, segment, offset, size);
        self.track_resident_loc(&header, payload, segment, offset, size);
        self.marker_ids.record(
            header.record_type,
            header.stream_id,
            payload,
            header.identity(),
        );
        self.track_snapshot_loc(&header, segment, offset, size);
        self.retire_snapshot_on_distill(&header, payload);
        self.track_npc_loc(&header, segment, offset, size);
        // Header + location only. This used to build a `WalkEntry`, whose
        // `Record` owns its payload — so every appended record cloned its whole
        // payload for a call that reads nothing but the header, then dropped it.
        self.manifest
            .ingest_located(&header, segment, offset, size)?;
        // Digest every data record for the header-index chain. Index
        // records themselves are self-describing via their `prev` links
        // and are never digested.
        if header.record_type != RecordType::HeaderIndex {
            self.pending_index
                .push(IndexEntry::from_header(&header, offset, size));
            if self.pending_index.len() >= INDEX_FLUSH_ENTRIES {
                self.flush_header_index()?;
            }
            self.rotate_if_over_target()?;
        }
        // Last, once the record is accounted and indexed: a write that fails
        // leaves it staged and known, for the next commit to carry.
        self.write_through_if_over_stage()
    }

    /// Append an **already-encoded** record verbatim to the active segment —
    /// the same accounting + header-index bookkeeping as [`Self::append_record`]
    /// but with **no** re-encode: the caller supplies the raw 4 KB-aligned
    /// record bytes, read verbatim from another segment. Used by background
    /// maintenance to relocate `Chunk`/`Tokens` records with no
    /// decode/CRC-verify/re-encode round trip. Returns `(segment, offset,
    /// size)`. Not for singletons — those go through [`Self::append_carried`]
    /// so `manifest.ingest` repoints them. A read-only handle refuses with
    /// [`PersistenceError::ReadOnly`].
    ///
    /// The bookkeeping reads the header out of `raw` — one JSON line — so the
    /// copy is indexed with the identity and digest it was written with, which
    /// no location map holds.
    pub fn append_raw_record(&mut self, raw: &[u8]) -> Result<Appended> {
        if self.is_read_only() {
            return Err(PersistenceError::ReadOnly);
        }
        let (header, _) = decode_header(raw)?;
        let header = &header;
        let (segment, offset) = self.segments.stage(raw);
        let size = raw.len() as u64;
        // The copy keeps the record's identity, so it is now where the record
        // is.
        self.track_identity(header, segment, offset);
        // Maintenance relocates verbatim only records the accounting keys by
        // header alone — never a section tombstone, which it re-emits.
        self.accounting.record(header, &[], size);
        // A relocated `Snapshot` becomes the stream's current on-disk tail —
        // the persistence-side map must follow it, or the next maintenance
        // pass would mis-read the relocated (live) copy as superseded and skip
        // carrying it out of a segment about to be dropped.
        self.track_snapshot_loc(header, segment, offset, size);
        // A relocated `Npc` record is that character's new on-disk home, for
        // exactly the same reason.
        self.track_npc_loc(header, segment, offset, size);
        // A relocated QSA index page likewise: its copy is where the record is,
        // and the segment liveness counts it from `metadata_locs`.
        if header.record_type == RecordType::TurnIndexPage {
            self.metadata_locs.offer(
                metadata_key(header),
                Some(Self::placed_loc(header, segment, offset, size)),
                header.identity(),
            );
        }
        // `Chunk` / `Tokens` are indexed on the substrate, not the manifest, so
        // no `manifest.ingest` — the caller repoints the substrate index.
        if header.record_type != RecordType::HeaderIndex {
            self.pending_index
                .push(IndexEntry::from_header(header, offset, size));
            if self.pending_index.len() >= INDEX_FLUSH_ENTRIES {
                self.flush_header_index()?;
            }
            self.rotate_if_over_target()?;
        }
        // Last, as in `append_record`.
        self.write_through_if_over_stage()?;
        Ok(Appended {
            segment,
            offset,
            size,
            identity: header.identity(),
        })
    }

    /// Flush the accumulated digests as `HeaderIndex` record(s) chained
    /// to the previous index, commit them durably, and publish the new
    /// chain head in the superblock. The commit-before-publish order
    /// means the hint never points at un-flushed bytes; recovery's
    /// fallback covers the crash window between the two writes either
    /// way. A read-only handle flushes nothing: its un-indexed tail is the one
    /// it found on disk, and the next writable open indexes it.
    pub(super) fn flush_header_index(&mut self) -> Result<()> {
        if self.pending_index.is_empty() || self.is_read_only() {
            return Ok(());
        }
        while !self.pending_index.is_empty() {
            let take = self.pending_index.len().min(INDEX_FLUSH_ENTRIES);
            let batch: Vec<IndexEntry> = self.pending_index.drain(..take).collect();
            let payload = encode_index_payload(self.last_index.unwrap_or((0, 0)), &batch);
            let at = self.append_record(RecordType::HeaderIndex, 0, 0, 0, 0, 0, &payload)?;
            self.last_index = Some((at.offset, at.size));
        }
        self.segments.commit()?;
        if let Some(li) = self.last_index {
            self.segments.set_last_index(li)?;
        }
        Ok(())
    }

    /// The newest committed `HeaderIndex` chain head, if any — the
    /// record the next flush chains to and the superblock hint's value.
    /// Diagnostic accessor.
    pub fn last_index(&self) -> Option<(u64, u64)> {
        self.last_index
    }

    /// Number of appended records not yet covered by a `HeaderIndex`
    /// record. Diagnostic accessor.
    pub fn pending_index_len(&self) -> usize {
        self.pending_index.len()
    }

    /// Data records the open-time recovery replayed (chain digests +
    /// walked tail). Diagnostic accessor.
    pub fn recovered_record_count(&self) -> usize {
        self.recovered_records
    }

    /// Declare a stream — append its `StreamDecl` record. Returns the
    /// stream's derived id and the identity the record was written with. A
    /// section's declaration starts its generation, which every record this
    /// node writes for the section from here on carries.
    pub fn declare_stream(&mut self, decl: &StreamDecl) -> Result<Stamped<StreamId>> {
        let stream = decl.stream_id();
        let at =
            self.append_record(RecordType::StreamDecl, 0, stream.0, 0, 0, 0, &decl.encode())?;
        if matches!(decl, StreamDecl::PromptSection(_)) {
            self.minter.start_section(stream.0, at.identity.oseq);
        }
        Ok(Stamped {
            value: stream,
            id: at.identity,
        })
    }
}
