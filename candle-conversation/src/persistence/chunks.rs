//! Writes and reads the KV chunk records that hold each sealed or partial chunk's bytes and quantization metadata.
//!
//! A chunk's golden is the Fletcher-32 the seal computed over the arena bytes when the seal supplies it, and is otherwise computed on the host over the same bytes. Golden mismatches on write and on read are logged rather than raised, so a bad byte path is flagged without aborting a write or a load, and the restart path reads metadata only and never reaches the check. Batched reads coalesce adjacent records into one read per stripe. An inherited log fills only the indices the active log lacks, so an inherited record never shadows a live one.

use super::*;

/// Read-into-RAM/VRAM golden check (non-fatal). Recompute a chunk's golden over
/// the KV bytes just read off disk and warn on mismatch — flagging on-disk or
/// read-path corruption without failing the latency-sensitive load. The
/// restart/recovery read never reaches here: it reads only metadata records.
pub(super) fn warn_on_chunk_golden_mismatch(
    header: &RecordHeader,
    payload: &ChunkPayload,
    chunk_idx: u64,
) {
    let recomputed = candle::fletcher::fletcher32(&payload.kv_bytes);
    if recomputed != header.crc {
        tracing::warn!(
            target: "candle_conversation::persistence::golden",
            chunk_idx,
            stored = header.crc,
            recomputed,
            "chunk golden mismatch on batched read into RAM — possible on-disk/read corruption"
        );
    }
}

impl SubstratePersistence {
    /// Append a `Chunk` record — one sealed or partial KV chunk's bytes and
    /// quantization metadata. `token_count` is 32 for a sealed chunk, less
    /// for a partial tail; `format` is the `KvFormat` tag. Returns where the
    /// record landed and its identity.
    pub fn write_chunk(
        &mut self,
        stream_id: StreamId,
        chunk_index: u64,
        token_count: u64,
        format: u8,
        golden: Option<u32>,
        payload: &ChunkPayload,
    ) -> Result<Appended> {
        // `Some(g)` is the seal path's GPU-computed golden (Fletcher-32 over the
        // arena bytes before the DtoH copy). `None` means no precomputed golden
        // is available, so take it host-side over these bytes now — the same
        // host-truth the CPU gather path uses.
        let golden = golden.unwrap_or_else(|| candle::fletcher::fletcher32(&payload.kv_bytes));
        self.append_record(
            RecordType::Chunk,
            format,
            stream_id.0,
            chunk_index,
            token_count,
            golden,
            &payload.encode(),
        )
    }

    /// Read one chunk's payload — from the active log, else any inherited
    /// log (§13.5). The chunk must be written to the file — committed, or
    /// written through by a long append; it is read from the file, not the
    /// staging buffer.
    pub fn read_chunk(
        &mut self,
        substrate: &Substrate,
        stream_id: StreamId,
        chunk_index: u64,
    ) -> Result<ChunkPayload> {
        if let Some(loc) = substrate
            .stream_of(stream_id)
            .and_then(|s| s.chunks.get(&chunk_index))
            .copied()
        {
            let record = self
                .segments
                .read_record_at(loc.segment, loc.offset, loc.record_size)?;
            return ChunkPayload::decode(&record.payload);
        }
        for inherited in &self.inherited {
            if let Some(loc) = inherited
                .substrate()
                .stream_of(stream_id)
                .and_then(|s| s.chunks.get(&chunk_index))
                .copied()
            {
                let record = inherited.read_record(loc.offset, loc.record_size)?;
                return ChunkPayload::decode(&record.payload);
            }
        }
        Err(PersistenceError::Corrupt(format!(
            "no chunk record for stream {} index {chunk_index}",
            stream_id.0
        )))
    }

    /// Read every chunk of a stream, in chunk-index order — the cold-load
    /// disk read. Resolves across the active and inherited logs.
    ///
    /// Allocates a scratch buffer internally; callers that load many turns
    /// in sequence (e.g. `elevate_to_hot`) should call
    /// [`Self::read_stream_chunks_batched`] directly with a reusable buffer.
    pub fn read_stream_chunks(
        &mut self,
        substrate: &Substrate,
        stream_id: StreamId,
    ) -> Result<Vec<(u64, ChunkPayload)>> {
        let mut buf: Vec<u8> = Vec::new();
        self.read_stream_chunks_batched(substrate, stream_id, &mut buf)
    }

    /// Read every chunk of a stream in a small number of stripe-coalesced
    /// I/Os — the cold-load hot path.
    ///
    /// For each chunk, the manifest already carries `offset` and the
    /// exact padded `record_size`. We sort chunks by file offset per
    /// source log (active + inherited), coalesce adjacent records into
    /// stripes, and issue **one read per stripe** into the caller's
    /// reusable scratch buffer. For a freshly-persisted turn the records
    /// are contiguous on disk → ~1 syscall covers all of them.
    ///
    /// Decoding (CRC verify + payload parse) happens in memory against
    /// slices of `buf`. The returned `ChunkPayload`s own their data;
    /// `buf` can be reused for the next stream.
    ///
    /// `buf` is resized as needed (grows monotonically across calls).
    pub fn read_stream_chunks_batched(
        &mut self,
        substrate: &Substrate,
        stream_id: StreamId,
        buf: &mut Vec<u8>,
    ) -> Result<Vec<(u64, ChunkPayload)>> {
        use std::collections::{BTreeMap, HashSet};

        #[derive(Copy, Clone)]
        struct ChunkSource {
            chunk_idx: u64,
            file_offset: u64,
            record_size: u64,
        }
        #[derive(Copy, Clone)]
        struct Stripe {
            file_offset: u64,
            len: usize,
        }

        fn build_stripes(chunks: &[ChunkSource]) -> Vec<Stripe> {
            let mut out: Vec<Stripe> = Vec::new();
            let mut it = chunks.iter();
            let Some(first) = it.next() else {
                return out;
            };
            let mut start = first.file_offset;
            let mut end = first.file_offset + first.record_size;
            for c in it {
                if c.file_offset == end {
                    end = c.file_offset + c.record_size;
                } else {
                    out.push(Stripe {
                        file_offset: start,
                        len: (end - start) as usize,
                    });
                    start = c.file_offset;
                    end = c.file_offset + c.record_size;
                }
            }
            out.push(Stripe {
                file_offset: start,
                len: (end - start) as usize,
            });
            out
        }

        // Collect from active first; remember which indices we've seen so
        // inherited logs don't shadow live entries. A live stream's chunks
        // may span several segments (sealed ones plus the active), so group
        // them by segment — stripe coalescing runs within one segment file.
        let mut seen: HashSet<u64> = HashSet::new();
        let mut active_by_seg: BTreeMap<SegmentId, Vec<ChunkSource>> = BTreeMap::new();
        if let Some(s) = substrate.stream_of(stream_id) {
            for (&chunk_idx, loc) in &s.chunks {
                seen.insert(chunk_idx);
                active_by_seg
                    .entry(loc.segment)
                    .or_default()
                    .push(ChunkSource {
                        chunk_idx,
                        file_offset: loc.offset,
                        record_size: loc.record_size,
                    });
            }
        }
        // Then each inherited log, filling in indices not present in active.
        let mut per_inh: Vec<Vec<ChunkSource>> =
            self.inherited.iter().map(|_| Vec::new()).collect();
        for (i, inh) in self.inherited.iter().enumerate() {
            if let Some(s) = inh.substrate().stream_of(stream_id) {
                for (&chunk_idx, loc) in &s.chunks {
                    if seen.insert(chunk_idx) {
                        per_inh[i].push(ChunkSource {
                            chunk_idx,
                            file_offset: loc.offset,
                            record_size: loc.record_size,
                        });
                    }
                }
            }
        }

        // Per-source: sort by file_offset, build stripes. The active source
        // is split per segment; each segment's stripes read from its own file.
        let mut active_seg_stripes: Vec<(SegmentId, Vec<ChunkSource>, Vec<Stripe>)> = Vec::new();
        for (seg, mut chunks) in active_by_seg {
            chunks.sort_unstable_by_key(|c| c.file_offset);
            let stripes = build_stripes(&chunks);
            active_seg_stripes.push((seg, chunks, stripes));
        }
        for v in &mut per_inh {
            v.sort_unstable_by_key(|c| c.file_offset);
        }
        let inh_stripes: Vec<Vec<Stripe>> = per_inh.iter().map(|v| build_stripes(v)).collect();

        // Size the scratch buffer to the exact sum of stripe spans.
        let total: usize = active_seg_stripes
            .iter()
            .flat_map(|(_, _, st)| st.iter())
            .map(|s| s.len)
            .sum::<usize>()
            + inh_stripes
                .iter()
                .flat_map(|v| v.iter())
                .map(|s| s.len)
                .sum::<usize>();
        if buf.len() < total {
            buf.resize(total, 0);
        }

        // Walk each source's stripes; read each stripe once, then decode
        // every chunk that falls within it.
        let mut out: Vec<(u64, ChunkPayload)> = Vec::with_capacity(
            active_seg_stripes
                .iter()
                .map(|(_, c, _)| c.len())
                .sum::<usize>()
                + per_inh.iter().map(|v| v.len()).sum::<usize>(),
        );
        let mut buf_cur: usize = 0;

        // Active source, one segment file at a time.
        for (seg, chunks, stripes) in &active_seg_stripes {
            let mut chunk_iter = chunks.iter().peekable();
            for stripe in stripes {
                let region = &mut buf[buf_cur..buf_cur + stripe.len];
                self.segments.read_into(*seg, stripe.file_offset, region)?;
                let stripe_end = stripe.file_offset + stripe.len as u64;
                while let Some(c) = chunk_iter.peek() {
                    if c.file_offset >= stripe_end {
                        break;
                    }
                    let within = (c.file_offset - stripe.file_offset) as usize;
                    let start = buf_cur + within;
                    let end = start + c.record_size as usize;
                    let (header, payload_bytes, _) = decode_record(&buf[start..end])?;
                    let payload = ChunkPayload::decode(payload_bytes)?;
                    warn_on_chunk_golden_mismatch(&header, &payload, c.chunk_idx);
                    out.push((c.chunk_idx, payload));
                    chunk_iter.next();
                }
                buf_cur += stripe.len;
            }
        }

        // Each inherited source.
        for (i, chunks) in per_inh.iter().enumerate() {
            let stripes = &inh_stripes[i];
            let mut chunk_iter = chunks.iter().peekable();
            for stripe in stripes {
                let region = &mut buf[buf_cur..buf_cur + stripe.len];
                self.inherited[i].read_into(stripe.file_offset, region)?;
                let stripe_end = stripe.file_offset + stripe.len as u64;
                while let Some(c) = chunk_iter.peek() {
                    if c.file_offset >= stripe_end {
                        break;
                    }
                    let within = (c.file_offset - stripe.file_offset) as usize;
                    let start = buf_cur + within;
                    let end = start + c.record_size as usize;
                    let (header, payload_bytes, _) = decode_record(&buf[start..end])?;
                    let payload = ChunkPayload::decode(payload_bytes)?;
                    warn_on_chunk_golden_mismatch(&header, &payload, c.chunk_idx);
                    out.push((c.chunk_idx, payload));
                    chunk_iter.next();
                }
                buf_cur += stripe.len;
            }
        }

        out.sort_unstable_by_key(|(idx, _)| *idx);
        Ok(out)
    }

    /// Build a chunked-read plan for `stream_id`, sized so each
    /// chunk's bytes fit within `buffer_size`. Used by the cold-load
    /// orchestrator to stream a turn through a fixed-size pinned
    /// scratch — see [`chunk_plan`] for the partition semantics.
    pub fn plan_chunked_read(
        &self,
        substrate: &Substrate,
        stream_id: StreamId,
        buffer_size: usize,
    ) -> ChunkedReadPlan {
        let active_chunks = substrate.stream_of(stream_id).map(|s| &s.chunks);
        let inherited_chunks: Vec<Option<&POrdMap<u64, ChunkLoc>>> = self
            .inherited
            .iter()
            .map(|i| i.substrate().stream_of(stream_id).map(|s| &s.chunks))
            .collect();
        chunk_plan::plan_chunked_read(
            self.segments.active_id(),
            active_chunks,
            &inherited_chunks,
            buffer_size,
        )
    }
}
