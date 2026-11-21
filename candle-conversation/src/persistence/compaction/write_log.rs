//! Writes the live set to a fresh log by copying verbatim records in coalesced
//! stripes.
//!
//! Records that sit in one source segment are read back in stripes of bounded
//! size and staged straight from the read buffer. Synthesised items are encoded
//! from memory. The write buffer is flushed between stripes, so neither buffer
//! grows with the store. The new log gets a fresh header chain and publishes its
//! head, so it recovers like any other log. Each copied record keeps the identity
//! it was written with, or is stamped with a new one when adoption is in effect.

use super::*;

/// Write the live set to a fresh log at `path`, interleaving a fresh
/// `HeaderIndex` chain and publishing its head in the superblock — a
/// just-compacted log recovers through the chain like any other.
///
/// `Raw` items are read back from their source segment through `read_into` in
/// **coalesced stripes** (a contiguous run of records → one sequential read)
/// and staged **byte-for-byte** — no decode, no CRC re-verify, no re-encode.
/// `Synth` items are encoded from their in-RAM payload. The write buffer is
/// flushed incrementally so the whole new log is never resident at once.
/// Returns the open [`LogFile`] and its (singleton-only) manifest. `path` is
/// removed first if it already exists.
///
/// With `adoption`, a record with no identity is stamped as it is written
/// ([`Adoption::restamp`]) — a read-back one decoded and encoded again under
/// its new header — and every other record is written as without.
///
/// `written` is told how many of `items` are staged, after each record or
/// stripe: a whole-store rewrite runs for minutes, and this is its counter.
pub fn write_compacted_log(
    path: &Path,
    items: &[CompactItem],
    read_into: &mut dyn FnMut(SegmentId, u64, &mut [u8]) -> Result<()>,
    mut adoption: Option<&mut Adoption>,
    written: &mut dyn FnMut(usize),
) -> Result<(LogFile, Manifest)> {
    if path.exists() {
        std::fs::remove_file(path)?;
    }
    let mut log = LogFile::create(path)?;
    let mut manifest = Manifest::new();

    let mut pending: Vec<IndexEntry> = Vec::new();
    let mut last_index: (u64, u64) = (0, 0);
    let flush_index =
        |log: &mut LogFile, pending: &mut Vec<IndexEntry>, last_index: &mut (u64, u64)| {
            let payload = encode_index_payload(*last_index, pending);
            let h = RecordHeader {
                record_type: RecordType::HeaderIndex,
                format: 0,
                payload_len: payload.len() as u64,
                crc: 0,
                stream_id: 0,
                chunk_index: 0,
                token_count: 0,
                ..RecordHeader::NONE
            };
            let bytes = encode_record(&h, &payload);
            let offset = log.stage(&bytes);
            *last_index = (offset, bytes.len() as u64);
            pending.clear();
        };

    // Single streaming pass in dependency order. `Synth` items encode from their
    // in-RAM payload; a `Raw` item begins a **run** — the maximal span of
    // consecutive `Raw` items that is also physically contiguous in one source
    // segment (a stream's chunk/token run: adjacent on disk and adjacent in the
    // list). The run is read in **coalesced stripes** of at most
    // `MAX_STRIPE_BYTES` and each record staged straight from the read buffer:
    // one copy per record, no per-record allocation, and no buffer — read or
    // write — larger than a stripe plus `COMPACT_FLUSH_BYTES`.
    let mut buf: Vec<u8> = Vec::new();
    let mut i = 0;
    while i < items.len() {
        match &items[i] {
            CompactItem::Synth { header, payload } => {
                let mut h = *header;
                h.payload_len = payload.len() as u64;
                let stamped = match adoption.as_deref_mut() {
                    Some(a) => a.restamp(&h, payload)?,
                    None => None,
                };
                let (written_header, bytes) = match &stamped {
                    Some((h, payload)) => encode_record_written(h, payload),
                    None => encode_record_written(&h, payload),
                };
                let offset = log.stage(&bytes);
                pending.push(IndexEntry::from_header(
                    &written_header,
                    offset,
                    bytes.len() as u64,
                ));
                i += 1;
                written(i);
            }
            CompactItem::Raw {
                segment,
                offset,
                record_size,
                ..
            } => {
                let seg = *segment;
                let start = *offset;
                let mut end = start + record_size;
                let mut j = i + 1;
                while let Some(CompactItem::Raw {
                    segment: s2,
                    offset: o2,
                    record_size: rs2,
                    ..
                }) = items.get(j)
                {
                    if *s2 != seg || *o2 != end {
                        break;
                    }
                    end += rs2;
                    j += 1;
                }
                // The run is read a bounded stripe at a time, and the write
                // buffer flushed between stripes: a segment of back-to-back
                // live chunks is one run of gigabytes (see `stripes`).
                let run = &items[i..j];
                let spans: Vec<(u64, u64)> = run
                    .iter()
                    .map(|it| match it {
                        CompactItem::Raw {
                            offset,
                            record_size,
                            ..
                        } => (*offset, *record_size),
                        CompactItem::Synth { .. } => unreachable!("a Raw run holds only Raw items"),
                    })
                    .collect();
                for (a, b) in stripes(&spans, MAX_STRIPE_BYTES) {
                    let from = spans[a].0;
                    let len = (spans[b - 1].0 + spans[b - 1].1 - from) as usize;
                    if buf.len() < len {
                        buf.resize(len, 0);
                    }
                    read_into(seg, from, &mut buf[..len])?;
                    let mut within = 0usize;
                    for it in &run[a..b] {
                        let CompactItem::Raw { record_size, .. } = it else {
                            unreachable!("a Raw run holds only Raw items");
                        };
                        let read = *record_size as usize;
                        let raw = &buf[within..within + read];
                        within += read;
                        // Indexed from the copied record's own header, so the copy
                        // keeps the identity and digest it was written with — or,
                        // adopted, the identity it is stamped with here.
                        let (header, _) = decode_header(raw)?;
                        let stamped = match adoption.as_deref_mut() {
                            Some(a) if header.identity().is_none() => {
                                let (_, payload, _) = decode_record(raw)?;
                                a.restamp(&header, payload)?
                                    .map(|(h, p)| encode_record_written(&h, &p))
                            }
                            _ => None,
                        };
                        let (header, staged, sz) = match &stamped {
                            Some((written, bytes)) => (*written, log.stage(bytes), bytes.len()),
                            None => (header, log.stage(raw), read),
                        };
                        let header = &header;
                        // Singletons resolve from the manifest (the substrate
                        // replay never rebuilds them); `Chunk` / `Tokens` are
                        // substrate-indexed and never enter the manifest.
                        let loc = RecordLoc {
                            segment: FIRST_SEGMENT,
                            offset: staged,
                            payload_len: header.payload_len,
                            record_size: sz as u64,
                        };
                        match header.record_type {
                            RecordType::ModelKey => manifest.model_key = Some(loc),
                            RecordType::Template => manifest.template = Some(loc),
                            RecordType::Tokenizer => manifest.tokenizer = Some(loc),
                            _ => {}
                        }
                        pending.push(IndexEntry::from_header(header, staged, sz as u64));
                        if pending.len() >= INDEX_FLUSH_ENTRIES {
                            flush_index(&mut log, &mut pending, &mut last_index);
                        }
                    }
                    if log.pending_len() >= COMPACT_FLUSH_BYTES {
                        log.flush()?;
                    }
                    written(i + b);
                }
                i = j;
            }
        }
        if pending.len() >= INDEX_FLUSH_ENTRIES {
            flush_index(&mut log, &mut pending, &mut last_index);
        }
        // Bound the in-RAM write buffer for a whole-store rewrite.
        if log.pending_len() >= COMPACT_FLUSH_BYTES {
            log.flush()?;
        }
    }
    if !pending.is_empty() {
        flush_index(&mut log, &mut pending, &mut last_index);
    }
    log.commit()?;
    if last_index != (0, 0) {
        log.set_last_index(last_index)?;
    }

    Ok((log, manifest))
}
