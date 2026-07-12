//! Log recovery — rebuilding the in-RAM state from the header-index
//! chain, with a filtered forward walk as the universal fallback.
//!
//! The fast path (§5.6) never probes record headers one by one. The
//! superblock carries an advisory hint to the newest committed
//! `HeaderIndex` record; recovery follows the backward `prev` chain
//! from there (a handful of reads regardless of log size), replays the
//! collected digests in append order, batch-fetches the payloads of
//! the few metadata record types whose bytes feed in-RAM state, and
//! forward-walks only the un-indexed tail after the hinted index.
//!
//! Every failure mode of the fast path — zero/garbage hint (including
//! the retired checkpoint offset old superblocks carry), a hint into
//! the pre-grown zero tail, a torn or wrong-typed or wrong-version
//! index record, a non-monotonic chain — degrades to the **full
//! filtered forward walk**: correct on any log, just slower. The walk
//! reads payload bytes only for the metadata types; the bulk types
//! (`Chunk` / `Tokens` / `Signatures`) contribute headers only.
//!
//! The recovery also reports the digests of every record past the
//! hinted index (the un-indexed tail) so the writer can seed its
//! accumulator and the next index flush covers them — the chain heals
//! forward across restarts instead of accumulating a permanent gap.

use std::collections::HashMap;

use super::header_index::{decode_index_payload, IndexEntry};
use super::log_file::{read_record_at, LogSource, SUPERBLOCK_SIZE};
use super::manifest::Manifest;
use super::record::{decode_record, verify_record_crc, Record, RecordType};
use super::walker::{self, WalkEntry};
use super::{PersistenceError, Result};

/// The outcome of recovering a log.
#[derive(Clone, Debug)]
pub struct Recovered {
    /// The reconstructed manifest (singleton record locations).
    pub manifest: Manifest,
    /// First byte past the last valid record — the live log tail.
    pub tail_offset: u64,
    /// Whether a torn record was found at `tail_offset` (the file must be
    /// truncated there).
    pub torn: bool,
    /// The newest `HeaderIndex` record the chain recovery validated —
    /// the link the writer's next index flush chains to. `None` when
    /// recovery fell back to the full walk (the next flush starts a
    /// fresh chain).
    pub last_index: Option<(u64, u64)>,
    /// Digests of every record **not** covered by the validated chain
    /// (the whole log on the fallback path). Seeds the writer's
    /// accumulator so the next flush indexes them.
    pub tail_digests: Vec<IndexEntry>,
}

/// Whether recovery needs a record's payload bytes in RAM.
///
/// `Chunk` / `Tokens` / `Signatures` payloads are stored by reference —
/// `Substrate::apply_walker_entry` keeps only their `(offset, len)` — and
/// the `ModelSpec` / `Template` / `Tokenizer` singleton payloads are read
/// back individually after recovery from their manifest locations.
/// `HeaderIndex` payloads are consumed only by the chain reader, never
/// by replay.
fn payload_needed(rt: RecordType) -> bool {
    !matches!(
        rt,
        RecordType::Chunk
            | RecordType::Tokens
            | RecordType::Signatures
            | RecordType::ModelSpec
            | RecordType::Template
            | RecordType::Tokenizer
            | RecordType::HeaderIndex
    )
}

/// Recover a log into a manifest and its true tail. `index_hint` is the
/// superblock's `last_index` — pass `(0, 0)` to force the full walk.
pub fn recover(src: &mut dyn LogSource, index_hint: (u64, u64)) -> Result<Recovered> {
    recover_with_sink(src, index_hint, |_| {})
}

/// Recovery + per-record sink. Identical to [`recover`] except every
/// replayed record — chain digest or walked tail record — is also passed
/// to `sink`, in append order. The production open paths use the sink to
/// dispatch records straight into a
/// [`Substrate`](crate::substrate::Substrate) (via
/// `Substrate::apply_walker_entry`) and the dead-byte accounting.
///
/// Entries for payload-skipped record types (see [`payload_needed`])
/// carry an empty `payload`; their headers, offsets, and sizes are exact
/// (digest-synthesized headers carry `crc = 0`, which nothing consumes).
pub fn recover_with_sink<F>(
    src: &mut dyn LogSource,
    index_hint: (u64, u64),
    mut sink: F,
) -> Result<Recovered>
where
    F: FnMut(&WalkEntry),
{
    // The chain and the metadata fetch are both validated BEFORE any
    // entry reaches the sink, so every fast-path failure can fall back
    // to the full walk without double-replaying anything into the
    // caller's state.
    let digests = match load_index_chain(src, index_hint) {
        Ok(Some(chain)) => Some(chain),
        // No usable chain — zero hint, or any inconsistency in it.
        // The full walk is always correct.
        Ok(None) => None,
        Err(e) => {
            tracing::warn!(
                "header-index chain unreadable ({e}); falling back to full recovery walk"
            );
            None
        }
    };
    if let Some(digests) = digests {
        // Batch-fetch the payload-bearing metadata records. The digests
        // expose their offsets up front, which is exactly what the
        // serial forward walk never had — nearby records coalesce into
        // shared reads instead of one probe each.
        let fetch: Vec<(u64, u64)> = digests
            .iter()
            .filter(|d| payload_needed(d.record_type))
            .map(|d| (d.offset, d.record_size as u64))
            .collect();
        match fetch_records_coalesced(src, &fetch) {
            Ok(payloads) => {
                return recover_from_chain(src, index_hint, digests, payloads, sink);
            }
            Err(e) => {
                tracing::warn!(
                    "header-index metadata fetch failed ({e}); falling back to full recovery walk"
                );
            }
        }
    }
    recover_full_walk(src, &mut sink)
}

/// Follow the backward `HeaderIndex` chain from `hint`, returning the
/// digests in **append order**, or `None` when the hint is absent /
/// invalid. I/O or decode errors bubble as `Err` so the caller can log
/// the reason before falling back — both outcomes mean "walk instead".
fn load_index_chain(src: &mut dyn LogSource, hint: (u64, u64)) -> Result<Option<Vec<IndexEntry>>> {
    if hint.0 < SUPERBLOCK_SIZE || hint.1 == 0 {
        return Ok(None);
    }
    let size = src.size()?;
    let mut batches: Vec<Vec<IndexEntry>> = Vec::new();
    let mut cur = hint;
    while cur != (0, 0) {
        if cur.0 < SUPERBLOCK_SIZE || cur.1 == 0 || cur.0 + cur.1 > size {
            return Ok(None);
        }
        // CRC-verified read — a torn or bit-rotten index record fails here.
        let rec = read_record_at(src, cur.0, cur.1)?;
        if rec.header.record_type != RecordType::HeaderIndex {
            return Ok(None);
        }
        let (prev, entries) = decode_index_payload(&rec.payload)?;
        // The chain must walk strictly backwards — a forward or
        // self-referencing link would loop.
        if prev != (0, 0) && prev.0 >= cur.0 {
            return Ok(None);
        }
        batches.push(entries);
        cur = prev;
    }
    batches.reverse();
    let digests: Vec<IndexEntry> = batches.concat();
    // Digest offsets must be strictly increasing and precede the hinted
    // index record itself — anything else means the chain and the file
    // disagree.
    let mut prev_off = 0u64;
    for d in &digests {
        if d.offset <= prev_off || d.offset + d.record_size as u64 > hint.0 {
            return Ok(None);
        }
        prev_off = d.offset;
    }
    Ok(Some(digests))
}

/// The chain fast path: replay digests (with the pre-fetched metadata
/// payloads), then forward-walk the tail after the hinted index record.
fn recover_from_chain<F>(
    src: &mut dyn LogSource,
    hint: (u64, u64),
    digests: Vec<IndexEntry>,
    mut payloads: HashMap<u64, Record>,
    mut sink: F,
) -> Result<Recovered>
where
    F: FnMut(&WalkEntry),
{
    let mut manifest = Manifest::new();

    // Replay in append order: digest headers for reference-stored types,
    // fetched records (real header + payload) for metadata types.
    let mut ingest_err: Option<PersistenceError> = None;
    for d in &digests {
        let entry = if payload_needed(d.record_type) {
            let record = payloads
                .remove(&d.offset)
                .expect("fetch covered every payload_needed digest");
            WalkEntry {
                offset: d.offset,
                record,
                size: d.record_size as u64,
            }
        } else {
            d.to_walk_entry()
        };
        if ingest_err.is_none() {
            if let Err(e) = manifest.ingest(&entry) {
                ingest_err = Some(e);
            }
        }
        sink(&entry);
    }
    if let Some(e) = ingest_err {
        return Err(e);
    }

    // Forward-walk the un-indexed tail: everything after the hinted
    // index record, including any newer HeaderIndex records whose
    // superblock update didn't land — their digested records are
    // visited directly here, so nothing is replayed twice.
    let tail_start = hint.0 + hint.1;
    let mut tail_digests: Vec<IndexEntry> = Vec::new();
    let mut walk_err: Option<PersistenceError> = None;
    let outcome = walker::walk_filtered(src, tail_start, payload_needed, |entry| {
        if entry.record.header.record_type != RecordType::HeaderIndex {
            tail_digests.push(IndexEntry::from_header(
                &entry.record.header,
                entry.offset,
                entry.size,
            ));
        }
        if walk_err.is_none() {
            if let Err(e) = manifest.ingest(entry) {
                walk_err = Some(e);
            }
        }
        sink(entry);
    })?;
    if let Some(e) = walk_err {
        return Err(e);
    }

    Ok(Recovered {
        manifest,
        tail_offset: outcome.tail_offset,
        torn: outcome.torn,
        last_index: Some(hint),
        tail_digests,
    })
}

/// The universal fallback: one filtered forward walk from the first
/// record. Correct on any log — including logs with no index chain at
/// all — at O(records) serial header probes.
fn recover_full_walk<F>(src: &mut dyn LogSource, sink: &mut F) -> Result<Recovered>
where
    F: FnMut(&WalkEntry),
{
    let mut manifest = Manifest::new();
    let mut tail_digests: Vec<IndexEntry> = Vec::new();
    let mut ingest_err: Option<PersistenceError> = None;
    let outcome = walker::walk_filtered(src, SUPERBLOCK_SIZE, payload_needed, |entry| {
        if entry.record.header.record_type != RecordType::HeaderIndex {
            tail_digests.push(IndexEntry::from_header(
                &entry.record.header,
                entry.offset,
                entry.size,
            ));
        }
        if ingest_err.is_none() {
            if let Err(e) = manifest.ingest(entry) {
                ingest_err = Some(e);
            }
        }
        sink(entry);
    })?;
    if let Some(e) = ingest_err {
        return Err(e);
    }
    Ok(Recovered {
        manifest,
        tail_offset: outcome.tail_offset,
        torn: outcome.torn,
        last_index: None,
        tail_digests,
    })
}

/// Reads within a coalesced span merge into one I/O when the gap
/// between consecutive records is at most this many bytes — wasted gap
/// bytes are cheaper than another round trip.
const FETCH_GAP_BYTES: u64 = 256 * 1024;

/// Upper bound on one coalesced fetch read.
const FETCH_SPAN_BYTES: u64 = 8 * 1024 * 1024;

/// Read the records at `locs` (`(offset, padded_size)`, any order) in
/// coalesced spans and return them keyed by offset. Every record is
/// CRC-verified — recovery is the consumption point for these payloads.
fn fetch_records_coalesced(
    src: &mut dyn LogSource,
    locs: &[(u64, u64)],
) -> Result<HashMap<u64, Record>> {
    let mut sorted: Vec<(u64, u64)> = locs.to_vec();
    sorted.sort_unstable_by_key(|(off, _)| *off);
    let mut out: HashMap<u64, Record> = HashMap::with_capacity(sorted.len());

    let mut i = 0usize;
    let mut buf: Vec<u8> = Vec::new();
    while i < sorted.len() {
        // Grow the span while the next record starts within the gap
        // threshold and the span stays bounded.
        let span_start = sorted[i].0;
        let mut span_end = sorted[i].0 + sorted[i].1;
        let mut j = i + 1;
        while j < sorted.len()
            && sorted[j].0 <= span_end + FETCH_GAP_BYTES
            && sorted[j].0 + sorted[j].1 - span_start <= FETCH_SPAN_BYTES
        {
            span_end = span_end.max(sorted[j].0 + sorted[j].1);
            j += 1;
        }
        let span_len = (span_end - span_start) as usize;
        buf.resize(span_len, 0);
        src.read_into(span_start, &mut buf)?;
        for &(off, size) in &sorted[i..j] {
            let lo = (off - span_start) as usize;
            let bytes = &buf[lo..lo + size as usize];
            let (header, payload, _) = decode_record(bytes)?;
            verify_record_crc(&header, payload)?;
            out.insert(
                off,
                Record {
                    header,
                    payload: payload.to_vec(),
                },
            );
        }
        i = j;
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::persistence::header_index::encode_index_payload;
    use crate::persistence::log_file::{LogFile, MemLog};
    use crate::persistence::record::{encode_record, RecordHeader, RecordType};
    use crate::persistence::streams::{ContentAddress, SectionDecl, StreamDecl};
    use crate::substrate::Substrate;

    fn record(rt: RecordType, stream_id: u64, chunk_index: u64, payload: &[u8]) -> Vec<u8> {
        encode_record(
            &RecordHeader {
                record_type: rt,
                format: 0,
                payload_len: payload.len() as u64,
                crc: 0, // overwritten by encode_record
                stream_id,
                chunk_index,
                token_count: if rt == RecordType::Chunk { 32 } else { 0 },
            },
            payload,
        )
    }

    fn tmp_path(tag: &str) -> std::path::PathBuf {
        let mut p = std::env::temp_dir();
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        p.push(format!("kvtier_recovery_{tag}_{nanos}.log"));
        p
    }

    /// Build a log blob + its index chain by hand: `chunks_per_index`
    /// records per HeaderIndex record. Returns (blob, hint).
    fn build_indexed_log(records: &[Vec<u8>], per_index: usize) -> (Vec<u8>, (u64, u64)) {
        let mut blob = Vec::new();
        let mut pending: Vec<IndexEntry> = Vec::new();
        let mut last_index = (0u64, 0u64);
        for bytes in records {
            let offset = SUPERBLOCK_SIZE + blob.len() as u64;
            let (header, payload, total) = decode_record(bytes).unwrap();
            let _ = payload;
            blob.extend_from_slice(bytes);
            pending.push(IndexEntry::from_header(&header, offset, total as u64));
            if pending.len() >= per_index {
                let payload = encode_index_payload(last_index, &pending);
                let rec = record(RecordType::HeaderIndex, 0, 0, &payload);
                let off = SUPERBLOCK_SIZE + blob.len() as u64;
                last_index = (off, rec.len() as u64);
                blob.extend_from_slice(&rec);
                pending.clear();
            }
        }
        (blob, last_index)
    }

    fn sample_records() -> Vec<Vec<u8>> {
        let decl = StreamDecl::PromptSection(SectionDecl {
            address: ContentAddress::default(),
            debug_name: "sect".to_string(),
        });
        vec![
            record(RecordType::StreamDecl, 7, 0, &decl.encode()),
            record(RecordType::Chunk, 7, 0, &vec![0xAB; 9000]),
            record(RecordType::Chunk, 7, 1, b"c1"),
            record(RecordType::Tokens, 7, 0, b"tok-bytes"),
            record(RecordType::ModelSpec, 0, 0, b"model-v1"),
            record(RecordType::Chunk, 8, 0, b"other-stream"),
            record(RecordType::ModelSpec, 0, 0, b"model-v2"),
        ]
    }

    /// Chain recovery and the full walk must produce identical
    /// manifests, substrate state, and sink entry sequences.
    #[test]
    fn chain_recovery_matches_full_walk() {
        let records = sample_records();
        // 2 records per index → a 4-link chain with an un-indexed tail.
        let (blob, hint) = build_indexed_log(&records, 2);
        assert!(hint.1 > 0, "test log must have an index chain");

        let mut mem_a = MemLog::with_records(&blob);
        let mut sub_a = Substrate::new();
        let mut seen_a: Vec<(RecordType, u64, usize)> = Vec::new();
        let rec_a = recover_with_sink(&mut mem_a, hint, |e| {
            seen_a.push((
                e.record.header.record_type,
                e.offset,
                e.record.payload.len(),
            ));
            sub_a.apply_walker_entry(e);
        })
        .unwrap();
        assert_eq!(rec_a.last_index, Some(hint), "chain path must be taken");

        let mut mem_b = MemLog::with_records(&blob);
        let mut sub_b = Substrate::new();
        let mut seen_b: Vec<(RecordType, u64, usize)> = Vec::new();
        let rec_b = recover_with_sink(&mut mem_b, (0, 0), |e| {
            seen_b.push((
                e.record.header.record_type,
                e.offset,
                e.record.payload.len(),
            ));
            sub_b.apply_walker_entry(e);
        })
        .unwrap();
        assert_eq!(rec_b.last_index, None, "zero hint forces the full walk");

        // The chain path never visits HeaderIndex records; the walk
        // path visits (and skips the payloads of) them. Filter them out
        // of the walk side for the comparison.
        let seen_b_filtered: Vec<_> = seen_b
            .iter()
            .filter(|(rt, _, _)| *rt != RecordType::HeaderIndex)
            .cloned()
            .collect();
        assert_eq!(seen_a.len(), seen_b_filtered.len());
        for (a, b) in seen_a.iter().zip(seen_b_filtered.iter()) {
            assert_eq!(a, b, "entry mismatch between chain and walk recovery");
        }
        assert_eq!(rec_a.manifest, rec_b.manifest);
        assert_eq!(rec_a.tail_offset, rec_b.tail_offset);
        assert_eq!(sub_a.live_chunk_count(), sub_b.live_chunk_count());
        assert_eq!(sub_a.live_chunk_count(), 3);
        // LWW singleton resolved identically (v2 wins on both paths).
        assert_eq!(rec_a.manifest.model_spec, rec_b.manifest.model_spec);
    }

    /// The un-indexed tail (records after the hinted index) is walked
    /// forward and reported as tail digests for the writer's
    /// accumulator.
    #[test]
    fn tail_after_hinted_index_is_walked_and_digested() {
        let records = sample_records();
        let (mut blob, hint) = build_indexed_log(&records, 2);
        // Three more records after the last index flush.
        let tail = [
            record(RecordType::Chunk, 9, 0, b"tail-chunk"),
            record(RecordType::Tokens, 9, 0, b"tail-tokens"),
            record(RecordType::Commit, 9, 0, b""),
        ];
        for t in &tail {
            blob.extend_from_slice(t);
        }

        let mut mem = MemLog::with_records(&blob);
        let mut sub = Substrate::new();
        let rec = recover_with_sink(&mut mem, hint, |e| sub.apply_walker_entry(e)).unwrap();
        assert_eq!(rec.last_index, Some(hint));
        // sample_records leaves 1 record un-indexed (7 % 2) + 3 tail.
        assert_eq!(rec.tail_digests.len(), 4);
        assert!(sub.has_stream(crate::persistence::streams::StreamId(9)));
        assert_eq!(sub.live_chunk_count(), 4);
    }

    /// Every corrupt-hint shape degrades to the full walk, silently
    /// producing the same state.
    #[test]
    fn bad_hints_fall_back_to_full_walk() {
        let records = sample_records();
        let (blob, hint) = build_indexed_log(&records, 3);

        // (offset of a NON-index record, plausible size) — wrong type.
        let wrong_type = (SUPERBLOCK_SIZE, 4096u64);
        // Dangling offset past EOF.
        let dangling = (SUPERBLOCK_SIZE + blob.len() as u64 + 40960, 4096u64);
        // Legacy checkpoint shape: offset with zero size.
        let legacy = (hint.0, 0u64);
        for bad in [wrong_type, dangling, legacy] {
            let mut mem = MemLog::with_records(&blob);
            let mut sub = Substrate::new();
            let rec = recover_with_sink(&mut mem, bad, |e| sub.apply_walker_entry(e)).unwrap();
            assert_eq!(
                rec.last_index, None,
                "hint {bad:?} must fall back to the walk"
            );
            assert_eq!(sub.live_chunk_count(), 3);
        }
    }

    /// A self-referencing / forward `prev` link is rejected as a loop
    /// instead of hanging.
    #[test]
    fn chain_loops_are_rejected() {
        // One index record whose prev points at itself.
        let mut blob = Vec::new();
        blob.extend_from_slice(&record(RecordType::Chunk, 1, 0, b"c0"));
        let idx_off = SUPERBLOCK_SIZE + blob.len() as u64;
        // Encode with a placeholder, measure, then re-encode pointing at
        // itself (same length: fixed-width fields).
        let entry = {
            let bytes = record(RecordType::Chunk, 1, 0, b"c0");
            let (h, _, total) = decode_record(&bytes).unwrap();
            IndexEntry::from_header(&h, SUPERBLOCK_SIZE, total as u64)
        };
        let probe = record(
            RecordType::HeaderIndex,
            0,
            0,
            &encode_index_payload((0, 0), &[entry]),
        );
        let idx_size = probe.len() as u64;
        let looped = record(
            RecordType::HeaderIndex,
            0,
            0,
            &encode_index_payload((idx_off, idx_size), &[entry]),
        );
        assert_eq!(looped.len() as u64, idx_size);
        blob.extend_from_slice(&looped);

        let mut mem = MemLog::with_records(&blob);
        let mut sub = Substrate::new();
        let rec = recover_with_sink(&mut mem, (idx_off, idx_size), |e| sub.apply_walker_entry(e))
            .unwrap();
        assert_eq!(rec.last_index, None, "looped chain must fall back");
        assert_eq!(sub.live_chunk_count(), 1);
    }

    /// A torn tail after the hinted index truncates exactly as the
    /// plain walk would.
    #[test]
    fn torn_tail_after_index_is_detected() {
        let records = sample_records();
        let (mut blob, hint) = build_indexed_log(&records, 7); // exactly one index, no gap
        let good_len = blob.len();
        let mut bad = record(RecordType::Chunk, 9, 0, b"torn");
        bad[0] = b'X';
        blob.extend_from_slice(&bad);

        let mut mem = MemLog::with_records(&blob);
        let rec = recover(&mut mem, hint).unwrap();
        assert_eq!(rec.last_index, Some(hint));
        assert!(rec.torn);
        assert_eq!(rec.tail_offset, SUPERBLOCK_SIZE + good_len as u64);
    }

    #[test]
    fn crash_recovery_truncates_a_torn_tail() {
        let path = tmp_path("crash");
        let good_records;
        {
            let mut log = LogFile::create(&path).unwrap();
            log.stage(&record(RecordType::Chunk, 1, 0, b"alpha"));
            log.stage(&record(RecordType::Chunk, 1, 1, b"beta"));
            log.commit().unwrap();
            good_records = log.write_offset();
            // Append a third record, then simulate a real crash by
            // **physically truncating** the file partway through it.
            log.stage(&record(RecordType::Chunk, 1, 2, b"gamma"));
            log.commit().unwrap();
        }
        // Drop the file back to `good_records + 64` bytes — the third
        // record's header is partially on disk but the payload isn't.
        // The walker stops here because the (parseable) header promises
        // a record the file doesn't fully hold.
        {
            let f = std::fs::OpenOptions::new().write(true).open(&path).unwrap();
            f.set_len(good_records + 64).unwrap();
        }
        {
            let mut log = LogFile::open(&path).unwrap();
            let hint = log.superblock().last_index;
            let mut substrate = Substrate::new();
            let rec =
                recover_with_sink(&mut log, hint, |e| substrate.apply_walker_entry(e)).unwrap();
            assert!(rec.torn, "the truncated third record must be detected");
            assert_eq!(rec.tail_offset, good_records);
            assert_eq!(substrate.live_chunk_count(), 2);
            // Applying the recovery: truncate to the good tail.
            log.truncate_to(rec.tail_offset).unwrap();
            log.set_write_offset(rec.tail_offset);
            assert_eq!(log.write_offset(), good_records);
        }
        std::fs::remove_file(&path).ok();
    }

    /// Bulk record payloads are skipped in both recovery modes — the
    /// sink sees their exact headers, offsets, and sizes but an empty
    /// payload, while payload-bearing types arrive intact.
    #[test]
    fn recovery_skips_bulk_payloads_but_keeps_framing() {
        let records = sample_records();
        let (blob, hint) = build_indexed_log(&records, 2);

        let mut seen: Vec<(RecordType, usize, u64)> = Vec::new();
        let mut mem = MemLog::with_records(&blob);
        recover_with_sink(&mut mem, hint, |e| {
            seen.push((
                e.record.header.record_type,
                e.record.payload.len(),
                e.record.header.payload_len,
            ));
        })
        .unwrap();
        let chunk9000 = seen
            .iter()
            .find(|(rt, _, plen)| *rt == RecordType::Chunk && *plen == 9000)
            .expect("the large chunk digest is replayed");
        assert_eq!(chunk9000.1, 0, "Chunk payload must not be read");
        let decl = seen
            .iter()
            .find(|(rt, _, _)| *rt == RecordType::StreamDecl)
            .expect("the decl is replayed");
        assert_eq!(
            decl.1 as u64, decl.2,
            "StreamDecl payload must be fetched in full"
        );
    }
}
