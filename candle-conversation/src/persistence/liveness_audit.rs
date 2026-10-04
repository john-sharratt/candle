//! Audit of the maintenance liveness count against what maintenance carries.
//!
//! Maintenance compacts a segment once enough of it reads dead, and "dead" is
//! whatever [`SubstratePersistence::segment_liveness`] does not count. The count
//! and the carry are written separately: relocation moves the read-back records
//! a target segment holds, and the resident re-emission rewrites every current
//! metadata record from RAM. Where the two disagree the store churns or loses
//! data:
//!
//! - **carried but counted dead** — the record is rewritten on every op, and the
//!   segment it lands in reads that much dead and is compacted again, forever;
//! - **counted live but not carried** — a drop of that segment loses a record
//!   the count said was needed, and a segment holding only such records is
//!   pinned.
//!
//! This walks every record of every segment, reading payloads only for the
//! resident types keyed by their payload, and reports both disagreements by
//! segment and record type. It reads; it writes nothing.

use std::collections::{BTreeMap, HashMap, HashSet};
use std::path::Path;

use super::log_file::{LogFile, SUPERBLOCK_SIZE};
use super::maintenance::gather_resident_set;
use super::manifest::{decode_conv_state_payload, decode_label_payload};
use super::record::{
    DebugIdPayload, DistillPayload, RecordType, TombstonePayload, TreeMetadataPayload,
    TurnCouplingPayload,
};
use super::segment::SegmentId;
use super::segmented_log::segment_path;
use super::walker::walk_filtered;
use super::{Result, SubstratePersistence};
use crate::substrate::Substrate;

/// The identity a resident record supersedes by: a newer record with the same
/// key replaces the older on reload.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ResidentKey {
    pub record_type: RecordType,
    pub id: u64,
    pub sub: u64,
}

/// Resident types whose key is in the payload rather than the header — the
/// ones `SubstratePersistence::resident_locs` tracks. Every other resident type
/// keys by its header and is tracked in `metadata_locs`.
pub fn keyed_by_payload(rt: RecordType) -> bool {
    matches!(
        rt,
        RecordType::Label
            | RecordType::ConvState
            | RecordType::TreeMetadata
            | RecordType::TurnCoupling
            | RecordType::DebugId
            | RecordType::Tombstone
            | RecordType::Distilled
    )
}

/// The supersession key of a resident record, or `None` for a type maintenance
/// does not re-emit (or a payload that does not decode).
pub fn resident_key(rt: RecordType, stream_id: u64, payload: &[u8]) -> Option<ResidentKey> {
    let key = |id: u64, sub: u64| {
        Some(ResidentKey {
            record_type: rt,
            id,
            sub,
        })
    };
    match rt {
        RecordType::StreamDecl
        | RecordType::ProjectionEvents
        | RecordType::WideQSig
        | RecordType::TurnIndexPage
        | RecordType::Commit
        | RecordType::SectionTombstone
        | RecordType::CustomObject => key(stream_id, 0),
        RecordType::ConvState => key(decode_conv_state_payload(payload).ok()?.0, 0),
        RecordType::Label => key(decode_label_payload(payload).ok()?.0, 0),
        RecordType::TreeMetadata => {
            let p = TreeMetadataPayload::decode(payload).ok()?;
            key(p.timeline_id, p.turn_index as u64)
        }
        RecordType::TurnCoupling => {
            let p = TurnCouplingPayload::decode(payload).ok()?;
            key(p.timeline_id, p.from_turn as u64)
        }
        RecordType::DebugId => key(DebugIdPayload::decode(payload).ok()?.timeline_id, 0),
        RecordType::Distilled => key(DistillPayload::decode(payload).ok()?.timeline_id, 0),
        RecordType::Tombstone => {
            let p = TombstonePayload::decode(payload).ok()?;
            // A whole-timeline tombstone and each turn's are separate records.
            key(p.timeline_id, p.turn_index.map_or(0, |t| t as u64 + 1))
        }
        _ => None,
    }
}

/// One segment's records of one type.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct AuditCell {
    pub records: u64,
    pub bytes: u64,
    /// Bytes [`SubstratePersistence::segment_liveness`] counts live.
    pub counted: u64,
    /// Bytes maintenance carries forward — relocated, or the current copy of a
    /// resident record it re-emits.
    pub carried: u64,
    /// Carried but counted dead: what makes a segment churn.
    pub carried_uncounted: u64,
    pub carried_uncounted_records: u64,
    /// Counted live but never carried: what a drop would lose.
    pub counted_uncarried: u64,
    pub counted_uncarried_records: u64,
}

/// The audit, keyed by segment then record type.
#[derive(Debug, Default)]
pub struct LivenessAudit {
    pub cells: BTreeMap<(SegmentId, RecordType), AuditCell>,
}

impl LivenessAudit {
    /// Every segment's cells summed by record type.
    pub fn by_type(&self) -> BTreeMap<RecordType, AuditCell> {
        let mut out: BTreeMap<RecordType, AuditCell> = BTreeMap::new();
        for (&(_, rt), c) in &self.cells {
            add(out.entry(rt).or_default(), c);
        }
        out
    }

    /// Every type's cells summed by segment.
    pub fn by_segment(&self) -> BTreeMap<SegmentId, AuditCell> {
        let mut out: BTreeMap<SegmentId, AuditCell> = BTreeMap::new();
        for (&(seg, _), c) in &self.cells {
            add(out.entry(seg).or_default(), c);
        }
        out
    }
}

fn add(into: &mut AuditCell, c: &AuditCell) {
    into.records += c.records;
    into.bytes += c.bytes;
    into.counted += c.counted;
    into.carried += c.carried;
    into.carried_uncounted += c.carried_uncounted;
    into.carried_uncounted_records += c.carried_uncounted_records;
    into.counted_uncarried += c.counted_uncarried;
    into.counted_uncarried_records += c.counted_uncarried_records;
}

impl SubstratePersistence {
    /// Audit the liveness count against the carry over every segment. Read-only.
    pub fn liveness_audit(&self, substrate: &Substrate) -> Result<LivenessAudit> {
        let mut segments: Vec<SegmentId> = self.segments.sealed_ids().to_vec();
        segments.push(self.segments.active_id());

        let counted: HashSet<(SegmentId, u64)> = self
            .counted_live_records(substrate)
            .0
            .iter()
            .map(|r| (r.segment, r.offset))
            .collect();

        // What a compaction of every segment at once would relocate.
        let mut carried: HashSet<(SegmentId, u64)> = HashSet::new();
        let (chunks, tokens, snapshots, branches, singletons) =
            self.gather_relocations(substrate, &segments);
        carried.extend(chunks.iter().map(|(_, _, l)| (l.segment, l.offset)));
        for loc in tokens
            .iter()
            .chain(&snapshots)
            .chain(&branches)
            .map(|(_, l)| l)
            .chain(singletons.iter().map(|(_, l)| l))
        {
            carried.insert((loc.segment, loc.offset));
        }
        for (_, loc) in self.npc_relocations(&segments) {
            carried.insert((loc.segment, loc.offset));
        }
        for r in self.vfs_relocations(substrate, &segments) {
            carried.insert((r.loc.segment, r.loc.offset));
        }
        // What the resident re-emission writes, by key.
        let emitted: HashSet<ResidentKey> = gather_resident_set(substrate)
            .iter()
            .filter_map(|r| resident_key(r.rt, r.stream_id, &r.payload))
            .collect();

        // One walk: tally every record, and find each resident key's newest copy.
        let mut seen: Vec<(SegmentId, u64, RecordType, u64)> = Vec::new();
        let mut newest: HashMap<ResidentKey, (SegmentId, u64)> = HashMap::new();
        for &seg in &segments {
            walk_segment(self.dir(), seg, |rt, stream_id, offset, size, payload| {
                seen.push((seg, offset, rt, size));
                if let Some(k) = resident_key(rt, stream_id, payload) {
                    newest.insert(k, (seg, offset));
                }
            })?;
        }
        for (k, at) in &newest {
            if emitted.contains(k) {
                carried.insert(*at);
            }
        }

        let mut audit = LivenessAudit::default();
        for (seg, offset, rt, size) in seen {
            let cell = audit.cells.entry((seg, rt)).or_default();
            let is_counted = counted.contains(&(seg, offset));
            let is_carried = carried.contains(&(seg, offset));
            cell.records += 1;
            cell.bytes += size;
            if is_counted {
                cell.counted += size;
            }
            if is_carried {
                cell.carried += size;
            }
            if is_carried && !is_counted {
                cell.carried_uncounted += size;
                cell.carried_uncounted_records += 1;
            }
            if is_counted && !is_carried {
                cell.counted_uncarried += size;
                cell.counted_uncarried_records += 1;
            }
        }
        Ok(audit)
    }
}

/// Visit every record of one segment: its type, header stream id, offset,
/// padded size, and — for a payload-keyed resident type only — its payload.
fn walk_segment(
    dir: &Path,
    seg: SegmentId,
    mut visit: impl FnMut(RecordType, u64, u64, u64, &[u8]),
) -> Result<()> {
    let mut log = LogFile::open_read_only(&segment_path(dir, seg))?;
    walk_filtered(&mut log, seg, SUPERBLOCK_SIZE, keyed_by_payload, |e| {
        let h = &e.record.header;
        visit(
            h.record_type,
            h.stream_id,
            e.offset,
            e.size,
            &e.record.payload,
        );
    })?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::persistence::manifest::encode_label_payload;
    use crate::persistence::record::TombstonePayload;
    use crate::persistence::streams::{StreamDecl, TurnDecl};
    use std::collections::BTreeMap;

    /// **A conversation's current Label is live weight, and the audit agrees.**
    /// Re-emitted from RAM on every op, it read dead wherever it landed, so the
    /// segment holding it looked reclaimable and was compacted again.
    #[test]
    fn a_current_label_is_counted_live_and_the_audit_is_consistent() {
        let dir = tempfile::tempdir().unwrap();
        // The Label's timeline is a live conversation: a turn declares it.
        let decl = StreamDecl::Turn(TurnDecl {
            timeline_id: 0x7171,
            turn_index: 0,
            turn_id_day: 0,
            turn_id_seq: 1,
            role: 2,
            block_start: 0,
            block_end: 1,
            layer_id: 1,
            group_id: 1,
            anchored_prefix: Vec::new(),
            view: Vec::new(),
            segments: Vec::new(),
            tags: Vec::new(),
        });
        let label = encode_label_payload(0x7171, "conv-a", "a title", &BTreeMap::new());
        {
            let mut substrate = Substrate::new();
            let mut sp =
                SubstratePersistence::open_in_with_substrate(dir.path(), &mut substrate).unwrap();
            sp.declare_stream(&decl).unwrap();
            substrate.apply_stream_decl(decl.stream_id(), decl.clone());
            sp.append_record(RecordType::Label, 0, 0, 0, 0, 0, &label)
                .unwrap();
            sp.commit().unwrap();
        }
        let mut substrate = Substrate::new();
        let sp = SubstratePersistence::open_in_with_substrate_read_only(dir.path(), &mut substrate)
            .unwrap();
        let audit = sp.liveness_audit(&substrate).unwrap();
        let labels = audit.by_type()[&RecordType::Label];
        assert_eq!(labels.records, 1);
        assert_eq!(labels.counted, labels.bytes, "the current Label is live");
        for (key, cell) in &audit.cells {
            assert_eq!(
                cell.carried_uncounted, 0,
                "{key:?} carried but counted dead"
            );
            assert_eq!(cell.counted_uncarried, 0, "{key:?} counted but not carried");
        }
    }

    #[test]
    fn header_keyed_records_key_on_their_stream() {
        let k = resident_key(RecordType::WideQSig, 42, &[]).unwrap();
        assert_eq!((k.id, k.sub), (42, 0));
        assert!(resident_key(RecordType::Chunk, 42, &[]).is_none());
    }

    #[test]
    fn a_turn_tombstone_and_its_timelines_are_different_keys() {
        let whole = TombstonePayload {
            timeline_id: 9,
            turn_index: None,
            reason: None,
        }
        .encode();
        let turn = TombstonePayload {
            timeline_id: 9,
            turn_index: Some(0),
            reason: None,
        }
        .encode();
        let a = resident_key(RecordType::Tombstone, 0, &whole).unwrap();
        let b = resident_key(RecordType::Tombstone, 0, &turn).unwrap();
        assert_ne!(a, b);
        assert_eq!((a.id, a.sub, b.sub), (9, 0, 1));
    }
}
