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
//!
//! **Two halves, because the walk is as long as the disk is slow.** What the
//! count and the carry say comes from the log's indexes, so
//! [`SubstratePersistence::liveness_audit_plan`] takes it on the log's owner.
//! [`AuditPlan::walk`] then needs only the files, and runs anywhere. A sealed
//! segment never changes, and the active one is read to the commit the plan saw,
//! so the walk meets the log as the plan left it; a segment maintenance unlinks
//! before the walk reaches it is reported in [`LivenessAudit::gone`].

use std::collections::{BTreeMap, HashMap, HashSet};
use std::io::ErrorKind;
use std::path::{Path, PathBuf};

use super::identity::RecordIdentity;
use super::log_file::{LogFile, LogSource, SUPERBLOCK_SIZE};
use super::maintenance::gather_resident_set;
use super::manifest::{decode_conv_state_payload, decode_label_payload};
use super::record::{
    DebugIdPayload, DistillPayload, RecordType, SectionTombstonePayload, TombstonePayload,
    TreeMetadataPayload, TurnCouplingPayload,
};
use super::segment::SegmentId;
use super::segmented_log::segment_path;
use super::walker::walk_filtered;
use super::{PersistenceError, Result, SubstratePersistence};
use crate::substrate::Substrate;

/// The identity a resident record supersedes by: a record with the same key
/// and a greater stamp replaces it on reload.
///
/// A `Label` record is its own key — `(timeline, origin, oseq)` — because a
/// timeline's label is assembled field by field from every record that won a
/// field (§6.5), so several are live at once and none supersedes another by
/// key. A label written before identities existed keys by its timeline alone,
/// and the last of those replaces the rest, as it always has.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ResidentKey {
    pub record_type: RecordType,
    pub id: u64,
    pub sub: u64,
    /// An origin the key names: the writing origin for a key that is one
    /// record, and a turn tombstone's turn writer; zero otherwise.
    pub origin: u64,
}

/// Resident types whose key is in the payload rather than the header — the
/// ones `SubstratePersistence::resident_locs` tracks. Every other resident type
/// keys by its header and is tracked in `metadata_locs`.
///
/// A `SectionTombstone` names its section in the header but its generation in
/// the payload, and each ended generation is its own marker.
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
            | RecordType::SectionTombstone
    )
}

/// The supersession key of resident record `record`, or `None` for a type
/// maintenance does not re-emit (or a payload that does not decode).
pub fn resident_key(
    rt: RecordType,
    stream_id: u64,
    payload: &[u8],
    record: &RecordIdentity,
) -> Option<ResidentKey> {
    let key = |id: u64, sub: u64| {
        Some(ResidentKey {
            record_type: rt,
            id,
            sub,
            origin: 0,
        })
    };
    match rt {
        // A stream's own records, one per writer: two members' turn N share a
        // stream id and are two streams (§6.6).
        RecordType::StreamDecl
        | RecordType::ProjectionEvents
        | RecordType::WideQSig
        | RecordType::TurnIndexPage
        | RecordType::Commit => Some(ResidentKey {
            record_type: rt,
            id: stream_id,
            sub: 0,
            origin: record.origin,
        }),
        // Each ended generation of a section is its own marker.
        RecordType::SectionTombstone => {
            let generation = SectionTombstonePayload::decode(payload).ok()?.generation;
            Some(ResidentKey {
                record_type: rt,
                id: stream_id,
                sub: generation.map_or(0, |g| g.oseq),
                origin: generation.map_or(0, |g| g.writer),
            })
        }
        RecordType::CustomObject => key(stream_id, 0),
        RecordType::ConvState => key(decode_conv_state_payload(payload).ok()?.0, 0),
        RecordType::Label => Some(ResidentKey {
            record_type: rt,
            id: decode_label_payload(payload).ok()?.0,
            sub: record.oseq,
            origin: record.origin,
        }),
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
            // A whole-timeline tombstone and each turn's are separate records,
            // a turn's named by its index and its writer.
            Some(ResidentKey {
                record_type: rt,
                id: p.timeline_id,
                sub: p.turn_index.map_or(0, |t| t as u64 + 1),
                origin: p.turn_writer,
            })
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
    /// Segments the plan named that were unlinked before the walk reached
    /// them: maintenance finished with them, so nothing here reads them.
    pub gone: Vec<SegmentId>,
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

/// What the log's indexes say of a liveness audit, ready to check against the
/// files. See the module documentation.
pub struct AuditPlan {
    dir: PathBuf,
    sealed: Vec<SegmentId>,
    active: SegmentId,
    /// How far the plan saw `active` durable: the walk stops there.
    active_end: u64,
    counted: HashSet<(SegmentId, u64)>,
    carried: HashSet<(SegmentId, u64)>,
    emitted: HashSet<ResidentKey>,
}

impl SubstratePersistence {
    /// Audit the liveness count against the carry over every segment. Read-only.
    pub fn liveness_audit(&self, substrate: &Substrate) -> Result<LivenessAudit> {
        self.liveness_audit_plan(substrate).walk()
    }

    /// The half of an audit that reads the log's indexes.
    pub fn liveness_audit_plan(&self, substrate: &Substrate) -> AuditPlan {
        let sealed: Vec<SegmentId> = self.segments.sealed_ids().to_vec();
        let active = self.segments.active_id();
        let active_end = self
            .sealed_reader()
            .committed_end(active)
            .unwrap_or(SUPERBLOCK_SIZE);
        let mut segments = sealed.clone();
        segments.push(active);

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
            .map(|(_, l)| l)
            .chain(snapshots.iter().chain(&branches).map(|(_, l)| l))
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
        let emitted: HashSet<ResidentKey> =
            gather_resident_set(substrate, self.marker_ids(), self.node())
                .iter()
                .filter_map(|r| resident_key(r.rt, r.stream_id, &r.payload, &r.id))
                .collect();
        AuditPlan {
            dir: self.dir().to_path_buf(),
            sealed,
            active,
            active_end,
            counted,
            carried,
            emitted,
        }
    }
}

impl AuditPlan {
    /// The half of an audit that reads every segment. Needs nothing of the log
    /// but its files.
    pub fn walk(self) -> Result<LivenessAudit> {
        let AuditPlan {
            dir,
            sealed,
            active,
            active_end,
            counted,
            mut carried,
            emitted,
        } = self;
        // One walk: tally every record, and find each resident key's current
        // copy — the greatest stamp, and of equal stamps the later copy.
        let mut seen: Vec<(SegmentId, u64, RecordType, u64)> = Vec::new();
        let mut current: HashMap<ResidentKey, (RecordIdentity, SegmentId, u64)> = HashMap::new();
        let mut audit = LivenessAudit::default();
        let ends = sealed
            .iter()
            .map(|&seg| (seg, u64::MAX))
            .chain([(active, active_end)]);
        for (seg, end) in ends {
            let walked = walk_segment(
                &dir,
                seg,
                end,
                |rt, stream_id, offset, size, payload, id| {
                    seen.push((seg, offset, rt, size));
                    if let Some(k) = resident_key(rt, stream_id, payload, &id) {
                        let held = current.get(&k).map(|(held, ..)| *held).unwrap_or_default();
                        if id.supersedes(&held) {
                            current.insert(k, (id, seg, offset));
                        }
                    }
                },
            );
            match walked {
                Err(e) if unlinked(&e) => audit.gone.push(seg),
                other => other?,
            }
        }
        for (k, (_, seg, offset)) in &current {
            if emitted.contains(k) {
                carried.insert((*seg, *offset));
            }
        }

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

/// Whether opening a segment failed because maintenance unlinked it: `NotFound`,
/// or `PermissionDenied` while a Windows volume has the name pending deletion.
fn unlinked(e: &PersistenceError) -> bool {
    matches!(
        e,
        PersistenceError::Io(e)
            if matches!(e.kind(), ErrorKind::NotFound | ErrorKind::PermissionDenied)
    )
}

/// A log read no further than `end`: the walker frames records up to its
/// source's size, and what lies past a commit may still be arriving.
struct Upto<'a> {
    log: &'a mut LogFile,
    end: u64,
}

impl LogSource for Upto<'_> {
    fn read_at(&mut self, offset: u64, len: usize) -> Result<Vec<u8>> {
        self.log.read_at(offset, len)
    }

    fn read_into(&mut self, offset: u64, dest: &mut [u8]) -> Result<()> {
        self.log.read_into(offset, dest)
    }

    fn size(&mut self) -> Result<u64> {
        Ok(LogSource::size(self.log)?.min(self.end))
    }
}

/// Visit every record of one segment below `end`: its type, header stream id,
/// offset, padded size, — for a payload-keyed resident type only — its payload,
/// and its identity.
fn walk_segment(
    dir: &Path,
    seg: SegmentId,
    end: u64,
    mut visit: impl FnMut(RecordType, u64, u64, u64, &[u8], RecordIdentity),
) -> Result<()> {
    let mut log = LogFile::open_read_only(&segment_path(dir, seg))?;
    let mut upto = Upto { log: &mut log, end };
    walk_filtered(&mut upto, seg, SUPERBLOCK_SIZE, keyed_by_payload, |e| {
        let h = &e.record.header;
        visit(
            h.record_type,
            h.stream_id,
            e.offset,
            e.size,
            &e.record.payload,
            h.identity(),
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
            let declared = sp.declare_stream(&decl).unwrap();
            substrate.apply_stream_decl(declared.value, decl.clone(), declared.id);
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

    fn label(sp: &mut SubstratePersistence, timeline: u64) {
        let label = encode_label_payload(timeline, "conv", "title", &BTreeMap::new());
        sp.append_record(RecordType::Label, 0, 0, 0, 0, 0, &label)
            .unwrap();
    }

    /// **The walk stops at the commit the plan saw.** What the owner staged
    /// and what it appends while the walk runs may be half on disk; a record
    /// past that commit is the next audit's.
    #[test]
    fn the_walk_stops_at_the_commit_the_plan_saw() {
        let dir = tempfile::tempdir().unwrap();
        let mut substrate = Substrate::new();
        let mut sp =
            SubstratePersistence::open_in_with_substrate(dir.path(), &mut substrate).unwrap();
        label(&mut sp, 1);
        sp.commit().unwrap();
        label(&mut sp, 2);
        let plan = sp.liveness_audit_plan(&substrate);
        sp.commit().unwrap();
        label(&mut sp, 3);
        sp.commit().unwrap();
        let audit = plan.walk().unwrap();
        assert_eq!(audit.by_type()[&RecordType::Label].records, 1);
        assert_eq!(audit.gone, []);
    }

    /// **A segment unlinked under the walk is reported, not fatal.**
    /// Maintenance goes on running beside it, and the audit of the rest stands.
    #[test]
    fn a_segment_unlinked_after_the_plan_is_reported() {
        let dir = tempfile::tempdir().unwrap();
        let mut substrate = Substrate::new();
        let mut sp =
            SubstratePersistence::open_in_with_substrate(dir.path(), &mut substrate).unwrap();
        label(&mut sp, 1);
        sp.commit().unwrap();
        let first = sp.segments.active_id();
        sp.seal_active().unwrap();
        label(&mut sp, 2);
        sp.commit().unwrap();
        let plan = sp.liveness_audit_plan(&substrate);
        sp.segments.drop_sealed(first).unwrap();
        let audit = plan.walk().unwrap();
        assert_eq!(audit.gone, [first]);
        assert_eq!(audit.by_type()[&RecordType::Label].records, 1);
    }

    #[test]
    fn header_keyed_records_key_on_their_stream() {
        let none = RecordIdentity::NONE;
        let k = resident_key(RecordType::WideQSig, 42, &[], &none).unwrap();
        assert_eq!((k.id, k.sub, k.origin), (42, 0, 0));
        assert!(resident_key(RecordType::Chunk, 42, &[], &none).is_none());
    }

    /// Each stamped label record is a key of its own, so two that won
    /// different fields of one timeline are both live; unstamped labels share
    /// their timeline's key.
    #[test]
    fn a_stamped_label_is_its_own_key() {
        let label = encode_label_payload(9, "c", "t", &BTreeMap::new());
        let at = |origin, oseq| RecordIdentity {
            origin,
            oseq,
            gen: 0,
            epoch: 0,
            hlc: oseq << 16,
        };
        let a = resident_key(RecordType::Label, 0, &label, &at(5, 1)).unwrap();
        let b = resident_key(RecordType::Label, 0, &label, &at(5, 2)).unwrap();
        let c = resident_key(RecordType::Label, 0, &label, &at(6, 1)).unwrap();
        assert!(a != b && a != c && b != c);
        assert_eq!((a.id, a.sub, a.origin), (9, 1, 5));
        let none = RecordIdentity::NONE;
        assert_eq!(
            resident_key(RecordType::Label, 0, &label, &none),
            resident_key(RecordType::Label, 0, &label, &none)
        );
    }

    #[test]
    fn a_turn_tombstone_and_its_timelines_are_different_keys() {
        let whole = TombstonePayload {
            timeline_id: 9,
            turn_index: None,
            turn_writer: 0,
            reason: None,
        }
        .encode();
        let turn = TombstonePayload {
            timeline_id: 9,
            turn_index: Some(0),
            turn_writer: 0,
            reason: None,
        }
        .encode();
        let others = TombstonePayload {
            timeline_id: 9,
            turn_index: Some(0),
            turn_writer: 7,
            reason: None,
        }
        .encode();
        let none = RecordIdentity::NONE;
        let a = resident_key(RecordType::Tombstone, 0, &whole, &none).unwrap();
        let b = resident_key(RecordType::Tombstone, 0, &turn, &none).unwrap();
        let c = resident_key(RecordType::Tombstone, 0, &others, &none).unwrap();
        assert_ne!(a, b);
        assert_ne!(b, c, "another writer's turn 0 is another turn");
        assert_eq!((a.id, a.sub, b.sub), (9, 0, 1));
    }
}
