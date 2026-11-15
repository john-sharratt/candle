//! The one-time rewrite of a store's QSA index keys from `F32` to Q8_0
//! (`docs/qsa_index_q8.md` §8), run against a stopped daemon's substrate by
//! `substrate_inspect migrate-index-q8`.
//!
//! Completed index keys are persisted only inside the model's aux container,
//! which three records carry: a conversation's `Snapshot` and a prompt
//! branch's `BranchCheckpoint` (their `aux`), and a `TurnIndexPage` — a turn's
//! framed list of pages, or a prompt section's single page. A build reads the
//! Q8_0 container only, so every record still holding the `F32` one is
//! rewritten here: decoded, its completed keys quantized by the host
//! definition, and appended again.
//!
//! **Under a new stamp.** Of two records with one identity a reload keeps the
//! one it meets first, so a rewrite that kept the old stamp would be dropped.
//! The rewritten record is a later write by this node and supersedes the one
//! it replaces, whose bytes maintenance then reclaims.
//!
//! **Only what this node wrote.** A page belongs to the member that wrote its
//! stream, and a record this node stamps is a record of this node's stream:
//! it cannot supersede another member's page. Those are counted and left —
//! they arrive rewritten from the member that wrote them, once it has run
//! this.

use std::path::Path;

use candle_transformers::models::qwen4exp::key_q8::Fidelity;
use candle_transformers::models::qwen4exp::paged_index::{migrate_aux, AuxMigration};

use super::record::{BranchCheckpointPayload, RecordType, SnapshotPayload};
use super::streams::StreamId;
use super::{Result, SubstratePersistence};
use crate::index_pages;
use crate::substrate::{IndexPage, Substrate};

/// What one kind of record held.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Tally {
    /// Records already Q8_0, or holding no index at all.
    pub current: usize,
    /// Records holding an `F32` index that this node can rewrite.
    pub f32: usize,
    /// Their payload bytes as they stand.
    pub f32_bytes: u64,
    /// Their payload bytes once rewritten.
    pub q8_bytes: u64,
    /// Records holding an `F32` index another member wrote.
    pub foreign: usize,
    /// Records whose payload is neither container.
    pub unreadable: usize,
}

/// What a store holds, or what a migration did to it.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Report {
    /// `TurnIndexPage` records: a turn's pages or a section's.
    pub pages: Tally,
    /// `Snapshot` and `BranchCheckpoint` records.
    pub recurrent: Tally,
    /// How closely the Q8_0 keys hold the `F32` ones, over every key of every
    /// record this node can rewrite.
    pub fidelity: Fidelity,
    /// The streams whose payload could not be read as an index, with why.
    pub unreadable: Vec<(u64, String)>,
}

impl Report {
    /// `F32` records this node can rewrite.
    pub fn to_rewrite(&self) -> usize {
        self.pages.f32 + self.recurrent.f32
    }
}

/// A payload read as an index.
enum Read {
    Current,
    /// The payload rewritten, and how closely its keys are held.
    Migrated(Vec<u8>, Fidelity),
}

/// One aux container read as an index.
fn read_aux(aux: &[u8]) -> std::result::Result<Read, String> {
    match migrate_aux(aux).map_err(|e| e.to_string())? {
        AuxMigration::Current => Ok(Read::Current),
        AuxMigration::Migrated { bytes, fidelity } => Ok(Read::Migrated(bytes, fidelity)),
    }
}

/// A `TurnIndexPage` payload as an index: one aux container (a section's
/// page), or a framed list of them (a turn's pages). Told apart by structure —
/// each is strict about accounting for every byte — and refused when it reads
/// as both or as neither.
fn read_page(payload: &[u8]) -> std::result::Result<Read, String> {
    let whole = read_aux(payload);
    let framed = index_pages::decode(payload)
        .map_err(|e| e.to_string())
        .and_then(|pages| {
            let mut fidelity = Fidelity::default();
            let mut any = false;
            let mut out = Vec::with_capacity(pages.len());
            for (tokens, page) in pages {
                match read_aux(page)? {
                    Read::Current => out.push((tokens, page.to_vec())),
                    Read::Migrated(new, held) => {
                        any = true;
                        fidelity.merge(&held);
                        out.push((tokens, new));
                    }
                }
            }
            Ok(if any {
                Read::Migrated(index_pages::encode(&out), fidelity)
            } else {
                Read::Current
            })
        });
    match (whole, framed) {
        (Ok(_), Ok(_)) => Err("reads both as one page and as a framed list".to_string()),
        (Ok(read), Err(_)) | (Err(_), Ok(read)) => Ok(read),
        (Err(a), Err(b)) => Err(format!("as one page: {a}; as a framed list: {b}")),
    }
}

/// A recurrent record's payload, rewritten when its aux holds an `F32` index.
fn read_recurrent(payload: &[u8]) -> std::result::Result<(RecordType, Read), String> {
    if payload.is_empty() {
        // A retraction: no state, no index.
        return Ok((RecordType::Snapshot, Read::Current));
    }
    // A record with no aux carries no index.
    let aux_of = |aux: &[u8]| match aux.is_empty() {
        true => Ok(Read::Current),
        false => read_aux(aux),
    };
    if let Ok(mut branch) = BranchCheckpointPayload::decode(payload) {
        return Ok((
            RecordType::BranchCheckpoint,
            match aux_of(&branch.aux)? {
                Read::Current => Read::Current,
                Read::Migrated(aux, held) => {
                    branch.aux = aux;
                    Read::Migrated(branch.encode(), held)
                }
            },
        ));
    }
    let mut snapshot = SnapshotPayload::decode(payload).map_err(|e| e.to_string())?;
    Ok((
        RecordType::Snapshot,
        match aux_of(&snapshot.aux)? {
            Read::Current => Read::Current,
            Read::Migrated(aux, held) => {
                snapshot.aux = aux;
                Read::Migrated(snapshot.encode(), held)
            }
        },
    ))
}

/// Count the store under `workspace` (the directory holding `substrate/`)
/// without writing to it.
pub fn survey(workspace: &Path) -> Result<Report> {
    let mut substrate = Substrate::new();
    let mut persistence =
        SubstratePersistence::open_in_with_substrate_read_only(workspace, &mut substrate)?;
    walk(&mut persistence, &substrate, false)
}

/// Rewrite every `F32` index this node wrote in the store under `workspace`,
/// and commit. The report is of what was found; [`survey`] afterwards says
/// what is left.
pub fn migrate(workspace: &Path) -> Result<Report> {
    let mut substrate = Substrate::new();
    let mut persistence = SubstratePersistence::open_in_with_substrate(workspace, &mut substrate)?;
    let report = walk(&mut persistence, &substrate, true)?;
    persistence.commit()?;
    Ok(report)
}

fn walk(
    persistence: &mut SubstratePersistence,
    substrate: &Substrate,
    write: bool,
) -> Result<Report> {
    let mut report = Report::default();
    let node = persistence.node();

    let pages: Vec<_> = substrate
        .all_physical_streams()
        .filter_map(|(key, entry)| match &entry.index_page {
            Some(IndexPage::Stored(loc)) => Some((key, *loc, entry.ids.index_page)),
            _ => None,
        })
        .collect();
    for (key, loc, id) in pages {
        let payload = persistence.read_record_payload(&loc)?;
        match read_page(&payload) {
            Ok(Read::Current) => report.pages.current += 1,
            Ok(Read::Migrated(..)) if key.writer != node => report.pages.foreign += 1,
            Ok(Read::Migrated(new, held)) => {
                report.pages.f32 += 1;
                report.pages.f32_bytes += payload.len() as u64;
                report.pages.q8_bytes += new.len() as u64;
                report.fidelity.merge(&held);
                if write {
                    // The page's own generation, so a section's page is still
                    // the section's on replay.
                    let stamp = persistence.mint_under(id.epoch);
                    persistence.append_turn_index_page(key.stream_id, &new, stamp)?;
                }
            }
            Err(why) => {
                report.pages.unreadable += 1;
                report.unreadable.push((key.stream_id.0, why));
            }
        }
    }

    let tails: Vec<(StreamId, _)> = persistence.recurrent_tails();
    for (stream, loc) in tails {
        let payload = persistence.read_record_payload(&loc)?;
        match read_recurrent(&payload) {
            Ok((_, Read::Current)) => report.recurrent.current += 1,
            Ok((rt, Read::Migrated(new, held))) => {
                report.recurrent.f32 += 1;
                report.recurrent.f32_bytes += payload.len() as u64;
                report.recurrent.q8_bytes += new.len() as u64;
                report.fidelity.merge(&held);
                if write {
                    match rt {
                        RecordType::BranchCheckpoint => {
                            persistence.write_branch_checkpoint(stream, &[&new])?
                        }
                        _ => persistence.write_snapshot(stream, &[&new])?,
                    };
                }
            }
            Err(why) => {
                report.recurrent.unreadable += 1;
                report.unreadable.push((stream.0, why));
            }
        }
    }
    Ok(report)
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use candle_transformers::models::qwen4exp::key_q8::Q8Keys;
    use candle_transformers::models::qwen4exp::paged_index::{
        decode_aux, encode_aux, SealedIndex, F32_INDEX_VERSION,
    };

    use super::super::content_hash::{snapshot_stream_id, ContentHash};
    use super::super::streams::{ContentAddress, SectionDecl, StreamDecl};
    use super::*;

    const DIM: usize = 32;

    fn tmp_dir(tag: &str) -> PathBuf {
        let mut p = std::env::temp_dir();
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        p.push(format!("index_q8_{tag}_{nanos}"));
        std::fs::create_dir_all(&p).unwrap();
        p
    }

    fn values(n: usize, seed: u32) -> Vec<f32> {
        (0..n)
            .map(|i| (((i as u32).wrapping_mul(2_654_435_761) ^ seed) % 2001) as f32 / 1000.0 - 1.0)
            .collect()
    }

    /// An aux container as version 3 wrote it: one layer of `rows` `F32` keys
    /// and `open` raw rows behind the PLE bytes.
    fn f32_aux(ple: &[u8], rows: &[f32], last_cells: u32, open: &[f32]) -> Vec<u8> {
        let mut out = Vec::new();
        out.extend_from_slice(&F32_INDEX_VERSION.to_le_bytes());
        out.extend_from_slice(&(ple.len() as u32).to_le_bytes());
        out.extend_from_slice(ple);
        out.extend_from_slice(&1u32.to_le_bytes());
        let len = 16 + (rows.len() + open.len()) * 4;
        out.extend_from_slice(&(len as u32).to_le_bytes());
        out.extend_from_slice(&((rows.len() / DIM) as u32).to_le_bytes());
        out.extend_from_slice(&(DIM as u32).to_le_bytes());
        out.extend_from_slice(&last_cells.to_le_bytes());
        out.extend_from_slice(&((open.len() / DIM) as u32).to_le_bytes());
        for v in rows.iter().chain(open) {
            out.extend_from_slice(&v.to_le_bytes());
        }
        out
    }

    /// The same container as this build writes it.
    fn q8_aux(ple: &[u8], rows: &[f32], last_cells: usize, open: &[f32]) -> Vec<u8> {
        encode_aux(
            ple,
            &[SealedIndex {
                keys: Q8Keys::quantize(rows, DIM),
                last_cells,
                open: open.to_vec(),
            }],
        )
        .unwrap()
    }

    fn section(lo: u64) -> StreamDecl {
        StreamDecl::PromptSection(SectionDecl {
            address: ContentAddress {
                prefix_hash: ContentHash { lo, hi: 0x22 },
                section_hash: ContentHash { lo: 0x33, hi: lo },
            },
            debug_name: format!("section_{lo}"),
        })
    }

    /// **A store's `F32` indexes are rewritten as exactly the Q8_0 ones, and a
    /// reload reads the rewritten records.** A section's single page, a turn's
    /// framed list of three — the count a version word of 3 would be mistaken
    /// for — a page already Q8_0, and a snapshot whose aux carries the index:
    /// the survey counts them, the migration rewrites them, and the store read
    /// back holds the bytes this build writes, with nothing `F32` left.
    #[test]
    fn a_store_is_rewritten_and_reads_back_q8() {
        let dir = tmp_dir("rewrite");
        let (one, three, done) = (section(0x11), section(0x12), section(0x13));
        let rows = |n: usize, seed: u32| values(n * DIM, seed);
        let single = f32_aux(&[], &rows(5, 1), 2, &[]);
        let framed: Vec<(usize, Vec<u8>)> = (0..3)
            .map(|i| {
                (
                    4 * (i + 2),
                    f32_aux(&[], &rows(i + 2, 10 + i as u32), 4, &[]),
                )
            })
            .collect();
        let current = q8_aux(&[], &rows(3, 30), 4, &[]);
        let timeline = 77u64;
        let snapshot = SnapshotPayload {
            timeline_id: timeline,
            turn_index: 4,
            schedule_hash: 0xABCD,
            layers: Vec::new(),
            aux: f32_aux(&[7, 8, 9], &rows(6, 40), 4, &values(2 * DIM, 41)),
        };

        {
            let mut sp = SubstratePersistence::open_in(&dir).unwrap();
            for (decl, payload) in [
                (&one, single.clone()),
                (&three, index_pages::encode(&framed)),
                (&done, current.clone()),
            ] {
                sp.declare_stream(decl).unwrap();
                let stream = decl.stream_id();
                let id = sp.mint_for(RecordType::TurnIndexPage, stream);
                sp.append_turn_index_page(stream, &payload, id).unwrap();
            }
            sp.write_snapshot(snapshot_stream_id(timeline), &[&snapshot.encode()])
                .unwrap();
            sp.commit().unwrap();
        }

        let before = survey(&dir).unwrap();
        assert_eq!(
            (before.pages.f32, before.pages.current),
            (2, 1),
            "{before:?}"
        );
        assert_eq!((before.recurrent.f32, before.recurrent.current), (1, 0));
        assert_eq!(before.to_rewrite(), 3);
        assert!(before.unreadable.is_empty(), "{:?}", before.unreadable);
        assert!(before.pages.q8_bytes * 2 < before.pages.f32_bytes);
        // Every key of every record to rewrite: 5, then 2 + 3 + 4, then 6.
        assert_eq!(before.fidelity.keys, 5 + 9 + 6);
        assert!(before.fidelity.snr_db() > 35.0, "{:?}", before.fidelity);

        let did = migrate(&dir).unwrap();
        assert_eq!(
            did, before,
            "the migration rewrites what the survey counted"
        );

        let after = survey(&dir).unwrap();
        assert_eq!(after.to_rewrite(), 0, "{after:?}");
        assert_eq!((after.pages.current, after.recurrent.current), (3, 1));

        // The records a reload reads are the ones this build writes.
        let mut substrate = Substrate::new();
        let mut sp = SubstratePersistence::open_in_with_substrate(&dir, &mut substrate).unwrap();
        let page = |decl: &StreamDecl, sp: &mut SubstratePersistence| {
            let loc = substrate
                .index_page_of(decl.stream_id())
                .and_then(|p| p.stored())
                .expect("the page is in the log");
            sp.read_record_payload(&loc).unwrap()
        };
        assert_eq!(page(&one, &mut sp), q8_aux(&[], &rows(5, 1), 2, &[]));
        let want: Vec<(usize, Vec<u8>)> = (0..3)
            .map(|i| {
                (
                    4 * (i + 2),
                    q8_aux(&[], &rows(i + 2, 10 + i as u32), 4, &[]),
                )
            })
            .collect();
        assert_eq!(page(&three, &mut sp), index_pages::encode(&want));
        assert_eq!(page(&done, &mut sp), current, "a Q8_0 page is left alone");

        let tails = sp.recurrent_tails();
        assert_eq!(tails.len(), 1);
        let back = SnapshotPayload::decode(&sp.read_record_payload(&tails[0].1).unwrap()).unwrap();
        assert_eq!(
            (back.timeline_id, back.turn_index, back.schedule_hash),
            (timeline, 4, 0xABCD)
        );
        assert_eq!(
            back.aux,
            q8_aux(&[7, 8, 9], &rows(6, 40), 4, &values(2 * DIM, 41))
        );
        let (ple, layers) = decode_aux(&back.aux).unwrap();
        assert_eq!(ple, vec![7, 8, 9]);
        assert_eq!(layers[0].n_open(), 2);
        drop(sp);

        // And running it again finds nothing to do.
        assert_eq!(migrate(&dir).unwrap().to_rewrite(), 0);
        std::fs::remove_dir_all(&dir).ok();
    }

    /// A payload that is neither container is reported and left as it stands.
    #[test]
    fn a_payload_that_is_not_an_index_is_reported_and_left() {
        let dir = tmp_dir("unreadable");
        let decl = section(0x21);
        let stream = decl.stream_id();
        let junk: Vec<u8> = (0..97u32).map(|i| (i * 7 % 251) as u8).collect();
        {
            let mut sp = SubstratePersistence::open_in(&dir).unwrap();
            sp.declare_stream(&decl).unwrap();
            let id = sp.mint_for(RecordType::TurnIndexPage, stream);
            sp.append_turn_index_page(stream, &junk, id).unwrap();
            sp.commit().unwrap();
        }
        let did = migrate(&dir).unwrap();
        assert_eq!((did.pages.unreadable, did.to_rewrite()), (1, 0));
        assert_eq!(did.unreadable.len(), 1);
        assert_eq!(did.unreadable[0].0, stream.0);

        let mut substrate = Substrate::new();
        let mut sp = SubstratePersistence::open_in_with_substrate(&dir, &mut substrate).unwrap();
        let loc = substrate
            .index_page_of(stream)
            .and_then(|p| p.stored())
            .unwrap();
        assert_eq!(sp.read_record_payload(&loc).unwrap(), junk);
        drop(sp);
        std::fs::remove_dir_all(&dir).ok();
    }
}
