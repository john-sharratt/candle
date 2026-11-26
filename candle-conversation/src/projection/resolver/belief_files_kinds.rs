//! Scores the knowledge base's files, kind by kind
//! (`docs/provenance_system_collection.md` §4).
//!
//! Each kind's files are assembled from their members
//! ([`super::super::belief_members`]) and scored as a turn group's files are:
//! one batched launch with the pinned question window as a second probe, each
//! file one case (`cases: file`), read against its kind — `kind_z` over the
//! files of its own extension — or the kind's learned background, the tail and
//! the question max-fused after normalization. The result is each file's
//! fresh score, what the knowledge base's working set builds its momentum
//! from.

use super::super::belief_members::{assemble_member_scans, file_of};
use super::super::files_kinds::kind_records;
use super::*;
use crate::files::{FileKey, FileRecord, FILES_PREFIX};
use crate::persistence::content_hash::section_stream_id;

impl Conversation {
    /// Every live file record of the knowledge base.
    pub fn files_records(&self) -> Vec<FileRecord> {
        self.inner
            .read()
            .custom_objects_under(FILES_PREFIX)
            .into_iter()
            .filter_map(|(obj, _)| FileRecord::from_object(&obj))
            .collect()
    }

    /// Score every file of each of `kinds` against `probe` — and `probe_q`,
    /// the question window, for a kind that pins it — and return each kind's
    /// files with their fresh scores.
    pub fn score_belief_files(
        &self,
        kinds: &[&GroupSchema],
        probe: &[WideQSig],
        probe_q: Option<&[WideQSig]>,
        observe: Observe<'_>,
        arena: Option<&dyn BeliefArena>,
    ) -> Vec<(GroupId, Vec<(FileKey, f32)>)> {
        let mut out = Vec::new();
        if probe.is_empty() {
            return out;
        }
        let records = self.files_records();
        for kind in kinds {
            let held = kind_records(kind, kinds, &records);
            if held.is_empty() {
                continue;
            }
            let members: Vec<(FileKey, Vec<StreamId>)> = held
                .iter()
                .map(|r| {
                    (
                        r.key(),
                        r.chunks
                            .iter()
                            .map(|c| section_stream_id(c.member.address))
                            .collect(),
                    )
                })
                .collect();
            let paths: HashMap<FileKey, &str> =
                held.iter().map(|r| (r.key(), r.path.as_str())).collect();
            let files = {
                let sub = self.inner.read();
                assemble_member_scans(&sub, &members)
            };
            if files.is_empty() {
                continue;
            }
            let scores = self.score_kind(kind, &files, &paths, probe, probe_q, observe, arena);
            out.push((kind.id, scores));
        }
        out
    }

    /// One kind's assembled `files` scored and normalized.
    #[allow(clippy::too_many_arguments)]
    fn score_kind(
        &self,
        kind: &GroupSchema,
        files: &[FileScan],
        paths: &HashMap<FileKey, &str>,
        probe: &[WideQSig],
        probe_q: Option<&[WideQSig]>,
        observe: Observe<'_>,
        arena: Option<&dyn BeliefArena>,
    ) -> Vec<(FileKey, f32)> {
        let plans = ScanPlans::default();
        let question = match probe_q {
            Some(q) if kind.policy.scan.question_pin && !q.is_empty() => Some(q),
            _ => None,
        };
        let mut scanned = match question {
            Some(q) => scan_file_scans(files, kind, &[probe, q], arena, Holder::Shared, &plans),
            None => scan_file_scans(files, kind, &[probe], arena, Holder::Shared, &plans),
        };
        let question_scan = question.and_then(|_| scanned.pop());
        let tail_scan = scanned.pop().unwrap_or_default();
        let raw = |per_file: &[(Vec<f32>, Vec<f32>)]| -> Vec<f32> {
            per_file
                .iter()
                .map(|(fused, _)| fused.first().copied().unwrap_or(0.0))
                .collect()
        };
        let tail_raw = raw(&tail_scan);
        let question_raw = question_scan.as_deref().map(raw);
        let by_kind = kind.policy.scan.normalize == ScanNorm::KindZ;
        let extensions: Vec<String> = files
            .iter()
            .map(|f| {
                paths
                    .get(&file_of(f.timeline))
                    .map_or(String::new(), |p| path_kind(p).to_string())
            })
            .collect();
        let pool_scope = ScopeKey::pool(kind.id.raw() as u64);
        let pairs = |scores: &[f32]| -> Vec<(ChildKey, f32)> {
            files
                .iter()
                .zip(scores)
                .map(|(f, &s)| (ChildKey::timeline(f.timeline.raw()), s))
                .collect()
        };
        let normalize = |scores: &[f32], t: usize| -> Vec<f32> {
            if by_kind {
                kind_z_band(scores, &extensions)
            } else {
                self.normalization
                    .normalize(&pool_scope, &pairs(scores), t)
                    .into_iter()
                    .map(|(_, s)| s)
                    .collect()
            }
        };
        let t_tail = probe.len();
        let t_q = probe_q.map_or(1, |q| q.len());
        let tail_normed = normalize(&tail_raw, t_tail);
        let question_normed = question_raw.as_ref().map(|q| normalize(q, t_q));
        // A kind read against its extensions keeps no background to teach.
        if let (false, true, Some(source)) = (
            by_kind,
            observe.teaches(&kind.policy.tags),
            observe.source(),
        ) {
            self.normalization
                .observe(&pool_scope, source, &pairs(&tail_raw), None, t_tail);
        }
        files
            .iter()
            .enumerate()
            .map(|(i, f)| {
                let t = tail_normed.get(i).copied().unwrap_or(0.0);
                let q = question_normed
                    .as_ref()
                    .and_then(|q| q.get(i).copied())
                    .unwrap_or(0.0);
                (file_of(f.timeline), t.max(q))
            })
            .collect()
    }
}
