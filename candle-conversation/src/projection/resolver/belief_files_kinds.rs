//! Scores the knowledge base's sections, kind by kind
//! (`docs/provenance_system_collection.md` §4,
//! `docs/placed_section_layout.md` §3.5c).
//!
//! Each kind's sections are assembled from their members
//! ([`super::super::belief_members`]) and scored as a turn group's files are:
//! one batched launch with the pinned question window as a second probe, each
//! section one case (`cases: section`), read against its kind — `kind_z` over
//! the sections of files of its own extension — or the kind's learned
//! background, the tail and the question max-fused after normalization. The
//! result is each section's fresh score with its file, what the knowledge
//! base's working set builds its momentum from.

use super::super::belief_members::{assemble_member_scans, section_of};
use super::super::files_kinds::kind_records;
use super::*;
use crate::files::{FileKey, FileRecord, FreshSection, SectionKey, FILES_PREFIX};
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

    /// Score every section of each of `kinds` against `probe` — and
    /// `probe_q`, the question window, for a kind that pins it — and return
    /// each kind's sections with their files and fresh scores.
    pub fn score_belief_files(
        &self,
        kinds: &[&GroupSchema],
        probe: &[WideQSig],
        probe_q: Option<&[WideQSig]>,
        observe: Observe<'_>,
        arena: Option<&dyn BeliefArena>,
    ) -> Vec<(GroupId, Vec<FreshSection>)> {
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
            // Every section of the kind's files, each its own case, with its
            // file and path.
            let mut members: Vec<(SectionKey, StreamId)> = Vec::new();
            let mut of: HashMap<SectionKey, (FileKey, &str)> = HashMap::new();
            for r in &held {
                for (i, c) in r.chunks.iter().enumerate() {
                    let key = r.section_key(i);
                    members.push((key, section_stream_id(c.member.address)));
                    of.insert(key, (r.key(), r.path.as_str()));
                }
            }
            let sections = {
                let sub = self.inner.read();
                assemble_member_scans(&sub, &members)
            };
            if sections.is_empty() {
                continue;
            }
            let scores = self.score_kind(kind, &sections, &of, probe, probe_q, observe, arena);
            out.push((kind.id, scores));
        }
        out
    }

    /// One kind's assembled `sections` scored and normalized.
    #[allow(clippy::too_many_arguments)]
    fn score_kind(
        &self,
        kind: &GroupSchema,
        files: &[FileScan],
        of: &HashMap<SectionKey, (FileKey, &str)>,
        probe: &[WideQSig],
        probe_q: Option<&[WideQSig]>,
        observe: Observe<'_>,
        arena: Option<&dyn BeliefArena>,
    ) -> Vec<FreshSection> {
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
                of.get(&section_of(f.timeline))
                    .map_or(String::new(), |(_, p)| path_kind(p).to_string())
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
            .filter_map(|(i, f)| {
                let t = tail_normed.get(i).copied().unwrap_or(0.0);
                let q = question_normed
                    .as_ref()
                    .and_then(|q| q.get(i).copied())
                    .unwrap_or(0.0);
                let section = section_of(f.timeline);
                let &(file, _) = of.get(&section)?;
                Some((section, file, t.max(q)))
            })
            .collect()
    }
}
