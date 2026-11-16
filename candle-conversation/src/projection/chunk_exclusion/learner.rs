//! A group's incremental exclusion learner: its state held between folds,
//! loaded from the group's replicated records when it has none, and written
//! back to them after each fold (`docs/incremental_chunk_exclusion.md` §4.2,
//! §4.7).

use std::collections::{BTreeMap, HashMap, HashSet};

use candle::Result;

use super::cut::{cut_of, CompetitionCut};
use super::fold::{full_width, sample_of, FoldCounts, LearnerState, SampleQuestion};
use super::gallery::{Competition, GalleryStream, GalleryView};
use super::scan_jobs::ScanShape;
use crate::persistence::record::CustomObjectPayload;
use crate::persistence::streams::StreamId;
use crate::projection::ids::GroupId;
use crate::projection::Conversation;
use crate::provenance::chunk_exclusion::records::{
    from_json, group_key, question_key, question_of_key, questions_prefix, to_json, CutRecord,
    QuestionRecord,
};
use crate::provenance::chunk_exclusion::sample::sample_hash;
use crate::provenance::gallery_arena::{BeliefArena, GroupSet, Keep};
use crate::provenance::FOLD_GROUPS;

/// Question records written in one commit.
const RECORDS_PER_COMMIT: usize = 256;

/// A question's record is written again once its competition's gallery count
/// has moved by at least this share since it was written — between it and
/// twice it, fixed per question ([`drifted`]). A record's sums are read only
/// by a member taking over, and normalise its votes from there.
const DRIFT: f64 = 0.05;

/// What one fold did, and the cut it left.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct FoldReport {
    pub counts: FoldCounts,
    /// Sample questions held after the fold.
    pub sample: usize,
    /// Each competition's gallery count.
    pub n: [u64; 2],
    /// Entries left waiting on a rescan, and entries in all.
    pub dirty: (usize, usize),
    /// Question records written and tombstoned.
    pub written: usize,
    pub tombstoned: usize,
    pub content: CompetitionCut,
    pub calibration: CompetitionCut,
    /// Whether the state was read from the group's records first.
    pub loaded: bool,
}

/// One group's learner. See the module documentation.
pub struct ExclusionLearner {
    group: GroupId,
    state: Option<LearnerState>,
}

impl ExclusionLearner {
    pub fn new(group: GroupId) -> Self {
        Self { group, state: None }
    }

    pub fn group(&self) -> GroupId {
        self.group
    }

    /// Fold what changed in `conv`'s galleries for the group into the
    /// learner, scanning on `arena` — the content competition in `content`'s
    /// fold groups and kept tokens, the calibration one in `calibration`'s —
    /// and write the records that moved and the group's cut.
    pub fn fold(
        &mut self,
        conv: &Conversation,
        arena: &dyn BeliefArena,
        content: (GroupSet, Keep),
        calibration: (GroupSet, Keep),
    ) -> Result<FoldReport> {
        let view = {
            let sub = conv.inner.read();
            GalleryView::of(&sub, self.group)
        };
        // Another member that held the lease since this one last folded wrote
        // a cursor past this state's: what it folded is in the records, not
        // here.
        let recorded = conv
            .shared()
            .custom_object(&group_key(self.group.raw()))
            .and_then(|o| from_json(&o.blob))
            .and_then(|b| CutRecord::decode(&b))
            .map_or(0, |r| r.cursor);
        if self.state.as_ref().is_some_and(|s| s.cursor < recorded) {
            self.state = None;
        }
        let loaded = self.state.is_none();
        let mut state = match self.state.take() {
            Some(s) => s,
            None => load(conv, self.group, &view),
        };
        let shapes = [
            ScanShape {
                groups: content.0,
                keep: content.1,
            },
            ScanShape {
                groups: calibration.0,
                keep: calibration.1,
            },
        ];
        // A fold cut short leaves the learner with no state, so the next one
        // starts again from the records rather than from a half-folded one.
        let (counts, left) = state.fold(&view, arena, shapes)?;
        let sample: HashSet<StreamId> = state.sample.keys().copied().collect();
        let calibration_now: Vec<&GalleryStream> = view
            .calibration
            .iter()
            .filter(|s| !sample.contains(&s.sid))
            .collect();
        let content_now: Vec<&GalleryStream> = view.content.iter().collect();
        let content_cut = cut_of(&mut state, Competition::Content, &content_now);
        let calibration_cut = cut_of(&mut state, Competition::Calibration, &calibration_now);
        let (written, tombstoned) = write(
            conv,
            self.group,
            &mut state,
            &left,
            &content_cut,
            &calibration_cut,
        )?;
        let report = FoldReport {
            counts,
            sample: state.sample.len(),
            n: state.n,
            dirty: state.dirty(),
            written,
            tombstoned,
            content: content_cut,
            calibration: calibration_cut,
            loaded,
        };
        self.state = Some(state);
        Ok(report)
    }
}

/// The learner's state as `group`'s records hold it: its sample questions'
/// entries, the gallery counts and the cursor, every stream of `view` at or
/// below the cursor taken as folded. A group with no record starts empty, and
/// its first fold is the whole gallery.
fn load(conv: &Conversation, group: GroupId, view: &GalleryView) -> LearnerState {
    let shared = conv.shared();
    let Some(cut) = shared
        .custom_object(&group_key(group.raw()))
        .and_then(|o| from_json(&o.blob))
        .and_then(|b| CutRecord::decode(&b))
    else {
        return LearnerState::default();
    };
    let width = view.width().unwrap_or(0);
    let questions: HashMap<StreamId, _> = view.questions.iter().map(|q| (q.sid, q)).collect();
    let mut sample: BTreeMap<StreamId, SampleQuestion> = BTreeMap::new();
    let mut orphans: Vec<StreamId> = Vec::new();
    for (obj, _) in shared.custom_objects_under(&questions_prefix(group.raw())) {
        let Some(sid) = question_of_key(&obj.key) else {
            continue;
        };
        // A record whose question is gone, or that no longer fits its
        // question's tokens, holds nothing a fold can use: it goes.
        let held = questions.get(&sid).and_then(|q| {
            let rec = from_json(&obj.blob).and_then(|b| QuestionRecord::decode(&b))?;
            let tokens = full_width(q.sig.decode(), width);
            let want = tokens.len() * FOLD_GROUPS;
            rec.entries
                .iter()
                .all(|e| e.len() == want && want > 0)
                .then_some((rec, tokens))
        });
        let Some((rec, tokens)) = held else {
            orphans.push(sid);
            continue;
        };
        sample.insert(
            sid,
            SampleQuestion {
                owner: rec.owner,
                tokens,
                entries: rec.entries,
                written_n: rec.n,
                changed: false,
            },
        );
    }
    // The calibration gallery the records were folded against held no
    // sample question: neither the ones with a record, nor today's sample's
    // others, which a fold takes as entering rather than as removed.
    let chosen = sample_of(view, width);
    let mut folded: [HashMap<StreamId, GalleryStream>; 2] = [HashMap::new(), HashMap::new()];
    for comp in Competition::BOTH {
        for s in view.streams(comp) {
            let in_sample = comp == Competition::Calibration
                && (sample.contains_key(&s.sid) || chosen.contains(&s.sid));
            if s.hlc <= cut.cursor && !in_sample {
                folded[comp.index()].insert(s.sid, s.clone());
            }
        }
    }
    LearnerState {
        sample,
        n: cut.n,
        folded,
        cursor: cut.cursor,
        orphans,
        written_cut: Some(cut),
    }
}

/// Whether question `sid`'s record has drifted far enough to write again: its
/// gallery count has moved by between [`DRIFT`] and twice it, the share fixed
/// per question by its hash so the sample's records come due a few at a time
/// rather than all on one fold.
fn drifted(sid: StreamId, written: u64, now: u64) -> bool {
    if now == 0 {
        return false;
    }
    let spread = 1.0 + (sample_hash(sid) % 1_000) as f64 / 1_000.0;
    (now as f64 - written as f64).abs() >= DRIFT * spread * now as f64
}

/// Write what the fold moved: the group's cut, cursor and counts when they
/// changed, then every sample question whose entries changed or whose sums
/// have drifted ([`drifted`]), and a tombstone for every question that left
/// and every orphaned record load found. Returns the records written and
/// tombstoned.
///
/// **The group record goes first.** A fold that dies between the writes leaves
/// a cursor with some question records behind it; a reload takes each missing
/// question as entering the sample and scans it whole. Written last, a death
/// before it would leave question records with no group record — a reload
/// that starts from nothing beside records nothing tombstones.
fn write(
    conv: &Conversation,
    group: GroupId,
    state: &mut LearnerState,
    left: &[StreamId],
    content: &CompetitionCut,
    calibration: &CompetitionCut,
) -> Result<(usize, usize)> {
    let shared = conv.shared();
    let n = state.n;
    let cut = CutRecord {
        cursor: state.cursor,
        n,
        cut: content
            .cut
            .iter()
            .chain(&calibration.cut)
            .copied()
            .collect(),
    };
    if state.written_cut.as_ref() != Some(&cut) {
        shared
            .put_custom_object(CustomObjectPayload::new(
                group_key(group.raw()),
                to_json(&cut.encode()),
            ))
            .map_err(|e| candle::Error::Msg(format!("exclusion cut record: {e}")))?;
        state.written_cut = Some(cut);
    }
    let mut objs: Vec<CustomObjectPayload> = Vec::new();
    for (sid, q) in state.sample.iter_mut() {
        let moved = (0..2).any(|c| drifted(*sid, q.written_n[c], n[c]));
        if !q.changed && !moved {
            continue;
        }
        let rec = QuestionRecord {
            owner: q.owner,
            n,
            entries: q.entries.clone(),
        };
        objs.push(CustomObjectPayload::new(
            question_key(group.raw(), *sid),
            to_json(&rec.encode()),
        ));
        q.written_n = n;
        q.changed = false;
    }
    let written = objs.len();
    let gone: Vec<StreamId> = left.iter().chain(&state.orphans).copied().collect();
    objs.extend(
        gone.iter()
            .map(|&sid| CustomObjectPayload::ceased(question_key(group.raw(), sid))),
    );
    for batch in objs.chunks(RECORDS_PER_COMMIT) {
        shared
            .put_custom_objects_under(batch.to_vec(), 0)
            .map_err(|e| candle::Error::Msg(format!("exclusion records: {e}")))?;
    }
    state.orphans.clear();
    Ok((written, gone.len()))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// **A record comes due between 5% and 10% of drift, fixed per
    /// question**: under 5% never, at 10% always, and between them as its
    /// hash places it — so not every question at once.
    #[test]
    fn a_record_comes_due_between_five_and_ten_percent() {
        for sid in (0..200).map(StreamId) {
            assert!(!drifted(sid, 1_000, 1_049));
            assert!(drifted(sid, 1_000, 1_112));
            assert!(!drifted(sid, 1_000, 0), "an empty gallery is never due");
        }
        let due_at_7: usize = (0..200)
            .map(StreamId)
            .filter(|&s| drifted(s, 1_000, 1_076))
            .count();
        assert!(due_at_7 > 0 && due_at_7 < 200, "{due_at_7} of 200 due at 7%");
    }
}
