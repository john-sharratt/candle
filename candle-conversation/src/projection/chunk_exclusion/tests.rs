//! The learner folded against the host reference arena: folding a gallery a
//! part at a time equals folding it whole, a removal equals never having
//! folded, the cursor and the sample move as the design says, and a hub chunk
//! is cut (`docs/incremental_chunk_exclusion.md` §7).

use std::collections::HashMap;
use std::sync::Arc;

use super::cut::cut_of;
use super::fold::{sample_of, LearnerState};
use super::gallery::{Competition, GalleryStream, GalleryView, Question};
use super::host_arena::HostArena;
use super::learner::ExclusionLearner;
use super::scan_jobs::ScanShape;
use crate::persistence::content_hash::turn_stream_id;
use crate::persistence::record::CustomObjectPayload;
use crate::persistence::streams::StreamId;
use crate::projection::{Conversation, GroupId, LayerId, TimelineId};
use crate::provenance::chunk_exclusion::records::{
    group_key, question_key, questions_prefix, to_json, CutRecord, Entry, QuestionRecord,
};
use crate::provenance::gallery_arena::{GroupSet, Keep};
use crate::provenance::{encode_wide_sigs, SigWindow, WideQSig, FOLD_GROUPS};
use crate::substrate::{Substrate, TurnPartWrite};
use crate::turn::Role;

fn xorshift(state: &mut u64) -> u64 {
    let mut x = *state;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    *state = x;
    x
}

fn rand_sig(state: &mut u64) -> WideQSig {
    WideQSig {
        n_heads: 12,
        words: (0..24).map(|_| xorshift(state)).collect(),
    }
}

fn window(tokens: &[WideQSig]) -> SigWindow {
    SigWindow::new(Arc::from(encode_wide_sigs(tokens))).expect("a stored window")
}

fn stream(sid: u64, file: u64, tokens: &[WideQSig], hlc: u64) -> GalleryStream {
    GalleryStream {
        sid: StreamId(sid),
        file,
        sig: window(tokens),
        windows: vec![(0, tokens.len())],
        hlc,
    }
}

fn shapes() -> [ScanShape; 2] {
    let shape = ScanShape {
        groups: GroupSet::all(FOLD_GROUPS),
        keep: Keep::All,
    };
    [shape, shape]
}

/// Six files of two turns each, and four questions about them — each a
/// question stream in the calibration gallery too.
fn gallery(seed: u64) -> GalleryView {
    let mut rng = seed;
    let mut view = GalleryView::default();
    for f in 0..6u64 {
        for t in 0..2u64 {
            let toks: Vec<WideQSig> = (0..20 + 7 * t as usize)
                .map(|_| rand_sig(&mut rng))
                .collect();
            view.content
                .push(stream(1_000 + f * 10 + t, 100 + f, &toks, 10 + f * 2 + t));
        }
    }
    for q in 0..4u64 {
        let toks: Vec<WideQSig> = (0..6).map(|_| rand_sig(&mut rng)).collect();
        let s = stream(5_000 + q, 100 + q, &toks, 50 + q);
        view.questions.push(Question {
            sid: s.sid,
            owner: s.file,
            sig: s.sig.clone(),
            hlc: s.hlc,
        });
        view.calibration.push(s);
    }
    view
}

/// `view` with only the content streams `keep` holds.
fn with_content(view: &GalleryView, keep: impl Fn(&GalleryStream) -> bool) -> GalleryView {
    GalleryView {
        content: view.content.iter().filter(|s| keep(s)).cloned().collect(),
        ..view.clone()
    }
}

fn entries(s: &LearnerState) -> HashMap<StreamId, [Vec<Entry>; 2]> {
    s.sample
        .iter()
        .map(|(sid, q)| (*sid, q.entries.clone()))
        .collect()
}

/// **A question reloaded behind the cursor catches up exactly.** Files
/// stamped after the first fold are folded by a learner that holds the sample
/// throughout; another state reloads one question from its record as the
/// first fold left it, the rest current, and catches it up on the streams above
/// its own cursor alone — ending with the same entries.
#[test]
fn a_question_reloaded_behind_catches_up_exactly() {
    let mut view = gallery(0xCA7C);
    // Files 103.. arrive after the first fold: their stamps are above the
    // questions'.
    for s in &mut view.content {
        if s.file >= 103 {
            s.hlc += 1_000;
        }
    }
    let early = with_content(&view, |s| s.file < 103);
    let arena = HostArena::default();
    let mut held = LearnerState::default();
    held.fold(&early, &arena, shapes()).unwrap();
    let first_cursor = held.cursor;
    let behind_q = *held.sample.keys().next().unwrap();
    let record = held.sample[&behind_q].entries.clone();
    held.fold(&view, &arena, shapes()).unwrap();

    let mut reloaded = held.clone();
    let q = reloaded.sample.get_mut(&behind_q).unwrap();
    q.entries = record;
    q.behind = Some(first_cursor);
    let (counts, _) = reloaded.fold(&view, &arena, shapes()).unwrap();
    assert_eq!(counts.caught_up, 1);
    assert_eq!(counts.appended, 0, "nothing new in the gallery");
    assert_eq!(
        reloaded.sample[&behind_q].entries,
        held.sample[&behind_q].entries
    );
    assert_eq!(reloaded.sample[&behind_q].behind, None);
}

/// **Folding a gallery a part at a time is folding it whole.** One state
/// folds the whole gallery at once; another folds a third of the files, then
/// two thirds, then all: their entries and gallery counts are identical.
#[test]
fn folding_in_parts_is_folding_whole() {
    let view = gallery(0xC0FFEE);
    let arena = HostArena::default();
    let mut whole = LearnerState::default();
    whole.fold(&view, &arena, shapes()).unwrap();

    let mut parts = LearnerState::default();
    for upto in [102u64, 104, 106] {
        let part = with_content(&view, |s| s.file < upto);
        let (counts, _) = parts.fold(&part, &arena, shapes()).unwrap();
        assert_eq!(counts.removed, 0);
    }
    assert_eq!(parts.n, whole.n);
    assert_eq!(entries(&parts), entries(&whole));
    assert_eq!(whole.sample.len(), 4);
    assert_eq!(whole.n[0], 6 * (20 + 27), "every content token");
    assert_eq!(whole.n[1], 0, "every question is in the sample");
}

/// **A removal is never having folded the stream**: its sums come out
/// exactly, and the entries it led are rescanned to the leaders a fold
/// without it finds.
#[test]
fn a_removal_is_never_having_folded() {
    let view = gallery(0xBEEF);
    let arena = HostArena::default();
    let gone = StreamId(1_021);
    let without = with_content(&view, |s| s.sid != gone);

    let mut never = LearnerState::default();
    never.fold(&without, &arena, shapes()).unwrap();

    let mut removed = LearnerState::default();
    removed.fold(&view, &arena, shapes()).unwrap();
    let (counts, _) = removed.fold(&without, &arena, shapes()).unwrap();
    assert_eq!(counts.removed, 1);
    assert_eq!(removed.n, never.n);
    assert_eq!(removed.dirty().0, 0, "every entry it led was rescanned");
    let strip = |m: HashMap<StreamId, [Vec<Entry>; 2]>| -> HashMap<StreamId, Vec<_>> {
        m.into_iter()
            .map(|(k, v)| (k, v.iter().flatten().map(|e| e.stats).collect()))
            .collect()
    };
    assert_eq!(strip(entries(&removed)), strip(entries(&never)));
}

/// **A file that vanished while no member held the state is rescanned
/// around.** A takeover's state holds only what its gallery still holds — the
/// vanished file's sums stay counted, the loss the design takes — but every
/// entry the file led waits on a rescan and gets the leaders a fold without
/// it finds.
#[test]
fn a_file_that_vanished_unheld_is_rescanned_around() {
    let view = gallery(0xF11E);
    let arena = HostArena::default();
    let gone = 103u64;
    let without = with_content(&view, |s| s.file != gone);
    let mut never = LearnerState::default();
    never.fold(&without, &arena, shapes()).unwrap();

    let mut taken_over = LearnerState::default();
    taken_over.fold(&view, &arena, shapes()).unwrap();
    // A takeover loads the folded set from the gallery as it stands.
    taken_over.folded[0].retain(|_, s| s.file != gone);
    let (counts, _) = taken_over.fold(&without, &arena, shapes()).unwrap();
    assert_eq!(counts.removed, 0, "nothing held to take out");
    assert_eq!(taken_over.dirty().0, 0);
    let leaders = |s: &LearnerState| -> Vec<(u32, u64, u32, u64)> {
        s.sample
            .values()
            .flat_map(|q| q.entries.iter().flatten())
            .map(|e| (e.stats.a1, e.stats.c1, e.stats.a2, e.stats.c2))
            .collect()
    };
    assert_eq!(leaders(&taken_over), leaders(&never));
}

/// **The cursor rises to the highest stamp folded**, and a fold with nothing
/// new scans nothing.
#[test]
fn the_cursor_rises_and_a_quiet_fold_scans_nothing() {
    let view = gallery(0xFACE);
    let arena = HostArena::default();
    let mut s = LearnerState::default();
    s.fold(&view, &arena, shapes()).unwrap();
    assert_eq!(s.cursor, 53, "the last question's stamp");
    let launched = arena.launches.get();
    let (counts, left) = s.fold(&view, &arena, shapes()).unwrap();
    assert_eq!(
        (counts.appended, counts.removed, counts.entered, left.len()),
        (0, 0, 0, 0)
    );
    assert_eq!(arena.launches.get(), launched);
}

/// **A question that leaves the sample goes, and joins the calibration
/// gallery.** Its question tombstoned, the sample's rest keep their entries'
/// leaders whatever they were, and the calibration gallery grows by it.
#[test]
fn a_question_leaving_joins_the_calibration_gallery() {
    let view = gallery(0xD00D);
    let arena = HostArena::default();
    let mut s = LearnerState::default();
    s.fold(&view, &arena, shapes()).unwrap();
    let gone = StreamId(5_002);
    let fewer = GalleryView {
        questions: view
            .questions
            .iter()
            .filter(|q| q.sid != gone)
            .cloned()
            .collect(),
        ..view.clone()
    };
    let n_before = s.n[1];
    let (counts, left) = s.fold(&fewer, &arena, shapes()).unwrap();
    assert_eq!(left, vec![gone]);
    assert_eq!(
        counts.appended, 1,
        "its stream enters the calibration gallery"
    );
    assert!(s.n[1] > n_before);
    assert!(!s.sample.contains_key(&gone));
}

/// **A hub chunk is cut.** One token planted in the second chunk of file 0's
/// turn is every question's token too: it wins every question about another
/// file, its chunk's mass stands far above every other chunk's, and the
/// content cut is that chunk.
#[test]
fn a_hub_chunk_is_cut() {
    let mut rng = 0x4B1Du64;
    let hub = rand_sig(&mut rng);
    let mut view = GalleryView::default();
    for f in 0..30u64 {
        let mut toks: Vec<WideQSig> = (0..64).map(|_| rand_sig(&mut rng)).collect();
        if f == 0 {
            toks[40] = hub.clone();
        }
        view.content.push(stream(2_000 + f, 300 + f, &toks, f + 1));
    }
    for q in 0..30u64 {
        let mut toks: Vec<WideQSig> = (0..12).map(|_| rand_sig(&mut rng)).collect();
        toks[5] = hub.clone();
        let s = stream(9_000 + q, 300 + q, &toks, 100 + q);
        view.questions.push(Question {
            sid: s.sid,
            owner: s.file,
            sig: s.sig.clone(),
            hlc: s.hlc,
        });
        view.calibration.push(s);
    }
    let arena = HostArena::default();
    let mut s = LearnerState::default();
    s.fold(&view, &arena, shapes()).unwrap();
    let content: Vec<&GalleryStream> = view.content.iter().collect();
    let cut = cut_of(&mut s, Competition::Content, &content);
    assert_eq!(cut.cut, vec![(StreamId(2_000), 32, 64)]);
    assert_eq!(cut.chunks, 60);
    assert!(
        cut.weakest_cut_votes
            .is_some_and(|v| v >= 29 * FOLD_GROUPS as u32),
        "every other file's question, in every fold group: {:?}",
        cut.weakest_cut_votes
    );
}

/// A conversation holding `files` files of one 40-token turn each in
/// `group`, each with one calibration of one 8-token question; the group.
fn calibrated_conversation(files: u64) -> (Conversation, GroupId) {
    let conv = Conversation::ephemeral();
    let (layer, group) = (LayerId::from_raw(1).unwrap(), GroupId::from_raw(1).unwrap());
    let hidden = (LayerId::from_raw(2).unwrap(), GroupId::from_raw(2).unwrap());
    let mut rng = 0x1EA2_u64;
    let write = |tl: TimelineId, sigs: &[WideQSig]| {
        let idx = conv
            .record_turn(
                tl,
                Role::User,
                TurnPartWrite {
                    token_count: sigs.len(),
                    ..Default::default()
                },
                |seqs| Ok(seqs.to_vec()),
            )
            .expect("record_turn");
        conv.persist_wide_q_sigs(turn_stream_id(tl.raw(), idx.0), &encode_wide_sigs(sigs))
            .expect("persist sigs");
    };
    for f in 0..files {
        let (file_tl, cal_tl) = (TimelineId::for_test(100 + f), TimelineId::for_test(200 + f));
        conv.register_timeline(file_tl, layer, group);
        conv.register_timeline(cal_tl, hidden.0, hidden.1);
        write(
            file_tl,
            &(0..40).map(|_| rand_sig(&mut rng)).collect::<Vec<_>>(),
        );
        write(
            cal_tl,
            &(0..8).map(|_| rand_sig(&mut rng)).collect::<Vec<_>>(),
        );
        conv.set_conversation_metadata(
            cal_tl,
            Substrate::CALIBRATES_KEY,
            &file_tl.raw().to_string(),
        )
        .unwrap();
    }
    (conv, group)
}

fn every() -> (GroupSet, Keep) {
    (GroupSet::all(FOLD_GROUPS), Keep::All)
}

/// **A learner restarted from the records resumes there**: the first fold
/// builds from nothing and writes one record a question and the group's cut;
/// a new learner reads them back, folds nothing — no launch — and leaves the
/// same cut; a turn appended to one file is then the only stream folded.
#[test]
fn a_restarted_learner_resumes_from_the_records() {
    let (conv, group) = calibrated_conversation(5);
    let arena = HostArena::default();
    let mut first = ExclusionLearner::new(group);
    let built = first.fold(&conv, &arena, every(), every()).unwrap();
    assert!(built.loaded);
    assert_eq!((built.sample, built.counts.entered), (5, 5));
    assert_eq!(built.n, [5 * 40, 0]);
    assert_eq!(built.written, 5);
    let shared = conv.shared();
    assert_eq!(
        shared
            .custom_objects_under(&questions_prefix(group.raw()))
            .len(),
        5
    );
    assert!(shared.custom_object(&group_key(group.raw())).is_some());

    let launched = arena.launches.get();
    let mut second = ExclusionLearner::new(group);
    let resumed = second.fold(&conv, &arena, every(), every()).unwrap();
    assert!(resumed.loaded);
    assert_eq!(
        (
            resumed.counts.appended,
            resumed.counts.entered,
            resumed.sample
        ),
        (0, 0, 5)
    );
    assert_eq!(arena.launches.get(), launched, "nothing was scanned");
    assert_eq!(resumed.content.cut, built.content.cut);
    assert_eq!(resumed.n, built.n);
    assert_eq!(resumed.written, 0, "no record moved");
    // A fold that changes nothing writes nothing, the group record included.
    let version = shared.custom_objects_version();
    second.fold(&conv, &arena, every(), every()).unwrap();
    assert_eq!(shared.custom_objects_version(), version);

    let file = TimelineId::for_test(103);
    let mut rng = 0xA11_u64;
    let more: Vec<WideQSig> = (0..12).map(|_| rand_sig(&mut rng)).collect();
    let idx = conv
        .record_turn(
            file,
            Role::User,
            TurnPartWrite {
                token_count: more.len(),
                ..Default::default()
            },
            |seqs| Ok(seqs.to_vec()),
        )
        .unwrap();
    conv.persist_wide_q_sigs(turn_stream_id(file.raw(), idx.0), &encode_wide_sigs(&more))
        .unwrap();
    let grown = second.fold(&conv, &arena, every(), every()).unwrap();
    assert_eq!((grown.counts.appended, grown.counts.removed), (1, 0));
    assert_eq!(grown.n, [5 * 40 + 12, 0]);
}

/// **A record whose question is gone is tombstoned** by the first fold of the
/// learner that loads it.
#[test]
fn an_orphaned_question_record_is_tombstoned() {
    let (conv, group) = calibrated_conversation(3);
    let arena = HostArena::default();
    ExclusionLearner::new(group)
        .fold(&conv, &arena, every(), every())
        .unwrap();
    let shared = conv.shared();
    let orphan = question_key(group.raw(), StreamId(424_242));
    shared
        .put_custom_object(CustomObjectPayload::new(
            orphan.clone(),
            to_json(&QuestionRecord::default().encode()),
        ))
        .unwrap();
    let report = ExclusionLearner::new(group)
        .fold(&conv, &arena, every(), every())
        .unwrap();
    assert_eq!(report.tombstoned, 1);
    assert!(shared.custom_object(&orphan).is_none());
    assert_eq!(report.sample, 3, "the live questions' records still stand");
}

/// **A question at another width never takes a place in the sample**: it
/// has no token to scan.
#[test]
fn a_question_at_another_width_takes_no_place() {
    let mut view = gallery(0x7001);
    let narrow = WideQSig {
        n_heads: 8,
        words: vec![7; 16],
    };
    let s = stream(7_777, 100, &[narrow.clone(), narrow], 90);
    view.questions.push(Question {
        sid: s.sid,
        owner: s.file,
        sig: s.sig.clone(),
        hlc: s.hlc,
    });
    let chosen = sample_of(&view, view.width().unwrap());
    assert_eq!(chosen.len(), 4);
    assert!(!chosen.contains(&StreamId(7_777)));
}

/// **Every member installs a group's cut as it applies the record**, and a
/// tombstone on it leaves the group with no cut.
#[test]
fn a_group_record_installs_its_cut_on_apply() {
    let (conv, group) = calibrated_conversation(1);
    let shared = conv.shared();
    let rec = CutRecord {
        cursor: 9,
        n: [1, 2],
        cut: vec![(StreamId(77), 32, 64)],
    };
    shared
        .put_custom_object(CustomObjectPayload::new(
            group_key(group.raw()),
            to_json(&rec.encode()),
        ))
        .unwrap();
    let held = conv.inner.read().chunk_exclusion(group);
    let cut: Vec<(usize, usize)> = held
        .of(StreamId(77))
        .iter()
        .map(|r| (r.start, r.end))
        .collect();
    assert_eq!(cut, vec![(32, 64)]);
    shared
        .put_custom_object(CustomObjectPayload::ceased(group_key(group.raw())))
        .unwrap();
    assert!(conv.inner.read().chunk_exclusion(group).is_empty());
}
