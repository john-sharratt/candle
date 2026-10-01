//! The `working_set` selection rule through the whole projection
//! (`docs/zend_working_set.md` §4.4, §7 "Selection").

use std::collections::HashMap;

use super::builder::Builder;
use super::ids::{GroupId, TimelineId, TurnIndex, TurnKey};
use super::project::{
    OptionalState, ProjectionMode, ProjectionTarget, SelectionState, TOOL_ROUND_SELECTOR,
};
use super::working_set_pick::{Standing, WorkingSetMembers, PINNED_SCORE};
use crate::substrate::ContentResolver;
use crate::summary_tree::SelectionOrigin;

/// A folder layer, a file layer and a dialogue that carries a 1,000-token
/// working set, 300 of it for folders.
const WS_YAML: &str = r#"
system_prompt:
  sections:
    - id: frame
      content: "You are a helpful assistant."
layers:
  - name: repo_map
    window: 6000
    summary: &summary
      turns:
        max_tokens: 256
        user:
          system_prompt: compress
          user_prompt: compress
        assistant:
          system_prompt: compress
          user_prompt: compress
    groups:
      - id: structure
        selection: { kind: working_set, share: folders }

  - name: code_reading
    window: 40000
    summary: *summary
    groups:
      - id: scopes
        selection: { kind: working_set, share: remainder }

  - name: dialogue
    window: 500
    summary: *summary
    working_set:
      budget_tokens: 1000
      folder_tokens: 300
      beta: 0.2
      min_momentum: 100
      max_file_tokens: 800
      release_on: [write]
    budget:
      priority: 100
    groups:
      - id: conversation
        selection:
          kind: conversation
          recent: 4
          historical_top_k: 0
"#;

/// Conversations are whole timelines here, each in one group, each turn
/// `tokens` long; the dialogue's own turns sit on [`DIALOGUE`].
#[derive(Default)]
struct WsResolver {
    /// timeline → (group, turn count, tokens per turn)
    timelines: HashMap<u64, (GroupId, u32, usize)>,
    /// group → the working set's members there; absent ⇒ the target carries none.
    members: HashMap<GroupId, WorkingSetMembers>,
    ingest_self: bool,
    /// Every turn answers "carries none of these tags" — what a turn holding
    /// only working-set marks answers a tag-scoped group.
    carries_nothing: bool,
}

const DIALOGUE: u64 = 1;

fn tl(raw: u64) -> TimelineId {
    TimelineId::from_raw(raw).unwrap()
}

impl WsResolver {
    fn with_conversation(mut self, raw: u64, group: GroupId, turns: u32, tokens: usize) -> Self {
        self.timelines.insert(raw, (group, turns, tokens));
        self
    }

    /// The next member of `group`'s working set, after the ones added before.
    fn member(mut self, group: GroupId, raw: u64, standing: Standing) -> Self {
        self.members
            .entry(group)
            .or_default()
            .members
            .push((tl(raw), standing));
        self
    }

    fn lock(self, group: GroupId, raw: u64) -> Self {
        self.member(group, raw, Standing::Locked)
    }

    fn provenance(self, group: GroupId, raw: u64, momentum: f32) -> Self {
        self.member(group, raw, Standing::Provenance(momentum))
    }
}

impl ContentResolver for WsResolver {
    fn group_turns(&self, group: GroupId) -> Vec<TurnKey> {
        let mut raws: Vec<u64> = self
            .timelines
            .iter()
            .filter(|(_, (g, _, _))| *g == group)
            .map(|(raw, _)| *raw)
            .collect();
        raws.sort_unstable();
        raws.into_iter()
            .flat_map(|raw| self.timeline_turns(tl(raw)))
            .collect()
    }

    fn turn_token_count(&self, turn: TurnKey) -> usize {
        self.timelines
            .get(&turn.timeline.raw())
            .map_or(0, |(_, _, tokens)| *tokens)
    }

    fn turn_score(&self, _turn: TurnKey) -> f32 {
        0.0
    }

    fn target_is_ingest_self(&self) -> bool {
        self.ingest_self
    }

    fn turn_carries(&self, _turn: TurnKey, _tags: &[String]) -> bool {
        !self.carries_nothing
    }

    fn working_set_members(&self, group: GroupId) -> Option<WorkingSetMembers> {
        self.members.get(&group).cloned()
    }

    fn timeline_turns(&self, timeline: TimelineId) -> Vec<TurnKey> {
        let turns = self
            .timelines
            .get(&timeline.raw())
            .map_or(0, |(_, turns, _)| *turns);
        (0..turns)
            .map(|i| TurnKey::new(timeline, TurnIndex(i)))
            .collect()
    }
}

struct Fixture {
    b: Builder,
    structure: GroupId,
    scopes: GroupId,
    conversation: GroupId,
    target: ProjectionTarget,
}

fn fixture() -> Fixture {
    let b = Builder::from_yaml(WS_YAML).unwrap();
    let conversation = b.id_for_group("conversation").unwrap();
    let target = ProjectionTarget {
        layer: b.id_for_layer("dialogue").unwrap(),
        group: conversation,
        timeline: tl(DIALOGUE),
    };
    Fixture {
        structure: b.id_for_group("structure").unwrap(),
        scopes: b.id_for_group("scopes").unwrap(),
        conversation,
        target,
        b,
    }
}

/// Each emitted conversation once, in emission order, as `(timeline, turns)`.
fn emitted(b: &Builder, target: ProjectionTarget, r: &WsResolver) -> Vec<(u64, usize)> {
    emitted_with(b, target, r, &SelectionState::new())
}

fn emitted_with(
    b: &Builder,
    target: ProjectionTarget,
    r: &WsResolver,
    selection: &SelectionState,
) -> Vec<(u64, usize)> {
    let proj = b.project_with_selection(target, r, ProjectionMode::Decode, selection);
    let mut out: Vec<(u64, usize)> = Vec::new();
    for t in proj.sealed_turns() {
        let raw = t.timeline.unwrap().raw();
        match out.last_mut() {
            Some((last, n)) if *last == raw => *n += 1,
            _ => out.push((raw, 1)),
        }
    }
    out
}

/// Every member emits whole, in the working set's order — never re-sorted by
/// momentum, standing or timeline id. The set enforced the budget when its
/// members entered, so the projection trims nothing.
#[test]
fn every_member_emits_whole_in_the_sets_order() {
    let f = fixture();
    let r = WsResolver::default()
        .with_conversation(DIALOGUE, f.conversation, 1, 10)
        .with_conversation(10, f.scopes, 2, 50)
        .with_conversation(12, f.scopes, 1, 50)
        .with_conversation(13, f.scopes, 3, 50)
        .with_conversation(14, f.scopes, 1, 50)
        .provenance(f.scopes, 13, 500.0)
        .provenance(f.scopes, 10, 9_000.0)
        .lock(f.scopes, 14)
        .lock(f.scopes, 12);
    assert_eq!(
        emitted(&f.b, f.target, &r),
        vec![(13, 3), (10, 2), (14, 1), (12, 1), (DIALOGUE, 1)],
    );
}

/// A lock is recorded at the pinned score, provenance at its momentum.
#[test]
fn each_member_carries_its_standing_into_the_selection_record() {
    let f = fixture();
    let r = WsResolver::default()
        .with_conversation(DIALOGUE, f.conversation, 1, 10)
        .with_conversation(10, f.scopes, 1, 50)
        .with_conversation(12, f.scopes, 1, 50)
        .provenance(f.scopes, 12, 4_000.0)
        .lock(f.scopes, 10);
    let proj = f.b.project(f.target, &r);
    let key = |raw: u64| TurnKey::new(tl(raw), TurnIndex(0));
    assert_eq!(
        proj.selection_origins[&key(10)],
        SelectionOrigin::WorkingSetLock
    );
    assert_eq!(
        proj.selection_origins[&key(12)],
        SelectionOrigin::WorkingSetMomentum
    );
    assert_eq!(proj.selection_scores.turn(key(10)), PINNED_SCORE);
    assert_eq!(proj.selection_scores.turn(key(12)), 4_000.0);
}

/// The folder group and the file group each emit their own members, the
/// folders' layer first.
#[test]
fn folders_and_files_each_emit_their_own_members() {
    let f = fixture();
    let r = WsResolver::default()
        .with_conversation(DIALOGUE, f.conversation, 1, 10)
        .with_conversation(20, f.structure, 1, 200)
        .with_conversation(30, f.scopes, 1, 500)
        .provenance(f.scopes, 30, 900.0)
        .provenance(f.structure, 20, 900.0);
    let picked: Vec<u64> = emitted(&f.b, f.target, &r)
        .into_iter()
        .map(|(raw, _)| raw)
        .collect();
    assert_eq!(picked, vec![20, 30, DIALOGUE]);
}

/// The working set is off the flexbox: the dialogue's own selection under a
/// full working set is exactly what it is with none.
#[test]
fn the_dialogue_selects_the_same_with_or_without_a_working_set() {
    let f = fixture();
    let bare = WsResolver::default().with_conversation(DIALOGUE, f.conversation, 6, 100);
    let full = WsResolver::default()
        .with_conversation(DIALOGUE, f.conversation, 6, 100)
        .with_conversation(30, f.scopes, 1, 700)
        .with_conversation(20, f.structure, 1, 300)
        .lock(f.scopes, 30)
        .lock(f.structure, 20);
    let own = |r: &WsResolver| -> Vec<(u64, usize)> {
        emitted(&f.b, f.target, r)
            .into_iter()
            .filter(|(raw, _)| *raw == DIALOGUE)
            .collect()
    };
    assert_eq!(own(&bare), vec![(DIALOGUE, 4)], "recent 4 of a 500 window");
    assert_eq!(own(&full), own(&bare));
}

/// The marks on a dialogue's turns change none of its selection: the target's
/// own group never consults tags, so turns that answer every tag question with
/// "none" project exactly as untagged ones.
#[test]
fn the_marks_change_no_dialogue_selection() {
    let f = fixture();
    let plain = WsResolver::default().with_conversation(DIALOGUE, f.conversation, 6, 100);
    let marked = WsResolver {
        carries_nothing: true,
        ..WsResolver::default()
    }
    .with_conversation(DIALOGUE, f.conversation, 6, 100);
    assert_eq!(
        emitted(&f.b, f.target, &marked),
        emitted(&f.b, f.target, &plain)
    );
    assert_eq!(emitted(&f.b, f.target, &plain), vec![(DIALOGUE, 4)]);
}

/// The working set emits ahead of the dialogue — its layers sit below it — and
/// it is in tool rounds too.
#[test]
fn the_working_set_sits_ahead_of_the_dialogue_in_every_round() {
    let f = fixture();
    let r = WsResolver::default()
        .with_conversation(DIALOGUE, f.conversation, 1, 10)
        .with_conversation(20, f.structure, 1, 100)
        .with_conversation(30, f.scopes, 1, 100)
        .lock(f.structure, 20)
        .lock(f.scopes, 30);
    let mut round = SelectionState::new();
    round.set_optional(TOOL_ROUND_SELECTOR, OptionalState::Present);
    for selection in [SelectionState::new(), round] {
        assert_eq!(
            emitted_with(&f.b, f.target, &r, &selection),
            vec![(20, 1), (30, 1), (DIALOGUE, 1)],
        );
    }
}

/// A target whose layer declares no working set — every ingest conversation —
/// selects nothing into a working-set group, whatever the resolver holds.
#[test]
fn a_target_without_a_working_set_selects_nothing() {
    let f = fixture();
    let scopes_target = ProjectionTarget {
        layer: f.b.id_for_layer("code_reading").unwrap(),
        group: f.scopes,
        timeline: tl(40),
    };
    let r = WsResolver::default()
        .with_conversation(40, f.scopes, 2, 10)
        .with_conversation(20, f.structure, 1, 100)
        .lock(f.structure, 20);
    assert!(
        emitted(&f.b, scopes_target, &r)
            .iter()
            .all(|(raw, _)| *raw != 20),
        "a code_reading target carries no working set",
    );
}

/// An ingest conversation generating into a working-set group reads its own
/// turns there, trimmed to its window like any other self-read.
#[test]
fn an_ingest_conversation_reads_its_own_working_set_group() {
    let f = fixture();
    let own = ProjectionTarget {
        layer: f.b.id_for_layer("code_reading").unwrap(),
        group: f.scopes,
        timeline: tl(40),
    };
    let r = WsResolver {
        ingest_self: true,
        ..WsResolver::default()
    }
    .with_conversation(40, f.scopes, 3, 10);
    assert_eq!(emitted(&f.b, own, &r), vec![(40, 3)]);
}
