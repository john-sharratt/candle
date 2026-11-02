//! A turn's recorded projections, played back instead of chosen.
//!
//! A decode's context is whatever its projections placed, and each of those was
//! chosen against the substrate as it stood. Rebuilt later over a substrate
//! that has moved on, the same scan picks something else, so a replay that
//! re-selects is not a replay of what the decode read. A script holds the
//! recorded choice of each projection point: the members of every collection and
//! the turns, in order. The scan still runs, because its cost is part of what
//! is measured, and its answer is set aside.
//!
//! A point with no recorded choice is live: the projection runs as production
//! runs it. A script ends on one to time a real reprojection over the context
//! the decode finished on.

use std::collections::HashSet;
use std::sync::Arc;

use super::event::{ProjectionEvent, SelectedTurn, SystemItem};
use super::ids::{TimelineId, TurnId, TurnIndex};
use super::project::{Projection, ProjectionSegment, ResolvedTurn, SealedKind};
use super::schema::Schema;

/// What one recorded projection chose.
#[derive(Debug, Clone, PartialEq)]
pub struct ScriptedChoice {
    /// Every collection member it selected, by section name.
    pub members: Arc<HashSet<String>>,
    /// Its turns, in slot order.
    pub turns: Vec<ResolvedTurn>,
}

/// One projection point of a turn.
#[derive(Debug, Clone, PartialEq)]
struct ScriptPoint {
    /// The generated-token count it governs from.
    at: usize,
    /// `None` runs the projection live.
    choice: Option<ScriptedChoice>,
}

/// The projection points of one turn, in token order.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct ProjectionScript {
    points: Vec<ScriptPoint>,
    /// Recorded turns with nothing to stand for them here: summary nodes of the
    /// recorded timeline, which a replay's timeline never grew.
    pub dropped_turns: usize,
}

/// The section names a recorded selection chose from its collections.
fn members_of(system: &[SystemItem]) -> HashSet<String> {
    let mut out = HashSet::new();
    for item in system {
        match item {
            SystemItem::Section { name, .. } => {
                out.insert(name.clone());
            }
            SystemItem::Collection { sections, .. } => {
                out.extend(
                    sections
                        .iter()
                        .filter(|s| s.selected)
                        .map(|s| s.name.clone()),
                );
            }
            SystemItem::Glue { .. } => {}
        }
    }
    out
}

/// `turn` as this schema names it, where `moved` puts it. `None` for the live
/// message and for a turn `moved` has no place for.
fn resolve(
    turn: &SelectedTurn,
    schema: &Schema,
    moved: &dyn Fn(u64, u32) -> Option<(TimelineId, TurnIndex)>,
) -> Option<ResolvedTurn> {
    let (timeline, index) = moved(turn.timeline?, turn.index)?;
    let (layer_id, group_id) = schema.layers.iter().find_map(|l| {
        l.groups
            .iter()
            .find(|g| g.name == turn.group)
            .map(|g| (l.id, g.id))
    })?;
    Some(ResolvedTurn {
        id: TurnId {
            layer_id,
            group_id,
            index,
        },
        timeline: Some(timeline),
    })
}

impl ProjectionScript {
    /// The script of a turn recorded as `events`, in the order they were made.
    /// `moved` says where a recorded `(timeline, index)` stands now.
    pub fn from_events(
        events: &[ProjectionEvent],
        schema: &Schema,
        moved: &dyn Fn(u64, u32) -> Option<(TimelineId, TurnIndex)>,
    ) -> Self {
        let mut dropped_turns = 0;
        let points = events
            .iter()
            .map(|event| {
                let recorded: Vec<&SelectedTurn> = event
                    .selection
                    .turns
                    .iter()
                    .filter(|t| t.timeline.is_some())
                    .collect();
                let turns: Vec<ResolvedTurn> = recorded
                    .iter()
                    .filter_map(|t| resolve(t, schema, moved))
                    .collect();
                dropped_turns = dropped_turns.max(recorded.len() - turns.len());
                ScriptPoint {
                    at: event.start_token as usize,
                    choice: Some(ScriptedChoice {
                        members: Arc::new(members_of(&event.selection.system)),
                        turns,
                    }),
                }
            })
            .collect();
        Self {
            points,
            dropped_turns,
        }
    }

    /// Keep only the points before generated token `at`, and end on a live
    /// point there.
    pub fn ending_live_at(mut self, at: usize) -> Self {
        self.points.retain(|p| p.at < at);
        self.points.push(ScriptPoint { at, choice: None });
        self
    }

    /// Take `turns`, by `(timeline, index)`, out of every point: the control
    /// that asks what a decode does without them.
    pub fn without(mut self, turns: &HashSet<(u64, u32)>) -> Self {
        for point in &mut self.points {
            if let Some(choice) = &mut point.choice {
                choice.turns.retain(|t| {
                    !t.timeline
                        .is_some_and(|tl| turns.contains(&(tl.raw(), t.index().0)))
                });
            }
        }
        self
    }

    /// The recorded choice governing generated token `pos`; `None` when that
    /// point is live.
    pub fn choice_at(&self, pos: usize) -> Option<&ScriptedChoice> {
        let i = self.points.partition_point(|p| p.at <= pos);
        self.points[..i].last()?.choice.as_ref()
    }

    /// Whether a reprojection falls due at generated token `count`.
    pub fn fires_at(&self, count: usize) -> bool {
        count > 0 && self.points.iter().any(|p| p.at == count)
    }

    /// How many points the script holds.
    pub fn len(&self) -> usize {
        self.points.len()
    }

    /// Whether it holds none.
    pub fn is_empty(&self) -> bool {
        self.points.is_empty()
    }
}

impl ScriptedChoice {
    /// Put the recorded turns in `projection` in place of the ones chosen. A
    /// projection's turns close it, after the system prompt.
    pub fn place_turns(&self, projection: &mut Projection) {
        projection
            .segments
            .retain(|s| !matches!(s, ProjectionSegment::Sealed(SealedKind::Turn(..))));
        projection.segments.extend(
            self.turns
                .iter()
                .map(|t| ProjectionSegment::Sealed(SealedKind::Turn(*t, crate::Role::Assistant))),
        );
    }
}

#[cfg(test)]
mod tests {
    use super::super::builder::Builder;
    use super::super::event::{ProjectionSelection, SelectedSection};
    use super::super::ids::SectionId;
    use super::super::ids::{GroupId, LayerId};
    use super::*;
    use crate::summary_tree::TurnKind;

    fn timeline(raw: u64) -> TimelineId {
        TimelineId::from_raw(raw).unwrap()
    }

    fn turn(group: &str, timeline: Option<u64>, index: u32) -> SelectedTurn {
        SelectedTurn {
            layer: "dialogue".into(),
            group: group.into(),
            index,
            role: "assistant".into(),
            tokens: 1,
            kind: TurnKind::Normal,
            reason: None,
            qualified: false,
            timeline,
            selected: true,
            score: 0.0,
        }
    }

    fn event(at: u32, turns: Vec<SelectedTurn>) -> ProjectionEvent {
        ProjectionEvent {
            start_token: at,
            selection: ProjectionSelection {
                system: vec![
                    SystemItem::Glue {
                        name: "open".into(),
                        content: String::new(),
                        tokens: 1,
                    },
                    SystemItem::Section {
                        name: "frame".into(),
                        tokens: 1,
                    },
                    SystemItem::Collection {
                        name: "tools".into(),
                        sections: vec![
                            SelectedSection {
                                name: "file_read".into(),
                                tokens: 1,
                                selected: true,
                                score: 0.0,
                                qualified: false,
                            },
                            SelectedSection {
                                name: "file_edit".into(),
                                tokens: 1,
                                selected: false,
                                score: 0.0,
                                qualified: false,
                            },
                        ],
                        member_glue: String::new(),
                        member_glue_tokens: 0,
                    },
                ],
                turns,
            },
            ..Default::default()
        }
    }

    fn script(events: &[ProjectionEvent]) -> ProjectionScript {
        let builder = Builder::for_plain_prompt("x", SectionId::new(1));
        ProjectionScript::from_events(events, builder.schema(), &|tl, index| match tl {
            7 if index == 1 => Some((timeline(9), TurnIndex(100))),
            7 => None,
            other => Some((timeline(other), TurnIndex(index))),
        })
    }

    /// **A recorded turn lands where the replay holds it**, its own timeline's
    /// on the replay's and another's where it was. The live message is no turn,
    /// and a summary node the replay never grew is counted, not guessed at.
    #[test]
    fn a_choice_moves_the_recorded_turns_onto_the_replay() {
        let s = script(&[event(
            0,
            vec![
                turn("primary_conversation", Some(5), 3),
                turn("primary_conversation", Some(7), 1),
                turn("primary_conversation", Some(7), 2),
                turn("current message", None, u32::MAX),
            ],
        )]);
        let choice = s.choice_at(0).unwrap();
        let keys: Vec<(u64, u32)> = choice
            .turns
            .iter()
            .map(|t| (t.timeline.unwrap().raw(), t.index().0))
            .collect();
        assert_eq!(keys, vec![(5, 3), (9, 100)]);
        assert_eq!(s.dropped_turns, 1);
        assert_eq!(choice.turns[0].id.layer_id, LayerId::new(1));
        assert_eq!(choice.turns[0].id.group_id, GroupId::new(1));
    }

    /// A bare section and a selected member are pinned; a skipped member and
    /// structural glue are not.
    #[test]
    fn only_what_was_selected_is_pinned() {
        let s = script(&[event(0, Vec::new())]);
        let mut members: Vec<String> = s.choice_at(0).unwrap().members.iter().cloned().collect();
        members.sort();
        assert_eq!(members, vec!["file_read".to_string(), "frame".to_string()]);
    }

    /// A point governs up to the next, and the opening one never fires: the
    /// submission makes it.
    #[test]
    fn a_point_governs_until_the_next() {
        let s = script(&[
            event(0, vec![turn("primary_conversation", Some(5), 1)]),
            event(40, vec![turn("primary_conversation", Some(5), 2)]),
        ]);
        let index_at = |pos| s.choice_at(pos).unwrap().turns[0].index().0;
        assert_eq!(index_at(0), 1);
        assert_eq!(index_at(39), 1);
        assert_eq!(index_at(40), 2);
        assert_eq!(index_at(4000), 2);
        assert!(!s.fires_at(0));
        assert!(!s.fires_at(39));
        assert!(s.fires_at(40));
    }

    /// The closing point is live, and takes the place of everything recorded
    /// from there on.
    #[test]
    fn a_script_can_end_on_a_live_point() {
        let s = script(&[
            event(0, Vec::new()),
            event(40, Vec::new()),
            event(80, Vec::new()),
        ])
        .ending_live_at(60);
        assert_eq!(s.len(), 3);
        assert!(s.choice_at(59).is_some());
        assert!(s.choice_at(60).is_none());
        assert!(s.fires_at(60));
        assert!(!s.fires_at(80));
    }

    /// Recorded turns replace the chosen ones and close the projection; the
    /// system prompt keeps its place.
    #[test]
    fn placing_turns_keeps_the_system_prompt() {
        let chosen = ResolvedTurn {
            id: TurnId {
                layer_id: LayerId::new(1),
                group_id: GroupId::new(1),
                index: TurnIndex(1),
            },
            timeline: Some(timeline(1)),
        };
        let recorded = ResolvedTurn {
            id: TurnId {
                index: TurnIndex(2),
                ..chosen.id
            },
            ..chosen
        };
        let glue = ProjectionSegment::NewUserMessage {
            tokens: Arc::new(vec![1]),
        };
        let mut projection = Projection {
            segments: vec![
                glue,
                ProjectionSegment::Sealed(SealedKind::Turn(chosen, crate::Role::Assistant)),
            ],
            selections: Vec::new(),
            selection_scores: Default::default(),
            selection_origins: Default::default(),
        };
        ScriptedChoice {
            members: Arc::new(HashSet::new()),
            turns: vec![recorded],
        }
        .place_turns(&mut projection);
        assert_eq!(projection.segments.len(), 2);
        assert!(matches!(
            projection.segments[0],
            ProjectionSegment::NewUserMessage { .. }
        ));
        assert_eq!(
            projection.sealed_turns().collect::<Vec<_>>(),
            vec![&recorded]
        );
    }
}
