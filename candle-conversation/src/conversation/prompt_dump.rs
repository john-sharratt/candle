//! The exact tokens a conversation would be given if it decoded now.
//!
//! [`Sequence::prompt_dump`] recomputes the projection the way a turn does —
//! belief scores from the last sealed turn, the conversation's own selection —
//! and walks its segments in injection order. Each becomes a [`PromptPiece`]
//! carrying the token ids the model is handed and their decoded text, so a
//! caller can check that a section it submitted, a tool it expected or a turn it
//! sent is really in the context, and what scored it there.

use std::collections::HashMap;
use std::sync::Arc;

use serde::Serialize;

use super::Sequence;
use crate::projection::event::{group_name_of, role_str};
use crate::projection::{
    GroupId, ProjectionMode, ProjectionSegment, ResolvedTurn, Schema, SealedKind,
    SectionCollection, SectionId, SelectionScores, SystemPromptItem, TimelineId, TurnIndex,
};

/// What a piece of the prompt is.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum PieceKind {
    /// A system-prompt section, sealed in the substrate.
    Section,
    /// A sealed conversation turn, boundary markers included.
    Turn,
    /// The user half of a turn, injected alone by a compression pass.
    TurnHalf,
    /// Structural template tokens prefilled live (role markers, envelopes).
    Glue,
    /// The user message of the turn being submitted.
    UserMessage,
}

/// One piece of the prompt, in injection order.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct PromptPiece {
    pub kind: PieceKind,
    /// `collection/section` or the bare section name for a section, the group
    /// name for a turn, the template's name for glue.
    pub name: String,
    /// The substrate id of a section.
    pub section_id: Option<u32>,
    /// The timeline and index of a turn.
    pub timeline: Option<u64>,
    pub index: Option<u32>,
    /// `user` / `assistant` / `system` for a turn.
    pub role: Option<String>,
    /// The belief score that put the section or turn here; `0.0` when unscored.
    pub score: f32,
    /// Whether the score cleared its own gate rather than the piece being
    /// filled in to a budget.
    pub qualified: bool,
    pub token_count: usize,
    #[serde(skip)]
    pub token_ids: Vec<u32>,
    pub text: String,
}

/// The whole prompt: every piece, and the same tokens concatenated.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct PromptDump {
    pub pieces: Vec<PromptPiece>,
    pub token_count: usize,
    #[serde(skip)]
    pub token_ids: Vec<u32>,
    pub text: String,
}

/// Where the dump reads the tokens and names behind a segment's ids.
pub(crate) struct DumpSource<'a> {
    pub section_name: &'a dyn Fn(SectionId) -> String,
    pub group_name: &'a dyn Fn(GroupId) -> String,
    pub section_tokens: &'a dyn Fn(SectionId) -> Arc<Vec<u32>>,
    pub turn_tokens: &'a dyn Fn(TimelineId, TurnIndex) -> Vec<u32>,
    pub decode: &'a dyn Fn(&[u32]) -> String,
}

/// Lay `segments` out as pieces in injection order.
pub(crate) fn lay_out(
    segments: &[ProjectionSegment],
    scores: &SelectionScores,
    source: &DumpSource<'_>,
) -> PromptDump {
    let mut pieces = Vec::with_capacity(segments.len());
    for segment in segments {
        let piece = match segment {
            ProjectionSegment::Sealed(SealedKind::Section(section)) => {
                let ids = (source.section_tokens)(section.id).as_ref().clone();
                PromptPiece {
                    section_id: Some(section.id.raw()),
                    score: scores.section(section.id),
                    qualified: scores.section_qualified(section.id),
                    ..piece_of(
                        PieceKind::Section,
                        (source.section_name)(section.id),
                        ids,
                        source,
                    )
                }
            }
            ProjectionSegment::Sealed(SealedKind::Turn(turn, role)) => {
                turn_piece(PieceKind::Turn, turn, Some(role_str(*role)), scores, source)
            }
            ProjectionSegment::Sealed(SealedKind::TurnHalf(turn)) => {
                turn_piece(PieceKind::TurnHalf, turn, None, scores, source)
            }
            ProjectionSegment::Generated { tokens, identity } => piece_of(
                PieceKind::Glue,
                identity.name.clone(),
                tokens.as_ref().clone(),
                source,
            ),
            ProjectionSegment::NewUserMessage { tokens } => piece_of(
                PieceKind::UserMessage,
                "current message".to_string(),
                tokens.as_ref().clone(),
                source,
            ),
        };
        pieces.push(piece);
    }
    let token_ids: Vec<u32> = pieces
        .iter()
        .flat_map(|p| p.token_ids.iter().copied())
        .collect();
    PromptDump {
        pieces,
        token_count: token_ids.len(),
        text: (source.decode)(&token_ids),
        token_ids,
    }
}

fn piece_of(
    kind: PieceKind,
    name: String,
    token_ids: Vec<u32>,
    source: &DumpSource<'_>,
) -> PromptPiece {
    PromptPiece {
        kind,
        name,
        section_id: None,
        timeline: None,
        index: None,
        role: None,
        score: 0.0,
        qualified: false,
        token_count: token_ids.len(),
        text: (source.decode)(&token_ids),
        token_ids,
    }
}

fn turn_piece(
    kind: PieceKind,
    turn: &ResolvedTurn,
    role: Option<&str>,
    scores: &SelectionScores,
    source: &DumpSource<'_>,
) -> PromptPiece {
    let key = turn.key();
    let ids = key
        .map(|k| (source.turn_tokens)(k.timeline, k.index))
        .unwrap_or_default();
    PromptPiece {
        timeline: key.map(|k| k.timeline.raw()),
        index: Some(turn.index().0),
        role: role.map(str::to_string),
        score: key.map_or(0.0, |k| scores.turn(k)),
        qualified: key.is_some_and(|k| scores.turn_qualified(k)),
        ..piece_of(kind, (source.group_name)(turn.group()), ids, source)
    }
}

/// Every substrate section id the schema names, with the name it goes by.
///
/// Two sections that share a name keep separate ids, so a dump that shows the
/// same name twice shows two different sealed sections.
pub(crate) fn section_names(schema: &Schema) -> HashMap<SectionId, String> {
    let mut names = HashMap::new();
    for item in &schema.system_prompt.items {
        match item {
            SystemPromptItem::Section(s) => {
                names.insert(s.id, s.name.clone());
            }
            SystemPromptItem::Collection(c) => name_collection(c, &mut names),
            SystemPromptItem::SectionTree(tree) => {
                for node in &tree.nodes {
                    for option in &node.options {
                        for variant in &option.variants {
                            names.insert(variant.id, format!("{}:{}", node.name, option.id));
                        }
                    }
                    if let Some(tc) = &node.collection {
                        name_collection(&tc.collection, &mut names);
                        for (member, variants) in tc.collection.sections.iter().zip(&tc.variants) {
                            for variant in variants {
                                names.insert(
                                    variant.id,
                                    format!("{}/{}", tc.collection.name, member.name),
                                );
                            }
                        }
                    }
                }
            }
        }
    }
    names
}

fn name_collection(collection: &SectionCollection, names: &mut HashMap<SectionId, String>) {
    for s in &collection.sections {
        names.insert(s.id, format!("{}/{}", collection.name, s.name));
    }
    if let Some(id) = collection.summary_section {
        names.insert(id, format!("{}/summary", collection.name));
    }
}

impl Sequence {
    /// The tokens this conversation would be given if it decoded now, piece by
    /// piece with the scores that selected them. `None` for a layer that does
    /// not reproject (`disable_reprojection`), whose prompt is not a selection.
    pub fn prompt_dump(&self) -> Option<PromptDump> {
        if self.config.disable_reprojection {
            return None;
        }
        let scores = self.last_turn_belief_scores();
        let resolver = self.substrate.read_for_scored(self.target, &scores);
        let projection = self.projection.project_with_selection(
            self.target,
            &resolver,
            ProjectionMode::Decode,
            &self.selection,
        );
        let schema = self.projection.schema();
        let names = section_names(schema);
        let section_name = |id: SectionId| {
            names
                .get(&id)
                .cloned()
                .unwrap_or_else(|| format!("section#{}", id.raw()))
        };
        let group_name = |id: GroupId| {
            group_name_of(schema, id)
                .unwrap_or("conversation")
                .to_string()
        };
        let section_tokens = |id: SectionId| resolver.section_tokens_of(id);
        let turn_tokens =
            |timeline: TimelineId, index: TurnIndex| resolver.token_ids_of(timeline, index);
        let decode = |ids: &[u32]| self.tokenizer.decode(ids, false).unwrap_or_default();
        Some(lay_out(
            &projection.segments,
            &projection.selection_scores,
            &DumpSource {
                section_name: &section_name,
                group_name: &group_name,
                section_tokens: &section_tokens,
                turn_tokens: &turn_tokens,
                decode: &decode,
            },
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::projection::{GeneratedIdentity, LayerId, ResolvedSection, TurnId};
    use crate::Role;

    fn decode(ids: &[u32]) -> String {
        ids.iter().map(|i| format!("<{i}>")).collect()
    }

    fn dump(segments: &[ProjectionSegment], scores: &SelectionScores) -> PromptDump {
        let section_name = |id: SectionId| format!("name{}", id.raw());
        let group_name = |_: GroupId| "dialogue".to_string();
        let section_tokens = |id: SectionId| Arc::new(vec![id.raw() * 10, id.raw() * 10 + 1]);
        let turn_tokens = |_: TimelineId, index: TurnIndex| vec![900 + index.0, 901 + index.0];
        lay_out(
            segments,
            scores,
            &DumpSource {
                section_name: &section_name,
                group_name: &group_name,
                section_tokens: &section_tokens,
                turn_tokens: &turn_tokens,
                decode: &decode,
            },
        )
    }

    fn section(n: u32) -> ProjectionSegment {
        ProjectionSegment::Sealed(SealedKind::Section(ResolvedSection {
            id: SectionId::new(n),
        }))
    }

    #[test]
    fn pieces_keep_injection_order_and_concatenate() {
        let turn = ResolvedTurn {
            id: TurnId {
                layer_id: LayerId::for_test(1),
                group_id: GroupId::for_test(1),
                index: TurnIndex(3),
            },
            timeline: Some(TimelineId::for_test(7)),
        };
        let segments = vec![
            section(1),
            ProjectionSegment::Generated {
                tokens: Arc::new(vec![5, 6]),
                identity: GeneratedIdentity {
                    name: "tools_open".into(),
                    position: 1,
                },
            },
            section(2),
            ProjectionSegment::Sealed(SealedKind::Turn(turn, Role::Assistant)),
            ProjectionSegment::NewUserMessage {
                tokens: Arc::new(vec![42]),
            },
        ];
        let d = dump(&segments, &SelectionScores::default());
        let kinds: Vec<_> = d.pieces.iter().map(|p| p.kind).collect();
        assert_eq!(
            kinds,
            [
                PieceKind::Section,
                PieceKind::Glue,
                PieceKind::Section,
                PieceKind::Turn,
                PieceKind::UserMessage
            ]
        );
        assert_eq!(d.token_ids, [10, 11, 5, 6, 20, 21, 903, 904, 42]);
        assert_eq!(d.token_count, 9);
        assert_eq!(d.text, "<10><11><5><6><20><21><903><904><42>");
        assert_eq!(d.pieces[0].name, "name1");
        assert_eq!(d.pieces[0].section_id, Some(1));
        assert_eq!(d.pieces[1].name, "tools_open");
        let t = &d.pieces[3];
        assert_eq!(
            (t.name.as_str(), t.index, t.role.as_deref(), t.timeline),
            ("dialogue", Some(3), Some("assistant"), Some(7))
        );
        assert_eq!(t.token_count, 2);
    }

    #[test]
    fn a_section_carries_the_score_that_selected_it() {
        let mut scores = SelectionScores::default();
        scores.set_section(SectionId::new(2), 7.5, true);
        let d = dump(&[section(1), section(2)], &scores);
        assert_eq!((d.pieces[0].score, d.pieces[0].qualified), (0.0, false));
        assert_eq!((d.pieces[1].score, d.pieces[1].qualified), (7.5, true));
    }

    #[test]
    fn the_same_name_under_two_ids_stays_two_pieces() {
        let section_name = |_: SectionId| "mission/standing".to_string();
        let group_name = |_: GroupId| String::new();
        let section_tokens = |id: SectionId| Arc::new(vec![id.raw()]);
        let turn_tokens = |_: TimelineId, _: TurnIndex| Vec::new();
        let d = lay_out(
            &[section(1), section(2)],
            &SelectionScores::default(),
            &DumpSource {
                section_name: &section_name,
                group_name: &group_name,
                section_tokens: &section_tokens,
                turn_tokens: &turn_tokens,
                decode: &decode,
            },
        );
        assert_eq!(d.pieces.len(), 2);
        assert_eq!(d.pieces[0].name, d.pieces[1].name);
        assert_ne!(d.pieces[0].section_id, d.pieces[1].section_id);
    }
}
