//! A character's system prompt as the engine assembled it.
//!
//! The projection event a turn leaves behind names, in prompt order, every glue
//! marker, bare section and selected collection member that went into the
//! system prompt; the conversation holds the authored text of each by name. This
//! walks the one against the other, so what is shown is the selection the engine
//! actually made for the turn — including which identity members were pinned and
//! whether the mission member was — and not a second renderer's opinion of it.

use std::collections::HashMap;

use candle_conversation::projection::{ProjectionEvent, SystemItem};
use serde::Serialize;

/// One run of the system prompt, in the order the character reads it.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct Piece {
    /// The section's name; a collection member is `collection/member`.
    pub name: String,
    /// `glue` for a structural marker, `section` for a bare section, `member`
    /// for a selected collection member.
    pub kind: &'static str,
    pub text: String,
}

/// A rendered system prompt: the pieces, and the text they make end to end.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct Rendered {
    pub pieces: Vec<Piece>,
    pub text: String,
}

/// Lay `event`'s system selection out as text, looking each name up in
/// `contents` (the schema's `(name, authored text)` pairs).
///
/// A name with no authored text — a runtime summary section the engine
/// synthesises — contributes nothing. A collection's `member_glue` goes between
/// consecutive selected members and never before the first, as the projection
/// emits it.
pub fn render(event: &ProjectionEvent, contents: &[(String, String)]) -> Rendered {
    let text_of: HashMap<&str, &str> = contents
        .iter()
        .map(|(name, text)| (name.as_str(), text.as_str()))
        .collect();
    let mut pieces = Vec::new();
    for item in &event.selection.system {
        match item {
            SystemItem::Glue { name, content, .. } => pieces.push(Piece {
                name: name.clone(),
                kind: "glue",
                text: content.clone(),
            }),
            SystemItem::Section { name, .. } => {
                if let Some(text) = text_of.get(name.as_str()) {
                    pieces.push(Piece {
                        name: name.clone(),
                        kind: "section",
                        text: (*text).to_string(),
                    });
                }
            }
            SystemItem::Collection {
                sections,
                member_glue,
                ..
            } => {
                let mut first = true;
                for member in sections.iter().filter(|s| s.selected) {
                    let Some(text) = text_of.get(member.name.as_str()) else {
                        continue;
                    };
                    if !first && !member_glue.is_empty() {
                        pieces.push(Piece {
                            name: "member_glue".to_string(),
                            kind: "glue",
                            text: member_glue.clone(),
                        });
                    }
                    first = false;
                    pieces.push(Piece {
                        name: member.name.clone(),
                        kind: "member",
                        text: (*text).to_string(),
                    });
                }
            }
        }
    }
    let text = pieces.iter().map(|p| p.text.as_str()).collect();
    Rendered { pieces, text }
}

#[cfg(test)]
mod tests {
    use candle_conversation::projection::{ProjectionSelection, SelectedSection};

    use super::*;

    fn member(name: &str, selected: bool) -> SelectedSection {
        SelectedSection {
            name: name.to_string(),
            tokens: 1,
            selected,
            score: 0.0,
            qualified: false,
        }
    }

    fn event(system: Vec<SystemItem>) -> ProjectionEvent {
        ProjectionEvent {
            selection: ProjectionSelection {
                system,
                turns: Vec::new(),
            },
            ..Default::default()
        }
    }

    fn contents() -> Vec<(String, String)> {
        [
            ("intro", "You are in a world.\n"),
            ("mission/carrying", "A mission you took up."),
            ("tools/tell", "tell: speak."),
            ("tools/walk", "walk: move."),
        ]
        .iter()
        .map(|(n, t)| (n.to_string(), t.to_string()))
        .collect()
    }

    #[test]
    fn only_selected_members_render_and_in_prompt_order() {
        let ev = event(vec![
            SystemItem::Section {
                name: "intro".into(),
                tokens: 1,
            },
            SystemItem::Collection {
                name: "mission".into(),
                sections: vec![member("mission/carrying", true)],
                member_glue: String::new(),
                member_glue_tokens: 0,
            },
            SystemItem::Collection {
                name: "tools".into(),
                sections: vec![member("tools/tell", true), member("tools/walk", false)],
                member_glue: String::new(),
                member_glue_tokens: 0,
            },
        ]);
        let r = render(&ev, &contents());
        let names: Vec<&str> = r.pieces.iter().map(|p| p.name.as_str()).collect();
        assert_eq!(names, ["intro", "mission/carrying", "tools/tell"]);
        assert_eq!(
            r.text,
            "You are in a world.\nA mission you took up.tell: speak."
        );
    }

    #[test]
    fn a_collection_with_nothing_selected_leaves_no_trace() {
        let ev = event(vec![SystemItem::Collection {
            name: "mission".into(),
            sections: vec![member("mission/carrying", false)],
            member_glue: String::new(),
            member_glue_tokens: 0,
        }]);
        let r = render(&ev, &contents());
        assert!(r.pieces.is_empty());
        assert_eq!(r.text, "");
    }

    #[test]
    fn member_glue_sits_between_members_and_never_before_the_first() {
        let ev = event(vec![SystemItem::Collection {
            name: "tools".into(),
            sections: vec![member("tools/tell", true), member("tools/walk", true)],
            member_glue: "\n".into(),
            member_glue_tokens: 1,
        }]);
        assert_eq!(render(&ev, &contents()).text, "tell: speak.\nwalk: move.");
    }

    #[test]
    fn glue_renders_verbatim_and_an_unknown_section_is_skipped() {
        let ev = event(vec![
            SystemItem::Glue {
                name: "open".into(),
                content: "<sys>".into(),
                tokens: 1,
            },
            SystemItem::Section {
                name: "tools summary".into(),
                tokens: 1,
            },
            SystemItem::Glue {
                name: "close".into(),
                content: "</sys>".into(),
                tokens: 1,
            },
        ]);
        let r = render(&ev, &contents());
        assert_eq!(r.text, "<sys></sys>");
        assert_eq!(r.pieces.len(), 2);
    }
}
