//! What follows npcd's system-prompt collections stands still
//! (`docs/placed_section_layout.md` §3.1c): whoever the character is, whichever
//! template and dials a turn has, the first item after the collections is read
//! at one position — the one the creation walk lays down in front of it, which
//! every member is sealed behind.

use std::collections::HashMap;
use std::path::Path;

use candle_conversation::projection::{
    prefix_before_collection, Builder, CollectionId, ContentResolver, GroupId, LayerId, PrefixItem,
    ProjectionMode, ProjectionSegment, ProjectionTarget, SealedKind, SectionId, SelectionRule,
    SelectionState, SystemPromptItem, TimelineId, TurnKey,
};

use crate::engine::identity::{self, Authored, Deliberation};
use crate::engine::schema;
use crate::engine::talk::templates::Destination;
use crate::engine::tests_slot_sizes::{
    authored, installed, live_mind, token_counter, Live, NATIVE, WORLD,
};
use crate::mind::files::MindFiles;

/// Every sealed section's tokens, and what each collection member scores.
#[derive(Default)]
struct Sealed {
    tokens: HashMap<SectionId, usize>,
    scores: HashMap<SectionId, f32>,
}

impl ContentResolver for Sealed {
    fn group_turns(&self, _group: GroupId) -> Vec<TurnKey> {
        Vec::new()
    }

    fn turn_token_count(&self, _turn: TurnKey) -> usize {
        0
    }

    fn trained_window(&self) -> Option<usize> {
        Some(NATIVE)
    }

    fn turn_score(&self, _turn: TurnKey) -> f32 {
        0.0
    }

    fn section_token_count(&self, section: SectionId) -> usize {
        self.tokens.get(&section).copied().unwrap_or(0)
    }

    fn section_score(&self, section: SectionId) -> f32 {
        self.scores.get(&section).copied().unwrap_or(0.0)
    }
}

/// `b`'s system prompt sealed as `count` reads it: every section, every
/// collection member and every tree variant.
fn sealed(b: &Builder, count: &impl Fn(&str) -> usize) -> Sealed {
    let mut out = Sealed::default();
    for item in &b.schema().system_prompt.items {
        match item {
            SystemPromptItem::Section(s) => {
                out.tokens.insert(s.id, count(&s.content));
            }
            SystemPromptItem::Collection(c) => {
                for s in &c.sections {
                    out.tokens.insert(s.id, count(&s.content));
                }
            }
            SystemPromptItem::SectionTree(t) => {
                for o in t.nodes.iter().flat_map(|n| &n.options) {
                    for v in &o.variants {
                        out.tokens.insert(v.id, count(&o.content));
                    }
                }
            }
        }
    }
    out
}

/// The position section `id` is read at when `b` projects `selection` onto
/// `(layer, group)`.
fn position_of(
    b: &Builder,
    r: &Sealed,
    target: (LayerId, GroupId),
    selection: &SelectionState,
    id: SectionId,
) -> usize {
    let target = ProjectionTarget {
        layer: target.0,
        group: target.1,
        timeline: TimelineId::from_raw(1).expect("a timeline id is non-zero"),
    };
    let segments = b
        .project_with_selection(target, r, ProjectionMode::Decode, selection)
        .segments;
    let mut at = 0;
    for s in &segments {
        match s {
            ProjectionSegment::Sealed(SealedKind::Section(x)) if x.id == id => return at,
            ProjectionSegment::Sealed(SealedKind::Section(x)) => at += r.section_token_count(x.id),
            ProjectionSegment::Hole { positions } => at += positions,
            ProjectionSegment::Generated { tokens, .. } => at += tokens.len(),
            _ => {}
        }
    }
    panic!("section {id:?} is not projected")
}

/// What the creation walk lays down in front of collection `before`, and its
/// range: where the item after it stands.
fn walked_to_after(b: &Builder, r: &Sealed, before: CollectionId) -> usize {
    let items = &b.schema().system_prompt.items;
    let prefix =
        prefix_before_collection(items, before, Some(NATIVE), |s| r.section_token_count(s))
            .expect("the collection is declared");
    let laid: usize = prefix
        .iter()
        .map(|p| match p {
            PrefixItem::Section(id) => r.section_token_count(*id),
            PrefixItem::Hole(n) => *n,
        })
        .sum();
    let range = items
        .iter()
        .find_map(|it| match it {
            SystemPromptItem::Collection(c) if c.id == before => Some(c.range),
            _ => None,
        })
        .flatten()
        .map_or(0, |range| range.positions(Some(NATIVE)));
    laid + range
}

/// The schema bundled beside this crate, with two personalities of unlike
/// sizes in its identity collections and three response templates.
fn bundled() -> (Builder, (LayerId, GroupId)) {
    let mind = Path::new(env!("CARGO_MANIFEST_DIR"));
    let p =
        schema::build(Some(&MindFiles::folder(mind)), WORLD).expect("the bundled schema parses");
    let mut b = p.builder;
    for name in [identity::ANCHOR, identity::WHO] {
        b.set_collection_selection(
            name,
            SelectionRule::Named {
                selector: name.to_string(),
            },
        )
        .unwrap();
    }
    let add = |b: &mut Builder, collection: &str, member: &str, content: &str| {
        let cid = b.id_for_system_collection(collection).unwrap();
        b.add_section_to_collection(cid, format!("{collection}/{member}"), content, 100.0)
            .unwrap();
    };
    add(&mut b, identity::ANCHOR, "brief", "You keep the gate.");
    add(
        &mut b,
        identity::ANCHOR,
        "full",
        &"You remember every name. ".repeat(20),
    );
    add(&mut b, identity::WHO, "brief", "You are Ash.");
    add(
        &mut b,
        identity::WHO,
        "full",
        &"You are Mercer, who trades. ".repeat(12),
    );
    add(&mut b, "response", "terse", "One line.");
    add(
        &mut b,
        "response",
        "told",
        &"Tell it as it happened. ".repeat(8),
    );
    add(
        &mut b,
        "response",
        "asked",
        "Answer the question, then stop.",
    );
    (b, (p.layer, p.group))
}

/// **What follows the bundled schema's collections stands at one position**:
/// for a character of a few words and one of a paragraph, under each response
/// template, the grounding after them is read where the creation walk lays it
/// down — in front of `mood`, the last of them, which holds nothing.
#[test]
fn the_bundled_grounding_stands_still() {
    let (b, target) = bundled();
    let mut r = sealed(&b, &|text: &str| text.split_whitespace().count());
    let grounding = b.id_for_system_section("grounding").unwrap();
    let mood = b.id_for_system_collection("mood").unwrap();
    let walked = walked_to_after(&b, &r, mood);
    let templates: Vec<SectionId> = ["terse", "told", "asked"]
        .iter()
        .map(|m| b.id_for_system_section(&format!("response/{m}")).unwrap())
        .collect();
    let mut seen = Vec::new();
    for who in ["brief", "full"] {
        for &pick in &templates {
            r.scores = templates
                .iter()
                .map(|&t| (t, if t == pick { 900.0 } else { 600.0 }))
                .collect();
            let mut sel = SelectionState::new();
            sel.select(identity::ANCHOR, format!("{}/{who}", identity::ANCHOR));
            sel.select(identity::WHO, format!("{}/{who}", identity::WHO));
            seen.push(position_of(&b, &r, target, &sel, grounding));
        }
    }
    assert_eq!(seen, vec![walked; 6], "the grounding moved");
}

/// Every turn shape a live character projects under: each personality and
/// character, awake or dreaming, each stance, each response length and
/// deliberation, each talk template's destination, on a mission or not.
fn live_selections(live: &Live, personalities: &[&str]) -> Vec<SelectionState> {
    let mut out = Vec::new();
    let building = live
        .identities
        .buildings
        .first()
        .cloned()
        .unwrap_or_default();
    let thinking = [
        Deliberation::None,
        Deliberation::Quick,
        Deliberation::Balanced,
        Deliberation::Deep,
        Deliberation::Exhaustive,
    ];
    let lengths = ["terse", "concise", "standard", "detailed", "comprehensive"];
    let stances = ["acting", "talking", "reflecting", "dreaming"];
    for (npc, personality) in personalities.iter().enumerate() {
        let npc = npc as u64 + 1;
        for &deliberation in &thinking {
            for length in lengths {
                for stance in stances {
                    for dest in Destination::ALL {
                        for on_mission in [false, true] {
                            let mut sel = match stance {
                                "dreaming" => {
                                    live.identities
                                        .dreaming_selection_for(npc, personality, WORLD)
                                }
                                _ => live.identities.selection_for(
                                    npc,
                                    personality,
                                    WORLD,
                                    &building,
                                    deliberation,
                                ),
                            };
                            if stance == "acting" || stance == "talking" {
                                live.identities.act_as(&mut sel, personality, on_mission);
                            }
                            sel.select("stance", stance);
                            sel.select("response_length", length);
                            live.templates.offer(&mut sel, dest);
                            out.push(sel);
                        }
                    }
                }
            }
        }
    }
    out
}

/// **What follows the live mind's collections stands at one position**: the
/// close of the system block, after the act list, is read at the same position
/// for a bodiless mind and a body of a few lines, for a character block of a
/// line and one of a paragraph, under every stance, length, deliberation and
/// template the daemon sets, and on a mission or not — and it is where the
/// creation walk lays it down.
#[test]
#[ignore = "reads the live mind checked out beside this repository and the model pack's tokenizer"]
fn what_follows_the_live_minds_collections_stands_still() {
    let mind = live_mind();
    let personalities = ["zen-moon", "maker"];
    let authored = Authored {
        characters: vec![
            (1, "You are Zen.".to_string()),
            (2, "You are Perrin, a Maker of the vault. ".repeat(12)),
        ],
        ..authored(&mind)
    };
    let live = installed(&mind, &authored);
    let b = &live.builder;
    let count = token_counter();
    let mut r = sealed(b, &count);
    for c in b
        .schema()
        .system_prompt
        .items
        .iter()
        .filter_map(|it| match it {
            SystemPromptItem::Collection(c) => Some(c),
            _ => None,
        })
    {
        for (i, s) in c.sections.iter().enumerate() {
            r.scores.insert(s.id, 900.0 - i as f32);
        }
    }
    let close = b.id_for_system_section("system_close").unwrap();
    let tools = b.id_for_system_collection("tools").unwrap();
    let walked = walked_to_after(b, &r, tools);
    let target = (live.layer, live.group);
    let selections = live_selections(&live, &personalities);
    let moved: Vec<usize> = selections
        .iter()
        .map(|sel| position_of(b, &r, target, sel, close))
        .filter(|&at| at != walked)
        .collect();
    assert!(
        moved.is_empty(),
        "of {} turn shapes, {} read the close somewhere other than {walked}: {:?}",
        selections.len(),
        moved.len(),
        &moved[..moved.len().min(8)]
    );
}
