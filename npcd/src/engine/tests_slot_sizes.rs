//! The live mind's system prompt holds what it is given, measured in the
//! model's own tokens (`docs/placed_section_layout.md` §3.1c): every
//! collection the daemon fills has a range its members fit, and every section
//! and tree node whose text can change declares a `min_tokens` its text fits.
//!
//! An item past its slot moves everything after it, and the K/V sealed behind
//! the old positions is stale; nothing at run time fails, so this measures it.
//! The members are installed as the loader installs them — the identities and
//! frames from the personalities and worlds, a building per part of each map,
//! the talk templates and every act — so what is measured is what a character
//! reads. The members the loader cannot know until a character lives — its
//! character block, its mission's wording, its journal, channels and catch-up
//! — are bounded by their ranges and warned of at projection when they run
//! over (`emit_in_range`).

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use candle_conversation::models::{cache_root, ResolvedModel};
use candle_conversation::projection::{
    Builder, GroupId, LayerId, SectionCollection, SectionSchema, SystemPromptItem, TreeNode,
};
use npc_map::{describe, MapSet};
use serde_json::Value;

use crate::engine::agenda::ordered::ORDERED_ACTS;
use crate::engine::identity::{self, Authored};
use crate::engine::talk::templates;
use crate::engine::{prompt, reflect, schema, tools};
use crate::mind::files::MindFiles;
use crate::model::MODEL;
use crate::registry::Registry;

/// The world the schema's `{workspace}` names.
pub(super) const WORLD: &str = "battle-cities";

/// Qwen3.6's native context, which picks a range's full size.
pub(super) const NATIVE: usize = 262_144;

/// The collections the daemon declares and never fills: each holds nothing, so
/// each takes no positions and moves nothing, and a range there would be a
/// hole for nothing.
const NEVER_FILLED: &[&str] = &["mood", "task"];

/// The collections filled per conversation as a character lives, which the
/// loader installs at most a stand-in for. Their ranges bound them at run time.
const LIVED: &[&str] = &["identity", "mission", "journal", "channels", "catchup"];

/// The mind checked out beside this repository.
pub(super) fn live_mind() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../mind")
}

/// How many tokens the model npcd runs reads `text` as, from the tokenizer in
/// its model pack.
pub(super) fn token_counter() -> impl Fn(&str) -> usize {
    let dir = cache_root().join(MODEL.spec().model_repo.replace('/', "--"));
    let pack = std::fs::read_dir(&dir)
        .unwrap_or_else(|e| panic!("the model pack's folder {}: {e}", dir.display()))
        .filter_map(Result::ok)
        .map(|e| e.path())
        .find(|p| p.to_string_lossy().ends_with(".pack.gguf"))
        .unwrap_or_else(|| panic!("no model pack in {}", dir.display()));
    let tokenizer = ResolvedModel::open(pack)
        .and_then(|m| m.tokenizer())
        .expect("the model pack carries its tokenizer");
    move |text| {
        tokenizer
            .encode(text, false)
            .expect("the tokenizer encodes the mind's text")
            .len()
    }
}

/// The projection the loader builds from a mind, and what it installed.
pub(super) struct Live {
    pub builder: Builder,
    pub identities: identity::Installed,
    pub templates: templates::Installed,
    /// The live conversation layer and its group.
    pub layer: LayerId,
    pub group: GroupId,
}

/// The projection the loader builds from the mind at `mind`: the schema, every
/// act, the identities and frames from `authored`, a building and a being per
/// part of every map, and the talk templates.
pub(super) fn installed(mind: &Path, authored: &Authored) -> Live {
    let files = MindFiles::folder(mind);
    let p = schema::build(Some(&files), WORLD).expect("the live mind's schema parses");
    let mut builder = p.builder;
    let acts = tools::CATALOG
        .iter()
        .chain(ORDERED_ACTS)
        .chain(reflect::ASKED);
    tools::install(&mut builder, acts, |_, _| {}).expect("every act installs");
    let identities =
        identity::install(&mut builder, authored, &places(mind)).expect("the identities install");
    let templates =
        templates::install(&mut builder, Some(&files)).expect("the talk templates install");
    Live {
        builder,
        identities,
        templates,
        layer: p.layer,
        group: p.group,
    }
}

/// What the loader reads from the personalities and worlds for the identity
/// collections: each personality's anchor, inner life, voice and acting, and
/// each world's setting.
pub(super) fn authored(mind: &Path) -> Authored {
    let personalities =
        Registry::load("personality", mind.join("personalities")).expect("the personalities");
    let worlds = Registry::load("world", mind.join("worlds")).expect("the worlds");
    let field = |b: &Value, k: &str| {
        b.get(k)
            .and_then(Value::as_str)
            .unwrap_or_default()
            .to_string()
    };
    let of = |k: &str| -> Vec<(String, String)> {
        personalities
            .iter()
            .map(|p| (p.id.clone(), field(&p.body, k)))
            .collect()
    };
    Authored {
        anchors: of("anchor"),
        inner_lives: of("inner_life"),
        settings: worlds
            .iter()
            .map(|w| (w.id.clone(), field(&w.body, "setting")))
            .collect(),
        voices: of("voice"),
        actings: of("acting"),
        ..Authored::default()
    }
}

/// Every part of every map, as a body standing in it reads it and as the mind
/// that is the place reads it: the two members the loader installs per part.
fn places(mind: &Path) -> BTreeMap<String, String> {
    let mut out = BTreeMap::new();
    let maps = std::fs::read_dir(mind.join("map")).expect("the mind's maps");
    for world in maps.filter_map(Result::ok).filter(|e| e.path().is_dir()) {
        let id = world.file_name().to_string_lossy().into_owned();
        let set = MapSet::load_dir(world.path()).unwrap_or_else(|e| panic!("map {id}: {e:#}"));
        for (part, text) in describe::places(&set) {
            let key = identity::building_key(&id, &part);
            out.insert(identity::whole_key(&key), prompt::being(&text));
            out.insert(key, prompt::building(&text));
        }
    }
    out
}

/// One item that does not fit the slot the schema gives it, or that is given
/// none and needs one.
fn misfit(name: &str, tokens: usize, slot: Option<usize>) -> Option<String> {
    match slot {
        Some(slot) if tokens <= slot => None,
        Some(slot) => Some(format!("{name}: {tokens} tokens past its slot of {slot}")),
        None => Some(format!("{name}: {tokens} tokens and no slot declared")),
    }
}

/// A collection's tokens at the most it ever places: its mandatory members and
/// its `most` largest others, or its largest member when it places one.
fn collection_tokens(c: &SectionCollection, tokens: &impl Fn(&str) -> usize) -> usize {
    let mut sizes: Vec<usize> = c
        .sections
        .iter()
        .filter(|s| !c.mandatory.contains(&s.id))
        .map(|s| tokens(&s.content))
        .collect();
    sizes.sort_unstable_by(|a, b| b.cmp(a));
    let most = c.range.and_then(|r| r.most).unwrap_or(1);
    let mandatory: usize = c
        .sections
        .iter()
        .filter(|s| c.mandatory.contains(&s.id))
        .map(|s| tokens(&s.content))
        .sum();
    mandatory + sizes.iter().take(most).sum::<usize>()
}

/// A section that is fixed dialect text: its slot is its own tokens.
fn is_marker(s: &SectionSchema) -> bool {
    s.is_template
}

/// A tree node's longest option.
fn node_tokens(n: &TreeNode, tokens: &impl Fn(&str) -> usize) -> usize {
    n.options
        .iter()
        .map(|o| tokens(&o.content))
        .max()
        .unwrap_or(0)
}

/// **Every item of the live mind's system prompt fits its slot**: each
/// collection the daemon fills, the largest it places against its range; each
/// section and tree node whose text can change, its text against its
/// `min_tokens`. Prints what each takes, so a margin can be read off it.
#[test]
#[ignore = "reads the live mind checked out beside this repository and the model pack's tokenizer"]
fn the_live_minds_items_fit_their_slots() {
    let tokens = token_counter();
    let mind = live_mind();
    let b = installed(&mind, &authored(&mind)).builder;
    let mut misfits = Vec::new();
    for item in &b.schema().system_prompt.items {
        match item {
            SystemPromptItem::Section(s) if is_marker(s) => {}
            SystemPromptItem::Section(s) => {
                let t = tokens(&s.content);
                println!("section {:<20} {t:>6} of {}", s.name, s.min_tokens);
                misfits.extend(misfit(
                    &s.name,
                    t,
                    (s.min_tokens > 0).then_some(s.min_tokens),
                ));
            }
            SystemPromptItem::SectionTree(tree) => {
                for n in &tree.nodes {
                    let t = node_tokens(n, &tokens);
                    println!("node    {:<20} {t:>6} of {}", n.name, n.min_tokens);
                    misfits.extend(misfit(
                        &n.name,
                        t,
                        (n.min_tokens > 0).then_some(n.min_tokens),
                    ));
                }
            }
            SystemPromptItem::Collection(c) if NEVER_FILLED.contains(&c.name.as_str()) => {
                assert!(c.sections.is_empty(), "{} is filled after all", c.name);
                assert!(
                    c.range.is_none(),
                    "{} is never filled and needs no range",
                    c.name
                );
            }
            SystemPromptItem::Collection(c) => {
                let t = collection_tokens(c, &tokens);
                let range = c.range.map(|r| r.positions(Some(NATIVE)));
                println!(
                    "range   {:<20} {t:>6} of {} ({} members{})",
                    c.name,
                    range.unwrap_or(0),
                    c.sections.len(),
                    if LIVED.contains(&c.name.as_str()) {
                        ", more as it lives"
                    } else {
                        ""
                    }
                );
                misfits.extend(misfit(&c.name, t, range));
            }
        }
    }
    assert!(misfits.is_empty(), "{}", misfits.join("\n"));
}
