//! Which shared material a character draws on, by what stands where it is.
//!
//! **Provenance follows the bench.** The world, the eras and the stories are
//! scored against every turn a character takes, and scored alone they answered
//! a Maker writing the Golden Age with six ammunition cards — a few dozen tokens
//! each, they match anything a little. What a character is doing decides what is
//! worth reaching for: at a story desk, the storyline and the stories told about
//! it; at the map table, the places; walking the halls, the places too, and
//! nothing of the chronicle.
//!
//! The mind says so in `bearings.yaml`: for each kind of part, which shared
//! layers it draws on, and of the world which topics. A layer a bearing does not
//! name offers nothing there. The first bearing whose parts stand in the room is
//! the one; a room with none of them — a corridor, a lift, a plant room — is
//! `elsewhere`. A mind with no `bearings.yaml` draws on everything everywhere.
//!
//! Each shared group is scoped ([`ConversationEngine::mark_group_scoped`]), and
//! before each turn its conversation is given the documents its bearing draws
//! on ([`ConversationEngine::set_retrieval_scope`]); the belief scan and the
//! selection both read only those.

use std::collections::{BTreeMap, HashMap, HashSet};
use std::path::Path;
use std::sync::{Arc, Mutex};

use candle_conversation::projection::{GroupId, TimelineId};
use candle_conversation::ConversationEngine;
use serde::Deserialize;

use crate::engine::chronology;

/// The file a mind declares its bearings in.
pub const FILE: &str = "bearings.yaml";

/// What a bearing draws from one layer: every document, or the world's
/// documents under the named topics.
#[derive(Clone, Debug, PartialEq, Eq, Deserialize)]
#[serde(untagged)]
pub enum Draw {
    /// `all`.
    Every(String),
    /// Topics: the folder under the layer (`world/locations/…` is
    /// `locations`), or a document's own name for one standing at its root
    /// (`world/cities`).
    Topics(Vec<String>),
}

impl Draw {
    fn admits(&self, address: &str) -> bool {
        match self {
            Draw::Every(_) => true,
            Draw::Topics(topics) => topic(address).is_some_and(|t| topics.iter().any(|x| x == t)),
        }
    }
}

/// Per layer, what is drawn.
pub type Draws = BTreeMap<String, Draw>;

#[derive(Debug, Deserialize)]
struct Rule {
    /// Part ids, any one of which standing in the room makes this the bearing.
    at: Vec<String>,
    #[serde(default)]
    draws: Draws,
}

/// How a layer answers to the year a character works in, set at a time machine.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Timed {
    /// The storyline: the era the year falls in and the one before it —
    /// nothing after the year, and not the distant past either.
    Era,
    /// Documents that say when they are set: those set on or before the year.
    /// One that does not say is held back, since nothing shows it is not later.
    Dated,
}

#[derive(Debug, Deserialize)]
struct File {
    bearings: Vec<Rule>,
    #[serde(default)]
    elsewhere: Draws,
    /// The layers that answer to a character's year, and how.
    #[serde(default)]
    timed: BTreeMap<String, Timed>,
}

/// A mind's bearings, in the order it declares them.
#[derive(Debug)]
pub struct Bearings {
    rules: Vec<Rule>,
    elsewhere: Draws,
    timed: BTreeMap<String, Timed>,
}

/// The name of the bearing a room with none of the declared parts falls to.
pub const ELSEWHERE: &str = "elsewhere";

impl Bearings {
    /// Read `<mind>/bearings.yaml`. `Ok(None)` when the mind has none.
    pub fn read(mind: &Path) -> anyhow::Result<Option<Self>> {
        match std::fs::read_to_string(mind.join(FILE)) {
            Ok(text) => Self::parse(&text).map(Some),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(None),
            Err(e) => Err(e.into()),
        }
    }

    pub fn parse(yaml: &str) -> anyhow::Result<Self> {
        let file: File = serde_yaml::from_str(yaml)?;
        let draws = file
            .bearings
            .iter()
            .map(|r| &r.draws)
            .chain(std::iter::once(&file.elsewhere));
        for d in draws.flat_map(|d| d.values()) {
            if let Draw::Every(word) = d {
                anyhow::ensure!(
                    word == "all",
                    "a layer draws `all` or a list of topics, not `{word}`"
                );
            }
        }
        Ok(Self {
            rules: file.bearings,
            elsewhere: file.elsewhere,
            timed: file.timed,
        })
    }

    /// The bearing of a room holding `parts`: its index (the count of rules
    /// for `elsewhere`), its name, and what it draws.
    pub fn of(&self, parts: &[String]) -> (usize, &str, &Draws) {
        self.rules
            .iter()
            .enumerate()
            .find(|(_, r)| r.at.iter().any(|a| parts.contains(a)))
            .map(|(i, r)| (i, r.at[0].as_str(), &r.draws))
            .unwrap_or((self.rules.len(), ELSEWHERE, &self.elsewhere))
    }
}

/// A document's topic within its layer: the folder beneath the layer, or its
/// own name at the layer's root. `world/locations/the-shelf` → `locations`;
/// `world/cities` → `cities`.
pub fn topic(address: &str) -> Option<&str> {
    let (_, rest) = address.split_once('/')?;
    Some(rest.split('/').next().unwrap_or(rest))
}

/// One shared document: the conversation it was written as, the address it was
/// ingested under, and the year it says it is set in.
pub struct Doc {
    pub timeline: TimelineId,
    pub address: String,
    pub year: Option<u32>,
}

/// One shared group's documents.
pub struct Shelf {
    pub layer: String,
    pub group: GroupId,
    pub docs: Vec<Doc>,
}

impl Shelf {
    /// Read `group`'s documents off the engine: each conversation's first turn
    /// is its document — the address, then the text it is dated by.
    pub fn read(engine: &ConversationEngine, layer: &str, group: GroupId) -> Self {
        let docs = engine
            .group_conversations(group)
            .into_iter()
            .filter_map(|t| {
                let (address, text) = engine.turn_texts(t).into_iter().next()?;
                Some(Doc {
                    timeline: t,
                    address,
                    year: chronology::dated(&text),
                })
            })
            .collect();
        Self {
            layer: layer.to_string(),
            group,
            docs,
        }
    }

    /// Whether `doc` is in reach of a character working in `year`, as this
    /// layer answers to time.
    fn in_time(&self, doc: &Doc, timed: Option<Timed>, year: Option<u32>) -> bool {
        match (timed, year) {
            (None, _) | (_, None) => true,
            (Some(Timed::Dated), Some(y)) => doc.year.is_some_and(|d| d <= y),
            (Some(Timed::Era), Some(y)) => {
                let mut opens: Vec<u32> = self.docs.iter().filter_map(|d| d.year).collect();
                opens.sort_unstable();
                opens.dedup();
                let Some(at) = opens.iter().rposition(|o| *o <= y) else {
                    return false;
                };
                doc.year
                    .is_some_and(|d| opens[at.saturating_sub(1)..=at].contains(&d))
            }
        }
    }
}

/// The documents one shared group offers at a bearing.
pub type Offered = (GroupId, Arc<HashSet<TimelineId>>);

/// A bearing (its index in the mind's list) and the year worked in there.
type Key = (usize, Option<u32>);

/// The shared groups and the bearings that scope them, with each bearing's
/// scopes built once per year a character works in.
pub struct Scopes {
    bearings: Bearings,
    shelves: Vec<Shelf>,
    built: Mutex<HashMap<Key, Arc<Vec<Offered>>>>,
}

impl Scopes {
    pub fn new(bearings: Bearings, shelves: Vec<Shelf>) -> Self {
        Self {
            bearings,
            shelves,
            built: Mutex::new(HashMap::new()),
        }
    }

    /// The address the shared document written as `timeline` was ingested
    /// under, if it is one.
    pub fn address(&self, timeline: u64) -> Option<&str> {
        self.shelves
            .iter()
            .flat_map(|s| s.docs.iter())
            .find(|d| d.timeline.raw() == timeline)
            .map(|d| d.address.as_str())
    }

    /// The groups to scope at setup.
    pub fn groups(&self) -> impl Iterator<Item = GroupId> + '_ {
        self.shelves.iter().map(|s| s.group)
    }

    /// The bearing of a room holding `parts`, and the documents each shared
    /// group offers there to a character working in `year` (`None`: the
    /// present) — an empty set for a layer the bearing does not draw on.
    pub fn at(&self, parts: &[String], year: Option<u32>) -> (&str, Arc<Vec<Offered>>) {
        let (index, name, draws) = self.bearings.of(parts);
        let mut built = self.built.lock().unwrap();
        let scopes = built.entry((index, year)).or_insert_with(|| {
            Arc::new(
                self.shelves
                    .iter()
                    .map(|shelf| {
                        let timed = self.bearings.timed.get(&shelf.layer).copied();
                        let allowed = match draws.get(&shelf.layer) {
                            Some(draw) => shelf
                                .docs
                                .iter()
                                .filter(|d| draw.admits(&d.address))
                                .filter(|d| shelf.in_time(d, timed, year))
                                .map(|d| d.timeline)
                                .collect(),
                            None => HashSet::new(),
                        };
                        (shelf.group, Arc::new(allowed))
                    })
                    .collect(),
            )
        });
        (name, Arc::clone(scopes))
    }

    /// Scope `timeline`'s turns to the bearing of a room holding `parts`, for
    /// a character working in `year`. Answers the bearing's name.
    pub fn apply(
        &self,
        engine: &ConversationEngine,
        timeline: TimelineId,
        parts: &[String],
        year: Option<u32>,
    ) -> String {
        let (name, scopes) = self.at(parts, year);
        for (group, allowed) in scopes.iter() {
            engine.set_retrieval_scope(timeline, *group, Arc::clone(allowed));
        }
        name.to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const YAML: &str = "\
bearings:
  - at: [story-desk, chronicle-terminal]
    draws:
      eras: all
      stories: all
      world: [history, factions]
  - at: [map-table]
    draws:
      world: [locations, cities]
elsewhere:
  world: [locations, cities]
timed:
  eras: era
  stories: dated
";

    fn parts(p: &[&str]) -> Vec<String> {
        p.iter().map(|s| s.to_string()).collect()
    }

    fn doc(timeline: u64, address: &str, year: Option<u32>) -> Doc {
        Doc {
            timeline: TimelineId::from_raw(timeline).unwrap(),
            address: address.into(),
            year,
        }
    }

    /// What `s` offers a character at `p` working in `year`, by group.
    fn offered(s: &Scopes, p: &[&str], year: Option<u32>) -> Vec<(GroupId, Vec<u64>)> {
        let (_, scopes) = s.at(&parts(p), year);
        scopes
            .iter()
            .map(|(g, a)| {
                let mut v: Vec<u64> = a.iter().map(|t| t.raw()).collect();
                v.sort();
                (*g, v)
            })
            .collect()
    }

    #[test]
    fn a_documents_topic_is_its_folder_or_its_own_name_at_the_root() {
        assert_eq!(topic("world/locations/the-shelf"), Some("locations"));
        assert_eq!(topic("world/cities"), Some("cities"));
        assert_eq!(topic("eras/the-fall"), Some("the-fall"));
        assert_eq!(topic("loose"), None);
    }

    /// **The first bearing whose part stands in the room is the one**, and a
    /// room with none of them is `elsewhere`.
    #[test]
    fn the_bearing_is_the_first_whose_part_is_here() {
        let b = Bearings::parse(YAML).unwrap();
        let (i, name, draws) = b.of(&parts(&["seat", "story-desk"]));
        assert_eq!((i, name), (0, "story-desk"));
        assert_eq!(draws.get("eras"), Some(&Draw::Every("all".into())));
        assert_eq!(b.of(&parts(&["map-table"])).1, "map-table");
        let (i, name, draws) = b.of(&parts(&[]));
        assert_eq!((i, name), (2, ELSEWHERE));
        assert_eq!(
            draws.get("eras"),
            None,
            "nothing of the chronicle in a corridor"
        );
    }

    #[test]
    fn a_layer_draws_all_or_topics_and_nothing_else() {
        let bad = "bearings:\n  - at: [x]\n    draws:\n      eras: some\n";
        assert!(Bearings::parse(bad)
            .unwrap_err()
            .to_string()
            .contains("not `some`"));
    }

    /// **What a bearing does not name, it does not offer**, and of the world
    /// only the topics it names.
    #[test]
    fn each_group_offers_only_what_the_bearing_draws() {
        let world = GroupId::from_raw(1).unwrap();
        let eras = GroupId::from_raw(2).unwrap();
        let shelves = vec![
            Shelf {
                layer: "world".into(),
                group: world,
                docs: vec![
                    doc(1, "world/ammo/shotgun", None),
                    doc(2, "world/history/the-burning", None),
                    doc(3, "world/cities", None),
                ],
            },
            Shelf {
                layer: "eras".into(),
                group: eras,
                docs: vec![doc(4, "eras/the-fall", Some(2487))],
            },
        ];
        let s = Scopes::new(Bearings::parse(YAML).unwrap(), shelves);
        assert_eq!(
            offered(&s, &["story-desk"], None),
            [(world, vec![2]), (eras, vec![4])]
        );
        assert_eq!(offered(&s, &[], None), [(world, vec![3]), (eras, vec![])]);
    }

    /// **Nothing after the year a character works in reaches it.** Standing in
    /// 2950 it recalls the era the year falls in and the one before — not the
    /// era after, and not the distant past — and only the stories that say they
    /// are set by then; the world's undated lore is untouched.
    #[test]
    fn a_year_keeps_the_future_out() {
        let world = GroupId::from_raw(1).unwrap();
        let eras = GroupId::from_raw(2).unwrap();
        let stories = GroupId::from_raw(3).unwrap();
        let shelves = vec![
            Shelf {
                layer: "world".into(),
                group: world,
                docs: vec![doc(1, "world/history/the-burning", None)],
            },
            Shelf {
                layer: "eras".into(),
                group: eras,
                docs: vec![
                    doc(10, "eras/the-golden-age", Some(2151)),
                    doc(11, "eras/the-awakening", Some(2487)),
                    doc(12, "eras/the-tower-age", Some(2837)),
                    doc(13, "eras/the-contested-cities", Some(2937)),
                    doc(14, "eras/the-spawn", Some(3087)),
                ],
            },
            Shelf {
                layer: "stories".into(),
                group: stories,
                docs: vec![
                    doc(20, "stories/the-ledger", Some(2792)),
                    doc(21, "stories/the-final-count", Some(3087)),
                    doc(22, "stories/what-vasko-knew", None),
                ],
            },
        ];
        let s = Scopes::new(Bearings::parse(YAML).unwrap(), shelves);
        assert_eq!(
            offered(&s, &["story-desk"], Some(2950)),
            [(world, vec![1]), (eras, vec![12, 13]), (stories, vec![20])]
        );
        assert_eq!(
            offered(&s, &["story-desk"], None),
            [
                (world, vec![1]),
                (eras, vec![10, 11, 12, 13, 14]),
                (stories, vec![20, 21, 22])
            ],
            "the present keeps everything"
        );
    }
}
