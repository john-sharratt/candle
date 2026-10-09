//! What the command table's generator is configured to do: `<mind>/missions.yaml`.
//!
//! The file is authored content, like `projection.yaml`, so how a mission is
//! asked for can change without touching the engine. It names the generators —
//! each a kind of work and the prompt that writes one mission of it — how
//! heavily each is drawn, and how many generated missions wait at the table.
//!
//! **No template slots.** A prompt is the instruction alone; the engine appends
//! what the mission is about (the character, its life so far, the documents in
//! question) under headings of its own — see [`super::material`]. A slot in the
//! prompt would be a second place the material's shape is decided, and the two
//! would drift.

use std::collections::BTreeSet;
use std::io::ErrorKind;
use std::path::Path;

use serde::Deserialize;

use super::target::Kind;

/// The file a mind configures its generator in.
pub const FILE: &str = "missions.yaml";

/// How many generated missions wait at the table when the file does not say.
const DEFAULT_KEEP: usize = 4;

/// One configured generator: a kind of work and the prompt that asks for it.
#[derive(Clone, Debug, PartialEq, Eq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Generator {
    /// Its name — what a generated mission's origin and the ledger carry.
    pub id: String,
    /// What it finds work in, which decides the material it is handed.
    pub kind: Kind,
    /// How often it is drawn against the others. Zero switches it off.
    #[serde(default = "one")]
    pub weight: u32,
    /// What the model is asked, before the material.
    pub prompt: String,
}

fn one() -> u32 {
    1
}

/// The whole configuration.
#[derive(Clone, Debug, PartialEq, Eq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Config {
    /// The voice every generator writes in.
    pub system: String,
    /// How many generated missions to keep waiting at the table.
    #[serde(default = "default_keep")]
    pub keep: usize,
    /// What the table is asked when it reads a draft, before a review is set —
    /// see [`super::reading`].
    pub reading: String,
    /// The voice a Maker writes a piece in when it sits down to compose —
    /// see [`crate::engine::compose`].
    pub writing: String,
    pub generators: Vec<Generator>,
}

fn default_keep() -> usize {
    DEFAULT_KEEP
}

impl Config {
    /// Parse and check a configuration.
    pub fn parse(yaml: &str) -> Result<Config, String> {
        let c: Config = serde_yaml::from_str(yaml).map_err(|e| format!("{FILE}: {e}"))?;
        if c.system.trim().is_empty() {
            return Err(format!("{FILE}: `system` is empty"));
        }
        if c.reading.trim().is_empty() {
            return Err(format!("{FILE}: `reading` is empty"));
        }
        if c.writing.trim().is_empty() {
            return Err(format!("{FILE}: `writing` is empty"));
        }
        if c.generators.is_empty() {
            return Err(format!("{FILE}: no generators"));
        }
        let mut seen = BTreeSet::new();
        for g in &c.generators {
            if g.id.trim().is_empty() || g.prompt.trim().is_empty() {
                return Err(format!("{FILE}: a generator needs an `id` and a `prompt`"));
            }
            if !seen.insert(g.id.as_str()) {
                return Err(format!("{FILE}: generator `{}` is named twice", g.id));
            }
        }
        if c.generators.iter().all(|g| g.weight == 0) {
            return Err(format!("{FILE}: every generator has weight 0"));
        }
        Ok(c)
    }

    /// Read `<mind>/missions.yaml`. `Ok(None)` when the mind has none — a mind
    /// with no generator gives out the routine bank alone, as it always has.
    pub fn load(mind: &Path) -> Result<Option<Config>, String> {
        let path = mind.join(FILE);
        match std::fs::read_to_string(&path) {
            Ok(text) => Config::parse(&text).map(Some),
            Err(e) if e.kind() == ErrorKind::NotFound => Ok(None),
            Err(e) => Err(format!("{}: {e}", path.display())),
        }
    }

    /// The generator by name.
    pub fn generator(&self, id: &str) -> Option<&Generator> {
        self.generators.iter().find(|g| g.id == id)
    }

    /// The generators in the order the `turn`th draw tries them: weighted
    /// round-robin, so over a full cycle each is first in proportion to its
    /// weight, and the rest follow in declaration order as fallbacks for when the
    /// first finds no work.
    pub fn order(&self, turn: u64) -> Vec<&Generator> {
        let cycle: Vec<&Generator> = self
            .generators
            .iter()
            .flat_map(|g| std::iter::repeat_n(g, g.weight as usize))
            .collect();
        let first = cycle[(turn % cycle.len() as u64) as usize];
        std::iter::once(first)
            .chain(
                self.generators
                    .iter()
                    .filter(|g| g.weight > 0 && g.id != first.id),
            )
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const YAML: &str = "\
system: You set work.
keep: 3
reading: Read the draft.
writing: You write for the record.
generators:
  - id: lives
    kind: life_event
    weight: 2
    prompt: Find the next event.
  - id: boundaries
    kind: contradiction
    prompt: Find what disagrees.
  - id: off
    kind: gap
    weight: 0
    prompt: Never drawn.
";

    #[test]
    fn a_configuration_parses_with_its_defaults() {
        let c = Config::parse(YAML).unwrap();
        assert_eq!(c.keep, 3);
        assert_eq!(c.generators.len(), 3);
        assert_eq!(c.generators[0].kind, Kind::LifeEvent);
        assert_eq!(c.generators[1].weight, 1, "weight defaults to one");
        assert_eq!(c.generator("boundaries").unwrap().kind, Kind::Contradiction);
    }

    /// **Drawn by weight, never the one switched off, and every other as a
    /// fallback.** Over three turns `lives` (weight 2) leads twice and
    /// `boundaries` once.
    #[test]
    fn the_draw_order_follows_the_weights() {
        let c = Config::parse(YAML).unwrap();
        let firsts: Vec<&str> = (0..3).map(|t| c.order(t)[0].id.as_str()).collect();
        assert_eq!(firsts, ["lives", "lives", "boundaries"]);
        let ids: Vec<&str> = c.order(2).iter().map(|g| g.id.as_str()).collect();
        assert_eq!(ids, ["boundaries", "lives"]);
    }

    #[test]
    fn a_broken_configuration_is_refused_with_why() {
        for (yaml, why) in [
            ("system: ''\nreading: r\nwriting: w\ngenerators: []\n", "`system` is empty"),
            ("system: x\nreading: ' '\nwriting: w\ngenerators: []\n", "`reading` is empty"),
            ("system: x\nreading: r\nwriting: ' '\ngenerators: []\n", "`writing` is empty"),
            ("system: x\nwriting: w\ngenerators: []\n", "missing field `reading`"),
            ("system: x\nreading: r\ngenerators: []\n", "missing field `writing`"),
            ("system: x\nreading: r\nwriting: w\ngenerators: []\n", "no generators"),
            (
                "system: x\nreading: r\nwriting: w\ngenerators:\n  - {id: a, kind: gap, prompt: p}\n  - {id: a, kind: gap, prompt: q}\n",
                "named twice",
            ),
            (
                "system: x\nreading: r\nwriting: w\ngenerators:\n  - {id: a, kind: gap, weight: 0, prompt: p}\n",
                "weight 0",
            ),
            (
                "system: x\nreading: r\nwriting: w\ngenerators:\n  - {id: a, kind: review, prompt: p}\n",
                "unknown variant",
            ),
        ] {
            let e = Config::parse(yaml).unwrap_err();
            assert!(e.contains(why), "{e}");
        }
    }

    #[test]
    fn a_mind_with_no_file_has_no_generator() {
        let dir = tempfile::tempdir().unwrap();
        assert_eq!(Config::load(dir.path()), Ok(None));
        std::fs::write(dir.path().join(FILE), YAML).unwrap();
        assert!(Config::load(dir.path()).unwrap().is_some());
    }
}
