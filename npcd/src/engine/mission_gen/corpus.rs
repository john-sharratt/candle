//! What the mind holds, read once per generation: the eras, every character's
//! life, the stories, as the generator needs them.
//!
//! Plain data read straight off the mind folder — the same documents a Maker
//! reads and writes at a bench — so a target is chosen from what is actually on
//! the page, including whatever a Maker committed a minute ago.

use std::path::{Path, PathBuf};

use super::fingerprint::Fingerprint;
use crate::engine::life;

/// How many years apart two written events must be for the years between them
/// to count as a stretch the next event belongs in.
pub const STRETCH_MIN_YEARS: u32 = 3;

/// One era of the world's history.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Era {
    /// Its mind path, `layers/eras/the-salvation.md`.
    pub path: String,
    /// Its own title, from its first heading.
    pub title: String,
    /// The year it opens, from its `… 2787 CE` line.
    pub year: Option<u32>,
    pub text: String,
}

/// One written event in a character's life.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Event {
    pub date: String,
    pub title: String,
    pub path: String,
}

impl Event {
    /// The year it falls in.
    pub fn year(&self) -> Option<u32> {
        self.date.get(..4)?.parse().ok()
    }
}

/// A character whose life the world holds.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Life {
    /// The personality id — `layers/life/<who>/`.
    pub who: String,
    pub name: String,
    /// Who they are, from their personality's anchor.
    pub anchor: String,
    /// Their life story, from `layers/memory/<who>/life-story.md`, when written.
    pub story: Option<String>,
    /// The events already written, in order.
    pub events: Vec<Event>,
}

impl Life {
    /// The longest run of years between two written events with nothing
    /// written in it, as the first and last unwritten year — `None` when no two
    /// written years are more than [`STRETCH_MIN_YEARS`] apart.
    ///
    /// **Where the next event goes.** Left to choose, the model wrote the day
    /// after the latest event every time — a Mech with one event written and
    /// three hundred and ninety-five years of service before it was given the
    /// afternoon that followed it. A life is filled from its largest silences.
    pub fn longest_stretch(&self) -> Option<(u32, u32)> {
        let mut years: Vec<u32> = self.events.iter().filter_map(Event::year).collect();
        years.sort_unstable();
        years.dedup();
        years
            .windows(2)
            .filter(|w| w[1] - w[0] > STRETCH_MIN_YEARS)
            .max_by_key(|w| w[1] - w[0])
            .map(|w| (w[0] + 1, w[1] - 1))
    }

    /// What the life holds, as a number that changes when an event is added,
    /// removed or renamed.
    pub fn fingerprint(&self) -> u64 {
        let mut h = Fingerprint::new();
        for e in &self.events {
            h.add(&e.path);
        }
        h.finish()
    }
}

/// A told story.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Story {
    pub path: String,
    pub title: String,
    pub text: String,
}

/// Everything the generator reads.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Corpus {
    pub root: PathBuf,
    /// The world's own `setting`, from `worlds/<id>.yaml` — what kind of place
    /// this is, which a story or a life written in it must not step outside.
    pub setting: Option<String>,
    /// The documents Makers have written for generated missions, oldest first —
    /// what a review reads. Not read off the disk: the ledger knows which
    /// documents are Makers' work, and the caller hands them in.
    pub reviewable: Vec<String>,
    /// In the order they happened.
    pub eras: Vec<Era>,
    /// By personality id.
    pub lives: Vec<Life>,
    pub stories: Vec<Story>,
}

impl Corpus {
    /// Read the mind at `root`, as the world `world` sees it.
    ///
    /// **The cast is whoever has a life or a memory folder.** A personality with
    /// neither is not somebody in this world's history — a Maker, which writes
    /// the world and is not in it, is the case that matters.
    pub fn read(root: &Path, world: &str) -> Corpus {
        let layers = root.join("layers");
        let setting = std::fs::read_to_string(root.join("worlds").join(format!("{world}.yaml")))
            .ok()
            .and_then(|t| serde_yaml::from_str::<serde_yaml::Value>(&t).ok())
            .and_then(|y| y.get("setting")?.as_str().map(|s| s.trim().to_string()))
            .filter(|s| !s.is_empty());
        let mut eras: Vec<Era> = docs(&layers.join("eras"))
            .into_iter()
            .map(|(name, text)| Era {
                path: format!("layers/eras/{name}"),
                title: heading(&text).unwrap_or_else(|| name.trim_end_matches(".md").into()),
                year: era_year(&text),
                text,
            })
            .collect();
        eras.sort_by_key(|e| (e.year.unwrap_or(u32::MAX), e.path.clone()));

        let stories = docs(&layers.join("stories"))
            .into_iter()
            .map(|(name, text)| Story {
                path: format!("layers/stories/{name}"),
                title: heading(&text).unwrap_or_else(|| name.trim_end_matches(".md").into()),
                text,
            })
            .collect();

        let mut lives = Vec::new();
        for (who, yaml) in personalities(&root.join("personalities")) {
            let life_dir = layers.join("life").join(&who);
            let memory_dir = layers.join("memory").join(&who);
            if !life_dir.is_dir() && !memory_dir.is_dir() {
                continue;
            }
            let (episodes, _) = life::episodes(&life_dir, &who);
            let events = episodes
                .into_iter()
                .filter_map(|e| {
                    let file = e.path.file_name()?.to_str()?.to_string();
                    Some(Event {
                        date: e.date,
                        title: e.title,
                        path: format!("layers/life/{who}/{file}"),
                    })
                })
                .collect();
            let field = |k: &str| yaml.get(k).and_then(|v| v.as_str()).unwrap_or_default();
            lives.push(Life {
                name: match field("name") {
                    "" => who.clone(),
                    n => n.to_string(),
                },
                anchor: field("anchor").trim().to_string(),
                story: std::fs::read_to_string(memory_dir.join("life-story.md")).ok(),
                events,
                who,
            });
        }
        lives.sort_by(|a, b| a.who.cmp(&b.who));
        Corpus {
            root: root.to_path_buf(),
            setting,
            reviewable: Vec::new(),
            eras,
            lives,
            stories,
        }
    }

    /// A document's text by mind path, when it is one this corpus holds or the
    /// disk has.
    pub fn text(&self, path: &str) -> Option<String> {
        if let Some(e) = self.eras.iter().find(|e| e.path == path) {
            return Some(e.text.clone());
        }
        if let Some(s) = self.stories.iter().find(|s| s.path == path) {
            return Some(s.text.clone());
        }
        std::fs::read_to_string(self.root.join(path)).ok()
    }

    /// Whether a document exists, by mind path.
    pub fn exists(&self, path: &str) -> bool {
        self.root.join(path).is_file()
    }

    /// The world's present: the year its latest era opens. Nothing in a life
    /// is written after it.
    pub fn present(&self) -> Option<u32> {
        self.eras.iter().filter_map(|e| e.year).max()
    }

    /// The era a year falls in: the last that opened on or before it.
    pub fn era_of(&self, year: u32) -> Option<&Era> {
        self.eras
            .iter()
            .rev()
            .find(|e| e.year.is_some_and(|y| y <= year))
    }

    pub fn life(&self, who: &str) -> Option<&Life> {
        self.lives.iter().find(|l| l.who == who)
    }
}

/// Every `.md` directly in `dir`, by file name, sorted, skipping the `_` notes.
fn docs(dir: &Path) -> Vec<(String, String)> {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return Vec::new();
    };
    let mut out: Vec<(String, String)> = entries
        .flatten()
        .filter_map(|e| {
            let name = e.file_name().to_str()?.to_string();
            if !name.ends_with(".md") || name.starts_with('_') || !e.path().is_file() {
                return None;
            }
            Some((name, std::fs::read_to_string(e.path()).ok()?))
        })
        .collect();
    out.sort();
    out
}

/// Every personality file, by id.
fn personalities(dir: &Path) -> Vec<(String, serde_yaml::Value)> {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return Vec::new();
    };
    let mut out: Vec<(String, serde_yaml::Value)> = entries
        .flatten()
        .filter_map(|e| {
            let name = e.file_name().to_str()?.to_string();
            let id = name.strip_suffix(".yaml")?.to_string();
            let yaml = serde_yaml::from_str(&std::fs::read_to_string(e.path()).ok()?).ok()?;
            Some((id, yaml))
        })
        .collect();
    out.sort_by(|a, b| a.0.cmp(&b.0));
    out
}

/// A document's first `# ` heading.
fn heading(text: &str) -> Option<String> {
    text.lines()
        .find_map(|l| l.strip_prefix("# "))
        .map(|h| h.trim().to_string())
}

/// The year an era opens: the number before the first `CE`.
fn era_year(text: &str) -> Option<u32> {
    let at = text.find(" CE")?;
    let head = &text[..at];
    let digits: String = head
        .chars()
        .rev()
        .take_while(|c| c.is_ascii_digit())
        .collect::<Vec<_>>()
        .into_iter()
        .rev()
        .collect();
    digits.parse().ok()
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    /// A small mind on disk: three eras, two characters with lives, a Maker
    /// with neither, and one story.
    pub fn mind() -> tempfile::TempDir {
        let dir = tempfile::tempdir().unwrap();
        let w = |path: &str, text: &str| {
            let p = dir.path().join(path);
            std::fs::create_dir_all(p.parent().unwrap()).unwrap();
            std::fs::write(p, text).unwrap();
        };
        w(
            "layers/eras/the-fall.md",
            "# The Fall\n\n**Era 0 · 2487 CE**\n\nThe sky went out over every world at once.\n",
        );
        w(
            "layers/eras/the-salvation.md",
            "# The Salvation\n\n**Era 300 · 2787 CE**\n\nThe towers were built.\n",
        );
        w(
            "layers/eras/the-retreat.md",
            "# The Retreat\n\n**Era 120 · 2607 CE**\n\nEveryone went underground.\n",
        );
        w("layers/eras/_notes.md", "not an era");
        w(
            "layers/stories/the-charge.md",
            "# The Charge\n\nKeeper is given the plan.\n",
        );
        w(
            "personalities/keeper.yaml",
            "id: keeper\nname: Keeper\nanchor: |\n  You keep the towers.\n",
        );
        w(
            "personalities/kaelor.yaml",
            "id: kaelor\nname: Kaelor\nanchor: You lead.\n",
        );
        w(
            "personalities/maker.yaml",
            "id: maker\nname: Maker\nanchor: You write a world.\n",
        );
        w(
            "layers/life/keeper/2487-03-08 The Second the Sky Went Out.md",
            "We were reconciling a water schedule.\n",
        );
        w(
            "layers/life/keeper/2786 The Charge.md",
            "We were given the plan.\n",
        );
        w(
            "layers/memory/kaelor/life-story.md",
            "Kaelor fought for three decades.\n",
        );
        w(
            "worlds/test.yaml",
            "id: test\nsetting: >-\n  A world of towers after the sky went out.\n",
        );
        dir
    }

    /// The fixture mind, read as its one world.
    pub fn corpus(dir: &tempfile::TempDir) -> Corpus {
        Corpus::read(dir.path(), "test")
    }

    #[test]
    fn the_corpus_reads_the_eras_in_order_and_the_cast_with_their_lives() {
        let dir = mind();
        let c = corpus(&dir);
        assert_eq!(
            c.setting.as_deref(),
            Some("A world of towers after the sky went out.")
        );
        let eras: Vec<(&str, Option<u32>)> =
            c.eras.iter().map(|e| (e.title.as_str(), e.year)).collect();
        assert_eq!(
            eras,
            [
                ("The Fall", Some(2487)),
                ("The Retreat", Some(2607)),
                ("The Salvation", Some(2787))
            ]
        );
        let cast: Vec<&str> = c.lives.iter().map(|l| l.who.as_str()).collect();
        assert_eq!(cast, ["kaelor", "keeper"], "a Maker is not in the world");
        let keeper = c.life("keeper").unwrap();
        assert_eq!(keeper.name, "Keeper");
        assert_eq!(keeper.anchor, "You keep the towers.");
        assert_eq!(
            keeper
                .events
                .iter()
                .map(|e| e.date.as_str())
                .collect::<Vec<_>>(),
            ["2487-03-08", "2786"]
        );
        assert_eq!(
            keeper.events[1].path,
            "layers/life/keeper/2786 The Charge.md"
        );
        assert_eq!(
            c.life("kaelor").unwrap().story.as_deref(),
            Some("Kaelor fought for three decades.\n")
        );
        assert_eq!(c.stories.len(), 1);
        assert_eq!(c.era_of(2700).unwrap().title, "The Retreat");
        assert_eq!(c.era_of(2000), None);
        assert!(c.exists("layers/stories/the-charge.md"));
        assert!(!c.exists("layers/stories/nothing.md"));
    }

    /// The widest silence between written years, and none when every written
    /// year is close to the next.
    #[test]
    fn the_longest_stretch_is_the_widest_silence_between_written_years() {
        let dir = mind();
        let c = corpus(&dir);
        assert_eq!(
            c.life("keeper").unwrap().longest_stretch(),
            Some((2488, 2785))
        );
        assert_eq!(
            c.life("kaelor").unwrap().longest_stretch(),
            None,
            "nothing written"
        );
        std::fs::write(dir.path().join("layers/life/keeper/2600 Middle.md"), "x").unwrap();
        let c = corpus(&dir);
        assert_eq!(
            c.life("keeper").unwrap().longest_stretch(),
            Some((2601, 2785))
        );
    }

    #[test]
    fn a_life_fingerprint_moves_when_an_event_is_added() {
        let dir = mind();
        let before = corpus(&dir).life("keeper").unwrap().fingerprint();
        std::fs::write(
            dir.path().join("layers/life/keeper/2500 Later.md"),
            "Later.",
        )
        .unwrap();
        let after = corpus(&dir).life("keeper").unwrap().fingerprint();
        assert_ne!(before, after);
    }
}
