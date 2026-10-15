//! What the mind holds, read once per generation: the eras, every character's
//! life, the stories, as the generator needs them.
//!
//! Plain data read straight off the mind folder — the same documents a Maker
//! reads and writes at a bench — so a target is chosen from what is actually on
//! the page, including whatever a Maker committed a minute ago.

use std::collections::BTreeMap;
use std::sync::Arc;

use super::fingerprint::Fingerprint;
use super::glossary::{self, Term};
use crate::engine::life;
use crate::mind::files::{MindFiles, MindRef};

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
    /// The mind as it stood when this corpus was read.
    pub mind: MindRef,
    /// The world's own `setting`, from `worlds/<id>.yaml` — what kind of place
    /// this is, which a story or a life written in it must not step outside.
    pub setting: Option<String>,
    /// In the order they happened.
    pub eras: Vec<Era>,
    /// By personality id.
    pub lives: Vec<Life>,
    pub stories: Vec<Story>,
    /// The world's own terms, from `layers/world/` ([`glossary`]).
    pub terms: Vec<Term>,
    /// Documents as an operation's store holds them over the published mind
    /// (`docs/replicated_substrate.md` §10.3), read before the disk.
    pub held: BTreeMap<String, String>,
}

impl Corpus {
    /// This corpus as operation work sees it: `path` holding `text`, the
    /// document as its store has it.
    pub fn holding(mut self, path: &str, text: String) -> Corpus {
        self.held.insert(path.to_string(), text);
        self
    }

    /// Read the mind at `root`, as the world `world` sees it.
    ///
    /// **The cast is whoever has a life or a memory folder.** A personality with
    /// neither is not somebody in this world's history — a Maker, which writes
    /// the world and is not in it, is the case that matters.
    pub fn read(root: &Arc<MindFiles>, world: &str) -> Corpus {
        let setting = root
            .read(&format!("worlds/{world}.yaml"))
            .and_then(|t| serde_yaml::from_str::<serde_yaml::Value>(&t).ok())
            .and_then(|y| y.get("setting")?.as_str().map(|s| s.trim().to_string()))
            .filter(|s| !s.is_empty());
        let mut eras: Vec<Era> = docs(root, "layers/eras")
            .into_iter()
            .map(|(name, text)| Era {
                path: format!("layers/eras/{name}"),
                title: heading(&text).unwrap_or_else(|| name.trim_end_matches(".md").into()),
                year: era_year(&text),
                text,
            })
            .collect();
        eras.sort_by_key(|e| (e.year.unwrap_or(u32::MAX), e.path.clone()));

        let stories = docs(root, "layers/stories")
            .into_iter()
            .map(|(name, text)| Story {
                path: format!("layers/stories/{name}"),
                title: heading(&text).unwrap_or_else(|| name.trim_end_matches(".md").into()),
                text,
            })
            .collect();

        let mut lives = Vec::new();
        for (who, yaml) in personalities(root) {
            let life_dir = format!("layers/life/{who}");
            let memory_dir = format!("layers/memory/{who}");
            if !root.is_dir(&life_dir) && !root.is_dir(&memory_dir) {
                continue;
            }
            let (episodes, _) = life::episodes(root, &life_dir, &who);
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
                story: root.read(&format!("{memory_dir}/life-story.md")),
                events,
                who,
            });
        }
        lives.sort_by(|a, b| a.who.cmp(&b.who));
        Corpus {
            mind: MindRef(Some(Arc::clone(root))),
            setting,
            eras,
            lives,
            stories,
            terms: glossary::terms(root),
            held: BTreeMap::new(),
        }
    }

    /// A document's text by mind path, when it is one this corpus holds or the
    /// disk has.
    pub fn text(&self, path: &str) -> Option<String> {
        if let Some(t) = self.held.get(path) {
            return Some(t.clone());
        }
        if let Some(e) = self.eras.iter().find(|e| e.path == path) {
            return Some(e.text.clone());
        }
        if let Some(s) = self.stories.iter().find(|s| s.path == path) {
            return Some(s.text.clone());
        }
        self.on_disk(path)
    }

    /// A document exactly as the published mind holds it, by mind path —
    /// `None` when there is none.
    pub fn on_disk(&self, path: &str) -> Option<String> {
        self.mind.0.as_ref()?.read(path)
    }

    /// Whether a document exists, by mind path.
    pub fn exists(&self, path: &str) -> bool {
        self.held.contains_key(path) || self.mind.0.as_ref().is_some_and(|m| m.is_file(path))
    }

    /// The world's present: the year its latest era opens. Nothing in a life
    /// is written after it.
    pub fn present(&self) -> Option<u32> {
        self.eras.iter().filter_map(|e| e.year).max()
    }

    /// The years `era` covers: from its opening to the year before the next
    /// era opens — `None` for the last, which runs to the present.
    pub fn span(&self, era: &Era) -> Option<(u32, Option<u32>)> {
        let from = era.year?;
        let to = self
            .eras
            .iter()
            .filter_map(|e| e.year)
            .filter(|&y| y > from)
            .min()
            .map(|y| y - 1);
        Some((from, to))
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
fn docs(mind: &MindFiles, dir: &str) -> Vec<(String, String)> {
    let mut out: Vec<(String, String)> = mind
        .list(dir)
        .into_iter()
        .filter_map(|name| {
            if !name.ends_with(".md") || name.starts_with('_') {
                return None;
            }
            let text = mind.read(&format!("{dir}/{name}"))?;
            Some((name, text))
        })
        .collect();
    out.sort();
    out
}

/// Every personality file, by id.
fn personalities(mind: &MindFiles) -> Vec<(String, serde_yaml::Value)> {
    let mut out: Vec<(String, serde_yaml::Value)> = mind
        .list("personalities")
        .into_iter()
        .filter_map(|name| {
            let id = name.strip_suffix(".yaml")?.to_string();
            let yaml = serde_yaml::from_str(&mind.read(&format!("personalities/{name}"))?).ok()?;
            Some((id, yaml))
        })
        .collect();
    out.sort_by(|a, b| a.0.cmp(&b.0));
    out
}

/// A document's first `# ` heading.
pub(crate) fn heading(text: &str) -> Option<String> {
    text.lines()
        .find_map(|l| l.strip_prefix("# "))
        .map(|h| h.trim().to_string())
}

/// The year an era opens: the first year of the date its first `CE` closes —
/// `**Era 350–450 · 2837–2937 CE**` opens in 2837, `**Era 0 · 2487 CE**` in
/// 2487. The date runs from the last `·` before the `CE` (or the line's start).
///
/// **The opening year, not the closing one.** Read as the digits just before
/// `CE`, a ranged era gave its last year, so every year inside it was placed in
/// the era before — a life set in 2850 was checked against the Concord.
pub(crate) fn era_year(text: &str) -> Option<u32> {
    let at = text.find(" CE")?;
    let line = &text[text[..at].rfind('\n').map_or(0, |i| i + 1)..at];
    let date = line.rsplit('·').next().unwrap_or(line);
    let digits: String = date
        .chars()
        .skip_while(|c| !c.is_ascii_digit())
        .take_while(|c| c.is_ascii_digit())
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
        w(
            "layers/world/tower.md",
            "# Tower\n\nA [tower](/tower) is a mobile fortress the minds live in.\n",
        );
        dir
    }

    /// The fixture mind, read as its one world.
    pub fn corpus(dir: &tempfile::TempDir) -> Corpus {
        Corpus::read(&Arc::new(MindFiles::folder(dir.path())), "test")
    }

    #[test]
    fn the_corpus_reads_the_eras_in_order_and_the_cast_with_their_lives() {
        let dir = mind();
        let c = corpus(&dir);
        assert_eq!(
            c.setting.as_deref(),
            Some("A world of towers after the sky went out.")
        );
        // The year an era opens, whichever way its date is written.
        assert_eq!(era_year("# T\n\n**Era 0 · 2487 CE**\n"), Some(2487));
        assert_eq!(era_year("**Era 350–450 · 2837–2937 CE**"), Some(2837));
        assert_eq!(era_year("**Era −336 to −1 · 2151–2486 CE**"), Some(2151));
        assert_eq!(era_year("**Era 600 · 3087 CE — present day**"), Some(3087));
        assert_eq!(era_year("In 2791 CE it ended."), Some(2791));
        assert_eq!(era_year("no date here"), None);
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

    /// **An operation's document is read as its store holds it**, over what
    /// the published mind says, and is there although no file is.
    #[test]
    fn a_held_document_is_read_over_the_published_mind() {
        let dir = mind();
        let c = corpus(&dir)
            .holding(
                "layers/stories/the-charge.md",
                "# The Charge\n\nRedrawn.\n".into(),
            )
            .holding("layers/stories/new.md", "# New\n\nA draft.\n".into());
        assert_eq!(
            c.text("layers/stories/the-charge.md").as_deref(),
            Some("# The Charge\n\nRedrawn.\n")
        );
        assert_eq!(
            c.text("layers/stories/new.md").as_deref(),
            Some("# New\n\nA draft.\n")
        );
        assert!(c.exists("layers/stories/new.md"));
        assert_eq!(c.on_disk("layers/stories/new.md"), None, "not published");
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
