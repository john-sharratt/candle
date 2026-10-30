//! One entry of a journal (§3.2): a note, an episode, or a fold, written as a
//! list item with one field to a line and its text folded to the file's
//! width.

use std::fmt;
use std::ops::Range;

use sha2::{Digest, Sha256};

use crate::error::Problem;
use crate::period::{Level, Period};
use crate::scope::Channel;
use crate::slug::Slug;
use crate::time::Stamp;
use crate::yaml::{quote, scalar, text_field, Fields};

/// The longest a line's text may be, in characters.
pub const MAX_TEXT: usize = 2_000;

/// A line's address within its journal, and its file's name:
/// `yyyy-mm-dd-hhmm-xxxx`, the minute it was written and four hex digits of
/// the hash of its time, writer, text and salt — so a journal's entries list
/// in the order they were written. Stable across edits; never a position.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct LineId(String);

impl LineId {
    /// The id of a line written at `at` by `by`, saying `text`; `salt` is
    /// raised until the id is free among the journal's.
    pub fn mint(at: Stamp, by: Writer, text: &str, salt: u32) -> LineId {
        let mut h = Sha256::new();
        h.update(at.secs().to_le_bytes());
        h.update([0]);
        h.update(by.name().as_bytes());
        h.update([0]);
        h.update(text.as_bytes());
        h.update([0]);
        h.update(salt.to_le_bytes());
        let d = h.finalize();
        LineId(format!("{}-{:02x}{:02x}", at.dated_minute(), d[0], d[1]))
    }

    pub fn parse(s: &str) -> Result<LineId, String> {
        let b = s.as_bytes();
        let digits = |r: Range<usize>| b[r].iter().all(u8::is_ascii_digit);
        let ok = b.len() == 20
            && [4, 7, 10, 15].iter().all(|&i| b[i] == b'-')
            && digits(0..4)
            && digits(5..7)
            && digits(8..10)
            && digits(11..15)
            && b[16..]
                .iter()
                .all(|c| c.is_ascii_digit() || (b'a'..=b'f').contains(c));
        if ok {
            Ok(LineId(s.to_string()))
        } else {
            Err(format!(
                "`{s}` is not a line id, written 2026-10-30-0912-7e3a"
            ))
        }
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl fmt::Display for LineId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

/// What a line is, set by who writes it.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Kind {
    /// A model was told or chose to keep it: a claim, shown as one.
    Note,
    /// The server recorded something that happened.
    Episode,
    /// A folded journal or period.
    Fold,
}

impl Kind {
    pub fn name(self) -> &'static str {
        match self {
            Self::Note => "note",
            Self::Episode => "episode",
            Self::Fold => "fold",
        }
    }

    pub fn parse(s: &str) -> Result<Kind, String> {
        match s {
            "note" => Ok(Self::Note),
            "episode" => Ok(Self::Episode),
            "fold" => Ok(Self::Fold),
            _ => Err(format!("`{s}` is not a kind: note, episode or fold")),
        }
    }
}

/// Who wrote a line.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Writer {
    /// A model, through the channel a person speaks to it in.
    Channel(Channel),
    /// The connector's server, recording what a call did.
    Server,
    /// zend: summaries, folds, a worker's export.
    Zend,
    /// npcd, for a bot.
    Npcd,
    /// A Zen worker, writing on its own account.
    Worker,
}

impl Writer {
    pub fn name(self) -> &'static str {
        match self {
            Self::Channel(c) => c.name(),
            Self::Server => "server",
            Self::Zend => "zend",
            Self::Npcd => "npcd",
            Self::Worker => "worker",
        }
    }

    pub fn parse(s: &str) -> Result<Writer, String> {
        if let Some(c) = Channel::parse(s) {
            return Ok(Self::Channel(c));
        }
        match s {
            "server" => Ok(Self::Server),
            "zend" => Ok(Self::Zend),
            "npcd" => Ok(Self::Npcd),
            "worker" => Ok(Self::Worker),
            _ => Err(format!(
                "`{s}` is not a writer: a channel, server, zend, npcd or worker"
            )),
        }
    }
}

/// What a fold line folds.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum Folds {
    /// A period of this journal's own lines, moved to its archive.
    Period(Period),
    /// A child journal, left out of the live index.
    Journal(Slug),
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Line {
    pub id: LineId,
    pub at: Stamp,
    pub kind: Kind,
    pub by: Writer,
    /// The person it was written for, when one is.
    pub person: Option<Slug>,
    /// The channel they spoke through, when it is not the writer itself.
    pub channel: Option<Channel>,
    /// For a fold: what it folds.
    pub folds: Option<Folds>,
    /// A worker's note the person has not yet accepted.
    pub proposed: bool,
    pub text: String,
}

impl Line {
    /// The level it stands at: a step, or its folded period's.
    pub fn level(&self) -> Level {
        match &self.folds {
            Some(Folds::Period(p)) => p.level(),
            _ => Level::Step,
        }
    }

    /// The period it is counted in when its window folds: its folded period,
    /// or the day it was written.
    pub fn unit(&self) -> Period {
        match &self.folds {
            Some(Folds::Period(p)) => *p,
            _ => Period::day_of(self.at),
        }
    }

    /// The line's rules, apart from where it is.
    pub fn check(&self) -> Result<(), String> {
        let n = self.text.chars().count();
        if self.text.trim().is_empty() {
            return Err("its text is empty".into());
        }
        if n > MAX_TEXT {
            return Err(format!("its text is {n} characters, over {MAX_TEXT}"));
        }
        match (self.kind, &self.folds) {
            (Kind::Fold, None) => Err("a fold names its `period` or `journal`".into()),
            (Kind::Note | Kind::Episode, Some(_)) => {
                Err("only a fold names a `period` or `journal`".into())
            }
            _ if self.proposed && self.kind != Kind::Note => {
                Err("only a note can be proposed".into())
            }
            _ => Ok(()),
        }
    }

    /// The line's own file, `<id>.yaml`: one field to a text line, its text
    /// last. The id is the file's name and is not written in it.
    pub fn render(&self) -> String {
        let mut out = format!(
            "at: {}\nkind: {}\nby: {}\n",
            self.at,
            self.kind.name(),
            self.by.name()
        );
        if let Some(p) = &self.person {
            out.push_str(&format!("person: {}\n", scalar(p.as_str())));
        }
        if let Some(c) = self.channel {
            out.push_str(&format!("channel: {c}\n"));
        }
        match &self.folds {
            // A year is digits alone, which YAML reads as a number: quoted, so
            // every period is read back as the text it is.
            Some(Folds::Period(p)) => out.push_str(&format!("period: {}\n", quote(&p.to_string()))),
            Some(Folds::Journal(j)) => out.push_str(&format!("journal: {}\n", scalar(j.as_str()))),
            None => {}
        }
        if self.proposed {
            out.push_str("proposed: true\n");
        }
        out.push_str(&text_field("", "text", &self.text));
        out
    }

    /// The line `id` read back from its file's fields.
    pub(crate) fn read(f: &mut Fields<'_>, id: LineId) -> Result<Line, Problem> {
        let at = f.parsed("at", Stamp::parse)?;
        let kind = f.parsed("kind", Kind::parse)?;
        let by = f.parsed("by", Writer::parse)?;
        let person = f.opt_parsed("person", Slug::parse)?;
        let channel = f.opt_parsed("channel", |s| {
            Channel::parse(s).ok_or_else(|| format!("`{s}` is not a channel"))
        })?;
        let period = match f.opt_text_or_whole("period")? {
            None => None,
            Some(s) => Some(
                Period::parse(&s).map_err(|why| f.problem("period", format!("`period`: {why}")))?,
            ),
        };
        let journal = f.opt_parsed("journal", Slug::parse)?;
        let folds = match (period, journal) {
            (Some(_), Some(_)) => {
                return Err(f.problem(
                    "journal",
                    "a fold names a `period` or a `journal`, not both",
                ))
            }
            (Some(p), None) => Some(Folds::Period(p)),
            (None, Some(j)) => Some(Folds::Journal(j)),
            (None, None) => None,
        };
        let proposed = f.opt_bool("proposed")?.unwrap_or(false);
        let text = f.str("text")?;
        let line = Line {
            id,
            at,
            kind,
            by,
            person,
            channel,
            folds,
            proposed,
            text,
        };
        line.check()
            .map_err(|why| f.problem("text", format!("the line is wrong: {why}")))?;
        Ok(line)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testing::note;
    use crate::testing::slug;
    use crate::yaml::parse_doc;

    /// The line `s` holds, read as its file `2026-10-30-0912-7e3a.yaml`.
    fn read_as(id: &LineId, s: &str) -> Result<Line, Problem> {
        let file = format!("{id}.yaml");
        let v = parse_doc(&file, s)?;
        let mut f = Fields::of(&v, &file, None, Some(s))?;
        let line = Line::read(&mut f, id.clone())?;
        f.done()?;
        Ok(line)
    }

    fn read(s: &str) -> Result<Line, Problem> {
        read_as(&LineId::parse("2026-10-30-0912-7e3a").unwrap(), s)
    }

    #[test]
    fn a_line_is_written_exactly_and_read_back() {
        let line = note(
            "2026-10-30T09:12:40Z",
            "Workers push branches; merging is the user's decision.",
        );
        assert_eq!(line.id.as_str().len(), 20);
        assert!(line.id.as_str().starts_with("2026-10-30-0912-"));
        let text = line.render();
        assert_eq!(
            text,
            "at: 2026-10-30T09:12:40Z\nkind: note\nby: claude-code\nperson: john\n\
             text: Workers push branches; merging is the user's decision.\n"
        );
        assert_eq!(read_as(&line.id, &text).unwrap(), line);
    }

    /// **Long text is folded within the file's width**, and read back as
    /// the one paragraph it is.
    #[test]
    fn a_long_line_is_folded_and_read_back_whole() {
        let line = note(
            "2026-10-30T09:12:40Z",
            "The candle fork powers two products: Zen Code, a persistent AI coding assistant \
             with institutional memory across sessions, and Battle Cities, an NPC narrative \
             game whose agents have unbounded memory.",
        );
        assert_eq!(
            line.render(),
            "at: 2026-10-30T09:12:40Z\nkind: note\nby: claude-code\nperson: john\ntext: >-\n\
             \x20 The candle fork powers two products: Zen Code, a persistent AI coding\n\
             \x20 assistant with institutional memory across sessions, and Battle Cities, an NPC\n\
             \x20 narrative game whose agents have unbounded memory.\n"
        );
        assert_eq!(read_as(&line.id, &line.render()).unwrap(), line);
    }

    #[test]
    fn every_field_round_trips() {
        let at = Stamp::parse("2026-10-31T00:00:00Z").unwrap();
        for folds in [
            None,
            Some(Folds::Period(Period::parse("2026").unwrap())),
            Some(Folds::Period(Period::parse("2026-W44").unwrap())),
            Some(Folds::Journal(slug("permits.2"))),
        ] {
            let kind = if folds.is_some() {
                Kind::Fold
            } else {
                Kind::Note
            };
            let line = Line {
                id: LineId::mint(at, Writer::Zend, "x", 7),
                at,
                kind,
                by: Writer::Zend,
                person: None,
                channel: Some(Channel::Grok),
                folds,
                proposed: kind == Kind::Note,
                text: "a: b, {c} #d \"e\"\nf".into(),
            };
            assert_eq!(read_as(&line.id, &line.render()).unwrap(), line);
        }
    }

    #[test]
    fn names_yaml_would_misread_round_trip() {
        let mut line = note("2026-10-30T09:12:40Z", "x");
        line.person = Some(slug("null"));
        line.kind = Kind::Fold;
        line.folds = Some(Folds::Journal(slug("true")));
        assert_eq!(read_as(&line.id, &line.render()).unwrap(), line);
        line.folds = Some(Folds::Journal(slug("1e3")));
        assert_eq!(read_as(&line.id, &line.render()).unwrap(), line);
    }

    #[test]
    fn ids_are_stable_and_salted() {
        let at = Stamp::parse("2026-10-30T09:12:40Z").unwrap();
        let a = LineId::mint(at, Writer::Server, "x", 0);
        assert_eq!(a, LineId::mint(at, Writer::Server, "x", 0));
        assert_ne!(a, LineId::mint(at, Writer::Server, "x", 1));
        assert_ne!(a, LineId::mint(at, Writer::Zend, "x", 0));
        assert_ne!(a, LineId::mint(at, Writer::Server, "y", 0));
        assert!(LineId::parse(a.as_str()).is_ok());
        for bad in [
            "",
            "2026-10-30-0912-7E3A",
            "2026-10-30-0912-7e3",
            "2026-10-30_0912-7e3a",
            "2026-1a-30-0912-7e3a",
            "2026-10-30-0912-7e3az",
            "20261030-0912-7e3a",
        ] {
            assert!(LineId::parse(bad).is_err(), "{bad:?}");
        }
    }

    /// **Each rule broken is an error at the line of its field.**
    #[test]
    fn the_rules_of_a_line_are_errors_naming_its_line() {
        let base = "at: 2026-10-30T09:12:40Z\n";
        for (body, line, says) in [
            (
                "kind: note\nby: grok\ntext: \"\"\n",
                Some(4),
                "its text is empty",
            ),
            (
                "kind: note\nby: grok\ntext: \"   \"\n",
                Some(4),
                "its text is empty",
            ),
            ("kind: fold\nby: zend\ntext: x\n", Some(4), "a fold names"),
            (
                "kind: note\nby: grok\nperiod: \"2026\"\ntext: x\n",
                Some(5),
                "only a fold",
            ),
            (
                "kind: episode\nby: server\nproposed: true\ntext: x\n",
                Some(5),
                "only a note can be proposed",
            ),
            (
                "kind: fold\nby: zend\nperiod: \"2026\"\njournal: a\ntext: x\n",
                Some(5),
                "not both",
            ),
            (
                "kind: note\nby: grok\ntext: x\ncolour: red\n",
                Some(5),
                "`colour` is not a field here",
            ),
            ("kind: remark\nby: grok\ntext: x\n", Some(2), "not a kind"),
            ("kind: note\nby: alice\ntext: x\n", Some(3), "not a writer"),
            (
                "kind: note\nby: grok\nchannel: tesla\ntext: x\n",
                Some(4),
                "not a channel",
            ),
            (
                "kind: note\nby: grok\nperson: Yen\ntext: x\n",
                Some(4),
                "not a name",
            ),
            ("kind: note\nby: grok\n", None, "`text` is missing"),
            (
                "kind: note\nby: grok\nproposed: yes\ntext: x\n",
                Some(4),
                "true or false",
            ),
        ] {
            let err = read(&format!("{base}{body}")).unwrap_err();
            assert_eq!(err.line, line, "{body}: {err}");
            assert!(err.what.contains(says), "{body}: {err}");
        }
        let long = "y".repeat(MAX_TEXT + 1);
        let entry = |text: &str| format!("{base}kind: note\nby: grok\ntext: {text}\n");
        let err = read(&entry(&long)).unwrap_err();
        assert!(err.what.contains("over 2000"), "{err}");
        assert!(read(&entry(&"y".repeat(MAX_TEXT))).is_ok());
    }

    #[test]
    fn a_year_period_is_quoted_so_it_reads_back_as_text() {
        let at = Stamp::parse("2027-01-01T00:00:00Z").unwrap();
        let line = Line {
            id: LineId::mint(at, Writer::Zend, "2026", 0),
            at,
            kind: Kind::Fold,
            by: Writer::Zend,
            person: None,
            channel: None,
            folds: Some(Folds::Period(Period::parse("2026").unwrap())),
            proposed: false,
            text: "The year.".into(),
        };
        assert!(line.render().contains("period: \"2026\""));
        // Written bare by hand, it is still read as the year.
        let bare = line.render().replace("\"2026\"", "2026");
        assert_eq!(read_as(&line.id, &bare).unwrap(), line);
        assert_eq!(line.level(), Level::Year);
        assert_eq!(line.unit(), Period::parse("2026").unwrap());
    }
}
