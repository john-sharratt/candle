//! One entry of a journal (§3.2): a note, an episode, or a fold, written as a
//! one-line flow mapping so that union merges of concurrent appends never
//! conflict.

use std::fmt;

use sha2::{Digest, Sha256};

use crate::error::Problem;
use crate::period::{Level, Period};
use crate::scope::Client;
use crate::slug::Slug;
use crate::time::Stamp;
use crate::yaml::{quote, scalar, Fields};

/// The longest a line's text may be, in characters.
pub const MAX_TEXT: usize = 2_000;

/// A line's address within its journal: `yyyymmdd-hhmm-xxxx`, the minute it
/// was written and four hex digits of the hash of its time, writer, text and
/// salt. Stable across edits; never a position.
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
        LineId(format!("{}-{:02x}{:02x}", at.compact_minute(), d[0], d[1]))
    }

    pub fn parse(s: &str) -> Result<LineId, String> {
        let b = s.as_bytes();
        let ok = b.len() == 18
            && b[8] == b'-'
            && b[13] == b'-'
            && b[..8].iter().all(u8::is_ascii_digit)
            && b[9..13].iter().all(u8::is_ascii_digit)
            && b[14..]
                .iter()
                .all(|c| c.is_ascii_digit() || (b'a'..=b'f').contains(c));
        if ok {
            Ok(LineId(s.to_string()))
        } else {
            Err(format!(
                "`{s}` is not a line id, written 20261030-0912-7e3a"
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
    /// A model, through the client a person speaks to it in.
    Client(Client),
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
            Self::Client(c) => c.name(),
            Self::Server => "server",
            Self::Zend => "zend",
            Self::Npcd => "npcd",
            Self::Worker => "worker",
        }
    }

    pub fn parse(s: &str) -> Result<Writer, String> {
        if let Some(c) = Client::parse(s) {
            return Ok(Self::Client(c));
        }
        match s {
            "server" => Ok(Self::Server),
            "zend" => Ok(Self::Zend),
            "npcd" => Ok(Self::Npcd),
            "worker" => Ok(Self::Worker),
            _ => Err(format!(
                "`{s}` is not a writer: a client, server, zend, npcd or worker"
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
    /// The client they spoke through, when it is not the writer itself.
    pub client: Option<Client>,
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

    /// The line as `lines.yaml` writes it, without the `- `.
    pub fn render(&self) -> String {
        let mut out = format!(
            "{{id: {}, at: {}, kind: {}, by: {}",
            scalar(self.id.as_str()),
            self.at,
            self.kind.name(),
            self.by.name()
        );
        if let Some(p) = &self.person {
            out.push_str(&format!(", person: {}", scalar(p.as_str())));
        }
        if let Some(c) = self.client {
            out.push_str(&format!(", client: {c}"));
        }
        match &self.folds {
            // A year is digits alone, which YAML reads as a number: quoted, so
            // every period is read back as the text it is.
            Some(Folds::Period(p)) => out.push_str(&format!(", period: {}", quote(&p.to_string()))),
            Some(Folds::Journal(j)) => out.push_str(&format!(", journal: {}", scalar(j.as_str()))),
            None => {}
        }
        if self.proposed {
            out.push_str(", proposed: true");
        }
        out.push_str(&format!(", text: {}}}", quote(&self.text)));
        out
    }

    /// One entry read back; `file` and `line` say where, for its errors.
    pub(crate) fn read(f: &mut Fields<'_>) -> Result<Line, Problem> {
        let id = f.parsed("id", LineId::parse)?;
        let at = f.parsed("at", Stamp::parse)?;
        let kind = f.parsed("kind", Kind::parse)?;
        let by = f.parsed("by", Writer::parse)?;
        let person = f.opt_parsed("person", Slug::parse)?;
        let client = f.opt_parsed("client", |s| {
            Client::parse(s).ok_or_else(|| format!("`{s}` is not a client"))
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
            client,
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
    use crate::yaml::parse_flow_line;

    fn read(s: &str) -> Result<Line, Problem> {
        let v = parse_flow_line("lines.yaml", 3, s)?;
        let mut f = Fields::of(&v, "lines.yaml", Some(3), None)?;
        let line = Line::read(&mut f)?;
        f.done()?;
        Ok(line)
    }

    #[test]
    fn a_line_is_written_exactly_and_read_back() {
        let line = note(
            "2026-10-30T09:12:40Z",
            "Workers push branches; merging is the user's decision.",
        );
        assert_eq!(line.id.as_str().len(), 18);
        assert!(line.id.as_str().starts_with("20261030-0912-"));
        let text = format!("- {}", line.render());
        assert_eq!(
            text,
            format!(
                "- {{id: {}, at: 2026-10-30T09:12:40Z, kind: note, by: claude-code, person: john, \
                 text: \"Workers push branches; merging is the user's decision.\"}}",
                line.id
            )
        );
        assert_eq!(read(&text).unwrap(), line);
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
                client: Some(Client::Grok),
                folds,
                proposed: kind == Kind::Note,
                text: "a: b, {c} #d \"e\"\nf".into(),
            };
            assert_eq!(read(&format!("- {}", line.render())).unwrap(), line);
        }
    }

    #[test]
    fn names_yaml_would_misread_round_trip() {
        let mut line = note("2026-10-30T09:12:40Z", "x");
        line.person = Some(slug("null"));
        line.kind = Kind::Fold;
        line.folds = Some(Folds::Journal(slug("true")));
        assert_eq!(read(&format!("- {}", line.render())).unwrap(), line);
        line.folds = Some(Folds::Journal(slug("1e3")));
        assert_eq!(read(&format!("- {}", line.render())).unwrap(), line);
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
            "20261030-0912-7E3A",
            "20261030-0912-7e3",
            "20261030_0912-7e3a",
            "2026103a-0912-7e3a",
            "20261030-0912-7e3az",
        ] {
            assert!(LineId::parse(bad).is_err(), "{bad:?}");
        }
    }

    #[test]
    fn the_rules_of_a_line_are_errors_naming_its_line() {
        let base = "id: 20261030-0912-7e3a, at: 2026-10-30T09:12:40Z";
        for (body, says) in [
            ("kind: note, by: grok, text: \"\"", "its text is empty"),
            ("kind: note, by: grok, text: \"   \"", "its text is empty"),
            ("kind: fold, by: zend, text: \"x\"", "a fold names"),
            (
                "kind: note, by: grok, period: \"2026\", text: \"x\"",
                "only a fold",
            ),
            (
                "kind: episode, by: server, proposed: true, text: \"x\"",
                "only a note can be proposed",
            ),
            (
                "kind: fold, by: zend, period: \"2026\", journal: a, text: \"x\"",
                "not both",
            ),
            (
                "kind: note, by: grok, text: \"x\", colour: red",
                "`colour` is not a field here",
            ),
            ("kind: remark, by: grok, text: \"x\"", "not a kind"),
            ("kind: note, by: alice, text: \"x\"", "not a writer"),
            (
                "kind: note, by: grok, client: tesla, text: \"x\"",
                "not a client",
            ),
            (
                "kind: note, by: grok, person: Yen, text: \"x\"",
                "not a name",
            ),
            ("kind: note, by: grok", "`text` is missing"),
            (
                "kind: note, by: grok, proposed: yes, text: \"x\"",
                "true or false",
            ),
        ] {
            let err = read(&format!("- {{{base}, {body}}}")).unwrap_err();
            assert_eq!(err.line, Some(3), "{body}");
            assert!(err.what.contains(says), "{body}: {err}");
        }
        let long = "y".repeat(MAX_TEXT + 1);
        let err = read(&format!(
            "- {{{base}, kind: note, by: grok, text: \"{long}\"}}"
        ))
        .unwrap_err();
        assert!(err.what.contains("over 2000"), "{err}");
        assert!(read(&format!(
            "- {{{base}, kind: note, by: grok, text: \"{}\"}}",
            "y".repeat(MAX_TEXT)
        ))
        .is_ok());
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
            client: None,
            folds: Some(Folds::Period(Period::parse("2026").unwrap())),
            proposed: false,
            text: "The year.".into(),
        };
        assert!(line.render().contains("period: \"2026\""));
        // Written bare by hand, it is still read as the year.
        let bare = line.render().replace("\"2026\"", "2026");
        assert_eq!(read(&format!("- {bare}")).unwrap(), line);
        assert_eq!(line.level(), Level::Year);
        assert_eq!(line.unit(), Period::parse("2026").unwrap());
    }
}
