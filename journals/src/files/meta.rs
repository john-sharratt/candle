//! `journal.yaml`: a journal's name, cue, dates, state and summary (§3.2). It
//! merges as git always does, so two rebuilds of one summary conflict.

use crate::error::Problem;
use crate::files::summary::Summary;
use crate::time::Stamp;
use crate::yaml::{parse_doc, quote, scalar, Fields};

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Meta {
    /// Its folder's name; for a fixed journal, its fixed name.
    pub name: String,
    /// "Open this when…": what its parent's index shows. A fixed journal's
    /// is fixed in code and not written.
    pub cue: Option<String>,
    pub created: Stamp,
    /// When its conversation, thread or task ended.
    pub closed: Option<Stamp>,
    /// Left out of its parent's live index (§2.3).
    pub folded: bool,
    /// The group journal it was shared into, written `@group/path`.
    pub shared: Option<String>,
    pub summary: Option<Summary>,
}

impl Meta {
    pub fn new(name: impl Into<String>, cue: Option<String>, created: Stamp) -> Meta {
        Meta {
            name: name.into(),
            cue,
            created,
            closed: None,
            folded: false,
            shared: None,
            summary: None,
        }
    }

    pub fn render(&self) -> String {
        let mut out = format!("name: {}\n", scalar(&self.name));
        if let Some(cue) = &self.cue {
            out.push_str(&format!("cue: {}\n", quote(cue)));
        }
        out.push_str(&format!("created: {}\n", self.created));
        if let Some(c) = self.closed {
            out.push_str(&format!("closed: {c}\n"));
        }
        out.push_str(&format!("folded: {}\n", self.folded));
        if let Some(s) = &self.shared {
            out.push_str(&format!("shared: {}\n", quote(s)));
        }
        if let Some(s) = &self.summary {
            s.render(&mut out);
        }
        out
    }

    pub fn read(file: &str, text: &str) -> Result<Meta, Problem> {
        let v = parse_doc(file, text)?;
        let mut f = Fields::of(&v, file, None, Some(text))?;
        let name = f.str("name")?;
        let cue = f.opt_str("cue")?;
        if cue.as_deref().is_some_and(|c| c.trim().is_empty()) {
            return Err(f.problem("cue", "`cue` is empty"));
        }
        let created = f.parsed("created", Stamp::parse)?;
        let closed = f.opt_parsed("closed", Stamp::parse)?;
        let folded = f
            .opt_bool("folded")?
            .ok_or_else(|| f.problem("folded", "`folded` is missing"))?;
        let shared = f.opt_str("shared")?;
        if shared.as_deref().is_some_and(|s| !s.starts_with('@')) {
            return Err(f.problem("shared", "`shared` names a group journal, `@group/path`"));
        }
        let summary = match f.opt_map("summary")? {
            None => None,
            Some(sv) => {
                let mut s = Fields::of(sv, file, None, Some(text))?;
                let summary = Summary::read(&mut s, file, text)?;
                s.done()?;
                Some(summary)
            }
        };
        f.done()?;
        Ok(Meta {
            name,
            cue,
            created,
            closed,
            folded,
            shared,
            summary,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::files::summary::{Cite, Fingerprint, Sentence};
    use crate::line::LineId;
    use crate::testing::slug;

    fn at(s: &str) -> Stamp {
        Stamp::parse(s).unwrap()
    }

    fn full() -> Meta {
        Meta {
            name: "candle".into(),
            cue: Some("open for the inference engine, its releases and cluster".into()),
            created: at("2026-10-30T09:10:02Z"),
            closed: Some(at("2026-10-31T01:00:00Z")),
            folded: true,
            shared: Some("@sharratt/topics/candle".into()),
            summary: Some(Summary {
                over: Fingerprint::of_bytes(b"inputs"),
                built: at("2026-10-30T11:02:40Z"),
                lines: 12,
                children: 2,
                stale: true,
                digest: Some("Candle: releases go through keep.".into()),
                sentences: vec![
                    Sentence {
                        text: "Releases go through keep trials on BlackWorld, then roll out."
                            .into(),
                        cites: vec![Cite::Line(LineId::parse("20261029-0410-a3b4").unwrap())],
                    },
                    Sentence {
                        text: "The connector now remembers its user.".into(),
                        cites: vec![Cite::Child(slug("user-context"))],
                    },
                ],
            }),
        }
    }

    #[test]
    fn journal_yaml_is_written_exactly() {
        let m = full();
        assert_eq!(
            m.render(),
            format!(
                "name: candle\n\
                 cue: \"open for the inference engine, its releases and cluster\"\n\
                 created: 2026-10-30T09:10:02Z\n\
                 closed: 2026-10-31T01:00:00Z\n\
                 folded: true\n\
                 shared: \"@sharratt/topics/candle\"\n\
                 summary:\n  over: {}\n  built: 2026-10-30T11:02:40Z\n  lines: 12\n  children: 2\n  stale: true\n\
                 \x20 digest: \"Candle: releases go through keep.\"\n  sentences:\n\
                 \x20   - {{text: \"Releases go through keep trials on BlackWorld, then roll out.\", cites: [\"#20261029-0410-a3b4\"]}}\n\
                 \x20   - {{text: \"The connector now remembers its user.\", cites: [\"user-context\"]}}\n",
                Fingerprint::of_bytes(b"inputs")
            )
        );
        assert_eq!(Meta::read("journal.yaml", &m.render()).unwrap(), m);
    }

    #[test]
    fn a_bare_journal_is_four_lines() {
        let m = Meta::new(
            "permits",
            Some("open for council permits".into()),
            at("2026-10-30T09:10:02Z"),
        );
        assert_eq!(
            m.render(),
            "name: permits\ncue: \"open for council permits\"\ncreated: 2026-10-30T09:10:02Z\nfolded: false\n"
        );
        assert_eq!(Meta::read("j", &m.render()).unwrap(), m);
        let fixed = Meta::new("projects", None, at("2026-10-30T09:10:02Z"));
        assert_eq!(Meta::read("j", &fixed.render()).unwrap(), fixed);
    }

    #[test]
    fn names_and_fingerprints_yaml_would_misread_round_trip() {
        for name in ["true", "null", "1e3", "0x1f", "0123"] {
            let mut m = full();
            m.name = name.into();
            m.summary.as_mut().unwrap().over = Fingerprint::parse("1234567890123e45").unwrap();
            assert_eq!(Meta::read("j", &m.render()).unwrap(), m, "{name}");
        }
    }

    #[test]
    fn an_empty_sentence_list_round_trips() {
        let mut m = full();
        m.summary.as_mut().unwrap().sentences.clear();
        m.summary.as_mut().unwrap().digest = None;
        m.summary.as_mut().unwrap().stale = false;
        assert!(m.render().contains("  sentences: []\n"));
        assert_eq!(Meta::read("j", &m.render()).unwrap(), m);
    }

    #[test]
    fn each_fault_names_its_line() {
        let good = full().render();
        for (from, to, line, says) in [
            ("folded: true", "folded: yes", 5, "true or false"),
            ("folded: true\n", "", 0, "`folded` is missing"),
            (
                "created: 2026-10-30T09:10:02Z",
                "created: yesterday",
                3,
                "not a time",
            ),
            (
                "name: candle",
                "colour: red\nname: candle",
                1,
                "`colour` is not a field",
            ),
            (
                "shared: \"@sharratt/topics/candle\"",
                "shared: \"topics/candle\"",
                6,
                "`@group/path`",
            ),
            ("over: ", "over: xyz", 8, "not a fingerprint"),
            ("lines: 12", "lines: many", 10, "not a count"),
            (
                "[\"#20261029-0410-a3b4\"]",
                "[\"#bad\"]",
                15,
                "not a line id",
            ),
            (
                "[\"user-context\"]",
                "[\"User Context\"]",
                16,
                "cites neither",
            ),
            (
                "The connector now remembers its user.",
                "  ",
                16,
                "a sentence is empty",
            ),
            (
                "cue: \"open for the inference engine, its releases and cluster\"",
                "cue: \"\"",
                2,
                "`cue` is empty",
            ),
        ] {
            assert!(good.contains(from), "{from}");
            let bad = good.replacen(from, to, 1);
            let err = Meta::read("projects/candle/journal.yaml", &bad).unwrap_err();
            assert!(err.what.contains(says), "{from}: {err}");
            if line > 0 {
                assert_eq!(err.line, Some(line), "{from}: {err}");
            }
        }
    }
}
