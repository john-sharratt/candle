//! `journal.yaml`: a journal's name, cue, dates, state, repository,
//! references, focus and summary (§3.2). It merges as git always does, so two
//! rebuilds of one summary conflict.

use crate::error::Problem;
use crate::files::focus::{Focus, MAX_ABOUT};
use crate::files::reference::{Ref, MAX_REFS};
use crate::files::summary::Summary;
use crate::path::Below;
use crate::slug::Slug;
use crate::time::Stamp;
use crate::yaml::{item_line, parse_doc, scalar, scalar_list, text_field, Fields};

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
    /// The repository a topic is, for a Zen worker's frame (§2.2).
    pub repo: Option<Slug>,
    /// Its references to other journals of the group (§2.5).
    pub refs: Vec<Ref>,
    /// A conversation's focus: who is speaking, and what it is about (§2.4).
    pub focus: Option<Focus>,
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
            repo: None,
            refs: Vec::new(),
            focus: None,
            summary: None,
        }
    }

    pub fn render(&self) -> String {
        let mut out = format!("name: {}\n", scalar(&self.name));
        if let Some(cue) = &self.cue {
            out.push_str(&text_field("", "cue", cue));
        }
        out.push_str(&format!("created: {}\n", self.created));
        if let Some(c) = self.closed {
            out.push_str(&format!("closed: {c}\n"));
        }
        out.push_str(&format!("folded: {}\n", self.folded));
        if let Some(r) = &self.repo {
            out.push_str(&format!("repo: {}\n", scalar(r.as_str())));
        }
        if !self.refs.is_empty() {
            out.push_str("refs:\n");
            for r in &self.refs {
                out.push_str(&format!("  - name: {}\n", scalar(r.name.as_str())));
                out.push_str(&format!("    to: {}\n", scalar(&r.to.to_string())));
                out.push_str(&text_field("    ", "cue", &r.cue));
            }
        }
        if let Some(f) = self.focus.as_ref().filter(|f| !f.is_empty()) {
            out.push_str("focus:\n");
            if let Some(s) = &f.speaker {
                out.push_str(&format!("  speaker: {}\n", scalar(s.as_str())));
            }
            if !f.about.is_empty() {
                let about: Vec<String> = f.about.iter().map(Below::to_string).collect();
                out.push_str(&format!("  about: {}\n", scalar_list(&about)));
            }
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
        let repo = f.opt_parsed("repo", Slug::parse)?;
        let mut refs: Vec<Ref> = Vec::new();
        for (i, rv) in f.opt_seq("refs")?.into_iter().flatten().enumerate() {
            let mut rf = Fields::of(rv, file, item_line(text, "refs", i), None)?;
            let name = rf.parsed("name", Slug::parse)?;
            let to = rf.parsed("to", Below::parse)?;
            let cue = rf.str("cue")?;
            if cue.trim().is_empty() {
                return Err(rf.problem("cue", "a reference's `cue` is empty"));
            }
            if refs.iter().any(|r| r.name == name) {
                return Err(rf.problem("name", format!("the reference {name} is listed twice")));
            }
            rf.done()?;
            refs.push(Ref { name, to, cue });
        }
        if refs.len() > MAX_REFS {
            return Err(f.problem(
                "refs",
                format!("{} references: at most {MAX_REFS}", refs.len()),
            ));
        }
        let focus = match f.opt_map("focus")? {
            None => None,
            Some(fv) => {
                let mut ff = Fields::of(fv, file, None, Some(text))?;
                let speaker = ff.opt_parsed("speaker", Slug::parse)?;
                let about = ff
                    .opt_strs("about")?
                    .unwrap_or_default()
                    .iter()
                    .map(|s| Below::parse(s))
                    .collect::<Result<Vec<_>, _>>()
                    .map_err(|why| ff.problem("about", format!("`about`: {why}")))?;
                if about.len() > MAX_ABOUT {
                    return Err(ff.problem(
                        "about",
                        format!(
                            "a focus is about at most {MAX_ABOUT} journals, not {}",
                            about.len()
                        ),
                    ));
                }
                ff.done()?;
                Some(Focus { speaker, about })
            }
        };
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
            repo,
            refs,
            focus,
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
            repo: Some(slug("candle")),
            refs: vec![
                Ref {
                    name: slug("wasmer"),
                    to: Below::parse("topics/wasmer").unwrap(),
                    cue: "open for the WebAssembly work the engine's sandboxing came from".into(),
                },
                Ref {
                    name: slug("john"),
                    to: Below::parse("people/john").unwrap(),
                    cue: "open for who leads it".into(),
                },
            ],
            focus: Some(Focus {
                speaker: Some(slug("yen")),
                about: vec![Below::parse("topics/house-extension/permits").unwrap()],
            }),
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
                        cites: vec![Cite::Line(LineId::parse("2026-10-29-0410-a3b4").unwrap())],
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
                 cue: open for the inference engine, its releases and cluster\n\
                 created: 2026-10-30T09:10:02Z\n\
                 closed: 2026-10-31T01:00:00Z\n\
                 folded: true\n\
                 repo: candle\n\
                 refs:\n\
                 \x20 - name: wasmer\n\
                 \x20   to: topics/wasmer\n\
                 \x20   cue: open for the WebAssembly work the engine's sandboxing came from\n\
                 \x20 - name: john\n\
                 \x20   to: people/john\n\
                 \x20   cue: open for who leads it\n\
                 focus:\n\
                 \x20 speaker: yen\n\
                 \x20 about: [topics/house-extension/permits]\n\
                 summary:\n  over: {}\n  built: 2026-10-30T11:02:40Z\n  lines: 12\n  children: 2\n  stale: true\n\
                 \x20 digest: \"Candle: releases go through keep.\"\n  sentences:\n\
                 \x20   - text: Releases go through keep trials on BlackWorld, then roll out.\n\
                 \x20     cites: [\"#2026-10-29-0410-a3b4\"]\n\
                 \x20   - text: The connector now remembers its user.\n\
                 \x20     cites: [user-context]\n",
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
            "name: permits\ncue: open for council permits\ncreated: 2026-10-30T09:10:02Z\nfolded: false\n"
        );
        assert_eq!(Meta::read("j", &m.render()).unwrap(), m);
        let fixed = Meta::new("topics", None, at("2026-10-30T09:10:02Z"));
        assert_eq!(Meta::read("j", &fixed.render()).unwrap(), fixed);
    }

    /// **A focus writes only what it has**: a speaker alone, or journals
    /// alone; an empty one writes nothing.
    #[test]
    fn a_focus_writes_only_what_it_has() {
        let mut m = Meta::new(
            "2026-10-31-0712-grok",
            Some("open for a drive".into()),
            at("2026-10-31T07:12:04Z"),
        );
        m.focus = Some(Focus {
            speaker: Some(slug("john")),
            about: vec![],
        });
        assert!(
            m.render()
                .ends_with("folded: false\nfocus:\n  speaker: john\n"),
            "{}",
            m.render()
        );
        assert_eq!(Meta::read("j", &m.render()).unwrap(), m);
        m.focus = Some(Focus {
            speaker: None,
            about: vec![Below::parse("topics/candle").unwrap()],
        });
        assert!(
            m.render().ends_with("focus:\n  about: [topics/candle]\n"),
            "{}",
            m.render()
        );
        assert_eq!(Meta::read("j", &m.render()).unwrap(), m);
        m.focus = Some(Focus::default());
        assert!(m.render().ends_with("folded: false\n"));
    }

    #[test]
    fn names_and_fingerprints_yaml_would_misread_round_trip() {
        for name in ["true", "null", "1e3", "0x1f", "0123"] {
            let mut m = full();
            m.name = name.into();
            m.repo = Some(slug(name));
            m.refs[0].name = slug(name);
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
            ("repo: candle", "repo: Candle", 6, "not a name"),
            (
                "to: topics/wasmer",
                "to: topics/Wasmer",
                8,
                "not a path below a group",
            ),
            (
                "    to: people/john\n",
                "    to: people/john\n    via: x\n",
                0,
                "`via` is not a field",
            ),
            ("name: john", "name: wasmer", 11, "listed twice"),
            ("speaker: yen", "speaker: Yen", 0, "not a name"),
            (
                "about: [topics/house-extension/permits]",
                "about: [a, b, c, d]",
                0,
                "at most 3 journals, not 4",
            ),
            ("over: ", "over: xyz", 0, "not a fingerprint"),
            ("lines: 12", "lines: many", 0, "not a count"),
            ("[user-context]", "[User Context]", 0, "cites neither"),
            (
                "cue: open for the inference engine, its releases and cluster",
                "cue: \"\"",
                2,
                "`cue` is empty",
            ),
        ] {
            assert!(good.contains(from), "{from}");
            let bad = good.replacen(from, to, 1);
            let err = Meta::read("topics/candle/journal.yaml", &bad).unwrap_err();
            assert!(err.what.contains(says), "{from}: {err}");
            if line > 0 {
                assert_eq!(err.line, Some(line), "{from}: {err}");
            }
        }
        // Everything in a group is shared: no journal says it was.
        let err = Meta::read("j", &format!("{good}shared: \"@zen/topics/x\"\n")).unwrap_err();
        assert!(err.what.contains("`shared` is not a field"), "{err}");
        let eleven = (0..11)
            .map(|i| format!("  - name: r{i}\n    to: topics/x\n    cue: c\n"))
            .collect::<String>();
        let err = Meta::read(
            "j",
            &format!(
                "name: x\ncue: c\ncreated: 2026-10-30T09:10:02Z\nfolded: false\nrefs:\n{eleven}"
            ),
        )
        .unwrap_err();
        assert!(err.what.contains("11 references: at most 10"), "{err}");
    }
}
