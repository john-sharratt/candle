//! A journal's summary (§5): cited sentences, capped, rebuilt from its window
//! and its live children's summaries, and the fingerprint of what it was
//! built over — so it is dirty exactly when that changed.

use std::fmt;

use sha2::{Digest, Sha256};

use crate::error::Problem;
use crate::line::LineId;
use crate::slug::Slug;
use crate::time::Stamp;
use crate::yaml::{item_line, quote, quote_list, scalar, Fields};

/// What a summary was built over: a hash of its journal's window and its
/// live children's names, cues and summaries, 16 hex digits.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct Fingerprint(String);

impl Fingerprint {
    pub fn of_bytes(bytes: &[u8]) -> Fingerprint {
        let d = Sha256::digest(bytes);
        Fingerprint(d[..8].iter().map(|b| format!("{b:02x}")).collect())
    }

    pub fn parse(s: &str) -> Result<Fingerprint, String> {
        if s.len() == 16
            && s.bytes()
                .all(|c| c.is_ascii_digit() || (b'a'..=b'f').contains(&c))
        {
            Ok(Fingerprint(s.into()))
        } else {
            Err(format!(
                "`{s}` is not a fingerprint: 16 lowercase hex digits"
            ))
        }
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl fmt::Display for Fingerprint {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

/// What a sentence rests on.
#[derive(Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Cite {
    /// One of the journal's own lines, in its window or archives: `#<id>`.
    Line(LineId),
    /// One of its children, by name.
    Child(Slug),
}

impl Cite {
    pub fn parse(s: &str) -> Result<Cite, String> {
        match s.strip_prefix('#') {
            Some(id) => LineId::parse(id).map(Cite::Line),
            None => Slug::parse(s)
                .map(Cite::Child)
                .map_err(|_| format!("`{s}` cites neither `#<line id>` nor a child's name")),
        }
    }
}

impl fmt::Display for Cite {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Cite::Line(id) => write!(f, "#{id}"),
            Cite::Child(s) => write!(f, "{s}"),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Sentence {
    pub text: String,
    pub cites: Vec<Cite>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Summary {
    pub over: Fingerprint,
    pub built: Stamp,
    /// Window lines, and live children with something to say, it was built
    /// over — so a rebuild that shrinks it by half with nothing gone can be
    /// told from one whose inputs went: a line forgotten, a child emptied.
    pub lines: usize,
    pub children: usize,
    /// The last rebuild was refused; this is the summary before it.
    pub stale: bool,
    /// The short form a voice client is given, for a root (§6.4).
    pub digest: Option<String>,
    pub sentences: Vec<Sentence>,
}

impl Summary {
    /// The summary as one text: its sentences, space-joined.
    pub fn text(&self) -> String {
        self.sentences
            .iter()
            .map(|s| s.text.trim())
            .collect::<Vec<_>>()
            .join(" ")
    }

    /// Its length, in characters, as caps measure it.
    pub fn len(&self) -> usize {
        self.text().chars().count()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Lines of `journal.yaml` under `summary:`.
    pub(crate) fn render(&self, out: &mut String) {
        out.push_str("summary:\n");
        out.push_str(&format!("  over: {}\n", scalar(self.over.as_str())));
        out.push_str(&format!("  built: {}\n", self.built));
        out.push_str(&format!("  lines: {}\n", self.lines));
        out.push_str(&format!("  children: {}\n", self.children));
        if self.stale {
            out.push_str("  stale: true\n");
        }
        if let Some(d) = &self.digest {
            out.push_str(&format!("  digest: {}\n", quote(d)));
        }
        if self.sentences.is_empty() {
            out.push_str("  sentences: []\n");
            return;
        }
        out.push_str("  sentences:\n");
        for s in &self.sentences {
            let cites: Vec<String> = s.cites.iter().map(Cite::to_string).collect();
            out.push_str(&format!(
                "    - {{text: {}, cites: {}}}\n",
                quote(&s.text),
                quote_list(&cites)
            ));
        }
    }

    pub(crate) fn read(f: &mut Fields<'_>, file: &str, text: &str) -> Result<Summary, Problem> {
        let over = f.parsed("over", Fingerprint::parse)?;
        let built = f.parsed("built", Stamp::parse)?;
        let lines = f.count("lines")?;
        let children = f.count("children")?;
        let stale = f.opt_bool("stale")?.unwrap_or(false);
        let digest = f.opt_str("digest")?;
        let mut sentences = Vec::new();
        for (i, v) in f.opt_seq("sentences")?.into_iter().flatten().enumerate() {
            let line = item_line(text, "sentences", i);
            let mut s = Fields::of(v, file, line, None)?;
            let st = s.str("text")?;
            if st.trim().is_empty() {
                return Err(s.problem("text", "a sentence is empty"));
            }
            let cites = s
                .opt_strs("cites")?
                .unwrap_or_default()
                .iter()
                .map(|c| Cite::parse(c))
                .collect::<Result<Vec<_>, _>>()
                .map_err(|why| s.problem("cites", why))?;
            s.done()?;
            sentences.push(Sentence { text: st, cites });
        }
        Ok(Summary {
            over,
            built,
            lines,
            children,
            stale,
            digest,
            sentences,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cites_are_line_ids_or_child_names() {
        assert_eq!(
            Cite::parse("#20261029-0410-a3b4").unwrap().to_string(),
            "#20261029-0410-a3b4"
        );
        assert_eq!(
            Cite::parse("user-context").unwrap().to_string(),
            "user-context"
        );
        for bad in ["#nope", "Projects/x", "a/b", "", "#"] {
            assert!(Cite::parse(bad).is_err(), "{bad:?}");
        }
    }

    #[test]
    fn a_fingerprint_is_sixteen_hex_digits_of_sha256() {
        let f = Fingerprint::of_bytes(b"");
        assert_eq!(f.as_str(), "e3b0c44298fc1c14");
        assert_eq!(Fingerprint::parse("e3b0c44298fc1c14"), Ok(f));
        assert!(Fingerprint::parse("E3B0C44298FC1C14").is_err());
        assert!(Fingerprint::parse("e3b0c44298fc1c1").is_err());
    }

    #[test]
    fn the_text_is_the_sentences_joined() {
        let s = Summary {
            over: Fingerprint::of_bytes(b"x"),
            built: Stamp::from_secs(0),
            lines: 0,
            children: 0,
            stale: false,
            digest: None,
            sentences: vec![
                Sentence {
                    text: " One. ".into(),
                    cites: vec![],
                },
                Sentence {
                    text: "Two.".into(),
                    cites: vec![],
                },
            ],
        };
        assert_eq!(s.text(), "One. Two.");
        assert_eq!(s.len(), 9);
    }
}
