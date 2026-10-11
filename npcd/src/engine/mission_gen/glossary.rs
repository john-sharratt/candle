//! The world's own words, as the world says them.
//!
//! **A term the reader cannot look up is a fault it invents.** The setting
//! says "humanity survives as uploaded consciousness in the towers… Avatars
//! walk the surface", and the table, shown that and an era, failed a story of
//! the Contested Cities for "biological humans in a post-biological era" —
//! eight readings in sixteen. The world's own `avatar` document opens:
//! "Avatars are biological clones vat-grown in tower factories, serving as
//! physical bodies for human consciousness". The story was canon; the reader
//! did not know what the word meant.
//!
//! Every top-level document of `layers/world/` is the world's account of one
//! of its terms — `avatar`, `tower`, `zenlings`, `cities` — and opens by
//! saying what it is. The terms a draft and what it answers to name are given
//! with that opening, most named first.

use std::path::Path;

use super::answer::unlink;
use super::material::cut;

/// Where the world's terms are, under the mind root.
const WORLD: &str = "layers/world";

/// How much of a term's opening is given.
const GIST_WORDS: usize = 50;

/// The most terms given at once.
pub const MOST: usize = 6;

/// One of the world's terms and what its document opens by saying.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Term {
    /// The term as a word: `avatar`, `alien wildlife`.
    pub word: String,
    /// Its document, by mind path.
    pub path: String,
    /// Its document's opening paragraph, links read as their words.
    pub gist: String,
}

/// The world's terms in the mind at `root`, in alphabetical order — none
/// when it has no world layer.
pub fn terms(root: &Path) -> Vec<Term> {
    let Ok(entries) = std::fs::read_dir(root.join(WORLD)) else {
        return Vec::new();
    };
    let mut out: Vec<Term> = entries
        .flatten()
        .filter_map(|e| {
            let p = e.path();
            if !p.is_file() || p.extension()? != "md" {
                return None;
            }
            let stem = p.file_stem()?.to_str()?.to_string();
            let gist = gist(&std::fs::read_to_string(&p).ok()?)?;
            Some(Term {
                word: stem.replace('_', " "),
                path: format!("{WORLD}/{stem}.md"),
                gist,
            })
        })
        .collect();
    out.sort_by(|a, b| a.word.cmp(&b.word));
    out
}

/// A document's first paragraph that is not a heading, cut to
/// [`GIST_WORDS`].
fn gist(text: &str) -> Option<String> {
    text.split("\n\n")
        .map(str::trim)
        .find(|p| !p.is_empty() && !p.starts_with('#'))
        .map(|p| cut(&unlink(p), GIST_WORDS))
}

/// The forms a term is written in: itself, and its singular or plural.
fn forms(word: &str) -> Vec<String> {
    let w = word.to_lowercase();
    let other = match (w.strip_suffix("ies"), w.strip_suffix('s')) {
        (Some(stem), _) => format!("{stem}y"),
        (None, Some(stem)) => stem.to_string(),
        (None, None) => format!("{w}s"),
    };
    vec![w, other]
}

/// How many times `form` stands in `text` as whole words.
fn mentions(text: &str, form: &str) -> usize {
    let boundary = |c: Option<char>| c.is_none_or(|c| !c.is_alphanumeric());
    text.match_indices(form)
        .filter(|(at, _)| {
            boundary(text[..*at].chars().next_back())
                && boundary(text[at + form.len()..].chars().next())
        })
        .count()
}

/// The terms of `terms` named in `draft` or in what it answers to (`around`),
/// at most [`MOST`], most named first. A name in the draft counts twice: it is
/// what the reading judges.
pub fn named<'a>(terms: &'a [Term], draft: &str, around: &[&str]) -> Vec<&'a Term> {
    let draft = unlink(draft).to_lowercase();
    let around: Vec<String> = around.iter().map(|a| unlink(a).to_lowercase()).collect();
    let mut scored: Vec<(usize, &Term)> = terms
        .iter()
        .filter_map(|t| {
            let n: usize = forms(&t.word)
                .iter()
                .map(|f| {
                    2 * mentions(&draft, f) + around.iter().map(|a| mentions(a, f)).sum::<usize>()
                })
                .sum();
            (n > 0).then_some((n, t))
        })
        .collect();
    scored.sort_by(|a, b| b.0.cmp(&a.0).then_with(|| a.1.word.cmp(&b.1.word)));
    scored.into_iter().take(MOST).map(|(_, t)| t).collect()
}

/// The section a reading is shown the `named` terms under — empty for none.
pub fn render(named: &[&Term]) -> String {
    if named.is_empty() {
        return String::new();
    }
    let mut s =
        String::from("\n## What the world's own words mean — as its record defines them\n\n");
    for t in named {
        s.push_str(&format!("- **{}** (`{}`): {}\n", t.word, t.path, t.gist));
    }
    s
}

#[cfg(test)]
mod tests {
    use super::*;

    fn term(word: &str, gist: &str) -> Term {
        Term {
            word: word.into(),
            path: format!("{WORLD}/{}.md", word.replace(' ', "_")),
            gist: gist.into(),
        }
    }

    /// **A term's document is read for its opening**, its heading skipped and
    /// its links read as words.
    #[test]
    fn the_terms_are_read_from_the_world_layer() {
        let dir = tempfile::tempdir().unwrap();
        let world = dir.path().join(WORLD);
        std::fs::create_dir_all(world.join("avatar")).unwrap();
        std::fs::write(
            world.join("avatar.md"),
            "# Avatar - Comprehensive Details\n\n[Avatars](/avatar) are biological clones \
             vat-grown in [tower](/tower) factories.\n\n## More\n\nDetail.",
        )
        .unwrap();
        std::fs::write(world.join("alien_wildlife.md"), "Beasts of the waste.").unwrap();
        std::fs::write(world.join("avatar").join("biological.md"), "Not a term.").unwrap();
        assert_eq!(
            terms(dir.path()),
            [
                term("alien wildlife", "Beasts of the waste."),
                term(
                    "avatar",
                    "Avatars are biological clones vat-grown in tower factories."
                ),
            ]
        );
        assert!(terms(&dir.path().join("nothing")).is_empty());
    }

    /// **The terms a draft names come first**, singular or plural, as whole
    /// words, and a term named nowhere is not given.
    #[test]
    fn the_named_terms_are_given_most_named_first() {
        let terms = [
            term("avatar", "a"),
            term("cities", "c"),
            term("tower", "t"),
            term("zenlings", "z"),
            term("core", "k"),
        ];
        let draft = "The Avatar raised the core of the vault. A Zenling watched the avatar.";
        let era = "The [cities](/cities) opened; the towers grew; the city burned.";
        let words: Vec<&str> = named(&terms, draft, &[era])
            .iter()
            .map(|t| t.word.as_str())
            .collect();
        assert_eq!(words, ["avatar", "cities", "core", "zenlings", "tower"]);
        let words: Vec<&str> = named(&terms, "Kernel and towering cores.", &[])
            .iter()
            .map(|t| t.word.as_str())
            .collect();
        assert_eq!(words, ["core"]);
    }

    #[test]
    fn the_section_names_each_term_and_where_it_is_written() {
        let a = term("avatar", "Avatars are vat-grown clones.");
        assert_eq!(
            render(&[&a]),
            "\n## What the world's own words mean — as its record defines them\n\n- **avatar** \
             (`layers/world/avatar.md`): Avatars are vat-grown clones.\n"
        );
        assert_eq!(render(&[]), "");
    }
}
