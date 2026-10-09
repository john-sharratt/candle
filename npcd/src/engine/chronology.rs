//! The world's history as a list of eras by the year each opens — what a time
//! machine checks a year against, and what decides which eras reach a
//! character standing in one.

use std::path::Path;

use crate::engine::mission_gen::corpus::{era_year, heading};

/// One era: where it is written, what it is called, and the year it opens.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Era {
    /// Its address as ingested: `eras/the-tower-age`.
    pub address: String,
    pub title: String,
    pub opens: u32,
}

/// Every dated era under `<mind>/layers/eras`, earliest first.
pub fn eras(mind: &Path) -> Vec<Era> {
    let Ok(entries) = std::fs::read_dir(mind.join("layers").join("eras")) else {
        return Vec::new();
    };
    let mut out: Vec<Era> = entries
        .flatten()
        .filter_map(|e| {
            let name = e.file_name().to_string_lossy().into_owned();
            let stem = name.strip_suffix(".md")?.to_string();
            let text = std::fs::read_to_string(e.path()).ok()?;
            Some(Era {
                address: format!("eras/{stem}"),
                title: heading(&text).unwrap_or_else(|| stem.clone()),
                opens: era_year(&text)?,
            })
        })
        .collect();
    out.sort_by(|a, b| (a.opens, &a.address).cmp(&(b.opens, &b.address)));
    out
}

/// The era `year` falls in — the last to open on or before it — and the one
/// before that: what a character standing in `year` recalls of the storyline.
/// Empty for a year before the first era opens.
pub fn standing_in(eras: &[Era], year: u32) -> Vec<&Era> {
    let Some(at) = eras.iter().rposition(|e| e.opens <= year) else {
        return Vec::new();
    };
    eras[at.saturating_sub(1)..=at].iter().collect()
}

/// The year a document is set in, from its text: the first `NNNN CE`, or
/// "the year is NNNN" — the forms the record dates itself in. `None` for a
/// document that does not say.
pub fn dated(text: &str) -> Option<u32> {
    if let Some(year) = era_year(text) {
        return Some(year);
    }
    let lower = text.to_ascii_lowercase();
    let at = lower.find("the year is ")? + "the year is ".len();
    let digits: String = lower[at..]
        .chars()
        .take_while(|c| c.is_ascii_digit())
        .collect();
    digits.parse().ok()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn era(address: &str, opens: u32) -> Era {
        Era {
            address: address.into(),
            title: address.into(),
            opens,
        }
    }

    /// **Standing in a year recalls its era and the one before — nothing
    /// after it, and not the distant past.**
    #[test]
    fn a_year_recalls_its_era_and_the_one_before() {
        let eras = [
            era("eras/golden", 2151),
            era("eras/awakening", 2487),
            era("eras/tower", 2837),
            era("eras/contested", 2937),
        ];
        let at = |y| -> Vec<&str> {
            standing_in(&eras, y)
                .iter()
                .map(|e| e.address.as_str())
                .collect()
        };
        assert_eq!(at(2950), ["eras/tower", "eras/contested"]);
        assert_eq!(at(2936), ["eras/awakening", "eras/tower"]);
        assert_eq!(at(2491), ["eras/golden", "eras/awakening"]);
        assert_eq!(at(2200), ["eras/golden"], "nothing before the first");
        assert!(at(2000).is_empty(), "before the world's history opens");
    }

    #[test]
    fn a_document_is_dated_by_its_era_line_or_its_own_words() {
        assert_eq!(dated("# T\n\n*Tessin, 2788 CE*\n\nIt rained."), Some(2788));
        assert_eq!(
            dated("Dawn. The year is 3087, and the count is done."),
            Some(3087)
        );
        assert_eq!(dated("Nobody wrote down when."), None);
    }

    #[test]
    fn the_eras_are_read_off_the_mind_earliest_first() {
        let dir = tempfile::tempdir().unwrap();
        let eras_dir = dir.path().join("layers").join("eras");
        std::fs::create_dir_all(&eras_dir).unwrap();
        std::fs::write(
            eras_dir.join("the-tower-age.md"),
            "# The Tower Age\n\n**Era 350–450 · 2837–2937 CE**\n",
        )
        .unwrap();
        std::fs::write(
            eras_dir.join("the-awakening.md"),
            "# The Awakening\n\n**Era 0 · 2487 CE**\n",
        )
        .unwrap();
        assert_eq!(
            eras(dir.path()),
            [
                Era {
                    address: "eras/the-awakening".into(),
                    title: "The Awakening".into(),
                    opens: 2487
                },
                Era {
                    address: "eras/the-tower-age".into(),
                    title: "The Tower Age".into(),
                    opens: 2837
                }
            ]
        );
    }
}
