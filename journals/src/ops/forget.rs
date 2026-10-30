//! Forgetting a line (§8): it leaves the journal's current state — its
//! window or archive, and so frames, summaries at their next rebuild, and
//! grep. The commit that removes it keeps it in history, forever.

use crate::error::{Error, Result};
use crate::line::Line;
use crate::ops::{Applied, Which};
use crate::path::JournalPath;
use crate::tree::{Held, Journal, Tree};

/// The one line `which` names in `j`.
pub fn find_line<'a>(j: &'a Journal, which: &Which, path: &JournalPath) -> Result<&'a Line> {
    match which {
        Which::Id(id) => j
            .line(id)
            .ok_or_else(|| Error::not_found(format!("{path}#{id} is no line"))),
        Which::Says(phrase) => {
            let want = phrase.trim().to_lowercase();
            if want.is_empty() {
                return Err(Error::invalid("say which line: the phrase is empty"));
            }
            let hits: Vec<&Line> = j
                .all_lines()
                .filter(|l| l.text.to_lowercase().contains(&want))
                .collect();
            match hits.as_slice() {
                [one] => Ok(one),
                [] => Err(Error::not_found(format!(
                    "no line of {path} says “{}”",
                    phrase.trim()
                ))),
                many => Err(Error::invalid(format!(
                    "{} lines of {path} say “{}” — name one: {}",
                    many.len(),
                    phrase.trim(),
                    many.iter()
                        .map(|l| format!("#{} “{}”", l.id, l.text))
                        .collect::<Vec<_>>()
                        .join("; ")
                ))),
            }
        }
    }
}

pub fn forget(tree: &mut Tree, path: &JournalPath, which: &Which) -> Result<Applied> {
    let j = tree.need_mut(path)?;
    j.writable(&path.show())?;
    let id = find_line(j, which, path)?.id.clone();
    match j.find(&id).expect("just found") {
        Held::Window(i) => {
            j.window.remove(i);
        }
        Held::Archive(period, i) => {
            let lines = j.archives.get_mut(&period).expect("just found");
            lines.remove(i);
            if lines.is_empty() {
                j.archives.remove(&period);
            }
        }
    }
    Ok(Applied {
        line: Some(id.clone()),
        ..Applied::touching(format!("Forget {path}#{id}"), vec![path.clone()])
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ops::fixture::{john, now, p, zen};
    use crate::ops::{apply, Op};
    use crate::period::Period;
    use crate::testing::note;

    fn seeded() -> (Tree, JournalPath) {
        let mut t = zen();
        let path = p("people");
        for text in [
            "Prefers tea.",
            "Drives a Model Y.",
            "Prefers short answers.",
        ] {
            apply(
                &mut t,
                &john(),
                &Op::Note {
                    path: path.clone(),
                    text: text.into(),
                    speaker: None,
                },
                now(),
            )
            .unwrap();
        }
        (t, path)
    }

    #[test]
    fn a_line_is_forgotten_by_a_phrase_it_alone_says() {
        let (mut t, path) = seeded();
        let a = apply(
            &mut t,
            &john(),
            &Op::Forget {
                path: path.clone(),
                which: Which::Says("MODEL y".into()),
            },
            now(),
        )
        .unwrap();
        let left: Vec<&str> = t
            .get(&path)
            .unwrap()
            .window
            .iter()
            .map(|l| l.text.as_str())
            .collect();
        assert_eq!(left, ["Prefers tea.", "Prefers short answers."]);
        assert!(a.subject.starts_with("Forget @zen/people#"));
    }

    #[test]
    fn a_phrase_two_lines_say_names_both_and_forgets_neither() {
        let (mut t, path) = seeded();
        let err = apply(
            &mut t,
            &john(),
            &Op::Forget {
                path: path.clone(),
                which: Which::Says("prefers".into()),
            },
            now(),
        )
        .unwrap_err();
        let msg = err.to_string();
        assert!(
            msg.starts_with("2 lines of @zen/people say “prefers” — name one: #"),
            "{msg}"
        );
        assert!(
            msg.contains("“Prefers tea.”") && msg.contains("“Prefers short answers.”"),
            "{msg}"
        );
        assert_eq!(t.get(&path).unwrap().window.len(), 3);
        assert!(matches!(
            apply(
                &mut t,
                &john(),
                &Op::Forget {
                    path: path.clone(),
                    which: Which::Says("coffee".into())
                },
                now()
            ),
            Err(Error::NotFound(_))
        ));
        assert!(matches!(
            apply(
                &mut t,
                &john(),
                &Op::Forget {
                    path,
                    which: Which::Says("  ".into())
                },
                now()
            ),
            Err(Error::Invalid(_))
        ));
    }

    #[test]
    fn forgetting_an_archives_last_line_removes_the_archive() {
        let mut t = zen();
        let path = p("people");
        let old = note("2026-09-01T10:00:00Z", "old");
        let period = Period::parse("2026-09-01").unwrap();
        t.get_mut(&path)
            .unwrap()
            .archives
            .insert(period, vec![old.clone()]);
        apply(
            &mut t,
            &john(),
            &Op::Forget {
                path: path.clone(),
                which: Which::Id(old.id),
            },
            now(),
        )
        .unwrap();
        assert!(t.get(&path).unwrap().archives.is_empty());
    }
}
