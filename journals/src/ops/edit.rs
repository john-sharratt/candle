//! Changing a line's text. Its id, time and writer stay: an edit is a commit
//! that changes one line, and history keeps what it said before.

use crate::error::{Error, Result};
use crate::line::Kind;
use crate::ops::append::clean;
use crate::ops::Applied;
use crate::path::Address;
use crate::tree::Tree;

pub fn edit(tree: &mut Tree, address: &Address, text: &str) -> Result<Applied> {
    let text = clean(text)?;
    let j = tree.need_mut(&address.path)?;
    j.writable(&address.path.show())?;
    let line = j
        .line_mut(&address.id)
        .ok_or_else(|| Error::not_found(format!("{address} is no line")))?;
    if line.kind == Kind::Fold {
        return Err(Error::refused(format!(
            "{address} is a fold, which zend writes; edit the lines it folded"
        )));
    }
    if line.text == text {
        return Err(Error::refused(format!("{address} already says that")));
    }
    line.text = text;
    Ok(Applied {
        line: Some(address.id.clone()),
        ..Applied::touching(format!("Edit of {address}"), vec![address.path.clone()])
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::line::{Folds, LineId};
    use crate::ops::fixture::{john, now, p, zen};
    use crate::ops::{apply, Op};
    use crate::period::Period;
    use crate::testing::note;

    #[test]
    fn an_edit_changes_the_text_and_nothing_else() {
        let mut t = zen();
        let path = p("people");
        let id = apply(
            &mut t,
            &john(),
            &Op::Note {
                path: path.clone(),
                text: "Prefers tea.".into(),
                speaker: None,
            },
            now(),
        )
        .unwrap()
        .line
        .unwrap();
        let before = t.get(&path).unwrap().window[0].clone();
        let address = Address {
            path: path.clone(),
            id: id.clone(),
        };
        let a = apply(
            &mut t,
            &john(),
            &Op::Edit {
                address: address.clone(),
                text: "Prefers green tea.".into(),
            },
            now().plus(60),
        )
        .unwrap();
        assert_eq!(a.subject, format!("Edit of @zen/people#{id}"));
        let after = &t.get(&path).unwrap().window[0];
        assert_eq!(after.text, "Prefers green tea.");
        assert_eq!(
            (&after.id, after.at, after.by),
            (&before.id, before.at, before.by)
        );
        assert!(apply(
            &mut t,
            &john(),
            &Op::Edit {
                address,
                text: "Prefers green tea.".into()
            },
            now()
        )
        .unwrap_err()
        .to_string()
        .contains("already says that"));
    }

    #[test]
    fn an_archived_line_is_edited_where_it_is() {
        let mut t = zen();
        let path = p("people");
        let line = note("2026-09-01T10:00:00Z", "old");
        let period = Period::parse("2026-09-01").unwrap();
        t.get_mut(&path)
            .unwrap()
            .archives
            .insert(period, vec![line.clone()]);
        apply(
            &mut t,
            &john(),
            &Op::Edit {
                address: Address {
                    path: path.clone(),
                    id: line.id.clone(),
                },
                text: "old, corrected".into(),
            },
            now(),
        )
        .unwrap();
        assert_eq!(
            t.get(&path).unwrap().archives[&period][0].text,
            "old, corrected"
        );
    }

    #[test]
    fn no_such_line_and_folds_are_refused() {
        let mut t = zen();
        let path = p("people");
        let id = LineId::parse("2026-10-30-0912-7e3a").unwrap();
        assert!(matches!(
            apply(
                &mut t,
                &john(),
                &Op::Edit {
                    address: Address {
                        path: path.clone(),
                        id
                    },
                    text: "x".into()
                },
                now()
            ),
            Err(Error::NotFound(_))
        ));
        let mut fold = note("2026-10-29T10:00:00Z", "a day");
        fold.kind = Kind::Fold;
        fold.folds = Some(Folds::Period(Period::parse("2026-10-29").unwrap()));
        t.get_mut(&path).unwrap().window.push(fold.clone());
        assert!(matches!(
            apply(
                &mut t,
                &john(),
                &Op::Edit {
                    address: Address { path, id: fold.id },
                    text: "x".into()
                },
                now()
            ),
            Err(Error::Refused(_))
        ));
    }
}
