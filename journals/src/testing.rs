//! Fixtures the tests share.

use crate::line::{Kind, Line, LineId, Writer};
use crate::slug::Slug;
use crate::terms::Terms;
use crate::time::Stamp;

pub(crate) fn at(s: &str) -> Stamp {
    Stamp::parse(s).unwrap()
}

pub(crate) fn slug(s: &str) -> Slug {
    Slug::parse(s).unwrap()
}

/// A note John's assistant kept at `when`.
pub(crate) fn note(when: &str, text: &str) -> Line {
    let at = at(when);
    let by = Writer::Person;
    Line {
        id: LineId::mint(at, by, text, 0),
        at,
        kind: Kind::Note,
        by,
        account: Some(slug("john")),
        speaker: None,
        folds: None,
        proposed: false,
        terms: Terms::default(),
        experience: None,
        text: text.into(),
    }
}
