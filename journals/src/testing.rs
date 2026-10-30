//! Fixtures the tests share.

use crate::line::{Kind, Line, LineId, Writer};
use crate::scope::Channel;
use crate::slug::Slug;
use crate::time::Stamp;

pub(crate) fn at(s: &str) -> Stamp {
    Stamp::parse(s).unwrap()
}

pub(crate) fn slug(s: &str) -> Slug {
    Slug::parse(s).unwrap()
}

/// A note John's Claude Code kept at `when`.
pub(crate) fn note(when: &str, text: &str) -> Line {
    let at = at(when);
    let by = Writer::Channel(Channel::ClaudeCode);
    Line {
        id: LineId::mint(at, by, text, 0),
        at,
        kind: Kind::Note,
        by,
        account: Some(slug("john")),
        speaker: None,
        channel: None,
        folds: None,
        proposed: false,
        text: text.into(),
    }
}
