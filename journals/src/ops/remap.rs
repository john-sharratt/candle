//! Line ids an operation names, rewritten. A line replayed onto a newer
//! origin keeps the id it was written with unless that id is now another
//! line's; then it takes a new one, and every later replayed operation that
//! named the old id is rewritten to the new before it runs.

use std::collections::BTreeMap;

use crate::files::summary::Cite;
use crate::line::LineId;
use crate::ops::{Op, Which};

pub fn remap(op: &mut Op, renamed: &BTreeMap<LineId, LineId>) {
    if renamed.is_empty() {
        return;
    }
    let swap = |id: &mut LineId| {
        if let Some(new) = renamed.get(id) {
            *id = new.clone();
        }
    };
    match op {
        Op::Edit { address, .. } | Op::Accept { address } => swap(&mut address.id),
        Op::Forget {
            which: Which::Id(id),
            ..
        } => swap(id),
        Op::FoldPeriod { ids, .. } => ids.iter_mut().for_each(swap),
        Op::Summarize { sentences, .. } => {
            for s in sentences {
                for c in &mut s.cites {
                    if let Cite::Line(id) = c {
                        swap(id);
                    }
                }
            }
        }
        Op::Note { .. }
        | Op::Episode { .. }
        | Op::Forget {
            which: Which::Says(_),
            ..
        }
        | Op::Create { .. }
        | Op::Close { .. }
        | Op::Fold { .. }
        | Op::Unfold { .. }
        | Op::Merge { .. }
        | Op::MarkStale { .. }
        | Op::Share { .. }
        | Op::CreateGroup { .. }
        | Op::DescribeGroup { .. }
        | Op::ChooseGroups { .. }
        | Op::AddMember { .. }
        | Op::AddBot { .. }
        | Op::RemoveMember { .. }
        | Op::Bind { .. } => {}
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::files::summary::{Fingerprint, Sentence};
    use crate::ops::fixture::p;
    use crate::path::Address;
    use crate::period::Period;
    use crate::testing::slug;

    fn id(s: &str) -> LineId {
        LineId::parse(s).unwrap()
    }

    #[test]
    fn every_named_id_is_rewritten_and_nothing_else() {
        let old = id("2026-10-30-0912-aaaa");
        let new = id("2026-10-30-0912-bbbb");
        let other = id("2026-10-30-0912-cccc");
        let renamed = BTreeMap::from([(old.clone(), new.clone())]);
        let address = |i: &LineId| Address {
            path: p("topics"),
            id: i.clone(),
        };
        let cases = [
            (
                Op::Edit {
                    address: address(&old),
                    text: "x".into(),
                },
                Op::Edit {
                    address: address(&new),
                    text: "x".into(),
                },
            ),
            (
                Op::Accept {
                    address: address(&old),
                },
                Op::Accept {
                    address: address(&new),
                },
            ),
            (
                Op::Forget {
                    path: p("topics"),
                    which: Which::Id(old.clone()),
                },
                Op::Forget {
                    path: p("topics"),
                    which: Which::Id(new.clone()),
                },
            ),
            (
                Op::FoldPeriod {
                    path: p("~"),
                    period: Period::parse("2026-10-30").unwrap(),
                    ids: vec![old.clone(), other.clone()],
                    over: Fingerprint::of_bytes(b"x"),
                    text: "x".into(),
                },
                Op::FoldPeriod {
                    path: p("~"),
                    period: Period::parse("2026-10-30").unwrap(),
                    ids: vec![new.clone(), other.clone()],
                    over: Fingerprint::of_bytes(b"x"),
                    text: "x".into(),
                },
            ),
            (
                Op::Summarize {
                    path: p("~"),
                    over: Fingerprint::of_bytes(b""),
                    sentences: vec![Sentence {
                        text: "s".into(),
                        cites: vec![Cite::Line(old.clone()), Cite::Child(slug("channels"))],
                    }],
                    digest: None,
                    cue: None,
                },
                Op::Summarize {
                    path: p("~"),
                    over: Fingerprint::of_bytes(b""),
                    sentences: vec![Sentence {
                        text: "s".into(),
                        cites: vec![Cite::Line(new.clone()), Cite::Child(slug("channels"))],
                    }],
                    digest: None,
                    cue: None,
                },
            ),
            (
                Op::Edit {
                    address: address(&other),
                    text: "x".into(),
                },
                Op::Edit {
                    address: address(&other),
                    text: "x".into(),
                },
            ),
        ];
        for (mut before, after) in cases {
            remap(&mut before, &renamed);
            assert_eq!(before, after);
        }
    }
}
