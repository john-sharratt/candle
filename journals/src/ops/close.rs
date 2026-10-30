//! Closing a journal: its conversation, thread or task has ended. A closed
//! thread or worker is what an eleventh may fold (§2.3).

use crate::error::{Error, Result};
use crate::ops::Applied;
use crate::path::JournalPath;
use crate::time::Stamp;
use crate::tree::Tree;

pub fn close(tree: &mut Tree, path: &JournalPath, now: Stamp) -> Result<Applied> {
    let j = tree.need_mut(path)?;
    j.writable(&path.show())?;
    let fixed = j.is_fixed();
    let meta = match (&mut j.meta, fixed) {
        (Some(m), false) => m,
        _ => return Err(Error::refused(format!("{path} is fixed: it never closes"))),
    };
    if let Some(at) = meta.closed {
        return Err(Error::refused(format!("{path} closed at {at}")));
    }
    meta.closed = Some(now);
    Ok(Applied::touching(
        format!("Close {path}"),
        vec![path.clone()],
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::actor::Actor;
    use crate::ops::fixture::{now, p, sharratt};
    use crate::ops::{apply, OnTaken, Op};
    use crate::scope::Client;
    use crate::testing::slug;

    #[test]
    fn a_conversation_closes_once() {
        let mut t = sharratt();
        let server = Actor::Server {
            person: slug("john"),
            client: Client::Grok,
        };
        let created = apply(
            &mut t,
            &server,
            &Op::Create {
                under: p("~/clients/grok"),
                name: slug("2026-10-30-2105"),
                cue: "a drive".into(),
                on_taken: OnTaken::Number,
            },
            now(),
        )
        .unwrap()
        .created
        .unwrap();
        apply(
            &mut t,
            &server,
            &Op::Close {
                path: created.clone(),
            },
            now().plus(600),
        )
        .unwrap();
        assert_eq!(
            t.get(&created).unwrap().meta.as_ref().unwrap().closed,
            Some(now().plus(600))
        );
        assert!(apply(&mut t, &server, &Op::Close { path: created }, now())
            .unwrap_err()
            .to_string()
            .contains("closed at"));
        assert!(close(&mut t, &p("~/clients/grok"), now())
            .unwrap_err()
            .to_string()
            .contains("is fixed"));
    }
}
