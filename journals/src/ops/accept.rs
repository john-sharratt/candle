//! Accepting a worker's proposal (§7.2): the note stays, no longer proposed.

use crate::error::{Error, Result};
use crate::ops::Applied;
use crate::path::Address;
use crate::tree::Tree;

pub fn accept(tree: &mut Tree, address: &Address) -> Result<Applied> {
    let j = tree.need_mut(&address.path)?;
    j.writable(&address.path.show())?;
    let line = j
        .line_mut(&address.id)
        .ok_or_else(|| Error::not_found(format!("{address} is no line")))?;
    if !line.proposed {
        return Err(Error::refused(format!("{address} is no proposal")));
    }
    line.proposed = false;
    Ok(Applied {
        line: Some(address.id.clone()),
        ..Applied::touching(format!("Accept {address}"), vec![address.path.clone()])
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::actor::Actor;
    use crate::ops::fixture::{john, now, p, sharratt, thanhlong};
    use crate::ops::{apply, Op};
    use crate::testing::slug;

    #[test]
    fn the_person_a_proposal_was_made_for_accepts_it_once() {
        let mut t = sharratt();
        let path = p("projects");
        let w = Actor::Worker {
            person: slug("john"),
        };
        let id = apply(
            &mut t,
            &w,
            &Op::Note {
                path: path.clone(),
                text: "Trials run on BlackWorld.".into(),
            },
            now(),
        )
        .unwrap()
        .line
        .unwrap();
        let address = Address {
            path: path.clone(),
            id,
        };
        assert!(apply(
            &mut t,
            &thanhlong(),
            &Op::Accept {
                address: address.clone()
            },
            now()
        )
        .is_err());
        let a = apply(
            &mut t,
            &john(),
            &Op::Accept {
                address: address.clone(),
            },
            now(),
        )
        .unwrap();
        assert!(a.subject.starts_with("Accept @sharratt/projects#"));
        assert!(!t.get(&path).unwrap().window[0].proposed);
        assert!(apply(&mut t, &john(), &Op::Accept { address }, now())
            .unwrap_err()
            .to_string()
            .contains("is no proposal"));
    }
}
