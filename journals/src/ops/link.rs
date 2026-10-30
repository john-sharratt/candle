//! A new reference (§2.5): a named link from one journal to another in the
//! same group, so a path decoded from one end reaches the other. Only a
//! restructuring removes or retargets one (§2.6).

use crate::error::{Error, Result};
use crate::files::{Meta, Ref, MAX_REFS};
use crate::ops::create::clean_cue;
use crate::ops::summarize::fixed_name;
use crate::ops::Applied;
use crate::path::{Below, JournalPath};
use crate::slug::Slug;
use crate::time::Stamp;
use crate::tree::Tree;

pub fn link(
    tree: &mut Tree,
    under: &JournalPath,
    name: &Slug,
    to: &JournalPath,
    cue: &str,
    now: Stamp,
) -> Result<Applied> {
    let cue = clean_cue(cue)?;
    if to.group() != under.group() {
        return Err(Error::invalid(format!(
            "a reference stays in its group: {to} is not in @{}",
            under.group()
        )));
    }
    if to == under {
        return Err(Error::invalid(format!("{under} would lead to itself")));
    }
    if to.is_root() {
        return Err(Error::invalid(format!(
            "every path starts at {to} already: a reference leads to a journal below it"
        )));
    }
    tree.need(to)?;
    let j = tree.need_mut(under)?;
    j.writable(&under.show())?;
    if j.children.contains_key(name) {
        return Err(Error::refused(format!(
            "{under}/{name} is a sub-journal already"
        )));
    }
    // A master or section has no `journal.yaml` until something needs one.
    let own = fixed_name(under, j);
    let meta = j.meta.get_or_insert_with(|| Meta::new(own, None, now));
    if let Some(r) = meta.refs.iter().find(|r| &r.name == name) {
        return Err(Error::refused(format!(
            "{under} has a reference {name} already, to {}",
            r.to
        )));
    }
    if meta.refs.len() >= MAX_REFS {
        return Err(Error::refused(format!(
            "{under} has {MAX_REFS} references; changing them is a coding task on the journals repository"
        )));
    }
    meta.refs.push(Ref {
        name: name.clone(),
        to: Below::from(to),
        cue,
    });
    Ok(Applied::touching(
        format!("Link {under}/{name} to {to}"),
        vec![under.clone()],
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ops::fixture::{john, now, p, zen};
    use crate::ops::{apply, OnTaken, Op};
    use crate::testing::slug;

    fn with_topics() -> Tree {
        let mut t = zen();
        for name in ["candle", "wasmer"] {
            apply(
                &mut t,
                &john(),
                &Op::Create {
                    under: p("topics"),
                    name: slug(name),
                    cue: format!("open for {name}"),
                    on_taken: OnTaken::Refuse,
                },
                now(),
            )
            .unwrap();
        }
        t
    }

    fn link_op(under: &str, name: &str, to: &str) -> Op {
        Op::Link {
            under: p(under),
            name: slug(name),
            to: p(to),
            cue: format!("open for {name}"),
        }
    }

    /// **A reference makes a second way to a journal**, resolved to the
    /// journal itself.
    #[test]
    fn a_reference_leads_to_its_journal() {
        let mut t = with_topics();
        let a = apply(
            &mut t,
            &john(),
            &link_op("topics/candle", "wasmer", "topics/wasmer"),
            now(),
        )
        .unwrap();
        assert_eq!(
            a.subject,
            "Link @zen/topics/candle/wasmer to @zen/topics/wasmer"
        );
        assert_eq!(a.touched, vec![p("topics/candle")]);
        assert_eq!(
            t.resolve(&p("topics/candle/wasmer")).unwrap(),
            p("topics/wasmer")
        );
        // A section takes one too, its `journal.yaml` made with its fixed name.
        apply(
            &mut t,
            &john(),
            &link_op("people", "engine", "topics/candle"),
            now(),
        )
        .unwrap();
        let people = t.get(&p("people")).unwrap().meta.as_ref().unwrap();
        assert_eq!(people.name, "people");
        assert_eq!(people.refs[0].to, Below::parse("topics/candle").unwrap());
    }

    #[test]
    fn a_reference_is_refused_where_it_would_mislead() {
        let mut t = with_topics();
        apply(
            &mut t,
            &john(),
            &link_op("topics/candle", "wasmer", "topics/wasmer"),
            now(),
        )
        .unwrap();
        for (op, says) in [
            (
                link_op("topics/candle", "wasmer", "topics/wasmer"),
                "has a reference wasmer already",
            ),
            (
                link_op("topics", "candle", "topics/wasmer"),
                "is a sub-journal already",
            ),
            (
                link_op("topics/candle", "x", "topics/candle"),
                "would lead to itself",
            ),
            (
                link_op("topics/candle", "x", "topics/nothing"),
                "there is no journal @zen/topics/nothing",
            ),
            (
                link_op("topics/candle", "x", "@guest/topics"),
                "a reference stays in its group",
            ),
            (
                link_op("topics/candle", "home", "@zen"),
                "every path starts at @zen already",
            ),
        ] {
            let err = apply(&mut t, &john(), &op, now()).unwrap_err().to_string();
            assert!(err.contains(says), "{err}");
        }
        for i in 1..MAX_REFS {
            apply(
                &mut t,
                &john(),
                &link_op("topics/candle", &format!("r{i}"), "topics/wasmer"),
                now(),
            )
            .unwrap();
        }
        assert_eq!(
            apply(&mut t, &john(), &link_op("topics/candle", "r10", "topics/wasmer"), now())
                .unwrap_err()
                .to_string(),
            "@zen/topics/candle has 10 references; changing them is a coding task on the journals repository"
        );
    }
}
