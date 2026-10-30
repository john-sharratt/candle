//! The tree the operation tests start from: the Sharratt family — John and
//! Yen its owners, Thanh Long a member — and Stranger, in no family.

use std::collections::BTreeMap;

use crate::actor::{viewer, Actor};
use crate::path::{JournalPath, Viewer};
use crate::scope::Client;
use crate::testing::slug;
use crate::time::Stamp;
use crate::tree::load::load;
use crate::tree::Tree;

pub(crate) fn sharratt() -> Tree {
    let person = |p: &str, fam: bool| {
        let fams = if fam {
            "families: [sharratt]\ndefault_family: sharratt\n"
        } else {
            "families: []\n"
        };
        (
            format!("persons/{p}/person.yaml"),
            format!(
                "person: {p}\nname: \"{p}\"\naccounts:\n  - {{email: \"{p}@example.com\"}}\n{fams}"
            )
            .into_bytes(),
        )
    };
    let mut files: BTreeMap<String, Vec<u8>> = BTreeMap::from([
        person("john", true),
        person("yen", true),
        person("thanhlong", true),
        person("stranger", false),
    ]);
    files.insert(
        "families/sharratt/family.yaml".into(),
        "family: sharratt\nname: \"Sharratt\"\nmembers:\n\
         \x20 - {person: john, role: owner, joined: 2026-10-30T00:00:00Z}\n\
         \x20 - {person: yen, role: owner, joined: 2026-10-30T00:00:00Z}\n\
         \x20 - {person: thanhlong, role: member, joined: 2026-10-30T00:00:00Z}\n"
            .into(),
    );
    let tree = load(&files);
    assert!(tree.problems.is_empty(), "{:?}", tree.problems);
    tree
}

pub(crate) fn john() -> Actor {
    Actor::Person {
        person: slug("john"),
        client: Client::ClaudeCode,
    }
}

pub(crate) fn yen() -> Actor {
    Actor::Person {
        person: slug("yen"),
        client: Client::Grok,
    }
}

pub(crate) fn thanhlong() -> Actor {
    Actor::Person {
        person: slug("thanhlong"),
        client: Client::Other,
    }
}

pub(crate) fn stranger() -> Actor {
    Actor::Person {
        person: slug("stranger"),
        client: Client::Other,
    }
}

/// A path as John writes it.
pub(crate) fn p(s: &str) -> JournalPath {
    p_as("john", s)
}

/// A path as `person` writes it, in the fixture's families.
pub(crate) fn p_as(person: &str, s: &str) -> JournalPath {
    let families = if person == "stranger" {
        vec![]
    } else {
        vec![slug("sharratt")]
    };
    Viewer {
        person: slug(person),
        families,
    }
    .parse(s)
    .unwrap()
}

/// The fixture's family is what the tree's files say it is.
#[test]
fn the_fixture_viewers_match_the_tree() {
    let t = sharratt();
    for person in ["john", "yen", "thanhlong", "stranger"] {
        let v = viewer(&t, &slug(person));
        let families = if person == "stranger" {
            vec![]
        } else {
            vec![slug("sharratt")]
        };
        assert_eq!(
            v,
            Viewer {
                person: slug(person),
                families
            }
        );
    }
}

pub(crate) fn now() -> Stamp {
    Stamp::parse("2026-10-30T12:00:00Z").unwrap()
}
