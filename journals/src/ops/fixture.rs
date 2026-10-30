//! The tree the operation tests start from: the Sharratt family — John and
//! Yen its owners, Thanh Long a member, the bot c3f1a07e a reader — and
//! Stranger, in no group. John is the estate's admin.

use std::collections::BTreeMap;

use crate::actor::{viewer, Actor};
use crate::path::{JournalPath, Viewer};
use crate::scope::Channel;
use crate::testing::slug;
use crate::time::Stamp;
use crate::tree::load::load;
use crate::tree::Tree;

pub(crate) fn sharratt() -> Tree {
    let person = |p: &str, in_group: bool| {
        let groups = if in_group {
            "groups: [sharratt]\ndefault_group: sharratt\n"
        } else {
            "groups: []\n"
        };
        (
            format!("persons/{p}/person.yaml"),
            format!(
                "person: {p}\nname: \"{p}\"\naccounts:\n  - {{email: \"{p}@example.com\"}}\n{groups}"
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
        "groups/sharratt/group.yaml".into(),
        "group: sharratt\nname: \"Sharratt\"\nkind: family\nmembers:\n\
         \x20 - {person: john, role: owner, joined: 2026-10-30T00:00:00Z}\n\
         \x20 - {person: yen, role: owner, joined: 2026-10-30T00:00:00Z}\n\
         \x20 - {person: thanhlong, role: member, joined: 2026-10-30T00:00:00Z}\n\
         \x20 - {bot: c3f1a07e, role: reader, joined: 2026-10-30T00:00:00Z}\n"
            .into(),
    );
    let tree = load(&files);
    assert!(tree.problems.is_empty(), "{:?}", tree.problems);
    tree
}

/// John, the estate's admin: he lists accounts' emails.
pub(crate) fn john() -> Actor {
    Actor::Person {
        person: slug("john"),
        channel: Channel::ClaudeCode,
        admin: true,
    }
}

/// Yen, an owner of @sharratt and no admin.
pub(crate) fn yen() -> Actor {
    Actor::Person {
        person: slug("yen"),
        channel: Channel::Grok,
        admin: false,
    }
}

pub(crate) fn thanhlong() -> Actor {
    Actor::Person {
        person: slug("thanhlong"),
        channel: Channel::Other,
        admin: false,
    }
}

pub(crate) fn stranger() -> Actor {
    Actor::Person {
        person: slug("stranger"),
        channel: Channel::Other,
        admin: false,
    }
}

/// The fixture's bot, a reader of @sharratt.
pub(crate) fn bot() -> Actor {
    Actor::Bot {
        bot: slug("c3f1a07e"),
    }
}

/// A path as John writes it.
pub(crate) fn p(s: &str) -> JournalPath {
    p_as("john", s)
}

/// A path as `person` writes it, in the fixture's groups.
pub(crate) fn p_as(person: &str, s: &str) -> JournalPath {
    Viewer {
        person: slug(person),
        default_group: (person != "stranger").then(|| slug("sharratt")),
    }
    .parse(s)
    .unwrap()
}

/// The fixture's groups are what the tree's files say they are.
#[test]
fn the_fixture_viewers_match_the_tree() {
    let t = sharratt();
    for person in ["john", "yen", "thanhlong", "stranger"] {
        assert_eq!(
            viewer(&t, &slug(person)),
            Viewer {
                person: slug(person),
                default_group: (person != "stranger").then(|| slug("sharratt")),
            }
        );
    }
}

pub(crate) fn now() -> Stamp {
    Stamp::parse("2026-10-30T12:00:00Z").unwrap()
}
