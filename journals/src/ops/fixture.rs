//! The tree the operation tests start from: the group Zen — John and Yen its
//! owners, Thanh Long and Zen itself members, the bot c3f1a07e a reader of
//! `topics/game` — `@guest`, which John owns, and Stranger, in no group. John
//! is the estate's admin.

use std::collections::BTreeMap;

use crate::actor::{viewer, Actor};
use crate::path::{JournalPath, Viewer};
use crate::scope::Channel;
use crate::testing::slug;
use crate::time::Stamp;
use crate::tree::load::load;
use crate::tree::Tree;

pub(crate) fn zen() -> Tree {
    let person = |p: &str, accounts: bool, groups: &str| {
        let accounts = if accounts {
            format!("accounts:\n  - {{email: \"{p}@example.com\"}}\n")
        } else {
            "accounts: []\n".into()
        };
        (
            format!("persons/{p}/person.yaml"),
            format!("person: {p}\nname: \"{p}\"\n{accounts}{groups}").into_bytes(),
        )
    };
    let in_zen = "groups: [zen]\ndefault_group: zen\n";
    let mut files: BTreeMap<String, Vec<u8>> = BTreeMap::from([
        person("john", true, "groups: [zen, guest]\ndefault_group: zen\n"),
        person("yen", true, in_zen),
        person("thanhlong", true, in_zen),
        person("zen", false, in_zen),
        person("stranger", true, "groups: []\n"),
    ]);
    files.insert(
        "groups/zen/group.yaml".into(),
        "group: zen\nname: \"Zen\"\nmembers:\n\
         \x20 - {person: john, role: owner, joined: 2026-10-30T00:00:00Z}\n\
         \x20 - {person: yen, role: owner, joined: 2026-10-30T00:00:00Z}\n\
         \x20 - {person: thanhlong, role: member, joined: 2026-10-30T00:00:00Z}\n\
         \x20 - {person: zen, role: member, joined: 2026-10-31T00:00:00Z}\n\
         \x20 - {bot: c3f1a07e, role: reader, joined: 2026-10-30T00:00:00Z, reads: [topics/game]}\n"
            .into(),
    );
    files.insert(
        "groups/guest/group.yaml".into(),
        "group: guest\nname: \"Guest\"\nmembers:\n\
         \x20 - {person: john, role: owner, joined: 2026-10-31T00:00:00Z}\n"
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

/// Yen, an owner of @zen and no admin.
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

/// An account no person lists.
pub(crate) fn guest() -> Actor {
    Actor::Guest {
        channel: Channel::Other,
    }
}

/// The fixture's bot, a reader of @zen's `topics/game`.
pub(crate) fn bot() -> Actor {
    Actor::Bot {
        bot: slug("c3f1a07e"),
    }
}

/// A path as written in @zen.
pub(crate) fn p(s: &str) -> JournalPath {
    Viewer {
        group: Some(slug("zen")),
        here: None,
    }
    .parse(s)
    .unwrap()
}

/// The fixture's people write their paths from @zen.
#[test]
fn the_fixture_viewers_match_the_tree() {
    let t = zen();
    for person in ["john", "yen", "thanhlong", "zen"] {
        assert_eq!(
            viewer(&t, &slug(person), None).group,
            Some(slug("zen")),
            "{person}"
        );
    }
    assert_eq!(viewer(&t, &slug("stranger"), None).group, None);
}

pub(crate) fn now() -> Stamp {
    Stamp::parse("2026-10-30T12:00:00Z").unwrap()
}
