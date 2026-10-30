//! Who is writing (§7): a member's AI through a channel, the connector's
//! server, zend, a Zen worker, npcd, a bot, a guest — each with what it may
//! read and write (§7.2). The writer a line records and the trailers a commit
//! carries come from here, never from what a caller says of itself.

use crate::error::{Error, Result};
use crate::files::Role;
use crate::line::Writer;
use crate::path::{JournalPath, Viewer};
use crate::scope::Channel;
use crate::slug::Slug;
use crate::tree::Tree;

/// The group every account no person lists reads, and writes nothing in
/// (§3.6).
pub const GUEST: &str = "guest";

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Actor {
    /// A member's AI, through `channel`. `admin` is the estate's role table
    /// naming them an admin, as the gateway's verified identity says: only
    /// an admin lists an account's email (§3.5).
    Person {
        person: Slug,
        channel: Channel,
        admin: bool,
    },
    /// The connector's server, recording what a call of `person`'s did.
    Server { person: Slug, channel: Channel },
    /// zend's background thread: summaries, folds, workers' exports — Zen
    /// writing as itself.
    Zend,
    /// A Zen worker of `person`'s, on its own account.
    Worker { person: Slug },
    /// npcd, for a bot of a group `person` belongs to.
    Npcd { person: Slug },
    /// A bot, reading what its group grants it. It writes nothing.
    Bot { bot: Slug },
    /// An account no person lists, through `channel`: it reads `@guest` and
    /// writes nothing (§3.6).
    Guest { channel: Channel },
    /// The service itself: binding an account at its first sign-in.
    System,
}

impl Actor {
    /// The writer its lines record. A bot and a guest write none, and are
    /// named for what runs them should a refusal be logged.
    pub fn writer(&self) -> Writer {
        match self {
            Actor::Person { channel, .. } | Actor::Guest { channel } => Writer::Channel(*channel),
            Actor::Server { .. } => Writer::Server,
            Actor::Zend | Actor::System => Writer::Zend,
            Actor::Worker { .. } => Writer::Worker,
            Actor::Npcd { .. } | Actor::Bot { .. } => Writer::Npcd,
        }
    }

    /// The person whose account the call is: the line's `account`.
    pub fn person(&self) -> Option<&Slug> {
        match self {
            Actor::Person { person, .. }
            | Actor::Server { person, .. }
            | Actor::Worker { person }
            | Actor::Npcd { person } => Some(person),
            Actor::Zend | Actor::System | Actor::Bot { .. } | Actor::Guest { .. } => None,
        }
    }

    pub fn channel(&self) -> Option<Channel> {
        match self {
            Actor::Person { channel, .. }
            | Actor::Server { channel, .. }
            | Actor::Guest { channel } => Some(*channel),
            _ => None,
        }
    }

    /// Commit trailers naming who wrote (§7.1).
    pub fn trailers(&self) -> Vec<(&'static str, String)> {
        let mut out = vec![("Journal-Writer", self.writer().name().to_string())];
        if let Some(p) = self.person() {
            out.push(("Journal-Account", p.to_string()));
        }
        if let Some(c) = self.channel() {
            out.push(("Journal-Channel", c.name().to_string()));
        }
        out
    }
}

/// `person` is a member of `group`, by its `group.yaml`; their role.
pub fn role_in(tree: &Tree, person: &Slug, group: &Slug) -> Option<Role> {
    tree.group_file(group)
        .and_then(|f| f.member(person))
        .map(|m| m.role)
}

/// The groups `person` reads: those whose `group.yaml` lists them.
pub fn groups_of(tree: &Tree, person: &Slug) -> Vec<Slug> {
    tree.groups
        .keys()
        .filter(|g| role_in(tree, person, g).is_some())
        .cloned()
        .collect()
}

/// The groups that list `bot`.
pub fn groups_of_bot(tree: &Tree, bot: &Slug) -> Vec<Slug> {
    tree.groups
        .iter()
        .filter(|(_, g)| g.file.as_ref().is_ok_and(|f| f.bot(bot).is_some()))
        .map(|(slug, _)| slug.clone())
        .collect()
}

/// The subtrees `bot` reads: each grant of each group that lists it.
pub fn grants_of_bot(tree: &Tree, bot: &Slug) -> Vec<JournalPath> {
    let mut out = Vec::new();
    for group in groups_of_bot(tree, bot) {
        let member = tree.group_file(&group).and_then(|f| f.bot(bot));
        for r in member.map(|m| m.reads.as_slice()).unwrap_or_default() {
            if let Ok(p) = r.at(&group) {
                out.push(p);
            }
        }
    }
    out
}

/// The group `person`'s conversations open in: their default, while it lists
/// them. With none, none is chosen for them — even a group they are left
/// alone in would be the pick of whoever left them there, by removing them
/// from the rest (§3.5).
pub fn home_group(tree: &Tree, person: &Slug) -> Option<Slug> {
    tree.person_file(person)
        .and_then(|f| f.default_group.clone())
        .filter(|g| role_in(tree, person, g).is_some())
}

/// The viewer `person`'s paths are written relative to, in the conversation
/// `here` when there is one: its group, else their home group.
pub fn viewer(tree: &Tree, person: &Slug, here: Option<&JournalPath>) -> Viewer {
    Viewer {
        group: here
            .map(|h| h.group().clone())
            .or_else(|| home_group(tree, person)),
        here: here.cloned(),
    }
}

/// A guest's viewer: `@guest`, and no conversation of its own.
pub fn guest_viewer() -> Viewer {
    Viewer {
        group: Some(Slug::parse(GUEST).expect("a name")),
        here: None,
    }
}

/// `person` may read `path`: it is in a group that lists them, and every
/// member reads all of their group (§3.5).
pub fn may_read(tree: &Tree, person: &Slug, path: &JournalPath) -> bool {
    role_in(tree, person, path.group()).is_some()
}

/// `bot` may read `path`: it lies within one of the grants of `path`'s own
/// group, the only group whose grants can hold it.
pub fn bot_may_read(tree: &Tree, bot: &Slug, path: &JournalPath) -> bool {
    let group = path.group();
    tree.group_file(group)
        .and_then(|f| f.bot(bot))
        .is_some_and(|m| {
            m.reads
                .iter()
                .any(|r| r.at(group).is_ok_and(|g| g.holds(path)))
        })
}

/// Refuses `actor` reading `path`. zend and the service read everything; a
/// person their groups; a bot its grants; a guest `@guest`.
pub fn need_read(tree: &Tree, actor: &Actor, path: &JournalPath) -> Result<()> {
    let may = match actor {
        Actor::Zend | Actor::System => true,
        Actor::Bot { bot } => bot_may_read(tree, bot, path),
        Actor::Guest { .. } => path.group().as_str() == GUEST,
        _ => actor.person().is_some_and(|p| may_read(tree, p, path)),
    };
    if may {
        Ok(())
    } else {
        Err(Error::not_found(format!("there is no journal {path}")))
    }
}

/// `actor` is an owner of the group `path` is in.
pub fn owns_group(tree: &Tree, actor: &Actor, path: &JournalPath) -> bool {
    actor
        .person()
        .is_some_and(|p| role_in(tree, p, path.group()) == Some(Role::Owner))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ops::fixture::{p, zen};
    use crate::testing::slug;

    #[test]
    fn each_actor_writes_as_itself_and_says_so() {
        let a = Actor::Person {
            person: slug("yen"),
            channel: Channel::Grok,
            admin: false,
        };
        assert_eq!(a.writer(), Writer::Channel(Channel::Grok));
        assert_eq!(
            a.trailers(),
            vec![
                ("Journal-Writer", "grok".to_string()),
                ("Journal-Account", "yen".to_string()),
                ("Journal-Channel", "grok".to_string()),
            ]
        );
        assert_eq!(
            Actor::Zend.trailers(),
            vec![("Journal-Writer", "zend".to_string())]
        );
        assert_eq!(
            Actor::Worker {
                person: slug("john")
            }
            .writer(),
            Writer::Worker
        );
        let guest = Actor::Guest {
            channel: Channel::Other,
        };
        assert_eq!(guest.person(), None);
        assert_eq!(
            guest.trailers(),
            vec![
                ("Journal-Writer", "other".to_string()),
                ("Journal-Channel", "other".to_string()),
            ]
        );
        let bot = Actor::Bot { bot: slug("b1") };
        assert_eq!(bot.person(), None);
        assert_eq!(bot.writer(), Writer::Npcd);
    }

    /// **A bot reads only its grants**: never the rest of its group, never
    /// another group — which a missing person must not be read as "zend,
    /// reading everything".
    #[test]
    fn a_bot_reads_only_its_grants() {
        let t = zen();
        let listed = Actor::Bot {
            bot: slug("c3f1a07e"),
        };
        let stranger = Actor::Bot { bot: slug("b9") };
        assert!(need_read(&t, &listed, &p("@zen/topics/game")).is_ok());
        assert!(need_read(&t, &listed, &p("@zen/topics/game/canon")).is_ok());
        assert!(need_read(&t, &listed, &p("@zen/topics")).is_err());
        assert!(need_read(&t, &listed, &p("@zen/people/john")).is_err());
        assert!(need_read(&t, &stranger, &p("@zen/topics/game")).is_err());
        assert_eq!(groups_of_bot(&t, &slug("c3f1a07e")), [slug("zen")]);
        assert_eq!(
            grants_of_bot(&t, &slug("c3f1a07e")),
            [p("@zen/topics/game")]
        );
        assert!(need_read(&t, &Actor::Zend, &p("@zen/people/john")).is_ok());
    }

    /// **Every member reads the whole group**; a guest reads `@guest` alone.
    #[test]
    fn members_read_their_group_and_a_guest_reads_guest() {
        let t = zen();
        assert!(may_read(&t, &slug("yen"), &p("@zen/people/john")));
        assert!(may_read(&t, &slug("thanhlong"), &p("@zen/conversations")));
        assert!(!may_read(&t, &slug("stranger"), &p("@zen")));
        assert_eq!(groups_of(&t, &slug("yen")), [slug("zen")]);
        let guest = Actor::Guest {
            channel: Channel::Other,
        };
        assert!(need_read(&t, &guest, &p("@guest/topics")).is_ok());
        assert!(need_read(&t, &guest, &p("@zen")).is_err());
        assert_eq!(guest_viewer().parse("topics").unwrap(), p("@guest/topics"));
    }

    /// **A conversation's group is where its paths are written from**; out
    /// of a conversation, the person's home group: their default, or none.
    #[test]
    fn paths_are_written_from_the_conversation_or_the_home_group() {
        let mut t = zen();
        let here = p("@zen/conversations/2026-10-31-0712-grok");
        let v = viewer(&t, &slug("yen"), Some(&here));
        assert_eq!(v.group, Some(slug("zen")));
        assert_eq!(v.parse("here").unwrap(), here);
        assert_eq!(home_group(&t, &slug("john")), Some(slug("zen")));
        let default = |t: &mut Tree, person: &str, to: Option<&str>| {
            t.persons
                .get_mut(&slug(person))
                .unwrap()
                .file
                .as_mut()
                .unwrap()
                .default_group = to.map(slug);
        };
        default(&mut t, "yen", None);
        assert_eq!(
            home_group(&t, &slug("yen")),
            None,
            "her one group, yet hers to choose: whoever left her in it chose nothing"
        );
        default(&mut t, "john", None);
        assert_eq!(
            home_group(&t, &slug("john")),
            None,
            "@zen and @guest: his to choose"
        );
        default(&mut t, "john", Some("gone"));
        assert_eq!(
            home_group(&t, &slug("john")),
            None,
            "a default that lists him not"
        );
        assert_eq!(home_group(&t, &slug("stranger")), None);
        assert_eq!(viewer(&t, &slug("stranger"), None).group, None);
    }
}
