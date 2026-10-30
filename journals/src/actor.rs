//! Who is writing (§7): a person's AI through a channel, the connector's
//! server, zend, a Zen worker, npcd, a bot — each with what it may read and
//! write (§7.2). The writer a line records and the trailers a commit carries
//! come from here, never from what a caller says of itself.

use crate::error::{Error, Result};
use crate::files::Role;
use crate::line::Writer;
use crate::path::{JournalPath, Owner, Viewer};
use crate::scope::Channel;
use crate::slug::Slug;
use crate::tree::Tree;

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Actor {
    /// A person's AI, through `channel`. `admin` is the estate's role table
    /// naming them an admin, as the gateway's verified identity says: only
    /// an admin lists an account's email (§3.5).
    Person {
        person: Slug,
        channel: Channel,
        admin: bool,
    },
    /// The connector's server, recording what a call of `person`'s did.
    Server { person: Slug, channel: Channel },
    /// zend's background thread: summaries, folds, workers' exports.
    Zend,
    /// A Zen worker of `person`'s, on its own account.
    Worker { person: Slug },
    /// npcd, for a bot of a group `person` belongs to.
    Npcd { person: Slug },
    /// A bot, reading the groups that list it as a reader. It writes nothing.
    Bot { bot: Slug },
    /// The service itself: binding an account at its first sign-in.
    System,
}

impl Actor {
    /// The writer its lines record. A bot writes none, and is named for the
    /// daemon that runs it should a refusal be logged.
    pub fn writer(&self) -> Writer {
        match self {
            Actor::Person { channel, .. } => Writer::Channel(*channel),
            Actor::Server { .. } => Writer::Server,
            Actor::Zend | Actor::System => Writer::Zend,
            Actor::Worker { .. } => Writer::Worker,
            Actor::Npcd { .. } | Actor::Bot { .. } => Writer::Npcd,
        }
    }

    pub fn person(&self) -> Option<&Slug> {
        match self {
            Actor::Person { person, .. }
            | Actor::Server { person, .. }
            | Actor::Worker { person }
            | Actor::Npcd { person } => Some(person),
            Actor::Zend | Actor::System | Actor::Bot { .. } => None,
        }
    }

    pub fn channel(&self) -> Option<Channel> {
        match self {
            Actor::Person { channel, .. } | Actor::Server { channel, .. } => Some(*channel),
            _ => None,
        }
    }

    /// Commit trailers naming who wrote (§7.1).
    pub fn trailers(&self) -> Vec<(&'static str, String)> {
        let mut out = vec![("Journal-Writer", self.writer().name().to_string())];
        if let Some(p) = self.person() {
            out.push(("Journal-Person", p.to_string()));
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

/// The groups `bot` reads: those whose `group.yaml` lists it.
pub fn groups_of_bot(tree: &Tree, bot: &Slug) -> Vec<Slug> {
    tree.groups
        .iter()
        .filter(|(_, g)| g.file.as_ref().is_ok_and(|f| f.bot(bot).is_some()))
        .map(|(slug, _)| slug.clone())
        .collect()
}

/// The viewer paths are written relative to, for `person`: their default
/// group when it is one that lists them. Without one, none is chosen for
/// them — whichever group stood in would be the pick of whoever left them in
/// it — so they name a group, or choose a default.
pub fn viewer(tree: &Tree, person: &Slug) -> Viewer {
    let default_group = tree
        .person_file(person)
        .and_then(|f| f.default_group.clone())
        .filter(|g| role_in(tree, person, g).is_some());
    Viewer {
        person: person.clone(),
        default_group,
    }
}

/// The groups that open every conversation of `person`'s: their default as
/// [`viewer`] resolves it, then the others they frame — each one that lists
/// them.
pub fn framed_groups(tree: &Tree, person: &Slug) -> Vec<Slug> {
    let default = viewer(tree, person).default_group;
    let mut out: Vec<Slug> = default.iter().cloned().collect();
    if let Some(f) = tree.person_file(person) {
        for g in &f.framed {
            if !out.contains(g) && role_in(tree, person, g).is_some() {
                out.push(g.clone());
            }
        }
    }
    out
}

/// `person` may read what is under `owner`: their own folder, or a group's
/// that lists them.
pub fn may_read(tree: &Tree, person: &Slug, owner: &Owner) -> bool {
    match owner {
        Owner::Person(p) => p == person,
        Owner::Group(g) => role_in(tree, person, g).is_some(),
    }
}

/// `bot` may read what is under `owner`: a group's that lists it, never a
/// person's folder.
pub fn bot_may_read(tree: &Tree, bot: &Slug, owner: &Owner) -> bool {
    match owner {
        Owner::Person(_) => false,
        Owner::Group(g) => tree.group_file(g).is_some_and(|f| f.bot(bot).is_some()),
    }
}

/// Refuses `actor` reading under `path`. zend and the service read
/// everything; a person their own folder and their groups'; a bot its
/// groups'.
pub fn need_read(tree: &Tree, actor: &Actor, path: &JournalPath) -> Result<()> {
    let may = match actor {
        Actor::Zend | Actor::System => true,
        Actor::Bot { bot } => bot_may_read(tree, bot, path.owner()),
        _ => actor
            .person()
            .is_some_and(|p| may_read(tree, p, path.owner())),
    };
    if may {
        Ok(())
    } else {
        Err(Error::not_found(format!("there is no journal {path}")))
    }
}

/// `actor` is an owner of the group `path` is in.
pub fn owns_group(tree: &Tree, actor: &Actor, path: &JournalPath) -> bool {
    match (actor.person(), path.owner()) {
        (Some(p), Owner::Group(g)) => role_in(tree, p, g) == Some(Role::Owner),
        _ => false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ops::fixture::{p, sharratt};
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
                ("Journal-Person", "yen".to_string()),
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
        assert_eq!(
            Actor::Server {
                person: slug("john"),
                channel: Channel::Other
            }
            .writer(),
            Writer::Server
        );
        assert_eq!(
            Actor::Npcd {
                person: slug("john")
            }
            .channel(),
            None
        );
        let bot = Actor::Bot { bot: slug("b1") };
        assert_eq!(bot.person(), None);
        assert_eq!(bot.writer(), Writer::Npcd);
    }

    /// **A bot reads the groups that list it and nothing else**: never a
    /// person's folder, never a group it is not in — which a missing person
    /// must not be read as "zend, reading everything".
    #[test]
    fn a_bot_reads_only_its_groups() {
        let t = sharratt();
        let listed = Actor::Bot {
            bot: slug("c3f1a07e"),
        };
        let stranger = Actor::Bot { bot: slug("b9") };
        assert!(need_read(&t, &listed, &p("@sharratt/topics")).is_ok());
        assert!(need_read(&t, &listed, &p("~")).is_err());
        assert!(need_read(&t, &stranger, &p("@sharratt/topics")).is_err());
        assert_eq!(groups_of_bot(&t, &slug("c3f1a07e")), [slug("sharratt")]);
        assert!(groups_of_bot(&t, &slug("b9")).is_empty());
        assert!(need_read(&t, &Actor::Zend, &p("~")).is_ok());
    }

    #[test]
    fn a_person_reads_their_folder_and_their_groups() {
        let t = sharratt();
        assert!(may_read(&t, &slug("yen"), p("@sharratt").owner()));
        assert!(!may_read(&t, &slug("yen"), p("~").owner()), "~ is john's");
        assert_eq!(groups_of(&t, &slug("yen")), [slug("sharratt")]);
        assert_eq!(
            viewer(&t, &slug("john")).default_group,
            Some(slug("sharratt"))
        );
    }

    /// **A default no longer listing them is no default**, and no other
    /// group stands in for it.
    #[test]
    fn a_default_that_does_not_list_them_is_none() {
        let mut t = sharratt();
        t.persons
            .get_mut(&slug("john"))
            .unwrap()
            .file
            .as_mut()
            .unwrap()
            .default_group = Some(slug("gone"));
        assert_eq!(viewer(&t, &slug("john")).default_group, None);
        assert_eq!(groups_of(&t, &slug("john")), [slug("sharratt")]);
        assert_eq!(viewer(&t, &slug("stranger")).default_group, None);
    }
}
