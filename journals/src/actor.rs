//! Who is writing (§7): a person's AI through a client, the connector's
//! server, zend, a Zen worker, npcd — each with what it may read and write
//! (§7.2). The writer a line records and the trailers a commit carries come
//! from here, never from what a caller says of itself.

use crate::error::{Error, Result};
use crate::files::Role;
use crate::line::Writer;
use crate::path::{JournalPath, Owner, Viewer};
use crate::scope::Client;
use crate::slug::Slug;
use crate::tree::Tree;

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Actor {
    /// A person's AI, through `client`.
    Person { person: Slug, client: Client },
    /// The connector's server, recording what a call of `person`'s did.
    Server { person: Slug, client: Client },
    /// zend's background thread: summaries, folds, workers' exports.
    Zend,
    /// A Zen worker of `person`'s, on its own account.
    Worker { person: Slug },
    /// npcd, for a bot of a family `person` belongs to.
    Npcd { person: Slug },
    /// The service itself: binding an account at its first sign-in.
    System,
}

impl Actor {
    pub fn writer(&self) -> Writer {
        match self {
            Actor::Person { client, .. } => Writer::Client(*client),
            Actor::Server { .. } => Writer::Server,
            Actor::Zend | Actor::System => Writer::Zend,
            Actor::Worker { .. } => Writer::Worker,
            Actor::Npcd { .. } => Writer::Npcd,
        }
    }

    pub fn person(&self) -> Option<&Slug> {
        match self {
            Actor::Person { person, .. }
            | Actor::Server { person, .. }
            | Actor::Worker { person }
            | Actor::Npcd { person } => Some(person),
            Actor::Zend | Actor::System => None,
        }
    }

    pub fn client(&self) -> Option<Client> {
        match self {
            Actor::Person { client, .. } | Actor::Server { client, .. } => Some(*client),
            _ => None,
        }
    }

    /// Commit trailers naming who wrote (§7.1).
    pub fn trailers(&self) -> Vec<(&'static str, String)> {
        let mut out = vec![("Journal-Writer", self.writer().name().to_string())];
        if let Some(p) = self.person() {
            out.push(("Journal-Person", p.to_string()));
        }
        if let Some(c) = self.client() {
            out.push(("Journal-Client", c.name().to_string()));
        }
        out
    }
}

/// `person` is a member of `family`, by its `family.yaml`; their role.
pub fn role_in(tree: &Tree, person: &Slug, family: &Slug) -> Option<Role> {
    tree.family_file(family)
        .and_then(|f| f.member(person))
        .map(|m| m.role)
}

/// The families `person` reads: those whose `family.yaml` lists them.
pub fn families_of(tree: &Tree, person: &Slug) -> Vec<Slug> {
    tree.families
        .keys()
        .filter(|f| role_in(tree, person, f).is_some())
        .cloned()
        .collect()
}

/// The viewer paths are written relative to, for `person`.
pub fn viewer(tree: &Tree, person: &Slug) -> Viewer {
    Viewer {
        person: person.clone(),
        families: families_of(tree, person),
    }
}

/// `person` may read what is under `owner`: their own folder, or a family's
/// that lists them.
pub fn may_read(tree: &Tree, person: &Slug, owner: &Owner) -> bool {
    match owner {
        Owner::Person(p) => p == person,
        Owner::Family(f) => role_in(tree, person, f).is_some(),
    }
}

/// Refuses `actor` reading under `path`.
pub fn need_read(tree: &Tree, actor: &Actor, path: &JournalPath) -> Result<()> {
    match actor.person() {
        None => Ok(()),
        Some(p) if may_read(tree, p, path.owner()) => Ok(()),
        Some(_) => Err(Error::not_found(format!("there is no journal {path}"))),
    }
}

/// `actor` is an owner of the family `path` is in.
pub fn owns_family(tree: &Tree, actor: &Actor, path: &JournalPath) -> bool {
    match (actor.person(), path.owner()) {
        (Some(p), Owner::Family(f)) => role_in(tree, p, f) == Some(Role::Owner),
        _ => false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testing::slug;

    #[test]
    fn each_actor_writes_as_itself_and_says_so() {
        let a = Actor::Person {
            person: slug("yen"),
            client: Client::Grok,
        };
        assert_eq!(a.writer(), Writer::Client(Client::Grok));
        assert_eq!(
            a.trailers(),
            vec![
                ("Journal-Writer", "grok".to_string()),
                ("Journal-Person", "yen".to_string()),
                ("Journal-Client", "grok".to_string()),
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
                client: Client::Other
            }
            .writer(),
            Writer::Server
        );
        assert_eq!(
            Actor::Npcd {
                person: slug("john")
            }
            .client(),
            None
        );
    }
}
