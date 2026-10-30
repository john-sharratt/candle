//! What every memory tool starts from: the caller as a person on the tree,
//! the tree as last published, the client they speak through, and how their
//! paths and addresses are read.

use std::sync::Arc;

use journals::actor::{groups_of, viewer, Actor};
use journals::files::Who;
use journals::path::{Address, JournalPath, Owner};
use journals::scope::Client;
use journals::slug::Slug;
use journals::tree::Tree;
use journals::view::who::name_of;
use mcp_server::Client as McpClient;

use super::Ctx;
use crate::mcp::caller::Caller;
use crate::mcp::memory::{conversations, moved, who, Memory};
use crate::session::ZendSession;

pub struct Mem {
    pub memory: Arc<Memory>,
    pub tree: Arc<Tree>,
    pub person: Slug,
    pub client: Client,
    /// The estate's role table names the caller an admin: they list
    /// accounts' emails.
    pub admin: bool,
}

/// The journal client an MCP session's client is.
pub fn client_of(c: McpClient) -> Client {
    Client::parse(c.name()).unwrap_or(Client::Other)
}

/// The caller as a person, through an MCP session: [`Mem::of`], and their
/// conversation's clock moved on.
pub fn open(ctx: &Ctx) -> Result<Mem, String> {
    let m = Mem::of(&ctx.session, &ctx.caller, client_of(ctx.mcp.client))?;
    conversations::touch(&m.memory, &ctx.mcp_id);
    Ok(m)
}

/// Whether what [`open`] answered stays true however often it is asked —
/// this account is in no group — rather than passing: the repository still
/// being read, this member holding no journal checkout while another may.
/// A lasting answer is given as an answer; a passing one as an error, which
/// the aggregator asks again.
pub fn lasting(why: &str) -> bool {
    why == who::STRANGER
}

impl Mem {
    /// `caller` as a person on `session`'s memory, speaking through `client`:
    /// bound at their account's first call, and their old notes moved onto
    /// the tree once.
    pub fn of(session: &ZendSession, caller: &Caller, client: Client) -> Result<Mem, String> {
        let memory = Arc::clone(session.mcp().memory());
        let person = who::person(&memory, caller)?;
        if let Some(records) = session.shared_substrate() {
            moved::once(&memory, records, &caller.key(), &person);
        }
        Ok(Mem {
            tree: memory.tree()?,
            memory,
            person,
            client,
            admin: caller.is_admin(),
        })
    }

    /// The same caller, reading `tree`: a batch's operations, each seen
    /// against the tree as the ones before it leave it.
    pub fn seeing(&self, tree: Arc<Tree>) -> Mem {
        Mem {
            memory: Arc::clone(&self.memory),
            tree,
            person: self.person.clone(),
            client: self.client,
            admin: self.admin,
        }
    }

    /// The caller's AI, as a writer.
    pub fn actor(&self) -> Actor {
        Actor::Person {
            person: self.person.clone(),
            client: self.client,
            admin: self.admin,
        }
    }

    pub fn path(&self, text: &str) -> Result<JournalPath, String> {
        viewer(&self.tree, &self.person)
            .parse(text)
            .map_err(|e| e.to_string())
    }

    /// `path#line-id`.
    pub fn address(&self, text: &str) -> Result<Address, String> {
        viewer(&self.tree, &self.person)
            .parse_address(text)
            .map_err(|e| e.to_string())
    }

    /// Who reads what is under `path`: the caller alone, or their group's
    /// other people and its bots.
    pub fn readers(&self, path: &JournalPath) -> String {
        match path.owner() {
            Owner::Person(_) => "only you read it".into(),
            Owner::Group(g) => {
                let file = self.tree.group_file(g);
                let others: Vec<String> = file
                    .map(|f| {
                        f.persons()
                            .filter(|(p, _)| *p != &self.person)
                            .map(|(p, _)| name_of(&self.tree, p))
                            .collect()
                    })
                    .unwrap_or_default();
                let bots = file.map_or(0, |f| {
                    f.members
                        .iter()
                        .filter(|m| matches!(m.who, Who::Bot(_)))
                        .count()
                });
                let people = match others.as_slice() {
                    [] if bots == 0 => {
                        return format!("shared with @{g}, which has no other member yet")
                    }
                    [] => String::new(),
                    [one] => format!("shared with {one}"),
                    [rest @ .., last] => format!("shared with {} and {last}", rest.join(", ")),
                };
                let bots = match bots {
                    0 => String::new(),
                    1 => "read by its bot".to_string(),
                    n => format!("read by its {n} bots"),
                };
                match (people.is_empty(), bots.is_empty()) {
                    (false, false) => format!("{people}, and {bots}"),
                    (false, true) => people,
                    _ => format!("{bots} only"),
                }
            }
        }
    }

    /// The groups the caller belongs to.
    pub fn groups(&self) -> Vec<Slug> {
        groups_of(&self.tree, &self.person)
    }

    /// The caller's default group: where a bare shared section is.
    pub fn default_group(&self) -> Option<Slug> {
        viewer(&self.tree, &self.person).default_group
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use std::collections::BTreeMap;

    use journals::files::{Member, Role};
    use journals::time::Stamp;
    use journals::tree::load::load;
    use serde_json::json;

    use super::*;
    use crate::mcp::tools::group_add::group_of;

    /// The Sharratts: John and Yen owners, Thanh Long a member; `solo` in no
    /// group. John is the estate's admin.
    pub(crate) fn mem(person: &str) -> Mem {
        let p = |slug: &str, name: &str, groups: &str| {
            (
                format!("persons/{slug}/person.yaml"),
                format!("person: {slug}\nname: \"{name}\"\naccounts: []\n{groups}").into_bytes(),
            )
        };
        let sharratt = "groups: [sharratt]\ndefault_group: sharratt\n";
        let files: BTreeMap<String, Vec<u8>> = BTreeMap::from([
            p("john", "John", sharratt),
            p("yen", "Yen", sharratt),
            p("thanhlong", "Thanh Long", sharratt),
            p("solo", "Solo", "groups: []\n"),
            (
                "groups/sharratt/group.yaml".into(),
                b"group: sharratt\nname: \"Sharratt\"\nkind: family\nmembers:\n  - {person: john, role: owner, joined: 2026-10-30T00:00:00Z}\n  - {person: yen, role: owner, joined: 2026-10-30T00:00:00Z}\n  - {person: thanhlong, role: member, joined: 2026-10-30T00:00:00Z}\n".to_vec(),
            ),
        ]);
        Mem {
            memory: Arc::default(),
            tree: Arc::new(load(&files)),
            person: Slug::parse(person).unwrap(),
            client: Client::Grok,
            admin: person == "john",
        }
    }

    #[test]
    fn a_write_says_who_reads_it() {
        let m = mem("john");
        assert_eq!(m.readers(&m.path("~").unwrap()), "only you read it");
        assert_eq!(
            m.readers(&m.path("topics").unwrap()),
            "shared with Yen and Thanh Long"
        );
        let y = mem("yen");
        assert_eq!(
            y.readers(&y.path("@sharratt/projects").unwrap()),
            "shared with John and Thanh Long"
        );
    }

    #[test]
    fn paths_and_addresses_are_read_for_the_caller() {
        let m = mem("john");
        assert_eq!(
            m.path("projects/candle").unwrap().show(),
            "@sharratt/projects/candle"
        );
        assert_eq!(
            m.address("~#20261030-0912-7e3a").unwrap().to_string(),
            "~#20261030-0912-7e3a"
        );
        assert!(m.address("~").is_err());
        let solo = mem("solo");
        assert_eq!(
            solo.path("topics").unwrap_err(),
            "`topics` is shared, and you have no default group: name one, as `@group/topics`"
        );
        assert_eq!(
            m.actor(),
            Actor::Person {
                person: Slug::parse("john").unwrap(),
                client: Client::Grok,
                admin: true,
            }
        );
        assert_eq!(client_of(McpClient::ClaudeCode), Client::ClaudeCode);
    }

    /// **Only a caller in no group is a lasting answer**; the memory still
    /// opening, or held on another member, is asked again.
    #[test]
    fn only_a_stranger_is_answered_as_lasting() {
        use crate::mcp::memory::NO_MEMORY;
        assert!(lasting(who::STRANGER));
        assert!(!lasting(NO_MEMORY));
        assert!(!lasting(
            "the journal repository is still being read; try again shortly"
        ));
    }

    #[test]
    fn a_group_is_named_or_the_callers_default() {
        let m = mem("john");
        assert_eq!(m.groups(), [Slug::parse("sharratt").unwrap()]);
        assert_eq!(group_of(&m, &json!({})).unwrap().as_str(), "sharratt");
        assert_eq!(
            group_of(&m, &json!({"group": "@work"})).unwrap().as_str(),
            "work"
        );
        assert_eq!(
            group_of(&mem("solo"), &json!({})).unwrap_err(),
            "you have no default group: name the `group`"
        );
    }

    /// **A write into a group says who else reads it**, its bots included.
    #[test]
    fn a_groups_readers_include_its_bots() {
        let mut m = mem("john");
        let mut tree = (*m.tree).clone();
        let g = tree
            .groups
            .get_mut(&Slug::parse("sharratt").unwrap())
            .unwrap();
        let file = g.file.as_mut().unwrap();
        file.members.push(Member {
            who: Who::Bot(Slug::parse("c3f1a07e").unwrap()),
            role: Role::Reader,
            joined: Stamp::parse("2026-10-30T00:00:00Z").unwrap(),
        });
        m.tree = Arc::new(tree);
        assert_eq!(
            m.readers(&m.path("topics").unwrap()),
            "shared with Yen and Thanh Long, and read by its bot"
        );
    }
}
