//! What every memory tool starts from: the caller as a person on the tree,
//! the tree as last published, the client they speak through, and how their
//! paths and addresses are read.

use std::sync::Arc;

use journals::actor::{families_of, viewer, Actor};
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
/// this account is in no family — rather than passing: the repository still
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
        })
    }

    /// The caller's AI, as a writer.
    pub fn actor(&self) -> Actor {
        Actor::Person {
            person: self.person.clone(),
            client: self.client,
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

    /// Who reads what is under `path`: the caller alone, or their family.
    pub fn readers(&self, path: &JournalPath) -> String {
        match path.owner() {
            Owner::Person(_) => "only you read it".into(),
            Owner::Family(f) => {
                let others: Vec<String> = self
                    .tree
                    .family_file(f)
                    .map(|file| {
                        file.members
                            .iter()
                            .filter(|m| m.person != self.person)
                            .map(|m| name_of(&self.tree, &m.person))
                            .collect()
                    })
                    .unwrap_or_default();
                match others.as_slice() {
                    [] => format!("shared with @{f}, which has no other member yet"),
                    [one] => format!("shared with {one}"),
                    [rest @ .., last] => format!("shared with {} and {last}", rest.join(", ")),
                }
            }
        }
    }

    /// The families the caller belongs to.
    pub fn families(&self) -> Vec<Slug> {
        families_of(&self.tree, &self.person)
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use std::collections::BTreeMap;

    use journals::tree::load::load;
    use serde_json::json;

    use super::*;
    use crate::mcp::tools::family_add::family_of;

    /// The Sharratts: John and Yen owners, Thanh Long a member; `solo` in no
    /// family.
    pub(crate) fn mem(person: &str) -> Mem {
        let p = |slug: &str, name: &str, fams: &str| {
            (
                format!("persons/{slug}/person.yaml"),
                format!("person: {slug}\nname: \"{name}\"\naccounts: []\n{fams}").into_bytes(),
            )
        };
        let fam = "families: [sharratt]\ndefault_family: sharratt\n";
        let files: BTreeMap<String, Vec<u8>> = BTreeMap::from([
            p("john", "John", fam),
            p("yen", "Yen", fam),
            p("thanhlong", "Thanh Long", fam),
            p("solo", "Solo", "families: []\n"),
            (
                "families/sharratt/family.yaml".into(),
                b"family: sharratt\nname: \"Sharratt\"\nmembers:\n  - {person: john, role: owner, joined: 2026-10-30T00:00:00Z}\n  - {person: yen, role: owner, joined: 2026-10-30T00:00:00Z}\n  - {person: thanhlong, role: member, joined: 2026-10-30T00:00:00Z}\n".to_vec(),
            ),
        ]);
        Mem {
            memory: Arc::default(),
            tree: Arc::new(load(&files)),
            person: Slug::parse(person).unwrap(),
            client: Client::Grok,
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
        assert!(solo
            .path("topics")
            .unwrap_err()
            .contains("belong to no family"));
        assert_eq!(
            m.actor(),
            Actor::Person {
                person: Slug::parse("john").unwrap(),
                client: Client::Grok
            }
        );
        assert_eq!(client_of(McpClient::ClaudeCode), Client::ClaudeCode);
    }

    /// **Only a caller in no family is a lasting answer**; the memory still
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
    fn a_family_is_named_or_the_callers_only_one() {
        let m = mem("john");
        assert_eq!(m.families(), [Slug::parse("sharratt").unwrap()]);
        assert_eq!(family_of(&m, &json!({})).unwrap().as_str(), "sharratt");
        assert_eq!(
            family_of(&m, &json!({"family": "@work"})).unwrap().as_str(),
            "work"
        );
        assert_eq!(
            family_of(&mem("solo"), &json!({})).unwrap_err(),
            "you belong to no family"
        );
    }
}
