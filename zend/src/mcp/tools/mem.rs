//! What every memory tool starts from: the caller — a member of a group or a
//! guest — the tree as last published, the channel they speak through, the
//! conversation they are in and its focus, and how their paths and addresses
//! are read: from the conversation's group, `here` the conversation itself,
//! each reference on the way followed (§2.5).

use std::sync::Arc;

use journals::actor::{
    groups_of, guest_viewer, home_group, need_read, role_in, viewer, Actor, GUEST,
};
use journals::files::{Role, Who as Member};
use journals::path::{Address, JournalPath, Viewer};
use journals::scope::Channel;
use journals::slug::Slug;
use journals::time::Stamp;
use journals::tree::Tree;
use journals::view::index::{index, refs_lines};
use journals::view::who::name_of;
use mcp_server::Client as McpClient;

use super::Ctx;
use crate::mcp::caller::Caller;
use crate::mcp::memory::who::{self, Who};
use crate::mcp::memory::{conversations, moved, Memory};
use crate::session::ZendSession;

/// How every memory tool's paths are read, as its description says it
/// (`docs/memory_backbone.md` §6.9).
pub const PATHS: &str = "A memory path starts at `@<group>`, or is bare for this conversation's \
    group: `topics/candle`, `people/yen`; `here` is this conversation's own journal. A path may pass \
    through a reference (an index lists one as `→ name (target)`), and goes on from the journal it \
    leads to.";

/// What no memory tool does, as its description says it (§2.6).
pub const APPEND_ONLY: &str =
    "The tools only add: moving, merging, renaming or splitting journals is a \
    coding task, `repo_task_start` on the `journals` checkout.";

/// Why a path or an address names no journal.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Miss {
    /// It is not written as a path is.
    Malformed(String),
    /// It is, and no journal the caller reads is there.
    Absent(String),
}

impl From<Miss> for String {
    fn from(m: Miss) -> String {
        match m {
            Miss::Malformed(why) | Miss::Absent(why) => why,
        }
    }
}

/// Anything else said wrong about a call is how it was written.
impl From<String> for Miss {
    fn from(why: String) -> Miss {
        Miss::Malformed(why)
    }
}

impl From<&str> for Miss {
    fn from(why: &str) -> Miss {
        Miss::Malformed(why.to_string())
    }
}

pub struct Mem {
    pub memory: Arc<Memory>,
    pub tree: Arc<Tree>,
    pub who: Who,
    pub channel: Channel,
    /// The estate's role table names the caller an admin: they list
    /// accounts' emails.
    pub admin: bool,
    /// The group paths are written from: the conversation's, else the
    /// person's home group; `@guest` for a guest.
    pub group: Option<Slug>,
    /// This conversation's journal, once the person has a group to hold it.
    pub here: Option<JournalPath>,
}

/// The channel an MCP session's client app speaks through.
pub fn channel_of(c: McpClient) -> Channel {
    Channel::parse(c.name()).unwrap_or(Channel::Other)
}

/// The caller through an MCP session: [`Mem::of`], in this session's
/// conversation — opened at its first call (§6.6), for a person who writes
/// in their group. A reader writes nothing there, not even a conversation
/// of their own, and reads without one. A conversation that could not be
/// opened leaves the call to read without it: what fails is the write that
/// needs it, never the read.
pub fn open(ctx: &Ctx) -> Result<Mem, String> {
    let mut m = Mem::of(&ctx.session, &ctx.caller, channel_of(ctx.mcp.client))?;
    if let (Who::Person(person), Some(group)) = (&m.who, &m.group) {
        if writes_in(&m.tree, person, group) {
            match conversations::of(&m.memory, person, group, m.channel, &ctx.mcp_id) {
                Ok(here) => {
                    m.tree = m.memory.tree()?;
                    m.here = Some(here);
                }
                Err(e) => {
                    tracing::warn!(%person, "journals: this conversation's journal was not opened: {e}")
                }
            }
        }
    }
    Ok(m)
}

/// `person` writes in `group`: a member or an owner of it, not a reader.
fn writes_in(tree: &Tree, person: &Slug, group: &Slug) -> bool {
    role_in(tree, person, group).is_some_and(|r| r != Role::Reader)
}

impl Mem {
    /// `caller` on `session`'s memory, speaking through `channel`: bound at
    /// their account's first call, and their old notes moved onto their own
    /// journal once.
    pub fn of(session: &ZendSession, caller: &Caller, channel: Channel) -> Result<Mem, String> {
        let memory = Arc::clone(session.mcp().memory());
        let who = who::who(&memory, caller)?;
        let tree = memory.tree()?;
        let group = match &who {
            Who::Person(p) => home_group(&tree, p),
            Who::Guest => Some(Slug::parse(GUEST).expect("a name")),
        };
        if let (Who::Person(person), Some(group), Some(records)) =
            (&who, &group, session.shared_substrate())
        {
            moved::once(&memory, records, &caller.key(), group, person);
        }
        Ok(Mem {
            tree,
            memory,
            who,
            channel,
            admin: caller.is_admin(),
            group,
            here: None,
        })
    }

    /// The same caller, reading `tree`: a batch's operations, each seen
    /// against the tree as the ones before it leave it.
    pub fn seeing(&self, tree: Arc<Tree>) -> Mem {
        Mem {
            memory: Arc::clone(&self.memory),
            tree,
            who: self.who.clone(),
            channel: self.channel,
            admin: self.admin,
            group: self.group.clone(),
            here: self.here.clone(),
        }
    }

    /// The caller's person, when they are one.
    pub fn person(&self) -> Option<&Slug> {
        match &self.who {
            Who::Person(p) => Some(p),
            Who::Guest => None,
        }
    }

    /// Why this caller has no conversation of their own here, for a call
    /// that needs one.
    pub fn why_no_here(&self) -> String {
        let Some(person) = self.person() else {
            return "this account is mapped to no one, so it reads @guest and writes nothing"
                .into();
        };
        let groups = groups_of(&self.tree, person);
        match (&self.group, groups.as_slice()) {
            (_, []) => "you are in no group yet: an owner adds you with group_add".into(),
            (None, groups) => format!(
                "you are in {} and have chosen none for your conversations: group_default \
                 chooses one",
                groups
                    .iter()
                    .map(|g| format!("@{g}"))
                    .collect::<Vec<_>>()
                    .join(" and ")
            ),
            (Some(g), _) if !writes_in(&self.tree, person, g) => {
                format!(
                    "you read @{g} and write nothing there, not even a conversation of your own"
                )
            }
            (Some(_), _) => {
                "this conversation's journal could not be opened; try again shortly".into()
            }
        }
    }

    /// The person, or why a guest may not do what needs one.
    pub fn need_person(&self) -> Result<&Slug, String> {
        self.person().ok_or_else(|| {
            "this account is mapped to no one, so it reads @guest and writes nothing: an \
             estate admin maps it to a person with group_add"
                .into()
        })
    }

    /// The caller's AI, as a writer.
    pub fn actor(&self) -> Actor {
        match &self.who {
            Who::Person(person) => Actor::Person {
                person: person.clone(),
                channel: self.channel,
                admin: self.admin,
            },
            Who::Guest => Actor::Guest {
                channel: self.channel,
            },
        }
    }

    fn viewer(&self) -> Viewer {
        match &self.who {
            Who::Person(p) => viewer(&self.tree, p, self.here.as_ref()),
            Who::Guest => guest_viewer(),
        }
    }

    /// `text` as a journal: parsed from the caller's group, and each
    /// reference on the way followed to the journal it leads to.
    pub fn path(&self, text: &str) -> Result<JournalPath, String> {
        self.find(text).map_err(String::from)
    }

    /// [`Mem::path`], telling a path written wrong from one that names no
    /// journal the caller reads.
    pub fn find(&self, text: &str) -> Result<JournalPath, Miss> {
        let written = self
            .viewer()
            .parse(text)
            .map_err(|e| Miss::Malformed(e.to_string()))?;
        self.resolve(&written).map_err(Miss::Absent)
    }

    /// `path#line-id`, its path resolved as [`Mem::path`] resolves one.
    pub fn address(&self, text: &str) -> Result<Address, String> {
        self.find_address(text).map_err(String::from)
    }

    /// [`Mem::address`], telling the two misses apart as [`Mem::find`] does.
    pub fn find_address(&self, text: &str) -> Result<Address, Miss> {
        let a = self
            .viewer()
            .parse_address(text)
            .map_err(|e| Miss::Malformed(e.to_string()))?;
        Ok(Address {
            path: self.resolve(&a.path).map_err(Miss::Absent)?,
            id: a.id,
        })
    }

    /// `written` resolved; or why not, with what the deepest journal it
    /// reaches holds, when the caller may read it — so a wrong turn costs
    /// one call, not a search (§6.8).
    fn resolve(&self, written: &JournalPath) -> Result<JournalPath, String> {
        // A group the caller is outside is not there for them, whatever it
        // holds: its shape is never told.
        let group = JournalPath::root(written.group().clone());
        need_read(&self.tree, &self.actor(), &group)
            .map_err(|_| format!("there is no journal {written}"))?;
        self.tree.resolve(written).map_err(|e| {
            let why = e.to_string();
            let Some(at) = self.tree.reached(written) else {
                return why;
            };
            let Some(j) = self.tree.get(&at) else {
                return why;
            };
            let mut out = vec![why, format!("{at} holds:")];
            out.extend(refs_lines(j).into_iter().map(|l| format!("  {l}")));
            out.extend(
                index(&at, j, Stamp::now())
                    .into_iter()
                    .map(|l| format!("  {l}")),
            );
            if out.len() == 2 {
                out[1] = format!("{at} holds nothing below it yet.");
            }
            out.join("\n")
        })
    }

    /// Who the conversation's focus says is speaking, when it says.
    pub fn speaker(&self) -> Option<Slug> {
        let here = self.tree.get(self.here.as_ref()?)?;
        here.meta.as_ref()?.focus.as_ref()?.speaker.clone()
    }

    /// Who reads what is under `path`: every other person in its group, and
    /// its bots that are granted it.
    pub fn readers(&self, path: &JournalPath) -> String {
        let g = path.group();
        let file = self.tree.group_file(g);
        let others: Vec<String> = file
            .map(|f| {
                f.persons()
                    .filter(|(p, _)| Some(*p) != self.person())
                    .map(|(p, _)| name_of(&self.tree, p))
                    .collect()
            })
            .unwrap_or_default();
        let bots = file.map_or(0, |f| {
            f.members
                .iter()
                .filter(|m| matches!(m.who, Member::Bot(_)))
                .filter(|m| m.reads.iter().any(|r| r.at(g).is_ok_and(|r| r.holds(path))))
                .count()
        });
        let people = match others.as_slice() {
            [] if bots == 0 => return format!("read in @{g}, which has no other member yet"),
            [] => String::new(),
            [one] => format!("read by everyone in @{g}: {one}"),
            [rest @ .., last] => {
                format!("read by everyone in @{g}: {} and {last}", rest.join(", "))
            }
        };
        let bots = match bots {
            0 => String::new(),
            1 => "a bot granted it".to_string(),
            n => format!("{n} bots granted it"),
        };
        match (people.is_empty(), bots.is_empty()) {
            (false, false) => format!("{people}, and {bots}"),
            (false, true) => people,
            _ => format!("read by {bots} only"),
        }
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use std::collections::BTreeMap;

    use journals::files::Member as GroupMember;
    use journals::path::Below;
    use journals::tree::load::load;
    use serde_json::json;

    use super::*;
    use crate::mcp::tools::group_add::group_of;

    /// Zen: John and Yen owners, Thanh Long a member; `solo` in no group; and
    /// a guest. John is the estate's admin.
    pub(crate) fn mem(person: &str) -> Mem {
        let p = |slug: &str, name: &str, groups: &str| {
            (
                format!("persons/{slug}/person.yaml"),
                format!("person: {slug}\nname: \"{name}\"\naccounts: []\n{groups}").into_bytes(),
            )
        };
        let zen = "groups: [zen]\ndefault_group: zen\n";
        let files: BTreeMap<String, Vec<u8>> = BTreeMap::from([
            p("john", "John", zen),
            p("yen", "Yen", zen),
            p("thanhlong", "Thanh Long", zen),
            p("solo", "Solo", "groups: []\n"),
            (
                "groups/zen/group.yaml".into(),
                b"group: zen\nname: \"Zen\"\nmembers:\n  - {person: john, role: owner, joined: 2026-10-30T00:00:00Z}\n  - {person: yen, role: owner, joined: 2026-10-30T00:00:00Z}\n  - {person: thanhlong, role: member, joined: 2026-10-30T00:00:00Z}\n".to_vec(),
            ),
        ]);
        let tree = Arc::new(load(&files));
        let who = match person {
            "guest" => Who::Guest,
            p => Who::Person(Slug::parse(p).unwrap()),
        };
        let group = match &who {
            Who::Person(p) => home_group(&tree, p),
            Who::Guest => Some(Slug::parse(GUEST).unwrap()),
        };
        Mem {
            memory: Arc::default(),
            tree,
            who,
            channel: Channel::Grok,
            admin: person == "john",
            group,
            here: None,
        }
    }

    #[test]
    fn a_write_says_who_reads_it() {
        let m = mem("john");
        assert_eq!(
            m.readers(&m.path("topics").unwrap()),
            "read by everyone in @zen: Yen and Thanh Long"
        );
        let y = mem("yen");
        assert_eq!(
            y.readers(&y.path("@zen/people").unwrap()),
            "read by everyone in @zen: John and Thanh Long"
        );
    }

    #[test]
    fn paths_and_addresses_are_read_for_the_caller() {
        let m = mem("john");
        assert_eq!(m.path("topics").unwrap().show(), "@zen/topics");
        assert_eq!(
            m.address("@zen#2026-10-30-0912-7e3a").unwrap().to_string(),
            "@zen#2026-10-30-0912-7e3a"
        );
        assert!(m.address("@zen").is_err());
        assert!(m.path("~").unwrap_err().contains("people/<name>"));
        assert_eq!(
            mem("solo").path("topics").unwrap_err(),
            "`topics` names no group, and you are in none: write `@group/topics`"
        );
        assert_eq!(
            m.actor(),
            Actor::Person {
                person: Slug::parse("john").unwrap(),
                channel: Channel::Grok,
                admin: true,
            }
        );
        assert_eq!(channel_of(McpClient::ClaudeCode), Channel::ClaudeCode);
        let g = mem("guest");
        assert_eq!(
            g.actor(),
            Actor::Guest {
                channel: Channel::Grok
            }
        );
        assert!(g.need_person().is_err());
    }

    /// **Only someone who writes in the group is given a conversation**: a
    /// reader's calls read, and write nothing, not even one.
    #[test]
    fn a_reader_reads_without_a_conversation_of_their_own() {
        let m = mem("john");
        let zen = Slug::parse("zen").unwrap();
        let p = |s: &str| Slug::parse(s).unwrap();
        assert!(writes_in(&m.tree, &p("john"), &zen), "an owner");
        assert!(writes_in(&m.tree, &p("thanhlong"), &zen), "a member");
        let mut tree = (*m.tree).clone();
        let g = tree.groups.get_mut(&zen).unwrap();
        for member in &mut g.file.as_mut().unwrap().members {
            if member.who == Member::Person(p("thanhlong")) {
                member.role = Role::Reader;
            }
        }
        assert!(!writes_in(&tree, &p("thanhlong"), &zen), "a reader");
        assert!(!writes_in(&tree, &p("solo"), &zen), "no one there");
    }

    /// **A call that needs a conversation says why there is none**, in
    /// terms of what the caller can do about it.
    #[test]
    fn a_missing_conversation_says_what_would_give_one() {
        assert_eq!(
            mem("solo").why_no_here(),
            "you are in no group yet: an owner adds you with group_add"
        );
        assert!(mem("guest")
            .why_no_here()
            .starts_with("this account is mapped to no one"));
        let mut yen = mem("yen");
        yen.group = None;
        assert_eq!(
            yen.why_no_here(),
            "you are in @zen and have chosen none for your conversations: group_default \
             chooses one"
        );
        let mut reader = mem("thanhlong");
        let mut tree = (*reader.tree).clone();
        let zen = Slug::parse("zen").unwrap();
        for member in &mut tree
            .groups
            .get_mut(&zen)
            .unwrap()
            .file
            .as_mut()
            .unwrap()
            .members
        {
            if member.who == Member::Person(Slug::parse("thanhlong").unwrap()) {
                member.role = Role::Reader;
            }
        }
        reader.tree = Arc::new(tree);
        assert_eq!(
            reader.why_no_here(),
            "you read @zen and write nothing there, not even a conversation of your own"
        );
    }

    /// **A wrong turn is answered with what is there**: the deepest journal
    /// the path reached, and what it holds — never for a reader outside it.
    #[test]
    fn a_path_that_misses_shows_where_it_stopped() {
        let m = mem("john");
        assert_eq!(
            m.path("topics/house/permits").unwrap_err(),
            "@zen/topics has no sub-journal or reference `house`\n\
             @zen/topics holds nothing below it yet."
        );
        assert_eq!(
            mem("guest").path("@zen/topics/house").unwrap_err(),
            "there is no journal @zen/topics/house",
            "a guest is not shown what a group it is outside holds"
        );
        assert_eq!(
            mem("guest").path("@zen/topics").unwrap_err(),
            "there is no journal @zen/topics",
            "nor told that a journal there exists"
        );
    }

    #[test]
    fn a_group_is_named_or_the_callers_own() {
        let m = mem("john");
        assert_eq!(group_of(&m, &json!({})).unwrap().as_str(), "zen");
        assert_eq!(
            group_of(&m, &json!({"group": "@work"})).unwrap().as_str(),
            "work"
        );
        assert_eq!(
            group_of(&mem("solo"), &json!({})).unwrap_err(),
            "you have no default group: name the `group`"
        );
    }

    /// **A write says which bots read it**: only those granted that subtree.
    #[test]
    fn a_groups_readers_include_the_bots_granted_it() {
        let mut m = mem("john");
        let mut tree = (*m.tree).clone();
        let g = tree.groups.get_mut(&Slug::parse("zen").unwrap()).unwrap();
        g.file.as_mut().unwrap().members.push(GroupMember {
            who: Member::Bot(Slug::parse("c3f1a07e").unwrap()),
            role: Role::Reader,
            joined: Stamp::parse("2026-10-30T00:00:00Z").unwrap(),
            reads: vec![Below::parse("topics").unwrap()],
        });
        m.tree = Arc::new(tree);
        assert_eq!(
            m.readers(&m.path("topics").unwrap()),
            "read by everyone in @zen: Yen and Thanh Long, and a bot granted it"
        );
        assert_eq!(
            m.readers(&m.path("people").unwrap()),
            "read by everyone in @zen: Yen and Thanh Long"
        );
    }
}
