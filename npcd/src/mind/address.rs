//! What the console and the API talk about, instead of files.
//!
//! The mind is a corpus, and a corpus has topics and entries. It is *stored* as
//! a directory of `.md` and `.yaml`, but that is an implementation of it — and
//! an API that says `layers/world/ammo/bolt.md` has published the
//! implementation as its contract. Every one of those tokens is a promise:
//! that canon lives under `layers/`, that a world's tags are a directory, that
//! prose is markdown. Change any of them later and every client breaks.
//!
//! So the wire says `canon/ammo/bolt`, and this module is the only place that
//! knows what that is on disk.
//!
//! # The shape
//!
//! An [`Address`] is a [`Section`] and a chain of names. A section is a fixed,
//! named part of the corpus — there are nine, they are listed below, and a
//! client cannot invent a tenth. A name is one step down: a topic, then an
//! entry.
//!
//! ```text
//!   canon                     the world knowledge
//!   canon/ammo                one topic of it
//!   canon/ammo/bolt           one entry in that topic
//!   responses/blush_then_own  one response section
//!   characters/cindy-tan      one character
//!   settings/projection       the projection schema
//! ```
//!
//! # What the section owns, so the address does not
//!
//! **The file extension is a property of the section, never of the address.**
//! Responses are YAML because the response library is structured; canon is
//! markdown because canon is prose. A caller that had to type `.md` would be
//! deciding a storage question, and would be wrong the day a section changes
//! format. So `canon/ammo/bolt` names the entry and this module supplies `.md`.
//!
//! The same goes for the directory. `canon` is `layers/world` today; if the
//! corpus is ever reorganised, that is one line here and no change anywhere
//! else.
//!
//! # A collection can have a body
//!
//! `layers/world/ammo.md` is the overview of the `ammo` topic, and
//! `layers/world/ammo/` holds its entries — one idea stored as two things.
//! Rather than leak that, a collection simply *may have text of its own*, and
//! `canon/ammo` addresses both: listing it gives the entries, reading it gives
//! the overview. Nothing on the wire suggests there are two files, because to
//! a reader there are not.

use super::path::{MindPath, PathError};

/// How a section stores its entries.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Format {
    /// Prose. Canon, memories, beliefs.
    Markdown,
    /// A structured document. Sections, characters, worlds, settings.
    Yaml,
}

impl Format {
    fn ext(self) -> &'static str {
        match self {
            Format::Markdown => "md",
            Format::Yaml => "yaml",
        }
    }
}

/// A named part of the corpus. Fixed: a client cannot address anything else.
///
/// This is the whole of what the mind directory is allowed to expose. A folder
/// that is not one of these — `node_modules`, the daemon's own `.substrate`, a
/// scratch directory somebody left — is not addressable, so it cannot be
/// listed, opened or written by mistake. The old path-shaped API had to filter
/// those out by name; this one cannot name them in the first place.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Section {
    Canon,
    Agency,
    Beliefs,
    Memory,
    Responses,
    Moods,
    Characters,
    Worlds,
    Settings,
}

/// Every section, in the order the console shows them: the world first, then
/// what characters are made of, then the settings that bind them.
pub const SECTIONS: [Section; 9] = [
    Section::Canon,
    Section::Agency,
    Section::Beliefs,
    Section::Memory,
    Section::Responses,
    Section::Moods,
    Section::Characters,
    Section::Worlds,
    Section::Settings,
];

impl Section {
    /// The token in an address.
    pub fn slug(self) -> &'static str {
        match self {
            Section::Canon => "canon",
            Section::Agency => "agency",
            Section::Beliefs => "beliefs",
            Section::Memory => "memory",
            Section::Responses => "responses",
            Section::Moods => "moods",
            Section::Characters => "characters",
            Section::Worlds => "worlds",
            Section::Settings => "settings",
        }
    }

    /// What a reader calls it.
    pub fn title(self) -> &'static str {
        match self {
            Section::Canon => "World knowledge",
            Section::Agency => "Agency",
            Section::Beliefs => "Beliefs",
            Section::Memory => "Memory",
            Section::Responses => "Responses",
            Section::Moods => "Moods",
            Section::Characters => "Characters",
            Section::Worlds => "Worlds",
            Section::Settings => "Settings",
        }
    }

    /// One line saying what lives here, for the console's section list.
    pub fn blurb(self) -> &'static str {
        match self {
            // Named by what is in it. "The setting itself" is true and told a
            // reader looking for the technology tree nothing — this is the
            // largest section in the mind and the one most often come for, so
            // its blurb is the sign that says so.
            Section::Canon => {
                "History, technology, factions, geography, combat — the game's own knowledge, \
                 and the largest part of the mind."
            }
            Section::Agency => "What characters want, and how they go about it.",
            Section::Beliefs => "What each character holds to be true.",
            Section::Memory => "What each character remembers having lived.",
            Section::Responses => "The structural shapes a reply can take.",
            Section::Moods => "The registers a reply can be spoken in.",
            Section::Characters => "Who a character is before they have lived anything.",
            Section::Worlds => "The settings, and the filters that scope each one.",
            Section::Settings => "How the mind itself is configured.",
        }
    }

    pub fn from_slug(slug: &str) -> Option<Section> {
        SECTIONS.into_iter().find(|s| s.slug() == slug)
    }

    /// Where it lives, relative to the mind root. `None` for the root itself.
    fn dir(self) -> Option<&'static str> {
        match self {
            Section::Canon => Some("layers/world"),
            Section::Agency => Some("layers/agency"),
            Section::Beliefs => Some("layers/beliefs"),
            Section::Memory => Some("layers/memory"),
            Section::Responses => Some("responses"),
            Section::Moods => Some("moods"),
            Section::Characters => Some("personalities"),
            Section::Worlds => Some("worlds"),
            // The settings are loose files at the mind root, which is why they
            // are a named set below rather than a directory listing: the root
            // also holds a lock file, a scratch folder and the daemon's redo
            // log, and none of those is a setting.
            Section::Settings => None,
        }
    }

    pub fn format(self) -> Format {
        match self {
            Section::Canon | Section::Agency | Section::Beliefs | Section::Memory => {
                Format::Markdown
            }
            Section::Responses
            | Section::Moods
            | Section::Characters
            | Section::Worlds
            | Section::Settings => Format::Yaml,
        }
    }

    /// Whether entries nest. A response is one document; a canon topic holds
    /// entries, and some of those hold more.
    pub fn nests(self) -> bool {
        matches!(
            self,
            Section::Canon | Section::Agency | Section::Beliefs | Section::Memory
        )
    }
}

/// The settings, named one by one.
///
/// A fixed table rather than a listing, because the mind root is not a folder
/// of settings — it is a folder that happens to contain some. `CLAUDE.md` is
/// markdown while the other two are YAML, which is exactly why the file name is
/// written out here instead of derived from the section's format.
const SETTINGS: [(&str, &str, &str); 3] = [
    ("mind", "Mind", "mind.yaml"),
    ("projection", "Projection schema", "projection.yaml"),
    ("guidance", "Authoring guidance", "CLAUDE.md"),
];

/// A setting whose document has addressable *parts*, and how to find them:
/// (setting, the key holding the list, the key inside each item that names it).
///
/// The projection schema's substance is its nine layers — `interaction`,
/// `world`, `beliefs` and the rest — each one a window, a threshold, a budget,
/// a summarisation prompt and a set of selection groups. As one document that
/// is seven hundred lines of YAML and the only way to change a layer's budget
/// is to find it in a textarea. As parts, `settings/projection/world` is a
/// layer, and it opens as fields like anything else.
///
/// This is the same idea as a canon topic having both entries and a body: the
/// document still reads whole at `settings/projection`, and the parts are
/// addressable underneath it.
///
/// **A part can be edited but not added or removed.** An address names a part
/// that exists, so there is no address for a tenth layer and none for deleting
/// the ninth — deliberately. Between the layers of `projection.yaml` are the
/// author's banner comments, and adding or removing an item means rewriting the
/// list, which would take those with it. Adding a layer is an act for the whole
/// document, where the author can see the comments they are moving.
const PARTED: [(&str, &str, &str); 1] = [("projection", "layers", "name")];

fn parted(setting: &str) -> Option<(&'static str, &'static str)> {
    PARTED
        .iter()
        .find(|(slug, _, _)| *slug == setting)
        .map(|(_, list, id)| (*list, *id))
}

/// One item inside a settings document.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Part<'a> {
    /// The key holding the list — `layers`.
    pub list: &'static str,
    /// The key inside an item that names it — `name`.
    pub id_key: &'static str,
    /// Which item.
    pub name: &'a str,
}

/// A place in the corpus.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Address {
    section: Section,
    names: Vec<String>,
}

/// Why an address was refused.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AddressError {
    /// The first token is not one of the nine sections.
    UnknownSection(String),
    /// A name that could not be part of a file name — see [`MindPath`].
    BadName(PathError),
    /// This section does not nest, so it has entries and no topics.
    TooDeep,
    /// `settings/…` naming something that is not a setting.
    UnknownSetting(String),
}

impl std::fmt::Display for AddressError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            AddressError::UnknownSection(s) => {
                write!(f, "`{s}` is not part of the mind")
            }
            AddressError::BadName(e) => write!(f, "{e}"),
            AddressError::TooDeep => {
                write!(f, "that part of the mind holds entries, not topics")
            }
            AddressError::UnknownSetting(s) => write!(f, "there is no `{s}` setting"),
        }
    }
}

impl Address {
    /// Parse `section/name/name…`. The empty string is the corpus itself.
    pub fn parse(raw: &str) -> Result<Option<Address>, AddressError> {
        let trimmed = raw.trim_matches('/');
        if trimmed.is_empty() {
            return Ok(None);
        }
        let mut parts = trimmed.split('/');
        let head = parts.next().unwrap_or_default();
        let section = Section::from_slug(head)
            .ok_or_else(|| AddressError::UnknownSection(head.to_owned()))?;

        let names: Vec<String> = parts.map(str::to_owned).collect();
        // Every name has to survive becoming part of a file name. Checked here
        // rather than at resolution so a bad address is refused before anything
        // touches a disk.
        for name in &names {
            MindPath::root().join(name).map_err(AddressError::BadName)?;
        }
        if section == Section::Settings {
            if let Some(name) = names.first() {
                if !SETTINGS.iter().any(|(slug, _, _)| slug == name) {
                    return Err(AddressError::UnknownSetting(name.clone()));
                }
            }
            // A setting whose document has parts admits one name more: which
            // part. Everything else is one deep.
            let depth = match names.first() {
                Some(first) if parted(first).is_some() => 2,
                _ => 1,
            };
            if names.len() > depth {
                return Err(AddressError::TooDeep);
            }
        } else if !section.nests() && names.len() > 1 {
            return Err(AddressError::TooDeep);
        }
        Ok(Some(Address { section, names }))
    }

    pub fn section(&self) -> Section {
        self.section
    }

    /// The address of the thing this is inside, or `None` at a section.
    pub fn parent(&self) -> Option<Address> {
        if self.names.is_empty() {
            return None;
        }
        let mut names = self.names.clone();
        names.pop();
        Some(Address {
            section: self.section,
            names,
        })
    }

    /// This address with one more name. Checked, so a listing cannot produce an
    /// address a parse would refuse.
    pub fn child(&self, name: &str) -> Result<Address, AddressError> {
        MindPath::root().join(name).map_err(AddressError::BadName)?;
        if !self.holds_children() {
            return Err(AddressError::TooDeep);
        }
        let mut names = self.names.clone();
        names.push(name.to_owned());
        Ok(Address {
            section: self.section,
            names,
        })
    }

    /// Whether anything can sit one level below this.
    fn holds_children(&self) -> bool {
        if self.section == Section::Settings {
            return match self.names.first() {
                None => true,
                Some(first) => self.names.len() == 1 && parted(first).is_some(),
            };
        }
        self.section.nests() || self.names.is_empty()
    }

    /// The list of parts this address holds, for a settings document that has
    /// them — `settings/projection` holds its `layers`.
    pub fn parts(&self) -> Option<(&'static str, &'static str)> {
        if self.section != Section::Settings || self.names.len() != 1 {
            return None;
        }
        parted(&self.names[0])
    }

    /// The single part this address names — `settings/projection/world` is the
    /// layer called `world` inside `projection.yaml`.
    pub fn part(&self) -> Option<Part<'_>> {
        if self.section != Section::Settings || self.names.len() != 2 {
            return None;
        }
        let (list, id_key) = parted(&self.names[0])?;
        Some(Part {
            list,
            id_key,
            name: &self.names[1],
        })
    }

    /// The wire form.
    pub fn as_str(&self) -> String {
        let mut s = String::from(self.section.slug());
        for n in &self.names {
            s.push('/');
            s.push_str(n);
        }
        s
    }

    /// What a reader calls this. The last name, made readable.
    pub fn title(&self) -> String {
        match self.names.last() {
            None => self.section.title().to_owned(),
            Some(name) => {
                // Only at the top of `settings`: a *part* is named by its
                // author, and a layer that happened to be called `mind` must
                // not borrow a setting's title.
                if self.section == Section::Settings && self.names.len() == 1 {
                    if let Some((_, title, _)) = SETTINGS.iter().find(|(slug, _, _)| slug == name) {
                        return (*title).to_owned();
                    }
                }
                titleize(name)
            }
        }
    }

    /// Is this a section or a topic — something that holds other things?
    ///
    /// Answered from the address alone where the section's shape decides it: a
    /// response is always an entry, a canon topic is always a collection. Where
    /// both are possible the caller checks the disk, which is what
    /// [`Self::collection_path`] returning `Some` is for.
    pub fn is_section(&self) -> bool {
        self.names.is_empty()
    }

    /// The directory this addresses, if it could be one.
    ///
    /// `None` for a section that does not nest below its own folder — a
    /// response has no children, so `responses/blush_then_own` is never a
    /// directory — and for `settings`, whose entries are loose files.
    pub fn collection_path(&self) -> Option<MindPath> {
        let dir = self.section.dir()?;
        if !self.section.nests() && !self.names.is_empty() {
            return None;
        }
        let mut p = MindPath::parse(dir).ok()?;
        for n in &self.names {
            p = p.join(n).ok()?;
        }
        Some(p)
    }

    /// The document this addresses, if it could be one.
    ///
    /// `None` for a section itself — `canon` is not a document — and for
    /// anything whose name would not make a file.
    pub fn entry_path(&self) -> Option<MindPath> {
        let last = self.names.last()?;
        if self.section == Section::Settings {
            // The *first* name, not the last: a part lives inside the setting's
            // document, so `settings/projection/world` is still
            // `projection.yaml` — which of its layers is [`Self::part`].
            let first = self.names.first()?;
            let (_, _, file) = SETTINGS.iter().find(|(slug, _, _)| slug == first)?;
            return MindPath::parse(file).ok();
        }
        let dir = self.section.dir()?;
        let mut p = MindPath::parse(dir).ok()?;
        for n in &self.names[..self.names.len() - 1] {
            p = p.join(n).ok()?;
        }
        p.join(&format!("{last}.{}", self.section.format().ext()))
            .ok()
    }

    /// The settings, as addresses. The console lists them from this rather than
    /// from the mind root, so a stray file at the root is not a setting.
    pub fn settings() -> Vec<Address> {
        SETTINGS
            .iter()
            .map(|(slug, _, _)| Address {
                section: Section::Settings,
                names: vec![(*slug).to_owned()],
            })
            .collect()
    }

    /// The address of a whole section.
    pub fn of_section(section: Section) -> Address {
        Address {
            section,
            names: Vec::new(),
        }
    }
}

/// `blush_then_own` → `Blush then own`; `cindy-tan` → `Cindy tan`.
///
/// A display name, never an id: the id is the name as stored, and this is only
/// ever shown. Making it reversible would mean constraining what an author may
/// call a file, which is the opposite of the point.
fn titleize(name: &str) -> String {
    let spaced = name.replace(['_', '-'], " ");
    let mut chars = spaced.chars();
    match chars.next() {
        None => String::new(),
        Some(first) => first.to_uppercase().collect::<String>() + chars.as_str(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn a(s: &str) -> Address {
        Address::parse(s).expect("parses").expect("not the root")
    }

    #[test]
    fn the_corpus_itself_is_the_empty_address() {
        for raw in ["", "/", "///"] {
            assert_eq!(Address::parse(raw), Ok(None), "{raw:?}");
        }
    }

    #[test]
    fn an_address_names_a_section_and_a_chain_of_names() {
        let x = a("canon/ammo/bolt");
        assert_eq!(x.section(), Section::Canon);
        assert_eq!(x.as_str(), "canon/ammo/bolt");
        assert_eq!(x.title(), "Bolt");
        assert_eq!(x.parent().unwrap().as_str(), "canon/ammo");
        assert_eq!(x.parent().unwrap().parent().unwrap().as_str(), "canon");
        assert!(x.parent().unwrap().parent().unwrap().is_section());
    }

    /// **The point of the whole module.** No extension appears in an address,
    /// and the section decides it — so a caller never states, and can never get
    /// wrong, a storage question.
    #[test]
    fn the_section_supplies_the_extension_not_the_caller() {
        assert_eq!(
            a("canon/ammo/bolt").entry_path().unwrap().as_str(),
            "layers/world/ammo/bolt.md"
        );
        assert_eq!(
            a("responses/blush_then_own").entry_path().unwrap().as_str(),
            "responses/blush_then_own.yaml"
        );
        assert_eq!(
            a("characters/cindy-tan").entry_path().unwrap().as_str(),
            "personalities/cindy-tan.yaml"
        );
        assert_eq!(
            a("worlds/earth").entry_path().unwrap().as_str(),
            "worlds/earth.yaml"
        );
        assert_eq!(
            a("memory/cindy-tan/first-kiss")
                .entry_path()
                .unwrap()
                .as_str(),
            "layers/memory/cindy-tan/first-kiss.md"
        );
    }

    /// A canon topic is a folder of entries *and* a page of its own. One idea,
    /// two files, and the address is the same for both.
    #[test]
    fn a_topic_addresses_both_its_entries_and_its_overview() {
        let ammo = a("canon/ammo");
        assert_eq!(
            ammo.collection_path().unwrap().as_str(),
            "layers/world/ammo"
        );
        assert_eq!(ammo.entry_path().unwrap().as_str(), "layers/world/ammo.md");
    }

    #[test]
    fn a_section_is_a_collection_and_never_a_document() {
        let canon = a("canon");
        assert!(canon.is_section());
        assert_eq!(canon.collection_path().unwrap().as_str(), "layers/world");
        assert_eq!(canon.entry_path(), None);
        assert_eq!(canon.title(), "World knowledge");
    }

    #[test]
    fn a_flat_section_has_entries_and_no_topics() {
        assert!(!Section::Responses.nests());
        assert_eq!(Address::parse("responses/a/b"), Err(AddressError::TooDeep));
        // And a response is never a folder.
        assert_eq!(a("responses/blush_then_own").collection_path(), None);
        // Building one by hand is refused too, so a listing cannot make one.
        assert_eq!(
            a("responses/blush_then_own").child("x"),
            Err(AddressError::TooDeep)
        );
    }

    /// Only these nine, so nothing else in the mind directory is addressable —
    /// `node_modules` and the daemon's own `.substrate` cannot be *named*,
    /// which is a stronger guarantee than filtering them out of a listing.
    #[test]
    fn only_the_named_sections_exist() {
        for good in [
            "canon",
            "agency",
            "beliefs",
            "memory",
            "responses",
            "moods",
            "characters",
            "worlds",
            "settings",
        ] {
            assert!(Address::parse(good).is_ok(), "{good}");
        }
        for bad in ["layers", "node_modules", ".substrate", "scratchpad", "etc"] {
            assert_eq!(
                Address::parse(bad),
                Err(AddressError::UnknownSection(bad.to_owned())),
                "{bad} was addressable"
            );
        }
    }

    /// The names still have to survive becoming a file name, so every rule in
    /// `MindPath` still applies — an address is a nicer spelling of a path, not
    /// a way around one.
    #[test]
    fn a_name_that_could_not_be_a_file_is_refused() {
        for bad in [
            "canon/..",
            "canon/../../etc",
            "canon/a/../b",
            r"canon/a\b",
            "canon/nul",
            "canon/x.",
        ] {
            assert!(
                matches!(Address::parse(bad), Err(AddressError::BadName(_))),
                "{bad} was accepted"
            );
        }
    }

    #[test]
    fn the_settings_are_a_named_set_not_a_folder() {
        assert_eq!(Section::Settings.dir(), None);
        assert_eq!(
            a("settings/projection").entry_path().unwrap().as_str(),
            "projection.yaml"
        );
        assert_eq!(
            a("settings/mind").entry_path().unwrap().as_str(),
            "mind.yaml"
        );
        // Markdown among the YAML, which is why the file name is written out
        // rather than derived from the section's format.
        assert_eq!(
            a("settings/guidance").entry_path().unwrap().as_str(),
            "CLAUDE.md"
        );
        assert_eq!(a("settings/projection").title(), "Projection schema");
        // A file at the root that is not a setting cannot be reached by
        // pretending it is one.
        assert_eq!(
            Address::parse("settings/package"),
            Err(AddressError::UnknownSetting("package".into()))
        );
        assert_eq!(Address::settings().len(), 3);
    }

    #[test]
    fn a_title_is_for_reading_and_the_name_stays_the_id() {
        assert_eq!(a("responses/blush_then_own").title(), "Blush then own");
        assert_eq!(a("canon/alien_wildlife").title(), "Alien wildlife");
        assert_eq!(a("characters/cindy-tan").title(), "Cindy tan");
        // The id is untouched by any of that.
        assert_eq!(
            a("responses/blush_then_own").as_str(),
            "responses/blush_then_own"
        );
    }

    /// The one non-ASCII name in the real mind still addresses.
    #[test]
    fn a_unicode_name_addresses_and_resolves() {
        let x = a("canon/relationships/bonds/protégés");
        assert_eq!(
            x.entry_path().unwrap().as_str(),
            "layers/world/relationships/bonds/protégés.md"
        );
        assert_eq!(x.title(), "Protégés");
    }

    #[test]
    fn child_and_parent_are_inverses() {
        let topic = a("canon/ammo");
        let entry = topic.child("bolt").unwrap();
        assert_eq!(entry.as_str(), "canon/ammo/bolt");
        assert_eq!(entry.parent().unwrap(), topic);
        assert_eq!(Address::of_section(Section::Canon).parent(), None);
    }
}
