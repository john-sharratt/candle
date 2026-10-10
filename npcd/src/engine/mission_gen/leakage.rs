//! Where the writing happens, found in what was written.
//!
//! **The writers' room is not the world's history.** The Makers write the
//! record from a vault — its levels and rooms, its desks and terminals, its
//! lift, its light rings and air handlers, one another — and that is the most
//! concrete thing in front of a Maker every turn. It went into the work: a
//! Conan scene in the Contested Cities "smelled of hot plastic" while "Ione
//! Valtiere tells me the ground is shifting"; a story of a governor's ledger
//! had "Sila Vane" for its hero and "the light ring was steady while its tubes
//! held" for its rule. Of eleven pieces judged against the lore, nine had the
//! vault in them, and review and the canon check passed every one.
//!
//! The check needs no list of words. The writers' vocabulary is the map they
//! stand in — every level, room and part named on it — and the names of the
//! people in it, with the bench's own words for the act of writing. The world's
//! vocabulary is its lore: the eras and the world documents. A term of the
//! first that the second never uses is the writers' room in the work. A term
//! the lore uses — a tower world has its terminals — is the world's, and
//! passes.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, OnceLock};

use npc_map::world::World;

/// The bench's words for the act of writing — not the world's, unless its
/// lore says them.
const WRITING: &[&str] = &[
    "commit",
    "committed",
    "the draft",
    "this draft",
    "the mission",
    "my mission",
    "maker",
    "makers",
];

/// The shortest term weighed: shorter words are too common to be anybody's.
const SHORTEST: usize = 4;

/// The writers' vocabulary in `world`: the names of its levels, rooms and
/// parts, and of the people standing in it — each lower case, without a
/// leading article, but for a place or thing named in one word, which keeps
/// it: `the stacks`, `the lift`.
///
/// **One word names the vault only as the vault's own.** "The stacks" is a
/// room the Makers walk through; "stacks of requisition forms" in a logistics
/// office is English. Matched bare, a story set over a quartermaster's paper
/// was refused for "stacks" at every report, composed again, and refused
/// again. A name of two words — "light ring", "air handler" — is the vault's
/// however it is used, and so is a person's name.
pub fn vocabulary(world: &World) -> Vec<String> {
    let map = world.map();
    let mut named: Vec<String> = Vec::new();
    for area in map.areas() {
        named.push(area.name.clone());
        for node in &area.nodes {
            named.push(node.name.clone());
            named.extend(map.parts_at(node).map(|(part, _)| part.name.clone()));
        }
    }
    let mut terms: Vec<String> = named
        .iter()
        .map(|t| bare(t))
        .filter(|t| t.chars().count() >= SHORTEST)
        .map(|t| match t.contains(' ') {
            true => t,
            false => format!("the {t}"),
        })
        .collect();
    let mut people: Vec<String> = Vec::new();
    for actor in world.actors() {
        people.push(actor.name.clone());
        if let Some(first) = actor.name.split_whitespace().next() {
            people.push(first.to_string());
        }
    }
    people.extend(WRITING.iter().map(|w| w.to_string()));
    terms.extend(
        people
            .iter()
            .map(|t| bare(t))
            .filter(|t| t.chars().count() >= SHORTEST),
    );
    terms.sort();
    terms.dedup();
    terms
}

/// The world's own words: its eras and world documents, lower case, read once
/// per mind. The lives and stories are not lore — they are what is being
/// checked, and the pieces already written carry the very leakage this finds.
pub fn lore(root: &Path) -> Arc<String> {
    static READ: OnceLock<Mutex<HashMap<PathBuf, Arc<String>>>> = OnceLock::new();
    let cache = READ.get_or_init(|| Mutex::new(HashMap::new()));
    if let Some(known) = cache.lock().expect("lore lock").get(root) {
        return Arc::clone(known);
    }
    let mut text = String::new();
    for dir in ["layers/eras", "layers/world"] {
        gather(&root.join(dir), &mut text);
    }
    let text = Arc::new(text.to_lowercase());
    cache
        .lock()
        .expect("lore lock")
        .insert(root.to_path_buf(), Arc::clone(&text));
    text
}

/// Every markdown document under `dir`, appended to `into`.
fn gather(dir: &Path, into: &mut String) {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            gather(&path, into);
        } else if path.extension().is_some_and(|e| e == "md") {
            if let Ok(t) = std::fs::read_to_string(&path) {
                into.push_str(&t);
                into.push('\n');
            }
        }
    }
}

/// The writers' terms `text` holds that the lore never uses, in the order of
/// `vocabulary`. The lore is asked about the word itself, article or none: a
/// lore that ever speaks of stacks has made them the world's.
pub fn leaks(text: &str, vocabulary: &[String], lore: &str) -> Vec<String> {
    let text = text.to_lowercase();
    vocabulary
        .iter()
        .filter(|t| has_term(&text, t) && !has_term(lore, &bare(t)))
        .cloned()
        .collect()
}

/// The fault the gate reports for `leaked` terms.
pub fn fault(leaked: &[String]) -> String {
    let named: Vec<String> = leaked.iter().map(|t| format!("\"{t}\"")).collect();
    format!(
        "It has {} in it — the vault the Makers write in, not this world's history. Nothing of \
         where you write belongs in the piece: not its rooms or machines, not the Makers, not the \
         writing itself. Tell it from inside its own world, in its own time, from what the record \
         says that world held.",
        named.join(", ")
    )
}

/// `term` without a leading article, lower case.
fn bare(term: &str) -> String {
    let t = term.trim().to_lowercase();
    ["the ", "a ", "an "]
        .iter()
        .find_map(|a| t.strip_prefix(a))
        .unwrap_or(&t)
        .to_string()
}

/// Whether `term` stands in `text` as whole words.
fn has_term(text: &str, term: &str) -> bool {
    let boundary = |c: Option<char>| c.is_none_or(|c| !c.is_alphanumeric());
    text.match_indices(term).any(|(at, _)| {
        boundary(text[..at].chars().next_back()) && boundary(text[at + term.len()..].chars().next())
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn words(ws: &[&str]) -> Vec<String> {
        ws.iter().map(|w| w.to_string()).collect()
    }

    /// **The vault in a story is found; the world's own words are not.** A
    /// tower world's lore speaks of terminals, so a terminal in a story is the
    /// world's; it never speaks of a light ring or of the Makers by name.
    #[test]
    fn the_writers_room_is_found_and_the_worlds_words_pass() {
        let vocabulary = words(&["light ring", "air handler", "terminal", "sila vane", "sila"]);
        let lore = "the towers kept their records on a terminal in every hall.";
        let story = "Sila Vane bent over the terminal. The light ring was steady while its \
                     tubes held.";
        assert_eq!(
            leaks(story, &vocabulary, lore),
            ["light ring", "sila vane", "sila"]
        );
        let clean = "Kaelor held the breach at the terminal until the ammunition ran out.";
        assert!(leaks(clean, &vocabulary, lore).is_empty());
    }

    /// A term is a whole word: "silage" is not "sila", nor "lifted" "lift".
    #[test]
    fn a_term_is_matched_as_whole_words() {
        let vocabulary = words(&["sila", "the lift"]);
        assert!(leaks("The silage lifted in the wind.", &vocabulary, "").is_empty());
        assert_eq!(leaks("The lift sighed.", &vocabulary, ""), ["the lift"]);
    }

    /// **A one-word place is the vault's only as a place**: "the stacks" is
    /// the room, "stacks of forms" is English; and a lore that names the word
    /// at all has made it the world's.
    #[test]
    fn a_one_word_place_leaks_only_as_the_place() {
        let vocabulary = words(&["the stacks", "light ring"]);
        assert!(leaks(
            "Stacks of requisition forms covered the desk.",
            &vocabulary,
            ""
        )
        .is_empty());
        assert_eq!(
            leaks("She walked into the stacks.", &vocabulary, ""),
            ["the stacks"]
        );
        assert!(
            leaks(
                "She walked into the stacks.",
                &vocabulary,
                "ammunition in stacks"
            )
            .is_empty(),
            "the lore's own word"
        );
        assert_eq!(
            leaks("A light ring hummed.", &vocabulary, ""),
            ["light ring"]
        );
    }

    #[test]
    fn a_term_is_bare_and_lower_case() {
        assert_eq!(bare("The Casting Floor"), "casting floor");
        assert_eq!(bare("a light ring"), "light ring");
    }

    /// **Against the real lore, on the pieces that were judged.** Runs over the
    /// mind at `D:/prog/mind` — the one the daemon writes — with the shipped
    /// vault and the four Makers standing in it, so it is ignored by default and
    /// run by hand: `cargo test -p npcd --lib leakage -- --ignored`.
    ///
    /// A story judged to be the writers' own building is caught; a story that
    /// tells a canon event in the world's own terms is not.
    #[test]
    #[ignore = "reads the live mind at D:/prog/mind"]
    fn the_judged_pieces_against_the_real_lore() {
        use npc_map::world::Where;
        use npc_map::MapSet;

        let root = Path::new("D:/prog/mind");
        let map = MapSet::load_dir(concat!(env!("CARGO_MANIFEST_DIR"), "/../npc-map/maps"))
            .expect("the shipped maps load");
        let mut world = World::new(map);
        for (id, name) in [
            ("m1", "Paxon Vael"),
            ("m2", "Sila Vane"),
            ("m3", "Ione Valtiere"),
            ("m4", "Vespera Kaine"),
        ] {
            world
                .enter(id, name, Where::new("vault-command", "command-room"))
                .unwrap();
        }
        let vocabulary = vocabulary(&world);
        let lore = lore(root);
        let read = |p: &str| std::fs::read_to_string(root.join(p)).unwrap_or_default();

        for bad in [
            "rejected/cleanup/the-final-count.md",
            "rejected/cleanup/the-night-the-governor-s-tally-failed.md",
            "rejected/cleanup/the-arithmetic-of-survival.md",
        ] {
            let found = leaks(&read(bad), &vocabulary, &lore);
            assert!(
                !found.is_empty(),
                "{bad} carries the vault and nothing was found"
            );
        }
        let good = "layers/stories/the-attrition-finding.md";
        let found = leaks(&read(good), &vocabulary, &lore);
        assert!(found.is_empty(), "{good} is the world's own: {found:?}");
    }

    #[test]
    fn the_fault_names_what_it_found() {
        let f = fault(&words(&["light ring", "sila vane"]));
        assert!(
            f.starts_with("It has \"light ring\", \"sila vane\" in it"),
            "{f}"
        );
    }
}
