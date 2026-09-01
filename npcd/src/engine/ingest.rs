//! Putting the mind's layers into the substrate.
//!
//! # What this fixes
//!
//! The `Layers` load phase used to walk the mind directory, count what it found,
//! move the progress bar and write **nothing**. The phase was named "Ingesting
//! mind layers" and it ingested no layers — the exact shape of dishonesty the
//! engine module exists to refuse, sitting in the loading screen where it was
//! least visible. A world with 1,267 documents in `layers/world/` booted in a
//! second and the characters knew none of it.
//!
//! # A conversation per document, not per layer
//!
//! This is the whole performance story, and the first attempt got it backwards.
//!
//! Appending documents as turns to one conversation per layer **cannot batch**:
//! turn N+1's KV depends on turn N's, so 1,267 documents is 1,267 forward passes
//! at a batch width of one, each paying full wave latency. Measured at 3.5
//! seconds a document — 84 minutes for one layer.
//!
//! One conversation *per document* is embarrassingly parallel instead. Sixteen
//! independent sequences co-batch into a single prefill forward, and a wider
//! window just packs more sequences into it. This is what zend's calibration
//! ingest does and why its comments record ~6.9k tok/s, "at the batched-forward
//! gate's rate" — the model was never the bottleneck, the serial round-trips
//! were.
//!
//! **The conversation is a vehicle for writing, not a container for reading.**
//! That is the fact that makes one-per-document obviously right, and missing it
//! is what produced the wrong shape: the turns seal into the substrate and the
//! gather reaches them across conversations regardless of which one wrote them,
//! so the sequence can be dropped the moment its turn is sealed. zend frees its
//! per-file conversations for exactly this reason.
//!
//! It also fixes the memory: sixteen small resident sequences that seal and free,
//! rather than one conversation accumulating every document's K/V until the card
//! is full.
//!
//! # Prefilled, not decoded
//!
//! [`candle_conversation::Sequence::submit_prefilled_turn`] writes both halves
//! verbatim in one batched forward. The document's address is the user half and
//! its content is the assistant half — nothing is generated, because the text is
//! already on disk.
//!
//! The pairing is not decoration. The gather retrieves *turns*, and a turn whose
//! user half names what it is — `world/alpha_centauri` — gives provenance
//! selection something to match a question against, where a bare wall of content
//! would only match on the content's own words.
//!
//! # What stops a restart re-ingesting everything
//!
//! The **content-hash ledger**, persisted inside `.substrate/` so that wiping the
//! substrate wipes the ledger with it. A second boot finds every hash unchanged
//! and writes nothing.

use std::path::{Path, PathBuf};

use serde::Serialize;

use crate::engine::loading::LoadProgress;
use crate::engine::watcher::{is_ingestible, walk, Ledger, Reconcile};

/// How many documents are in flight at once.
///
/// Every slot is a live sequence holding its own K/V until the turn seals, so
/// this is a VRAM decision as much as a throughput one — zend runs the same
/// width for the same reason, sized to keep a concurrent window inside a 16 GB
/// card. Wider packs more sequences per forward and costs proportionally more
/// resident K/V; narrower starves the batch back towards the serial shape this
/// exists to escape.
pub const WINDOW: usize = 16;

/// How many slots must be free before the window refills.
///
/// Creation is served between waves, so a refill costs one wave boundary
/// whatever it creates. Refilling at half amortises that boundary over eight
/// documents rather than paying it for one.
pub const REFILL_AT: usize = WINDOW / 2;

// The window's bounds, held where a change to the constant cannot get past them
// rather than in a test that only runs when somebody runs it. Too narrow starves
// the batch back towards the serial shape this exists to escape; too wide is
// resident K/V for every slot until its turn seals.
const _: () = assert!(WINDOW >= 8, "too narrow to fill a forward");
const _: () = assert!(WINDOW <= 32, "every slot holds live K/V until it seals");

/// One layer with content to ingest.
#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct LayerSource {
    /// The layer's name in the schema — `world`, `agency`, `memory`.
    pub name: String,
    /// The folder its content lives in.
    pub dir: PathBuf,
    /// What the layer counts, from its `ingest_unit:`. Shown on the loading
    /// screen: "412 / 1204 documents" says more than "412 / 1204 files", and the
    /// schema is the only thing that knows which noun a layer uses.
    pub unit: String,
    /// Whose content this is, when the layer is per-character.
    ///
    /// `None` is shared ground — `world/`, `agency/` — which every character
    /// reads. `Some(personality)` is one character's own and must not be
    /// reachable by any other: a life story on a shared conversation lets every
    /// character recall every other character's childhood.
    pub owner: Option<String>,
}

/// The order layers are ingested in, lowest first.
///
/// **Not schema order.** The schema lists layers by what they are for; ingestion
/// has to follow what they *depend on*, because a turn's provenance is captured
/// against what is already in the substrate when it lands. A memory that refers
/// to a place has to be written after the place exists, or the reference has
/// nothing to attach to.
///
/// So: the world first, since it is the ground everything else stands on; then
/// the shared behavioural templates; then a character's stance on that world;
/// and **memory last**, because a life story refers to all of the others at once.
///
/// A layer not named here sorts after the named ones. New layers are then
/// ingested somewhere sane without this list becoming a thing you must edit
/// before a layer works at all.
const ORDER: &[&str] = &[
    "world",
    "environment",
    "perception",
    "action",
    "agency",
    "relationships",
    "beliefs",
    "memory",
];

fn rank(name: &str) -> usize {
    ORDER.iter().position(|n| *n == name).unwrap_or(ORDER.len())
}

/// Layers a life story produces, which are therefore **not** ingested from disk.
///
/// A belief is a conclusion, and a conclusion without an origin is the thing the
/// life-story design exists to end — so beliefs, relationships and standing
/// intentions come from `layers/life/<who>/` and its `<tool_call>` blocks, and
/// land in the substrate as tagged records. Reading them from disk as well would
/// give a character two sources for the same conviction: one with a history and
/// one asserted, disagreeing silently.
///
/// The files are **left alone**, not deleted. Twenty-nine of cindy-tan's beliefs
/// have no life episode yet, and losing them to a schema change is not a trade
/// worth making — they are content somebody wrote, waiting to be migrated.
const LIFE_DERIVED: &[&str] = &["beliefs", "relationships", "agency"];

/// What one layer's ingest did.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize)]
pub struct LayerReport {
    pub layer: String,
    /// Documents written as turns.
    pub written: usize,
    /// Documents whose content hash matched — nothing written.
    pub unchanged: usize,
    /// Documents that could not be read or whose turn failed. Counted rather
    /// than fatal: one bad document must not cost a world its other sixty-five.
    pub failed: usize,
}

/// Which layers have content, in **dependency order** — see [`ORDER`].
///
/// A layer is a turn sink when it declares an `ingest_unit:` **and** has a
/// folder. Both halves matter: the declaration is the schema saying this layer
/// holds authored content, and the folder is that content existing. A layer that
/// declares a unit and has no folder is simply empty, which is not an error —
/// most minds fill in over time.
///
/// `characters` is every personality id the mind declares. A layer directory
/// whose children are all personality ids is **per-character** and becomes one
/// source per character; anything else is shared ground. Detected from the
/// filesystem rather than declared, because `layers/memory/zen/` is unambiguous
/// and a second declaration to keep in step with it would be one more thing to
/// get wrong.
pub fn sources(
    mind: &Path,
    schema_layers: &[(String, Option<String>)],
    characters: &[String],
) -> Vec<LayerSource> {
    let root = mind.join("layers");
    let mut out: Vec<LayerSource> = Vec::new();

    for (name, unit) in schema_layers {
        let Some(unit) = unit else { continue };
        // A life story produces these; reading them from disk too would give a
        // character two sources for one conviction. See `LIFE_DERIVED`.
        if LIFE_DERIVED.contains(&name.as_str()) {
            continue;
        }
        let dir = root.join(name);
        if !dir.is_dir() {
            continue;
        }
        let owners = per_character_owners(&dir, characters);
        if owners.is_empty() {
            out.push(LayerSource {
                name: name.clone(),
                dir,
                unit: unit.clone(),
                owner: None,
            });
        } else {
            for owner in owners {
                out.push(LayerSource {
                    name: name.clone(),
                    dir: dir.join(&owner),
                    unit: unit.clone(),
                    owner: Some(owner),
                });
            }
        }
    }

    out.sort_by(|a, b| {
        rank(&a.name)
            .cmp(&rank(&b.name))
            .then_with(|| a.name.cmp(&b.name))
            .then_with(|| a.owner.cmp(&b.owner))
    });
    out
}

/// The character subdirectories of a layer, if that is what it holds.
///
/// Empty when the layer is shared ground. The test is that **every** immediate
/// subdirectory names a personality — one stray topic folder among them means
/// the layer is organised by subject, not by character, and splitting it would
/// scatter shared content into per-character silos.
fn per_character_owners(dir: &Path, characters: &[String]) -> Vec<String> {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return Vec::new();
    };
    let mut subdirs: Vec<String> = Vec::new();
    let mut has_loose_file = false;
    for e in entries.flatten() {
        let path = e.path();
        if path.is_dir() {
            if let Some(n) = path.file_name().and_then(|n| n.to_str()) {
                subdirs.push(n.to_string());
            }
        } else if is_ingestible(&path) {
            has_loose_file = true;
        }
    }
    // A layer with content directly in it is shared, whatever its subdirectories
    // look like — `world/` has `alliances.md` beside `ammo/`.
    if has_loose_file || subdirs.is_empty() {
        return Vec::new();
    }
    if subdirs.iter().all(|d| characters.iter().any(|c| c == d)) {
        subdirs.sort();
        subdirs
    } else {
        Vec::new()
    }
}

/// The address a document is known by inside its layer.
///
/// `world/factions/hess` — the layer, then the path beneath it, without the
/// extension. This is the user half of the turn, so it is what provenance
/// selection matches a question against; it has to read like a name, not a path.
pub fn address(layer: &str, owner: Option<&str>, dir: &Path, file: &Path) -> String {
    let rel = file.strip_prefix(dir).unwrap_or(file);
    let stem: Vec<String> = rel
        .components()
        .map(|c| c.as_os_str().to_string_lossy().into_owned())
        .collect();
    let mut joined = stem.join("/");
    for ext in [".md", ".yaml", ".yml"] {
        if let Some(cut) = joined.strip_suffix(ext) {
            joined = cut.to_string();
            break;
        }
    }
    // Backslashes on Windows would otherwise make the same document read as two
    // different names depending on which machine ingested it.
    let joined = joined.replace('\\', "/");
    // A per-character document names whose it is: `memory/zen/first_winter`.
    // Without the owner, two characters' documents with the same filename —
    // which biographies routinely have, `childhood.md` — would be
    // indistinguishable in the gather.
    match owner {
        Some(who) => format!("{layer}/{who}/{joined}"),
        None => format!("{layer}/{joined}"),
    }
}

/// Everything to write for one layer, as `(address, content)` pairs.
///
/// Separated from the substrate write so the decision — which documents moved —
/// is testable without an engine, which is most of what can go wrong here.
pub fn pending(
    source: &LayerSource,
    ledger: &Ledger,
) -> std::io::Result<(Vec<(String, String)>, LayerReport)> {
    let files = walk(&source.dir)?;
    let mut out = Vec::new();
    let mut report = LayerReport {
        layer: source.name.clone(),
        ..Default::default()
    };
    for file in files {
        let Ok(content) = std::fs::read_to_string(&file) else {
            // Unreadable — a permission, a half-written file, a stray binary
            // with a .md name. Counted and skipped.
            report.failed += 1;
            continue;
        };
        match ledger.reconcile(&file, Some(&content)) {
            Reconcile::Added | Reconcile::Changed => {
                out.push((
                    address(&source.name, source.owner.as_deref(), &source.dir, &file),
                    content,
                ));
                report.written += 1;
            }
            Reconcile::Unchanged | Reconcile::Removed => report.unchanged += 1,
        }
    }
    Ok((out, report))
}

/// Report progress for one layer on the loading screen.
pub fn announce(progress: &LoadProgress, source: &LayerSource, done: usize, total: usize) {
    progress.set_unit(source.unit.clone());
    progress.set_detail(source.label());
    progress.set_progress(done as u64, total as u64);
}

impl LayerSource {
    /// How this source reads on the loading screen and in the log.
    ///
    /// Names the character for a per-character layer — "memory · zen" rather
    /// than three consecutive phases all saying "memory", which is what an
    /// operator watching a slow start would otherwise see.
    pub fn label(&self) -> String {
        match &self.owner {
            Some(who) => format!("{} · {who}", self.name),
            None => self.name.clone(),
        }
    }

    /// The gather-scope tags a source's turns carry.
    ///
    /// A shared layer is tagged with its name. A per-character layer is tagged
    /// with the character too, which is what a projection policy scopes on so a
    /// character's own memory reaches it and nobody else's does.
    pub fn tags(&self) -> Vec<String> {
        match &self.owner {
            Some(who) => vec![self.name.clone(), format!("{}:{who}", self.name)],
            None => vec![self.name.clone()],
        }
    }

    /// The system prompt a document's conversation is opened under.
    ///
    /// Short on purpose. The conversation exists to carry one document into the
    /// substrate and is dropped once the turn seals, so a long prompt here would
    /// be prefilled once per document to no end.
    pub fn prompt(&self) -> String {
        match &self.owner {
            Some(who) => format!("The {} of {who}. Their own, and nobody else's.", self.name),
            None => format!("Reference material: the {} layer.", self.name),
        }
    }
}

/// How many documents to create in the next batch, given the window's state.
///
/// Zero until the window has drained past [`REFILL_AT`], so a refill fills half
/// the window rather than trickling one document per wave boundary — which is
/// the failure mode that starved zend's own batch to 2–4 wide before it was
/// pipelined.
pub fn refill(in_flight: usize, remaining: usize) -> usize {
    if remaining == 0 {
        return 0;
    }
    let free = WINDOW.saturating_sub(in_flight);
    if free < REFILL_AT && in_flight > 0 {
        return 0;
    }
    free.min(remaining)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tmp(tag: &str) -> PathBuf {
        let p = std::env::temp_dir().join(format!("npcd-ingest-{}-{tag}", std::process::id()));
        let _ = std::fs::remove_dir_all(&p);
        std::fs::create_dir_all(&p).unwrap();
        p
    }

    fn layers(pairs: &[(&str, Option<&str>)]) -> Vec<(String, Option<String>)> {
        pairs
            .iter()
            .map(|(n, u)| (n.to_string(), u.map(str::to_string)))
            .collect()
    }

    const CAST: &[&str] = &["zen", "keeper", "cindy-tan"];

    fn cast() -> Vec<String> {
        CAST.iter().map(|s| s.to_string()).collect()
    }

    fn source(name: &str, dir: PathBuf) -> LayerSource {
        LayerSource {
            name: name.into(),
            dir,
            unit: "documents".into(),
            owner: None,
        }
    }

    // ── the window ──────────────────────────────────────────────────────────

    /// **The property the whole rewrite exists for.** A full batch on the first
    /// pass is what makes sixteen documents co-batch into one prefill forward
    /// instead of sixteen forwards of width one.
    #[test]
    fn the_first_batch_fills_the_whole_window() {
        assert_eq!(refill(0, 1_267), WINDOW);
    }

    /// Refilling one slot at a time is the failure this avoids: creation is
    /// served between waves, so a one-document refill pays a whole wave boundary
    /// for one document.
    #[test]
    fn a_nearly_full_window_does_not_refill() {
        assert_eq!(refill(WINDOW, 100), 0);
        assert_eq!(refill(WINDOW - 1, 100), 0);
        // Not until half of it is free.
        assert_eq!(refill(WINDOW - REFILL_AT, 100), REFILL_AT);
    }

    /// The tail: fewer documents left than slots, so take what is left rather
    /// than waiting for a refill threshold that will never be reached.
    #[test]
    fn the_last_few_documents_still_go() {
        assert_eq!(refill(0, 3), 3);
        assert_eq!(refill(0, 0), 0);
        // An empty window always refills, however little is left — otherwise the
        // final documents of a layer would never be written.
        assert_eq!(refill(0, 1), 1);
    }

    /// The window's own bounds are held at compile time, beside the constant —
    /// see the `const _` assertions there. This checks the thing a constant
    /// cannot: that the refill logic actually respects them.
    #[test]
    fn the_window_is_wide_enough_to_batch_and_bounded_for_vram() {
        assert_eq!(
            refill(0, 10_000),
            WINDOW,
            "a refill did not fill the window"
        );
        assert!(refill(0, 10_000) >= 8, "too narrow to fill a forward");
    }

    // ── sources ─────────────────────────────────────────────────────────────

    /// **A conviction must have exactly one source.**
    ///
    /// Beliefs, relationships and intentions come from a life story's tool calls
    /// and land in the substrate. Reading `layers/beliefs/` from disk as well
    /// would give a character two of each — one with a history and one asserted,
    /// disagreeing with nothing to say which is right.
    #[test]
    fn life_derived_layers_are_not_ingested_from_disk() {
        let mind = tmp("derived");
        for l in ["beliefs", "relationships", "agency", "world"] {
            std::fs::create_dir_all(mind.join("layers").join(l)).unwrap();
            std::fs::write(mind.join("layers").join(l).join("x.md"), "x").unwrap();
        }
        let found = sources(
            &mind,
            &layers(&[
                ("beliefs", Some("beliefs")),
                ("relationships", Some("people")),
                ("agency", Some("goals")),
                ("world", Some("documents")),
            ]),
            &cast(),
        );
        let names: Vec<&str> = found.iter().map(|s| s.name.as_str()).collect();
        assert_eq!(
            names,
            vec!["world"],
            "a life-derived layer was read from disk"
        );
        let _ = std::fs::remove_dir_all(&mind);
    }

    /// **A life story must not be readable by another character.**
    #[test]
    fn a_per_character_layer_becomes_one_source_per_character() {
        let mind = tmp("owners");
        for who in ["zen", "keeper", "cindy-tan"] {
            std::fs::create_dir_all(mind.join("layers/memory").join(who)).unwrap();
        }
        let found = sources(&mind, &layers(&[("memory", Some("memories"))]), &cast());

        assert_eq!(
            found.len(),
            3,
            "the biographies were not split by character"
        );
        let owners: Vec<Option<&str>> = found.iter().map(|s| s.owner.as_deref()).collect();
        assert_eq!(owners, vec![Some("cindy-tan"), Some("keeper"), Some("zen")]);
        assert!(found[2].dir.ends_with("zen"));
        let _ = std::fs::remove_dir_all(&mind);
    }

    /// A layer with content directly in it is shared ground, whatever its
    /// subdirectories look like — `world/` has `alliances.md` beside `ammo/`.
    #[test]
    fn a_layer_with_loose_files_stays_shared() {
        let mind = tmp("shared");
        let dir = mind.join("layers/world");
        std::fs::create_dir_all(dir.join("zen")).unwrap();
        std::fs::write(dir.join("alliances.md"), "# Alliances").unwrap();

        let found = sources(&mind, &layers(&[("world", Some("documents"))]), &cast());
        assert_eq!(found.len(), 1);
        assert_eq!(
            found[0].owner, None,
            "a shared layer was split by character"
        );
        let _ = std::fs::remove_dir_all(&mind);
    }

    #[test]
    fn topic_subdirectories_do_not_make_a_layer_per_character() {
        let mind = tmp("topics");
        for topic in ["ammo", "armor"] {
            std::fs::create_dir_all(mind.join("layers/world").join(topic)).unwrap();
        }
        let found = sources(&mind, &layers(&[("world", Some("documents"))]), &cast());
        assert_eq!(found.len(), 1);
        assert_eq!(found[0].owner, None);
        let _ = std::fs::remove_dir_all(&mind);
    }

    /// **Dependency order, and memory last.**
    #[test]
    fn layers_ingest_lowest_first_with_memory_last() {
        let mind = tmp("order2");
        for l in ["memory", "world", "beliefs", "agency"] {
            std::fs::create_dir_all(mind.join("layers").join(l)).unwrap();
            std::fs::write(mind.join("layers").join(l).join("x.md"), "x").unwrap();
        }
        let found = sources(
            &mind,
            &layers(&[
                ("memory", Some("memories")),
                ("world", Some("documents")),
                ("agency", Some("goals")),
                ("beliefs", Some("beliefs")),
            ]),
            &cast(),
        );
        let names: Vec<&str> = found.iter().map(|s| s.name.as_str()).collect();
        // `agency` and `beliefs` are absent because a life story produces them —
        // see `life_derived_layers_are_not_ingested_from_disk`. What this test
        // pins is the ORDER of what remains: world before memory, always.
        assert_eq!(names, vec!["world", "memory"]);
        assert_eq!(
            *names.last().unwrap(),
            "memory",
            "memory was not ingested last"
        );
        let _ = std::fs::remove_dir_all(&mind);
    }

    #[test]
    fn an_unranked_layer_sorts_after_the_ranked_ones() {
        let mind = tmp("unranked");
        for l in ["world", "dreams"] {
            std::fs::create_dir_all(mind.join("layers").join(l)).unwrap();
            std::fs::write(mind.join("layers").join(l).join("x.md"), "x").unwrap();
        }
        let found = sources(
            &mind,
            &layers(&[("dreams", Some("dreams")), ("world", Some("documents"))]),
            &cast(),
        );
        let names: Vec<&str> = found.iter().map(|s| s.name.as_str()).collect();
        assert_eq!(names, vec!["world", "dreams"]);
        let _ = std::fs::remove_dir_all(&mind);
    }

    #[test]
    fn a_layer_is_a_source_only_with_a_unit_and_a_folder() {
        let mind = tmp("sources");
        std::fs::create_dir_all(mind.join("layers/world")).unwrap();
        std::fs::create_dir_all(mind.join("layers/interaction")).unwrap();

        let found = sources(
            &mind,
            &layers(&[
                ("world", Some("documents")),
                ("interaction", None),
                ("memory", Some("memories")),
            ]),
            &cast(),
        );
        assert_eq!(found.len(), 1);
        assert_eq!(found[0].name, "world");
        assert_eq!(found[0].unit, "documents");
        let _ = std::fs::remove_dir_all(&mind);
    }

    // ── addresses ───────────────────────────────────────────────────────────

    #[test]
    fn an_address_names_the_document_not_its_path() {
        let dir = Path::new("/mind/layers/world");
        assert_eq!(
            address("world", None, dir, Path::new("/mind/layers/world/alpha.md")),
            "world/alpha"
        );
        assert_eq!(
            address(
                "world",
                None,
                dir,
                Path::new("/mind/layers/world/factions/hess.md")
            ),
            "world/factions/hess"
        );
    }

    /// Two characters' biographies routinely share a filename — `childhood.md` —
    /// and without the owner they would be indistinguishable in the gather.
    #[test]
    fn a_per_character_address_names_whose_it_is() {
        let dir = Path::new("/m/layers/memory/zen");
        assert_eq!(
            address(
                "memory",
                Some("zen"),
                dir,
                Path::new("/m/layers/memory/zen/childhood.md")
            ),
            "memory/zen/childhood"
        );
        assert_ne!(
            address(
                "memory",
                Some("zen"),
                dir,
                Path::new("/m/layers/memory/zen/childhood.md")
            ),
            address(
                "memory",
                Some("keeper"),
                Path::new("/m/layers/memory/keeper"),
                Path::new("/m/layers/memory/keeper/childhood.md")
            ),
        );
    }

    #[test]
    fn a_windows_path_produces_the_same_address_as_a_unix_one() {
        let a = address(
            "world",
            None,
            Path::new("/m/layers/world"),
            Path::new("/m/layers/world/factions/hess.md"),
        );
        assert!(!a.contains('\\'), "a backslash reached the address: {a}");
        assert_eq!(a, "world/factions/hess");
    }

    // ── pending ─────────────────────────────────────────────────────────────

    #[test]
    fn an_unchanged_layer_writes_nothing_on_the_second_pass() {
        let mind = tmp("unchanged");
        let dir = mind.join("layers/world");
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("a.md"), "# Alpha").unwrap();
        std::fs::write(dir.join("b.md"), "# Beta").unwrap();

        let src = source("world", dir.clone());
        let ledger = Ledger::new();

        let (first, r1) = pending(&src, &ledger).unwrap();
        assert_eq!(first.len(), 2);
        assert_eq!(r1.written, 2);

        let (second, r2) = pending(&src, &ledger).unwrap();
        assert!(
            second.is_empty(),
            "a second boot rewrote unchanged documents"
        );
        assert_eq!(r2.unchanged, 2);

        std::fs::write(dir.join("a.md"), "# Alpha, revised").unwrap();
        let (third, r3) = pending(&src, &ledger).unwrap();
        assert_eq!(third.len(), 1);
        assert_eq!(third[0].0, "world/a");
        assert_eq!(r3.written, 1);
        let _ = std::fs::remove_dir_all(&mind);
    }

    /// A layer document is authored prose and the character has to read what was
    /// written, not a normalisation of it.
    #[test]
    fn content_reaches_the_turn_unaltered() {
        let mind = tmp("verbatim");
        let dir = mind.join("layers/world");
        std::fs::create_dir_all(&dir).unwrap();
        let body = "# Hess\n\nHe burned the east granary.\n\n  - indented\n";
        std::fs::write(dir.join("hess.md"), body).unwrap();

        let (out, _) = pending(&source("world", dir), &Ledger::new()).unwrap();
        assert_eq!(out[0].1, body);
        let _ = std::fs::remove_dir_all(&mind);
    }

    #[test]
    fn an_unreadable_document_is_counted_not_fatal() {
        let mind = tmp("bad");
        let dir = mind.join("layers/world");
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("good.md"), "fine").unwrap();
        std::fs::write(dir.join("bad.md"), [0xff, 0xfe, 0x00, 0x9c]).unwrap();

        let (out, report) = pending(&source("world", dir), &Ledger::new()).unwrap();
        assert_eq!(out.len(), 1, "the good document did not survive");
        assert_eq!(report.written, 1);
        assert_eq!(report.failed, 1);
        let _ = std::fs::remove_dir_all(&mind);
    }

    /// The conversation is dropped once its turn seals, so its prompt is
    /// prefilled once per document and must stay short.
    #[test]
    fn a_documents_prompt_is_short_and_names_its_owner() {
        let shared = source("world", PathBuf::new());
        assert!(shared.prompt().len() < 80);
        assert!(shared.prompt().contains("world"));

        let mine = LayerSource {
            owner: Some("zen".into()),
            ..source("memory", PathBuf::new())
        };
        assert!(mine.prompt().contains("zen"));
        assert!(mine.prompt().len() < 80);
    }
}
