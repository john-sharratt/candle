//! `substrate_inspect export-cast` and `import` — npcd's wipe-and-rebuild,
//! offline (`docs/hit_driven_projection.md` §13.4).
//!
//! A store a new model key refuses cannot be opened by its daemon, so neither
//! export route can answer, and an import route needs an admin's identity
//! through the gateway. These take the cast out of the old store read-only,
//! under no model, beside a daemon that may still have it open; and write a
//! cast export and a kept-objects export into a fresh store, with its daemon
//! stopped, which then binds its model at the daemon's first open.

use std::cell::RefCell;
use std::fs;
use std::path::Path;
use std::time::{SystemTime, UNIX_EPOCH};

use anyhow::{Context, Result};
use candle_conversation::persistence::{cast, Admit, SharedSubstrate, SubstratePersistence};
use candle_conversation::substrate::Substrate;

/// The cast export of the store at `substrate_dir` (the `substrate`
/// directory), taken at `taken_at_ms`.
pub fn exported(substrate_dir: &Path, taken_at_ms: u64) -> Result<String> {
    let workspace = substrate_dir
        .parent()
        .context("the substrate directory has no parent to open it from")?;
    let mut substrate = Substrate::new();
    let mut persistence =
        SubstratePersistence::open_in_with_substrate_read_only(workspace, &mut substrate)
            .with_context(|| format!("opening {} read-only", substrate_dir.display()))?;
    let ids: Vec<u64> = persistence.npc_locs().iter().map(|(id, _)| *id).collect();
    let mut npcs = Vec::with_capacity(ids.len());
    for id in ids {
        if let Some(npc) = persistence.read_npc(id)? {
            npcs.push(npc);
        }
    }
    let persistence = RefCell::new(persistence);
    cast::export(
        npcs,
        |name| Ok(persistence.borrow_mut().blob(name)?),
        taken_at_ms,
    )
}

/// Write the cast export of the store at `substrate_dir` to `out`, and say how
/// many characters and portraits it holds.
pub fn export_cast(substrate_dir: &Path, out: &Path) -> Result<()> {
    let now_ms = SystemTime::now().duration_since(UNIX_EPOCH)?.as_millis() as u64;
    let text = exported(substrate_dir, now_ms)?;
    let npcs = text.lines().filter(|l| l.starts_with("{\"npc\"")).count();
    let blobs = text.lines().filter(|l| l.starts_with("{\"blob\"")).count();
    fs::write(out, &text).with_context(|| format!("writing {}", out.display()))?;
    println!(
        "cast: {npcs} characters and {blobs} portraits, {} bytes, to {}",
        text.len(),
        out.display()
    );
    Ok(())
}

/// **Writes.** Write `kept` (an `export-kept` of `daemon`) and `cast` (an
/// `export-cast`) into the store at `substrate_dir`, which must not exist yet:
/// this builds a fresh one, never adds to a store that has history. The kept
/// objects go first, then the portraits and the characters, each character
/// verbatim. The daemon must be stopped, and binds the store's model at its
/// first open.
pub fn import(
    substrate_dir: &Path,
    daemon: &str,
    kept: Option<&Path>,
    cast_file: Option<&Path>,
) -> Result<()> {
    if substrate_dir.exists() {
        anyhow::bail!(
            "{} exists: an import builds a fresh store, so remove the old one first",
            substrate_dir.display()
        );
    }
    let workspace = substrate_dir
        .parent()
        .context("the substrate directory has no parent workspace")?;
    let shared = SharedSubstrate::open_in(workspace, Admit::Unbound)
        .with_context(|| format!("creating {}", substrate_dir.display()))?;
    if let Some(path) = kept {
        let text =
            fs::read_to_string(path).with_context(|| format!("reading {}", path.display()))?;
        let report = shared.import_kept_objects(&text, daemon)?;
        println!(
            "kept: {} objects and {} blobs written, {} deleted keys skipped",
            report.objects_written, report.blobs_written, report.ceased
        );
    }
    if let Some(path) = cast_file {
        let text =
            fs::read_to_string(path).with_context(|| format!("reading {}", path.display()))?;
        let report = cast::import(
            &text,
            |bytes| Ok(shared.put_blob(bytes)?),
            |npcs| {
                shared.persistence.call(|p| {
                    for npc in &npcs {
                        p.write_npc(npc)?;
                    }
                    p.commit()
                })?;
                Ok(())
            },
        )?;
        println!(
            "cast: {} characters and {} portraits written",
            report.npcs_written, report.blobs_written
        );
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use candle_conversation::persistence::blob::hex;
    use candle_conversation::persistence::record::NpcPayload;
    use candle_conversation::persistence::SUBSTRATE_DIR;

    use super::*;

    fn npc(id: u64, name: &str) -> NpcPayload {
        serde_json::from_value(serde_json::json!({
            "npc_id": id,
            "owner_id": "u_1",
            "revision": 4,
            "created_ms": 1,
            "updated_ms": 2,
            "state": "idle",
            "name": name,
            "world_id": "moon",
            "personality_id": "commander",
            "hidden": false,
            "heartbeat_ms": 1000,
            "salience_gate": 0.5,
            "tags": [],
            "persona_description": "p",
            "persona_origin": "authored",
            "portrait_image_id": null,
            "portrait_origin": null,
        }))
        .unwrap()
    }

    /// **A cast exported offline and imported into a fresh store reads back
    /// the same**, with its portrait; and an import refuses a store that
    /// already exists.
    #[test]
    fn a_cast_moves_offline_into_a_fresh_store() {
        let old = tempfile::tempdir().unwrap();
        let portrait = {
            let shared = SharedSubstrate::open_in(old.path(), Admit::Unbound).unwrap();
            let sha = shared.put_blob(b"a portrait".to_vec()).unwrap();
            let mut zen = npc(7, "Zen");
            zen.portrait_image_id = Some(format!("{}.png", hex(&sha)));
            shared
                .persistence
                .call(|p| {
                    p.write_npc(&zen)?;
                    p.write_npc(&npc(9, "Wick"))?;
                    p.commit()
                })
                .unwrap();
            sha
        };
        let text = exported(&old.path().join(SUBSTRATE_DIR), 1).unwrap();
        let file = old.path().join("cast.jsonl");
        fs::write(&file, &text).unwrap();

        let new = tempfile::tempdir().unwrap();
        let dir = new.path().join(SUBSTRATE_DIR);
        import(&dir, "npcd", None, Some(&file)).unwrap();
        let again = exported(&dir, 1).unwrap();
        assert_eq!(again, text, "the same cast, portrait and all");
        assert!(text.contains(&hex(&portrait)));
        assert!(
            import(&dir, "npcd", None, Some(&file)).is_err(),
            "a store that exists is not imported into"
        );
    }
}
