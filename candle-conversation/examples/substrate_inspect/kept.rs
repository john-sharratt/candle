//! `substrate_inspect export-kept` — a store's kept custom objects, the export
//! `GET /v1/substrate/<daemon>/objects/export` answers, taken offline.
//!
//! A store a new model key refuses cannot be opened by a daemon under that
//! key, so it cannot answer the route; and the build that wrote it may
//! predate the route. Its kept objects — accounts, MCP state, journals'
//! records — are no model's, so the store is opened read-only under no model
//! and exported by the same code the route runs
//! (`candle_conversation::persistence::kept_objects`). Read-only, it runs
//! beside a daemon that still has the store open.

use std::cell::RefCell;
use std::fs;
use std::path::Path;
use std::time::{SystemTime, UNIX_EPOCH};

use anyhow::{Context, Result};
use candle_conversation::persistence::kept_objects;
use candle_conversation::persistence::SubstratePersistence;
use candle_conversation::substrate::Substrate;

/// `daemon`'s export of the store at `substrate_dir` (the `substrate`
/// directory), taken at `taken_at_ms`.
pub fn exported(substrate_dir: &Path, daemon: &str, taken_at_ms: u64) -> Result<String> {
    let workspace = substrate_dir
        .parent()
        .context("the substrate directory has no parent to open it from")?;
    let mut substrate = Substrate::new();
    let persistence =
        SubstratePersistence::open_in_with_substrate_read_only(workspace, &mut substrate)
            .with_context(|| format!("opening {} read-only", substrate_dir.display()))?;
    let persistence = RefCell::new(persistence);
    kept_objects::export(
        substrate.held_custom_objects(),
        |name| Ok(persistence.borrow_mut().blob(name)?),
        daemon,
        taken_at_ms,
    )
}

/// Write `daemon`'s export of the store at `substrate_dir` to `out`, and say
/// how many objects and blobs it holds.
pub fn export_kept(substrate_dir: &Path, daemon: &str, out: &Path) -> Result<()> {
    let now_ms = SystemTime::now().duration_since(UNIX_EPOCH)?.as_millis() as u64;
    let text = exported(substrate_dir, daemon, now_ms)?;
    let objects = text
        .lines()
        .filter(|l| l.starts_with("{\"object\""))
        .count();
    let blobs = text.lines().filter(|l| l.starts_with("{\"blob\"")).count();
    fs::write(out, &text).with_context(|| format!("writing {}", out.display()))?;
    println!(
        "{daemon}: {objects} kept objects and {blobs} blobs, {} bytes, to {}",
        text.len(),
        out.display()
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use candle_conversation::persistence::record::CustomObjectPayload;
    use candle_conversation::persistence::{Admit, SharedSubstrate, SUBSTRATE_DIR};

    use super::*;

    /// **The offline export is the route's**: a store holding an account and a
    /// recurrent row — model state — exports, read-only with its writer closed,
    /// exactly the lines its own daemon's export gives: the account kept, the
    /// row left out. Past the header, which names the moment.
    #[test]
    fn the_offline_export_is_the_routes() {
        let dir = tempfile::tempdir().unwrap();
        let live = {
            let shared = SharedSubstrate::open_in(dir.path(), Admit::Unbound).unwrap();
            shared
                .put_custom_object(CustomObjectPayload::new(
                    "accounts/google:1",
                    serde_json::json!({"name": "a"}),
                ))
                .unwrap();
            shared
                .put_custom_object(CustomObjectPayload::new(
                    "recurrent/abc",
                    serde_json::json!({}),
                ))
                .unwrap();
            shared.export_kept_objects("zend").unwrap()
        };
        let offline = exported(&dir.path().join(SUBSTRATE_DIR), "zend", 1).unwrap();
        let body = |text: &str| text.lines().skip(1).map(str::to_string).collect::<Vec<_>>();
        assert_eq!(body(&offline), body(&live));
        assert_eq!(body(&offline).len(), 1, "{offline}");
        assert!(offline.contains("accounts/google:1"), "{offline}");
    }
}
