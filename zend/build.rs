//! Build script: make `include_dir!("$CARGO_MANIFEST_DIR/web")` (in
//! `src/api/mod.rs`) actually rebuild when the embedded web UI changes.
//!
//! `include_dir!` embeds `web/` at compile time but emits no `rerun-if-changed`,
//! so editing `index.html` / `zend-api*.js` alone would NOT recompile zend — it
//! would silently keep serving a stale UI. Emit a `rerun-if-changed` for every
//! file under `web/` so any web edit forces a re-embed.

use std::path::Path;

fn main() {
    watch(Path::new("web"));
}

fn watch(dir: &Path) {
    println!("cargo:rerun-if-changed={}", dir.display());
    let Ok(entries) = std::fs::read_dir(dir) else {
        return;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            watch(&path);
        } else {
            println!("cargo:rerun-if-changed={}", path.display());
        }
    }
}
