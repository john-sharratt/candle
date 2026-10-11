//! Build script: make `include_dir!("$CARGO_MANIFEST_DIR/src/prompts/tools")`
//! (in `src/tool_def.rs`) rebuild when a tool definition changes.
//!
//! `include_dir!` embeds the directory at compile time but emits no
//! `rerun-if-changed`, so editing a definition alone would NOT recompile zend —
//! it would silently keep the stale catalog. Emit a `rerun-if-changed` for every
//! file under it so any edit forces a re-embed. (The web UI is not embedded:
//! the host serves it from disk, and an edit there needs no build at all.)

use std::path::Path;

fn main() {
    watch(Path::new("src/prompts/tools"));
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
