//! Makes `models.override.yaml` optional.
//!
//! `src/models/overrides.rs` embeds the override document with `include_str!`.
//! Pointing that straight at the workspace root would make the file mandatory —
//! `include_str!` on a missing path is a compile error — and the file is
//! deliberately gitignored, so every fresh clone and every CI run would fail to
//! build on a file that is not supposed to be in the repository.
//!
//! So the document is copied into `OUT_DIR` and included from there. When the
//! real file is absent this writes an empty one, and an empty document overrides
//! nothing.

use std::path::PathBuf;

/// Relative to this crate's directory, so it resolves the same wherever cargo is
/// invoked from.
const OVERRIDE_FILE: &str = "../models.override.yaml";

fn main() {
    // Rebuild when the override appears, changes, or is deleted. `rerun-if-changed`
    // on a non-existent path is honoured — cargo re-runs the script when one
    // shows up — which is what makes "drop the file in and rebuild" work.
    println!("cargo:rerun-if-changed={OVERRIDE_FILE}");
    println!("cargo:rerun-if-changed=build.rs");

    let out =
        PathBuf::from(std::env::var_os("OUT_DIR").expect("OUT_DIR")).join("models.override.yaml");

    let document = match std::fs::read_to_string(OVERRIDE_FILE) {
        Ok(s) => {
            println!("cargo:warning=using local model overrides from {OVERRIDE_FILE}");
            s
        }
        // Any read failure is "no override": absent is the ordinary case, and a
        // file that exists but cannot be read is reported by the parse step in
        // `overrides.rs` rather than aborting the build here with less context.
        Err(_) => String::new(),
    };

    std::fs::write(&out, document).unwrap_or_else(|e| panic!("write {out:?}: {e}"));
}
