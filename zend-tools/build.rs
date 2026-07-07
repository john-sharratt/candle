//! Generates a compile-time table of calibration trajectories from the
//! `examples/*.md` files (produced by the `export_calibration` tool), keyed by
//! `(tool, example_index)`. Emits `$OUT_DIR/calibration_trajectories.rs` holding
//! `pub static TRAJECTORIES: &[(&str, usize, &str)]` with one `include_str!` per
//! file, so the trajectories are embedded in the binary for the prefill-based
//! calibration path. Re-runs whenever a `.md` file changes.

use std::{env, fs, path::PathBuf};

fn main() {
    let manifest = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    let examples = manifest.join("examples");
    let out = PathBuf::from(env::var("OUT_DIR").unwrap()).join("calibration_trajectories.rs");
    println!("cargo:rerun-if-changed={}", examples.display());

    // Collect `{tool}_{NN}.md` → (tool, index, path).
    let mut entries: Vec<(String, usize, PathBuf)> = Vec::new();
    if let Ok(rd) = fs::read_dir(&examples) {
        for e in rd.flatten() {
            let path = e.path();
            if path.extension().and_then(|s| s.to_str()) != Some("md") {
                continue;
            }
            let Some(stem) = path.file_stem().and_then(|s| s.to_str()) else {
                continue;
            };
            // Split on the LAST underscore: everything before is the tool name
            // (tools themselves contain underscores), the suffix is the index.
            let Some(us) = stem.rfind('_') else { continue };
            let tool = &stem[..us];
            let Ok(idx) = stem[us + 1..].parse::<usize>() else {
                continue;
            };
            println!("cargo:rerun-if-changed={}", path.display());
            entries.push((tool.to_string(), idx, path));
        }
    }
    entries.sort();

    let mut src = String::from("pub static TRAJECTORIES: &[(&str, usize, &str)] = &[\n");
    for (tool, idx, path) in &entries {
        // `{:?}` escapes the string / Windows path into a valid Rust literal.
        src.push_str(&format!(
            "    ({:?}, {}, include_str!({:?})),\n",
            tool,
            idx,
            path.to_str().unwrap()
        ));
    }
    src.push_str("];\n");
    fs::write(&out, src).unwrap();
}
