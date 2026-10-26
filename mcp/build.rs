//! Rebuild when a guide is added, changed or removed: `include_dir!` names
//! only the files it found to cargo, so a new `.md` alone would not.

fn main() {
    println!("cargo:rerun-if-changed=src/guides");
}
