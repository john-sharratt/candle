//! Stamps this build's version (§3.5) into the crate as `TOKERA_BUILD_COUNT`,
//! `TOKERA_BUILD_HASH` and `TOKERA_BUILD_LOCAL`.

#[path = "src/version/compute.rs"]
mod compute;

use std::path::Path;

fn main() {
    let manifest = std::env::var("CARGO_MANIFEST_DIR").expect("cargo sets CARGO_MANIFEST_DIR");
    let root = Path::new(&manifest)
        .parent()
        .expect("the cluster crate sits in the workspace root");
    for path in compute::rerun_paths(root) {
        println!("cargo:rerun-if-changed={}", path.display());
    }
    let v = compute::compute(root);
    println!("cargo:rustc-env=TOKERA_BUILD_COUNT={}", v.count);
    println!("cargo:rustc-env=TOKERA_BUILD_HASH={}", v.hash);
    println!("cargo:rustc-env=TOKERA_BUILD_LOCAL={}", v.local);
}
