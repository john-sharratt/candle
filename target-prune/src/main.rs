//! `cargo prune` — sweep superseded build artifacts by hand.
//!
//! The same pass [`build.rs`](../build.rs) runs automatically, exposed as a
//! command for when you want a report (`--dry-run`) or want to reclaim disk
//! without starting a build.

use std::path::PathBuf;
use target_prune::prune_target;

fn main() {
    let mut dry_run = false;
    let mut root: Option<PathBuf> = None;
    for a in std::env::args().skip(1) {
        match a.as_str() {
            // `cargo prune -- --dry-run` forwards the separator too.
            "--" => continue,
            "-n" | "--dry-run" => dry_run = true,
            "-h" | "--help" => {
                println!(
                    "usage: cargo prune [-- --dry-run] [TARGET_DIR]\n\n\
                     Keeps the newest generations per (stem, extension) in each \
                     TARGET_DIR/*/deps and deletes older ones."
                );
                return;
            }
            _ => root = Some(PathBuf::from(a)),
        }
    }

    let root = root
        .or_else(|| std::env::var_os("CARGO_TARGET_DIR").map(PathBuf::from))
        .unwrap_or_else(|| PathBuf::from("target"));

    if !root.is_dir() {
        eprintln!("target-prune: no such directory: {}", root.display());
        std::process::exit(1);
    }

    let p = prune_target(&root, dry_run);
    let verb = if dry_run { "would free" } else { "freed" };
    println!(
        "target-prune: {verb} {:.1} GB — {} stale files removed, {} kept",
        p.gb(),
        p.removed,
        p.kept
    );
}
