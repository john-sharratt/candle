//! Delete the stale generations cargo leaves behind in `target/*/deps`.
//!
//! # Why this has to exist
//!
//! Cargo names every artifact `<stem>-<16 hex>.<ext>`, where the hash covers the
//! inputs that produced it. Change a dependency and the next build writes a new
//! hash — and leaves the old file there for ever. Cargo has no garbage
//! collector, and no post-build hook to hang one on.
//!
//! On most projects that is invisible, because a test binary is a few megabytes.
//! Here it is not: every CUDA test binary statically links the ~919 MB
//! `candle-kernels` archive, so each one is ~900 MB against 6 MB for a binary
//! that does not (`web`). Cargo builds one executable per `tests/*.rs`, so a
//! single generation runs to tens of GB — measured at 134 files for 55 binaries,
//! 45 GB of them dead. That is what fills a 1.9 TB disk inside a working day.
//!
//! # What it deletes
//!
//! For each `(stem, extension)` in a `deps` directory it keeps the
//! [`KEEP_GENERATIONS`] newest files and removes the rest. Nothing else is
//! touched: not `build/` (which holds the compiled CUDA archives and is
//! expensive to regenerate), not the profile roots, not `incremental/`.
//!
//! # Why this is a command and not a hook
//!
//! Cargo has no post-build hook, and the obvious substitute does not survive
//! measurement. A build script can be made to run on every build by naming a
//! `rerun-if-changed` path that never exists — but cargo ties a script's
//! staleness to its crate's, so an always-dirty script is an always-dirty crate,
//! and that propagates to everything depending on it. Wired as a dev-dependency
//! of `candle-core`/`candle-nn`/`candle-transformers`, it recompiled `candle-nn`
//! and `candle-core` on *every* build: 5–7s each time, against an
//! `opt-level = 2` profile where those rebuilds are not cheap. A rebuild tax on
//! every build is a worse problem than the disk.
//!
//! (`cargo:warning=` makes it worse still: build-script *stdout* is metadata
//! cargo diffs run-to-run, so a line emitted only when there was something to
//! delete is itself a reason to rebuild. Report on stderr.)
//!
//! So this runs as `cargo prune`. The durable fix is to generate less: fewer
//! test binaries, and a kernel archive that is not 919 MB.
//!
//! # Safety
//!
//! Two rules keep this from breaking a build rather than merely tidying after
//! one:
//!
//! - **Files modified within [`QUIET_PERIOD`] are never touched**, so output
//!   being written right now is left alone.
//! - **Failure is not fatal.** A locked `.dll` (rust-analyzer holds proc-macro
//!   libraries open) or a running test binary is skipped, to be collected later.

use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::time::{Duration, SystemTime};

/// Leave anything this recent alone — it may belong to a build in flight.
pub const QUIET_PERIOD: Duration = Duration::from_secs(120);

/// How many generations of each artifact to keep.
///
/// **Two, not one.** Two configurations can both be current: build `web` without
/// CUDA and `candle-core` with it, and each leaves a differently-hashed artifact
/// that the *other* invocation has no use for. Keeping only the newest would make
/// alternating between them recompile every time, turning a disk saving into a
/// build-time cost. Two covers the common alternation; a third rebuild of the
/// same crate is genuinely superseded.
pub const KEEP_GENERATIONS: usize = 2;

/// What a prune pass did.
#[derive(Debug, Default, Clone, Copy)]
pub struct Pruned {
    pub freed: u64,
    pub removed: usize,
    pub kept: usize,
}

impl Pruned {
    pub fn gb(&self) -> f64 {
        self.freed as f64 / (1024.0 * 1024.0 * 1024.0)
    }
}

/// Prune every `<target>/<profile>/deps` under `root`.
pub fn prune_target(root: &Path, dry_run: bool) -> Pruned {
    let mut total = Pruned::default();
    let Ok(profiles) = fs::read_dir(root) else {
        return total;
    };
    // Every profile: `debug`, `release`, and anything custom like
    // `release-with-debug`. Each has its own `deps` and its own generations.
    for profile in profiles.flatten() {
        let deps = profile.path().join("deps");
        if deps.is_dir() {
            let p = prune_dir(&deps, dry_run);
            total.freed += p.freed;
            total.removed += p.removed;
            total.kept += p.kept;
        }
    }
    total
}

/// One generation of an artifact: when it was written, how big, and where.
type Generation = (SystemTime, u64, PathBuf);

/// One `deps` directory: keep the newest generations per `(stem, extension)`.
pub fn prune_dir(deps: &Path, dry_run: bool) -> Pruned {
    // The key includes the extension: `foo-<hash>.exe` and `foo-<hash>.pdb` are
    // separate series, and the newest of each has to survive.
    let mut groups: HashMap<(String, String), Vec<Generation>> = HashMap::new();

    let Ok(entries) = fs::read_dir(deps) else {
        return Pruned::default();
    };
    let now = SystemTime::now();

    for entry in entries.flatten() {
        let Ok(meta) = entry.metadata() else { continue };
        if !meta.is_file() {
            continue;
        }
        let modified = meta.modified().unwrap_or(now);
        // A build in flight owns its output; do not race it.
        if now
            .duration_since(modified)
            .is_ok_and(|age| age < QUIET_PERIOD)
        {
            continue;
        }
        let path = entry.path();
        let Some(name) = path.file_name().and_then(|n| n.to_str()) else {
            continue;
        };
        let Some((stem, ext)) = split_hashed(name) else {
            continue;
        };
        groups
            .entry((stem, ext))
            .or_default()
            .push((modified, meta.len(), path));
    }

    let mut out = Pruned::default();
    for (_, mut series) in groups {
        // Newest first, so everything past the keep window is superseded.
        series.sort_by_key(|(modified, _, _)| std::cmp::Reverse(*modified));
        out.kept += series.len().min(KEEP_GENERATIONS);
        for (_, len, path) in series.into_iter().skip(KEEP_GENERATIONS) {
            if dry_run || fs::remove_file(&path).is_ok() {
                out.freed += len;
                out.removed += 1;
            }
        }
    }
    out
}

/// Split `libfoo-0123456789abcdef.rlib` into (`libfoo`, `rlib`).
///
/// Returns `None` for anything not carrying cargo's 16-hex hash, which is how
/// files that are not generation-stamped — and therefore have no older siblings
/// to compare against — are left alone.
pub fn split_hashed(name: &str) -> Option<(String, String)> {
    let (base, ext) = name.rsplit_once('.')?;
    let (stem, hash) = base.rsplit_once('-')?;
    if hash.len() != 16 || !hash.bytes().all(|b| b.is_ascii_hexdigit()) {
        return None;
    }
    if stem.is_empty() {
        return None;
    }
    Some((stem.to_string(), ext.to_string()))
}

/// The `target` directory a build script is running inside.
///
/// `OUT_DIR` is `<target>/<profile>/build/<pkg>-<hash>/out`, so the root is four
/// levels up. Returns `None` if that shape does not hold, rather than guessing —
/// a pruner that deletes in the wrong place is worse than one that does nothing.
pub fn target_root_from_out_dir(out_dir: &Path) -> Option<PathBuf> {
    let build = out_dir.parent()?.parent()?; // .../build/<pkg>-<hash>/out → build
    if build.file_name()? != "build" {
        return None;
    }
    // `build`'s parent is the profile directory; its parent is the root.
    Some(build.parent()?.parent()?.to_path_buf())
}

#[cfg(test)]
mod tests {
    use super::{split_hashed, target_root_from_out_dir};
    use std::path::Path;

    #[test]
    fn recognises_cargos_generation_stamp() {
        assert_eq!(
            split_hashed("libcandle_kernels-604c335b093adf99.rlib"),
            Some(("libcandle_kernels".into(), "rlib".into()))
        );
        assert_eq!(
            split_hashed("quantized_tests-a04ae3f11e0811de.exe"),
            Some(("quantized_tests".into(), "exe".into()))
        );
        // Same stem, different extension: separate series, each keeps a newest.
        assert_eq!(
            split_hashed("web-7e96bc81057c0db1.pdb"),
            Some(("web".into(), "pdb".into()))
        );
    }

    /// Anything without the stamp has no generations to compare, so it is not
    /// ours to delete — the guard that stops this from eating `libstd` or a
    /// hand-placed file.
    #[test]
    fn leaves_unstamped_files_alone() {
        assert_eq!(split_hashed("candle_core.d"), None);
        assert_eq!(split_hashed("libstd.rlib"), None);
        assert_eq!(split_hashed("noextension"), None);
        // Right shape, wrong alphabet.
        assert_eq!(split_hashed("thing-zzzzzzzzzzzzzzzz.rlib"), None);
        // Right alphabet, wrong length.
        assert_eq!(split_hashed("thing-abc123.rlib"), None);
    }

    #[test]
    fn finds_the_target_root_from_a_build_scripts_out_dir() {
        let out = Path::new("/w/candle/target/debug/build/target-prune-abc/out");
        assert_eq!(
            target_root_from_out_dir(out).as_deref(),
            Some(Path::new("/w/candle/target"))
        );
    }

    /// A shape we do not recognise yields `None` rather than a guess.
    #[test]
    fn refuses_an_out_dir_it_does_not_understand() {
        assert_eq!(target_root_from_out_dir(Path::new("/tmp")), None);
    }
}
