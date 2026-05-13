// CLI tool for candle-kernels build cache management.
//
// Shares all hashing/compilation/compression logic with build.rs via include!().
// Run from the candle-kernels directory:
//
//   cargo run --bin kernel_tool -- status
//   cargo run --bin kernel_tool -- hash
//   cargo run --bin kernel_tool -- rebuild-archive-hashes
//   cargo run --bin kernel_tool -- rebuild-staging --build-dir target/debug/build/candle-kernels-.../out
//   cargo run --bin kernel_tool -- rebuild-staging-hashes
//   cargo run --bin kernel_tool -- check-for-changes
//   cargo run --bin kernel_tool -- rebuild-archives
//   cargo run --bin kernel_tool -- compile --group simple
//   cargo run --bin kernel_tool -- compress --group q4_k
//   cargo run --bin kernel_tool -- clean-staged

// Some build_utils functions are only used by build.rs, not by this binary.
#![allow(dead_code)]

include!("../../build_utils.rs");

use clap::{Parser, Subcommand};

#[derive(Parser)]
#[command(name = "kernel_tool", about = "Candle CUDA kernel build cache tool")]
struct Cli {
    #[command(subcommand)]
    command: Commands,

    /// Working directory (must contain src/ and precompiled/ directories).
    /// Defaults to the current directory.
    #[arg(long, default_value = ".")]
    dir: String,
}

#[derive(Subcommand)]
enum Commands {
    /// Show cache status for all archive groups
    Status,

    /// Compute and display hashes for all archive groups
    Hash {
        /// Only show specific archive group(s); repeatable
        #[arg(long)]
        group: Vec<String>,
    },

    /// Recompute .sha256 hash files for archives that have valid .a.gz files.
    /// Stamps existing .a.gz files as matching current source.
    RebuildArchiveHashes,

    /// Copy .o files from a build directory into staged/ cache.
    /// Use after a successful build to populate the staged cache.
    RebuildStaging {
        /// Directory containing compiled .o files (e.g. target/debug/build/candle-kernels-.../out).
        /// If omitted, auto-detects the most recent cargo build output.
        #[arg(long)]
        build_dir: Option<String>,
    },

    /// Recompute .o.sha256 hash files for .o files currently in staged/.
    /// Stamps existing staged .o files as matching current source.
    RebuildStagingHashes,

    /// Dry-run: report exactly which archives and kernels would be rebuilt.
    CheckForChanges,

    /// Force-rebuild archives from staged .o files, recompute all hashes.
    /// Links staged .o files into .a, compresses to .a.gz, updates .sha256.
    RebuildArchives {
        /// Only rebuild specific archive group(s); repeatable
        #[arg(long)]
        group: Vec<String>,
    },

    /// Compile only kernels missing or stale in staged/.
    /// Skips kernels that already have a valid .o + .sha256 in staged/.
    /// Saves compiled .o files + .sha256 hashes into staged/.
    Compile {
        /// Compile only specific archive group(s); repeatable (e.g. --group simple --group q4_k)
        #[arg(long)]
        group: Vec<String>,

        /// Number of parallel compilation threads
        #[arg(long, default_value = "8")]
        threads: usize,
    },

    /// Force-recompile ALL kernels in a group (ignores staged cache).
    /// Saves compiled .o files + .sha256 hashes into staged/.
    Recompile {
        /// Recompile only specific archive group(s); repeatable (e.g. --group simple --group sampling)
        #[arg(long)]
        group: Vec<String>,

        /// Number of parallel compilation threads
        #[arg(long, default_value = "8")]
        threads: usize,
    },

    /// Compress .a archive(s) into .a.gz in precompiled/
    Compress {
        /// Compress only specific archive group(s); repeatable
        #[arg(long)]
        group: Vec<String>,

        /// Directory containing .a files
        #[arg(long, default_value = "build_tool")]
        build_dir: String,
    },

    /// Wipe the staged .o cache directory
    CleanStaged,

    /// List all archive groups and their constituent kernels
    ListGroups,
}

fn main() -> Result<()> {
    let cli = Cli::parse();

    // Change to the specified working directory
    if cli.dir != "." {
        std::env::set_current_dir(&cli.dir)
            .with_context(|| format!("Failed to cd to {}", cli.dir))?;
    }

    // Validate we're in the right directory
    if !Path::new("src").is_dir() {
        anyhow::bail!(
            "Expected to find a src/ directory. Run this from candle-kernels/ \
             or pass --dir <path-to-candle-kernels>"
        );
    }

    let is_msvc = detect_is_msvc();
    let archive_groups = build_archive_groups(is_msvc);
    let precompiled_dir = PathBuf::from("precompiled");
    let staged_dir = PathBuf::from("staged");
    let base_dir = PathBuf::from(".");

    match cli.command {
        Commands::Status => cmd_status(&archive_groups, &precompiled_dir, &staged_dir, &base_dir),
        Commands::Hash { group } => {
            cmd_hash(&archive_groups, &base_dir, &group)
        }
        Commands::RebuildArchiveHashes => {
            cmd_rebuild_archive_hashes(&archive_groups, &precompiled_dir, &base_dir)
        }
        Commands::RebuildStaging { build_dir } => {
            cmd_rebuild_staging(&archive_groups, &staged_dir, build_dir.as_deref())
        }
        Commands::RebuildStagingHashes => {
            cmd_rebuild_staging_hashes(&archive_groups, &staged_dir, &base_dir)
        }
        Commands::CheckForChanges => {
            cmd_check_for_changes(&archive_groups, &precompiled_dir, &staged_dir, &base_dir)
        }
        Commands::RebuildArchives { group } => {
            cmd_rebuild_archives(
                &archive_groups,
                &staged_dir,
                &precompiled_dir,
                &base_dir,
                &group,
                is_msvc,
            )
        }
        Commands::Compile {
            group,
            threads,
        } => cmd_compile(
            &archive_groups,
            &staged_dir,
            &base_dir,
            &group,
            threads,
        ),
        Commands::Recompile {
            group,
            threads,
        } => cmd_recompile(
            &archive_groups,
            &staged_dir,
            &base_dir,
            &group,
            threads,
        ),
        Commands::Compress { group, build_dir } => cmd_compress(
            &archive_groups,
            &PathBuf::from(&build_dir),
            &precompiled_dir,
            &group,
            is_msvc,
        ),
        Commands::CleanStaged => cmd_clean_staged(&staged_dir),
        Commands::ListGroups => cmd_list_groups(&archive_groups),
    }
}

// ============================================================================
// Subcommand implementations
// ============================================================================

fn cmd_status(
    groups: &[ArchiveGroup],
    precompiled_dir: &Path,
    staged_dir: &Path,
    base_dir: &Path,
) -> Result<()> {
    let mut dep_cache: HashMap<String, String> = HashMap::new();

    println!("{:<20} {:>8}  {:>8}  {:>8}  {}", "GROUP", "KERNELS", "CACHED", "STAGED", "STATUS");
    println!("{}", "-".repeat(72));

    let mut total_kernels = 0usize;
    let mut cached_groups = 0usize;
    let mut staged_count = 0usize;

    // Collect problems for the detail section
    struct StagingProblem {
        kernel_path: String,
        kernel_name: String,
        group_name: String,
        kind: StagingProblemKind,
    }
    enum StagingProblemKind {
        MissingO,       // .o file not in staged/
        StaleHash,      // .o exists but .sha256 is wrong or missing
    }

    let mut problems: Vec<StagingProblem> = Vec::new();

    for group in groups {
        let (kernel_hashes, aggregate_hash) =
            compute_group_hashes(group, base_dir, &mut dep_cache)?;
        let is_cached = is_archive_cache_valid(&group.name, precompiled_dir, &aggregate_hash);

        // Check each kernel's staged status
        let mut group_staged = 0usize;
        for kernel_path in &group.kernels {
            let name = kernel_stem(kernel_path);
            let expected_hash = kernel_hashes
                .iter()
                .find(|(p, _)| p == kernel_path)
                .map(|(_, h)| h.as_str())
                .unwrap_or("");

            let staged_o = staged_dir.join(format!("{}.o", name));
            let staged_hash_file = staged_dir.join(format!("{}.o.sha256", name));

            if !staged_o.exists() {
                problems.push(StagingProblem {
                    kernel_path: kernel_path.clone(),
                    kernel_name: name,
                    group_name: group.name.clone(),
                    kind: StagingProblemKind::MissingO,
                });
            } else {
                let hash_ok = staged_hash_file.exists()
                    && fs::read_to_string(&staged_hash_file)
                        .map(|h| h.trim() == expected_hash)
                        .unwrap_or(false);
                if hash_ok {
                    group_staged += 1;
                } else {
                    problems.push(StagingProblem {
                        kernel_path: kernel_path.clone(),
                        kernel_name: name,
                        group_name: group.name.clone(),
                        kind: StagingProblemKind::StaleHash,
                    });
                }
            }
        }

        let status = if is_cached {
            cached_groups += 1;
            "OK (cached)"
        } else if group_staged == group.kernels.len() {
            "DIRTY (staged ready)"
        } else {
            "DIRTY"
        };

        total_kernels += group.kernels.len();
        staged_count += group_staged;

        println!(
            "{:<20} {:>8}  {:>8}  {:>8}  {}",
            group.name,
            group.kernels.len(),
            if is_cached { "yes" } else { "no" },
            format!("{}/{}", group_staged, group.kernels.len()),
            status,
        );
    }

    println!("{}", "-".repeat(72));
    println!(
        "Total: {} groups, {} kernels, {}/{} archives cached, {}/{} staged",
        groups.len(),
        total_kernels,
        cached_groups,
        groups.len(),
        staged_count,
        total_kernels,
    );

    // Detail section: show problems and repair commands
    if !problems.is_empty() {
        let missing: Vec<&StagingProblem> = problems
            .iter()
            .filter(|p| matches!(p.kind, StagingProblemKind::MissingO))
            .collect();
        let stale: Vec<&StagingProblem> = problems
            .iter()
            .filter(|p| matches!(p.kind, StagingProblemKind::StaleHash))
            .collect();

        println!("\n--- Staging problems ({} total) ---\n", problems.len());

        if !missing.is_empty() {
            println!("Missing .o files in staged/ ({}):", missing.len());
            for p in &missing {
                println!("  [{}] {} ({})", p.group_name, p.kernel_name, p.kernel_path);
            }
            println!("\n  To copy from last build output:");
            println!("    cargo run --bin kernel_tool -- rebuild-staging");
            println!("\n  Or compile just the affected group(s):");
            let mut affected_groups: Vec<&str> = missing.iter().map(|p| p.group_name.as_str()).collect();
            affected_groups.sort();
            affected_groups.dedup();
            for g in &affected_groups {
                println!("    cargo run --bin kernel_tool -- compile --group {}", g);
            }
            println!();
        }

        if !stale.is_empty() {
            println!("Stale/missing .sha256 for staged .o files ({}):", stale.len());
            for p in &stale {
                println!("  [{}] {} ({})", p.group_name, p.kernel_name, p.kernel_path);
            }
            println!("\n  To repair all staging hashes:");
            println!("    cargo run --bin kernel_tool -- rebuild-staging-hashes");
            println!();
        }

        // If everything is in staged/ (just needs hash repair), suggest the fast path
        if missing.is_empty() && !stale.is_empty() {
            println!("All .o files present — just the hashes need repair.");
            println!("After repairing, rebuild archives with:");
            println!("  cargo run --bin kernel_tool -- rebuild-staging-hashes");
            println!("  cargo run --bin kernel_tool -- rebuild-archives");
        }
    }

    Ok(())
}

fn cmd_hash(
    groups: &[ArchiveGroup],
    base_dir: &Path,
    filter_group: &[String],
) -> Result<()> {
    let mut dep_cache: HashMap<String, String> = HashMap::new();

    for group in groups {
        if !filter_group.is_empty() && !filter_group.iter().any(|f| f == &group.name) {
            continue;
        }

        let (kernel_hashes, aggregate_hash) =
            compute_group_hashes(group, base_dir, &mut dep_cache)?;

        println!("=== {} (aggregate: {}) ===", group.name, &aggregate_hash[..16]);
        for (path, hash) in &kernel_hashes {
            println!("  {} {}", &hash[..16], path);
        }
        println!();
    }

    Ok(())
}

fn cmd_rebuild_archive_hashes(
    groups: &[ArchiveGroup],
    precompiled_dir: &Path,
    base_dir: &Path,
) -> Result<()> {
    let mut dep_cache: HashMap<String, String> = HashMap::new();
    let mut repaired = 0;

    println!("Rebuilding archive hash files (.sha256) for existing .a.gz archives...");
    for group in groups {
        let gz_file = precompiled_dir.join(format!("lib{}.a.gz", group.name));
        if !gz_file.exists() {
            println!("  SKIP {} — no .a.gz file", group.name);
            continue;
        }

        let (_, aggregate_hash) = compute_group_hashes(group, base_dir, &mut dep_cache)?;
        save_archive_hash(&group.name, &aggregate_hash, precompiled_dir)?;
        println!("  OK   {} → {}", group.name, &aggregate_hash[..16]);
        repaired += 1;
    }

    println!("\nRepaired {} archive hash files", repaired);
    Ok(())
}

/// Find all cargo build output directories for candle-kernels.
/// Returns them sorted most-recent first (by directory mtime).
fn find_all_build_dirs() -> Result<Vec<PathBuf>> {
    let target_dir = PathBuf::from("target");
    let mut candidates: Vec<(PathBuf, std::time::SystemTime)> = Vec::new();

    for profile in &["debug", "release"] {
        let build_dir = target_dir.join(profile).join("build");
        if !build_dir.exists() {
            continue;
        }
        for entry in fs::read_dir(&build_dir)?.flatten() {
            let name = entry.file_name().to_string_lossy().to_string();
            if name.starts_with("candle-kernels-") {
                let out_dir = entry.path().join("out");
                if out_dir.is_dir() {
                    let has_objects = fs::read_dir(&out_dir)
                        .map(|entries| {
                            entries.flatten().any(|e| {
                                e.path().extension().map(|ext| ext == "o").unwrap_or(false)
                            })
                        })
                        .unwrap_or(false);
                    if has_objects {
                        let mtime = entry
                            .metadata()
                            .and_then(|m| m.modified())
                            .unwrap_or(std::time::SystemTime::UNIX_EPOCH);
                        candidates.push((out_dir, mtime));
                    }
                }
            }
        }
    }

    if candidates.is_empty() {
        anyhow::bail!(
            "No candle-kernels build output found in target/. \
             Pass --build-dir explicitly."
        );
    }

    candidates.sort_by(|a, b| b.1.cmp(&a.1)); // most recent first
    Ok(candidates.into_iter().map(|(path, _)| path).collect())
}

/// For a given kernel name, find the newest .o file across all candidate build dirs.
/// Returns the path to the newest .o, or None if not found in any.
fn find_newest_object(kernel_name: &str, build_dirs: &[PathBuf]) -> Option<PathBuf> {
    let mut best: Option<(PathBuf, std::time::SystemTime)> = None;

    for dir in build_dirs {
        let obj = dir.join(format!("{}.o", kernel_name));
        if obj.exists() {
            let mtime = fs::metadata(&obj)
                .and_then(|m| m.modified())
                .unwrap_or(std::time::SystemTime::UNIX_EPOCH);
            match &best {
                Some((_, best_mtime)) if mtime > *best_mtime => {
                    best = Some((obj, mtime));
                }
                None => {
                    best = Some((obj, mtime));
                }
                _ => {}
            }
        }
    }

    best.map(|(path, _)| path)
}

fn cmd_rebuild_staging(
    groups: &[ArchiveGroup],
    staged_dir: &Path,
    build_dir_override: Option<&str>,
) -> Result<()> {
    let build_dirs: Vec<PathBuf> = match build_dir_override {
        Some(dir) => {
            let p = PathBuf::from(dir);
            if !p.is_dir() {
                anyhow::bail!("Build directory does not exist: {}", p.display());
            }
            vec![p]
        }
        None => {
            let dirs = find_all_build_dirs()?;
            println!(
                "Found {} candle-kernels build dir(s), will pick newest .o per kernel:",
                dirs.len()
            );
            for d in &dirs {
                println!("  {}", d.display());
            }
            dirs
        }
    };

    fs::create_dir_all(staged_dir)?;

    let mut copied = 0usize;
    let mut skipped = 0usize;

    // Collect all kernel names across all groups
    let all_kernels: Vec<String> = groups
        .iter()
        .flat_map(|g| g.kernels.iter())
        .map(|k| kernel_stem(k))
        .collect();

    // Deduplicate (shouldn't be needed but just in case)
    let mut seen = HashSet::new();
    for name in &all_kernels {
        if !seen.insert(name.clone()) {
            continue;
        }

        if let Some(src) = find_newest_object(name, &build_dirs) {
            let dest = staged_dir.join(format!("{}.o", name));
            fs::copy(&src, &dest).with_context(|| {
                format!("Failed to copy {}.o to staged/", name)
            })?;
            // Remove any stale hash — staging hashes must be rebuilt separately
            let hash_file = staged_dir.join(format!("{}.o.sha256", name));
            let _ = fs::remove_file(&hash_file);
            copied += 1;
        } else {
            skipped += 1;
        }
    }

    println!(
        "Copied {} .o files to staged/ ({} not found in any build dir)",
        copied, skipped
    );
    if copied > 0 {
        println!("Run 'rebuild-staging-hashes' to stamp all staged .o files as current.");
    }
    Ok(())
}

fn cmd_rebuild_staging_hashes(
    groups: &[ArchiveGroup],
    staged_dir: &Path,
    base_dir: &Path,
) -> Result<()> {
    let mut dep_cache: HashMap<String, String> = HashMap::new();
    let mut stamped = 0usize;
    let mut missing = 0usize;

    println!("Rebuilding staged .o.sha256 hash files...");

    for group in groups {
        let canonical_args = canonical_args_for_hash(&group.compile_args);

        for kernel_path in &group.kernels {
            let name = kernel_stem(kernel_path);
            let staged_o = staged_dir.join(format!("{}.o", name));

            if !staged_o.exists() {
                missing += 1;
                continue;
            }

            // Compute the current hash for this kernel
            let hash = compute_kernel_hash(
                kernel_path,
                &canonical_args,
                base_dir,
                &group.include_dirs,
                &mut dep_cache,
            )?;

            // Write the hash file
            let hash_file = staged_dir.join(format!("{}.o.sha256", name));
            fs::write(&hash_file, &hash).with_context(|| {
                format!("Failed to write {}", hash_file.display())
            })?;
            stamped += 1;
        }
    }

    println!(
        "Stamped {} staged .o files ({} .o files not found in staged/)",
        stamped, missing
    );
    Ok(())
}

fn cmd_check_for_changes(
    groups: &[ArchiveGroup],
    precompiled_dir: &Path,
    staged_dir: &Path,
    base_dir: &Path,
) -> Result<()> {
    let mut dep_cache: HashMap<String, String> = HashMap::new();
    let mut any_dirty = false;

    println!("Checking what would be rebuilt...\n");

    for group in groups {
        let (kernel_hashes, aggregate_hash) =
            compute_group_hashes(group, base_dir, &mut dep_cache)?;

        if is_archive_cache_valid(&group.name, precompiled_dir, &aggregate_hash) {
            continue; // Archive is up-to-date
        }

        any_dirty = true;
        println!("DIRTY archive: {} ({} kernels)", group.name, group.kernels.len());

        let mut staged_hits = 0usize;
        let mut to_compile: Vec<String> = Vec::new();

        for kernel_path in &group.kernels {
            let name = kernel_stem(kernel_path);
            let current_hash = kernel_hashes
                .iter()
                .find(|(p, _)| p == kernel_path)
                .map(|(_, h)| h.as_str())
                .unwrap_or("");

            if is_staged_kernel_valid(staged_dir, &name, current_hash) {
                staged_hits += 1;
            } else {
                to_compile.push(kernel_path.clone());
            }
        }

        if staged_hits > 0 {
            println!(
                "  {} kernel(s) cached in staged/ (would skip compilation)",
                staged_hits
            );
        }
        if !to_compile.is_empty() {
            println!(
                "  {} kernel(s) would need compilation:",
                to_compile.len()
            );
            for k in &to_compile {
                println!("    {}", k);
            }
        }
        println!();
    }

    if !any_dirty {
        println!("All archives are up-to-date. Nothing would be rebuilt.");
    }

    Ok(())
}

fn cmd_rebuild_archives(
    groups: &[ArchiveGroup],
    staged_dir: &Path,
    precompiled_dir: &Path,
    base_dir: &Path,
    filter_group: &[String],
    is_msvc: bool,
) -> Result<()> {
    let mut dep_cache: HashMap<String, String> = HashMap::new();

    fs::create_dir_all(precompiled_dir)?;

    // Use a temp build dir for linking (we copy .o files there from staged)
    let link_dir = PathBuf::from("build_link_tmp");
    fs::create_dir_all(&link_dir)?;

    let mut rebuilt = 0usize;

    for group in groups {
        if !filter_group.is_empty() && !filter_group.iter().any(|f| f == &group.name) {
            continue;
        }

        // Collect all .o files from staged/
        let mut all_present = true;
        let mut object_files: Vec<PathBuf> = Vec::new();
        let mut missing_kernels: Vec<String> = Vec::new();

        for kernel_path in &group.kernels {
            let name = kernel_stem(kernel_path);
            let staged_o = staged_dir.join(format!("{}.o", name));
            let dest_o = link_dir.join(format!("{}.o", name));

            if staged_o.exists() {
                fs::copy(&staged_o, &dest_o).with_context(|| {
                    format!("Failed to copy staged {}.o", name)
                })?;
                object_files.push(dest_o);
            } else {
                all_present = false;
                missing_kernels.push(kernel_path.clone());
            }
        }

        if !all_present {
            println!(
                "  SKIP {} — missing {} .o files in staged/:",
                group.name,
                missing_kernels.len()
            );
            for k in &missing_kernels {
                println!("    {}", k);
            }
            continue;
        }

        // Link
        println!("Linking {}...", group.name);
        create_archive(&group.name, &object_files, &link_dir, is_msvc)?;

        // Compress
        let lib_a = link_dir.join(format!("lib{}.a", group.name));
        let lib_gz = precompiled_dir.join(format!("lib{}.a.gz", group.name));
        compress_gz(&lib_a, &lib_gz)?;

        // Compute and save the archive hash
        let (_, aggregate_hash) = compute_group_hashes(group, base_dir, &mut dep_cache)?;
        save_archive_hash(&group.name, &aggregate_hash, precompiled_dir)?;
        println!("  OK   {} → hash {}", group.name, &aggregate_hash[..16]);

        rebuilt += 1;
    }

    // Clean up temp link dir
    let _ = fs::remove_dir_all(&link_dir);

    println!("\nRebuilt {} archive(s)", rebuilt);
    Ok(())
}

/// Incremental compile: only compile kernels missing or stale in staged/.
fn cmd_compile(
    groups: &[ArchiveGroup],
    staged_dir: &Path,
    base_dir: &Path,
    filter_group: &[String],
    max_threads: usize,
) -> Result<()> {
    fs::create_dir_all(staged_dir)?;
    let mut dep_cache: HashMap<String, String> = HashMap::new();

    // Use a temp dir for nvcc output, then move .o to staged/
    let compile_dir = PathBuf::from("build_compile_tmp");
    fs::create_dir_all(&compile_dir)?;

    let mut total_compiled = 0usize;
    let mut total_skipped = 0usize;

    for group in groups {
        if !filter_group.is_empty() && !filter_group.iter().any(|f| f == &group.name) {
            continue;
        }

        let canonical_args = canonical_args_for_hash(&group.compile_args);

        // Figure out which kernels need compilation
        let mut to_compile: Vec<&str> = Vec::new();
        let mut kernel_hash_map: HashMap<String, String> = HashMap::new();

        for kernel_path in &group.kernels {
            let name = kernel_stem(kernel_path);
            let hash = compute_kernel_hash(
                kernel_path,
                &canonical_args,
                base_dir,
                &group.include_dirs,
                &mut dep_cache,
            )?;
            kernel_hash_map.insert(kernel_path.clone(), hash.clone());

            if is_staged_kernel_valid(staged_dir, &name, &hash) {
                total_skipped += 1;
            } else {
                to_compile.push(kernel_path);
            }
        }

        if to_compile.is_empty() {
            println!("{}: all {} kernels up-to-date in staged/", group.name, group.kernels.len());
            continue;
        }

        println!(
            "{}: compiling {} kernel(s) ({} up-to-date, {} threads)...",
            group.name,
            to_compile.len(),
            group.kernels.len() - to_compile.len(),
            max_threads,
        );

        let start = std::time::Instant::now();
        compile_kernels_parallel(&to_compile, &compile_dir, &group.compile_args, max_threads)?;

        // Move compiled .o to staged/ and write .sha256
        for kernel_path in &to_compile {
            let name = kernel_stem(kernel_path);
            if let Some(hash) = kernel_hash_map.get(*kernel_path) {
                save_to_staged_cache(&compile_dir, staged_dir, &name, hash)?;
            }
        }

        total_compiled += to_compile.len();
        println!(
            "  Done in {:.1}s",
            start.elapsed().as_secs_f64()
        );
    }

    // Clean up temp dir
    let _ = fs::remove_dir_all(&compile_dir);

    println!(
        "\nCompiled {} kernel(s), {} already up-to-date",
        total_compiled, total_skipped
    );
    Ok(())
}

/// Force-recompile ALL kernels in a group (ignores staged cache).
fn cmd_recompile(
    groups: &[ArchiveGroup],
    staged_dir: &Path,
    base_dir: &Path,
    filter_group: &[String],
    max_threads: usize,
) -> Result<()> {
    fs::create_dir_all(staged_dir)?;
    let mut dep_cache: HashMap<String, String> = HashMap::new();

    let compile_dir = PathBuf::from("build_compile_tmp");
    fs::create_dir_all(&compile_dir)?;

    for group in groups {
        if !filter_group.is_empty() && !filter_group.iter().any(|f| f == &group.name) {
            continue;
        }

        let canonical_args = canonical_args_for_hash(&group.compile_args);
        let kernel_refs: Vec<&str> = group.kernels.iter().map(|s| s.as_str()).collect();

        println!(
            "Recompiling {} ({} kernels, {} threads)...",
            group.name,
            kernel_refs.len(),
            max_threads,
        );

        let start = std::time::Instant::now();
        compile_kernels_parallel(&kernel_refs, &compile_dir, &group.compile_args, max_threads)?;

        // Save all .o to staged/ with hashes
        for kernel_path in &group.kernels {
            let name = kernel_stem(kernel_path);
            let hash = compute_kernel_hash(
                kernel_path,
                &canonical_args,
                base_dir,
                &group.include_dirs,
                &mut dep_cache,
            )?;
            save_to_staged_cache(&compile_dir, staged_dir, &name, &hash)?;
        }

        println!(
            "  Done in {:.1}s",
            start.elapsed().as_secs_f64()
        );
    }

    let _ = fs::remove_dir_all(&compile_dir);
    Ok(())
}

fn cmd_compress(
    groups: &[ArchiveGroup],
    build_dir: &Path,
    precompiled_dir: &Path,
    filter_group: &[String],
    is_msvc: bool,
) -> Result<()> {
    fs::create_dir_all(precompiled_dir)?;

    for group in groups {
        if !filter_group.is_empty() && !filter_group.iter().any(|f| f == &group.name) {
            continue;
        }

        // First link objects into .a
        let object_files: Vec<PathBuf> = group
            .kernels
            .iter()
            .map(|k| build_dir.join(format!("{}.o", kernel_stem(k))))
            .collect();

        let all_exist = object_files.iter().all(|f| f.exists());
        if !all_exist {
            let missing: Vec<_> = object_files
                .iter()
                .filter(|f| !f.exists())
                .map(|f| f.display().to_string())
                .collect();
            println!(
                "  SKIP {} — missing {} .o files ({})",
                group.name,
                missing.len(),
                missing.first().unwrap_or(&"?".to_string()),
            );
            continue;
        }

        println!("Linking + compressing {}...", group.name);
        create_archive(&group.name, &object_files, build_dir, is_msvc)?;

        let lib_a = build_dir.join(format!("lib{}.a", group.name));
        let lib_gz = precompiled_dir.join(format!("lib{}.a.gz", group.name));
        compress_gz(&lib_a, &lib_gz)?;
    }

    Ok(())
}

fn cmd_clean_staged(staged_dir: &Path) -> Result<()> {
    if staged_dir.exists() {
        let mut count = 0usize;
        for entry in fs::read_dir(staged_dir)?.flatten() {
            fs::remove_file(entry.path())?;
            count += 1;
        }
        println!("Removed {} files from {}", count, staged_dir.display());
    } else {
        println!("Staged directory does not exist: {}", staged_dir.display());
    }
    Ok(())
}

fn cmd_list_groups(groups: &[ArchiveGroup]) -> Result<()> {
    for group in groups {
        println!("=== {} ({} kernels) ===", group.name, group.kernels.len());
        for kernel in &group.kernels {
            println!("  {}", kernel);
        }
        println!();
    }
    Ok(())
}
