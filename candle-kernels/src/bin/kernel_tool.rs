// CLI tool for candle-kernels build cache management.
//
// Shares all hashing/compilation/compression logic with build.rs via include!().
// Run from the candle-kernels directory:
//
//   cargo run --bin kernel_tool -- status
//   cargo run --bin kernel_tool -- hash
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
        Commands::Hash { group } => cmd_hash(&archive_groups, &base_dir, &group),
        Commands::CheckForChanges => {
            cmd_check_for_changes(&archive_groups, &precompiled_dir, &staged_dir, &base_dir)
        }
        Commands::RebuildArchives { group } => cmd_rebuild_archives(
            &archive_groups,
            &staged_dir,
            &precompiled_dir,
            &base_dir,
            &group,
            is_msvc,
        ),
        Commands::Compile { group, threads } => {
            cmd_compile(&archive_groups, &staged_dir, &base_dir, &group, threads)
        }
        Commands::Recompile { group, threads } => {
            cmd_recompile(&archive_groups, &staged_dir, &base_dir, &group, threads)
        }
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

    println!(
        "{:<20} {:>8}  {:>8}  {:>8}  {}",
        "GROUP", "KERNELS", "CACHED", "STAGED", "STATUS"
    );
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
        MissingO,  // .o file not in staged/
        StaleHash, // .o exists but .sha256 is wrong or missing, or the .o predates its sources
    }

    let mut problems: Vec<StagingProblem> = Vec::new();

    for group in groups {
        let (kernel_hashes, aggregate_hash) =
            compute_group_hashes(group, base_dir, &mut dep_cache)?;
        let is_cached = is_archive_cache_valid(
            &group.name,
            precompiled_dir,
            &aggregate_hash,
            &group_sources(group, base_dir)?,
        ) && staged_objects_are_fresh(group, &kernel_hashes, staged_dir, base_dir)?;

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

            if !staged_o.exists() {
                problems.push(StagingProblem {
                    kernel_path: kernel_path.clone(),
                    kernel_name: name,
                    group_name: group.name.clone(),
                    kind: StagingProblemKind::MissingO,
                });
            } else {
                let sources = kernel_sources(kernel_path, base_dir, &group.include_dirs)?;
                if is_staged_kernel_valid(staged_dir, &name, expected_hash, &sources) {
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
            println!("\n  Compile just the affected group(s):");
            let mut affected_groups: Vec<&str> =
                missing.iter().map(|p| p.group_name.as_str()).collect();
            affected_groups.sort();
            affected_groups.dedup();
            for g in &affected_groups {
                println!("    cargo run --bin kernel_tool -- compile --group {}", g);
            }
            println!();
        }

        if !stale.is_empty() {
            println!(
                "Stale staged .o files ({}) — hash mismatch, or older than a source they \
                 were compiled from:",
                stale.len()
            );
            for p in &stale {
                println!("  [{}] {} ({})", p.group_name, p.kernel_name, p.kernel_path);
            }
            println!("\n  A stale object has to be recompiled. Re-stamping its hash does not");
            println!("  repair it: the label changes, the kernel does not, and the freshness");
            println!("  check still rejects it.");
            println!("\n  Recompile just the affected group(s):");
            let mut affected_groups: Vec<&str> =
                stale.iter().map(|p| p.group_name.as_str()).collect();
            affected_groups.sort();
            affected_groups.dedup();
            for g in &affected_groups {
                println!("    cargo run --bin kernel_tool -- compile --group {}", g);
            }
            println!();
        }
    }

    Ok(())
}

fn cmd_hash(groups: &[ArchiveGroup], base_dir: &Path, filter_group: &[String]) -> Result<()> {
    let mut dep_cache: HashMap<String, String> = HashMap::new();

    for group in groups {
        if !filter_group.is_empty() && !filter_group.iter().any(|f| f == &group.name) {
            continue;
        }

        let (kernel_hashes, aggregate_hash) =
            compute_group_hashes(group, base_dir, &mut dep_cache)?;

        println!(
            "=== {} (aggregate: {}) ===",
            group.name,
            &aggregate_hash[..16]
        );
        for (path, hash) in &kernel_hashes {
            println!("  {} {}", &hash[..16], path);
        }
        println!();
    }

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

        if is_archive_cache_valid(
            &group.name,
            precompiled_dir,
            &aggregate_hash,
            &group_sources(group, base_dir)?,
        ) && staged_objects_are_fresh(group, &kernel_hashes, staged_dir, base_dir)?
        {
            continue; // Archive is up-to-date
        }

        any_dirty = true;
        println!(
            "DIRTY archive: {} ({} kernels)",
            group.name,
            group.kernels.len()
        );

        let mut staged_hits = 0usize;
        let mut to_compile: Vec<String> = Vec::new();

        for kernel_path in &group.kernels {
            let name = kernel_stem(kernel_path);
            let current_hash = kernel_hashes
                .iter()
                .find(|(p, _)| p == kernel_path)
                .map(|(_, h)| h.as_str())
                .unwrap_or("");

            let sources = kernel_sources(kernel_path, base_dir, &group.include_dirs)?;
            if is_staged_kernel_valid(staged_dir, &name, current_hash, &sources) {
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
            println!("  {} kernel(s) would need compilation:", to_compile.len());
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
                fs::copy(&staged_o, &dest_o)
                    .with_context(|| format!("Failed to copy staged {}.o", name))?;
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

            let sources = kernel_sources(kernel_path, base_dir, &group.include_dirs)?;
            if is_staged_kernel_valid(staged_dir, &name, &hash, &sources) {
                total_skipped += 1;
            } else {
                to_compile.push(kernel_path);
            }
        }

        if to_compile.is_empty() {
            println!(
                "{}: all {} kernels up-to-date in staged/",
                group.name,
                group.kernels.len()
            );
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
                let sources = kernel_sources(kernel_path, base_dir, &group.include_dirs)?;
                save_to_staged_cache(&compile_dir, staged_dir, &name, hash, &sources)?;
            }
        }

        total_compiled += to_compile.len();
        println!("  Done in {:.1}s", start.elapsed().as_secs_f64());
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
            let sources = kernel_sources(kernel_path, base_dir, &group.include_dirs)?;
            save_to_staged_cache(&compile_dir, staged_dir, &name, &hash, &sources)?;
        }

        println!("  Done in {:.1}s", start.elapsed().as_secs_f64());
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
