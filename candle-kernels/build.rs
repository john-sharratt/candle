// Build script for CUDA kernel compilation with archive-level + staged .o caching.
//
// Shared utilities (constants, hashing, compilation, compression) live in
// build_utils.rs and are consumed via include!() so the kernel_tool CLI
// binary can reuse the same code.

include!("build_utils.rs");

fn main() -> Result<()> {
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=build_utils.rs");

    // Register all .cu and .cuh files in src/ for cargo:rerun-if-changed.
    // SHA256-based caching determines which actually need recompilation.
    fn register_source_files(dir: &Path) -> Result<()> {
        if !dir.exists() {
            return Ok(());
        }
        for entry in fs::read_dir(dir)? {
            let entry = entry?;
            let path = entry.path();
            if path.is_dir() {
                register_source_files(&path)?;
            } else if let Some(ext) = path.extension() {
                if ext == "cu" || ext == "cuh" {
                    let path_str = if cfg!(windows) {
                        path.to_string_lossy().replace('\\', "/")
                    } else {
                        path.to_string_lossy().to_string()
                    };
                    println!("cargo:rerun-if-changed={}", path_str);
                }
            }
        }
        Ok(())
    }

    register_source_files(Path::new("src"))?;

    // ================================================================
    // Build directory setup
    // ================================================================

    let out_dir = PathBuf::from(std::env::var("OUT_DIR").context("OUT_DIR not set")?);
    let build_dir = match std::env::var("CANDLE_KERNELS_BUILD_DIR") {
        Err(_) => out_dir.clone(),
        Ok(build_dir) => {
            let path = PathBuf::from(build_dir);
            path.canonicalize().unwrap_or_else(|_| {
                panic!(
                    "Directory doesn't exist: {} (current directory: {})",
                    &path.display(),
                    std::env::current_dir().unwrap().display()
                )
            })
        }
    };

    fs::create_dir_all(&build_dir)
        .with_context(|| format!("Failed to create build dir: {}", build_dir.display()))?;

    let precompiled_dir = PathBuf::from("precompiled");
    fs::create_dir_all(&precompiled_dir).context("Failed to create precompiled directory")?;

    let staged_dir = PathBuf::from("staged");
    fs::create_dir_all(&staged_dir).context("Failed to create staged directory")?;

    // GPU compute capability detection (for Rust-side runtime validation only)
    println!("cargo:rerun-if-env-changed=CUDA_COMPUTE_CAP");
    let compute_cap = detect_cuda_compute_cap()?;
    println!("cargo:rustc-env=CUDA_COMPUTE_CAP={compute_cap}");

    // ================================================================
    // Build archive groups and detect MSVC
    // ================================================================

    let is_target_msvc = detect_is_msvc();
    let archive_groups = build_archive_groups(is_target_msvc);
    let max_threads = 8;

    // ================================================================
    // Phase 1: Hash all groups, partition into cached vs dirty
    // ================================================================

    let mut dep_cache: HashMap<String, String> = HashMap::new();
    let base_dir = PathBuf::from(".");
    let mut all_archive_names: Vec<String> = Vec::new();

    struct DirtyGroup {
        name: String,
        kernels: Vec<String>,
        compile_args: Vec<String>,
        aggregate_hash: String,
        kernel_hashes: Vec<(String, String)>,
    }

    let mut dirty_groups: Vec<DirtyGroup> = Vec::new();

    for group in &archive_groups {
        all_archive_names.push(group.name.clone());

        let (kernel_hashes, aggregate_hash) =
            compute_group_hashes(group, &base_dir, &mut dep_cache)?;

        let build_lib = build_dir.join(format!("lib{}.a", group.name));

        if is_archive_cache_valid(&group.name, &precompiled_dir, &aggregate_hash) {
            // FAST PATH: decompress .a.gz into build_dir if needed
            let gz = precompiled_dir.join(format!("lib{}.a.gz", group.name));
            let needs_decompress = if build_lib.exists() {
                match (fs::metadata(&gz), fs::metadata(&build_lib)) {
                    (Ok(gz_meta), Ok(lib_meta)) => {
                        gz_meta
                            .modified()
                            .unwrap_or(std::time::SystemTime::UNIX_EPOCH)
                            > lib_meta
                                .modified()
                                .unwrap_or(std::time::SystemTime::UNIX_EPOCH)
                    }
                    _ => true,
                }
            } else {
                true
            };
            if needs_decompress {
                decompress_gz(&gz, &build_lib)
                    .with_context(|| format!("Failed to decompress lib{}.a", group.name))?;
            }
        } else {
            dirty_groups.push(DirtyGroup {
                name: group.name.clone(),
                kernels: group.kernels.clone(),
                compile_args: group.compile_args.clone(),
                aggregate_hash,
                kernel_hashes,
            });
        }
    }

    // ================================================================
    // Phase 2: Check staged .o cache, compile only truly dirty kernels
    // ================================================================

    if !dirty_groups.is_empty() {
        // Per-kernel hash map for staged lookup
        let mut kernel_hash_map: HashMap<String, String> = HashMap::new();
        for dg in &dirty_groups {
            for (path, hash) in &dg.kernel_hashes {
                kernel_hash_map.insert(path.clone(), hash.clone());
            }
        }

        // Partition: staged hit → skip, staged miss → compile
        let mut compile_jobs: Vec<(String, Vec<String>)> = Vec::new();
        let mut staged_hits = 0usize;

        for dg in &dirty_groups {
            for kernel_path in &dg.kernels {
                let name = kernel_stem(kernel_path);
                let current_hash = kernel_hash_map.get(kernel_path).unwrap();

                if is_staged_kernel_valid(&staged_dir, &name, current_hash) {
                    staged_hits += 1;
                } else {
                    let _ = fs::remove_file(build_dir.join(format!("{}.o", name)));
                    let _ = fs::remove_file(build_dir.join(format!("{}.obj", name)));
                    compile_jobs.push((kernel_path.clone(), dg.compile_args.clone()));
                }
            }
        }

        let total_needed: usize = dirty_groups.iter().map(|dg| dg.kernels.len()).sum();
        let to_compile = compile_jobs.len();
        if staged_hits > 0 {
            eprintln!(
                "  Staged cache: {}/{} kernels cached, {} to compile",
                staged_hits, total_needed, to_compile,
            );
        }

        if !compile_jobs.is_empty() {
            // Group by compile_args for batched compilation
            let mut batches: HashMap<Vec<String>, Vec<String>> = HashMap::new();
            for (kernel_path, args) in &compile_jobs {
                batches
                    .entry(args.clone())
                    .or_default()
                    .push(kernel_path.clone());
            }

            let total_archives = dirty_groups.len();
            eprintln!(
                "  Compiling {} kernel(s) across {} archive(s) ({} threads)...",
                to_compile, total_archives, max_threads,
            );

            let compile_start = std::time::Instant::now();

            for (args, kernels) in &batches {
                let kernel_refs: Vec<&str> = kernels.iter().map(|s| s.as_str()).collect();
                compile_kernels_parallel(&kernel_refs, &build_dir, args, max_threads)?;
            }

            let compile_elapsed = compile_start.elapsed();
            eprintln!(
                "  Compilation finished in {:.1}s",
                compile_elapsed.as_secs_f64()
            );

            // Save newly compiled .o files to staged cache
            for (kernel_path, _) in &compile_jobs {
                let name = kernel_stem(kernel_path);
                if let Some(hash) = kernel_hash_map.get(kernel_path) {
                    let _ = save_to_staged_cache(&build_dir, &staged_dir, &name, hash);
                }
            }
        } else {
            eprintln!(
                "  All {} kernels found in staged cache, skipping compilation",
                staged_hits
            );
        }

        // ================================================================
        // Phase 3: Link from staged/ + compress into precompiled/
        // ================================================================
        // staged/ is the source of truth for .o files.
        // All kernels are now in staged/ (either from cache hit or just compiled+saved).

        let link_start = std::time::Instant::now();

        for dg in &dirty_groups {
            let object_files: Vec<PathBuf> = dg
                .kernels
                .iter()
                .map(|k| staged_dir.join(format!("{}.o", kernel_stem(k))))
                .collect();
            create_archive(&dg.name, &object_files, &build_dir, is_target_msvc)?;

            let build_lib = build_dir.join(format!("lib{}.a", dg.name));
            compress_gz(
                &build_lib,
                &precompiled_dir.join(format!("lib{}.a.gz", dg.name)),
            )
            .with_context(|| format!("Failed to compress lib{}.a", dg.name))?;

            save_archive_hash(&dg.name, &dg.aggregate_hash, &precompiled_dir)?;
        }

        let link_elapsed = link_start.elapsed();
        eprintln!(
            "  Link + compress finished in {:.1}s",
            link_elapsed.as_secs_f64()
        );
    }

    // ================================================================
    // Link directives
    // ================================================================

    println!("cargo:rustc-link-search={}", build_dir.display());

    if let Ok(cuda_path) = std::env::var("CUDA_PATH") {
        let cuda_lib_dir = if cfg!(target_os = "windows") {
            PathBuf::from(&cuda_path).join("lib").join("x64")
        } else {
            PathBuf::from(&cuda_path).join("lib64")
        };
        if cuda_lib_dir.exists() {
            println!("cargo:rustc-link-search=native={}", cuda_lib_dir.display());
        }
    }

    for name in &all_archive_names {
        println!("cargo:rustc-link-lib=static={}", name);
    }

    println!("cargo:rustc-link-lib=dylib=cudart");
    if !is_target_msvc {
        println!("cargo:rustc-link-lib=dylib=stdc++");
    }

    // Register .a files in build_dir so downstream crates re-link when they change
    for entry in fs::read_dir(&build_dir)?.flatten() {
        let path = entry.path();
        if let Some(ext) = path.extension() {
            if ext == "a" {
                let path_str = if cfg!(windows) {
                    path.to_string_lossy().replace('\\', "/")
                } else {
                    path.to_string_lossy().to_string()
                };
                println!("cargo:rerun-if-changed={}", path_str);
            }
        }
    }

    for entry in fs::read_dir(&precompiled_dir)?.flatten() {
        let path = entry.path();
        if let Some(ext) = path.extension() {
            if ext == "gz" || ext == "sha256" {
                let path_str = if cfg!(windows) {
                    path.to_string_lossy().replace('\\', "/")
                } else {
                    path.to_string_lossy().to_string()
                };
                println!("cargo:rerun-if-changed={}", path_str);
            }
        }
    }

    Ok(())
}
