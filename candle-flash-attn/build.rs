// Build script to run nvcc and generate the C glue code for launching the flash-attention kernel.
// The cuda build time is very long so one can set the CANDLE_FLASH_ATTN_BUILD_DIR environment
// variable in order to cache the compiled artifacts and avoid recompiling too often.
//
// This build script implements intelligent caching:
// - Computes SHA256 hashes of each kernel and its dependencies
// - Stores precompiled .o files in precompiled/ directory
// - Stores .sha256 files in kernels/ directory alongside source
// - Only recompiles kernels whose dependencies have changed
use anyhow::{Context, Result};
use sha2::{Digest, Sha256};
use std::collections::HashMap;
use std::fs;
use std::path::PathBuf;

// Full specialization build.
// (You can run/abort compilation from `cargo` as desired.)
const KERNEL_FILES: [&str; 34] = [
    // Flash-attn API + kernels.
    "kernels/flash_api.cu",
    // Simple batched sampling (argmax, temperature/top-k/top-p)
    "kernels/batched_sampling_simple.cu",
    "kernels/flash_fwd_hdim128_fp16_sm80.cu",
    "kernels/flash_fwd_hdim160_fp16_sm80.cu",
    "kernels/flash_fwd_hdim192_fp16_sm80.cu",
    "kernels/flash_fwd_hdim224_fp16_sm80.cu",
    "kernels/flash_fwd_hdim256_fp16_sm80.cu",
    "kernels/flash_fwd_hdim32_fp16_sm80.cu",
    "kernels/flash_fwd_hdim64_fp16_sm80.cu",
    "kernels/flash_fwd_hdim96_fp16_sm80.cu",
    "kernels/flash_fwd_hdim128_bf16_sm80.cu",
    "kernels/flash_fwd_hdim160_bf16_sm80.cu",
    "kernels/flash_fwd_hdim192_bf16_sm80.cu",
    "kernels/flash_fwd_hdim224_bf16_sm80.cu",
    "kernels/flash_fwd_hdim256_bf16_sm80.cu",
    "kernels/flash_fwd_hdim32_bf16_sm80.cu",
    "kernels/flash_fwd_hdim64_bf16_sm80.cu",
    "kernels/flash_fwd_hdim96_bf16_sm80.cu",
    "kernels/flash_fwd_hdim128_fp16_causal_sm80.cu",
    "kernels/flash_fwd_hdim160_fp16_causal_sm80.cu",
    "kernels/flash_fwd_hdim192_fp16_causal_sm80.cu",
    "kernels/flash_fwd_hdim224_fp16_causal_sm80.cu",
    "kernels/flash_fwd_hdim256_fp16_causal_sm80.cu",
    "kernels/flash_fwd_hdim32_fp16_causal_sm80.cu",
    "kernels/flash_fwd_hdim64_fp16_causal_sm80.cu",
    "kernels/flash_fwd_hdim96_fp16_causal_sm80.cu",
    "kernels/flash_fwd_hdim128_bf16_causal_sm80.cu",
    "kernels/flash_fwd_hdim160_bf16_causal_sm80.cu",
    "kernels/flash_fwd_hdim192_bf16_causal_sm80.cu",
    "kernels/flash_fwd_hdim224_bf16_causal_sm80.cu",
    "kernels/flash_fwd_hdim256_bf16_causal_sm80.cu",
    "kernels/flash_fwd_hdim32_bf16_causal_sm80.cu",
    "kernels/flash_fwd_hdim64_bf16_causal_sm80.cu",
    "kernels/flash_fwd_hdim96_bf16_causal_sm80.cu",
];

// Header files that all kernels depend on (kept for backwards-compatible hashing).
const COMMON_HEADERS: [&str; 14] = [
    "kernels/flash_fwd_kernel.h",
    "kernels/flash_fwd_launch_template.h",
    "kernels/flash.h",
    "kernels/philox.cuh",
    "kernels/softmax.h",
    "kernels/utils.h",
    "kernels/kernel_traits.h",
    "kernels/block_info.h",
    "kernels/static_switch.h",
    "kernels/hardware_info.h",
    "kernels/error.h",
    "kernels/kernels.h",
    "kernels/kernel_helpers.h",
    "kernels/mask.h",
];

// Header files for paged-prefill kernels.
// Keep this list minimal so edits don't trigger flash kernel rebuilds.
const PAGED_PREFILL_HEADERS: [&str; 1] = ["kernels/kv_quantize.cuh"];

// Header files for paged-decode kernels.
const PAGED_DECODE_HEADERS: [&str; 1] = ["kernels/kv_quantize.cuh"];

// Header files for batched sampling kernel.
const BATCHED_SAMPLING_HEADERS: [&str; 2] = [
    "kernels/philox.cuh",
    "kernels/fast_exp.cuh", // For fast exponential
];

/// Compute SHA256 hash of a file
fn hash_file(path: &PathBuf) -> Result<String> {
    let contents =
        fs::read(path).with_context(|| format!("Failed to read file: {}", path.display()))?;
    let mut hasher = Sha256::new();
    hasher.update(&contents);
    Ok(format!("{:x}", hasher.finalize()))
}

/// Compute combined hash of kernel and all its dependencies
fn compute_kernel_hash(
    kernel_path: &str,
    build_args: &[String],
    header_hashes: &HashMap<String, String>,
) -> Result<String> {
    let mut hasher = Sha256::new();

    // Hash the kernel source file
    let kernel_file = PathBuf::from(kernel_path);
    let kernel_hash = hash_file(&kernel_file)?;
    hasher.update(kernel_hash.as_bytes());

    // Hash common headers for all kernels (backwards-compatible with previous behavior).
    for header in COMMON_HEADERS.iter() {
        if let Some(header_hash) = header_hashes.get(*header) {
            hasher.update(header_hash.as_bytes());
        }
    }

    // Paged-prefill kernels depend on the shared implementation header.
    // We also include the dispatcher TU (`paged_prefill_api.cu`) so that any ABI/dispatch changes
    // coupled to the shared header force a rebuild.
    if kernel_path.ends_with("paged_prefill_api.cu") || kernel_path.contains("paged_prefill_hdim") {
        for header in PAGED_PREFILL_HEADERS.iter() {
            if let Some(header_hash) = header_hashes.get(*header) {
                hasher.update(header_hash.as_bytes());
            }
        }
    }

    // Paged-decode kernels depend on the shared implementation header.
    if kernel_path.ends_with("paged_decode_api.cu") || kernel_path.contains("paged_decode_hdim") {
        for header in PAGED_DECODE_HEADERS.iter() {
            if let Some(header_hash) = header_hashes.get(*header) {
                hasher.update(header_hash.as_bytes());
            }
        }
    }

    // Batched sampling kernels depend on their header.
    if kernel_path.contains("batched_sampling") {
        for header in BATCHED_SAMPLING_HEADERS.iter() {
            if let Some(header_hash) = header_hashes.get(*header) {
                hasher.update(header_hash.as_bytes());
            }
        }
    }

    // Hash build arguments (compiler flags affect output)
    // Use canonical args to exclude platform-specific flags
    let canonical = canonical_args_for_hash(build_args);
    for arg in &canonical {
        hasher.update(arg.as_bytes());
    }

    Ok(format!("{:x}", hasher.finalize()))
}

/// Filter out platform-specific build args that don't affect CUDA device code output.
/// This ensures the same hash is computed on Windows (MSVC) and Linux, so precompiled
/// objects can be shared across platforms.
fn canonical_args_for_hash(args: &[String]) -> Vec<String> {
    args.iter()
        .filter(|a| {
            !matches!(
                a.as_str(),
                "-D_USE_MATH_DEFINES" | "-Xcompiler" | "-fPIC"
            )
        })
        .cloned()
        .collect()
}

/// Check if cached object is valid by comparing hashes
fn is_cache_valid(kernel_path: &str, precompiled_dir: &PathBuf, current_hash: &str) -> bool {
    // Check if .sha256 file exists in precompiled/ directory
    let kernel_name = PathBuf::from(kernel_path)
        .file_stem()
        .unwrap()
        .to_string_lossy()
        .to_string();

    // Hash file is stored in precompiled/ directory
    let hash_file = precompiled_dir.join(format!("{}.sha256", kernel_name));

    // Check if precompiled .o file exists
    let obj_file = precompiled_dir.join(format!("{}.o", kernel_name));

    if !hash_file.exists() || !obj_file.exists() {
        return false;
    }

    // Compare stored hash with current hash
    if let Ok(stored_hash) = fs::read_to_string(&hash_file) {
        stored_hash.trim() == current_hash
    } else {
        false
    }
}

/// Save hash file for a kernel
fn save_hash(kernel_path: &str, hash: &str, precompiled_dir: &PathBuf) -> Result<()> {
    let kernel_name = PathBuf::from(kernel_path)
        .file_stem()
        .unwrap()
        .to_string_lossy()
        .to_string();

    // Hash file is stored in precompiled/ directory
    let hash_file = precompiled_dir.join(format!("{}.sha256", kernel_name));
    fs::write(&hash_file, hash)
        .with_context(|| format!("Failed to write hash file: {}", hash_file.display()))?;
    Ok(())
}

fn cleanup_stale_kernel_outputs(build_dir: &PathBuf, kernel_name: &str) -> Result<()> {
    // bindgen_cuda emits hash-suffixed artifacts like `<kernel>-<hash>.o` into out_dir.
    // If we previously compiled another git commit, these stale artifacts can remain and be
    // accidentally picked up as the “compiled” output even when a rebuild is required.
    //
    // We only delete files that are clearly owned by this kernel to avoid triggering
    // broad rebuilds.
    let clean_obj = build_dir.join(format!("{}.o", kernel_name));
    let _ = fs::remove_file(&clean_obj);

    // On Windows/MSVC nvcc may emit `.obj`.
    let clean_obj_msvc = build_dir.join(format!("{}.obj", kernel_name));
    let _ = fs::remove_file(&clean_obj_msvc);

    let Ok(entries) = fs::read_dir(build_dir) else {
        return Ok(());
    };

    let prefix = format!("{}-", kernel_name);
    for entry in entries.flatten() {
        let path = entry.path();
        let Some(file_name) = path.file_name().and_then(|n| n.to_str()) else {
            continue;
        };

        if file_name.starts_with(&prefix) {
            let _ = fs::remove_file(&path);
        }
    }

    Ok(())
}

fn detect_cuda_compute_cap() -> Result<u32> {
    if let Ok(v) = std::env::var("CUDA_COMPUTE_CAP") {
        if let Ok(cap) = v.trim().parse::<u32>() {
            if cap >= 50 && cap <= 999 {
                return Ok(cap);
            }
        }
    }

    // Best-effort detection via nvidia-smi.
    // Output is typically like: "8.9".
    if let Ok(output) = std::process::Command::new("nvidia-smi")
        .args(["--query-gpu=compute_cap", "--format=csv,noheader"])
        .output()
    {
        if output.status.success() {
            let s = String::from_utf8_lossy(&output.stdout);
            if let Some(first_line) = s.lines().next() {
                let tok = first_line.trim();
                if let Some((maj, min)) = tok.split_once('.') {
                    if let (Ok(maj), Ok(min)) =
                        (maj.trim().parse::<u32>(), min.trim().parse::<u32>())
                    {
                        return Ok(maj * 10 + min);
                    }
                }
                if let Ok(cap) = tok.parse::<u32>() {
                    return Ok(cap);
                }
            }
        }
    }

    // Fallback: sm80 is a reasonable baseline for modern GPUs.
    Ok(80)
}

fn compile_kernel_nvcc(
    kernel_path: &str,
    kernel_name: &str,
    build_dir: &PathBuf,
    compute_cap: u32,
    build_args: &[String],
) -> Result<PathBuf> {
    let out_obj = build_dir.join(format!("{}.o", kernel_name));

    let mut cmd = std::process::Command::new("nvcc");
    cmd.arg("-c")
        .arg(kernel_path)
        .arg("-o")
        .arg(&out_obj)
        .arg(format!(
            "-gencode=arch=compute_{cap},code=sm_{cap}",
            cap = compute_cap
        ));

    for arg in build_args {
        cmd.arg(arg);
    }

    let output = cmd
        .output()
        .with_context(|| format!("Failed to run nvcc for {kernel_name}"))?;
    if !output.status.success() {
        anyhow::bail!(
            "nvcc failed for {kernel_name}:\n{}\n{}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        );
    }

    if !out_obj.exists() {
        anyhow::bail!(
            "nvcc reported success but did not produce output: {}",
            out_obj.display()
        );
    }

    Ok(out_obj)
}

fn main() -> Result<()> {
    println!("cargo:rerun-if-changed=build.rs");
    for kernel_file in KERNEL_FILES.iter() {
        println!("cargo:rerun-if-changed={kernel_file}");
    }
    for header in COMMON_HEADERS.iter() {
        println!("cargo:rerun-if-changed={header}");
    }
    for header in PAGED_PREFILL_HEADERS.iter() {
        println!("cargo:rerun-if-changed={header}");
    }
    for header in PAGED_DECODE_HEADERS.iter() {
        println!("cargo:rerun-if-changed={header}");
    }
    for header in BATCHED_SAMPLING_HEADERS.iter() {
        println!("cargo:rerun-if-changed={header}");
    }

    let out_dir = PathBuf::from(std::env::var("OUT_DIR").context("OUT_DIR not set")?);
    let build_dir = match std::env::var("CANDLE_FLASH_ATTN_BUILD_DIR") {
        Err(_) =>
        {
            #[allow(clippy::redundant_clone)]
            out_dir.clone()
        }
        Ok(build_dir) => {
            let path = PathBuf::from(build_dir);
            path.canonicalize().expect(&format!(
                "Directory doesn't exists: {} (the current directory is {})",
                &path.display(),
                std::env::current_dir()?.display()
            ))
        }
    };

    // Cargo may provide an OUT_DIR path that is not yet created.
    fs::create_dir_all(&build_dir)
        .with_context(|| format!("Failed to create build dir: {}", build_dir.display()))?;

    // Create precompiled directory for storing compiled objects
    let precompiled_dir = PathBuf::from("precompiled");
    fs::create_dir_all(&precompiled_dir).context("Failed to create precompiled directory")?;

    // Collect build arguments for hashing
    let mut build_args = vec![
        "-std=c++17".to_string(),
        "-O3".to_string(),
        "-U__CUDA_NO_HALF_OPERATORS__".to_string(),
        "-U__CUDA_NO_HALF_CONVERSIONS__".to_string(),
        "-U__CUDA_NO_HALF2_OPERATORS__".to_string(),
        "-U__CUDA_NO_BFLOAT16_CONVERSIONS__".to_string(),
        "-Icutlass/include".to_string(),
        "--expt-relaxed-constexpr".to_string(),
        "--expt-extended-lambda".to_string(),
        "--use_fast_math".to_string(),
    ];

    // Extra optimization flags for paged kernels
    // Common flags for all paged kernels
    let paged_common_args = vec![
        "-Xptxas=-O3".to_string(), // PTX assembler max optimization
        "--extra-device-vectorization".to_string(), // Better half2/float4 vectorization
    ];
    // paged_prefill: limit registers to 128 for better occupancy
    let paged_prefill_extra_args: Vec<String> = paged_common_args
        .iter()
        .cloned()
        .chain(std::iter::once("-maxrregcount=128".to_string()))
        .collect();
    // paged_decode: limit registers to 96 (decode is more occupancy-sensitive)
    let paged_decode_extra_args: Vec<String> = paged_common_args
        .iter()
        .cloned()
        .chain(std::iter::once("-maxrregcount=96".to_string()))
        .collect();

    let mut is_target_msvc = false;
    if let Ok(target) = std::env::var("TARGET") {
        if target.contains("msvc") {
            is_target_msvc = true;
            build_args.push("-D_USE_MATH_DEFINES".to_string());
        } else {
            build_args.push("-Xcompiler".to_string());
            build_args.push("-fPIC".to_string());
        }
    }

    // Compute capability controls codegen and performance. Keep it explicit so rebuilds are deterministic.
    println!("cargo:rerun-if-env-changed=CUDA_COMPUTE_CAP");
    let compute_cap = detect_cuda_compute_cap()?;
    println!("cargo:rustc-env=CUDA_COMPUTE_CAP={compute_cap}");

    // Always check hashes to detect changes in source files
    // Cargo's rerun-if-changed will prevent unnecessary reruns of build.rs
    let mut kernels_to_compile = Vec::new();
    let mut kernel_hashes = HashMap::new();

    {
        // Pre-compute header hashes once (avoid re-hashing for each kernel)
        let mut header_hashes = HashMap::new();
        for header in COMMON_HEADERS.iter() {
            if let Ok(hash) = hash_file(&PathBuf::from(header)) {
                header_hashes.insert(header.to_string(), hash);
            }
        }
        for header in PAGED_PREFILL_HEADERS.iter() {
            if let Ok(hash) = hash_file(&PathBuf::from(header)) {
                header_hashes.insert(header.to_string(), hash);
            }
        }
        for header in PAGED_DECODE_HEADERS.iter() {
            if let Ok(hash) = hash_file(&PathBuf::from(header)) {
                header_hashes.insert(header.to_string(), hash);
            }
        }
        for header in BATCHED_SAMPLING_HEADERS.iter() {
            if let Ok(hash) = hash_file(&PathBuf::from(header)) {
                header_hashes.insert(header.to_string(), hash);
            }
        }

        for kernel_path in KERNEL_FILES.iter() {
            // Use extra args for paged kernels (paged_prefill, paged_decode, etc.)
            let effective_args: Vec<String> = if kernel_path.contains("paged_prefill") {
                build_args
                    .iter()
                    .chain(paged_prefill_extra_args.iter())
                    .cloned()
                    .collect()
            } else if kernel_path.contains("paged_decode") {
                build_args
                    .iter()
                    .chain(paged_decode_extra_args.iter())
                    .cloned()
                    .collect()
            } else {
                build_args.clone()
            };

            let hash = compute_kernel_hash(kernel_path, &effective_args, &header_hashes)?;
            kernel_hashes.insert(kernel_path.to_string(), hash.clone());

            if !is_cache_valid(kernel_path, &precompiled_dir, &hash) {
                kernels_to_compile.push(*kernel_path);
            }
        }
    }

    // If all kernels are cached, we can skip nvcc compilation entirely
    if kernels_to_compile.is_empty() {
        // Check if .a file already exists in build_dir
        let build_lib = build_dir.join("libflashattention.a");

        if !build_lib.exists() {
            // .a file doesn't exist - need to copy .o files and link
            let mut object_files = Vec::new();
            for kernel_path in KERNEL_FILES.iter() {
                let kernel_name = PathBuf::from(kernel_path)
                    .file_stem()
                    .unwrap()
                    .to_string_lossy()
                    .to_string();
                let precompiled_obj = precompiled_dir.join(format!("{}.o", kernel_name));
                let build_obj = build_dir.join(format!("{}.o", kernel_name));

                // Copy from precompiled to build_dir
                fs::copy(&precompiled_obj, &build_obj).with_context(|| {
                    format!("Failed to copy precompiled {} to build dir", kernel_name)
                })?;

                object_files.push(build_obj);
            }

            // Create static library from precompiled objects
            link_objects(&object_files, &build_dir, is_target_msvc)?;
        }
    } else {
        // Compile only the kernels that need it
        println!(
            "cargo:warning=Compiling {} kernel(s)...",
            kernels_to_compile.len()
        );

        // Ensure we don't accidentally re-cache stale objects from previous runs.
        for kernel_path in kernels_to_compile.iter() {
            let kernel_name = PathBuf::from(kernel_path)
                .file_stem()
                .unwrap()
                .to_string_lossy()
                .to_string();
            cleanup_stale_kernel_outputs(&build_dir, &kernel_name)?;
        }

        // Compile kernels directly to deterministic object file names in build_dir.
        for kernel_path in kernels_to_compile.iter() {
            let kernel_name = PathBuf::from(kernel_path)
                .file_stem()
                .unwrap()
                .to_string_lossy()
                .to_string();

            // Use extra args for paged kernels (paged_prefill, paged_decode, etc.)
            let effective_args: Vec<String> = if kernel_path.contains("paged_prefill") {
                build_args
                    .iter()
                    .chain(paged_prefill_extra_args.iter())
                    .cloned()
                    .collect()
            } else if kernel_path.contains("paged_decode") {
                build_args
                    .iter()
                    .chain(paged_decode_extra_args.iter())
                    .cloned()
                    .collect()
            } else {
                build_args.clone()
            };

            let compiled_obj = compile_kernel_nvcc(
                kernel_path,
                &kernel_name,
                &build_dir,
                compute_cap,
                &effective_args,
            )?;

            let precompiled_obj = precompiled_dir.join(format!("{}.o", kernel_name));
            fs::copy(&compiled_obj, &precompiled_obj)
                .with_context(|| format!("Failed to copy {} to precompiled", kernel_name))?;

            // Also copy to clean name in build_dir for linking
            let build_obj = build_dir.join(format!("{}.o", kernel_name));
            if compiled_obj != build_obj {
                fs::copy(&compiled_obj, &build_obj)
                    .with_context(|| format!("Failed to copy {} to clean name", kernel_name))?;
            }

            eprintln!("  Compiled {}", kernel_name);

            // Save hash file (only after we have a compiled object)
            if let Some(hash) = kernel_hashes.get(&kernel_path.to_string()) {
                save_hash(kernel_path, hash, &precompiled_dir)?;
            }
        }

        // Now link all objects (both cached and newly compiled)
        // Copy all precompiled objects to build_dir
        let mut object_files = Vec::new();
        for kernel_path in KERNEL_FILES.iter() {
            let kernel_name = PathBuf::from(kernel_path)
                .file_stem()
                .unwrap()
                .to_string_lossy()
                .to_string();
            let precompiled_obj = precompiled_dir.join(format!("{}.o", kernel_name));
            let build_obj = build_dir.join(format!("{}.o", kernel_name));

            // Copy from precompiled to build_dir if not just compiled
            if !kernels_to_compile.contains(&kernel_path) {
                fs::copy(&precompiled_obj, &build_obj).with_context(|| {
                    format!("Failed to copy precompiled {} to build dir", kernel_name)
                })?;
            }

            object_files.push(build_obj);
        }

        link_objects(&object_files, &build_dir, is_target_msvc)?;
    }

    println!("cargo:rustc-link-search={}", build_dir.display());
    println!("cargo:rustc-link-lib=flashattention");
    println!("cargo:rustc-link-lib=dylib=cudart");
    if !is_target_msvc {
        println!("cargo:rustc-link-lib=dylib=stdc++");
    }
    Ok(())
}

/// Link object files into static library
fn link_objects(object_files: &[PathBuf], build_dir: &PathBuf, is_msvc: bool) -> Result<()> {
    let lib_file = build_dir.join("libflashattention.a");

    if is_msvc {
        // Use lib.exe on Windows
        let mut cmd = std::process::Command::new("lib.exe");
        cmd.arg(format!("/OUT:{}", lib_file.display()));
        for obj in object_files {
            cmd.arg(obj);
        }
        let output = cmd.output().context("Failed to run lib.exe")?;
        if !output.status.success() {
            anyhow::bail!(
                "lib.exe failed: {}",
                String::from_utf8_lossy(&output.stderr)
            );
        }
    } else {
        // Use ar on Unix-like systems
        let mut cmd = std::process::Command::new("ar");
        cmd.arg("rcs").arg(&lib_file);
        for obj in object_files {
            cmd.arg(obj);
        }
        let output = cmd.output().context("Failed to run ar")?;
        if !output.status.success() {
            anyhow::bail!("ar failed: {}", String::from_utf8_lossy(&output.stderr));
        }
    }

    println!(
        "cargo:warning=✓ Linked {} object files into {}",
        object_files.len(),
        lib_file.display()
    );
    Ok(())
}
