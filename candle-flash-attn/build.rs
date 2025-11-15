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
use std::path::PathBuf;
use sha2::{Sha256, Digest};
use std::fs;
use std::collections::HashMap;

const KERNEL_FILES: [&str; 33] = [
    "kernels/flash_api.cu",
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

// Header files that all kernels depend on
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

/// Compute SHA256 hash of a file
fn hash_file(path: &PathBuf) -> Result<String> {
    let contents = fs::read(path)
        .with_context(|| format!("Failed to read file: {}", path.display()))?;
    let mut hasher = Sha256::new();
    hasher.update(&contents);
    Ok(format!("{:x}", hasher.finalize()))
}

/// Compute combined hash of kernel and all its dependencies
fn compute_kernel_hash(kernel_path: &str, build_args: &[String], header_hashes: &HashMap<String, String>) -> Result<String> {
    let mut hasher = Sha256::new();
    
    // Hash the kernel source file
    let kernel_file = PathBuf::from(kernel_path);
    let kernel_hash = hash_file(&kernel_file)?;
    hasher.update(kernel_hash.as_bytes());
    
    // Hash all common header files (use cached hashes)
    for header in COMMON_HEADERS.iter() {
        if let Some(header_hash) = header_hashes.get(*header) {
            hasher.update(header_hash.as_bytes());
        }
    }
    
    // Hash build arguments (compiler flags affect output)
    for arg in build_args {
        hasher.update(arg.as_bytes());
    }
    
    Ok(format!("{:x}", hasher.finalize()))
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

fn main() -> Result<()> {
    println!("cargo:rerun-if-changed=build.rs");
    for kernel_file in KERNEL_FILES.iter() {
        println!("cargo:rerun-if-changed={kernel_file}");
    }
    for header in COMMON_HEADERS.iter() {
        println!("cargo:rerun-if-changed={header}");
    }
    
    let out_dir = PathBuf::from(std::env::var("OUT_DIR").context("OUT_DIR not set")?);
    let build_dir = match std::env::var("CANDLE_FLASH_ATTN_BUILD_DIR") {
        Err(_) => {
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

    // Create precompiled directory for storing compiled objects
    let precompiled_dir = PathBuf::from("precompiled");
    fs::create_dir_all(&precompiled_dir)
        .context("Failed to create precompiled directory")?;

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

    // Always check hashes to detect changes in source files
    // Cargo's rerun-if-changed will prevent unnecessary reruns of build.rs
    let mut kernels_to_compile = Vec::new();
    let mut kernel_hashes = HashMap::new();
    
    {
        println!("cargo:warning=Checking kernel cache validity...");
        
        // Pre-compute header hashes once (avoid re-hashing for each kernel)
        let mut header_hashes = HashMap::new();
        for header in COMMON_HEADERS.iter() {
            if let Ok(hash) = hash_file(&PathBuf::from(header)) {
                header_hashes.insert(header.to_string(), hash);
            }
        }
        
        for kernel_path in KERNEL_FILES.iter() {
            let kernel_name = PathBuf::from(kernel_path)
                .file_stem()
                .unwrap()
                .to_string_lossy()
                .to_string();
            
            let hash = compute_kernel_hash(kernel_path, &build_args, &header_hashes)?;
            kernel_hashes.insert(kernel_path.to_string(), hash.clone());
            
            if is_cache_valid(kernel_path, &precompiled_dir, &hash) {
                println!("cargo:warning=✓ Cache valid for {}", kernel_name);
            } else {
                println!("cargo:warning=⚠ Need to compile {}", kernel_name);
                kernels_to_compile.push(*kernel_path);
            }
        }
    }

    // If all kernels are cached, we can skip nvcc compilation entirely
    if kernels_to_compile.is_empty() {
        println!("cargo:warning=All kernels cached! Skipping nvcc compilation.");
        
        // Copy precompiled objects to build_dir (with clean names, no hash suffix)
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
            fs::copy(&precompiled_obj, &build_obj)
                .with_context(|| format!("Failed to copy precompiled {} to build dir", kernel_name))?;
            
            object_files.push(build_obj);
        }
        
        // Create static library from precompiled objects
        link_objects(&object_files, &build_dir, is_target_msvc)?;
    } else {
        // Compile only the kernels that need it
        println!("cargo:warning=Compiling {} kernel(s)...", kernels_to_compile.len());
        
        let kernels = kernels_to_compile.iter().collect();
        let mut builder = bindgen_cuda::Builder::default()
            .kernel_paths(kernels)
            .out_dir(build_dir.clone())
            .arg("-std=c++17")
            .arg("-O3")
            .arg("-U__CUDA_NO_HALF_OPERATORS__")
            .arg("-U__CUDA_NO_HALF_CONVERSIONS__")
            .arg("-U__CUDA_NO_HALF2_OPERATORS__")
            .arg("-U__CUDA_NO_BFLOAT16_CONVERSIONS__")
            .arg("-Icutlass/include")
            .arg("--expt-relaxed-constexpr")
            .arg("--expt-extended-lambda")
            .arg("--use_fast_math")
            .arg("--verbose");

        if is_target_msvc {
            builder = builder.arg("-D_USE_MATH_DEFINES");
        } else {
            builder = builder.arg("-Xcompiler").arg("-fPIC");
        }

        // Compile the kernels
        builder.build_lib(build_dir.join("libflashattention_partial.a"));
        
        // Copy newly compiled objects to precompiled directory and save hashes
        for kernel_path in kernels_to_compile.iter() {
            let kernel_name = PathBuf::from(kernel_path)
                .file_stem()
                .unwrap()
                .to_string_lossy()
                .to_string();
            
            // Find the compiled object file in build_dir (bindgen_cuda adds hash suffix)
            // Look for files matching pattern: kernel_name-<hash>.o
            let mut found = false;
            
            if let Ok(entries) = fs::read_dir(&build_dir) {
                for entry in entries.flatten() {
                    let file_name = entry.file_name().to_string_lossy().to_string();
                    if file_name.starts_with(&format!("{}-", kernel_name)) && file_name.ends_with(".o") {
                        // Found the compiled object, copy it with clean name
                        let compiled_obj = entry.path();
                        let precompiled_obj = precompiled_dir.join(format!("{}.o", kernel_name));
                        
                        fs::copy(&compiled_obj, &precompiled_obj)
                            .with_context(|| format!("Failed to copy {} to precompiled", kernel_name))?;
                        println!("cargo:warning=✓ Cached {}", kernel_name);
                        found = true;
                        
                        // Also copy to clean name in build_dir for linking
                        let build_obj = build_dir.join(format!("{}.o", kernel_name));
                        fs::copy(&compiled_obj, &build_obj)
                            .with_context(|| format!("Failed to copy {} to clean name", kernel_name))?;
                        break;
                    }
                }
            }
            
            if !found {
                println!("cargo:warning=⚠ Could not find compiled object for {}", kernel_name);
            }
            
            // Save hash file
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
                fs::copy(&precompiled_obj, &build_obj)
                    .with_context(|| format!("Failed to copy precompiled {} to build dir", kernel_name))?;
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
        let output = cmd.output()
            .context("Failed to run lib.exe")?;
        if !output.status.success() {
            anyhow::bail!("lib.exe failed: {}", String::from_utf8_lossy(&output.stderr));
        }
    } else {
        // Use ar on Unix-like systems
        let mut cmd = std::process::Command::new("ar");
        cmd.arg("rcs").arg(&lib_file);
        for obj in object_files {
            cmd.arg(obj);
        }
        let output = cmd.output()
            .context("Failed to run ar")?;
        if !output.status.success() {
            anyhow::bail!("ar failed: {}", String::from_utf8_lossy(&output.stderr));
        }
    }
    
    println!("cargo:warning=✓ Linked {} object files into {}", object_files.len(), lib_file.display());
    Ok(())
}
