// Shared utilities for CUDA kernel compilation, caching, and archive management.
// Used by both build.rs (cargo build script) and kernel_tool (CLI binary).
//
// This file is consumed via include!() — it is NOT a crate module.
// Both consumers must have anyhow, flate2, sha2 in their dependency graph.

use anyhow::{Context, Result};
use flate2::read::GzDecoder;
use flate2::write::GzEncoder;
use flate2::Compression;
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, HashMap, HashSet};
use std::fs;
use std::io::{Read, Write};
use std::path::{Path, PathBuf};

// ============================================================================
// Kernel lists
// ============================================================================

// Simple CUDA kernels: basic operations, indexed operations, and simple transformations
// NOTE: Only include properly formed dispatcher files that use <<<...>>> kernel launch syntax.
// (`src/simple/quantized_dispatcher.cu` — the seal-time quantize/select
// kernels — compiles in its own group under the bit-exact mirror contract
// flags; see the `quantize_dispatch` group below.)
const SIMPLE_KERNELS: [&str; 51] = [
    "src/api.cu", // FFI wrapper functions for all simple kernels
    "src/simple/nvtx.cu",
    "src/simple/corpus_gather.cu",
    // Kernel implementations
    "src/simple/add_at_indices.cu",
    "src/simple/router_topk.cu",
    "src/simple/gather_r16_kv.cu",
    "src/simple/gather_rows.cu",
    "src/simple/prov_sign_pack.cu",
    "src/simple/kv_migrate.cu",
    "src/simple/fletcher32.cu",
    "src/simple/moe_bucketize.cu",
    "src/simple/sinkhorn.cu",
    "src/simple/hyper_mhc.cu",
    "src/simple/compressor_pool.cu",
    "src/simple/indexer_score.cu",
    "src/simple/rows_scatter.cu",
    "src/simple/comp_idx.cu",
    "src/simple/bdp.cu",
    "src/simple/moe_scatter.cu",
    "src/simple/affine.cu",
    "src/simple/binary.cu",
    "src/simple/cast.cu",
    "src/simple/cast_dispatcher.cu",
    "src/simple/conv.cu",
    "src/simple/div_at_indices.cu",
    "src/simple/fast_exp.cu",
    "src/simple/fill.cu",
    "src/simple/indexing.cu",
    "src/simple/mul_at_indices.cu",
    "src/simple/multinomial.cu",
    "src/simple/quantized.cu",
    "src/simple/reduce.cu",
    "src/simple/repeat_penalty.cu",
    "src/simple/sort.cu",
    "src/simple/sub_at_indices.cu",
    "src/simple/sub_at_indices_with_values.cu",
    "src/simple/ternary.cu",
    "src/simple/unary.cu",
    "src/simple/fused_silu_mul.cu",
    // Properly formed dispatcher files with <<<...>>> kernel launch syntax
    "src/simple/affine_dispatcher.cu",
    "src/simple/conv_dispatcher.cu",
    "src/simple/fill_dispatcher.cu",
    "src/simple/multinomial_dispatcher.cu",
    "src/simple/reduce_dispatcher.cu",
    "src/simple/repeat_penalty_dispatcher.cu",
    "src/simple/scatter_op_dispatcher.cu",
    "src/simple/sort_dispatcher.cu",
    // Quantized batched matmul dispatcher
    "src/quantized/dispatcher.cu",
    // Weight repacking for GEMX format
    "src/quantized/repack_gemx.cu",
    // Dequantize repacked tensors (for debugging)
    "src/quantized/dequantize.cu",
    // GPU-side quantize kernels (for KV cache quantization)
    "src/quantize/quantize_kernels.cu",
    // Note: GEMX kernels are in impl/q4_0_f16.cu, q4_0_bf16.cu, q4_k_f16.cu, q4_k_bf16.cu
];

// Quantized kernel instantiations (14 loaders × 3 Y_types: F16, BF16, F32)
const QUANTIZED_KERNELS: [&str; 45] = [
    "src/quantized/impl/qkv_segmented_f32.cu",
    "src/quantized/impl/q2_K_f16.cu",
    "src/quantized/impl/q2_K_bf16.cu",
    "src/quantized/impl/q2_K_f32.cu",
    "src/quantized/impl/q3_k_f16.cu",
    "src/quantized/impl/q3_k_bf16.cu",
    "src/quantized/impl/q3_k_f32.cu",
    "src/quantized/impl/q4_0_f16.cu",
    "src/quantized/impl/q4_0_bf16.cu",
    "src/quantized/impl/q4_0_f32.cu",
    "src/quantized/impl/q4_1_f16.cu",
    "src/quantized/impl/q4_1_bf16.cu",
    "src/quantized/impl/q4_1_f32.cu",
    "src/quantized/impl/q4_k_f16.cu",
    "src/quantized/impl/q4_k_bf16.cu",
    "src/quantized/impl/q4_k_f32.cu",
    "src/quantized/impl/q5_0_f16.cu",
    "src/quantized/impl/q5_0_bf16.cu",
    "src/quantized/impl/q5_0_f32.cu",
    "src/quantized/impl/q5_1_f16.cu",
    "src/quantized/impl/q5_1_bf16.cu",
    "src/quantized/impl/q5_1_f32.cu",
    "src/quantized/impl/q5_k_f16.cu",
    "src/quantized/impl/q5_k_bf16.cu",
    "src/quantized/impl/q5_k_f32.cu",
    "src/quantized/impl/q6_k_f16.cu",
    "src/quantized/impl/q6_k_bf16.cu",
    "src/quantized/impl/q6_k_f32.cu",
    "src/quantized/impl/q8_0_f16.cu",
    "src/quantized/impl/q8_0_bf16.cu",
    "src/quantized/impl/q8_0_f32.cu",
    "src/quantized/impl/q8_1_f16.cu",
    "src/quantized/impl/q8_1_bf16.cu",
    "src/quantized/impl/q8_1_f32.cu",
    "src/quantized/impl/q8_k_f16.cu",
    "src/quantized/impl/q8_k_bf16.cu",
    "src/quantized/impl/q8_k_f32.cu",
    "src/quantized/impl/mxfp4_f32.cu",
    "src/quantized/impl/q2_ko_f32.cu",
    "src/quantized/impl/q_awq_f16.cu",
    "src/quantized/impl/q_awq_bf16.cu",
    "src/quantized/impl/q_awq_f32.cu",
    "src/quantized/impl/q_awq_g64_f16.cu",
    "src/quantized/impl/q_awq_g64_bf16.cu",
    "src/quantized/impl/q_awq_g64_f32.cu",
];

// Flash-attention kernels: 12 total
const FLASH_KERNELS: [&str; 14] = [
    // Batched sampling (1 api + 4 variants)
    "src/sampling/batched_sampling_api.cu",
    "src/sampling/batched_sampling_f32.cu",
    "src/sampling/batched_sampling_f16.cu",
    "src/sampling/batched_sampling_bf16.cu",
    "src/sampling/batched_sampling_fp8_e4m3.cu",
    // Paged decode: per-dtype dispatchers (hdim 64/96/128/256)
    "src/paged-decode/paged_decode_api_fp16.cu",
    "src/paged-decode/paged_decode_api_bf16.cu",
    // INT8 prefix-attention prefill (1 api dispatcher + fp16, bf16)
    "src/paged-prefill/paged_prefill_int8_api.cu",
    "src/paged-prefill/paged_prefill_int8_fp16.cu",
    "src/paged-prefill/paged_prefill_int8_bf16.cu",
    // Paged glue: reprojection glue forward (decode-derivative; fp16, bf16)
    "src/paged-glue/paged_glue_api_fp16.cu",
    "src/paged-glue/paged_glue_api_bf16.cu",
    // Paged latent attention: single-latent K≡V window + compressed top-k
    "src/paged-latent/paged_latent_api_bf16.cu",
    // Gated DeltaNet (Qwen3.5/3.8 hybrid layers): one F32 entry TU over the
    // decode-step/conv-step and fused-prefill-scan kernel headers
    "src/delta-net/delta_net_api_f32.cu",
];

/// Provenance BDP scan — the scalar backend, the b1 tensor-core (BMMA) backend
/// (sm_75..sm_89), and the INT8 tensor-core (IMMA) backend (sm_80+, incl.
/// Blackwell). All compile with the sampling args (fast-math): the scan result
/// is a relevance ranking, and the backends share `bdp_vote.cuh` so their float
/// finalize is one code path.
const PROVENANCE_KERNELS: [&str; 3] = [
    "src/provenance/bdp_scan.cu",
    "src/provenance/bdp_bmma.cu",
    "src/provenance/bdp_imma.cu",
];

// ============================================================================
// Archive group definition and construction
// ============================================================================

/// An archive group: a set of kernels compiled with the same flags, linked into one .a
struct ArchiveGroup {
    name: String,
    kernels: Vec<String>,
    compile_args: Vec<String>,
    include_dirs: Vec<String>,
}

/// Build all archive group definitions. `is_msvc` adds the MSVC-specific flags.
fn build_archive_groups(is_msvc: bool) -> Vec<ArchiveGroup> {
    // Base build arguments
    let mut base_args = vec![
        "-std=c++17".to_string(),
        "-O3".to_string(),
        "-U__CUDA_NO_HALF_OPERATORS__".to_string(),
        "-U__CUDA_NO_HALF_CONVERSIONS__".to_string(),
        "-U__CUDA_NO_HALF2_OPERATORS__".to_string(),
        "-U__CUDA_NO_BFLOAT16_CONVERSIONS__".to_string(),
        "-DCUDA_ARCH=800".to_string(),
        "-Icutlass/include".to_string(),
        "--expt-relaxed-constexpr".to_string(),
        "--expt-extended-lambda".to_string(),
        "--use_fast_math".to_string(),
        // Embed source-line info in PTX/CUBIN (no meaningful runtime overhead).
        // Enables compute-sanitizer and cuda-gdb to report the exact kernel
        // source file + line number on illegal-address and other faults.
        "--generate-line-info".to_string(),
        // Target archs: native SASS for Ada (sm_89) and Blackwell (sm_120),
        // plus compute_120 PTX as a forward-compat fallback. These live in the
        // shared compile args (rather than hardcoded at the nvcc call) so that
        // changing the target arch invalidates the per-kernel and archive
        // caches — otherwise stale cubins of the wrong arch get reused.
        "-gencode=arch=compute_89,code=sm_89".to_string(),
        "-gencode=arch=compute_120,code=[sm_120,compute_120]".to_string(),
    ];

    if is_msvc {
        base_args.push("-D_USE_MATH_DEFINES".to_string());
        // Match Rust's MSVC target, which links the *dynamic* CRT: without
        // this, cl.exe compiles nvcc's host-side launch stubs against the
        // static CRT (/MT → LIBCMT defaultlib directives in every object),
        // and the final link reports LNK4098 (`LIBCMT conflicts with use of
        // other libs`) — two CRTs in one binary, which is a real hazard
        // (duplicate allocators/locale state), not just noise.
        base_args.push("-Xcompiler".to_string());
        base_args.push("/MD".to_string());
    } else {
        base_args.push("-Xcompiler".to_string());
        base_args.push("-fPIC".to_string());
    }

    let simple_includes = vec!["-Isrc".to_string(), "-Isrc/simple".to_string()];
    let quantized_includes = vec![
        "-Isrc".to_string(),
        "-Isrc/quantized".to_string(),
        "-Isrc/quantized/impl".to_string(),
        "-Isrc/quantized/loader".to_string(),
    ];
    let flash_includes = vec!["-Isrc".to_string()];

    let paged_common_args = vec![
        "-Xptxas=-O3".to_string(),
        "--extra-device-vectorization".to_string(),
    ];
    let paged_prefill_extra: Vec<String> = paged_common_args
        .iter()
        .cloned()
        .chain(std::iter::once("-maxrregcount=128".to_string()))
        .collect();
    let paged_decode_extra: Vec<String> = paged_common_args
        .iter()
        .cloned()
        .chain(std::iter::once("-maxrregcount=96".to_string()))
        .collect();

    let simple_args: Vec<String> = base_args
        .iter()
        .chain(simple_includes.iter())
        .chain(paged_common_args.iter())
        .cloned()
        .collect();
    let quantized_args: Vec<String> = base_args
        .iter()
        .chain(quantized_includes.iter())
        .chain(paged_common_args.iter())
        .cloned()
        .collect();
    let sampling_args: Vec<String> = base_args
        .iter()
        .chain(flash_includes.iter())
        .cloned()
        .collect();
    let prefill_args: Vec<String> = base_args
        .iter()
        .chain(flash_includes.iter())
        .chain(paged_prefill_extra.iter())
        .cloned()
        .collect();
    let decode_args: Vec<String> = base_args
        .iter()
        .chain(flash_includes.iter())
        .chain(paged_decode_extra.iter())
        .cloned()
        .collect();

    let mut groups: Vec<ArchiveGroup> = Vec::new();

    // 1. Simple kernels
    groups.push(ArchiveGroup {
        name: "simple".to_string(),
        kernels: SIMPLE_KERNELS.iter().map(|s| s.to_string()).collect(),
        compile_args: simple_args.clone(),
        include_dirs: simple_includes.clone(),
    });

    // 1b. Seal-time quantize/select dispatcher — its own group under the
    //     bit-exact mirror contract flags. The CPU codecs (the latent band
    //     `band_chunk_roundtrip`, the Qwen selection oracles) replicate the
    //     encoders and error measurements op-for-op; `--use_fast_math`'s
    //     approximate division and implicit fma contraction break that at
    //     .5 rounding boundaries (measured: one-code quant flips). These
    //     kernels run at seal time, not on the decode hot path, so the
    //     precise-math cost is irrelevant.
    {
        let quantize_dispatch_args: Vec<String> = simple_args
            .iter()
            .cloned()
            .chain(
                [
                    "-fmad=false".to_string(),
                    "-prec-div=true".to_string(),
                    "-prec-sqrt=true".to_string(),
                ]
                .into_iter(),
            )
            .collect();
        groups.push(ArchiveGroup {
            name: "quantize_dispatch".to_string(),
            kernels: vec!["src/simple/quantized_dispatcher.cu".to_string()],
            compile_args: quantize_dispatch_args,
            include_dirs: simple_includes.clone(),
        });
    }

    // 2. Quantized kernels — split per quant type
    {
        let mut quant_groups: BTreeMap<String, Vec<String>> = BTreeMap::new();
        for kernel_path in QUANTIZED_KERNELS.iter() {
            let stem = Path::new(kernel_path)
                .file_stem()
                .unwrap()
                .to_str()
                .unwrap();
            let qtype = stem
                .strip_suffix("_f16")
                .or_else(|| stem.strip_suffix("_bf16"))
                .or_else(|| stem.strip_suffix("_f32"))
                .unwrap_or(stem);
            quant_groups
                .entry(qtype.to_string())
                .or_default()
                .push(kernel_path.to_string());
        }
        for (qtype, kernels) in quant_groups {
            groups.push(ArchiveGroup {
                name: qtype,
                kernels,
                compile_args: quantized_args.clone(),
                include_dirs: quantized_includes.clone(),
            });
        }
    }

    // 3. Sampling kernels
    {
        let sampling_kernels: Vec<String> = FLASH_KERNELS
            .iter()
            .filter(|k| k.contains("sampling/"))
            .map(|s| s.to_string())
            .collect();
        groups.push(ArchiveGroup {
            name: "sampling".to_string(),
            kernels: sampling_kernels,
            compile_args: sampling_args.clone(),
            include_dirs: flash_includes.clone(),
        });
    }

    // 3b. Provenance BDP scan — full fast-math (like sampling); the scan result
    // is a relevance ranking, so ~ULP float differences vs the CPU are fine.
    {
        groups.push(ArchiveGroup {
            name: "provenance".to_string(),
            kernels: PROVENANCE_KERNELS.iter().map(|s| s.to_string()).collect(),
            compile_args: sampling_args.clone(),
            include_dirs: flash_includes.clone(),
        });
    }

    // 4. Paged prefill kernels
    {
        let prefill_kernels: Vec<String> = FLASH_KERNELS
            .iter()
            .filter(|k| k.contains("paged-prefill") || k.contains("paged_prefill"))
            .map(|s| s.to_string())
            .collect();
        groups.push(ArchiveGroup {
            name: "paged_prefill".to_string(),
            kernels: prefill_kernels,
            compile_args: prefill_args,
            include_dirs: flash_includes.clone(),
        });
    }

    // 5. Paged decode kernels
    {
        let decode_kernels: Vec<String> = FLASH_KERNELS
            .iter()
            .filter(|k| k.contains("paged-decode") || k.contains("paged_decode"))
            .map(|s| s.to_string())
            .collect();
        groups.push(ArchiveGroup {
            name: "paged_decode".to_string(),
            kernels: decode_kernels,
            compile_args: decode_args.clone(),
            include_dirs: flash_includes.clone(),
        });
    }

    // 6. Paged glue kernels (decode-derivative — shares the decode device
    //    helpers and the same compile flags).
    {
        let glue_kernels: Vec<String> = FLASH_KERNELS
            .iter()
            .filter(|k| k.contains("paged-glue") || k.contains("paged_glue"))
            .map(|s| s.to_string())
            .collect();
        groups.push(ArchiveGroup {
            name: "paged_glue".to_string(),
            kernels: glue_kernels,
            compile_args: decode_args.clone(),
            include_dirs: flash_includes.clone(),
        });
    }

    // 7. Paged latent attention (decode-derivative fork: single-latent K≡V,
    //    HEAD_DIM=512). Same flags as the stock decode PLUS `-fmad=false`:
    //    implicit mul+add contraction is disabled so the CPU mirror oracle can
    //    reproduce the kernel bit-for-bit — fused multiply-adds exist only
    //    where the kernel writes an explicit `__fmaf_rn`.
    //
    //    NOTE: this group keeps `--use_fast_math` (from base_args) for the
    //    hot path — `-prec-div` cost a measured ~8% on prefill/decode. The
    //    handful of divisions that must match the mirror bit-for-bit (the
    //    int8-requant reciprocal, the per-128 constant scales) are written
    //    as explicit IEEE ops in the kernel (`__frcp_rn`, `x*(1.f/K)`), so
    //    they are precise regardless of the fast-math flag. The seal-time
    //    quantize/select kernels — where the encoders run — DO compile under
    //    `-prec-div=true` (the `quantize_dispatch` group), since perf is
    //    irrelevant at seal time.
    {
        let latent_kernels: Vec<String> = FLASH_KERNELS
            .iter()
            .filter(|k| k.contains("paged-latent") || k.contains("paged_latent"))
            .map(|s| s.to_string())
            .collect();
        let latent_args: Vec<String> = decode_args
            .iter()
            .cloned()
            .chain(std::iter::once("-fmad=false".to_string()))
            .collect();
        groups.push(ArchiveGroup {
            name: "paged_latent".to_string(),
            kernels: latent_kernels,
            compile_args: latent_args,
            include_dirs: flash_includes.clone(),
        });
    }

    // 8. Gated DeltaNet (Qwen3.5/3.8 hybrid layers). Plain F32 recurrence
    //    kernels — no tensor cores, no smem heroics — validated against the
    //    tolerance-based sequential reference, so the stock decode flags are
    //    all it needs.
    {
        let dn_kernels: Vec<String> = FLASH_KERNELS
            .iter()
            .filter(|k| k.contains("delta-net"))
            .map(|s| s.to_string())
            .collect();
        groups.push(ArchiveGroup {
            name: "delta_net".to_string(),
            kernels: dn_kernels,
            compile_args: decode_args.clone(),
            include_dirs: flash_includes.clone(),
        });
    }

    groups
}

// ============================================================================
// Hashing and caching utilities
// ============================================================================

/// Parse #include statements from a file and return list of included paths.
/// include_dirs: list of -I paths to search (e.g., ["-Isrc", "-Isrc/quantized"])
fn parse_includes(
    file_path: &Path,
    base_dir: &Path,
    include_dirs: &[String],
) -> Result<Vec<PathBuf>> {
    let content = fs::read_to_string(file_path)
        .with_context(|| format!("Failed to read file: {}", file_path.display()))?;

    let mut includes = Vec::new();

    for line in content.lines() {
        let trimmed = line.trim();

        if let Some(rest) = trimmed.strip_prefix("#include") {
            let rest = rest.trim();

            let quoted = if let Some(q) = rest.strip_prefix('"').and_then(|s| s.strip_suffix('"')) {
                q
            } else if rest
                .strip_prefix('<')
                .and_then(|s| s.strip_suffix('>'))
                .is_some()
            {
                continue; // System include
            } else {
                continue;
            };

            let file_dir = file_path.parent().unwrap_or(base_dir);
            let mut found_path: Option<PathBuf> = None;

            // Try relative to file's directory first
            let relative_path = file_dir.join(quoted);
            if relative_path.exists() {
                found_path = Some(relative_path.canonicalize()?);
            } else {
                // Try each include directory
                for inc_dir in include_dirs {
                    let dir = inc_dir.strip_prefix("-I").unwrap_or(inc_dir);
                    let inc_path = base_dir.join(dir).join(quoted);
                    if inc_path.exists() {
                        found_path = Some(inc_path.canonicalize()?);
                        break;
                    }
                }
            }

            if let Some(path) = found_path {
                includes.push(path);
            } else {
                if quoted.ends_with(".cu") || quoted.ends_with(".cuh") {
                    panic!(
                        "Include file not found: '{}' (referenced from {})\nSearched in:\n  - {}\n{}",
                        quoted,
                        file_path.display(),
                        file_dir.display(),
                        include_dirs
                            .iter()
                            .map(|d| format!("  - {}", d.strip_prefix("-I").unwrap_or(d)))
                            .collect::<Vec<_>>()
                            .join("\n")
                    );
                }
            }
        }
    }

    Ok(includes)
}

/// Recursively collect all dependencies for a file
fn collect_dependencies(
    file_path: &Path,
    base_dir: &Path,
    include_dirs: &[String],
    visited: &mut HashSet<PathBuf>,
) -> Result<Vec<PathBuf>> {
    let canonical = file_path.canonicalize()?;

    if visited.contains(&canonical) {
        return Ok(Vec::new());
    }
    visited.insert(canonical.clone());

    let mut deps = vec![canonical.clone()];
    let includes = parse_includes(&canonical, base_dir, include_dirs)?;

    for include in includes {
        let sub_deps = collect_dependencies(&include, base_dir, include_dirs, visited)?;
        deps.extend(sub_deps);
    }

    Ok(deps)
}

/// Compute SHA256 hash of a file (line-ending normalised so CRLF ≡ LF)
fn hash_file(path: &Path) -> Result<String> {
    let raw = fs::read(path).with_context(|| format!("Failed to read file: {}", path.display()))?;
    let contents: Vec<u8> = raw.into_iter().filter(|&b| b != b'\r').collect();
    let mut hasher = Sha256::new();
    hasher.update(&contents);
    Ok(format!("{:x}", hasher.finalize()))
}

/// Compute combined hash of kernel source + all its #include deps + compiler flags
fn compute_kernel_hash(
    kernel_path: &str,
    build_args: &[String],
    base_dir: &Path,
    include_dirs: &[String],
    dep_cache: &mut HashMap<String, String>,
) -> Result<String> {
    let mut hasher = Sha256::new();
    let kernel_file = base_dir.join(kernel_path);

    let mut visited = HashSet::new();
    let dependencies = collect_dependencies(&kernel_file, base_dir, include_dirs, &mut visited)?;

    let mut dep_paths: Vec<_> = dependencies.iter().collect();
    dep_paths.sort();

    for dep in &dep_paths {
        let dep_str = dep.to_string_lossy().to_string();
        let hash = if let Some(cached) = dep_cache.get(&dep_str) {
            cached.clone()
        } else {
            let h = hash_file(dep)?;
            dep_cache.insert(dep_str, h.clone());
            h
        };
        hasher.update(hash.as_bytes());
    }

    for arg in build_args {
        hasher.update(arg.as_bytes());
    }

    Ok(format!("{:x}", hasher.finalize()))
}

/// Check if a precompiled archive is valid by comparing aggregate hash
fn is_archive_cache_valid(archive_name: &str, precompiled_dir: &Path, current_hash: &str) -> bool {
    let hash_file = precompiled_dir.join(format!("lib{}.a.sha256", archive_name));
    let gz_file = precompiled_dir.join(format!("lib{}.a.gz", archive_name));

    if !hash_file.exists() || !gz_file.exists() {
        return false;
    }

    if let Ok(stored_hash) = fs::read_to_string(&hash_file) {
        stored_hash.trim() == current_hash
    } else {
        false
    }
}

/// Save aggregate hash for an archive
fn save_archive_hash(archive_name: &str, hash: &str, precompiled_dir: &Path) -> Result<()> {
    let hash_file = precompiled_dir.join(format!("lib{}.a.sha256", archive_name));
    fs::write(&hash_file, hash)
        .with_context(|| format!("Failed to write hash file: {}", hash_file.display()))?;
    Ok(())
}

/// Compute aggregate hash for an archive group from individual kernel hashes
fn compute_archive_hash(kernel_hashes: &[(String, String)]) -> String {
    let mut hasher = Sha256::new();
    for (path, hash) in kernel_hashes {
        hasher.update(path.as_bytes());
        hasher.update(b":");
        hasher.update(hash.as_bytes());
        hasher.update(b"\n");
    }
    format!("{:x}", hasher.finalize())
}

/// Filter out platform-specific build args that don't affect CUDA device code output.
fn canonical_args_for_hash(args: &[String]) -> Vec<String> {
    args.iter()
        .filter(|a| !matches!(a.as_str(), "-D_USE_MATH_DEFINES" | "-Xcompiler" | "-fPIC"))
        .cloned()
        .collect()
}

/// Compute per-kernel hashes and aggregate hash for an archive group.
/// Returns (kernel_hashes, aggregate_hash).
fn compute_group_hashes(
    group: &ArchiveGroup,
    base_dir: &Path,
    dep_cache: &mut HashMap<String, String>,
) -> Result<(Vec<(String, String)>, String)> {
    let canonical_args = canonical_args_for_hash(&group.compile_args);
    let mut kernel_hashes: Vec<(String, String)> = Vec::new();
    for kernel_path in &group.kernels {
        let hash = compute_kernel_hash(
            kernel_path,
            &canonical_args,
            base_dir,
            &group.include_dirs,
            dep_cache,
        )?;
        kernel_hashes.push((kernel_path.clone(), hash));
    }
    kernel_hashes.sort_by(|a, b| a.0.cmp(&b.0));
    let aggregate_hash = compute_archive_hash(&kernel_hashes);
    Ok((kernel_hashes, aggregate_hash))
}

// ============================================================================
// GPU detection
// ============================================================================

fn detect_cuda_compute_cap() -> Result<u32> {
    if let Ok(v) = std::env::var("CUDA_COMPUTE_CAP") {
        if let Ok(cap) = v.trim().parse::<u32>() {
            if cap >= 50 && cap <= 999 {
                return Ok(cap);
            }
        }
    }

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

    Ok(80)
}

// ============================================================================
// Compilation
// ============================================================================

/// Compile a single kernel with nvcc. Target archs come from `build_args`.
fn compile_kernel_nvcc(
    kernel_path: &str,
    build_dir: &Path,
    build_args: &[String],
) -> Result<PathBuf> {
    let kernel_name = PathBuf::from(kernel_path)
        .file_stem()
        .unwrap()
        .to_string_lossy()
        .to_string();
    let out_obj = build_dir.join(format!("{}.o", kernel_name));

    let mut cmd = std::process::Command::new("nvcc");
    with_host_compiler(&mut cmd);
    cmd.arg("-c")
        .arg(kernel_path)
        .arg("-o")
        .arg(&out_obj)
        // Compile the gencode arches (sm_89 + sm_120) in parallel within this
        // nvcc invocation — halves the per-kernel tail now that we emit two.
        .arg("--threads")
        .arg("2");

    for arg in build_args {
        cmd.arg(arg);
    }

    let output = cmd
        .output()
        .with_context(|| format!("Failed to run nvcc for {}", kernel_name))?;
    if !output.status.success() {
        anyhow::bail!(
            "nvcc failed for {}:\n{}\n{}",
            kernel_name,
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

/// Compile multiple kernels in parallel using threads
fn compile_kernels_parallel(
    kernel_paths: &[&str],
    build_dir: &Path,
    build_args: &[String],
    max_threads: usize,
) -> Result<Vec<PathBuf>> {
    use std::sync::mpsc;
    use std::thread;

    if kernel_paths.is_empty() {
        return Ok(Vec::new());
    }

    let (tx, rx) = mpsc::channel();
    let mut handles = Vec::new();
    let mut pending = kernel_paths.iter().peekable();

    while pending.peek().is_some() || !handles.is_empty() {
        while handles.len() < max_threads {
            if let Some(&kernel_path) = pending.next() {
                let tx = tx.clone();
                let build_dir = build_dir.to_path_buf();
                let build_args = build_args.to_vec();
                let kernel_path = kernel_path.to_string();

                handles.push(thread::spawn(move || {
                    let result = compile_kernel_nvcc(&kernel_path, &build_dir, &build_args);
                    tx.send((kernel_path, result)).unwrap();
                }));
            } else {
                break;
            }
        }

        if !handles.is_empty() {
            let (kernel_path, result) = rx.recv().unwrap();
            handles.pop();
            result.with_context(|| format!("Failed to compile {}", kernel_path))?;
        }
    }

    let out_files: Vec<PathBuf> = kernel_paths
        .iter()
        .map(|p| {
            let name = PathBuf::from(p)
                .file_stem()
                .unwrap()
                .to_string_lossy()
                .to_string();
            build_dir.join(format!("{}.o", name))
        })
        .collect();

    Ok(out_files)
}

// ============================================================================
// Archive and compression
// ============================================================================

/// Compress a file with gzip and write to dest.
/// Retries on Windows file-lock errors (OS error 1224 / 32).
fn compress_gz(src: &Path, dest: &Path) -> Result<()> {
    let start = std::time::Instant::now();
    let data = fs::read(src).with_context(|| format!("Failed to read {}", src.display()))?;
    let mut encoder = GzEncoder::new(Vec::new(), Compression::fast());
    encoder
        .write_all(&data)
        .context("gzip compression failed")?;
    let compressed = encoder.finish().context("gzip finish failed")?;

    let max_retries = 3;
    let mut last_err = None;
    for attempt in 0..=max_retries {
        match fs::write(dest, &compressed) {
            Ok(()) => {
                if attempt > 0 {
                    eprintln!(
                        "  Compressed {} (succeeded on retry {})",
                        dest.file_name().unwrap().to_string_lossy(),
                        attempt,
                    );
                }
                last_err = None;
                break;
            }
            Err(e) => {
                let raw = e.raw_os_error().unwrap_or(0);
                if (raw == 1224 || raw == 32) && attempt < max_retries {
                    eprintln!(
                        "  Retry {}/{}: {} locked (OS error {}), waiting 2s...",
                        attempt + 1,
                        max_retries,
                        dest.file_name().unwrap().to_string_lossy(),
                        raw,
                    );
                    std::thread::sleep(std::time::Duration::from_secs(2));
                    last_err = Some(e);
                } else {
                    return Err(e).with_context(|| format!("Failed to write {}", dest.display()));
                }
            }
        }
    }
    if let Some(e) = last_err {
        return Err(e).with_context(|| {
            format!(
                "Failed to write {} after {} retries",
                dest.display(),
                max_retries
            )
        });
    }

    eprintln!(
        "  Compressed {} -> {} ({:.1} MB -> {:.1} MB, {:.1}s)",
        src.file_name().unwrap().to_string_lossy(),
        dest.file_name().unwrap().to_string_lossy(),
        data.len() as f64 / (1024.0 * 1024.0),
        compressed.len() as f64 / (1024.0 * 1024.0),
        start.elapsed().as_secs_f64(),
    );
    Ok(())
}

/// Decompress a gzip file and write to dest.
fn decompress_gz(src: &Path, dest: &Path) -> Result<()> {
    let start = std::time::Instant::now();
    let compressed = fs::read(src).with_context(|| format!("Failed to read {}", src.display()))?;
    let mut decoder = GzDecoder::new(&compressed[..]);
    let mut data = Vec::new();
    decoder
        .read_to_end(&mut data)
        .context("gzip decompression failed")?;
    fs::write(dest, &data).with_context(|| format!("Failed to write {}", dest.display()))?;
    eprintln!(
        "  Decompressed {} ({:.1} MB, {:.1}s)",
        dest.file_name().unwrap().to_string_lossy(),
        data.len() as f64 / (1024.0 * 1024.0),
        start.elapsed().as_secs_f64(),
    );
    Ok(())
}

/// Link object files into a static archive
fn create_archive(
    lib_name: &str,
    object_files: &[PathBuf],
    build_dir: &Path,
    is_msvc: bool,
) -> Result<()> {
    let lib_file = build_dir.join(format!("lib{}.a", lib_name));

    if is_msvc {
        // Same directory as `cl.exe`, and the same problem: reachable from a
        // Developer Command Prompt and from nowhere else.
        let mut cmd = std::process::Command::new("lib.exe");
        with_host_compiler(&mut cmd);
        cmd.arg(format!("/OUT:{}", lib_file.display()));
        for obj in object_files {
            cmd.arg(obj);
        }
        let output = cmd.output().context("Failed to run lib.exe")?;
        if !output.status.success() {
            anyhow::bail!(
                "lib.exe failed:\nstdout: {}\nstderr: {}",
                String::from_utf8_lossy(&output.stdout),
                String::from_utf8_lossy(&output.stderr)
            );
        }
    } else {
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
    Ok(())
}

// ============================================================================
// Staged .o cache operations
// ============================================================================

/// Check if a staged .o file is valid for the given kernel hash
fn is_staged_kernel_valid(staged_dir: &Path, kernel_name: &str, expected_hash: &str) -> bool {
    let staged_o = staged_dir.join(format!("{}.o", kernel_name));
    let staged_hash = staged_dir.join(format!("{}.o.sha256", kernel_name));

    staged_o.exists()
        && staged_hash.exists()
        && fs::read_to_string(&staged_hash)
            .map(|h| h.trim() == expected_hash)
            .unwrap_or(false)
}

/// Save a compiled .o file to the staged cache with its hash
fn save_to_staged_cache(
    build_dir: &Path,
    staged_dir: &Path,
    kernel_name: &str,
    hash: &str,
) -> Result<()> {
    let build_o = build_dir.join(format!("{}.o", kernel_name));
    let staged_o = staged_dir.join(format!("{}.o", kernel_name));
    let staged_hash = staged_dir.join(format!("{}.o.sha256", kernel_name));
    if build_o.exists() {
        fs::copy(&build_o, &staged_o)
            .with_context(|| format!("Failed to copy {}.o to staged cache", kernel_name))?;
        fs::write(&staged_hash, hash)
            .with_context(|| format!("Failed to write staged hash for {}", kernel_name))?;
    }
    Ok(())
}

/// Copy a staged .o file to the build directory
#[allow(dead_code)]
fn copy_from_staged_cache(staged_dir: &Path, build_dir: &Path, kernel_name: &str) -> Result<()> {
    let staged_o = staged_dir.join(format!("{}.o", kernel_name));
    let dest = build_dir.join(format!("{}.o", kernel_name));
    fs::copy(&staged_o, &dest)
        .with_context(|| format!("Failed to copy staged {} to build dir", kernel_name))?;
    Ok(())
}

/// Extract the kernel stem name from a kernel path like "src/quantized/impl/q4_0_f16.cu"
fn kernel_stem(kernel_path: &str) -> String {
    PathBuf::from(kernel_path)
        .file_stem()
        .unwrap()
        .to_string_lossy()
        .to_string()
}

/// Detect if the current target is MSVC (for build.rs context).
/// Falls back to cfg!(windows) heuristic when TARGET env is not set.
fn detect_is_msvc() -> bool {
    if let Ok(target) = std::env::var("TARGET") {
        target.contains("msvc")
    } else {
        cfg!(windows)
    }
}

/// The directory holding `cl.exe`, found the way MSVC itself expects.
///
/// `nvcc` compiles device code but hands the host half to the platform
/// compiler, which it locates **on `PATH`** — not through anything cargo sets
/// up. So a build that links perfectly well still dies with
/// `nvcc fatal : Cannot find compiler 'cl.exe' in PATH` in any shell that is
/// not a Developer Command Prompt: an ordinary terminal, a CI step, an editor's
/// integrated shell. The usual answer is "remember to open the right prompt",
/// which is a requirement nobody can see and everybody forgets.
///
/// So the build finds it. `vswhere.exe` ships at a fixed location with every
/// Visual Studio since 2017 and reports the newest installation carrying the
/// C++ toolset; the toolset version underneath moves with each VS update, so
/// the highest directory is taken rather than any pinned number.
///
/// Deliberately *not* solved by putting MSVC on the global `PATH`: that pins a
/// toolset version machine-wide and puts Microsoft's `link.exe` ahead of every
/// other `link` on the system, which is its own long-running confusion.
#[cfg(windows)]
fn msvc_bin_dir() -> Option<std::path::PathBuf> {
    // Already reachable — a Developer Command Prompt, or someone's own PATH.
    if std::process::Command::new("cl.exe")
        .arg("/?")
        .output()
        .is_ok()
    {
        return None;
    }

    let pf86 = std::env::var("ProgramFiles(x86)")
        .unwrap_or_else(|_| r"C:\Program Files (x86)".to_string());
    let vswhere =
        std::path::Path::new(&pf86).join(r"Microsoft Visual Studio\Installer\vswhere.exe");
    let out = std::process::Command::new(vswhere)
        .args([
            "-latest",
            "-products",
            "*",
            "-requires",
            "Microsoft.VisualStudio.Component.VC.Tools.x86.x64",
            "-property",
            "installationPath",
        ])
        .output()
        .ok()?;
    let root = String::from_utf8_lossy(&out.stdout).trim().to_string();
    if root.is_empty() {
        return None;
    }

    // Newest toolset present, by directory name — these sort correctly as
    // strings because the components are zero-padded (14.44.35207).
    let mut versions: Vec<_> = std::fs::read_dir(std::path::Path::new(&root).join(r"VC\Tools\MSVC"))
        .ok()?
        .flatten()
        .map(|e| e.path())
        .filter(|p| p.is_dir())
        .collect();
    versions.sort();
    let bin = versions.pop()?.join(r"bin\Hostx64\x64");
    bin.join("cl.exe").is_file().then_some(bin)
}

#[cfg(not(windows))]
fn msvc_bin_dir() -> Option<std::path::PathBuf> {
    None
}

/// Put `cl.exe` within nvcc's reach, if it is not already.
///
/// Applied to the child process only. Editing the build's own `PATH` would
/// leak into every other tool it shells out to, for a requirement that belongs
/// to one of them.
fn with_host_compiler(cmd: &mut std::process::Command) {
    let Some(bin) = msvc_bin_dir() else { return };
    let path = std::env::var_os("PATH").unwrap_or_default();
    let mut dirs = vec![bin];
    dirs.extend(std::env::split_paths(&path));
    if let Ok(joined) = std::env::join_paths(dirs) {
        cmd.env("PATH", joined);
    }
}
