//! Runs the component-isolation and advanced diagnostics at the batch sizes that
//! exercise each kernel.
//!
//! The advanced runner executes at batch 1 (the GEMV path), batch 16 (the
//! tensor-core GEMM path), batch 32 (the transition) and batch 64 (the FP8 MMA
//! path), together with the accumulation and FP8 tracing diagnostics. The
//! component runner executes the isolation diagnostics from the other files in
//! turn.

use super::*;

/// Run all kernel component isolation diagnostics
#[cfg(feature = "cuda")]
pub fn run_kernel_isolation_diagnostics(config: &QuantTestConfig, device: &Device) -> Result<()> {
    let _guard = acquire_cuda_test_lock();
    println!("\n================================================================================");
    println!("KERNEL COMPONENT ISOLATION DIAGNOSTICS for {}", config.name);
    println!("================================================================================");

    println!("\n=== BATCH=1 (GEMV path - known working) ===");
    run_stage_boundary_diagnostic(config.dtype, device, 1)?;
    run_thread_mapping_diagnostic(config.dtype, device, 1)?;
    run_scale_application_diagnostic(config.dtype, device, 1)?;
    run_ktile_pair_diagnostic(config.dtype, device, 1)?;
    run_global_to_shared_diagnostic(config.dtype, device, 1)?;

    println!("\n=== BATCH=16 (GEMM/tensor core path - broken) ===");
    run_stage_boundary_diagnostic(config.dtype, device, 16)?;
    run_thread_mapping_diagnostic(config.dtype, device, 16)?;
    run_scale_application_diagnostic(config.dtype, device, 16)?;
    run_ktile_pair_diagnostic(config.dtype, device, 16)?;
    run_global_to_shared_diagnostic(config.dtype, device, 16)?;
    run_mma_fragment_diagnostic(config.dtype, device)?;

    println!("\n=== ENHANCED K-ROW LOADING ANALYSIS ===");
    run_krow_wrap_diagnostic(config.dtype, device)?;
    run_y_input_isolation_diagnostic(config.dtype, device)?;
    run_first_4_krows_accuracy_diagnostic(config.dtype, device)?;
    run_krow_read_pattern_diagnostic(config.dtype, device)?;

    println!("\n=== THREAD/WARP MAPPING DIAGNOSTICS ===");
    run_column_identity_diagnostic(config.dtype, device)?;
    run_single_column_probe_diagnostic(config.dtype, device)?;
    run_warp_signature_diagnostic(config.dtype, device)?;
    run_column_pair_pattern_diagnostic(config.dtype, device)?;

    Ok(())
}

/// Run all advanced diagnostics for detailed debugging
#[cfg(feature = "cuda")]
pub fn run_advanced_diagnostics(config: &QuantTestConfig, device: &Device) -> Result<()> {
    let _guard = acquire_cuda_test_lock();
    println!("\n================================================================================");
    println!(
        "ADVANCED DIAGNOSTICS: Detailed flow tracing for {}",
        config.name
    );
    println!("================================================================================");

    // Run at batch=1 (GEMV path)
    println!("\n--- BATCH=1 (GEMV path) ---");
    run_subblock_isolation_diagnostic_batched(config.dtype, device, 1)?;
    run_y_stride_diagnostic_batched(config.dtype, device, 1)?;
    run_scale_correspondence_diagnostic(config.dtype, device)?;
    run_part_accumulation_diagnostic(config.dtype, device)?;

    // Run at batch=16 (GEMM path - tensor cores, s16_tc kernel)
    println!("\n--- BATCH=16 (GEMM/tensor core path, s16_tc) ---");
    run_subblock_isolation_diagnostic_batched(config.dtype, device, 16)?;
    run_y_stride_diagnostic_batched(config.dtype, device, 16)?;

    // Run at batch=32 (transition zone)
    println!("\n--- BATCH=32 (transition batch size) ---");
    run_subblock_isolation_diagnostic_batched(config.dtype, device, 32)?;
    run_y_stride_diagnostic_batched(config.dtype, device, 32)?;

    // Run at batch=64 (s32_tc kernel for FP8 - uses m16n8k32 MMA)
    println!("\n--- BATCH=64 (s32_tc kernel, m16n8k32 MMA for FP8) ---");
    run_subblock_isolation_diagnostic_batched(config.dtype, device, 64)?;
    run_y_stride_diagnostic_batched(config.dtype, device, 64)?;

    // NEW: Batch-specific accumulation diagnostics
    println!("\n--- BATCH ACCUMULATION DIAGNOSTICS ---");
    run_accumulation_ratio_diagnostic(config.dtype, device, 1)?;
    run_accumulation_ratio_diagnostic(config.dtype, device, 16)?;
    run_accumulation_ratio_diagnostic(config.dtype, device, 32)?;
    run_accumulation_ratio_diagnostic(config.dtype, device, 64)?;
    run_k_row_isolation_diagnostic(config.dtype, device, 1)?;
    run_k_row_isolation_diagnostic(config.dtype, device, 16)?;
    run_k_row_isolation_diagnostic(config.dtype, device, 32)?;
    run_k_row_isolation_diagnostic(config.dtype, device, 64)?;
    run_batch_comparison_diagnostic(config.dtype, device)?;

    // NEW: Even more targeted probe tests
    println!("\n--- SINGLE ELEMENT PROBE TESTS ---");
    run_single_k_probe_diagnostic(config.dtype, device, 1)?;
    run_single_k_probe_diagnostic(config.dtype, device, 16)?;
    run_single_k_probe_diagnostic(config.dtype, device, 32)?;
    run_single_k_probe_diagnostic(config.dtype, device, 64)?;
    run_k_iteration_count_diagnostic(config.dtype, device)?;

    // FP8-SPECIFIC DIAGNOSTICS - test with FP8 input dtype at various batch sizes
    println!("\n--- FP8 ACTIVATION DIAGNOSTICS (tests FP8 m16n8k32 MMA path) ---");
    run_fp8_activation_diagnostic(config.dtype, device, 1)?;
    run_fp8_activation_diagnostic(config.dtype, device, 16)?;
    run_fp8_activation_diagnostic(config.dtype, device, 32)?;
    run_fp8_activation_diagnostic(config.dtype, device, 64)?;

    // FP8 MMA indicator tracing - use specific values to trace data flow
    println!("\n--- FP8 MMA INDICATOR TRACING (batch=32) ---");
    run_fp8_mma_indicator_tracing(config.dtype, device)?;

    // New extended diagnostics (batch=1 only, for speed)
    println!("\n--- Extended diagnostics (batch=1) ---");
    run_nibble_pair_diagnostic(config.dtype, device)?;
    run_intra_tile_diagnostic(config.dtype, device)?;
    run_position_mapping_diagnostic(config.dtype, device)?;
    run_qh_bit_diagnostic(config.dtype, device)?;
    run_dequant_value_diagnostic(config.dtype, device)?;
    run_qh_raw_verification_diagnostic(config.dtype, device)?;
    run_dequant_kernel_diagnostic(config.dtype, device)?;

    Ok(())
}
