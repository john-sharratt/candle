// Test pinned memory allocation with cudarc for faster GPU transfers
// This tests page-locked host memory for optimal DMA performance

use std::time::Instant;

fn main() -> anyhow::Result<()> {
    println!("=== Testing Pinned vs Regular Memory for GPU Transfers ===\n");

    #[cfg(not(feature = "cuda"))]
    {
        println!("CUDA not enabled. Build with --features cuda");
        return Ok(());
    }

    #[cfg(feature = "cuda")]
    {
        use cudarc::driver::CudaContext;
        
        // Test size: 400 MB (similar to model loading)
        let test_size = 100_000_000; // 100M floats = 400 MB
        println!("Test size: {} MB\n", (test_size * 4) / 1_000_000);

        let ctx = CudaContext::new(0)?;
        let stream = ctx.default_stream();
        
        // Test 1: Regular Vec → GPU transfer (baseline)
        println!("Test 1: Regular memory (Vec → GPU, pageable)");
        let mut times = Vec::new();
        for run in 1..=3 {
            let data: Vec<f32> = vec![1.0; test_size];
            let start = Instant::now();
            let _dev_buffer = stream.memcpy_stod(&data)?;
            stream.synchronize()?;
            let elapsed = start.elapsed();
            let bandwidth_gbs = (test_size as f64 * 4.0) / elapsed.as_secs_f64() / 1e9;
            times.push(elapsed.as_secs_f64());
            println!("  Run {}: {:.3}s ({:.2} GB/s)", run, elapsed.as_secs_f32(), bandwidth_gbs);
        }
        let avg = times.iter().sum::<f64>() / times.len() as f64;
        let avg_bw = (test_size as f64 * 4.0) / avg / 1e9;
        println!("  Average: {:.3}s ({:.2} GB/s)\n", avg, avg_bw);
        
        // Test 2: Pinned memory → GPU transfer
        println!("Test 2: Pinned memory (page-locked → GPU)");
        let mut times = Vec::new();
        for run in 1..=3 {
            // Allocate pinned host memory using cudarc
            let mut pinned = unsafe { ctx.alloc_pinned::<f32>(test_size)? };
            
            // PinnedHostSlice provides as_slice() / as_mut_slice() for access
            // Initialize the data
            {
                let slice = pinned.as_mut_slice()?;
                for i in 0..test_size {
                    slice[i] = 1.0;
                }
            }
            
            let start = Instant::now();
            let _dev_buffer = stream.memcpy_stod(pinned.as_slice()?)?;
            stream.synchronize()?;
            let elapsed = start.elapsed();
            let bandwidth_gbs = (test_size as f64 * 4.0) / elapsed.as_secs_f64() / 1e9;
            times.push(elapsed.as_secs_f64());
            println!("  Run {}: {:.3}s ({:.2} GB/s)", run, elapsed.as_secs_f32(), bandwidth_gbs);
        }
        let avg = times.iter().sum::<f64>() / times.len() as f64;
        let avg_bw = (test_size as f64 * 4.0) / avg / 1e9;
        println!("  Average: {:.3}s ({:.2} GB/s)\n", avg, avg_bw);
        
        println!("=== Summary ===");
        println!("Pinned memory should show 20-50% improvement over regular memory");
        println!("Typical bandwidth:");
        println!("  - Regular (pageable): ~10 GB/s");
        println!("  - Pinned (page-locked): ~12-14 GB/s on PCIe 3.0 x16");
        println!("  - Theoretical PCIe 3.0 x16: ~16 GB/s");
    }
    
    Ok(())
}
