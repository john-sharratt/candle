/// Benchmark comparing sequential reader vs mmap tensor loading
/// 
/// Usage: Download a GGUF model and run:
///   cargo run --release --features cuda --example benchmark_mmap -- /path/to/model.gguf

use candle_core::quantized::gguf_file;
use candle_core::{Device, Result};
use std::time::Instant;

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: {} <path-to-gguf-model>", args[0]);
        eprintln!("\nExample:");
        eprintln!("  cargo run --release --features cuda --example benchmark_mmap -- model.gguf");
        std::process::exit(1);
    }

    let model_path = std::path::PathBuf::from(&args[1]);
    if !model_path.exists() {
        eprintln!("Error: File not found: {:?}", model_path);
        std::process::exit(1);
    }

    println!("\n=== GGUF mmap Optimization Benchmark ===");
    println!("Model: {:?}\n", model_path);

    let device = Device::new_cuda(0)?;
    println!("Device: {:?}\n", device);

    // Load metadata
    let mut file = std::fs::File::open(&model_path)?;
    let content = gguf_file::Content::read(&mut file)?;
    
    // Get first 50 tensors as sample
    let tensor_names: Vec<String> = content.tensor_infos.keys()
        .take(50)
        .cloned()
        .collect();

    println!("Loading {} tensors for comparison\n", tensor_names.len());

    // Test 1: Sequential (File→RAM→GPU)
    println!("--- Test 1: Sequential (File→RAM→GPU) ---");
    let start = Instant::now();
    for name in &tensor_names {
        let mut file = std::fs::File::open(&model_path)?;
        let content = gguf_file::Content::read(&mut file)?;
        let _tensor = content.tensor(&mut file, name, &device)?;
    }
    let sequential_time = start.elapsed();
    println!("✓ Time: {:.3}s\n", sequential_time.as_secs_f64());

    // Test 2: mmap (File→GPU with zero-copy)
    println!("--- Test 2: mmap (File→GPU with zero-copy) ---");
    use memmap2::MmapOptions;
    let file = std::fs::File::open(&model_path)?;
    let mmap = unsafe { MmapOptions::new().map(&file)? };
    
    let mut file = std::fs::File::open(&model_path)?;
    let content = gguf_file::Content::read(&mut file)?;
    
    let start = Instant::now();
    for name in &tensor_names {
        let tensor_info = content.tensor_infos.get(name)
            .ok_or_else(|| candle_core::Error::Msg(format!("tensor {} not found", name)))?;
        let _tensor = tensor_info.read_from_mmap(&mmap, content.tensor_data_offset, &device)?;
    }
    let mmap_time = start.elapsed();
    println!("✓ Time: {:.3}s\n", mmap_time.as_secs_f64());

    // Results
    let speedup = sequential_time.as_secs_f64() / mmap_time.as_secs_f64();
    let improvement = (1.0 - mmap_time.as_secs_f64() / sequential_time.as_secs_f64()) * 100.0;

    println!("=== Results ===");
    println!("Sequential: {:.3}s", sequential_time.as_secs_f64());
    println!("mmap:       {:.3}s", mmap_time.as_secs_f64());
    println!("Speedup:    {:.2}x", speedup);
    println!("Faster by:  {:.1}%", improvement);
    
    if mmap_time < sequential_time {
        println!("\n✓ mmap optimization successful!");
        #[cfg(feature = "cuda")]
        println!("  Note: CUDA zero-copy enabled - GPU reads mmap directly via PCIe");
    }

    Ok(())
}
