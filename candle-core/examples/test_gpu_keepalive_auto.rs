use candle_core::{Device, Result, Tensor, DType};
use candle_core::gpu_keepalive::GpuKeepalive;
use std::thread;
use std::time::Duration;

fn main() -> Result<()> {
    let device = Device::new_cuda(0)?;
    
    println!("=== GPU Keepalive Test ===\n");
    println!("Monitor GPU with:");
    println!("  nvidia-smi --query-gpu=clocks.current.graphics,power.draw --format=csv -l 1\n");
    
    println!("Phase 1: Without keepalive (10 seconds)");
    println!("  Expected: GPU drops to 210 MHz between operations\n");
    
    for i in 0..10 {
        let work = Tensor::zeros((4096, 4096), DType::F32, &device)?;
        let _ = work.matmul(&work)?;
        thread::sleep(Duration::from_secs(1));
        if i % 3 == 0 {
            println!("  {} seconds...", i);
        }
    }
    
    println!("\n✅ Phase 1 complete\n");
    println!("Phase 2: WITH keepalive (10 seconds)");
    println!("  Expected: GPU stays at 2295 MHz constantly\n");
    
    {
        // Create keepalive - GPU stays active while in scope
        let _keepalive = GpuKeepalive::new(&device)?;
        
        for i in 0..10 {
            let work = Tensor::zeros((4096, 4096), DType::F32, &device)?;
            let _ = work.matmul(&work)?;
            thread::sleep(Duration::from_secs(1));
            if i % 3 == 0 {
                println!("  {} seconds...", i);
            }
        }
        
        // Keepalive drops here
    }
    
    println!("\n✅ Phase 2 complete - keepalive stopped\n");
    println!("Phase 3: Without keepalive again (5 seconds)");
    println!("  Expected: GPU drops to 210 MHz again\n");
    
    for i in 0..5 {
        let work = Tensor::zeros((4096, 4096), DType::F32, &device)?;
        let _ = work.matmul(&work)?;
        thread::sleep(Duration::from_secs(1));
        if i % 2 == 0 {
            println!("  {} seconds...", i);
        }
    }
    
    println!("\n✅ Test complete!");
    println!("\nYou should have seen:");
    println!("  Phase 1: GPU at 210-400 MHz (idling)");
    println!("  Phase 2: GPU at 2295 MHz (keepalive active)");
    println!("  Phase 3: GPU at 210-400 MHz (idling again)");
    
    Ok(())
}
