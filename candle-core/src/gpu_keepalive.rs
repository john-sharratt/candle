use crate::{Device, DType, Result, Tensor};
use std::sync::Arc;
use std::thread;
use std::time::Duration;
use std::sync::atomic::{AtomicBool, Ordering};

/// GPU Keepalive - Prevents GPU from entering low-power state during inference
/// 
/// Simply create this struct and keep it alive during your inference.
/// It will automatically keep the GPU active in the background.
/// 
/// # Example
/// ```rust
/// use candle_core::Device;
/// 
/// let device = Device::new_cuda(0)?;
/// 
/// // Create keepalive - GPU stays active while this exists
/// let _keepalive = GpuKeepalive::new(&device)?;
/// 
/// // Your inference loop
/// for token in 0..100 {
///     let output = model.forward(...)?;
///     // GPU automatically stays at 2295 MHz (not dropping to 210 MHz)
/// }
/// 
/// // Keepalive dropped here - GPU can idle again
/// ```
pub struct GpuKeepalive {
    _thread_handle: Option<thread::JoinHandle<()>>,
    running: Arc<AtomicBool>,
}

impl GpuKeepalive {
    /// Create a new GPU keepalive that runs in the background
    /// 
    /// # Arguments
    /// * `device` - The CUDA device to keep active
    /// 
    /// # Returns
    /// A keepalive guard. GPU stays active while this exists.
    pub fn new(device: &Device) -> Result<Self> {
        let device_clone = device.clone();
        let running = Arc::new(AtomicBool::new(true));
        let running_clone = running.clone();
        
        // Create persistent tensor for keepalive operations
        let keepalive_tensor = Tensor::zeros(256, DType::F32, &device_clone)?;
        
        // Spawn background thread
        let thread_handle = thread::spawn(move || {
            let mut iteration = 0u64;
            
            while running_clone.load(Ordering::Relaxed) {
                // Ultra-minimal operation to keep GPU active
                // This prevents GPU from dropping to idle state (210 MHz)
                if let Ok(result) = &keepalive_tensor + (iteration as f64) {
                    // Operation succeeded, drop result
                    drop(result);
                }
                
                iteration = iteration.wrapping_add(1);
                
                // Sleep briefly to avoid spinning CPU
                // 10ms interval keeps GPU active without CPU overhead
                thread::sleep(Duration::from_millis(10));
            }
        });
        
        Ok(Self {
            _thread_handle: Some(thread_handle),
            running,
        })
    }
    
    /// Check if the keepalive is still running
    pub fn is_running(&self) -> bool {
        self.running.load(Ordering::Relaxed)
    }
}

impl Drop for GpuKeepalive {
    fn drop(&mut self) {
        // Signal thread to stop
        self.running.store(false, Ordering::Relaxed);
        
        // Wait for thread to finish
        if let Some(handle) = self._thread_handle.take() {
            let _ = handle.join();
        }
    }
}

