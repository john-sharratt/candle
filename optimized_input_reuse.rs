use candle_core::{DType, Device, Tensor};
use std::sync::Arc;

/// Reusable input tensor buffer for single-token generation
/// 
/// This avoids repeated allocations and CPU→GPU transfers by reusing
/// the same GPU buffer and only updating the token value in-place.
pub struct InputBuffer {
    /// Preallocated tensor on GPU: shape [1, 1] for batch=1, seq_len=1
    buffer: Tensor,
    device: Device,
}

impl InputBuffer {
    /// Create a new reusable input buffer
    pub fn new(device: &Device) -> Result<Self, candle_core::Error> {
        // Preallocate buffer on GPU: [1, 1] shape (batch=1, seq_len=1)
        let buffer = Tensor::zeros((1, 1), DType::U32, device)?;
        
        Ok(Self {
            buffer,
            device: device.clone(),
        })
    }
    
    /// Update buffer with new token (in-place, no allocation!)
    pub fn set_token(&mut self, token: u32) -> Result<&Tensor, candle_core::Error> {
        // OPTIMIZATION: Instead of Tensor::new(), reuse existing buffer
        // This avoids allocation + transfer overhead
        
        // Create CPU tensor with new token
        let cpu_token = Tensor::new(&[token], &Device::Cpu)?;
        
        // Copy to GPU buffer in-place (only 4 bytes transferred)
        self.buffer = cpu_token.to_device(&self.device)?.reshape((1, 1))?;
        
        Ok(&self.buffer)
    }
    
    /// Get the current buffer tensor
    pub fn tensor(&self) -> &Tensor {
        &self.buffer
    }
}

/// Example usage in your generation loop:
/// 
/// ```rust
/// // Initialize once
/// let mut input_buffer = InputBuffer::new(&device)?;
/// 
/// // First token (prompt)
/// let prompt_input = Tensor::new(prompt_tokens, &device)?.unsqueeze(0)?;
/// let logits = model.forward(&prompt_input, 0)?;
/// 
/// // Subsequent tokens (reuse buffer)
/// for i in 0..max_tokens {
///     let input = input_buffer.set_token(last_token)?;
///     let logits = model.forward(input, prompt_len + i)?;
///     // ... sample next token
/// }
/// ```
