use std::collections::HashMap;
use std::sync::RwLock;

use candle::{DType, Device, DeviceLocation, Result, Tensor};

#[derive(Debug, Clone)]
pub struct CisPrecomputation {
    pub cos: Tensor,
    pub sin: Tensor,
}

impl CisPrecomputation {
    pub fn to_dtype(&self, dtype: DType) -> Result<CisPrecomputation> {
        Ok(CisPrecomputation {
            cos: self.cos.to_dtype(dtype)?,
            sin: self.sin.to_dtype(dtype)?,
        })
    }

    pub fn to_device(&self, device: &Device) -> Result<CisPrecomputation> {
        Ok(CisPrecomputation {
            cos: self.cos.to_device(device)?,
            sin: self.sin.to_device(device)?,
        })
    }
}

/// Key for device cache: combines dtype and device location.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
struct CacheKey {
    dtype: DType,
    location: DeviceLocation,
}

impl CacheKey {
    fn new(dtype: DType, device: &Device) -> Self {
        Self {
            dtype,
            location: device.location(),
        }
    }
}

/// Precomputed RoPE tables (cos/sin) with CPU-backed storage and lazy GPU conversion.
///
/// - Source tables are stored on CPU in F32 for precision
/// - GPU variants are created on-demand when `get_for_dtype` is called
/// - `compact()` purges the device cache, freeing GPU memory
/// - Tables can grow dynamically if `extend_chunk > 0`
///
/// Shapes:
/// - `cos` and `sin`: (max_seq_len, rope_dim/2)
/// - `inv_freq_f32`: (1, rope_dim/2)
#[derive(Debug)]
pub struct CisPrecomputations {
    /// Source RoPE tables on CPU (F32 for precision).
    cpu_source: CisPrecomputation,
    /// Cached device variants keyed by (dtype, device location).
    device_cache: RwLock<HashMap<CacheKey, CisPrecomputation>>,
    /// Target device for GPU conversions.
    target_device: Device,

    /// Inverse frequency tensor on CPU (for growable tables).
    inv_freq_f32: Option<Tensor>,
    max_seq_len: usize,
    extend_chunk: usize,
}

// Manual Clone implementation since RwLock doesn't implement Clone
impl Clone for CisPrecomputations {
    fn clone(&self) -> Self {
        let device_cache = self.device_cache.read().unwrap().clone();
        Self {
            cpu_source: self.cpu_source.clone(),
            device_cache: RwLock::new(device_cache),
            target_device: self.target_device.clone(),
            inv_freq_f32: self.inv_freq_f32.clone(),
            max_seq_len: self.max_seq_len,
            extend_chunk: self.extend_chunk,
        }
    }
}

impl CisPrecomputations {
    pub fn max_seq_len(&self) -> usize {
        self.max_seq_len
    }

    /// Get the inv_freq tensor if this is a growable RoPE table.
    /// Returns None for fixed-size tables.
    pub fn inv_freq(&self) -> Option<&Tensor> {
        self.inv_freq_f32.as_ref()
    }

    /// Get the inv_freq values as a Vec<f32>.
    /// Returns None for fixed-size tables.
    pub fn inv_freq_vec(&self) -> Option<Vec<f32>> {
        self.inv_freq_f32
            .as_ref()
            .map(|t| t.flatten_all().unwrap().to_vec1::<f32>().unwrap())
    }

    /// Get RoPE tables for the requested dtype on GPU.
    ///
    /// Tables are lazily converted from CPU source and cached.
    /// Multiple dtype/device combinations can be cached.
    pub fn get_for_dtype(&self, dtype: DType) -> Result<CisPrecomputation> {
        let key = CacheKey::new(dtype, &self.target_device);

        // Fast path: check if we have this dtype cached
        {
            let cache = self.device_cache.read().unwrap();
            if let Some(cached) = cache.get(&key) {
                return Ok(cached.clone());
            }
        }

        // Slow path: convert from CPU and cache
        let mut cache = self.device_cache.write().unwrap();

        // Double-check after acquiring write lock
        if let Some(cached) = cache.get(&key) {
            return Ok(cached.clone());
        }

        // Convert from CPU source to GPU with requested dtype
        let gpu_cis = self
            .cpu_source
            .to_dtype(dtype)?
            .to_device(&self.target_device)?;

        // Cache and return
        cache.insert(key, gpu_cis.clone());

        Ok(gpu_cis)
    }

    /// Get borrowed (cos, sin) for common dtypes.
    ///
    /// Note: This returns a clone since we can't borrow through RwLock.
    /// For performance-critical paths, consider caching the result.
    #[inline]
    pub fn get_for_dtype_borrowed(&self, dtype: DType) -> Result<(Tensor, Tensor)> {
        let cis = self.get_for_dtype(dtype)?;
        Ok((cis.cos, cis.sin))
    }

    /// Purge the device cache, freeing GPU memory.
    ///
    /// The CPU source is preserved. The next access will re-convert from CPU.
    pub fn compact(&mut self) {
        let mut cache = self.device_cache.write().unwrap();
        cache.clear();
    }

    /// Create a fixed-size RoPE table (no extension).
    pub fn new_fixed(
        head_dim: usize,
        freq_base: f32,
        max_seq_len: usize,
        device: &Device,
    ) -> Result<Self> {
        let inv_freq = default_inv_freq(head_dim, freq_base);
        Self::new_fixed_with_inv_freq(inv_freq, max_seq_len, device)
    }

    /// Create a fixed-size RoPE table (no extension) using a caller-provided inv_freq.
    pub fn new_fixed_with_inv_freq(
        inv_freq: Vec<f32>,
        max_seq_len: usize,
        device: &Device,
    ) -> Result<Self> {
        // Compute on CPU for precision
        let inv_freq_f32 =
            Tensor::new(inv_freq.as_slice(), &Device::Cpu)?.reshape((1, inv_freq.len()))?;
        let cpu_source =
            precomput_freqs_cis_internal(max_seq_len, &inv_freq_f32, &Device::Cpu, DType::F32)?;

        Ok(Self {
            cpu_source,
            device_cache: RwLock::new(HashMap::new()),
            target_device: device.clone(),
            inv_freq_f32: None, // Fixed-size tables don't need inv_freq
            max_seq_len,
            extend_chunk: 0,
        })
    }

    /// Create a growable RoPE table (chunked extension). If `initial_len` is 0, a CUDA-safe
    /// 1-row placeholder is allocated but the table is treated as logically empty.
    pub fn new_growable(
        head_dim: usize,
        freq_base: f32,
        initial_len: usize,
        extend_chunk: usize,
        device: &Device,
    ) -> Result<Self> {
        if extend_chunk == 0 {
            candle::bail!("extend_chunk must be non-zero for growable RoPE")
        }

        let inv_freq = default_inv_freq(head_dim, freq_base);
        Self::new_growable_with_inv_freq(inv_freq, initial_len, extend_chunk, device)
    }

    /// Create a growable RoPE table (chunked extension) using a caller-provided inv_freq.
    ///
    /// This keeps the shared RoPE table code model-agnostic (e.g. Llama-3 scaling can live
    /// in the Llama model code).
    pub fn new_growable_with_inv_freq(
        inv_freq: Vec<f32>,
        initial_len: usize,
        extend_chunk: usize,
        device: &Device,
    ) -> Result<Self> {
        if extend_chunk == 0 {
            candle::bail!("extend_chunk must be non-zero for growable RoPE")
        }

        // Store inv_freq on CPU
        let inv_freq_f32 =
            Tensor::new(inv_freq.as_slice(), &Device::Cpu)?.reshape((1, inv_freq.len()))?;

        // Compute CPU source
        let cpu_source =
            precomput_freqs_cis_internal(initial_len, &inv_freq_f32, &Device::Cpu, DType::F32)?;

        Ok(Self {
            cpu_source,
            device_cache: RwLock::new(HashMap::new()),
            target_device: device.clone(),
            inv_freq_f32: Some(inv_freq_f32),
            max_seq_len: initial_len,
            extend_chunk,
        })
    }

    /// Ensure the tables have at least `required_len` positions.
    pub fn ensure_len(&mut self, required_len: usize) -> Result<()> {
        if required_len == 0 {
            return Ok(());
        }
        if required_len <= self.max_seq_len {
            return Ok(());
        }

        let inv_freq_f32 = self
            .inv_freq_f32
            .as_ref()
            .ok_or_else(|| candle::Error::Msg("rope tables are fixed-size".into()))?;

        let new_len = required_len.div_ceil(self.extend_chunk) * self.extend_chunk;

        // If the tables are logically empty (max_seq_len == 0), replace entirely
        if self.max_seq_len == 0 {
            self.cpu_source =
                precomput_freqs_cis_internal(new_len, inv_freq_f32, &Device::Cpu, DType::F32)?;
            self.max_seq_len = new_len;
            // Invalidate device cache
            self.device_cache.write().unwrap().clear();
            return Ok(());
        }

        let add_len = new_len
            .checked_sub(self.max_seq_len)
            .ok_or_else(|| candle::Error::Msg("rope extend underflow".into()))?;
        if add_len == 0 {
            self.max_seq_len = new_len;
            return Ok(());
        }

        // Extend CPU source
        let (cos_add, sin_add) =
            cos_sin_for_range(inv_freq_f32, self.max_seq_len, add_len, DType::F32)?;

        self.cpu_source.cos = Tensor::cat(&[&self.cpu_source.cos, &cos_add], 0)?;
        self.cpu_source.sin = Tensor::cat(&[&self.cpu_source.sin, &sin_add], 0)?;

        self.max_seq_len = new_len;

        // Invalidate device cache (it will be re-created on next access)
        self.device_cache.write().unwrap().clear();

        Ok(())
    }

    /// Narrow into the RoPE tables along dim 0, growing if configured.
    pub fn narrow_growable(
        &mut self,
        dim: usize,
        start: usize,
        len: usize,
        dtype: DType,
    ) -> Result<(Tensor, Tensor)> {
        if dim != 0 {
            candle::bail!("unsupported rope narrow dim {dim}, expected 0")
        }
        let end = start
            .checked_add(len)
            .ok_or_else(|| candle::Error::Msg("rope narrow overflow".into()))?;
        self.ensure_len(end)?;
        let cis = self.get_for_dtype(dtype)?;
        Ok((
            cis.cos.narrow(dim, start, len)?,
            cis.sin.narrow(dim, start, len)?,
        ))
    }

    /// Narrow into fixed-size RoPE tables along dim 0.
    pub fn narrow_fixed(
        &self,
        dim: usize,
        start: usize,
        len: usize,
        dtype: DType,
    ) -> Result<(Tensor, Tensor)> {
        if dim != 0 {
            candle::bail!("unsupported rope narrow dim {dim}, expected 0")
        }
        let cis = self.get_for_dtype(dtype)?;
        Ok((
            cis.cos.narrow(dim, start, len)?,
            cis.sin.narrow(dim, start, len)?,
        ))
    }

    /// Compute per-batch (cos, sin) for `seq_len == 1` decode.
    ///
    /// Returns tensors shaped like (B, 1, rope_dim/2), matching `decode_utils::gather_rope_cos_sin`.
    pub fn cos_sin_for_offsets(
        &self,
        offsets_t: &Tensor,
        dtype: DType,
    ) -> Result<(Tensor, Tensor)> {
        let inv_freq_f32 = self
            .inv_freq_f32
            .as_ref()
            .ok_or_else(|| candle::Error::Msg("rope tables are fixed-size".into()))?;

        let b = offsets_t.dim(0)?;
        if b == 0 {
            candle::bail!("offsets_t must be non-empty")
        }

        // Compute on the target device
        let inv_freq_device = inv_freq_f32.to_device(&self.target_device)?;
        let pos = offsets_t.to_dtype(DType::F32)?.reshape((b, 1))?;
        let freqs = pos.matmul(&inv_freq_device)?;
        let mut cos = freqs.cos()?.to_dtype(dtype)?;
        let mut sin = freqs.sin()?.to_dtype(dtype)?;
        if !cos.is_contiguous() {
            cos = cos.contiguous()?;
        }
        if !sin.is_contiguous() {
            sin = sin.contiguous()?;
        }
        Ok((
            cos.reshape((b, 1, cos.dim(1)?))?,
            sin.reshape((b, 1, sin.dim(1)?))?,
        ))
    }
}

fn default_inv_freq(head_dim: usize, freq_base: f32) -> Vec<f32> {
    (0..head_dim)
        .step_by(2)
        .map(|i| 1f32 / freq_base.powf(i as f32 / head_dim as f32))
        .collect()
}

fn cos_sin_for_range(
    inv_freq_f32: &Tensor,
    start: usize,
    len: usize,
    dtype: DType,
) -> Result<(Tensor, Tensor)> {
    if len == 0 {
        candle::bail!("rope len must be non-zero")
    }
    let end = start
        .checked_add(len)
        .ok_or_else(|| candle::Error::Msg("rope range overflow".into()))?;
    let device = inv_freq_f32.device();

    let pos = Tensor::arange(start as u32, end as u32, device)?
        .to_dtype(DType::F32)?
        .reshape((len, 1))?;
    let freqs = pos.matmul(inv_freq_f32)?;
    Ok((freqs.cos()?.to_dtype(dtype)?, freqs.sin()?.to_dtype(dtype)?))
}

fn precomput_freqs_cis_internal(
    max_seq_len: usize,
    inv_freq_f32: &Tensor,
    device: &Device,
    dtype: DType,
) -> Result<CisPrecomputation> {
    if max_seq_len == 0 {
        let d = inv_freq_f32.dim(1)?;
        // CUDA backend does not support 0-sized allocations. Allocate a 1-row placeholder;
        // the table is still treated as logically empty via `max_seq_len == 0`.
        let cos = Tensor::zeros((1, d), dtype, device)?;
        let sin = Tensor::zeros((1, d), dtype, device)?;
        return Ok(CisPrecomputation { cos, sin });
    }

    let idx_theta = Tensor::arange(0, max_seq_len as u32, device)?
        .to_dtype(DType::F32)?
        .reshape((max_seq_len, 1))?
        .matmul(inv_freq_f32)?;
    Ok(CisPrecomputation {
        cos: idx_theta.cos()?.to_dtype(dtype)?,
        sin: idx_theta.sin()?.to_dtype(dtype)?,
    })
}
