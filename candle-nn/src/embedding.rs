//! Embedding Layer with CPU-backed storage and lazy GPU conversion.
//!
//! The source embeddings are always stored on CPU to preserve precision.
//! GPU conversions are created on-demand and cached. Calling `compact()`
//! purges the device cache, and the next request will re-convert from CPU.
use candle::{DType, Device, DeviceLocation, Result, Tensor};
use std::collections::HashMap;
use std::sync::{Arc, RwLock};

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

/// Embedding layer with CPU-backed source and lazy GPU conversion.
///
/// - Source embeddings are stored on CPU (preserving original precision)
/// - GPU variants are created on-demand when `forward_as_dtype` is called
/// - `compact()` purges device cache, freeing GPU memory
#[derive(Clone, Debug)]
pub struct Embedding {
    /// Source embeddings on CPU - immutable after construction.
    cpu_source: Arc<Tensor>,
    /// Cached device variants keyed by (dtype, device).
    device_cache: Arc<RwLock<HashMap<CacheKey, Tensor>>>,
    hidden_size: usize,
}

impl Embedding {
    /// Create a new embedding layer.
    ///
    /// If the embeddings are on GPU, they will be copied to CPU for storage.
    /// The original tensor can then be dropped, freeing GPU memory.
    pub fn new(embeddings: Tensor, hidden_size: usize) -> Result<Self> {
        // Move to CPU if not already there
        let cpu_embeddings = if embeddings.device().is_cpu() {
            embeddings
        } else {
            embeddings.to_device(&Device::Cpu)?
        };

        Ok(Self {
            cpu_source: Arc::new(cpu_embeddings),
            device_cache: Arc::new(RwLock::new(HashMap::new())),
            hidden_size,
        })
    }

    /// Get embeddings on the specified device and dtype.
    ///
    /// This is the main method for getting embeddings ready for computation.
    /// Results are cached; subsequent calls with the same device/dtype are fast.
    pub fn embeddings_on_device(&self, dtype: DType, device: &Device) -> Result<Tensor> {
        let key = CacheKey::new(dtype, device);

        // Fast path: check if we already have this exact variant cached
        {
            let cache = self.device_cache.read().unwrap();
            if let Some(tensor) = cache.get(&key) {
                return Ok(tensor.clone());
            }
        }

        // Slow path: convert from CPU source and cache
        let mut cache = self.device_cache.write().unwrap();

        // Double-check after acquiring write lock
        if let Some(tensor) = cache.get(&key) {
            return Ok(tensor.clone());
        }

        // Convert from CPU source
        let converted = self.cpu_source.to_dtype(dtype)?.to_device(device)?;

        // Cache and return
        cache.insert(key, converted.clone());

        Ok(converted)
    }

    /// Get embeddings in the requested dtype from cache or CPU.
    ///
    /// This is a convenience method. For explicit device control, use `embeddings_on_device`.
    pub fn embeddings(&self, dtype: DType) -> Result<Tensor> {
        // Check if we have any cached variant with matching dtype
        {
            let cache = self.device_cache.read().unwrap();
            for (key, tensor) in cache.iter() {
                if key.dtype == dtype {
                    return Ok(tensor.clone());
                }
            }
        }

        // No matching cache - return CPU version converted to dtype
        self.cpu_source.to_dtype(dtype)
    }

    /// Get the CPU source embeddings in their native dtype.
    pub fn embeddings_native(&self) -> Tensor {
        (*self.cpu_source).clone()
    }

    /// Get the native dtype of the CPU source.
    pub fn native_dtype(&self) -> DType {
        self.cpu_source.dtype()
    }

    /// Get the hidden size of the embedding matrix.
    pub fn hidden_size(&self) -> usize {
        self.hidden_size
    }

    /// Purge the device cache, freeing GPU memory.
    ///
    /// The CPU source is preserved. The next forward pass will
    /// re-convert from CPU to GPU as needed.
    pub fn compact(&self) {
        let mut cache = self.device_cache.write().unwrap();
        cache.clear();
    }
}

impl crate::Module for Embedding {
    /// Forward pass using the native CPU dtype.
    ///
    /// Note: This returns a CPU tensor. For GPU inference, use `forward_as_dtype`
    /// with the indexes already on the target device.
    fn forward(&self, indexes: &Tensor) -> Result<Tensor> {
        let mut final_dims = indexes.dims().to_vec();
        final_dims.push(self.hidden_size);
        let indexes = indexes.flatten_all()?;

        // If indexes are on GPU, use GPU embeddings
        let device = indexes.device();
        if !device.is_cpu() {
            let key = CacheKey::new(self.native_dtype(), device);
            {
                let cache = self.device_cache.read().unwrap();
                if let Some(tensor) = cache.get(&key) {
                    let values = tensor.index_select(&indexes, 0)?;
                    return values.reshape(final_dims);
                }
            }
            // No cache for this device - create one with native dtype
            let embeddings = self.embeddings_on_device(self.native_dtype(), device)?;
            let values = embeddings.index_select(&indexes, 0)?;
            return values.reshape(final_dims);
        }

        // CPU path
        let values = self.cpu_source.index_select(&indexes, 0)?;
        values.reshape(final_dims)
    }

    /// Forward pass with explicit dtype conversion.
    ///
    /// The embeddings are converted to the requested dtype and moved to
    /// the same device as the input indexes.
    fn forward_as_dtype(&self, xs: &Tensor, dtype: DType) -> Result<Tensor> {
        let mut final_dims = xs.dims().to_vec();
        final_dims.push(self.hidden_size);
        let indexes = xs.flatten_all()?;

        let device = indexes.device();
        let embeddings = if device.is_cpu() {
            // CPU path - just convert dtype
            self.cpu_source.to_dtype(dtype)?
        } else {
            // GPU path - use cached or create
            self.embeddings_on_device(dtype, device)?
        };

        let values = embeddings.index_select(&indexes, 0)?;
        values.reshape(final_dims)
    }
}

pub fn embedding(in_size: usize, out_size: usize, vb: crate::VarBuilder) -> Result<Embedding> {
    let embeddings = vb.get_with_hints(
        (in_size, out_size),
        "weight",
        crate::Init::Randn {
            mean: 0.,
            stdev: 1.,
        },
    )?;
    Embedding::new(embeddings, out_size)
}
