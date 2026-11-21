//! Copies a tensor's values back to the host.
//!
//! `to_scalar` and `to_vec1` through `to_vec3` read values into host types, and
//! `to_scalar` copies a single element on its fast path. `cuda_readback`
//! reports what a readback copies off the card. A contiguous view copies its
//! own elements rather than its whole storage, so reading a band of a KV arena
//! does not move the whole arena across the bus.

use super::*;

/// The tensor API, generic over how long the storage is valid.
///
/// Unlike [`crate::quantized::LiveQTensor`], this block is *not* split into
/// `'static` constructors and `'w` accessors, because here the `-> Result<Self>`
/// methods are overwhelmingly **views** — `reshape`, `narrow`, `t`,
/// `broadcast_as` — and a view of a lease has to stay a lease. Returning `Self`
/// is what makes that true by construction. The handful of methods that
/// allocate leave `'w` free, which is sound because owned memory outlives every
/// choice of `'w`; the discipline that matters is applied at the *lease*
/// constructors instead, where `'w` is tied to a borrow of whatever owns the
/// bytes.
impl<'w> LiveTensor<'w> {
    /// Retrieves the single scalar value hold in the tensor. If the tensor contains multiple
    /// dimensions, an error is returned instead.
    pub fn to_scalar<S: crate::WithDType>(&self) -> Result<S> {
        if self.rank() != 0 {
            Err(Error::UnexpectedNumberOfDims {
                expected: 0,
                got: self.rank(),
                shape: self.shape().clone(),
            }
            .bt())?
        }
        let from_cpu_storage = |cpu_storage: &crate::CpuStorage| {
            let data = S::cpu_storage_as_slice(cpu_storage)?;
            Ok::<_, Error>(data[self.layout().start_offset()])
        };
        match &*self.storage() {
            Storage::Cpu(cpu_storage) => from_cpu_storage(cpu_storage),
            Storage::Cuda(storage) => {
                // Fast path for scalar transfers - only copy 1 element
                #[cfg(feature = "cuda")]
                {
                    // Try optimized path if type supports it, otherwise fallback
                    macro_rules! try_optimized {
                        ($ty:ty) => {
                            if S::DTYPE == <$ty as crate::WithDType>::DTYPE {
                                let val =
                                    storage.to_cpu_scalar::<$ty>(self.layout().start_offset())?;
                                return Ok(unsafe { std::mem::transmute_copy(&val) });
                            }
                        };
                    }

                    // Try each supported type
                    try_optimized!(u8);
                    try_optimized!(u32);
                    try_optimized!(i64);
                    try_optimized!(f32);
                    try_optimized!(f64);
                    try_optimized!(half::f16);
                    try_optimized!(half::bf16);

                    // Fast fallback: use slice-based transfer for all types
                    // This is much faster than to_cpu_storage() which transfers the entire buffer
                    from_cpu_storage(&storage.to_cpu_storage_scalar(self.layout().start_offset())?)
                }
                #[cfg(not(feature = "cuda"))]
                {
                    from_cpu_storage(&storage.to_cpu_storage()?)
                }
            }
            Storage::Metal(storage) => from_cpu_storage(&storage.to_cpu_storage()?),
        }
    }

    /// An alias for `to_scalar`.
    pub fn to_vec0<S: crate::WithDType>(&self) -> Result<S> {
        self.to_scalar::<S>()
    }

    /// Returns the data contained in a 1D tensor as a vector of scalar values.
    pub fn to_vec1<S: crate::WithDType>(&self) -> Result<Vec<S>> {
        if self.rank() != 1 {
            Err(Error::UnexpectedNumberOfDims {
                expected: 1,
                got: self.rank(),
                shape: self.shape().clone(),
            }
            .bt())?
        }
        let from_cpu_storage = |cpu_storage: &crate::CpuStorage, span: Option<(usize, usize)>| {
            let data = S::cpu_storage_as_slice(cpu_storage)?;
            let data = match span {
                Some((o1, o2)) => data[o1..o2].to_vec(),
                None => self.strided_index().map(|i| data[i]).collect(),
            };
            Ok::<Vec<_>, Error>(data)
        };
        match &*self.storage() {
            Storage::Cpu(storage) => from_cpu_storage(storage, self.layout.contiguous_offsets()),
            Storage::Cuda(storage) => {
                let (host, span) = self.cuda_readback(storage)?;
                from_cpu_storage(&host, span)
            }
            Storage::Metal(storage) => {
                from_cpu_storage(&storage.to_cpu_storage()?, self.layout.contiguous_offsets())
            }
        }
    }

    /// What a readback of this tensor copies off the card, and the span of it
    /// the tensor's elements occupy — `None` for a strided view, read through
    /// [`Self::strided_index`] over the whole storage.
    ///
    /// **A contiguous view copies itself, not its storage.** A view's storage
    /// is everything it was cut from, so copying that to read a 32-row band of
    /// a KV arena moved the whole arena across the bus for each band: 5 ms a
    /// band, and 0.85 s to read back a 145-token prefix that is a few
    /// kilobytes.
    pub(super) fn cuda_readback(
        &self,
        storage: &crate::CudaStorage,
    ) -> Result<(crate::CpuStorage, Option<(usize, usize)>)> {
        match self.layout.contiguous_offsets() {
            Some((o1, o2)) => Ok((storage.to_cpu_storage_range(o1, o2)?, Some((0, o2 - o1)))),
            None => Ok((storage.to_cpu_storage()?, None)),
        }
    }

    /// Returns the data contained in a 2D tensor as a vector of vector of scalar values.
    pub fn to_vec2<S: crate::WithDType>(&self) -> Result<Vec<Vec<S>>> {
        let (dim1, dim2) = self.dims2()?;
        let from_cpu_storage = |cpu_storage: &crate::CpuStorage, span: Option<(usize, usize)>| {
            let data = S::cpu_storage_as_slice(cpu_storage)?;
            let mut rows = vec![];
            match span {
                Some((o1, o2)) => {
                    let data = &data[o1..o2];
                    for idx_row in 0..dim1 {
                        rows.push(data[idx_row * dim2..(idx_row + 1) * dim2].to_vec())
                    }
                }
                None => {
                    let mut src_index = self.strided_index();
                    for _idx_row in 0..dim1 {
                        let row = (0..dim2).map(|_| data[src_index.next().unwrap()]).collect();
                        rows.push(row)
                    }
                    assert!(src_index.next().is_none());
                }
            }
            Ok(rows)
        };
        match &*self.storage() {
            Storage::Cpu(storage) => from_cpu_storage(storage, self.layout.contiguous_offsets()),
            Storage::Cuda(storage) => {
                let (host, span) = self.cuda_readback(storage)?;
                from_cpu_storage(&host, span)
            }
            Storage::Metal(storage) => {
                from_cpu_storage(&storage.to_cpu_storage()?, self.layout.contiguous_offsets())
            }
        }
    }

    /// Returns the data contained in a 3D tensor.
    pub fn to_vec3<S: crate::WithDType>(&self) -> Result<Vec<Vec<Vec<S>>>> {
        let (dim1, dim2, dim3) = self.dims3()?;
        let from_cpu_storage = |cpu_storage: &crate::CpuStorage, span: Option<(usize, usize)>| {
            let data = S::cpu_storage_as_slice(cpu_storage)?;
            let mut top_rows = vec![];
            match span {
                Some((o1, o2)) => {
                    let data = &data[o1..o2];
                    let dim23 = dim2 * dim3;
                    for idx1 in 0..dim1 {
                        let data = &data[idx1 * dim23..(idx1 + 1) * dim23];
                        let mut rows = vec![];
                        for idx2 in 0..dim2 {
                            rows.push(data[idx2 * dim3..(idx2 + 1) * dim3].to_vec())
                        }
                        top_rows.push(rows);
                    }
                }
                None => {
                    let mut src_index = self.strided_index();
                    for _idx in 0..dim1 {
                        let mut rows = vec![];
                        for _jdx in 0..dim2 {
                            let row = (0..dim3).map(|_| data[src_index.next().unwrap()]).collect();
                            rows.push(row)
                        }
                        top_rows.push(rows);
                    }
                    assert!(src_index.next().is_none());
                }
            }
            Ok(top_rows)
        };
        match &*self.storage() {
            Storage::Cpu(storage) => from_cpu_storage(storage, self.layout.contiguous_offsets()),
            Storage::Cuda(storage) => {
                let (host, span) = self.cuda_readback(storage)?;
                from_cpu_storage(&host, span)
            }
            Storage::Metal(storage) => {
                from_cpu_storage(&storage.to_cpu_storage()?, self.layout.contiguous_offsets())
            }
        }
    }
}
