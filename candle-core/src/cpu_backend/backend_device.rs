//! The `BackendDevice` implementation for the CPU, which has no device handle.
//!
//! `CpuDevice` is a zero-sized marker. The allocation code is unsafe, but it is
//! only applied to types that are `Copy` and not `Drop`, and for which every bit
//! pattern is a valid value. The comment cites the clippy issue that records the
//! risk of this pattern.

use super::*;

impl BackendDevice for CpuDevice {
    type Storage = CpuStorage;

    fn location(&self) -> crate::DeviceLocation {
        crate::DeviceLocation::Cpu
    }

    fn same_device(&self, _: &Self) -> bool {
        true
    }

    fn storage_from_slice<T: crate::WithDType>(&self, s: &[T]) -> Result<Self::Storage> {
        Ok(T::to_cpu_storage(s))
    }

    fn storage_from_cpu_storage(&self, s: &CpuStorage) -> Result<Self::Storage> {
        Ok(s.clone())
    }

    fn storage_from_cpu_storage_owned(&self, s: CpuStorage) -> Result<Self::Storage> {
        Ok(s)
    }

    fn new(_: usize) -> Result<Self> {
        Ok(Self)
    }

    fn set_seed(&self, _seed: u64) -> Result<()> {
        crate::bail!("cannot seed the CPU rng with set_seed")
    }

    fn rand_uniform(&self, shape: &Shape, dtype: DType, min: f64, max: f64) -> Result<CpuStorage> {
        use rand::prelude::*;

        let elem_count = shape.elem_count();
        let mut rng = rand::rng();
        match dtype {
            DType::U8 | DType::U32 | DType::I64 => {
                Err(Error::UnsupportedDTypeForOp(dtype, "rand_uniform").bt())
            }
            DType::BF16 => {
                let mut data = Vec::with_capacity(elem_count);
                let uniform = rand::distr::Uniform::new(bf16::from_f64(min), bf16::from_f64(max))
                    .map_err(Error::wrap)?;
                for _i in 0..elem_count {
                    data.push(rng.sample::<bf16, _>(uniform))
                }
                Ok(CpuStorage::BF16(data))
            }
            DType::F16 => {
                let mut data = Vec::with_capacity(elem_count);
                let uniform = rand::distr::Uniform::new(f16::from_f64(min), f16::from_f64(max))
                    .map_err(Error::wrap)?;
                for _i in 0..elem_count {
                    data.push(rng.sample::<f16, _>(uniform))
                }
                Ok(CpuStorage::F16(data))
            }
            DType::F8E4M3 => {
                let mut data = Vec::with_capacity(elem_count);
                let uniform =
                    rand::distr::Uniform::new(F8E4M3::from_f64(min), F8E4M3::from_f64(max))
                        .map_err(Error::wrap)?;
                for _i in 0..elem_count {
                    data.push(rng.sample::<F8E4M3, _>(uniform))
                }
                Ok(CpuStorage::F8E4M3(data))
            }
            DType::F32 => {
                let mut data = Vec::with_capacity(elem_count);
                let uniform =
                    rand::distr::Uniform::new(min as f32, max as f32).map_err(Error::wrap)?;
                for _i in 0..elem_count {
                    data.push(rng.sample::<f32, _>(uniform))
                }
                Ok(CpuStorage::F32(data))
            }
            DType::F64 => {
                let mut data = Vec::with_capacity(elem_count);
                let uniform = rand::distr::Uniform::new(min, max).map_err(Error::wrap)?;
                for _i in 0..elem_count {
                    data.push(rng.sample::<f64, _>(uniform))
                }
                Ok(CpuStorage::F64(data))
            }
        }
    }

    fn rand_normal(&self, shape: &Shape, dtype: DType, mean: f64, std: f64) -> Result<CpuStorage> {
        use rand::prelude::*;

        let elem_count = shape.elem_count();
        let mut rng = rand::rng();
        match dtype {
            DType::U8 | DType::U32 | DType::I64 => {
                Err(Error::UnsupportedDTypeForOp(dtype, "rand_normal").bt())
            }
            DType::BF16 => {
                let mut data = Vec::with_capacity(elem_count);
                let normal = rand_distr::Normal::new(bf16::from_f64(mean), bf16::from_f64(std))
                    .map_err(Error::wrap)?;
                for _i in 0..elem_count {
                    data.push(normal.sample(&mut rng))
                }
                Ok(CpuStorage::BF16(data))
            }
            DType::F16 => {
                let mut data = Vec::with_capacity(elem_count);
                let normal = rand_distr::Normal::new(f16::from_f64(mean), f16::from_f64(std))
                    .map_err(Error::wrap)?;
                for _i in 0..elem_count {
                    data.push(normal.sample(&mut rng))
                }
                Ok(CpuStorage::F16(data))
            }
            DType::F8E4M3 => {
                let mut data = Vec::with_capacity(elem_count);
                let normal = rand_distr::Normal::new(F8E4M3::from_f64(mean), F8E4M3::from_f64(std))
                    .map_err(Error::wrap)?;
                for _i in 0..elem_count {
                    data.push(normal.sample(&mut rng))
                }
                Ok(CpuStorage::F8E4M3(data))
            }
            DType::F32 => {
                let mut data = Vec::with_capacity(elem_count);
                let normal =
                    rand_distr::Normal::new(mean as f32, std as f32).map_err(Error::wrap)?;
                for _i in 0..elem_count {
                    data.push(normal.sample(&mut rng))
                }
                Ok(CpuStorage::F32(data))
            }
            DType::F64 => {
                let mut data = Vec::with_capacity(elem_count);
                let normal = rand_distr::Normal::new(mean, std).map_err(Error::wrap)?;
                for _i in 0..elem_count {
                    data.push(normal.sample(&mut rng))
                }
                Ok(CpuStorage::F64(data))
            }
        }
    }

    #[allow(clippy::uninit_vec)]
    unsafe fn alloc_uninit(&self, shape: &Shape, dtype: DType) -> Result<CpuStorage> {
        let elem_count = shape.elem_count();
        // The code below is highly unsafe but hopefully not directly unsound as we only consider
        // types that are Copy, not Drop, and for which all bit patterns are proper values.
        // It's still pretty risky, see the following for more details:
        // https://github.com/rust-lang/rust-clippy/issues/4483
        let storage = match dtype {
            DType::U8 => {
                let mut v = Vec::with_capacity(elem_count);
                v.set_len(elem_count);
                CpuStorage::U8(v)
            }
            DType::U32 => {
                let mut v = Vec::with_capacity(elem_count);
                v.set_len(elem_count);
                CpuStorage::U32(v)
            }
            DType::I64 => {
                let mut v = Vec::with_capacity(elem_count);
                v.set_len(elem_count);
                CpuStorage::I64(v)
            }
            DType::BF16 => {
                let mut v = Vec::with_capacity(elem_count);
                v.set_len(elem_count);
                CpuStorage::BF16(v)
            }
            DType::F16 => {
                let mut v = Vec::with_capacity(elem_count);
                v.set_len(elem_count);
                CpuStorage::F16(v)
            }
            DType::F32 => {
                let mut v = Vec::with_capacity(elem_count);
                v.set_len(elem_count);
                CpuStorage::F32(v)
            }
            DType::F64 => {
                let mut v = Vec::with_capacity(elem_count);
                v.set_len(elem_count);
                CpuStorage::F64(v)
            }
            DType::F8E4M3 => {
                let mut v = Vec::with_capacity(elem_count);
                v.set_len(elem_count);
                CpuStorage::F8E4M3(v)
            }
        };
        Ok(storage)
    }

    fn zeros_impl(&self, shape: &Shape, dtype: DType) -> Result<CpuStorage> {
        let elem_count = shape.elem_count();
        let storage = match dtype {
            DType::U8 => CpuStorage::U8(vec![0u8; elem_count]),
            DType::U32 => CpuStorage::U32(vec![0u32; elem_count]),
            DType::I64 => CpuStorage::I64(vec![0i64; elem_count]),
            DType::BF16 => CpuStorage::BF16(vec![bf16::ZERO; elem_count]),
            DType::F16 => CpuStorage::F16(vec![f16::ZERO; elem_count]),
            DType::F8E4M3 => CpuStorage::F8E4M3(vec![F8E4M3::ZERO; elem_count]),
            DType::F32 => CpuStorage::F32(vec![0f32; elem_count]),
            DType::F64 => CpuStorage::F64(vec![0f64; elem_count]),
        };
        Ok(storage)
    }

    fn synchronize(&self) -> Result<()> {
        Ok(())
    }
}
