//! In-place and sparse element updates on CPU storage, which change the buffer
//! directly instead of working on a cloned copy.
//!
//! The `_mut` helpers add, subtract, multiply or divide a value, or a value per
//! index, at a list of indices. Each mutates the storage it is called on, so a
//! sparse update does not clone the whole buffer. `as_slice` and `concat` give
//! access to the stored elements and join storages.

use super::*;

impl CpuStorage {
    pub fn as_slice<D: WithDType>(&self) -> Result<&[D]> {
        D::cpu_storage_as_slice(self)
    }

    pub fn concat(storages: &[CpuStorage]) -> Result<CpuStorage> {
        let storage0 = &storages[0];
        let s = match storage0 {
            Self::U8(_) => {
                let storages = storages
                    .iter()
                    .map(|s| match s {
                        Self::U8(s) => Ok(s.as_slice()),
                        _ => crate::bail!("dtype mismatch"),
                    })
                    .collect::<Result<Vec<_>>>()?
                    .concat();
                Self::U8(storages)
            }
            Self::U32(_) => {
                let storages = storages
                    .iter()
                    .map(|s| match s {
                        Self::U32(s) => Ok(s.as_slice()),
                        _ => crate::bail!("dtype mismatch"),
                    })
                    .collect::<Result<Vec<_>>>()?
                    .concat();
                Self::U32(storages)
            }
            Self::I64(_) => {
                let storages = storages
                    .iter()
                    .map(|s| match s {
                        Self::I64(s) => Ok(s.as_slice()),
                        _ => crate::bail!("dtype mismatch"),
                    })
                    .collect::<Result<Vec<_>>>()?
                    .concat();
                Self::I64(storages)
            }
            Self::BF16(_) => {
                let storages = storages
                    .iter()
                    .map(|s| match s {
                        Self::BF16(s) => Ok(s.as_slice()),
                        _ => crate::bail!("dtype mismatch"),
                    })
                    .collect::<Result<Vec<_>>>()?
                    .concat();
                Self::BF16(storages)
            }
            Self::F16(_) => {
                let storages = storages
                    .iter()
                    .map(|s| match s {
                        Self::F16(s) => Ok(s.as_slice()),
                        _ => crate::bail!("dtype mismatch"),
                    })
                    .collect::<Result<Vec<_>>>()?
                    .concat();
                Self::F16(storages)
            }
            Self::F32(_) => {
                let storages = storages
                    .iter()
                    .map(|s| match s {
                        Self::F32(s) => Ok(s.as_slice()),
                        _ => crate::bail!("dtype mismatch"),
                    })
                    .collect::<Result<Vec<_>>>()?
                    .concat();
                Self::F32(storages)
            }
            Self::F64(_) => {
                let storages = storages
                    .iter()
                    .map(|s| match s {
                        Self::F64(s) => Ok(s.as_slice()),
                        _ => crate::bail!("dtype mismatch"),
                    })
                    .collect::<Result<Vec<_>>>()?
                    .concat();
                Self::F64(storages)
            }
            Self::F8E4M3(_) => {
                let storages = storages
                    .iter()
                    .map(|s| match s {
                        Self::F8E4M3(s) => Ok(s.as_slice()),
                        _ => crate::bail!("dtype mismatch"),
                    })
                    .collect::<Result<Vec<_>>>()?
                    .concat();
                Self::F8E4M3(storages)
            }
        };
        Ok(s)
    }

    /// In-place sparse subtraction - mutates the storage directly without cloning.
    pub fn sub_at_indices_mut(
        &mut self,
        _layout: &Layout,
        indices: &[u32],
        value: f32,
    ) -> Result<()> {
        match self {
            Self::BF16(storage) => {
                let value_bf16 = bf16::from_f32(value);
                for &idx in indices {
                    storage[idx as usize] -= value_bf16;
                }
            }
            Self::F16(storage) => {
                let value_f16 = f16::from_f32(value);
                for &idx in indices {
                    storage[idx as usize] -= value_f16;
                }
            }
            Self::F32(storage) => {
                for &idx in indices {
                    storage[idx as usize] -= value;
                }
            }
            Self::F64(storage) => {
                let value_f64 = value as f64;
                for &idx in indices {
                    storage[idx as usize] -= value_f64;
                }
            }
            _ => return Err(Error::UnsupportedDTypeForOp(self.dtype(), "sub_at_indices").bt()),
        }
        Ok(())
    }

    /// In-place sparse addition - mutates the storage directly without cloning.
    /// Adds a scalar value to elements at the specified indices: data[indices[i]] += value
    pub fn add_at_indices_mut(
        &mut self,
        _layout: &Layout,
        indices: &[u32],
        value: f32,
    ) -> Result<()> {
        match self {
            Self::BF16(storage) => {
                let value_bf16 = bf16::from_f32(value);
                for &idx in indices {
                    storage[idx as usize] += value_bf16;
                }
            }
            Self::F16(storage) => {
                let value_f16 = f16::from_f32(value);
                for &idx in indices {
                    storage[idx as usize] += value_f16;
                }
            }
            Self::F32(storage) => {
                for &idx in indices {
                    storage[idx as usize] += value;
                }
            }
            Self::F64(storage) => {
                let value_f64 = value as f64;
                for &idx in indices {
                    storage[idx as usize] += value_f64;
                }
            }
            _ => return Err(Error::UnsupportedDTypeForOp(self.dtype(), "add_at_indices").bt()),
        }
        Ok(())
    }

    /// In-place sparse subtraction with per-index values - mutates the storage directly without cloning.
    /// Each index gets its own value to subtract: data[indices[i]] -= values[i]
    pub fn sub_at_indices_mut_with_values(
        &mut self,
        _layout: &Layout,
        indices: &[u32],
        values: &[f32],
    ) -> Result<()> {
        if indices.len() != values.len() {
            crate::bail!(
                "indices and values must have the same length, got {} and {}",
                indices.len(),
                values.len()
            );
        }

        match self {
            Self::BF16(storage) => {
                for (idx, &value) in indices.iter().zip(values.iter()) {
                    let value_bf16 = bf16::from_f32(value);
                    storage[*idx as usize] -= value_bf16;
                }
            }
            Self::F16(storage) => {
                for (idx, &value) in indices.iter().zip(values.iter()) {
                    let value_f16 = f16::from_f32(value);
                    storage[*idx as usize] -= value_f16;
                }
            }
            Self::F32(storage) => {
                for (idx, &value) in indices.iter().zip(values.iter()) {
                    storage[*idx as usize] -= value;
                }
            }
            Self::F64(storage) => {
                for (idx, &value) in indices.iter().zip(values.iter()) {
                    let value_f64 = value as f64;
                    storage[*idx as usize] -= value_f64;
                }
            }
            _ => {
                return Err(Error::UnsupportedDTypeForOp(
                    self.dtype(),
                    "sub_at_indices_with_values",
                )
                .bt())
            }
        }
        Ok(())
    }

    pub fn sub_at_indices(&self, _layout: &Layout, indices: &[u32], value: f32) -> Result<Self> {
        match self {
            Self::BF16(storage) => {
                let mut data = storage.to_vec();
                let value_bf16 = bf16::from_f32(value);
                for &idx in indices {
                    data[idx as usize] -= value_bf16;
                }
                Ok(Self::BF16(data))
            }
            Self::F16(storage) => {
                let mut data = storage.to_vec();
                let value_f16 = f16::from_f32(value);
                for &idx in indices {
                    data[idx as usize] -= value_f16;
                }
                Ok(Self::F16(data))
            }
            Self::F32(storage) => {
                let mut data = storage.to_vec();
                for &idx in indices {
                    data[idx as usize] -= value;
                }
                Ok(Self::F32(data))
            }
            Self::F64(storage) => {
                let mut data = storage.to_vec();
                let value_f64 = value as f64;
                for &idx in indices {
                    data[idx as usize] -= value_f64;
                }
                Ok(Self::F64(data))
            }
            _ => Err(Error::UnsupportedDTypeForOp(self.dtype(), "sub_at_indices").bt()),
        }
    }

    /// In-place sparse multiplication - mutates the storage directly without cloning.
    pub fn mul_at_indices_mut(
        &mut self,
        _layout: &Layout,
        indices: &[u32],
        value: f32,
    ) -> Result<()> {
        match self {
            Self::BF16(storage) => {
                let value_bf16 = bf16::from_f32(value);
                for &idx in indices {
                    storage[idx as usize] *= value_bf16;
                }
            }
            Self::F16(storage) => {
                let value_f16 = f16::from_f32(value);
                for &idx in indices {
                    storage[idx as usize] *= value_f16;
                }
            }
            Self::F32(storage) => {
                for &idx in indices {
                    storage[idx as usize] *= value;
                }
            }
            Self::F64(storage) => {
                let value_f64 = value as f64;
                for &idx in indices {
                    storage[idx as usize] *= value_f64;
                }
            }
            _ => return Err(Error::UnsupportedDTypeForOp(self.dtype(), "mul_at_indices").bt()),
        }
        Ok(())
    }

    /// In-place sparse division - mutates the storage directly without cloning.
    pub fn div_at_indices_mut(
        &mut self,
        _layout: &Layout,
        indices: &[u32],
        value: f32,
    ) -> Result<()> {
        match self {
            Self::BF16(storage) => {
                let value_bf16 = bf16::from_f32(value);
                for &idx in indices {
                    storage[idx as usize] /= value_bf16;
                }
            }
            Self::F16(storage) => {
                let value_f16 = f16::from_f32(value);
                for &idx in indices {
                    storage[idx as usize] /= value_f16;
                }
            }
            Self::F32(storage) => {
                for &idx in indices {
                    storage[idx as usize] /= value;
                }
            }
            Self::F64(storage) => {
                let value_f64 = value as f64;
                for &idx in indices {
                    storage[idx as usize] /= value_f64;
                }
            }
            _ => return Err(Error::UnsupportedDTypeForOp(self.dtype(), "div_at_indices").bt()),
        }
        Ok(())
    }

    pub fn div_at_indices(&self, _layout: &Layout, indices: &[u32], value: f32) -> Result<Self> {
        match self {
            Self::BF16(storage) => {
                let mut data = storage.to_vec();
                let value_bf16 = bf16::from_f32(value);
                for &idx in indices {
                    data[idx as usize] /= value_bf16;
                }
                Ok(Self::BF16(data))
            }
            Self::F16(storage) => {
                let mut data = storage.to_vec();
                let value_f16 = f16::from_f32(value);
                for &idx in indices {
                    data[idx as usize] /= value_f16;
                }
                Ok(Self::F16(data))
            }
            Self::F32(storage) => {
                let mut data = storage.to_vec();
                for &idx in indices {
                    data[idx as usize] /= value;
                }
                Ok(Self::F32(data))
            }
            Self::F64(storage) => {
                let mut data = storage.to_vec();
                let value_f64 = value as f64;
                for &idx in indices {
                    data[idx as usize] /= value_f64;
                }
                Ok(Self::F64(data))
            }
            _ => Err(Error::UnsupportedDTypeForOp(self.dtype(), "div_at_indices").bt()),
        }
    }
}
