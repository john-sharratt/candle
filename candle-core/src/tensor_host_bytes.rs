//! Writing raw bytes into a host byte tensor in place.
//!
//! A host KV arena is one contiguous `U8` tensor, and a hot→warm migrate
//! writes hundreds of thousands of small bands into it. Through `slice_set`
//! every band first became a tensor of its own — an allocation and a copy
//! before the write — which held the scatter at ~1.4 µs a band. This copies
//! the caller's bytes straight into the slab's storage, under the same storage
//! lock `slice_set` takes.

use crate::cpu_backend::CpuStorage;
use crate::{DType, Result, Storage, Tensor};

impl Tensor {
    /// Copy `src` into this tensor's bytes, `byte_offset` bytes in, in place.
    ///
    /// The tensor must be a contiguous `U8` tensor on the CPU, and the write must
    /// lie inside it. Like [`Tensor::slice_set`], it mutates the storage every
    /// view of this tensor shares, and is not compatible with back-propagation.
    pub fn write_host_bytes(&self, byte_offset: usize, src: &[u8]) -> Result<()> {
        if self.dtype() != DType::U8 {
            crate::bail!(
                "write_host_bytes: a U8 tensor is required, got {:?}",
                self.dtype()
            );
        }
        let (mut storage, layout) = self.storage_mut_and_layout();
        if !layout.is_contiguous() {
            crate::bail!("write_host_bytes: the tensor must be contiguous");
        }
        let len = self.elem_count();
        let end = byte_offset
            .checked_add(src.len())
            .filter(|&end| end <= len)
            .ok_or_else(|| {
                crate::Error::Msg(format!(
                    "write_host_bytes: bytes {byte_offset}..{} lie past the {len}-byte tensor",
                    byte_offset.saturating_add(src.len())
                ))
            })?;
        match &mut *storage {
            Storage::Cpu(CpuStorage::U8(bytes)) => {
                let start = layout.start_offset();
                bytes[start + byte_offset..start + end].copy_from_slice(src);
                Ok(())
            }
            _ => crate::bail!("write_host_bytes: the tensor is not on the host"),
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::{DType, Device, Tensor};

    #[test]
    fn bytes_land_in_place_and_nowhere_else() {
        let t = Tensor::zeros(8, DType::U8, &Device::Cpu).unwrap();
        t.write_host_bytes(3, &[7, 8, 9]).unwrap();
        assert_eq!(t.to_vec1::<u8>().unwrap(), [0, 0, 0, 7, 8, 9, 0, 0]);
    }

    /// A view starting partway into its storage writes relative to the view.
    #[test]
    fn a_view_writes_from_its_own_start() {
        let base = Tensor::zeros(8, DType::U8, &Device::Cpu).unwrap();
        let view = base.narrow(0, 2, 4).unwrap();
        view.write_host_bytes(1, &[5, 6]).unwrap();
        assert_eq!(base.to_vec1::<u8>().unwrap(), [0, 0, 0, 5, 6, 0, 0, 0]);
    }

    #[test]
    fn a_write_past_the_end_is_refused() {
        let t = Tensor::zeros(4, DType::U8, &Device::Cpu).unwrap();
        let e = t.write_host_bytes(2, &[1, 2, 3]).unwrap_err().to_string();
        assert!(e.contains("lie past"), "{e}");
        assert_eq!(t.to_vec1::<u8>().unwrap(), [0, 0, 0, 0]);
    }

    #[test]
    fn another_dtype_is_refused() {
        let t = Tensor::zeros(4, DType::F32, &Device::Cpu).unwrap();
        assert!(t.write_host_bytes(0, &[1]).is_err());
    }
}
