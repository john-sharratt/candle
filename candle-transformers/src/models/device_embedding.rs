//! A token-embedding table kept quantized in VRAM, looked up by one kernel.
//!
//! The table stays in its checkpoint form — Q8_0 for Qwen3.8-Flash-Next, where
//! the 248,320 x 2,560 table is 644 MiB resident against 1,212 MiB as BF16 —
//! and [`DeviceEmbedding::gather_into`] turns the rows a wave names into F32 in
//! a single launch (`simple/embed_gather_dequant.cu`). There is no widened
//! copy of the table, no staging buffer for the gathered bytes, and no
//! `to_dtype` after the lookup: the kernel emits the residual stream's type
//! (hot-path invariant 1).
//!
//! # Writing into the caller's buffers
//!
//! The lookup writes destinations the caller allocated, rather than returning
//! fresh tensors, because where those rows live is the caller's decision — a
//! wave places them on its forward span, a test on the pool — and because the
//! same launch can fill two of them:
//!
//! - `wide`, `[n, replicas, ncols]`: every row written `replicas` times. A
//!   hyper-connection residual is `hc` copies of the embedding, and writing the
//!   copies here is what removes the `broadcast_as(..).contiguous()` the
//!   consumer would otherwise pay (hot-path invariant 2).
//! - `narrow`, `[n, ncols]`: every row once, for a consumer that wants the bare
//!   embedding.
//!
//! Both are fully written, so both may be allocated uninitialised (hot-path
//! invariant 6).
//!
//! # Against [`HostEmbedding`](crate::models::host_embedding::HostEmbedding)
//!
//! That one serves the table from pinned host memory over PCIe and dequantizes
//! through a staged copy. This one keeps the table on the card: it costs the
//! quantized table in VRAM and no PCIe traffic per forward.

use candle::cuda_backend::cudarc::driver::{CudaView, DevicePtr};
use candle::quantized::{GgmlDType, QStorage, QTensor};
use candle::{DType, Device, Layout, Result, Storage, Tensor};
use candle_kernels::simple::embed_gather_dequant::{
    run_embed_gather_dequant_f32, EMBED_GATHER_RAGGED_ROW, EMBED_GATHER_UNSUPPORTED_FORMAT,
};

use crate::models::operand_guard::expect_dense_dtype;

/// Elements one warp dequantizes in one step, per dispatched format — the
/// granularity a row must be a whole number of. Mirrors the kernel's dispatch,
/// which covers Q8_0 alone: its header records why the other formats'
/// `dequant.cuh` functions cannot stand in for `QTensor::dequantize`.
fn unit_elems(dtype: GgmlDType) -> Option<usize> {
    match dtype {
        GgmlDType::Q8_0 => Some(2 * dtype.block_size()),
        _ => None,
    }
}

/// A destination's F32 device view, starting at its layout's offset.
fn f32_view<'a>(storage: &'a Storage, layout: &Layout, what: &str) -> Result<CudaView<'a, f32>> {
    let Storage::Cuda(cs) = storage else {
        candle::bail!("device embedding: {what} must be a CUDA tensor");
    };
    Ok(cs.as_cuda_slice::<f32>()?.slice(layout.start_offset()..))
}

/// A quantized `[n_rows, ncols]` embedding table resident on a CUDA device.
pub struct DeviceEmbedding {
    table: QTensor,
    n_rows: usize,
    ncols: usize,
}

impl DeviceEmbedding {
    /// Take ownership of a CUDA-resident table.
    ///
    /// Refused here rather than at the first forward: a table on the wrong
    /// device, in a format the kernel has no unit for, or with rows that are not
    /// a whole number of units would otherwise fail inside the wave, where the
    /// residual it was meant to seed is already allocated uninitialised.
    pub fn new(table: QTensor) -> Result<Self> {
        let (n_rows, ncols) = table.shape().dims2()?;
        let dtype = table.dtype();
        if !matches!(table.storage(), QStorage::Cuda(_)) {
            candle::bail!("device embedding: the table must be resident on a CUDA device");
        }
        let Some(unit) = unit_elems(dtype) else {
            candle::bail!(
                "device embedding: {dtype:?} has no dequantize unit in the gather kernel, \
                 which dispatches Q8_0 only (see `simple/embed_gather_dequant.cu`)"
            );
        };
        if !ncols.is_multiple_of(unit) {
            candle::bail!(
                "device embedding: a {ncols}-wide {dtype:?} row is not a whole number of \
                 {unit}-element units"
            );
        }
        Ok(Self {
            table,
            n_rows,
            ncols,
        })
    }

    pub fn n_rows(&self) -> usize {
        self.n_rows
    }

    pub fn ncols(&self) -> usize {
        self.ncols
    }

    pub fn dtype(&self) -> GgmlDType {
        self.table.dtype()
    }

    pub fn device(&self) -> Device {
        self.table.device()
    }

    /// Bytes the table holds in VRAM.
    pub fn table_bytes(&self) -> usize {
        self.table.storage_size_in_bytes()
    }

    /// Look up `ids` and write the rows as F32 into `wide` (`[n, replicas,
    /// ncols]`, each row repeated across the middle axis) and/or `narrow`
    /// (`[n, ncols]`).
    ///
    /// `ids` is a dense U32 device tensor. Both destinations must be dense F32
    /// on the table's device — validated, never converted — and are overwritten
    /// in full. An id past the table writes zeros rather than reading out of
    /// bounds.
    pub fn gather_into(
        &self,
        ids: &Tensor,
        wide: Option<&Tensor>,
        narrow: Option<&Tensor>,
    ) -> Result<()> {
        if wide.is_none() && narrow.is_none() {
            candle::bail!("device embedding: a lookup with no destination");
        }
        if ids.dtype() != DType::U32 || !ids.is_contiguous() {
            candle::bail!(
                "device embedding: ids must be dense U32, got {:?} {:?}",
                ids.dtype(),
                ids.stride()
            );
        }
        let n = ids.elem_count();
        let replicas = match wide {
            Some(w) => {
                expect_dense_dtype(w, DType::F32, "device embedding: wide")?;
                let (wn, r, wc) = w.dims3()?;
                if (wn, wc) != (n, self.ncols) {
                    candle::bail!(
                        "device embedding: wide is [{wn}, {r}, {wc}] for {n} ids of width {}",
                        self.ncols
                    );
                }
                r
            }
            None => 0,
        };
        if let Some(t) = narrow {
            expect_dense_dtype(t, DType::F32, "device embedding: narrow")?;
            if t.dims() != [n, self.ncols] {
                candle::bail!(
                    "device embedding: narrow is {:?} for {n} ids of width {}",
                    t.dims(),
                    self.ncols
                );
            }
        }
        if n == 0 {
            return Ok(());
        }
        let QStorage::Cuda(table) = self.table.storage() else {
            candle::bail!("device embedding: the table left the device");
        };
        let Device::Cuda(cuda) = self.table.device() else {
            candle::bail!("device embedding: the table's device is not CUDA");
        };
        let stream = cuda.cuda_stream();

        let (ids_storage, ids_layout) = ids.storage_and_layout();
        let Storage::Cuda(ids_cs) = &*ids_storage else {
            candle::bail!("device embedding: ids must be a CUDA tensor");
        };
        let ids_slice = ids_cs
            .as_cuda_slice::<u32>()?
            .slice(ids_layout.start_offset()..);
        let (ids_ptr, _ids_guard) = ids_slice.device_ptr(&stream);

        // Storage borrows, views and device-pointer guards are all held until
        // the launch has been issued.
        let wide_storage = wide.map(|t| t.storage_and_layout());
        let wide_view = match &wide_storage {
            Some((s, l)) => Some(f32_view(s, l, "wide")?),
            None => None,
        };
        let wide_dp = wide_view.as_ref().map(|v| v.device_ptr(&stream));
        let wide_ptr = wide_dp
            .as_ref()
            .map_or(std::ptr::null_mut(), |(p, _)| *p as *mut f32);
        let narrow_storage = narrow.map(|t| t.storage_and_layout());
        let narrow_view = match &narrow_storage {
            Some((s, l)) => Some(f32_view(s, l, "narrow")?),
            None => None,
        };
        let narrow_dp = narrow_view.as_ref().map(|v| v.device_ptr(&stream));
        let narrow_ptr = narrow_dp
            .as_ref()
            .map_or(std::ptr::null_mut(), |(p, _)| *p as *mut f32);

        candle::set_kernel_breadcrumb("run_embed_gather_dequant_f32", file!(), line!());
        let status = unsafe {
            run_embed_gather_dequant_f32(
                table.data_ptr() as *const std::ffi::c_void,
                self.table.dtype().to_u32() as i32,
                ids_ptr as *const u32,
                wide_ptr,
                replicas as i32,
                narrow_ptr,
                self.ncols as i64,
                self.n_rows as i64,
                n as i32,
                stream.cu_stream() as *mut std::ffi::c_void,
            )
        };
        match status {
            0 => Ok(()),
            EMBED_GATHER_UNSUPPORTED_FORMAT => candle::bail!(
                "device embedding: the kernel has no unit for {:?}",
                self.table.dtype()
            ),
            EMBED_GATHER_RAGGED_ROW => candle::bail!(
                "device embedding: the kernel refused a {}-wide {:?} row",
                self.ncols,
                self.table.dtype()
            ),
            other => candle::bail!("device embedding: unexpected launcher status {other}"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::gpu_test_lock::gpu_serial as gpu_guard;
    use candle::quantized::ggml_file::qtensor_from_ggml;

    fn cuda() -> Option<Device> {
        match Device::cuda_if_available(0) {
            Ok(d) if d.is_cuda() => Some(d),
            _ => {
                eprintln!("skipping: CUDA device required");
                None
            }
        }
    }

    /// Deterministic values spread over a few orders of magnitude, so every
    /// format's scales and mins take distinct values across blocks.
    fn lcg_table(n_rows: usize, ncols: usize) -> Tensor {
        let mut s = 0x9e37_79b9_7f4a_7c15u64;
        let vals: Vec<f32> = (0..n_rows * ncols)
            .map(|i| {
                s = s
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                let u = ((s >> 33) as f32 / (1u64 << 31) as f32) - 0.5;
                u * (1.0 + (i % 7) as f32)
            })
            .collect();
        Tensor::from_vec(vals, (n_rows, ncols), &Device::Cpu).unwrap()
    }

    /// Quantize on the CPU and place the bytes on the card through the same
    /// constructor a GGUF load uses.
    fn device_table(dtype: GgmlDType, n_rows: usize, ncols: usize, dev: &Device) -> QTensor {
        let cpu = QTensor::quantize(&lcg_table(n_rows, ncols), dtype).unwrap();
        qtensor_from_ggml(dtype, &cpu.data().unwrap(), vec![n_rows, ncols], dev).unwrap()
    }

    fn bits(t: &Tensor) -> Vec<Vec<u32>> {
        t.to_vec2::<f32>()
            .unwrap()
            .into_iter()
            .map(|r| r.into_iter().map(f32::to_bits).collect())
            .collect()
    }

    /// The lookup produces exactly what `QTensor::dequantize` produces for the
    /// same rows — compared as bit patterns. The two are separate
    /// implementations (`dequant.cuh` against `simple/quantized.cu`), so this is
    /// the check that licenses a format in the kernel, not a formality.
    #[test]
    fn the_lookup_matches_the_ordinary_dequantize_bit_for_bit() {
        let _gpu = gpu_guard();
        let Some(dev) = cuda() else { return };
        let (n_rows, ncols) = (7usize, 512usize);
        // Repeats and a non-monotonic order, so a gather that ignored the ids
        // and copied a contiguous span cannot pass.
        let ids_host = [3u32, 0, 6, 3, 1];
        let ids = Tensor::new(ids_host.as_slice(), &dev).unwrap();
        // Q8_0 is the one format the kernel dispatches. A format joins the
        // kernel's switch only with a parity test of its own like this one.
        let table = device_table(GgmlDType::Q8_0, n_rows, ncols, &dev);
        let reference = table
            .dequantize(&dev)
            .unwrap()
            .index_select(&ids, 0)
            .unwrap();
        let emb = DeviceEmbedding::new(table).unwrap();
        let narrow = Tensor::empty((ids_host.len(), ncols), DType::F32, &dev).unwrap();
        emb.gather_into(&ids, None, Some(&narrow)).unwrap();
        assert_eq!(bits(&narrow), bits(&reference));
    }

    /// The production format against values written out by hand: Q8_0 is
    /// `scale * q`, one f32 product per element, so the expected output is exact
    /// and independent of any other dequantize.
    #[test]
    fn q8_0_rows_are_scale_times_quant() {
        let _gpu = gpu_guard();
        let Some(dev) = cuda() else { return };
        let (n_rows, ncols) = (4usize, 64usize);
        let mut bytes = Vec::new();
        let mut expected = vec![Vec::new(); n_rows];
        for (r, row) in expected.iter_mut().enumerate() {
            for b in 0..ncols / 32 {
                let scale = half::f16::from_f32(0.25 + (r * 3 + b) as f32 * 0.125);
                bytes.extend_from_slice(&scale.to_le_bytes());
                for i in 0..32 {
                    let q = (((r * 7 + b * 13 + i * 3) % 251) as i32 - 125) as i8;
                    bytes.push(q as u8);
                    row.push((scale.to_f32() * q as f32).to_bits());
                }
            }
        }
        let table = qtensor_from_ggml(GgmlDType::Q8_0, &bytes, vec![n_rows, ncols], &dev).unwrap();
        let emb = DeviceEmbedding::new(table).unwrap();
        let ids = Tensor::new([2u32, 0, 3].as_slice(), &dev).unwrap();
        let narrow = Tensor::empty((3, ncols), DType::F32, &dev).unwrap();
        emb.gather_into(&ids, None, Some(&narrow)).unwrap();
        let got = bits(&narrow);
        assert_eq!(got[0], expected[2]);
        assert_eq!(got[1], expected[0]);
        assert_eq!(got[2], expected[3]);
    }

    /// `wide` holds every row `replicas` times back to back, and a lookup that
    /// fills both destinations writes the same values to each.
    #[test]
    fn wide_repeats_each_row_across_the_streams() {
        let _gpu = gpu_guard();
        let Some(dev) = cuda() else { return };
        let (n_rows, ncols, hc) = (6usize, 256usize, 4usize);
        let emb = DeviceEmbedding::new(device_table(GgmlDType::Q8_0, n_rows, ncols, &dev)).unwrap();
        let ids = Tensor::new([5u32, 1, 1].as_slice(), &dev).unwrap();
        let wide = Tensor::empty((3, hc, ncols), DType::F32, &dev).unwrap();
        let narrow = Tensor::empty((3, ncols), DType::F32, &dev).unwrap();
        emb.gather_into(&ids, Some(&wide), Some(&narrow)).unwrap();
        let rows = bits(&narrow);
        let streams = wide.to_vec3::<f32>().unwrap();
        for (t, per_stream) in streams.iter().enumerate() {
            for (s, stream) in per_stream.iter().enumerate() {
                let got: Vec<u32> = stream.iter().map(|v| v.to_bits()).collect();
                assert_eq!(got, rows[t], "token {t} stream {s} differs from its row");
            }
        }
        // A wide-only lookup writes the same thing.
        let wide_only = Tensor::empty((3, hc, ncols), DType::F32, &dev).unwrap();
        emb.gather_into(&ids, Some(&wide_only), None).unwrap();
        assert_eq!(wide_only.to_vec3::<f32>().unwrap(), streams);
    }

    /// An id past the table writes zeros to every destination instead of
    /// reading out of bounds — which is also what keeps an uninitialised
    /// destination fully written.
    #[test]
    fn an_id_past_the_table_writes_zero_rows() {
        let _gpu = gpu_guard();
        let Some(dev) = cuda() else { return };
        let (n_rows, ncols, hc) = (3usize, 64usize, 2usize);
        let emb = DeviceEmbedding::new(device_table(GgmlDType::Q8_0, n_rows, ncols, &dev)).unwrap();
        let ids = Tensor::new([1u32, 3, u32::MAX].as_slice(), &dev).unwrap();
        let wide = Tensor::full(f32::NAN, (3, hc, ncols), &dev).unwrap();
        let narrow = Tensor::full(f32::NAN, (3, ncols), &dev).unwrap();
        emb.gather_into(&ids, Some(&wide), Some(&narrow)).unwrap();
        let rows = narrow.to_vec2::<f32>().unwrap();
        assert!(
            rows[0].iter().any(|v| *v != 0.0),
            "a valid id gathered zeros"
        );
        for r in [1, 2] {
            assert!(
                rows[r].iter().all(|v| v.to_bits() == 0),
                "row {r} not zeroed"
            );
            let streams = wide.get(r).unwrap().to_vec2::<f32>().unwrap();
            assert!(streams.iter().flatten().all(|v| v.to_bits() == 0));
        }
    }

    #[test]
    fn a_row_that_is_not_whole_units_is_refused() {
        let _gpu = gpu_guard();
        let Some(dev) = cuda() else { return };
        // 96 is three Q8_0 blocks: whole blocks, but not whole two-block units.
        let table = device_table(GgmlDType::Q8_0, 2, 96, &dev);
        let err = DeviceEmbedding::new(table)
            .err()
            .expect("96 is not whole units");
        assert!(err.to_string().contains("whole number"), "{err}");
    }

    /// A dense float table and a quantized format the kernel does not dispatch
    /// are both refused at construction, by name.
    #[test]
    fn a_format_without_a_unit_is_refused() {
        let _gpu = gpu_guard();
        let Some(dev) = cuda() else { return };
        let table =
            QTensor::quantize(&lcg_table(2, 64).to_device(&dev).unwrap(), GgmlDType::F32).unwrap();
        let err = DeviceEmbedding::new(table).err().expect("F32 has no unit");
        assert!(err.to_string().contains("no dequantize unit"), "{err}");
        let err = DeviceEmbedding::new(device_table(GgmlDType::Q4_0, 2, 64, &dev))
            .err()
            .expect("Q4_0 is not dispatched");
        assert!(err.to_string().contains("Q8_0 only"), "{err}");
    }

    #[test]
    fn a_misshapen_destination_is_refused() {
        let _gpu = gpu_guard();
        let Some(dev) = cuda() else { return };
        let emb = DeviceEmbedding::new(device_table(GgmlDType::Q8_0, 4, 64, &dev)).unwrap();
        let ids = Tensor::new([0u32, 1].as_slice(), &dev).unwrap();
        let short = Tensor::empty((1, 64), DType::F32, &dev).unwrap();
        assert!(emb.gather_into(&ids, None, Some(&short)).is_err());
        let bf16 = Tensor::empty((2, 64), DType::BF16, &dev).unwrap();
        assert!(emb.gather_into(&ids, None, Some(&bf16)).is_err());
        assert!(emb.gather_into(&ids, None, None).is_err());
    }
}
