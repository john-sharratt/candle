//! Fletcher-32 KV-chunk golden checksum — the Rust side of the GPU kernel
//! (`candle-kernels` `simple/fletcher32.cu`).
//!
//! A golden plan is a flat list of `(src_ptr, byte_len)` records, each carrying
//! a device address the caller has already resolved — the same plan model as
//! [`super::migrate`]. [`fletcher32_golden`] checksums every record in a single
//! kernel launch and returns one `u32` per record: the KV bytes are hashed in
//! place on the GPU, so only the small result array crosses the bus. Computed
//! over the freshly-quantized arena bytes before any device→host copy, the
//! golden is the ground truth a later CPU recompute ([`candle::fletcher::fletcher32`])
//! checks the warm/cold copy against.

#[cfg(feature = "cuda")]
use candle::cuda_backend::cudarc::driver::CudaStream;

/// One record in a golden-checksum plan: `byte_len` bytes at `src_ptr` (a
/// resolved device address) to be checksummed in place on the GPU.
#[cfg(feature = "cuda")]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GoldenRecord {
    pub src_ptr: i64,
    pub byte_len: i64,
}

/// Compute a Fletcher-32 golden for each record's device byte span in one
/// kernel launch, one checksum per record with input order preserved. Uses the
/// default decode stream; see [`fletcher32_golden_on`] to pick a stream.
#[cfg(feature = "cuda")]
pub fn fletcher32_golden(
    device: &candle::Device,
    records: &[GoldenRecord],
) -> candle::Result<Vec<u32>> {
    fletcher32_golden_on(device, records, None)
}

/// [`fletcher32_golden`] on a caller-chosen stream.
#[cfg(feature = "cuda")]
pub fn fletcher32_golden_on(
    device: &candle::Device,
    records: &[GoldenRecord],
    stream: Option<&CudaStream>,
) -> candle::Result<Vec<u32>> {
    use candle::cuda_backend::cudarc::driver::DevicePtr;
    use candle::cuda_backend::kernels;

    let dev = match device {
        candle::Device::Cuda(d) => d,
        _ => {
            return Err(candle::Error::Msg(
                "fletcher32_golden requires a CUDA device".into(),
            ))
        }
    };
    if records.is_empty() {
        return Ok(Vec::new());
    }

    let src: Vec<i64> = records.iter().map(|r| r.src_ptr).collect();
    let lens: Vec<i64> = records.iter().map(|r| r.byte_len).collect();

    let src_gpu = dev
        .memcpy_stod(&src)
        .map_err(|e| candle::Error::Msg(format!("fletcher32_golden: src plan HtoD: {e}")))?;
    let len_gpu = dev
        .memcpy_stod(&lens)
        .map_err(|e| candle::Error::Msg(format!("fletcher32_golden: len plan HtoD: {e}")))?;
    let out_gpu = unsafe { dev.alloc::<u32>(records.len()) }
        .map_err(|e| candle::Error::Msg(format!("fletcher32_golden: out alloc: {e}")))?;

    let default_stream = dev.cuda_stream();
    let used_stream = stream.unwrap_or(&default_stream);
    // Same cross-stream ordering hazard as kv_migrate: on a dedicated stream the
    // plan uploads and the freshly-quantized source bytes retire in PRIMARY-stream
    // order, so drain the device once before the cross-stream launch. The
    // default-stream path is same-stream FIFO and needs no fence.
    if stream.is_some() {
        device
            .synchronize()
            .map_err(|e| candle::Error::Msg(format!("fletcher32_golden: pre-launch fence: {e}")))?;
    }
    let (sp, _sg) = src_gpu.device_ptr(used_stream);
    let (lp, _lg) = len_gpu.device_ptr(used_stream);
    let (op, _og) = out_gpu.device_ptr(used_stream);
    unsafe {
        candle::set_kernel_breadcrumb("run_fletcher32", file!(), line!());
        kernels::simple::fletcher32::run_fletcher32(
            sp as *const i64,
            lp as *const i64,
            op as *mut u32,
            records.len() as i32,
            used_stream.cu_stream() as *mut std::ffi::c_void,
        );
    }
    used_stream
        .synchronize()
        .map_err(|e| candle::Error::Msg(format!("fletcher32_golden: stream sync: {e}")))?;

    dev.memcpy_dtov(&out_gpu)
        .map_err(|e| candle::Error::Msg(format!("fletcher32_golden: out DtoH: {e}")))
}

#[cfg(all(test, feature = "cuda"))]
mod tests {
    use super::{fletcher32_golden, GoldenRecord};
    use candle::cuda_backend::cudarc::driver::DevicePtr;
    use candle::fletcher::fletcher32;

    #[test]
    fn golden_matches_cpu_reference_across_sizes() {
        let device = match candle::Device::cuda_if_available(0) {
            Ok(d @ candle::Device::Cuda(_)) => d,
            _ => return, // no GPU — skip
        };
        let dev = match &device {
            candle::Device::Cuda(d) => d,
            _ => unreachable!(),
        };

        // Varied lengths & alignments: the canonical vector, even/odd lengths,
        // a single byte, a long run that forces the running reduction, and an
        // empty span (len 0 → checksum 0).
        let chunks: Vec<Vec<u8>> = vec![
            b"abcdefgh".to_vec(),
            (0..256u32).map(|i| (i % 256) as u8).collect(),
            (0..177u32).map(|i| ((i * 13 + 1) % 256) as u8).collect(),
            vec![0x61u8],
            vec![0xFFu8; 4096],
            Vec::new(),
            (0..2048u32).map(|i| ((i * 7 + 3) % 256) as u8).collect(),
        ];

        // Every chunk gets a device allocation (>=1 byte so the pointer is real);
        // the plan records the true length (0 for the empty span).
        let src_gpus: Vec<_> = chunks
            .iter()
            .map(|c| {
                let bytes = if c.is_empty() { vec![0u8] } else { c.clone() };
                dev.memcpy_stod(&bytes).unwrap()
            })
            .collect();

        let stream = dev.cuda_stream();
        let records: Vec<GoldenRecord> = chunks
            .iter()
            .enumerate()
            .map(|(i, c)| GoldenRecord {
                src_ptr: src_gpus[i].device_ptr(&stream).0 as i64,
                byte_len: c.len() as i64,
            })
            .collect();
        drop(stream);

        let goldens = fletcher32_golden(&device, &records).unwrap();
        assert_eq!(goldens.len(), chunks.len());
        for (i, c) in chunks.iter().enumerate() {
            assert_eq!(
                goldens[i],
                fletcher32(c),
                "chunk {i} (len {}) GPU golden != CPU reference",
                c.len()
            );
        }
        // Spot-check the canonical Fletcher-32 vector went through the GPU path.
        assert_eq!(goldens[0], 0xEBE1_9591);
    }
}
