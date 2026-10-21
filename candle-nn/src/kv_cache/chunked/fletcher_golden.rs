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

/// One record in a golden-checksum plan: `byte_len` bytes at `src_ptr` (a
/// resolved device address) to be checksummed in place on the GPU.
#[cfg(feature = "cuda")]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GoldenRecord {
    pub src_ptr: i64,
    pub byte_len: i64,
}

/// Compute a Fletcher-32 golden for each record's device byte span in one
/// kernel launch on the device's stream, one checksum per record with input
/// order preserved.
///
/// The plan and the results share the device's held launch-table scratch
/// ([`candle::CudaDevice::with_synced_upload`]): one upload carries both plan
/// arrays, the kernel writes its checksums beside them, and one copy reads
/// them back before the scratch is let go. Nothing is allocated — this runs on
/// the persistence thread for every sealed sequence, and it used to take three
/// buffers from the driver each time.
#[cfg(feature = "cuda")]
pub fn fletcher32_golden(
    device: &candle::Device,
    records: &[GoldenRecord],
) -> candle::Result<Vec<u32>> {
    use candle::cuda_backend::cudarc::driver::result::memcpy_dtoh_async;
    use candle::cuda_backend::kernels;
    use candle::cuda_backend::staged::segment;
    use candle::cuda_backend::WrapErr;

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
    let stream = dev.cuda_stream();
    let mut out = vec![0u32; records.len()];
    dev.with_synced_upload(
        &stream,
        &[segment(&src), segment(&lens)],
        records.len() * std::mem::size_of::<u32>(),
        |at, room| {
            // SAFETY: `at` names the uploaded plan arrays and `room` the
            // checksums' space, all held until this closure returns; the kernel
            // writes every checksum, and the readback is ordered behind the
            // launch and waited for before the scratch is released.
            unsafe {
                candle::set_kernel_breadcrumb("run_fletcher32", file!(), line!());
                kernels::simple::fletcher32::run_fletcher32(
                    at[0] as *const i64,
                    at[1] as *const i64,
                    room as *mut u32,
                    records.len() as i32,
                    stream.cu_stream() as *mut std::ffi::c_void,
                );
                memcpy_dtoh_async(&mut out, room, stream.cu_stream()).w()?;
            }
            stream.synchronize().w()
        },
    )?;
    Ok(out)
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

    /// Records starting at every offset within a 16-byte vector and ending on
    /// both sides of each word / vector boundary: the aligned head, the 16-byte
    /// body, the tail words, the odd trailing byte, and the byte-assembled
    /// odd-address path all meet in one launch over one allocation.
    #[test]
    fn golden_matches_cpu_reference_at_every_alignment() {
        let device = match candle::Device::cuda_if_available(0) {
            Ok(d @ candle::Device::Cuda(_)) => d,
            _ => return, // no GPU — skip
        };
        let dev = match &device {
            candle::Device::Cuda(d) => d,
            _ => unreachable!(),
        };

        let buf: Vec<u8> = (0..20_000u32)
            .map(|i| (i.wrapping_mul(2_654_435_761) >> 13) as u8)
            .collect();
        let src_gpu = dev.memcpy_stod(&buf).unwrap();
        let stream = dev.cuda_stream();
        let base = src_gpu.device_ptr(&stream).0 as i64;
        drop(stream);

        let lens = [
            0usize, 1, 2, 3, 15, 16, 17, 30, 31, 32, 33, 47, 48, 49, 255, 256, 257, 4095, 4097,
            16_383,
        ];
        let spans: Vec<(usize, usize)> = (0..16usize)
            .flat_map(|off| lens.iter().map(move |&len| (off, len)))
            .collect();
        let records: Vec<GoldenRecord> = spans
            .iter()
            .map(|&(off, len)| GoldenRecord {
                src_ptr: base + off as i64,
                byte_len: len as i64,
            })
            .collect();

        let goldens = fletcher32_golden(&device, &records).unwrap();
        assert_eq!(goldens.len(), spans.len());
        for (i, &(off, len)) in spans.iter().enumerate() {
            assert_eq!(
                goldens[i],
                fletcher32(&buf[off..off + len]),
                "offset {off} len {len}: GPU golden != CPU reference"
            );
        }
    }
}
