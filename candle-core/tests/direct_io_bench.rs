//! Micro-benchmark: serial `read_at` vs `read_stripes_concurrent` on the real
//! expert pack, at the expert-record stride the pipeline uses. Run with
//! `cargo test -p candle-core --test direct_io_bench -- --ignored --nocapture`.

use candle_core::direct_io::{AlignedScratch, DirectFile, StripeRead};
use std::time::Instant;

#[test]
#[ignore]
fn pack_read_serial_vs_concurrent() {
    let path = std::path::PathBuf::from(r"D:\models\deepseek-v4-flash-mxfp4")
        .join("DeepSeek-V4-Flash-0731-MXFP4_KO.389b4f3e3199cd20.experts.pack");
    if !path.exists() {
        eprintln!("[skip] pack absent");
        return;
    }
    let stride: usize = 14_942_208; // ~14.2 MB, sector-rounded expert record
    let stride = candle_core::direct_io::round_up_sector(stride);
    let f = DirectFile::open(&path).expect("open pack");
    let n = 32usize;
    // Spread offsets like a layer's cold burst: consecutive records a few GB in.
    let base: u64 = 8 * 1024 * 1024 * 1024;
    let offsets: Vec<u64> = (0..n).map(|i| base + (i as u64) * stride as u64).collect();

    let mut bufs: Vec<AlignedScratch> = (0..n)
        .map(|_| {
            let mut a = AlignedScratch::new();
            a.ensure(stride).expect("scratch");
            a
        })
        .collect();

    // Warm-up one read (open cost, drive spin-up).
    f.read_at(offsets[0], bufs[0].as_mut_slice(stride)).unwrap();

    let t = Instant::now();
    for (i, &off) in offsets.iter().enumerate() {
        f.read_at(off, bufs[i].as_mut_slice(stride)).unwrap();
    }
    let serial = t.elapsed();

    let t = Instant::now();
    {
        let mut stripes: Vec<StripeRead<'_>> = Vec::with_capacity(n);
        for (i, buf) in bufs.iter_mut().enumerate() {
            stripes.push(StripeRead {
                file_offset: offsets[i],
                dest: buf.as_mut_slice(stride),
            });
        }
        f.read_stripes_concurrent(&mut stripes).unwrap();
    }
    let conc = t.elapsed();

    let gb = (n * stride) as f64 / 1e9;
    eprintln!(
        "[pack-io] {n} × {:.1} MB: serial {:.3}s ({:.2} GB/s) | concurrent {:.3}s ({:.2} GB/s)",
        stride as f64 / 1e6,
        serial.as_secs_f64(),
        gb / serial.as_secs_f64(),
        conc.as_secs_f64(),
        gb / conc.as_secs_f64(),
    );
}
