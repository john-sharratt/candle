//! Microbench + `ncu` target for `embed_gather_dequant_f32_kernel`, the
//! wave's embedding lookup, at Qwen3.8-Flash-Next's geometry: a Q8_0 table
//! 2560 wide written into the 4-stream residual and the head's bare rows.
//!
//! Three shapes per row count, each gated against the dequantized table and the
//! second table **on the device** before it is timed:
//!
//! - `text`  — every id in the vocabulary, no second table: the lookup every
//!   wave makes, which must not be slowed by the second table's argument;
//! - `text+x`— the same ids with a second table bound, which none of them read;
//! - `image` — half the ids past the vocabulary, reading the second table.
//!
//! Usage:
//!   cargo run -p candle-transformers --example embed_gather_bench \
//!       --features cuda --release -- [rows] [iters]
//!
//! Clock-locked A/B (the laptop's event timings move ±20%):
//!   ncu -k embed_gather_dequant_f32_kernel --launch-skip 20 --launch-count 10 \
//!       --metrics gpu__time_duration.sum --csv \
//!       target/release/examples/embed_gather_bench 8192 20

#[cfg(feature = "cuda")]
fn main() -> candle::Result<()> {
    use candle::quantized::{GgmlDType, QTensor};
    use candle::{DType, Device, Tensor};
    use candle_transformers::models::device_embedding::DeviceEmbedding;

    const NCOLS: usize = 2560;
    const HC: usize = 4;
    const TABLE_ROWS: usize = 16_384;

    let a: Vec<String> = std::env::args().collect();
    let parse = |i: usize, d: usize| a.get(i).and_then(|s| s.parse().ok()).unwrap_or(d);
    let rows = parse(1, 8192);
    let iters = parse(2, 50);

    let dev = Device::new_cuda(0)?;
    let mut s = 0x9e37_79b9_7f4a_7c15u64;
    let mut next = || {
        s = s
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        s
    };
    let vals: Vec<f32> = (0..TABLE_ROWS * NCOLS)
        .map(|_| ((next() >> 40) as f32 / (1u64 << 24) as f32) - 0.5)
        .collect();
    let host = Tensor::from_vec(vals, (TABLE_ROWS, NCOLS), &Device::Cpu)?;
    let q = QTensor::quantize(&host, GgmlDType::Q8_0)?;
    let reference = q.dequantize(&Device::Cpu)?;
    let table = candle::quantized::ggml_file::qtensor_from_ggml(
        GgmlDType::Q8_0,
        &q.data()?,
        vec![TABLE_ROWS, NCOLS],
        &dev,
    )?;
    let emb = DeviceEmbedding::new(table)?;

    let extra_rows = rows / 2;
    let extra_vals: Vec<f32> = (0..extra_rows * NCOLS)
        .map(|i| (i % 977) as f32 * 0.01 - 4.0)
        .collect();
    let extra = Tensor::from_vec(extra_vals.clone(), (extra_rows, NCOLS), &dev)?;

    let text_ids: Vec<u32> = (0..rows)
        .map(|_| (next() % TABLE_ROWS as u64) as u32)
        .collect();
    // Every other row an image row, in order through the second table.
    let image_ids: Vec<u32> = (0..rows)
        .map(|i| {
            if i % 2 == 1 && i / 2 < extra_rows {
                (TABLE_ROWS + i / 2) as u32
            } else {
                text_ids[i]
            }
        })
        .collect();

    let wide = Tensor::empty((rows, HC, NCOLS), DType::F32, &dev)?;
    let narrow = Tensor::empty((rows, NCOLS), DType::F32, &dev)?;
    let ref_rows = reference.to_vec2::<f32>()?;

    // The device's answer for `ids`, against the tables.
    let gate = |ids: &[u32], bound: Option<&Tensor>| -> candle::Result<()> {
        let t = Tensor::new(ids, &dev)?;
        emb.gather_into(&t, bound, Some(&wide), Some(&narrow))?;
        let got = narrow.to_vec2::<f32>()?;
        let streams = wide.to_vec3::<f32>()?;
        for (r, &id) in ids.iter().enumerate().step_by(97) {
            let want: &[f32] = if (id as usize) < TABLE_ROWS {
                &ref_rows[id as usize]
            } else {
                let e = id as usize - TABLE_ROWS;
                &extra_vals[e * NCOLS..(e + 1) * NCOLS]
            };
            if got[r] != want || streams[r].iter().any(|s| s.as_slice() != want) {
                candle::bail!("row {r} (id {id}) does not match its table");
            }
        }
        Ok(())
    };
    gate(&text_ids, None)?;
    gate(&text_ids, Some(&extra))?;
    gate(&image_ids, Some(&extra))?;
    println!("correctness gate: text, text+x and image rows match their tables");

    let time = |ids: &[u32], bound: Option<&Tensor>| -> candle::Result<f64> {
        let t = Tensor::new(ids, &dev)?;
        for _ in 0..10 {
            emb.gather_into(&t, bound, Some(&wide), Some(&narrow))?;
        }
        dev.synchronize()?;
        let t0 = std::time::Instant::now();
        for _ in 0..iters {
            emb.gather_into(&t, bound, Some(&wide), Some(&narrow))?;
        }
        dev.synchronize()?;
        Ok(t0.elapsed().as_secs_f64() * 1e6 / iters as f64)
    };
    // Bytes the lookup must move: each row's Q8_0 source (34 bytes per 32
    // values) or F32 extra row read once, and (HC + 1) F32 copies written.
    let moved = |image: bool| -> f64 {
        let written = rows * (HC + 1) * NCOLS * 4;
        let q8 = NCOLS / 32 * 34;
        let read = if image {
            (rows - extra_rows) * q8 + extra_rows * NCOLS * 4
        } else {
            rows * q8
        };
        (written + read) as f64
    };
    for (name, ids, bound, image) in [
        ("text", &text_ids, None, false),
        ("text+x", &text_ids, Some(&extra), false),
        ("image", &image_ids, Some(&extra), true),
    ] {
        let us = time(ids, bound)?;
        let bytes = moved(image);
        println!(
            "{name:<7} {rows:>6} rows {us:>9.1} us/call {:>8.1} GB/s ({:.1} MiB moved)",
            bytes / (us * 1e-6) / 1e9,
            bytes / (1024.0 * 1024.0)
        );
    }
    Ok(())
}

#[cfg(not(feature = "cuda"))]
fn main() {
    eprintln!("embed_gather_bench requires the `cuda` feature");
}
