//! Microbench + `ncu` target for `image_rope_rows::kernel`, the in-place turn
//! of an image's rows of Q and K, at Qwen3.8-Flash-Next's attention geometry:
//! 256-wide heads, 32 rotary pairs, BF16.
//!
//! A wave of `rows` rows holds one image of `image` rows in the middle. Q (16
//! heads) and K (2), turned in one launch, are gated **bit for bit** against
//! the host's f32 arithmetic before they are timed: every listed row's rotary
//! pairs turned, every other value of either tensor untouched.
//!
//! Usage:
//!   cargo run -p candle-transformers --example image_rope_bench \
//!       --features cuda --release -- [rows] [image] [iters]
//!
//! Clock-locked A/B:
//!   ncu -k regex:image_rope_rows --launch-skip 20 --launch-count 10 \
//!       --metrics gpu__time_duration.sum --csv \
//!       target/release/examples/image_rope_bench 8192 4096 20

#[cfg(feature = "cuda")]
fn main() -> candle::Result<()> {
    use candle::{Device, Tensor};
    use candle_transformers::models::qwen_vision::geometry::FACTOR;
    use candle_transformers::models::qwen_vision::{
        rotation_tables, Grid, ImageRotation, ImageRun,
    };
    use half::bf16;

    const HEAD_DIM: usize = 256;
    const ROPE_DIM: usize = 64;
    const THETA: f64 = 10_000_000.0;

    let a: Vec<String> = std::env::args().collect();
    let parse = |i: usize, d: usize| a.get(i).and_then(|s| s.parse().ok()).unwrap_or(d);
    let rows = parse(1, 8192);
    let image = parse(2, 4096);
    let iters = parse(3, 50);

    // A square image of `image` rows: side √image merged rows of 2×2 patches.
    let side = (image as f64).sqrt() as usize;
    let grid = Grid {
        height: side * FACTOR,
        width: side * FACTOR,
    };
    let image = side * side;
    let start = (rows - image) / 2;
    let run = ImageRun {
        row: start,
        len: image,
        grid,
        first: 0,
    };
    let t = rotation_tables(&[run], ROPE_DIM, THETA);

    let dev = Device::new_cuda(0)?;
    let n = t.rows.len();
    let rot = ImageRotation::from_tables(
        Tensor::from_vec(t.rows.clone(), n, &dev)?,
        t.rows_end,
        Tensor::from_vec(t.cos.clone(), (n, t.pairs), &dev)?,
        Tensor::from_vec(t.sin.clone(), (n, t.pairs), &dev)?,
    )?;

    let mut s = 0x9e37_79b9_7f4a_7c15u64;
    let mut next = || {
        s = s
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((s >> 40) as f32 / (1u64 << 24) as f32) - 0.5
    };
    const Q_HEADS: usize = 16;
    const K_HEADS: usize = 2;
    let mut fill = |heads: usize| -> Vec<bf16> {
        (0..rows * heads * HEAD_DIM)
            .map(|_| bf16::from_f32(next() * 4.0))
            .collect()
    };
    let (qv, kv) = (fill(Q_HEADS), fill(K_HEADS));
    let q = Tensor::from_vec(qv.clone(), (rows, Q_HEADS, HEAD_DIM), &dev)?;
    let k = Tensor::from_vec(kv.clone(), (rows, K_HEADS, HEAD_DIM), &dev)?;
    rot.apply(&q, &k)?;

    // The host's answer: the same f32 products and sums, rounded once.
    let half = HEAD_DIM / 2;
    for (name, x, vals, heads) in [("q", &q, &qv, Q_HEADS), ("k", &k, &kv, K_HEADS)] {
        let got = x.flatten_all()?.to_vec1::<bf16>()?;
        let mut want = vals.clone();
        for (i, &r) in t.rows.iter().enumerate() {
            for h in 0..heads {
                let base = (r as usize * heads + h) * HEAD_DIM;
                for f in 0..t.pairs {
                    let (c, sn) = (t.cos[i * t.pairs + f], t.sin[i * t.pairs + f]);
                    let lo = vals[base + f].to_f32();
                    let hi = vals[base + f + half].to_f32();
                    want[base + f] = bf16::from_f32(lo * c - hi * sn);
                    want[base + f + half] = bf16::from_f32(hi * c + lo * sn);
                }
            }
        }
        let bad = got
            .iter()
            .zip(&want)
            .position(|(g, w)| g.to_bits() != w.to_bits());
        if let Some(at) = bad {
            candle::bail!(
                "{name}: element {at} is {} on the device, {} on the host",
                got[at],
                want[at]
            );
        }
    }
    println!("correctness gate: q and k match the host bit for bit");

    for _ in 0..10 {
        rot.apply(&q, &k)?;
    }
    dev.synchronize()?;
    let t0 = std::time::Instant::now();
    for _ in 0..iters {
        rot.apply(&q, &k)?;
    }
    dev.synchronize()?;
    let us = t0.elapsed().as_secs_f64() * 1e6 / iters as f64;
    // Each listed row's rotary pairs read and written, its tables read.
    let heads = Q_HEADS + K_HEADS;
    let bytes = (n * heads * t.pairs * 2 * 2 * 2 + n * t.pairs * 2 * 4 + n * 4) as f64;
    println!(
        "q+k {rows:>6} rows, {n:>5} image rows, {Q_HEADS}+{K_HEADS} heads {us:>8.1} us/call \
         {:>7.1} GB/s",
        bytes / (us * 1e-6) / 1e9
    );
    Ok(())
}

#[cfg(not(feature = "cuda"))]
fn main() {
    eprintln!("image_rope_bench requires the `cuda` feature");
}
