//! **The prefill launches against recorded bits.**
//!
//! The three prefill kernels (conv, intra, state) are tuned for occupancy and
//! latency, never for their arithmetic: every sum each of them computes is a
//! fixed chain — the conv's j-ascending window, the norm's 128-wide tree, each
//! A/kq entry's d-ascending dot, the substitution's row order, each state-pass
//! dot over d_k, the intra-chunk sum over s and the update's sum over t — and a
//! change of thread mapping, staging or pipelining must leave every one of them
//! where it was. The tolerance tests beside this one cannot see a change of
//! order (it moves the last bits, well inside their bounds), and a rewind that
//! replays a wave through these launches diverges from the wave on exactly
//! those bits. So this test pins the bits themselves.
//!
//! Each case is one wave through `delta_net_conv_prefill` →
//! `delta_net_prefill_scan` from random projections, random entering states
//! and random conv tails, and records an FNV-1a hash of the f32 bit patterns of
//! every output: the conv output buffer, `o`, every advanced state and every
//! advanced conv tail. The shapes cover a sub-chunk span, a whole chunk, one
//! past it, a ragged multi-chunk span, long single sequences at Flash-Next's
//! 16:48 geometry (4096 and 7500 rows — the shapes the profile runs), and
//! multi-span waves of unequal lengths, advancing in place and into separate
//! buffers.
//!
//! The recorded hashes are the bits of a particular instruction sequence — the
//! fast-math `expf`/`log1pf` lower to the SFU approximations of the
//! architecture the archive was compiled for — so they are recorded per compute
//! capability, and a capability with no record prints its hashes for capture
//! instead of asserting.

use super::*;

/// One recorded wave: the K/V head counts, the span lengths, whether the
/// states advance in place, and the seed every operand derives from.
struct GoldenCase {
    h_k: usize,
    h_v: usize,
    lens: &'static [usize],
    in_place: bool,
    seed: u64,
}

const fn case(
    h_k: usize,
    h_v: usize,
    lens: &'static [usize],
    in_place: bool,
    seed: u64,
) -> GoldenCase {
    GoldenCase {
        h_k,
        h_v,
        lens,
        in_place,
        seed,
    }
}

const CASES: &[GoldenCase] = &[
    case(2, 4, &[7], true, 1101),
    case(2, 4, &[64], false, 1102),
    case(2, 4, &[65], true, 1103),
    case(2, 4, &[127], false, 1104),
    case(2, 4, &[5, 64, 65], false, 1105),
    case(2, 4, &[1000], true, 1106),
    case(16, 48, &[1000, 7, 130], true, 1107),
    case(16, 48, &[4096], false, 1108),
    case(16, 48, &[7500], true, 1109),
];

/// The recorded hashes on compute capability 12.0, per case in [`CASES`]
/// order: `[conv output, o, advanced states, advanced tails]`.
const GOLDEN_SM120: &[[u64; 4]] = &[
    // 2:4 [7]
    [
        0x19a91aba75307931,
        0xd064f6c40cbb58f1,
        0x58caf914400e81f8,
        0x8d8d5b0547439fc3,
    ],
    // 2:4 [64]
    [
        0xc408ea74c2654541,
        0x58c754d96fd06ad7,
        0x64c5826e7b926a8f,
        0x896aa006101ef02b,
    ],
    // 2:4 [65]
    [
        0x88ab079fb6fa6803,
        0xedc12b7cbdfdb138,
        0xfb11123e31713603,
        0xc932a2300dfb6e71,
    ],
    // 2:4 [127]
    [
        0xec86e7556e012c17,
        0xf642d39450a3cadc,
        0x04d58084be3db595,
        0x26a0faec666ca22f,
    ],
    // 2:4 [5, 64, 65]
    [
        0x6ff41414bdc6fd5f,
        0xf96a508d8dda66f0,
        0xa826e793c933d6de,
        0xae22b561994cf96d,
    ],
    // 2:4 [1000]
    [
        0xfe3931cd7ea12a76,
        0x4a3f9318286ab861,
        0x8b7be44abdf77877,
        0xe1abef1320f2fb58,
    ],
    // 16:48 [1000, 7, 130]
    [
        0x73fc6c994c0673bd,
        0x8d831813d5885417,
        0x2981f8880172c988,
        0xb8c9d15b34982b1c,
    ],
    // 16:48 [4096]
    [
        0x12ccc4f939dba039,
        0xb2cccfcd27918053,
        0xf6aa9be6c5b31b06,
        0xb2f490146d168cef,
    ],
    // 16:48 [7500]
    [
        0xa23b23db5a8f3ca3,
        0xfde368ad55cb96b3,
        0x8f18a2a52cda2c72,
        0x8de5678df9be35c9,
    ],
];

/// FNV-1a over the f32 bit patterns of every tensor in `ts`, in order.
fn fnv_bits(ts: &[&Tensor]) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for t in ts {
        for b in f32_bits(t) {
            for byte in b.to_le_bytes() {
                h ^= byte as u64;
                h = h.wrapping_mul(0x0000_0100_0000_01b3);
            }
        }
    }
    h
}

/// Runs one case through the conv and the scan and hashes what they wrote.
fn run_case(c: &GoldenCase, gpu: &Device) -> [u64; 4] {
    let cpu = Device::Cpu;
    let d = 128usize;
    let kw = 4usize;
    let eps = 1e-6f32;
    let t: usize = c.lens.iter().sum();
    let conv_dim = (2 * c.h_k + c.h_v) * d;
    let to = |x: Tensor| x.to_device(gpu).unwrap().contiguous().unwrap();
    // Projections at ±2 (and gates at ±2), so the SiLU, the norms and the
    // decays all leave their linear ranges.
    let wide =
        |shape: &[usize], seed: u64| to(lcg_tensor(shape, seed, &cpu).affine(4.0, 0.0).unwrap());
    let qkv = wide(&[t, conv_dim], c.seed);
    let kern = to(lcg_tensor(&[conv_dim, kw], c.seed + 1, &cpu));
    let alpha = wide(&[t, c.h_v], c.seed + 2);
    let blin = wide(&[t, c.h_v], c.seed + 3);
    let dt_bias = to(lcg_tensor(&[c.h_v], c.seed + 4, &cpu));
    let a_abs = lcg_tensor(&[c.h_v], c.seed + 5, &cpu).abs().unwrap();
    let a = to((a_abs + 0.1).unwrap().neg().unwrap());

    let n = c.lens.len();
    let starts: Vec<usize> = c
        .lens
        .iter()
        .scan(0usize, |acc, &l| {
            let s = *acc;
            *acc += l;
            Some(s)
        })
        .collect();
    let states: Vec<Tensor> = (0..n)
        .map(|i| to(lcg_tensor(&[c.h_v, d, d], c.seed + 10 + i as u64, &cpu)))
        .collect();
    let state_outs: Vec<Tensor> = (0..n)
        .map(|_| Tensor::zeros((c.h_v, d, d), DType::F32, gpu).unwrap())
        .collect();
    let tail_shape = [conv_dim, kw - 1];
    let tails: Vec<Tensor> = (0..n)
        .map(|i| to(lcg_tensor(&tail_shape, c.seed + 20 + i as u64, &cpu)))
        .collect();
    let tail_outs: Vec<Tensor> = (0..n)
        .map(|_| Tensor::zeros((conv_dim, kw - 1), DType::F32, gpu).unwrap())
        .collect();
    let rows: Vec<TestSpan<'_>> = (0..n)
        .map(|i| TestSpan {
            tail: &tails[i],
            tail_out: &tail_outs[i],
            state: &states[i],
            state_out: if c.in_place {
                &states[i]
            } else {
                &state_outs[i]
            },
            start: starts[i],
            len: c.lens[i],
        })
        .collect();
    let tbl = span_table(&rows);

    let conved = Tensor::zeros((t, conv_dim), DType::F32, gpu).unwrap();
    let o = Tensor::zeros((t, c.h_v * d), DType::F32, gpu).unwrap();
    delta_net_conv_prefill(&qkv, &kern, &tbl, 2 * c.h_k * d, eps, &conved).unwrap();
    let fused = DeltaNetFused {
        conved: &conved,
        alpha: &alpha,
        blin: &blin,
        dt_bias: &dt_bias,
        a: &a,
        o: &o,
        q_scale: 1.0 / (d as f32).sqrt(),
    };
    delta_net_prefill_scan(&fused, &tbl).unwrap();

    let advanced: Vec<&Tensor> = if c.in_place {
        states.iter().collect()
    } else {
        state_outs.iter().collect()
    };
    [
        fnv_bits(&[&conved]),
        fnv_bits(&[&o]),
        fnv_bits(&advanced),
        fnv_bits(&tail_outs.iter().collect::<Vec<_>>()),
    ]
}

#[test]
fn prefill_launches_reproduce_the_recorded_bits() {
    let Ok(gpu) = Device::new_cuda(0) else {
        eprintln!("skipping: no CUDA device");
        return;
    };
    let Device::Cuda(dev) = &gpu else {
        unreachable!("built as CUDA")
    };
    let cc = dev.compute_capability().unwrap();
    let got: Vec<[u64; 4]> = CASES.iter().map(|c| run_case(c, &gpu)).collect();
    for (c, g) in CASES.iter().zip(&got) {
        eprintln!(
            "    [{:#018x}, {:#018x}, {:#018x}, {:#018x}], // {}:{} {:?}",
            g[0], g[1], g[2], g[3], c.h_k, c.h_v, c.lens
        );
    }
    let golden = match cc {
        (12, 0) => GOLDEN_SM120,
        _ => {
            eprintln!(
                "no recorded prefill bits for compute capability {}.{} — the hashes above \
                 are this card's, for capture",
                cc.0, cc.1
            );
            return;
        }
    };
    assert_eq!(
        golden.len(),
        CASES.len(),
        "the record for {}.{} does not cover every case",
        cc.0,
        cc.1
    );
    for ((c, g), want) in CASES.iter().zip(&got).zip(golden) {
        let names = ["conv output", "o", "advanced states", "advanced tails"];
        for k in 0..4 {
            assert_eq!(
                g[k], want[k],
                "{}:{} {:?}: {} differs from the recorded bits",
                c.h_k, c.h_v, c.lens, names[k]
            );
        }
    }
}
