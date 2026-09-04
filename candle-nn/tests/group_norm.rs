/* Equivalent PyTorch code.
import torch
from torch.nn.functional import group_norm
t = torch.tensor(
        [[[-0.3034,  0.2726, -0.9659],
          [-1.1845, -1.3236,  0.0172],
          [ 1.9507,  1.2554, -0.8625],
          [ 1.0682,  0.3604,  0.3985],
          [-0.4957, -0.4461, -0.9721],
          [ 1.5157, -0.1546, -0.5596]],

         [[-1.6698, -0.4040, -0.7927],
          [ 0.3736, -0.0975, -0.1351],
          [-0.9461,  0.5461, -0.6334],
          [-1.0919, -0.1158,  0.1213],
          [-0.9535,  0.1281,  0.4372],
          [-0.2845,  0.3488,  0.5641]]])
print(group_norm(t, num_groups=2))
print(group_norm(t, num_groups=3))
*/
#[cfg(feature = "mkl")]
extern crate intel_mkl_src;

#[cfg(feature = "accelerate")]
extern crate accelerate_src;

use anyhow::Result;
use candle::test_utils::to_vec3_round;
use candle::{Device, Tensor};
use candle_nn::{GroupNorm, Module};

#[test]
fn group_norm() -> Result<()> {
    let device = &Device::Cpu;
    let w = Tensor::from_vec(vec![1f32; 6], 6, device)?;
    let b = Tensor::from_vec(vec![0f32; 6], 6, device)?;
    let gn2 = GroupNorm::new(w.clone(), b.clone(), 6, 2, 1e-5)?;
    let gn3 = GroupNorm::new(w, b, 6, 3, 1e-5)?;

    let input = Tensor::new(
        &[
            [
                [-0.3034f32, 0.2726, -0.9659],
                [-1.1845, -1.3236, 0.0172],
                [1.9507, 1.2554, -0.8625],
                [1.0682, 0.3604, 0.3985],
                [-0.4957, -0.4461, -0.9721],
                [1.5157, -0.1546, -0.5596],
            ],
            [
                [-1.6698, -0.4040, -0.7927],
                [0.3736, -0.0975, -0.1351],
                [-0.9461, 0.5461, -0.6334],
                [-1.0919, -0.1158, 0.1213],
                [-0.9535, 0.1281, 0.4372],
                [-0.2845, 0.3488, 0.5641],
            ],
        ],
        device,
    )?;
    assert_eq!(
        to_vec3_round(&gn2.forward(&input)?, 4)?,
        &[
            [
                [-0.1653, 0.3748, -0.7866],
                [-0.9916, -1.1220, 0.1353],
                [1.9485, 1.2965, -0.6896],
                [1.2769, 0.3628, 0.4120],
                [-0.7427, -0.6786, -1.3578],
                [1.8547, -0.3022, -0.8252]
            ],
            [
                [-1.9342, 0.0211, -0.5793],
                [1.2223, 0.4945, 0.4365],
                [-0.8163, 1.4887, -0.3333],
                [-1.7960, -0.0392, 0.3875],
                [-1.5469, 0.3998, 0.9561],
                [-0.3428, 0.7970, 1.1845]
            ]
        ]
    );
    assert_eq!(
        to_vec3_round(&gn3.forward(&input)?, 4)?,
        &[
            [
                [0.4560, 1.4014, -0.6313],
                [-0.9901, -1.2184, 0.9822],
                [1.4254, 0.6360, -1.7682],
                [0.4235, -0.3800, -0.3367],
                [-0.3890, -0.3268, -0.9862],
                [2.1325, 0.0386, -0.4691]
            ],
            [
                [-1.8797, 0.0777, -0.5234],
                [1.2802, 0.5517, 0.4935],
                [-1.0102, 1.5327, -0.4773],
                [-1.2587, 0.4047, 0.8088],
                [-1.9074, 0.1691, 0.7625],
                [-0.6230, 0.5928, 1.0061]
            ]
        ]
    );

    Ok(())
}

/// **The CUDA path must agree with the CPU one, at the shapes a real model
/// uses.**
///
/// The test above pins the numbers against PyTorch, but only on `Device::Cpu`,
/// and only for a 2×6×3 toy. Nothing exercised group norm on a card, and nothing
/// exercised it at a width where a kernel's reduction strategy changes — which
/// is exactly where a per-group reduction goes wrong.
///
/// The shapes here are Stable Diffusion's VAE decoder: 32 groups over 512, 256
/// and 128 channels at the resolutions the decoder upsamples through. A
/// diffusion pipeline whose decode collapses to flat grey looks like bad weights
/// and is indistinguishable from bad normalisation, so this is the check that
/// tells the two apart.
#[cfg(feature = "cuda")]
#[test]
fn group_norm_cuda_matches_cpu() -> Result<()> {
    let cuda = Device::new_cuda(0)?;

    // (channels, height, width) — the VAE decoder's three up-block stages, plus
    // a deliberately awkward one: 30 channels over 3 groups is 10 per group,
    // which no power-of-two tiling divides evenly.
    for &(c, h, w, groups) in &[
        (512usize, 64usize, 64usize, 32usize),
        (256, 128, 128, 32),
        (128, 256, 256, 32),
        (30, 7, 5, 3),
    ] {
        // Deterministic and non-trivial: a constant input normalises to zero
        // whatever the reduction does, so it would pass a broken kernel.
        let n = c * h * w;
        let data: Vec<f32> = (0..n)
            .map(|i| ((i % 97) as f32 - 48.0) / 17.0 + (i % 7) as f32 * 0.13)
            .collect();
        let weight: Vec<f32> = (0..c).map(|i| 0.5 + (i % 5) as f32 * 0.25).collect();
        let bias: Vec<f32> = (0..c).map(|i| (i % 3) as f32 * 0.1 - 0.1).collect();

        let on = |dev: &Device| -> Result<Vec<f32>> {
            let x = Tensor::from_vec(data.clone(), (1, c, h, w), dev)?;
            let g = GroupNorm::new(
                Tensor::from_vec(weight.clone(), c, dev)?,
                Tensor::from_vec(bias.clone(), c, dev)?,
                c,
                groups,
                1e-6,
            )?;
            Ok(g.forward(&x)?.flatten_all()?.to_vec1::<f32>()?)
        };

        let (a, b) = (on(&Device::Cpu)?, on(&cuda)?);
        let worst = a
            .iter()
            .zip(&b)
            .map(|(x, y)| (x - y).abs())
            .fold(0f32, f32::max);
        assert!(
            worst < 2e-4,
            "group_norm({groups} groups over {c}ch, {h}x{w}) differs between CPU and CUDA by \
             {worst} — a decoder built on this normalises to the wrong statistics, which reads \
             as a washed-out image rather than as an error"
        );
    }
    Ok(())
}
