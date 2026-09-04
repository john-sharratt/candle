use candle::{test_device, Device, IndexOp, Result, Tensor};
use candle_core as candle;

fn contiguous(device: &Device) -> Result<()> {
    let tensor = Tensor::arange(0u32, 24u32, device)?.reshape((2, 3, 4))?;
    assert_eq!(
        tensor.to_vec3::<u32>()?,
        &[
            [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11]],
            [[12, 13, 14, 15], [16, 17, 18, 19], [20, 21, 22, 23]]
        ]
    );
    assert_eq!(
        tensor.t()?.contiguous()?.to_vec3::<u32>()?,
        &[
            [[0, 4, 8], [1, 5, 9], [2, 6, 10], [3, 7, 11]],
            [[12, 16, 20], [13, 17, 21], [14, 18, 22], [15, 19, 23]]
        ]
    );
    assert_eq!(
        tensor.transpose(0, 1)?.contiguous()?.to_vec3::<u32>()?,
        &[
            [[0, 1, 2, 3], [12, 13, 14, 15]],
            [[4, 5, 6, 7], [16, 17, 18, 19]],
            [[8, 9, 10, 11], [20, 21, 22, 23]]
        ]
    );
    assert_eq!(
        tensor.transpose(0, 1)?.flatten_all()?.to_vec1::<u32>()?,
        &[0, 1, 2, 3, 12, 13, 14, 15, 4, 5, 6, 7, 16, 17, 18, 19, 8, 9, 10, 11, 20, 21, 22, 23]
    );
    assert_eq!(
        tensor
            .i(1..)?
            .transpose(0, 1)?
            .contiguous()?
            .to_vec3::<u32>()?,
        &[[[12, 13, 14, 15]], [[16, 17, 18, 19]], [[20, 21, 22, 23]]]
    );
    assert_eq!(
        tensor.transpose(0, 2)?.contiguous()?.to_vec3::<u32>()?,
        &[
            [[0, 12], [4, 16], [8, 20]],
            [[1, 13], [5, 17], [9, 21]],
            [[2, 14], [6, 18], [10, 22]],
            [[3, 15], [7, 19], [11, 23]]
        ]
    );
    Ok(())
}

test_device!(contiguous, contiguous_cpu, contiguous_gpu, contiguous_metal);

/// **Reshaping a transposed tensor must produce the same bytes on every
/// backend.**
///
/// `t()` leaves a non-contiguous layout, and `reshape` on one has to
/// materialise it before it can reinterpret the dims. That materialisation is
/// exactly what this repo's hot-path work removes wherever it can, so the
/// pattern is worth pinning: a `reshape` that reinterprets a *transposed*
/// layout without copying reads the right bytes in the wrong order, and the
/// result is a plausible tensor of the right shape with its contents
/// spatially scrambled.
///
/// The shape here is Stable Diffusion's VAE attention block, which ends on
/// `proj_attn.forward(xs).t().reshape((batch, channel, height, width))` — the
/// one place in that model where a transpose feeds a reshape directly.
fn reshape_of_a_transpose(device: &Device) -> Result<()> {
    // Distinct values, so any misordering is visible rather than averaged away.
    let (b, hw, c) = (2usize, 12usize, 6usize);
    let t = Tensor::arange(0u32, (b * hw * c) as u32, device)?.reshape((b, hw, c))?;

    // What the VAE does.
    let got = t
        .t()?
        .reshape((b, c, 3, 4))?
        .flatten_all()?
        .to_vec1::<u32>()?;

    // What it must equal: the transpose materialised first, which is the same
    // operation stated so that no backend can take a shortcut through it.
    let want = t
        .t()?
        .contiguous()?
        .reshape((b, c, 3, 4))?
        .flatten_all()?
        .to_vec1::<u32>()?;

    assert_eq!(
        got, want,
        "reshaping a transposed tensor took the un-transposed bytes — a model doing this gets \
         the right shape with its spatial layout scrambled, which reads as a periodic grid \
         rather than as an error"
    );

    // And against the value the indices define, so the test does not merely
    // assert two implementations agree with each other.
    let expect: Vec<u32> = (0..b)
        .flat_map(|bi| {
            (0..c).flat_map(move |ci| (0..hw).map(move |hi| (bi * hw * c + hi * c + ci) as u32))
        })
        .collect();
    assert_eq!(got, expect, "the transpose itself is wrong");
    Ok(())
}

test_device!(
    reshape_of_a_transpose,
    reshape_of_a_transpose_cpu,
    reshape_of_a_transpose_gpu,
    reshape_of_a_transpose_metal
);

#[test]
fn strided_blocks() -> Result<()> {
    use candle::Device::Cpu;
    let tensor = Tensor::arange(0u32, 24u32, &Cpu)?.reshape((2, 3, 4))?;
    match tensor.strided_blocks() {
        candle::StridedBlocks::SingleBlock { start_offset, len } => {
            assert_eq!(start_offset, 0);
            assert_eq!(len, 24);
        }
        candle::StridedBlocks::MultipleBlocks { .. } => {
            panic!("unexpected block structure")
        }
    };
    let tensor = Tensor::arange(0u32, 26u32, &Cpu)?
        .i(2..)?
        .reshape((2, 3, 4))?;
    match tensor.strided_blocks() {
        candle::StridedBlocks::SingleBlock { start_offset, len } => {
            assert_eq!(start_offset, 2);
            assert_eq!(len, 24);
        }
        candle::StridedBlocks::MultipleBlocks { .. } => {
            panic!("unexpected block structure")
        }
    };
    let tensor = Tensor::arange(0u32, 24u32, &Cpu)?.reshape((2, 3, 4))?;
    let tensor = tensor.i(1)?;
    match tensor.strided_blocks() {
        candle::StridedBlocks::SingleBlock { start_offset, len } => {
            assert_eq!(start_offset, 12);
            assert_eq!(len, 12);
        }
        candle::StridedBlocks::MultipleBlocks { .. } => {
            panic!("unexpected block structure")
        }
    };
    let tensor = Tensor::arange(0u32, 24u32, &Cpu)?.reshape((2, 3, 4))?;
    let tensor = tensor.i((.., 1))?.contiguous()?;
    match tensor.strided_blocks() {
        candle::StridedBlocks::SingleBlock { start_offset, len } => {
            assert_eq!(start_offset, 0);
            assert_eq!(len, 8);
            assert_eq!(tensor.to_vec2::<u32>()?, &[[4, 5, 6, 7], [16, 17, 18, 19]]);
        }
        candle::StridedBlocks::MultipleBlocks { .. } => {
            panic!("unexpected block structure")
        }
    };
    let tensor = Tensor::arange(0u32, 24u32, &Cpu)?.reshape((2, 3, 4))?;
    let tensor = tensor.i((.., 1))?;
    match tensor.strided_blocks() {
        candle::StridedBlocks::SingleBlock { .. } => {
            panic!("unexpected block structure")
        }
        candle::StridedBlocks::MultipleBlocks {
            block_len,
            block_start_index,
        } => {
            assert_eq!(block_len, 4);
            assert_eq!(block_start_index.collect::<Vec<_>>(), &[4, 16])
        }
    };
    let tensor = Tensor::arange(0u32, 24u32, &Cpu)?.reshape((2, 3, 4))?;
    match tensor.t()?.strided_blocks() {
        candle::StridedBlocks::SingleBlock { .. } => {
            panic!("unexpected block structure")
        }
        candle::StridedBlocks::MultipleBlocks {
            block_start_index,
            block_len,
        } => {
            assert_eq!(block_len, 1);
            assert_eq!(
                block_start_index.collect::<Vec<_>>(),
                &[
                    0, 4, 8, 1, 5, 9, 2, 6, 10, 3, 7, 11, 12, 16, 20, 13, 17, 21, 14, 18, 22, 15,
                    19, 23
                ]
            )
        }
    };
    let tensor = Tensor::arange(0u32, 24u32, &Cpu)?.reshape((2, 3, 4))?;
    match tensor.transpose(0, 1)?.strided_blocks() {
        candle::StridedBlocks::SingleBlock { .. } => {
            panic!("unexpected block structure")
        }
        candle::StridedBlocks::MultipleBlocks {
            block_start_index,
            block_len,
        } => {
            assert_eq!(block_len, 4);
            assert_eq!(
                block_start_index.collect::<Vec<_>>(),
                &[0, 12, 4, 16, 8, 20]
            )
        }
    };
    Ok(())
}
