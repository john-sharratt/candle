//! The tensor-assert sentinel: a grouped expert GEMM's rows no tile writes
//! read back as `0xFFFFFFFF`, and the rows its tiles write are the GEMM's.

use super::*;

/// One expert's tile covers rows `0..3` of an 8-row launch; rows `3..8` are
/// padding no kernel writes. Under tensor-assert they hold the sentinel bit
/// for bit, and the written rows hold finite products, so a NaN downstream of
/// this output tells an unwritten row from one something else wrote.
#[test]
pub(super) fn grouped_dev_gemm_padding_rows_read_as_the_sentinel() -> Result<()> {
    use rand::rngs::StdRng;
    use rand::SeedableRng;

    let dev = CudaDevice::new(0)?;
    let nrows = 256usize;
    let ncols = 1024usize;
    let rows = 8usize;
    let written = 3usize;
    let mut rng = StdRng::seed_from_u64(0x05e4_71e1);

    let shape = crate::Shape::from((nrows, ncols));
    let w: Vec<f32> = (0..nrows * ncols)
        .map(|_| rng.random_range(-1.0..1.0))
        .collect();
    let mut q = QCudaStorage::zeros(&dev, ncols * nrows, GgmlDType::Q6_K)?;
    q.quantize(&CudaStorage::wrap_cuda_slice(
        dev.memcpy_stod(&w)?,
        dev.clone(),
    ))?;
    let ko = q.repack_ko(&shape, GgmlDType::Q6_KO)?;
    let table = dev.memcpy_stod(&[ko.data_ptr()])?;

    let act: Vec<f32> = (0..rows * ncols)
        .map(|_| rng.random_range(-1.0..1.0))
        .collect();
    let op = quantize_acts_q8a128_test(&dev, &act, rows, ncols)?;
    let tile_expert = dev.memcpy_stod(&[0i32])?;
    let tile_b_start = dev.memcpy_stod(&[0i32])?;
    let tile_b_cnt = dev.memcpy_stod(&[written as i32])?;

    let out = grouped_qmatmul_dev_q8a128(
        &op,
        &table,
        0,
        1,
        GgmlDType::Q6_KO,
        nrows,
        &tile_expert,
        &tile_b_start,
        &tile_b_cnt,
        1,
        2,
        None,
        &dev,
    )?;
    let v = read_f32_tensor(&dev, &out)?;
    assert_eq!(v.len(), rows * nrows);
    assert!(
        v[..written * nrows].iter().all(|x| x.is_finite()),
        "the tile's rows are the GEMM's products"
    );
    assert_eq!(
        v[written * nrows..]
            .iter()
            .map(|x| x.to_bits())
            .collect::<Vec<u32>>(),
        vec![0xffff_ffffu32; (rows - written) * nrows],
        "the rows no tile writes read as the sentinel"
    );
    Ok(())
}
