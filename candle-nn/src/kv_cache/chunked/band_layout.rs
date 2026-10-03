//! How a chunk's bands hold one head's values, for the host-side readers and
//! writers (`read_contiguous`, `write_contiguous`, a fork's re-encode).
//!
//! Three rules, each mirroring the device kernels (`palette4_convert.cuh`,
//! `load_band_elem`):
//!
//! - **Orientation.** A float band is token-major, `[token][local_dim]`. A
//!   quantized band (R16 included) is channel-major: one 32-element block per
//!   local dim, holding that dim's 32 tokens.
//! - **Palette routing.** Band `p` of head `h` holds the global dims the head's
//!   palette map assigns to `p`, in increasing order. An empty map is the
//!   identity: band `p` holds dims `[p·sub, (p+1)·sub)`.
//! - **Outer scale.** A band stores `v · outer` and decodes `stored / outer`;
//!   an absent scale is 1.

use candle::quantized::k_quants::encode_q0_v_bytes;
use candle::quantized::{GgmlDType, QTensor};
use candle::{DType, Device, LiveTensor, Result, Tensor};

/// Which half of a head's K/V pair a band holds. Q0_V decodes each against its
/// own calibrated codebook.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum BandSide {
    K,
    V,
}

/// The palette band `pal` assigns global dim `d` to: two bits per dim, four
/// dims per byte, low bits first.
fn palette_of(map: &[u8], d: usize) -> usize {
    ((map[d / 4] >> (2 * (d % 4))) & 0x3) as usize
}

/// The global dims in the order the head's bands hold them — palette 0's dims,
/// then palette 1's, … — or `None` when that order is the identity, so the
/// caller need not route at all.
///
/// Routing applies to the four-palette band layout only; any other band count
/// carries no map.
pub(super) fn band_order(
    pal: &[u8],
    head: usize,
    n_palette: usize,
    head_dim: usize,
) -> Result<Option<Vec<u32>>> {
    let bytes = head_dim / 4;
    let Some(map) = pal.get(head * bytes..(head + 1) * bytes) else {
        return Ok(None);
    };
    if n_palette != 4 {
        return Ok(None);
    }
    let sub = head_dim / n_palette;
    let mut order = Vec::with_capacity(head_dim);
    for p in 0..n_palette {
        let start = order.len();
        order.extend(
            (0..head_dim)
                .filter(|&d| palette_of(map, d) == p)
                .map(|d| d as u32),
        );
        if order.len() - start != sub {
            candle::bail!(
                "palette map for head {head} gives palette {p} {} dims, not {sub}",
                order.len() - start
            );
        }
    }
    let identity = order.iter().enumerate().all(|(i, &d)| i == d as usize);
    Ok((!identity).then_some(order))
}

/// Put a head read band by band — `(.., head_dim)` with bands joined in palette
/// order on the last dim — back into global dim order.
pub(super) fn route_to_dims<'w>(
    joined: LiveTensor<'w>,
    order: Option<&[u32]>,
) -> Result<LiveTensor<'w>> {
    let Some(order) = order else {
        return Ok(joined);
    };
    let mut inverse = vec![0u32; order.len()];
    for (i, &d) in order.iter().enumerate() {
        inverse[d as usize] = i as u32;
    }
    let last = joined.rank() - 1;
    let idx = Tensor::from_vec(inverse, order.len(), joined.device())?;
    joined.index_select(&idx, last)
}

/// Band `p`'s values out of a head in global dim order, `(.., head_dim)` →
/// `(.., sub_head_dim)`.
pub(super) fn band_of<'w>(
    head: &LiveTensor<'w>,
    order: Option<&[u32]>,
    p: usize,
    sub: usize,
) -> Result<LiveTensor<'w>> {
    let last = head.rank() - 1;
    match order {
        None => head.narrow(last, p * sub, sub)?.contiguous(),
        Some(order) => {
            let idx = Tensor::from_vec(order[p * sub..(p + 1) * sub].to_vec(), sub, head.device())?;
            head.index_select(&idx, last)
        }
    }
}

/// The outer scale of band `idx`, 1 when the chunk records none.
pub(super) fn band_scale(scales: &[f32], idx: usize) -> f32 {
    scales.get(idx).copied().unwrap_or(1.0)
}

/// A decoded band's values: `stored / outer`.
pub(super) fn unscale(stored: LiveTensor<'_>, outer: f32) -> Result<LiveTensor<'_>> {
    if outer == 1.0 {
        return Ok(stored);
    }
    stored.affine(1.0 / outer as f64, 0.0)
}

/// What a band stores for `values`: `values · outer`.
pub(super) fn scale(values: LiveTensor<'_>, outer: f32) -> Result<LiveTensor<'_>> {
    if outer == 1.0 {
        return Ok(values);
    }
    values.affine(outer as f64, 0.0)
}

/// A quantized band's dequantized elements, channel-major as stored, turned
/// token-major: `(sub_head_dim · chunk_size)` → `(chunk_size, sub_head_dim)`.
pub(super) fn quantized_to_token_major(
    flat: Tensor,
    chunk_size: usize,
    sub_head_dim: usize,
) -> Result<Tensor> {
    flat.reshape((sub_head_dim, chunk_size))?.t()?.contiguous()
}

/// A token-major band `(chunk_size, sub_head_dim)` as the flat channel-major
/// element run a quantized band encodes.
pub(super) fn token_major_to_quantized<'w>(band: &LiveTensor<'w>) -> Result<LiveTensor<'w>> {
    band.to_dtype(DType::F32)?.t()?.contiguous()?.flatten_all()
}

/// The bytes a quantized band stores for `band` — token-major
/// `(chunk_size, sub_head_dim)` stored values — encoded on the host with the
/// format's block codec, channel-major. Q0_V takes its side's codebook. As a
/// 1-D `U8` tensor on `band`'s device, ready for the slot.
///
/// On the host because the device block-quantize kernel covers only some of
/// these formats; the codecs here mirror the KV kernels' encoders bit for bit.
pub(super) fn encode_quantized_band(
    format: GgmlDType,
    side: BandSide,
    band: &LiveTensor<'_>,
) -> Result<Tensor> {
    let flat: Vec<f32> = token_major_to_quantized(band)?.to_vec1()?;
    let bytes = if format == GgmlDType::Q0_V && side == BandSide::K {
        encode_q0_v_bytes::<true>(&flat)
    } else {
        let n = flat.len();
        let q = QTensor::quantize(&Tensor::from_vec(flat, n, &Device::Cpu)?, format)?;
        q.data()?.into_owned()
    };
    let n = bytes.len();
    Tensor::from_vec(bytes, n, band.device())
}

#[cfg(test)]
mod tests {
    use super::*;

    const HD: usize = 8;

    /// A tensor's values on the host, flattened.
    fn dims(t: &Tensor) -> Vec<f32> {
        t.flatten_all().unwrap().to_vec1().unwrap()
    }

    /// Pack a dim → palette assignment as the chunk stores it.
    fn pack(assign: [usize; HD]) -> Vec<u8> {
        let mut out = vec![0u8; HD / 4];
        for (d, &p) in assign.iter().enumerate() {
            out[d / 4] |= (p as u8) << (2 * (d % 4));
        }
        out
    }

    #[test]
    fn an_empty_or_banded_map_is_the_identity() {
        assert_eq!(band_order(&[], 0, 4, HD).unwrap(), None);
        let banded = pack([0, 0, 1, 1, 2, 2, 3, 3]);
        assert_eq!(band_order(&banded, 0, 4, HD).unwrap(), None);
    }

    #[test]
    fn a_striped_map_orders_dims_palette_by_palette() {
        let striped = pack([0, 1, 2, 3, 0, 1, 2, 3]);
        assert_eq!(
            band_order(&striped, 0, 4, HD).unwrap(),
            Some(vec![0, 4, 1, 5, 2, 6, 3, 7])
        );
    }

    /// The second head's map is read from its own bytes, not the first's.
    #[test]
    fn each_head_reads_its_own_map() {
        let mut maps = pack([0, 0, 1, 1, 2, 2, 3, 3]);
        maps.extend(pack([3, 3, 2, 2, 1, 1, 0, 0]));
        assert_eq!(band_order(&maps, 0, 4, HD).unwrap(), None);
        assert_eq!(
            band_order(&maps, 1, 4, HD).unwrap(),
            Some(vec![6, 7, 4, 5, 2, 3, 0, 1])
        );
    }

    #[test]
    fn a_map_that_unbalances_the_palettes_is_refused() {
        let lopsided = pack([0, 0, 0, 1, 2, 2, 3, 3]);
        assert!(band_order(&lopsided, 0, 4, HD).is_err());
    }

    /// Splitting a head into bands and routing them back is the identity, and
    /// each band holds exactly its palette's dims.
    #[test]
    fn bands_route_back_to_their_dims() {
        let striped = pack([0, 1, 2, 3, 0, 1, 2, 3]);
        let order = band_order(&striped, 0, 4, HD).unwrap();
        let head =
            Tensor::new(&[[10f32, 11., 12., 13., 14., 15., 16., 17.]], &Device::Cpu).unwrap();
        let bands: Vec<Tensor> = (0..4)
            .map(|p| band_of(&head, order.as_deref(), p, 2).unwrap())
            .collect();
        assert_eq!(dims(&bands[0]), vec![10., 14.]);
        assert_eq!(dims(&bands[3]), vec![13., 17.]);
        let joined = Tensor::cat(&bands, 1).unwrap();
        let back = route_to_dims(joined, order.as_deref()).unwrap();
        assert_eq!(dims(&back), dims(&head));
    }

    /// A quantized band's blocks run one dim at a time over the tokens.
    #[test]
    fn channel_major_round_trips_to_token_major() {
        // 2 dims × 3 tokens, stored dim 0's tokens then dim 1's.
        let flat = Tensor::new(&[0f32, 1., 2., 10., 11., 12.], &Device::Cpu).unwrap();
        let tm = quantized_to_token_major(flat.clone(), 3, 2).unwrap();
        assert_eq!(dims(&tm), vec![0., 10., 1., 11., 2., 12.]);
        assert_eq!(dims(&token_major_to_quantized(&tm).unwrap()), dims(&flat));
    }

    #[test]
    fn outer_scale_stores_multiplied_and_decodes_divided() {
        let v = Tensor::new(&[2f32, -4.], &Device::Cpu).unwrap();
        let stored = scale(v.clone(), 4.0).unwrap();
        assert_eq!(dims(&stored), vec![8., -16.]);
        assert_eq!(dims(&unscale(stored, 4.0).unwrap()), vec![2., -4.]);
        assert_eq!(band_scale(&[], 3), 1.0);
        assert_eq!(band_scale(&[1., 2., 3., 0.5], 3), 0.5);
    }
}
