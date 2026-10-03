//! The host side of the KV block encoders' warp arithmetic.
//!
//! The KV block encoders run one warp per 32-element block and reduce with
//! xor butterflies (`__shfl_xor_sync`). A butterfly sum adds in a fixed tree
//! order, so the host codecs that must reproduce the kernels' bytes add in the
//! same order; every lane ends holding the same value (each round adds a pair
//! both partners compute identically), so one scalar stands for the warp.
//! Max and min are order-independent and need no mirror.
//!
//! The kernels round a float to int with `__float2int_rn` (ties to even) in
//! some places and `roundf` (ties away from zero) in others; the helpers name
//! which.

/// `1.0f / 127.0f` as the kernels spell it: a rounded constant multiplied in,
/// not a division.
pub(crate) const INV_127: f32 = 1.0 / 127.0;
/// `INV_255` in `convert.cuh`.
pub(crate) const INV_255: f32 = 1.0 / 255.0;

/// A full-warp xor-butterfly sum (offsets 16, 8, 4, 2, 1).
pub(crate) fn warp_sum(lanes: &[f32]) -> f32 {
    debug_assert_eq!(lanes.len(), 32);
    let mut p: [f32; 32] = std::array::from_fn(|i| lanes[i]);
    for off in [16, 8, 4, 2, 1] {
        p = std::array::from_fn(|i| p[i] + p[i ^ off]);
    }
    p[0]
}

/// A width-8 xor-butterfly sum (offsets 4, 2, 1) over the eight lanes of a
/// vectorised encoder, each lane holding four elements' partial.
pub(crate) fn lane8_sum(lanes: &[f32; 8]) -> f32 {
    let mut p = *lanes;
    for off in [4, 2, 1] {
        p = std::array::from_fn(|i| p[i] + p[i ^ off]);
    }
    p[0]
}

/// `__float2int_rn`: round to nearest, ties to even, saturating.
pub(crate) fn rint(x: f32) -> i32 {
    x.round_ties_even() as i32
}

/// `q0_encode_centroid`: a float in outer-normalised units to its INT8 code.
pub(crate) fn encode_centroid(v: f32) -> i8 {
    rint((v * 127.0).clamp(-127.0, 127.0)) as i8
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn warp_sum_adds_in_butterfly_order() {
        // Lane 0 of the butterfly computes ((x0+x16)+(x8+x24)) + … — not a
        // left-to-right fold, which this corpus distinguishes.
        let mut lanes = [1.0e-8f32; 32];
        lanes[0] = 1.0;
        lanes[16] = -1.0;
        let sequential: f32 = lanes.iter().sum();
        // Lane 0's tree: the ±1 pair cancels in round one, and the other
        // thirty lanes arrive as partials of 2, 4, 8 and 16 terms.
        let p2 = 1.0e-8f32 + 1.0e-8;
        let (p4, p8) = (p2 + p2, (p2 + p2) + (p2 + p2));
        let p16 = p8 + p8;
        let tree = ((p2 + p4) + p8) + p16;
        assert_eq!(warp_sum(&lanes).to_bits(), tree.to_bits());
        assert_ne!(warp_sum(&lanes).to_bits(), sequential.to_bits());
    }

    #[test]
    fn rint_ties_to_even() {
        assert_eq!([rint(0.5), rint(1.5), rint(2.5), rint(-0.5)], [0, 2, 2, 0]);
    }
}
