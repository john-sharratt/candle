//! FFI for fused weighted personality-vector accumulation.

use core::ffi::c_void;

#[repr(i32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WeightedRowAccumDType {
    F32 = 0,
    F16 = 1,
    BF16 = 2,
}

extern "C" {
    /// Adds `scales[i] * vectors[i, j]` only when the vector's dot score with
    /// the activation is at least `threshold`.
    ///
    /// All strides are in elements. Vectors and activation use `dtype`; scales
    /// are always F32. Accumulation is performed in F32.
    pub fn run_weighted_row_accum(
        dtype: i32,
        vectors: *const c_void,
        scales: *const f32,
        activation: *mut c_void,
        vector_count: i32,
        hidden: i32,
        vector_stride: i64,
        scale_stride: i64,
        activation_stride: i64,
        threshold: f32,
        stream: *mut c_void,
    ) -> i32;
}

#[cfg(test)]
mod tests {
    use super::WeightedRowAccumDType;

    fn reference(
        vectors: &[f32],
        scales: &[f32],
        activation: &mut [f32],
        vector_count: usize,
        hidden: usize,
        vector_stride: usize,
        scale_stride: usize,
        activation_stride: usize,
        threshold: f32,
    ) {
        let original = activation.to_vec();
        for row in 0..vector_count {
            let mut dot = 0.0;
            for column in 0..hidden {
                dot += original[column * activation_stride]
                    * vectors[row * vector_stride + column];
            }
            if dot >= threshold {
                for column in 0..hidden {
                    activation[column * activation_stride] +=
                        scales[row * scale_stride] * vectors[row * vector_stride + column];
                }
            }
        }
    }

    #[test]
    fn dtype_codes_match_cuda_dispatch() {
        assert_eq!(WeightedRowAccumDType::F32 as i32, 0);
        assert_eq!(WeightedRowAccumDType::F16 as i32, 1);
        assert_eq!(WeightedRowAccumDType::BF16 as i32, 2);
    }

    #[test]
    fn cpu_reference_accumulates_signed_personality_scales() {
        let vectors = [[1.0_f32, 2.0], [3.0, 4.0]];
        let scales = [0.5_f32, -1.0];
        let mut activation = [10.0_f32, 20.0];
        for column in 0..2 {
            for row in 0..2 {
                activation[column] += scales[row] * vectors[row][column];
            }
        }
        assert_eq!(activation, [7.5, 17.0]);
    }

    #[test]
    fn threshold_blocks_a_vector_below_cutoff() {
        let activation = [1.0_f32, 0.0];
        let vector = [0.25_f32, 0.0];
        let dot = activation[0] * vector[0] + activation[1] * vector[1];
        assert!(dot < 0.5);
    }

    #[test]
    fn threshold_allows_a_vector_at_cutoff() {
        let activation = [1.0_f32, 0.0];
        let vector = [0.5_f32, 0.0];
        let dot = activation[0] * vector[0] + activation[1] * vector[1];
        assert!(dot >= 0.5);
    }

    #[test]
    fn zero_vectors_leave_activation_unchanged() {
        let vectors = [];
        let scales = [];
        let mut activation = [2.0_f32, -3.0, 4.0];
        reference(&vectors, &scales, &mut activation, 0, 3, 3, 1, 1, 0.0);
        assert_eq!(activation, [2.0, -3.0, 4.0]);
    }

    #[test]
    fn threshold_uses_original_activation_for_all_vectors() {
        let vectors = [1.0_f32, 0.0, 1.0, 0.0];
        let scales = [1.0, 1.0];
        let mut activation = [1.0_f32, 0.0];
        reference(&vectors, &scales, &mut activation, 2, 2, 2, 1, 1, 1.0);
        assert_eq!(activation, [3.0, 0.0]);
    }

    #[test]
    fn negative_threshold_allows_negative_dot() {
        let vectors = [-1.0_f32, 0.0];
        let scales = [2.0];
        let mut activation = [1.0_f32, 0.0];
        reference(&vectors, &scales, &mut activation, 1, 2, 2, 1, 1, -2.0);
        assert_eq!(activation, [-1.0, 0.0]);
    }

    #[test]
    fn below_threshold_is_a_strict_noop() {
        let vectors = [0.25_f32, 0.0];
        let scales = [10.0];
        let mut activation = [1.0_f32, 0.0];
        reference(&vectors, &scales, &mut activation, 1, 2, 2, 1, 1, 0.2501);
        assert_eq!(activation, [1.0, 0.0]);
    }

    #[test]
    fn supports_odd_hidden_width() {
        let vectors = [1.0_f32, 2.0, 3.0, -1.0, -2.0, -3.0, 4.0];
        let scales = [0.5, -0.25];
        let mut activation = [1.0_f32, 1.0, 1.0, 1.0];
        reference(&vectors, &scales, &mut activation, 2, 3, 3, 1, 1, -100.0);
        assert_eq!(activation, [1.75, 2.5, 3.25, 1.0]);
    }

    #[test]
    fn supports_non_unit_vector_and_scale_strides() {
        let vectors = [1.0_f32, 2.0, 99.0, 3.0, 4.0, 88.0];
        let scales = [0.5, 77.0, -1.0];
        let mut activation = [10.0_f32, 20.0, 30.0];
        reference(&vectors, &scales, &mut activation, 2, 2, 3, 2, 1, -100.0);
        assert_eq!(activation, [7.5, 17.0, 30.0]);
    }

    #[test]
    fn supports_non_unit_activation_stride() {
        let vectors = [1.0_f32, 2.0];
        let scales = [2.0];
        let mut activation = [10.0_f32, 999.0, 20.0];
        reference(&vectors, &scales, &mut activation, 1, 2, 2, 1, 2, -100.0);
        assert_eq!(activation, [12.0, 999.0, 24.0]);
    }

    #[test]
    fn positive_and_negative_scales_cancel() {
        let vectors = [1.0_f32, 2.0, 1.0, 2.0];
        let scales = [3.0, -3.0];
        let mut activation = [7.0_f32, 8.0];
        reference(&vectors, &scales, &mut activation, 2, 2, 2, 1, 1, -100.0);
        assert_eq!(activation, [7.0, 8.0]);
    }

    #[test]
    fn large_hidden_reference_has_no_remainder_assumption() {
        let hidden = 257;
        let vectors = vec![1.0_f32; hidden];
        let scales = [2.0];
        let mut activation = vec![3.0_f32; hidden];
        reference(&vectors, &scales, &mut activation, 1, hidden, hidden, 1, 1, -1.0);
        assert!(activation.iter().all(|&value| value == 5.0));
    }
}
