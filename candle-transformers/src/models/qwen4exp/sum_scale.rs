//! How this model's int8 activations store their per-128 `Σx`.

use candle::quantized::SumScale;

/// Every q8a128 activation this model quantizes stores `Σx/amax`.
///
/// **Its block sums pass f16.** A token's FFN input after a full-attention
/// layer can carry one channel large enough that its 128-element block sums
/// past 65504 while `amax/127` still fits: stored raw, that sum is `+inf`, and
/// the KO weights' min term (`m_w·Σx`) turns every expert row gathered from the
/// token non-finite. On WhiteWorld an expert call came out NaN in every element
/// of its gate and up outputs from a finite input, and NaN logits followed full
/// attention at layers 11, 19, 23, 27, 31 and 39. Stored by amax the field is
/// bounded at 128 whatever the magnitude, at the same cost and precision
/// (`SumScale::ByAmax`).
pub const ACT_SUM_SCALE: SumScale = SumScale::ByAmax;
