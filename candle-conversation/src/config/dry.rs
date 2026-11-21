//! Configures the DRY penalty that discourages continuing a repeated n-gram.
//!
//! The penalty grows exponentially with how far a repeated match runs past an
//! allowed length, so a short phrase that recurs is left alone while a longer
//! repeat is pushed off. A multiplier of zero disables it, and the look-back
//! range bounds how far back matches are searched.

/// DRY (Don't Repeat Yourself) penalty configuration.
///
/// Detects when the model is about to continue a repeated n-gram pattern
/// and applies exponential penalty based on match length.
///
/// Example: If the context contains "the cat sat on the mat" and later
/// "the cat", the DRY penalty will penalize "sat" to prevent repetition.
#[derive(Debug, Clone)]
pub struct DryConfig {
    /// Base penalty multiplier. `0.0` = disabled.
    /// Applied as: `penalty = multiplier * base^(match_length - allowed_length)`
    pub multiplier: f32,

    /// Exponential base for longer matches. Typical: 1.5 - 2.0.
    /// Higher values penalize longer matches more aggressively.
    pub base: f32,

    /// N-grams up to this length are allowed without penalty.
    /// Typical: 1-3. Allows common short phrases without penalty.
    pub allowed_length: i32,

    /// How far back to search for matching patterns (in tokens).
    /// `0` = search full history. Typical: 256 - 512 tokens.
    pub range: i32,
}

impl Default for DryConfig {
    fn default() -> Self {
        Self {
            multiplier: 0.8,
            base: 1.75,
            allowed_length: 2,
            range: 256,
        }
    }
}
