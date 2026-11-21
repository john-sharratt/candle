//! Tests that the int8 sizing rule chooses Precision or Performance on the
//! seven-tenths line of the card's total VRAM.
//!
//! The boundary is inclusive, so a model exactly at seven tenths of the card
//! keeps the accurate twin, and one byte more drops to the smaller one. The
//! fixtures are two real quantized models measured on a 16 GB card: a 30B
//! mixture-of-experts model and an 8B model, which fall on opposite sides of the
//! line.

use super::*;

#[cfg(test)]
pub(super) mod int8_mode_sizing_tests {
    use super::Int8Mode;

    pub(in super::super) const GIB: usize = 1 << 30;

    /// On the 16 GB card the 30B-A3B Q4_K_M (17.8 GB) and an 8B Q6_K (6.7 GB)
    /// fall either side of the 7/10 line.
    #[test]
    fn the_card_total_decides_the_mode() {
        let card = 16 * GIB;
        assert_eq!(
            Int8Mode::sized_for_card(17_817_403_392, card),
            Int8Mode::Performance
        );
        assert_eq!(
            Int8Mode::sized_for_card(6_700_000_000, card),
            Int8Mode::Precision
        );
    }

    /// The line itself is Precision; one byte past it is Performance.
    #[test]
    fn the_seven_tenths_line_is_inclusive() {
        let card = 10 * GIB;
        assert_eq!(Int8Mode::sized_for_card(7 * GIB, card), Int8Mode::Precision);
        assert_eq!(
            Int8Mode::sized_for_card(7 * GIB + 1, card),
            Int8Mode::Performance
        );
    }
}
