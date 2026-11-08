//! What one block of the room's audio holds (§4.4): its loudness, and its
//! energy in three bands — the low end a crowd moves to, the mids, the top.
//! Pure: samples in, levels out.

use std::sync::Arc;

use rustfft::num_complex::Complex;
use rustfft::{Fft, FftPlanner};

/// Samples a block is analysed over.
pub const BLOCK: usize = 2048;
/// The quietest level reported, in dBFS.
pub const FLOOR_DB: f32 = -90.0;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Levels {
    /// RMS level, dBFS.
    pub rms_db: f32,
    /// The fraction of the block's spectral energy below 150 Hz, 150 Hz to
    /// 2.5 kHz, and above.
    pub bands: [f32; 3],
}

pub struct Analyser {
    fft: Arc<dyn Fft<f32>>,
    rate: f32,
    buf: Vec<Complex<f32>>,
}

impl Analyser {
    pub fn new(rate: u32) -> Analyser {
        Analyser {
            fft: FftPlanner::new().plan_fft_forward(BLOCK),
            rate: rate as f32,
            buf: vec![Complex::default(); BLOCK],
        }
    }

    /// The levels of `mono`, [`BLOCK`] samples.
    pub fn levels(&mut self, mono: &[f32]) -> Levels {
        assert_eq!(mono.len(), BLOCK, "a block is {BLOCK} samples");
        let mean_sq = mono.iter().map(|x| x * x).sum::<f32>() / BLOCK as f32;
        let rms_db = if mean_sq > 0.0 {
            (10.0 * mean_sq.log10()).max(FLOOR_DB)
        } else {
            FLOOR_DB
        };
        for (i, (c, x)) in self.buf.iter_mut().zip(mono).enumerate() {
            // A Hann window, so a block's edges do not smear the spectrum.
            let w = 0.5 - 0.5 * (2.0 * std::f32::consts::PI * i as f32 / BLOCK as f32).cos();
            *c = Complex::new(x * w, 0.0);
        }
        self.fft.process(&mut self.buf);
        let hz = self.rate / BLOCK as f32;
        let mut bands = [0f32; 3];
        for (i, c) in self.buf[1..BLOCK / 2].iter().enumerate() {
            let f = (i + 1) as f32 * hz;
            let band = if f < 150.0 {
                0
            } else if f < 2500.0 {
                1
            } else {
                2
            };
            bands[band] += c.norm_sqr();
        }
        let total: f32 = bands.iter().sum();
        if total > 0.0 {
            for b in &mut bands {
                *b /= total;
            }
        }
        Levels { rms_db, bands }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tone(hz: f32, amp: f32, rate: u32) -> Vec<f32> {
        (0..BLOCK)
            .map(|i| amp * (2.0 * std::f32::consts::PI * hz * i as f32 / rate as f32).sin())
            .collect()
    }

    /// **A tone lands in its band, at its level**: a full-scale sine is
    /// −3 dBFS RMS; 60 Hz is the low end, 1 kHz the mids, 8 kHz the top.
    #[test]
    fn a_tone_lands_in_its_band_at_its_level() {
        let mut a = Analyser::new(48_000);
        let low = a.levels(&tone(60.0, 1.0, 48_000));
        assert!((low.rms_db + 3.01).abs() < 0.1, "{}", low.rms_db);
        assert!(low.bands[0] > 0.95, "{:?}", low.bands);
        assert!(a.levels(&tone(1000.0, 0.5, 48_000)).bands[1] > 0.95);
        let high = a.levels(&tone(8000.0, 0.1, 48_000));
        assert!(high.bands[2] > 0.95);
        assert!((high.rms_db + 23.01).abs() < 0.1);
        let silent = a.levels(&vec![0.0; BLOCK]);
        assert_eq!((silent.rms_db, silent.bands), (FLOOR_DB, [0.0; 3]));
    }
}
