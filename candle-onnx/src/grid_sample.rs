//! `GridSample`: an image read at normalised coordinates.
//!
//! <https://github.com/onnx/onnx/blob/main/docs/Operators.md#GridSample>. `x` is
//! `[N, C, H, W]` and `grid` is `[N, Ho, Wo, 2]`, each entry an `(x, y)` pair in
//! `[-1, 1]`; the answer is `[N, C, Ho, Wo]`. Exporters lower a deformable
//! convolution to it — BiRefNet's decoder samples each tap at a learnt offset —
//! so it is the one op such a graph cannot run without.
//!
//! Built from tensor ops that stay on the tensor's device: the four corners'
//! flat indices and weights are computed for every output position, gathered
//! from `x` viewed as `[N, C, H·W]`, and blended. Coordinates are worked in
//! `f32` whatever `x` is, and the answer is cast back to `x`'s dtype.

use candle::{bail, DType, Result, Tensor};

/// How a position between pixels is read.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Mode {
    /// The four surrounding pixels, weighted by distance — ONNX `linear`
    /// (opset 20) or `bilinear` (opset 16).
    Bilinear,
    /// The nearest pixel, ties to even as the reference's `np.rint`.
    Nearest,
}

/// What a position outside the image reads.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Padding {
    /// Zero: a tap outside the image contributes nothing.
    Zeros,
    /// The nearest edge pixel: the coordinate is clamped into the image.
    Border,
}

impl Mode {
    /// The ONNX attribute's value, as either opset spells it.
    pub fn parse(s: &str) -> Result<Self> {
        match s {
            "linear" | "bilinear" => Ok(Self::Bilinear),
            "nearest" => Ok(Self::Nearest),
            other => bail!("GridSample mode {other} is not implemented"),
        }
    }
}

impl Padding {
    pub fn parse(s: &str) -> Result<Self> {
        match s {
            "zeros" => Ok(Self::Zeros),
            "border" => Ok(Self::Border),
            other => bail!("GridSample padding_mode {other} is not implemented"),
        }
    }
}

/// Sample `x` at `grid` — see the module documentation.
pub fn grid_sample(
    x: &Tensor,
    grid: &Tensor,
    mode: Mode,
    padding: Padding,
    align_corners: bool,
) -> Result<Tensor> {
    let (n, c, h, w) = x.dims4()?;
    let (gn, ho, wo, two) = grid.dims4()?;
    if gn != n || two != 2 {
        bail!(
            "GridSample: grid is {:?} for an input of {:?}; it must be [N, Ho, Wo, 2]",
            grid.dims(),
            x.dims()
        );
    }
    let grid = grid.to_dtype(DType::F32)?;
    let ix = unnormalise(&grid.narrow(3, 0, 1)?.squeeze(3)?, w, align_corners)?;
    let iy = unnormalise(&grid.narrow(3, 1, 1)?.squeeze(3)?, h, align_corners)?;
    let (ix, iy) = match padding {
        Padding::Zeros => (ix, iy),
        Padding::Border => (
            ix.clamp(0f32, (w - 1) as f32)?,
            iy.clamp(0f32, (h - 1) as f32)?,
        ),
    };

    let flat = x.reshape((n, c, h * w))?;
    let positions = ho * wo;
    let read = |xs: &Tensor, ys: &Tensor, weight: &Tensor| -> Result<Tensor> {
        // A tap outside the image weighs nothing; its index is clamped only so
        // the gather stays in bounds, and what it reads is multiplied away.
        let inside = xs
            .ge(0f32)?
            .mul(&xs.le((w - 1) as f32)?)?
            .mul(&ys.ge(0f32)?)?
            .mul(&ys.le((h - 1) as f32)?)?
            .to_dtype(DType::F32)?;
        let index = (ys.clamp(0f32, (h - 1) as f32)? * w as f64)?
            .add(&xs.clamp(0f32, (w - 1) as f32)?)?
            .to_dtype(DType::U32)?
            .reshape((n, 1, positions))?
            .broadcast_as((n, c, positions))?
            .contiguous()?;
        let weight = weight.mul(&inside)?.reshape((n, 1, positions))?;
        flat.gather(&index, 2)?
            .to_dtype(DType::F32)?
            .broadcast_mul(&weight)
    };

    let out = match mode {
        Mode::Bilinear => {
            let (x0, y0) = (ix.floor()?, iy.floor()?);
            let (x1, y1) = ((&x0 + 1.0)?, (&y0 + 1.0)?);
            let (wx1, wy1) = (ix.sub(&x0)?, iy.sub(&y0)?);
            let (wx0, wy0) = (wx1.affine(-1.0, 1.0)?, wy1.affine(-1.0, 1.0)?);
            let a = read(&x0, &y0, &wx0.mul(&wy0)?)?;
            let b = read(&x1, &y0, &wx1.mul(&wy0)?)?;
            let cc = read(&x0, &y1, &wx0.mul(&wy1)?)?;
            let d = read(&x1, &y1, &wx1.mul(&wy1)?)?;
            ((a + b)? + (cc + d)?)?
        }
        Mode::Nearest => read(
            &round_half_even(&ix)?,
            &round_half_even(&iy)?,
            &ix.ones_like()?,
        )?,
    };
    out.reshape((n, c, ho, wo))?.to_dtype(x.dtype())
}

/// A normalised coordinate in pixels: `align_corners` puts -1 and 1 on the
/// centres of the corner pixels, and without it on their outer edges.
fn unnormalise(g: &Tensor, size: usize, align_corners: bool) -> Result<Tensor> {
    let size = size as f64;
    if align_corners {
        g.affine((size - 1.0) / 2.0, (size - 1.0) / 2.0)
    } else {
        g.affine(size / 2.0, (size - 1.0) / 2.0)
    }
}

/// Round to the nearest integer, a half to the even neighbour.
fn round_half_even(v: &Tensor) -> Result<Tensor> {
    let f = v.floor()?;
    let frac = v.sub(&f)?;
    // Odd when f − 2·⌊f/2⌋ is 1.
    let odd = f.sub(&(f.affine(0.5, 0.0)?.floor()? * 2.0)?)?.eq(1f32)?;
    let up = frac
        .gt(0.5f32)?
        .add(&frac.eq(0.5f32)?.mul(&odd)?)?
        .gt(0u8)?;
    up.where_cond(&(&f + 1.0)?, &f)
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle::Device;

    /// A 3×3 image whose pixels are their own row-major index.
    fn image() -> Tensor {
        Tensor::arange(0f32, 9.0, &Device::Cpu)
            .unwrap()
            .reshape((1, 1, 3, 3))
            .unwrap()
    }

    fn grid(points: &[(f32, f32)]) -> Tensor {
        let flat: Vec<f32> = points.iter().flat_map(|&(x, y)| [x, y]).collect();
        Tensor::from_vec(flat, (1, 1, points.len(), 2), &Device::Cpu).unwrap()
    }

    fn sample(points: &[(f32, f32)], mode: Mode, padding: Padding, align: bool) -> Vec<f32> {
        grid_sample(&image(), &grid(points), mode, padding, align)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1()
            .unwrap()
    }

    /// With corners aligned, -1 and 1 are the corner pixels' centres, and 0 the
    /// middle pixel's: the samples are the pixels themselves.
    #[test]
    fn aligned_corners_land_on_pixel_centres() {
        let got = sample(
            &[
                (-1.0, -1.0),
                (1.0, -1.0),
                (0.0, 0.0),
                (-1.0, 1.0),
                (1.0, 1.0),
            ],
            Mode::Bilinear,
            Padding::Zeros,
            true,
        );
        assert_eq!(got, vec![0.0, 2.0, 4.0, 6.0, 8.0]);
    }

    /// Halfway between pixels is their mean, and a quarter of the way in both
    /// axes is the four-way blend: (0, 1, 3, 4) weighted 9:3:3:1 over 16.
    #[test]
    fn bilinear_blends_by_distance() {
        let got = sample(
            &[(-0.5, -1.0), (-0.75, -0.75)],
            Mode::Bilinear,
            Padding::Zeros,
            true,
        );
        assert_eq!(got[0], 0.5);
        assert_eq!(
            got[1],
            (0.0 * 9.0 + 1.0 * 3.0 + 3.0 * 3.0 + 4.0 * 1.0) / 16.0
        );
    }

    /// Without aligned corners, -1 is the outer edge of the corner pixel — half
    /// a pixel outside its centre. Zero padding blends that with nothing; border
    /// padding reads the edge pixel itself.
    #[test]
    fn unaligned_corners_reach_half_a_pixel_out() {
        let zeros = sample(
            &[(-1.0, -1.0), (0.0, 0.0)],
            Mode::Bilinear,
            Padding::Zeros,
            false,
        );
        assert_eq!(zeros, vec![0.0 * 0.25, 4.0]);
        let edge = sample(&[(-1.0, 1.0)], Mode::Bilinear, Padding::Zeros, false);
        // Pixel 6 at a quarter weight; the other three taps are outside.
        assert_eq!(edge, vec![6.0 * 0.25]);
        let border = sample(&[(-1.0, 1.0)], Mode::Bilinear, Padding::Border, false);
        assert_eq!(border, vec![6.0]);
    }

    /// Far outside the image, zero padding reads zero.
    #[test]
    fn a_sample_outside_reads_zero() {
        let got = sample(
            &[(3.0, 0.0), (0.0, -3.0)],
            Mode::Bilinear,
            Padding::Zeros,
            true,
        );
        assert_eq!(got, vec![0.0, 0.0]);
    }

    /// Nearest takes the closer pixel, and a tie the even one: 0.5 → 0 and
    /// 1.5 → 2 (pixel coordinates, aligned corners on a 3-wide row).
    #[test]
    fn nearest_rounds_ties_to_even() {
        let got = sample(
            &[(-0.5, -1.0), (0.5, -1.0), (-0.4, -1.0), (0.1, 1.0)],
            Mode::Nearest,
            Padding::Zeros,
            true,
        );
        // x = -0.5 → 0.5 px → 0; x = 0.5 → 1.5 px → 2; x = -0.4 → 0.6 px → 1;
        // (0.1, 1.0) → (1.1, 2) px → pixel 7.
        assert_eq!(got, vec![0.0, 2.0, 1.0, 7.0]);
    }

    /// Every channel is read at the same positions.
    #[test]
    fn channels_share_the_grid() {
        let two = Tensor::cat(&[image(), (image() * 10.0).unwrap()], 1).unwrap();
        let got: Vec<f32> = grid_sample(
            &two,
            &grid(&[(0.0, 0.0)]),
            Mode::Bilinear,
            Padding::Zeros,
            true,
        )
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1()
        .unwrap();
        assert_eq!(got, vec![4.0, 40.0]);
    }

    #[test]
    fn a_grid_of_the_wrong_shape_is_refused() {
        let bad = Tensor::zeros((1, 1, 1, 3), DType::F32, &Device::Cpu).unwrap();
        assert!(grid_sample(&image(), &bad, Mode::Bilinear, Padding::Zeros, true).is_err());
        assert!(Mode::parse("cubic").is_err());
        assert_eq!(Mode::parse("linear").unwrap(), Mode::Bilinear);
        assert_eq!(Mode::parse("bilinear").unwrap(), Mode::Bilinear);
    }
}
