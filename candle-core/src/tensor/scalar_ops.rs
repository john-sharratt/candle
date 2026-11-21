//! Operators between an `f64` scalar and a tensor.
//!
//! Each of the four operators is implemented with the scalar on either side and
//! for both the owned and the borrowed tensor, so `2.0 * &t` and `2.0 * t` both
//! compile. The `bin_trait!` lines define how each operator applies a scalar
//! element by element.

use super::*;

bin_trait!(Add, add, |_| 1., |v| v);

bin_trait!(Sub, sub, |_| 1., |v: f64| -v);

bin_trait!(Mul, mul, |v| v, |_| 0.);

bin_trait!(Div, div, |v| 1. / v, |_| 0.);

impl<'w> std::ops::Add<LiveTensor<'w>> for f64 {
    type Output = Result<LiveTensor<'w>>;

    fn add(self, rhs: LiveTensor<'w>) -> Self::Output {
        rhs + self
    }
}

impl<'w> std::ops::Add<&LiveTensor<'w>> for f64 {
    type Output = Result<LiveTensor<'w>>;

    fn add(self, rhs: &LiveTensor<'w>) -> Self::Output {
        rhs + self
    }
}

impl<'w> std::ops::Mul<LiveTensor<'w>> for f64 {
    type Output = Result<LiveTensor<'w>>;

    fn mul(self, rhs: LiveTensor<'w>) -> Self::Output {
        rhs * self
    }
}

impl<'w> std::ops::Mul<&LiveTensor<'w>> for f64 {
    type Output = Result<LiveTensor<'w>>;

    fn mul(self, rhs: &LiveTensor<'w>) -> Self::Output {
        rhs * self
    }
}

impl<'w> std::ops::Sub<LiveTensor<'w>> for f64 {
    type Output = Result<LiveTensor<'w>>;

    fn sub(self, rhs: LiveTensor<'w>) -> Self::Output {
        rhs.affine(-1., self)
    }
}

impl<'w> std::ops::Sub<&LiveTensor<'w>> for f64 {
    type Output = Result<LiveTensor<'w>>;

    fn sub(self, rhs: &LiveTensor<'w>) -> Self::Output {
        rhs.affine(-1., self)
    }
}

impl<'w> std::ops::Div<LiveTensor<'w>> for f64 {
    type Output = Result<LiveTensor<'w>>;

    #[allow(clippy::suspicious_arithmetic_impl)]
    fn div(self, rhs: LiveTensor<'w>) -> Self::Output {
        rhs.recip()? * self
    }
}

impl<'w> std::ops::Div<&LiveTensor<'w>> for f64 {
    type Output = Result<LiveTensor<'w>>;

    #[allow(clippy::suspicious_arithmetic_impl)]
    fn div(self, rhs: &LiveTensor<'w>) -> Self::Output {
        rhs.recip()? * self
    }
}
