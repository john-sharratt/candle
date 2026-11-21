//! The CPU matrix multiplication for batched operands.
//!
//! `MatMul` is a map over a left and a right operand laid out as batch, m, k. Its
//! skip offsets are computed per batch. The comment records that the output skip
//! is not updated, because its step is always zero, so that omission does not
//! change the result.

use super::*;

pub(super) struct MatMul(pub(super) (usize, usize, usize, usize));

impl MatMul {
    pub(super) fn striding_error(
        &self,
        lhs_l: &Layout,
        rhs_l: &Layout,
        msg: &'static str,
    ) -> Error {
        Error::MatMulUnexpectedStriding(Box::new(crate::error::MatMulUnexpectedStriding {
            lhs_l: lhs_l.clone(),
            rhs_l: rhs_l.clone(),
            bmnk: self.0,
            msg,
        }))
        .bt()
    }

    pub(super) fn ab_skip(&self, lhs_l: &Layout, rhs_l: &Layout) -> Result<(usize, usize)> {
        let lhs_stride = lhs_l.stride();
        let rhs_stride = rhs_l.stride();
        let rank = lhs_stride.len();
        let (_b, m, n, k) = self.0;
        let a_skip: usize = match lhs_stride[..rank - 2] {
            [s1, stride] if s1 == stride * lhs_l.dims()[1] => stride,
            [_, stride] if lhs_l.dims()[0] == 1 => stride,
            [stride, _] if lhs_l.dims()[1] == 1 => stride,
            [stride] => stride,
            [] => m * k,
            _ => Err(self.striding_error(lhs_l, rhs_l, "non-contiguous lhs"))?,
        };
        let b_skip: usize = match rhs_stride[..rank - 2] {
            [s1, stride] if s1 == stride * rhs_l.dims()[1] => stride,
            [_, stride] if rhs_l.dims()[0] == 1 => stride,
            [stride, _] if rhs_l.dims()[1] == 1 => stride,
            [stride] => stride,
            [] => n * k,
            _ => Err(self.striding_error(lhs_l, rhs_l, "non-contiguous rhs"))?,
        };
        Ok((a_skip, b_skip))
    }
}

impl Map2 for MatMul {
    const OP: &'static str = "mat_mul";

    #[cfg(all(not(feature = "mkl"), not(feature = "accelerate")))]
    fn f<T: 'static + WithDType + num_traits::Num + Copy>(
        &self,
        lhs: &[T],
        lhs_l: &Layout,
        rhs: &[T],
        rhs_l: &Layout,
    ) -> Result<Vec<T>> {
        use gemm::{gemm, Parallelism};

        match T::DTYPE {
            DType::F16 | DType::F32 | DType::F64 => {}
            _ => Err(Error::UnsupportedDTypeForOp(T::DTYPE, "matmul").bt())?,
        }

        let (b, m, n, k) = self.0;
        let lhs = &lhs[lhs_l.start_offset()..];
        let rhs = &rhs[rhs_l.start_offset()..];

        let lhs_stride = lhs_l.stride();
        let rhs_stride = rhs_l.stride();
        let rank = lhs_stride.len();
        let lhs_cs = lhs_stride[rank - 1];
        let lhs_rs = lhs_stride[rank - 2];

        let rhs_cs = rhs_stride[rank - 1];
        let rhs_rs = rhs_stride[rank - 2];

        let (a_skip, b_skip) = self.ab_skip(lhs_l, rhs_l)?;
        let c_skip: usize = m * n;

        let dst_shape: Shape = (m, n).into();
        let dst_strides = dst_shape.stride_contiguous();
        let dst_rs = dst_strides[0];
        let dst_cs = dst_strides[1];

        let mut dst = vec![T::zero(); b * m * n];
        let num_threads = crate::utils::get_num_threads();
        let parallelism = if num_threads > 1 {
            Parallelism::Rayon(num_threads)
        } else {
            Parallelism::None
        };
        let (b, m, n, k) = if b_skip == 0 && a_skip == m * k {
            // a_skip and c_skip should be updated but step is always 0 so
            // it wouldn't matter.
            (1, b * m, n, k)
        } else if a_skip == 0 && b_skip == n * k {
            (1, m, b * n, k)
        } else {
            (b, m, n, k)
        };
        for step in 0..b {
            let lhs_p = &lhs[step * a_skip..];
            let rhs_p = &rhs[step * b_skip..];
            let dst_p = &mut dst[step * c_skip..];
            unsafe {
                gemm(
                    /* m: usize = */ m,
                    /* n: usize = */ n,
                    /* k: usize = */ k,
                    /* dst: *mut T = */ dst_p.as_mut_ptr(),
                    /* dst_cs: isize = */ dst_cs as isize,
                    /* dst_rs: isize = */ dst_rs as isize,
                    /* read_dst: bool = */ false,
                    /* lhs: *const T = */ lhs_p.as_ptr(),
                    /* lhs_cs: isize = */ lhs_cs as isize,
                    /* lhs_rs: isize = */ lhs_rs as isize,
                    /* rhs: *const T = */ rhs_p.as_ptr(),
                    /* rhs_cs: isize = */ rhs_cs as isize,
                    /* rhs_rs: isize = */ rhs_rs as isize,
                    /* alpha: T = */ T::zero(),
                    /* beta: T = */ T::one(),
                    /* conj_dst: bool = */ false,
                    /* conj_lhs: bool = */ false,
                    /* conj_rhs: bool = */ false,
                    parallelism,
                )
            }
        }
        Ok(dst)
    }

    #[cfg(feature = "accelerate")]
    fn f<T: 'static + WithDType + num_traits::Num + Copy>(
        &self,
        lhs: &[T],
        lhs_l: &Layout,
        rhs: &[T],
        rhs_l: &Layout,
    ) -> Result<Vec<T>> {
        let (b, m, n, k) = self.0;
        let lhs = &lhs[lhs_l.start_offset()..];
        let rhs = &rhs[rhs_l.start_offset()..];

        let lhs_stride = lhs_l.stride();
        let rhs_stride = rhs_l.stride();

        let (a_skip, b_skip) = self.ab_skip(lhs_l, rhs_l)?;
        let c_skip: usize = m * n;

        let rhs_m1 = rhs_stride[rhs_stride.len() - 1];
        let rhs_m2 = rhs_stride[rhs_stride.len() - 2];
        let lhs_m1 = lhs_stride[lhs_stride.len() - 1];
        let lhs_m2 = lhs_stride[lhs_stride.len() - 2];

        let (lda, transa) = if (rhs_m1 == 1 || n == 1) && (rhs_m2 == n || k == 1) {
            (n as i32, b'N')
        } else if rhs_m1 == k && rhs_m2 == 1 {
            (k as i32, b'T')
        } else {
            Err(self.striding_error(lhs_l, rhs_l, "non-contiguous rhs"))?
        };
        // The b tensor has dims batching, m, k (lhs)
        let (ldb, transb) = if (lhs_m1 == 1 || k == 1) && (lhs_m2 == k || m == 1) {
            (k as i32, b'N')
        } else if lhs_m1 == m && lhs_m2 == 1 {
            (m as i32, b'T')
        } else {
            Err(self.striding_error(lhs_l, rhs_l, "non-contiguous lhs"))?
        };

        let mut dst = vec![T::zero(); b * m * n];
        match T::DTYPE {
            DType::F16 => {
                crate::bail!("the accelerate backend does not support f16 matmul")
            }
            DType::F32 => {
                for step in 0..b {
                    let lhs_p = &lhs[step * a_skip..];
                    let rhs_p = &rhs[step * b_skip..];
                    let dst_p = &mut dst[step * c_skip..];
                    unsafe {
                        let a = rhs_p.as_ptr() as *const f32;
                        let b = lhs_p.as_ptr() as *const f32;
                        let c = dst_p.as_mut_ptr() as *mut f32;
                        let a = std::slice::from_raw_parts(a, a_skip);
                        let b = std::slice::from_raw_parts(b, b_skip);
                        let c = std::slice::from_raw_parts_mut(c, c_skip);
                        crate::accelerate::sgemm(
                            transa, transb, /* m= */ n as i32, /* n= */ m as i32,
                            /* k= */ k as i32, /* alpha= */ 1., /* a= */ a,
                            /* lda= */ lda, /* b= */ b, /* ldb= */ ldb,
                            /* beta= */ 0., /* c= */ c, /* ldc= */ n as i32,
                        )
                    }
                }
            }
            DType::F64 => {
                for step in 0..b {
                    let lhs_p = &lhs[step * a_skip..];
                    let rhs_p = &rhs[step * b_skip..];
                    let dst_p = &mut dst[step * c_skip..];
                    unsafe {
                        let a = rhs_p.as_ptr() as *const f64;
                        let b = lhs_p.as_ptr() as *const f64;
                        let c = dst_p.as_mut_ptr() as *mut f64;
                        let a = std::slice::from_raw_parts(a, a_skip);
                        let b = std::slice::from_raw_parts(b, b_skip);
                        let c = std::slice::from_raw_parts_mut(c, c_skip);
                        crate::accelerate::dgemm(
                            transa, transb, /* m= */ n as i32, /* n= */ m as i32,
                            /* k= */ k as i32, /* alpha= */ 1., /* a= */ a,
                            /* lda= */ lda, /* b= */ b, /* ldb= */ ldb,
                            /* beta= */ 0., /* c= */ c, /* ldc= */ n as i32,
                        )
                    }
                }
            }
            dtype => Err(Error::UnsupportedDTypeForOp(dtype, "matmul").bt())?,
        }
        Ok(dst)
    }

    #[cfg(feature = "mkl")]
    fn f<T: 'static + WithDType + num_traits::Num + Copy>(
        &self,
        lhs: &[T],
        lhs_l: &Layout,
        rhs: &[T],
        rhs_l: &Layout,
    ) -> Result<Vec<T>> {
        let (b, m, n, k) = self.0;
        let lhs = &lhs[lhs_l.start_offset()..];
        let rhs = &rhs[rhs_l.start_offset()..];

        let lhs_stride = lhs_l.stride();
        let rhs_stride = rhs_l.stride();

        let (a_skip, b_skip) = self.ab_skip(lhs_l, rhs_l)?;
        let c_skip: usize = m * n;

        let rhs_m1 = rhs_stride[rhs_stride.len() - 1];
        let rhs_m2 = rhs_stride[rhs_stride.len() - 2];
        let lhs_m1 = lhs_stride[lhs_stride.len() - 1];
        let lhs_m2 = lhs_stride[lhs_stride.len() - 2];

        let (lda, transa) = if (rhs_m1 == 1 || n == 1) && (rhs_m2 == n || k == 1) {
            (n as i32, b'N')
        } else if rhs_m1 == k && rhs_m2 == 1 {
            (k as i32, b'T')
        } else {
            Err(self.striding_error(lhs_l, rhs_l, "non-contiguous rhs"))?
        };
        // The b tensor has dims batching, m, k (lhs)
        let (ldb, transb) = if (lhs_m1 == 1 || k == 1) && (lhs_m2 == k || m == 1) {
            (k as i32, b'N')
        } else if lhs_m1 == m && lhs_m2 == 1 {
            (m as i32, b'T')
        } else {
            Err(self.striding_error(lhs_l, rhs_l, "non-contiguous lhs"))?
        };

        let mut dst = vec![T::zero(); b * m * n];
        match T::DTYPE {
            DType::F16 => {
                for step in 0..b {
                    let lhs_p = &lhs[step * a_skip..];
                    let rhs_p = &rhs[step * b_skip..];
                    let dst_p = &mut dst[step * c_skip..];
                    unsafe {
                        let a = rhs_p.as_ptr() as *const f16;
                        let b = lhs_p.as_ptr() as *const f16;
                        let c = dst_p.as_mut_ptr() as *mut f16;
                        let a = std::slice::from_raw_parts(a, a_skip);
                        let b = std::slice::from_raw_parts(b, b_skip);
                        let c = std::slice::from_raw_parts_mut(c, c_skip);
                        crate::mkl::hgemm(
                            transa,
                            transb,
                            /* m= */ n as i32,
                            /* n= */ m as i32,
                            /* k= */ k as i32,
                            /* alpha= */ f16::ONE,
                            /* a= */ a,
                            /* lda= */ lda,
                            /* b= */ b,
                            /* ldb= */ ldb,
                            /* beta= */ f16::ZERO,
                            /* c= */ c,
                            /* ldc= */ n as i32,
                        )
                    }
                }
            }
            DType::F32 => {
                for step in 0..b {
                    let lhs_p = &lhs[step * a_skip..];
                    let rhs_p = &rhs[step * b_skip..];
                    let dst_p = &mut dst[step * c_skip..];
                    unsafe {
                        let a = rhs_p.as_ptr() as *const f32;
                        let b = lhs_p.as_ptr() as *const f32;
                        let c = dst_p.as_mut_ptr() as *mut f32;
                        let a = std::slice::from_raw_parts(a, a_skip);
                        let b = std::slice::from_raw_parts(b, b_skip);
                        let c = std::slice::from_raw_parts_mut(c, c_skip);
                        crate::mkl::sgemm(
                            transa, transb, /* m= */ n as i32, /* n= */ m as i32,
                            /* k= */ k as i32, /* alpha= */ 1., /* a= */ a,
                            /* lda= */ lda, /* b= */ b, /* ldb= */ ldb,
                            /* beta= */ 0., /* c= */ c, /* ldc= */ n as i32,
                        )
                    }
                }
            }
            DType::F64 => {
                for step in 0..b {
                    let lhs_p = &lhs[step * a_skip..];
                    let rhs_p = &rhs[step * b_skip..];
                    let dst_p = &mut dst[step * c_skip..];
                    unsafe {
                        let a = rhs_p.as_ptr() as *const f64;
                        let b = lhs_p.as_ptr() as *const f64;
                        let c = dst_p.as_mut_ptr() as *mut f64;
                        let a = std::slice::from_raw_parts(a, a_skip);
                        let b = std::slice::from_raw_parts(b, b_skip);
                        let c = std::slice::from_raw_parts_mut(c, c_skip);
                        crate::mkl::dgemm(
                            transa, transb, /* m= */ n as i32, /* n= */ m as i32,
                            /* k= */ k as i32, /* alpha= */ 1., /* a= */ a,
                            /* lda= */ lda, /* b= */ b, /* ldb= */ ldb,
                            /* beta= */ 0., /* c= */ c, /* ldc= */ n as i32,
                        )
                    }
                }
            }
            dtype => Err(Error::UnsupportedDTypeForOp(dtype, "matmul").bt())?,
        }
        Ok(dst)
    }
}
