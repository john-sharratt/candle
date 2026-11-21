//! The CPU element-wise operations: unary and binary maps, comparisons, conditional
//! selects and ELU.
//!
//! Each operation is a small struct that the generic map driver applies per dtype,
//! with a fast path when both operands are contiguous and a strided slow path
//! otherwise. Min and max are not defined for integer types, to match CUDA. Half
//! and bfloat16 operations go through f32, because converting one type directly to
//! f64 never returned in the all-pairs cast test. In-place operations require the
//! left operand to be contiguous.

use super::*;

pub(super) struct Cmp(pub(super) CmpOp);

impl Map2U8 for Cmp {
    const OP: &'static str = "cmp";
    #[inline(always)]
    fn f<T: WithDType>(
        &self,
        lhs: &[T],
        lhs_l: &Layout,
        rhs: &[T],
        rhs_l: &Layout,
    ) -> Result<Vec<u8>> {
        let dst = match self.0 {
            CmpOp::Eq => binary_map(lhs_l, rhs_l, lhs, rhs, |x, y| u8::from(x == y)),
            CmpOp::Ne => binary_map(lhs_l, rhs_l, lhs, rhs, |x, y| u8::from(x != y)),
            CmpOp::Lt => binary_map(lhs_l, rhs_l, lhs, rhs, |x, y| u8::from(x < y)),
            CmpOp::Le => binary_map(lhs_l, rhs_l, lhs, rhs, |x, y| u8::from(x <= y)),
            CmpOp::Gt => binary_map(lhs_l, rhs_l, lhs, rhs, |x, y| u8::from(x > y)),
            CmpOp::Ge => binary_map(lhs_l, rhs_l, lhs, rhs, |x, y| u8::from(x >= y)),
        };
        Ok(dst)
    }
}

pub(super) struct WCond<'a, T: IntDType>(pub(super) &'a [T], pub(super) &'a Layout);

impl<I: IntDType> Map2 for WCond<'_, I> {
    const OP: &'static str = "where";
    #[inline(always)]
    fn f<T: WithDType>(&self, t: &[T], t_l: &Layout, f: &[T], f_l: &Layout) -> Result<Vec<T>> {
        let vs = match (
            self.1.contiguous_offsets(),
            t_l.contiguous_offsets(),
            f_l.contiguous_offsets(),
        ) {
            (Some((o1, o2)), Some((o_t1, o_t2)), Some((o_f1, o_f2))) => {
                let pred = &self.0[o1..o2];
                let t = &t[o_t1..o_t2];
                let f = &f[o_f1..o_f2];
                pred.iter()
                    .zip(t.iter().zip(f.iter()))
                    .map(|(p, (&t, &f))| if p.is_true() { t } else { f })
                    .collect::<Vec<_>>()
            }
            _ => self
                .1
                .strided_index()
                .zip(t_l.strided_index().zip(f_l.strided_index()))
                .map(|(i_p, (i_t, i_f))| {
                    if self.0[i_p].is_true() {
                        t[i_t]
                    } else {
                        f[i_f]
                    }
                })
                .collect::<Vec<_>>(),
        };
        Ok(vs)
    }
}

pub(super) struct Affine(pub(super) f64, pub(super) f64);

impl Map1 for Affine {
    fn f<T: WithDType>(&self, vs: &[T], layout: &Layout) -> Result<Vec<T>> {
        let mul = T::from_f64(self.0);
        let add = T::from_f64(self.1);
        Ok(unary_map(vs, layout, |v| v * mul + add))
    }
}

pub(super) fn elu<T: num_traits::Float>(v: T, alpha: T) -> T {
    if v.is_sign_positive() {
        v
    } else {
        (v.exp() - T::one()) * alpha
    }
}

impl BackendStorage for CpuStorage {
    type Device = CpuDevice;

    fn dtype(&self) -> DType {
        match self {
            Self::U8(_) => DType::U8,
            Self::U32(_) => DType::U32,
            Self::I64(_) => DType::I64,
            Self::BF16(_) => DType::BF16,
            Self::F16(_) => DType::F16,
            Self::F32(_) => DType::F32,
            Self::F64(_) => DType::F64,
            Self::F8E4M3(_) => DType::F8E4M3,
        }
    }

    fn to_dtype(&self, layout: &Layout, dtype: DType) -> Result<Self> {
        // TODO: find a way around the quadratic number of cases below.
        match (self, dtype) {
            (Self::U8(storage), DType::BF16) => {
                let data = unary_map(storage, layout, |v| bf16::from_f32(v as f32));
                Ok(Self::BF16(data))
            }
            (Self::U32(storage), DType::BF16) => {
                let data = unary_map(storage, layout, |v| bf16::from_f32(v as f32));
                Ok(Self::BF16(data))
            }
            (Self::I64(storage), DType::BF16) => {
                let data = unary_map(storage, layout, |v| bf16::from_f32(v as f32));
                Ok(Self::BF16(data))
            }
            (Self::BF16(storage), DType::BF16) => {
                let data = unary_map(storage, layout, |v| v);
                Ok(Self::BF16(data))
            }
            (Self::F16(storage), DType::BF16) => {
                let data = unary_map(storage, layout, |v| bf16::from_f32(v.to_f32()));
                Ok(Self::BF16(data))
            }
            (Self::F32(storage), DType::BF16) => {
                let data = unary_map(storage, layout, bf16::from_f32);
                Ok(Self::BF16(data))
            }
            (Self::F64(storage), DType::BF16) => {
                let data = unary_map(storage, layout, bf16::from_f64);
                Ok(Self::BF16(data))
            }
            (Self::F8E4M3(storage), DType::BF16) => {
                let data = unary_map(storage, layout, |v| bf16::from_f32(v.to_f32()));
                Ok(Self::BF16(data))
            }
            (Self::U8(storage), DType::F16) => {
                let data = unary_map(storage, layout, |v| f16::from_f32(v as f32));
                Ok(Self::F16(data))
            }
            (Self::U32(storage), DType::F16) => {
                let data = unary_map(storage, layout, |v| f16::from_f32(v as f32));
                Ok(Self::F16(data))
            }
            (Self::I64(storage), DType::F16) => {
                let data = unary_map(storage, layout, |v| f16::from_f32(v as f32));
                Ok(Self::F16(data))
            }
            (Self::BF16(storage), DType::F16) => {
                let data = unary_map(storage, layout, |v| f16::from_f32(v.to_f32()));
                Ok(Self::F16(data))
            }
            (Self::F16(storage), DType::F16) => {
                let data = unary_map(storage, layout, |v| v);
                Ok(Self::F16(data))
            }
            (Self::F32(storage), DType::F16) => {
                let data = unary_map(storage, layout, f16::from_f32);
                Ok(Self::F16(data))
            }
            (Self::F64(storage), DType::F16) => {
                let data = unary_map(storage, layout, f16::from_f64);
                Ok(Self::F16(data))
            }
            (Self::F8E4M3(storage), DType::F16) => {
                let data = unary_map(storage, layout, |v| f16::from_f32(v.to_f32()));
                Ok(Self::F16(data))
            }
            (Self::U8(storage), DType::F32) => {
                let data = unary_map(storage, layout, |v| v as f32);
                Ok(Self::F32(data))
            }
            (Self::U32(storage), DType::F32) => {
                let data = unary_map(storage, layout, |v| v as f32);
                Ok(Self::F32(data))
            }
            (Self::I64(storage), DType::F32) => {
                let data = unary_map(storage, layout, |v| v as f32);
                Ok(Self::F32(data))
            }
            (Self::BF16(storage), DType::F32) => {
                let data = unary_map(storage, layout, |v| v.to_f32());
                Ok(Self::F32(data))
            }
            (Self::F16(storage), DType::F32) => {
                let data = unary_map(storage, layout, |v| v.to_f32());
                Ok(Self::F32(data))
            }
            (Self::F32(storage), DType::F32) => {
                let data = unary_map(storage, layout, |v| v);
                Ok(Self::F32(data))
            }
            (Self::F64(storage), DType::F32) => {
                let data = unary_map(storage, layout, |v| v as f32);
                Ok(Self::F32(data))
            }
            (Self::F8E4M3(storage), DType::F32) => {
                let data = unary_map(storage, layout, |v| v.to_f32());
                Ok(Self::F32(data))
            }
            (Self::U8(storage), DType::U8) => {
                let data = unary_map(storage, layout, |v| v);
                Ok(Self::U8(data))
            }
            (Self::BF16(storage), DType::U8) => {
                let data = unary_map(storage, layout, |v| v.to_f32() as u8);
                Ok(Self::U8(data))
            }
            (Self::F16(storage), DType::U8) => {
                let data = unary_map(storage, layout, |v| v.to_f32() as u8);
                Ok(Self::U8(data))
            }
            (Self::F32(storage), DType::U8) => {
                let data = unary_map(storage, layout, |v| v as u8);
                Ok(Self::U8(data))
            }
            (Self::F64(storage), DType::U8) => {
                let data = unary_map(storage, layout, |v| v as u8);
                Ok(Self::U8(data))
            }
            (Self::U32(storage), DType::U8) => {
                let data = unary_map(storage, layout, |v| v as u8);
                Ok(Self::U8(data))
            }
            (Self::I64(storage), DType::U8) => {
                let data = unary_map(storage, layout, |v| v as u8);
                Ok(Self::U8(data))
            }
            (Self::F8E4M3(storage), DType::U8) => {
                let data = unary_map(storage, layout, |v| v.to_f32() as u8);
                Ok(Self::U8(data))
            }
            (Self::U8(storage), DType::U32) => {
                let data = unary_map(storage, layout, |v| v as u32);
                Ok(Self::U32(data))
            }
            (Self::U32(storage), DType::U32) => {
                let data = unary_map(storage, layout, |v| v);
                Ok(Self::U32(data))
            }
            (Self::I64(storage), DType::U32) => {
                let data = unary_map(storage, layout, |v| v as u32);
                Ok(Self::U32(data))
            }
            (Self::BF16(storage), DType::U32) => {
                let data = unary_map(storage, layout, |v| v.to_f32() as u32);
                Ok(Self::U32(data))
            }
            (Self::F16(storage), DType::U32) => {
                let data = unary_map(storage, layout, |v| v.to_f32() as u32);
                Ok(Self::U32(data))
            }
            (Self::F32(storage), DType::U32) => {
                let data = unary_map(storage, layout, |v| v as u32);
                Ok(Self::U32(data))
            }
            (Self::F64(storage), DType::U32) => {
                let data = unary_map(storage, layout, |v| v as u32);
                Ok(Self::U32(data))
            }
            (Self::F8E4M3(storage), DType::U32) => {
                let data = unary_map(storage, layout, |v| v.to_f32() as u32);
                Ok(Self::U32(data))
            }
            (Self::U8(storage), DType::I64) => {
                let data = unary_map(storage, layout, |v| v as i64);
                Ok(Self::I64(data))
            }
            (Self::U32(storage), DType::I64) => {
                let data = unary_map(storage, layout, |v| v as i64);
                Ok(Self::I64(data))
            }
            (Self::I64(storage), DType::I64) => {
                let data = unary_map(storage, layout, |v| v);
                Ok(Self::I64(data))
            }
            (Self::BF16(storage), DType::I64) => {
                let data = unary_map(storage, layout, |v| v.to_f32() as i64);
                Ok(Self::I64(data))
            }
            (Self::F16(storage), DType::I64) => {
                let data = unary_map(storage, layout, |v| v.to_f32() as i64);
                Ok(Self::I64(data))
            }
            (Self::F32(storage), DType::I64) => {
                let data = unary_map(storage, layout, |v| v as i64);
                Ok(Self::I64(data))
            }
            (Self::F64(storage), DType::I64) => {
                let data = unary_map(storage, layout, |v| v as i64);
                Ok(Self::I64(data))
            }
            (Self::F8E4M3(storage), DType::I64) => {
                let data = unary_map(storage, layout, |v| v.to_f32() as i64);
                Ok(Self::I64(data))
            }
            (Self::U8(storage), DType::F64) => {
                let data = unary_map(storage, layout, |v| v as f64);
                Ok(Self::F64(data))
            }
            (Self::U32(storage), DType::F64) => {
                let data = unary_map(storage, layout, |v| v as f64);
                Ok(Self::F64(data))
            }
            (Self::I64(storage), DType::F64) => {
                let data = unary_map(storage, layout, |v| v as f64);
                Ok(Self::F64(data))
            }
            (Self::BF16(storage), DType::F64) => {
                let data = unary_map(storage, layout, |v| v.to_f64());
                Ok(Self::F64(data))
            }
            (Self::F16(storage), DType::F64) => {
                let data = unary_map(storage, layout, |v| v.to_f64());
                Ok(Self::F64(data))
            }
            (Self::F32(storage), DType::F64) => {
                let data = unary_map(storage, layout, |v| v as f64);
                Ok(Self::F64(data))
            }
            (Self::F64(storage), DType::F64) => {
                let data = unary_map(storage, layout, |v| v);
                Ok(Self::F64(data))
            }
            (Self::F8E4M3(storage), DType::F64) => {
                // Through f32, as every other arm here: `to_f64` on this type
                // never returned (found by the all-pairs cast test).
                let data = unary_map(storage, layout, |v| v.to_f32() as f64);
                Ok(Self::F64(data))
            }
            (Self::U8(storage), DType::F8E4M3) => {
                let data = unary_map(storage, layout, |v| F8E4M3::from_f32(v as f32));
                Ok(Self::F8E4M3(data))
            }
            (Self::U32(storage), DType::F8E4M3) => {
                let data = unary_map(storage, layout, |v| F8E4M3::from_f32(v as f32));
                Ok(Self::F8E4M3(data))
            }
            (Self::I64(storage), DType::F8E4M3) => {
                let data = unary_map(storage, layout, |v| F8E4M3::from_f32(v as f32));
                Ok(Self::F8E4M3(data))
            }
            (Self::BF16(storage), DType::F8E4M3) => {
                let data = unary_map(storage, layout, |v| F8E4M3::from(v.to_f32()));
                Ok(Self::F8E4M3(data))
            }
            (Self::F16(storage), DType::F8E4M3) => {
                let data = unary_map(storage, layout, |v| F8E4M3::from_f32(v.to_f32()));
                Ok(Self::F8E4M3(data))
            }
            (Self::F32(storage), DType::F8E4M3) => {
                let data = unary_map(storage, layout, F8E4M3::from_f32);
                Ok(Self::F8E4M3(data))
            }
            (Self::F64(storage), DType::F8E4M3) => {
                let data = unary_map(storage, layout, F8E4M3::from_f64);
                Ok(Self::F8E4M3(data))
            }
            (Self::F8E4M3(storage), DType::F8E4M3) => {
                let data = unary_map(storage, layout, |v| v);
                Ok(Self::F8E4M3(data))
            }
        }
    }

    fn reduce_op(&self, op: ReduceOp, layout: &Layout, reduce_dims: &[usize]) -> Result<Self> {
        match op {
            ReduceOp::Sum => {
                let src_dims = layout.dims();
                let mut dst_dims = src_dims.to_vec();
                for &dim in reduce_dims.iter() {
                    dst_dims[dim] = 1;
                }
                let dst_shape = Shape::from(dst_dims);
                let mut reduce_dims = reduce_dims.to_vec();
                // Sort the reduce_dims as they have to be processed from left to right when converting the
                // indexes.
                reduce_dims.sort();
                let reduce_dims_and_stride: Vec<_> = reduce_dims
                    .iter()
                    .map(|&d| (src_dims[d], src_dims[d + 1..].iter().product::<usize>()))
                    .collect();
                ReduceSum {
                    dst_shape: &dst_shape,
                    reduce_dims: &reduce_dims,
                    reduce_dims_and_stride,
                }
                .map(self, layout)
            }
            ReduceOp::Min | ReduceOp::ArgMin | ReduceOp::Max | ReduceOp::ArgMax => {
                let reduce_dim_index = match reduce_dims {
                    [reduce_dim_index] => *reduce_dim_index,
                    _ => {
                        let op = match op {
                            ReduceOp::Min => "min",
                            ReduceOp::ArgMin => "argmin",
                            ReduceOp::Max => "max",
                            ReduceOp::ArgMax => "argmax",
                            _ => unreachable!(),
                        };
                        let dims = reduce_dims.to_vec();
                        Err(Error::OnlySingleDimension { op, dims })?
                    }
                };
                let (use_min, return_index) = match op {
                    ReduceOp::Min => (true, false),
                    ReduceOp::ArgMin => (true, true),
                    ReduceOp::Max => (false, false),
                    ReduceOp::ArgMax => (false, true),
                    _ => unreachable!(),
                };
                ReduceIndex {
                    reduce_dim_index,
                    use_min,
                    return_index,
                }
                .map(self, layout)
            }
        }
    }

    fn cmp(&self, op: CmpOp, rhs: &Self, lhs_l: &Layout, rhs_l: &Layout) -> Result<Self> {
        Cmp(op).map(self, lhs_l, rhs, rhs_l)
    }

    fn affine(&self, layout: &Layout, mul: f64, add: f64) -> Result<Self> {
        Affine(mul, add).map(self, layout)
    }

    fn avg_pool2d(
        &self,
        layout: &Layout,
        kernel_size: (usize, usize),
        stride: (usize, usize),
    ) -> Result<Self> {
        AvgPool2D(kernel_size, stride).map(self, layout)
    }

    fn max_pool2d(
        &self,
        layout: &Layout,
        kernel_size: (usize, usize),
        stride: (usize, usize),
    ) -> Result<Self> {
        MaxPool2D(kernel_size, stride).map(self, layout)
    }

    fn upsample_nearest1d(&self, layout: &Layout, sz: usize) -> Result<Self> {
        UpsampleNearest1D(sz).map(self, layout)
    }

    fn upsample_nearest2d(&self, layout: &Layout, h: usize, w: usize) -> Result<Self> {
        UpsampleNearest2D(h, w).map(self, layout)
    }

    fn powf(&self, layout: &Layout, e: f64) -> Result<Self> {
        use num_traits::Float;
        // TODO: Have some generic map for functions that apply on num_traits::Float elements.
        match self {
            Self::BF16(storage) => {
                let data = unary_map(storage, layout, |v| v.powf(bf16::from_f64(e)));
                Ok(Self::BF16(data))
            }
            Self::F16(storage) => {
                let data = unary_map(storage, layout, |v| v.powf(f16::from_f64(e)));
                Ok(Self::F16(data))
            }
            Self::F32(storage) => {
                let data = unary_map(storage, layout, |v| v.powf(e as f32));
                Ok(Self::F32(data))
            }
            Self::F64(storage) => {
                let data = unary_map(storage, layout, |v| v.powf(e));
                Ok(Self::F64(data))
            }
            Self::F8E4M3(storage) => {
                let data = unary_map(storage, layout, |v| v.powf(F8E4M3::from_f64(e)));
                Ok(Self::F8E4M3(data))
            }
            Self::U8(_) => Err(Error::UnsupportedDTypeForOp(DType::U8, "elu").bt()),
            Self::U32(_) => Err(Error::UnsupportedDTypeForOp(DType::U32, "elu").bt()),
            Self::I64(_) => Err(Error::UnsupportedDTypeForOp(DType::I64, "elu").bt()),
        }
    }

    fn elu(&self, layout: &Layout, alpha: f64) -> Result<Self> {
        // TODO: Have some generic map for functions that apply on num_traits::Float elements.
        match self {
            Self::BF16(storage) => {
                let data = unary_map(storage, layout, |v| elu(v, bf16::from_f64(alpha)));
                Ok(Self::BF16(data))
            }
            Self::F16(storage) => {
                let data = unary_map(storage, layout, |v| elu(v, f16::from_f64(alpha)));
                Ok(Self::F16(data))
            }
            Self::F32(storage) => {
                let data = unary_map(storage, layout, |v| elu(v, f32::from_f64(alpha)));
                Ok(Self::F32(data))
            }
            Self::F64(storage) => {
                let data = unary_map(storage, layout, |v| elu(v, alpha));
                Ok(Self::F64(data))
            }
            Self::F8E4M3(storage) => {
                let data = unary_map(storage, layout, |v| elu(v, F8E4M3::from_f64(alpha)));
                Ok(Self::F8E4M3(data))
            }
            Self::U8(_) => Err(Error::UnsupportedDTypeForOp(DType::U8, "elu").bt()),
            Self::U32(_) => Err(Error::UnsupportedDTypeForOp(DType::U32, "elu").bt()),
            Self::I64(_) => Err(Error::UnsupportedDTypeForOp(DType::I64, "elu").bt()),
        }
    }

    fn sub_at_indices(&self, layout: &Layout, indices: &[u32], value: f32) -> Result<Self> {
        // Delegate to the inherent method
        self.sub_at_indices(layout, indices, value)
    }

    fn div_at_indices(&self, layout: &Layout, indices: &[u32], value: f32) -> Result<Self> {
        // Delegate to the inherent method
        self.div_at_indices(layout, indices, value)
    }

    fn unary_impl<B: UnaryOpT>(&self, layout: &Layout) -> Result<Self> {
        match self {
            Self::BF16(storage) => {
                if B::BF16_VEC {
                    let data = unary_map_vec(storage, layout, B::bf16, B::bf16_vec);
                    Ok(Self::BF16(data))
                } else {
                    let data = unary_map(storage, layout, B::bf16);
                    Ok(Self::BF16(data))
                }
            }
            Self::F16(storage) => {
                if B::F16_VEC {
                    let data = unary_map_vec(storage, layout, B::f16, B::f16_vec);
                    Ok(Self::F16(data))
                } else {
                    let data = unary_map(storage, layout, B::f16);
                    Ok(Self::F16(data))
                }
            }
            Self::F32(storage) => {
                if B::F32_VEC {
                    let data = unary_map_vec(storage, layout, B::f32, B::f32_vec);
                    Ok(Self::F32(data))
                } else {
                    let data = unary_map(storage, layout, B::f32);
                    Ok(Self::F32(data))
                }
            }
            Self::F64(storage) => {
                if B::F64_VEC {
                    let data = unary_map_vec(storage, layout, B::f64, B::f64_vec);
                    Ok(Self::F64(data))
                } else {
                    let data = unary_map(storage, layout, B::f64);
                    Ok(Self::F64(data))
                }
            }
            Self::F8E4M3(storage) => {
                if B::F8E4M3_VEC {
                    let data = unary_map_vec(storage, layout, B::f8e4m3, B::f8e4m3_vec);
                    Ok(Self::F8E4M3(data))
                } else {
                    let data = unary_map(storage, layout, B::f8e4m3);
                    Ok(Self::F8E4M3(data))
                }
            }
            Self::U8(storage) => {
                let data = unary_map(storage, layout, B::u8);
                Ok(Self::U8(data))
            }
            Self::U32(storage) => {
                let data = unary_map(storage, layout, B::u32);
                Ok(Self::U32(data))
            }
            Self::I64(storage) => {
                let data = unary_map(storage, layout, B::i64);
                Ok(Self::I64(data))
            }
        }
    }

    fn binary_impl<B: BinaryOpT>(
        &self,
        rhs: &Self,
        lhs_l: &Layout,
        rhs_l: &Layout,
    ) -> Result<Self> {
        match (self, rhs) {
            (Self::BF16(lhs), Self::BF16(rhs)) => {
                let data = if B::BF16_VEC {
                    binary_map_vec(lhs_l, rhs_l, lhs, rhs, B::bf16, B::bf16_vec)
                } else {
                    binary_map(lhs_l, rhs_l, lhs, rhs, B::bf16)
                };
                Ok(Self::BF16(data))
            }
            (Self::F16(lhs), Self::F16(rhs)) => {
                let data = if B::F16_VEC {
                    binary_map_vec(lhs_l, rhs_l, lhs, rhs, B::f16, B::f16_vec)
                } else {
                    binary_map(lhs_l, rhs_l, lhs, rhs, B::f16)
                };
                Ok(Self::F16(data))
            }
            (Self::F32(lhs), Self::F32(rhs)) => {
                let data = if B::F32_VEC {
                    binary_map_vec(lhs_l, rhs_l, lhs, rhs, B::f32, B::f32_vec)
                } else {
                    binary_map(lhs_l, rhs_l, lhs, rhs, B::f32)
                };
                Ok(Self::F32(data))
            }
            (Self::F64(lhs), Self::F64(rhs)) => {
                let data = if B::F64_VEC {
                    binary_map_vec(lhs_l, rhs_l, lhs, rhs, B::f64, B::f64_vec)
                } else {
                    binary_map(lhs_l, rhs_l, lhs, rhs, B::f64)
                };
                Ok(Self::F64(data))
            }
            (Self::U32(lhs), Self::U32(rhs)) => {
                let data = if B::U32_VEC {
                    binary_map_vec(lhs_l, rhs_l, lhs, rhs, B::u32, B::u32_vec)
                } else {
                    binary_map(lhs_l, rhs_l, lhs, rhs, B::u32)
                };
                Ok(Self::U32(data))
            }
            (Self::I64(lhs), Self::I64(rhs)) => {
                let data = if B::I64_VEC {
                    binary_map_vec(lhs_l, rhs_l, lhs, rhs, B::i64, B::i64_vec)
                } else {
                    binary_map(lhs_l, rhs_l, lhs, rhs, B::i64)
                };
                Ok(Self::I64(data))
            }
            (Self::U8(lhs), Self::U8(rhs)) => {
                let data = if B::U8_VEC {
                    binary_map_vec(lhs_l, rhs_l, lhs, rhs, B::u8, B::u8_vec)
                } else {
                    binary_map(lhs_l, rhs_l, lhs, rhs, B::u8)
                };
                Ok(Self::U8(data))
            }
            _ => {
                // This should be covered by the dtype check above.
                Err(Error::DTypeMismatchBinaryOp {
                    lhs: self.dtype(),
                    rhs: rhs.dtype(),
                    op: B::NAME,
                }
                .bt())
            }
        }
    }

    fn binary_inplace_impl(
        &mut self,
        op: crate::op::BinaryInplaceOp,
        rhs: &Self,
        lhs_l: &Layout,
        rhs_l: &Layout,
    ) -> Result<()> {
        use crate::op::BinaryInplaceOp;

        // lhs must be contiguous for in-place operations
        if !lhs_l.is_contiguous() {
            return Err(Error::Msg(
                "in-place binary op requires contiguous lhs tensor".to_string(),
            )
            .bt());
        }

        // Save dtypes for error message before consuming self
        let lhs_dtype = self.dtype();
        let rhs_dtype = rhs.dtype();

        // Check for min/max on integer types (not supported for consistency with CUDA)
        let is_integer = matches!(lhs_dtype, DType::U8 | DType::U32 | DType::I64);
        if is_integer && matches!(op, BinaryInplaceOp::Min | BinaryInplaceOp::Max) {
            return Err(Error::Msg(format!(
                "min/max in-place ops not supported for integer dtype {:?}",
                lhs_dtype
            ))
            .bt());
        }

        // Helper macro to apply operation element-wise
        macro_rules! apply_inplace {
            ($lhs:expr, $rhs_slice:expr, $lhs_l:expr, $rhs_l:expr, $op_fn:expr) => {{
                let lhs_start = $lhs_l.start_offset();
                let rhs_start = $rhs_l.start_offset();
                let elem_count = $lhs_l.shape().elem_count();

                if $rhs_l.is_contiguous() {
                    // Fast path: both contiguous
                    for i in 0..elem_count {
                        $lhs[lhs_start + i] =
                            $op_fn($lhs[lhs_start + i], $rhs_slice[rhs_start + i]);
                    }
                } else {
                    // Slow path: rhs is strided
                    let dims = $lhs_l.dims();
                    let rhs_strides = $rhs_l.stride();
                    for i in 0..elem_count {
                        // Calculate strided index for rhs
                        let mut rhs_idx = rhs_start;
                        let mut tmp = i;
                        for d in (0..dims.len()).rev() {
                            let coord = tmp % dims[d];
                            tmp /= dims[d];
                            rhs_idx += coord * rhs_strides[d];
                        }
                        $lhs[lhs_start + i] = $op_fn($lhs[lhs_start + i], $rhs_slice[rhs_idx]);
                    }
                }
            }};
        }

        match (self, rhs, op) {
            // F32
            (Self::F32(lhs), Self::F32(rhs_data), BinaryInplaceOp::Add) => {
                apply_inplace!(lhs, rhs_data, lhs_l, rhs_l, |a, b| a + b);
            }
            (Self::F32(lhs), Self::F32(rhs_data), BinaryInplaceOp::Sub) => {
                apply_inplace!(lhs, rhs_data, lhs_l, rhs_l, |a, b| a - b);
            }
            (Self::F32(lhs), Self::F32(rhs_data), BinaryInplaceOp::Mul) => {
                apply_inplace!(lhs, rhs_data, lhs_l, rhs_l, |a, b| a * b);
            }
            (Self::F32(lhs), Self::F32(rhs_data), BinaryInplaceOp::Div) => {
                apply_inplace!(lhs, rhs_data, lhs_l, rhs_l, |a, b| a / b);
            }
            (Self::F32(lhs), Self::F32(rhs_data), BinaryInplaceOp::Min) => {
                apply_inplace!(lhs, rhs_data, lhs_l, rhs_l, |a: f32, b| a.min(b));
            }
            (Self::F32(lhs), Self::F32(rhs_data), BinaryInplaceOp::Max) => {
                apply_inplace!(lhs, rhs_data, lhs_l, rhs_l, |a: f32, b| a.max(b));
            }
            // F64
            (Self::F64(lhs), Self::F64(rhs_data), BinaryInplaceOp::Add) => {
                apply_inplace!(lhs, rhs_data, lhs_l, rhs_l, |a, b| a + b);
            }
            (Self::F64(lhs), Self::F64(rhs_data), BinaryInplaceOp::Sub) => {
                apply_inplace!(lhs, rhs_data, lhs_l, rhs_l, |a, b| a - b);
            }
            (Self::F64(lhs), Self::F64(rhs_data), BinaryInplaceOp::Mul) => {
                apply_inplace!(lhs, rhs_data, lhs_l, rhs_l, |a, b| a * b);
            }
            (Self::F64(lhs), Self::F64(rhs_data), BinaryInplaceOp::Div) => {
                apply_inplace!(lhs, rhs_data, lhs_l, rhs_l, |a, b| a / b);
            }
            (Self::F64(lhs), Self::F64(rhs_data), BinaryInplaceOp::Min) => {
                apply_inplace!(lhs, rhs_data, lhs_l, rhs_l, |a: f64, b| a.min(b));
            }
            (Self::F64(lhs), Self::F64(rhs_data), BinaryInplaceOp::Max) => {
                apply_inplace!(lhs, rhs_data, lhs_l, rhs_l, |a: f64, b| a.max(b));
            }
            // F16
            (Self::F16(lhs), Self::F16(rhs_data), BinaryInplaceOp::Add) => {
                apply_inplace!(lhs, rhs_data, lhs_l, rhs_l, |a, b| a + b);
            }
            (Self::F16(lhs), Self::F16(rhs_data), BinaryInplaceOp::Sub) => {
                apply_inplace!(lhs, rhs_data, lhs_l, rhs_l, |a, b| a - b);
            }
            (Self::F16(lhs), Self::F16(rhs_data), BinaryInplaceOp::Mul) => {
                apply_inplace!(lhs, rhs_data, lhs_l, rhs_l, |a, b| a * b);
            }
            (Self::F16(lhs), Self::F16(rhs_data), BinaryInplaceOp::Div) => {
                apply_inplace!(lhs, rhs_data, lhs_l, rhs_l, |a, b| a / b);
            }
            (Self::F16(lhs), Self::F16(rhs_data), BinaryInplaceOp::Min) => {
                apply_inplace!(lhs, rhs_data, lhs_l, rhs_l, |a: half::f16, b| a.min(b));
            }
            (Self::F16(lhs), Self::F16(rhs_data), BinaryInplaceOp::Max) => {
                apply_inplace!(lhs, rhs_data, lhs_l, rhs_l, |a: half::f16, b| a.max(b));
            }
            // BF16
            (Self::BF16(lhs), Self::BF16(rhs_data), BinaryInplaceOp::Add) => {
                apply_inplace!(lhs, rhs_data, lhs_l, rhs_l, |a, b| a + b);
            }
            (Self::BF16(lhs), Self::BF16(rhs_data), BinaryInplaceOp::Sub) => {
                apply_inplace!(lhs, rhs_data, lhs_l, rhs_l, |a, b| a - b);
            }
            (Self::BF16(lhs), Self::BF16(rhs_data), BinaryInplaceOp::Mul) => {
                apply_inplace!(lhs, rhs_data, lhs_l, rhs_l, |a, b| a * b);
            }
            (Self::BF16(lhs), Self::BF16(rhs_data), BinaryInplaceOp::Div) => {
                apply_inplace!(lhs, rhs_data, lhs_l, rhs_l, |a, b| a / b);
            }
            (Self::BF16(lhs), Self::BF16(rhs_data), BinaryInplaceOp::Min) => {
                apply_inplace!(lhs, rhs_data, lhs_l, rhs_l, |a: half::bf16, b| a.min(b));
            }
            (Self::BF16(lhs), Self::BF16(rhs_data), BinaryInplaceOp::Max) => {
                apply_inplace!(lhs, rhs_data, lhs_l, rhs_l, |a: half::bf16, b| a.max(b));
            }
            // F8E4M3
            (Self::F8E4M3(lhs), Self::F8E4M3(rhs_data), BinaryInplaceOp::Add) => {
                apply_inplace!(lhs, rhs_data, lhs_l, rhs_l, |a, b| a + b);
            }
            (Self::F8E4M3(lhs), Self::F8E4M3(rhs_data), BinaryInplaceOp::Sub) => {
                apply_inplace!(lhs, rhs_data, lhs_l, rhs_l, |a, b| a - b);
            }
            (Self::F8E4M3(lhs), Self::F8E4M3(rhs_data), BinaryInplaceOp::Mul) => {
                apply_inplace!(lhs, rhs_data, lhs_l, rhs_l, |a, b| a * b);
            }
            (Self::F8E4M3(lhs), Self::F8E4M3(rhs_data), BinaryInplaceOp::Div) => {
                apply_inplace!(lhs, rhs_data, lhs_l, rhs_l, |a, b| a / b);
            }
            (Self::F8E4M3(lhs), Self::F8E4M3(rhs_data), BinaryInplaceOp::Min) => {
                apply_inplace!(lhs, rhs_data, lhs_l, rhs_l, |a: float8::F8E4M3, b| a.min(b));
            }
            (Self::F8E4M3(lhs), Self::F8E4M3(rhs_data), BinaryInplaceOp::Max) => {
                apply_inplace!(lhs, rhs_data, lhs_l, rhs_l, |a: float8::F8E4M3, b| a.max(b));
            }
            // U8 (no min/max)
            (Self::U8(lhs), Self::U8(rhs_data), BinaryInplaceOp::Add) => {
                apply_inplace!(lhs, rhs_data, lhs_l, rhs_l, |a: u8, b| a.wrapping_add(b));
            }
            (Self::U8(lhs), Self::U8(rhs_data), BinaryInplaceOp::Sub) => {
                apply_inplace!(lhs, rhs_data, lhs_l, rhs_l, |a: u8, b| a.wrapping_sub(b));
            }
            (Self::U8(lhs), Self::U8(rhs_data), BinaryInplaceOp::Mul) => {
                apply_inplace!(lhs, rhs_data, lhs_l, rhs_l, |a: u8, b| a.wrapping_mul(b));
            }
            (Self::U8(lhs), Self::U8(rhs_data), BinaryInplaceOp::Div) => {
                apply_inplace!(lhs, rhs_data, lhs_l, rhs_l, |a: u8, b| a / b);
            }
            // U32 (no min/max)
            (Self::U32(lhs), Self::U32(rhs_data), BinaryInplaceOp::Add) => {
                apply_inplace!(lhs, rhs_data, lhs_l, rhs_l, |a: u32, b| a.wrapping_add(b));
            }
            (Self::U32(lhs), Self::U32(rhs_data), BinaryInplaceOp::Sub) => {
                apply_inplace!(lhs, rhs_data, lhs_l, rhs_l, |a: u32, b| a.wrapping_sub(b));
            }
            (Self::U32(lhs), Self::U32(rhs_data), BinaryInplaceOp::Mul) => {
                apply_inplace!(lhs, rhs_data, lhs_l, rhs_l, |a: u32, b| a.wrapping_mul(b));
            }
            (Self::U32(lhs), Self::U32(rhs_data), BinaryInplaceOp::Div) => {
                apply_inplace!(lhs, rhs_data, lhs_l, rhs_l, |a: u32, b| a / b);
            }
            // I64 (no min/max)
            (Self::I64(lhs), Self::I64(rhs_data), BinaryInplaceOp::Add) => {
                apply_inplace!(lhs, rhs_data, lhs_l, rhs_l, |a: i64, b| a.wrapping_add(b));
            }
            (Self::I64(lhs), Self::I64(rhs_data), BinaryInplaceOp::Sub) => {
                apply_inplace!(lhs, rhs_data, lhs_l, rhs_l, |a: i64, b| a.wrapping_sub(b));
            }
            (Self::I64(lhs), Self::I64(rhs_data), BinaryInplaceOp::Mul) => {
                apply_inplace!(lhs, rhs_data, lhs_l, rhs_l, |a: i64, b| a.wrapping_mul(b));
            }
            (Self::I64(lhs), Self::I64(rhs_data), BinaryInplaceOp::Div) => {
                apply_inplace!(lhs, rhs_data, lhs_l, rhs_l, |a: i64, b| a / b);
            }
            _ => {
                return Err(Error::DTypeMismatchBinaryOp {
                    lhs: lhs_dtype,
                    rhs: rhs_dtype,
                    op: "binary_inplace",
                }
                .bt())
            }
        }
        Ok(())
    }

    fn copy2d(
        &self,
        dst: &mut Self,
        d1: usize,
        d2: usize,
        src_s: usize,
        dst_s: usize,
        src_o: usize,
        dst_o: usize,
    ) -> Result<()> {
        match (self, dst) {
            (Self::U8(src), Self::U8(dst)) => copy2d_(src, dst, d1, d2, src_s, dst_s, src_o, dst_o),
            (Self::U32(src), Self::U32(dst)) => {
                copy2d_(src, dst, d1, d2, src_s, dst_s, src_o, dst_o)
            }
            (Self::I64(src), Self::I64(dst)) => {
                copy2d_(src, dst, d1, d2, src_s, dst_s, src_o, dst_o)
            }
            (Self::BF16(src), Self::BF16(dst)) => {
                copy2d_(src, dst, d1, d2, src_s, dst_s, src_o, dst_o)
            }
            (Self::F16(src), Self::F16(dst)) => {
                copy2d_(src, dst, d1, d2, src_s, dst_s, src_o, dst_o)
            }
            (Self::F32(src), Self::F32(dst)) => {
                copy2d_(src, dst, d1, d2, src_s, dst_s, src_o, dst_o)
            }
            (Self::F64(src), Self::F64(dst)) => {
                copy2d_(src, dst, d1, d2, src_s, dst_s, src_o, dst_o)
            }
            (Self::F8E4M3(src), Self::F8E4M3(dst)) => {
                copy2d_(src, dst, d1, d2, src_s, dst_s, src_o, dst_o)
            }
            (_, dst) => {
                return Err(Error::DTypeMismatchBinaryOp {
                    lhs: self.dtype(),
                    rhs: dst.dtype(),
                    op: "copy2d",
                }
                .bt());
            }
        }
        Ok(())
    }

    fn copy_strided_src(&self, dst: &mut Self, dst_offset: usize, src_l: &Layout) -> Result<()> {
        match (self, dst) {
            (Self::U8(src), Self::U8(dst)) => copy_strided_src_(src, dst, dst_offset, src_l),
            (Self::U32(src), Self::U32(dst)) => copy_strided_src_(src, dst, dst_offset, src_l),
            (Self::I64(src), Self::I64(dst)) => copy_strided_src_(src, dst, dst_offset, src_l),
            (Self::BF16(src), Self::BF16(dst)) => copy_strided_src_(src, dst, dst_offset, src_l),
            (Self::F16(src), Self::F16(dst)) => copy_strided_src_(src, dst, dst_offset, src_l),
            (Self::F32(src), Self::F32(dst)) => copy_strided_src_(src, dst, dst_offset, src_l),
            (Self::F64(src), Self::F64(dst)) => copy_strided_src_(src, dst, dst_offset, src_l),
            (_, dst) => {
                // This should be covered by the dtype check above.
                return Err(Error::DTypeMismatchBinaryOp {
                    lhs: self.dtype(),
                    rhs: dst.dtype(),
                    op: "copy_strided",
                }
                .bt());
            }
        }
        Ok(())
    }

    fn where_cond(
        &self,
        layout: &Layout,
        t: &Self,
        t_l: &Layout,
        f: &Self,
        f_l: &Layout,
    ) -> Result<Self> {
        match self {
            Self::U8(pred) => WCond(pred, layout).map(t, t_l, f, f_l),
            Self::U32(pred) => WCond(pred, layout).map(t, t_l, f, f_l),
            Self::I64(pred) => WCond(pred, layout).map(t, t_l, f, f_l),
            _ => Err(Error::UnsupportedDTypeForOp(self.dtype(), "where-cond")),
        }
    }

    fn conv1d(
        &self,
        l: &Layout,
        kernel: &Self,
        kernel_l: &Layout,
        params: &crate::conv::ParamsConv1D,
    ) -> Result<Self> {
        if !USE_IM2COL_CONV1D {
            return Conv1D(params).map(self, l, kernel, kernel_l);
        }
        let op = Im2Col1D {
            l_k: params.k_size,
            padding: params.padding,
            stride: params.stride,
            dilation: params.dilation,
        };
        let col = op.map(self, l)?;
        let b = params.b_size;
        let n = params.c_out;
        let l_out = params.l_out();
        let k = op.l_k * params.c_in;
        let m = l_out;
        let col_l = Layout::contiguous((b, m, k));
        let res = if kernel_l.is_contiguous() {
            let kernel_l = Layout::contiguous_with_offset((1, n, k), kernel_l.start_offset())
                .transpose(1, 2)?
                .broadcast_as((b, k, n))?;
            col.matmul(kernel, (b, m, n, k), &col_l, &kernel_l)?
        } else {
            // Make the kernel contiguous if not already the case.
            let mut kernel_c = unsafe {
                self.device()
                    .alloc_uninit(kernel_l.shape(), kernel.dtype())?
            };
            kernel.copy_strided_src(&mut kernel_c, 0, kernel_l)?;
            let kernel_l = Layout::contiguous((1, n, k))
                .transpose(1, 2)?
                .broadcast_as((b, k, n))?;
            col.matmul(&kernel_c, (b, m, n, k), &col_l, &kernel_l)?
        };
        let res_l = Layout::contiguous((b, l_out, params.c_out)).transpose(1, 2)?;
        let mut res_t = unsafe { self.device().alloc_uninit(res_l.shape(), res.dtype())? };
        res.copy_strided_src(&mut res_t, 0, &res_l)?;
        Ok(res_t)
    }

    fn conv_transpose1d(
        &self,
        l: &Layout,
        kernel: &Self,
        kernel_l: &Layout,
        params: &crate::conv::ParamsConvTranspose1D,
    ) -> Result<Self> {
        let can_use_col2im = kernel_l.is_contiguous()
            && params.dilation == 1
            && params.padding == 0
            && params.output_padding == 0;
        if USE_COL2IM_CONV1D_TR && can_use_col2im {
            let (b_size, c_in, l_in) = l.shape().dims3()?;
            let (c_in2, c_out, k_size) = kernel_l.shape().dims3()?;
            if !kernel_l.is_contiguous() {
                crate::bail!(
                    "convtr1d: the second argument (kernel) has to be contiguous {kernel_l:?}"
                )
            }
            if c_in != c_in2 {
                crate::bail!(
                    "convtr1d: shape mismatch on c_in {:?} {:?}",
                    l.shape(),
                    kernel_l.shape()
                )
            }
            let col = {
                // This merges the last two dimensions of the kernel together.
                let kernel_l_mm = Layout::new(
                    (b_size, c_in, k_size * c_out).into(),
                    vec![0, k_size * c_out, 1],
                    kernel_l.start_offset(),
                );
                self.matmul(
                    kernel,
                    (
                        b_size,
                        /* m */ l_in,
                        /* n */ c_out * k_size,
                        /* k */ c_in,
                    ),
                    &l.transpose(1, 2)?,
                    &kernel_l_mm,
                )?
            };
            let col_l = Layout::contiguous((b_size, l_in, c_out, k_size));
            Col2Im1D {
                stride: params.stride,
            }
            .map(&col, &col_l)
        } else {
            ConvTranspose1D(params).map(self, l, kernel, kernel_l)
        }
    }

    fn conv2d(
        &self,
        l: &Layout,
        kernel: &Self,
        kernel_l: &Layout,
        params: &crate::conv::ParamsConv2D,
    ) -> Result<Self> {
        if !USE_IM2COL_CONV2D {
            return Conv2D(params).map(self, l, kernel, kernel_l);
        }
        let op = Im2Col {
            h_k: params.k_h,
            w_k: params.k_w,
            padding: params.padding,
            stride: params.stride,
            dilation: params.dilation,
        };
        let col = op.map(self, l)?;
        let b = params.b_size;
        let n = params.c_out;
        let (h_out, w_out) = (params.out_h(), params.out_w());
        let k = op.h_k * op.w_k * params.c_in;
        let m = h_out * w_out;
        let col_l = Layout::contiguous((b, m, k));
        let res = if kernel_l.is_contiguous() {
            let kernel_l = Layout::contiguous_with_offset((1, n, k), kernel_l.start_offset())
                .transpose(1, 2)?
                .broadcast_as((b, k, n))?;
            col.matmul(kernel, (b, m, n, k), &col_l, &kernel_l)?
        } else {
            // Make the kernel contiguous if not already the case.
            let mut kernel_c = unsafe {
                self.device()
                    .alloc_uninit(kernel_l.shape(), kernel.dtype())?
            };
            kernel.copy_strided_src(&mut kernel_c, 0, kernel_l)?;
            let kernel_l = Layout::contiguous((1, n, k))
                .transpose(1, 2)?
                .broadcast_as((b, k, n))?;
            col.matmul(&kernel_c, (b, m, n, k), &col_l, &kernel_l)?
        };
        let res_l = Layout::contiguous((b, h_out, w_out, params.c_out))
            .transpose(1, 2)?
            .transpose(1, 3)?;
        let mut res_t = unsafe { self.device().alloc_uninit(res_l.shape(), res.dtype())? };
        res.copy_strided_src(&mut res_t, 0, &res_l)?;
        Ok(res_t)
    }

    fn conv_transpose2d(
        &self,
        l: &Layout,
        kernel: &Self,
        kernel_l: &Layout,
        params: &crate::conv::ParamsConvTranspose2D,
    ) -> Result<Self> {
        ConvTranspose2D(params).map(self, l, kernel, kernel_l)
    }

    fn index_select(&self, ids: &Self, l: &Layout, ids_l: &Layout, dim: usize) -> Result<Self> {
        match ids {
            Self::U8(ids) => IndexSelect { ids, ids_l, dim }.map(self, l),
            Self::U32(ids) => IndexSelect { ids, ids_l, dim }.map(self, l),
            Self::I64(ids) => IndexSelect { ids, ids_l, dim }.map(self, l),
            _ => Err(Error::UnsupportedDTypeForOp(self.dtype(), "index-select").bt()),
        }
    }

    fn gather(&self, l: &Layout, ids: &Self, ids_l: &Layout, dim: usize) -> Result<Self> {
        match ids {
            Self::U8(ids) => Gather { ids, ids_l, dim }.map(self, l),
            Self::U32(ids) => Gather { ids, ids_l, dim }.map(self, l),
            Self::I64(ids) => Gather { ids, ids_l, dim }.map(self, l),
            _ => Err(Error::UnsupportedDTypeForOp(self.dtype(), "gather").bt()),
        }
    }

    fn scatter_set(
        &mut self,
        l: &Layout,
        ids: &Self,
        ids_l: &Layout,
        src: &Self,
        src_l: &Layout,
        dim: usize,
    ) -> Result<()> {
        match ids {
            Self::U8(ids) => Scatter::<_, Set>::new(ids, ids_l, dim).map(self, l, src, src_l),
            Self::U32(ids) => Scatter::<_, Set>::new(ids, ids_l, dim).map(self, l, src, src_l),
            Self::I64(ids) => Scatter::<_, Set>::new(ids, ids_l, dim).map(self, l, src, src_l),
            _ => Err(Error::UnsupportedDTypeForOp(self.dtype(), "scatter").bt()),
        }
    }

    fn scatter_add_set(
        &mut self,
        l: &Layout,
        ids: &Self,
        ids_l: &Layout,
        src: &Self,
        src_l: &Layout,
        dim: usize,
    ) -> Result<()> {
        match ids {
            Self::U8(ids) => Scatter::<_, Add>::new(ids, ids_l, dim).map(self, l, src, src_l),
            Self::U32(ids) => Scatter::<_, Add>::new(ids, ids_l, dim).map(self, l, src, src_l),
            Self::I64(ids) => Scatter::<_, Add>::new(ids, ids_l, dim).map(self, l, src, src_l),
            _ => Err(Error::UnsupportedDTypeForOp(self.dtype(), "scatter-add").bt()),
        }
    }

    fn index_add(
        &self,
        l: &Layout,
        ids: &Self,
        ids_l: &Layout,
        src: &Self,
        src_l: &Layout,
        dim: usize,
    ) -> Result<Self> {
        match ids {
            Self::U8(ids) => {
                let ids = match ids_l.contiguous_offsets() {
                    Some((a, b)) => &ids[a..b],
                    None => Err(Error::RequiresContiguous { op: "index-add" }.bt())?,
                };
                IndexAdd { ids, dim }.map(self, l, src, src_l)
            }
            Self::U32(ids) => {
                let ids = match ids_l.contiguous_offsets() {
                    Some((a, b)) => &ids[a..b],
                    None => Err(Error::RequiresContiguous { op: "index-add" }.bt())?,
                };
                IndexAdd { ids, dim }.map(self, l, src, src_l)
            }
            Self::I64(ids) => {
                let ids = match ids_l.contiguous_offsets() {
                    Some((a, b)) => &ids[a..b],
                    None => Err(Error::RequiresContiguous { op: "index-add" }.bt())?,
                };
                IndexAdd { ids, dim }.map(self, l, src, src_l)
            }
            _ => Err(Error::UnsupportedDTypeForOp(self.dtype(), "index-add").bt()),
        }
    }

    fn matmul(
        &self,
        rhs: &Self,
        bmnk: (usize, usize, usize, usize),
        lhs_l: &Layout,
        rhs_l: &Layout,
    ) -> Result<Self> {
        MatMul(bmnk).map(self, lhs_l, rhs, rhs_l)
    }

    fn device(&self) -> &Self::Device {
        &CpuDevice
    }

    fn try_clone(&self, _: &Layout) -> Result<Self> {
        Ok(self.clone())
    }

    fn to_cpu_storage(&self) -> Result<CpuStorage> {
        Ok(self.clone())
    }

    fn const_set(&mut self, s: crate::scalar::Scalar, l: &Layout) -> Result<()> {
        use crate::scalar::Scalar;
        fn set<T: crate::WithDType>(src: &mut [T], l: &Layout, s: T) {
            match l.strided_blocks() {
                crate::StridedBlocks::SingleBlock { start_offset, len } => {
                    src[start_offset..start_offset + len].fill(s)
                }
                crate::StridedBlocks::MultipleBlocks {
                    block_start_index,
                    block_len: 1,
                } => {
                    for src_index in block_start_index {
                        src[src_index] = s
                    }
                }
                crate::StridedBlocks::MultipleBlocks {
                    block_start_index,
                    block_len,
                } => {
                    for src_index in block_start_index {
                        src[src_index..src_index + block_len].fill(s)
                    }
                }
            }
        }
        match (self, s) {
            (Self::BF16(storage), Scalar::BF16(v)) => set(storage, l, v),
            (Self::F16(storage), Scalar::F16(v)) => set(storage, l, v),
            (Self::F32(storage), Scalar::F32(v)) => set(storage, l, v),
            (Self::F64(storage), Scalar::F64(v)) => set(storage, l, v),
            (Self::U8(storage), Scalar::U8(v)) => set(storage, l, v),
            (Self::U32(storage), Scalar::U32(v)) => set(storage, l, v),
            (Self::I64(storage), Scalar::I64(v)) => set(storage, l, v),
            (Self::F8E4M3(storage), Scalar::F8E4M3(v)) => set(storage, l, v),
            (st, s) => crate::bail!(
                "const_set dtype mismatch, expected {:?} but got {:?}",
                st.dtype(),
                s
            ),
        }
        Ok(())
    }
}
