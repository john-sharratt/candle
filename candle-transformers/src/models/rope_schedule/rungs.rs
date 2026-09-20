//! A model's rungs on the device: every rung's factored table, end to end, and
//! each rung's Q rotary scale — the one argument every paged attention kernel
//! rotates from (`docs/progressive_yarn.md` §6.1).
//!
//! A kernel takes the whole set per launch and each sequence's rung from its
//! own `SlotHeader.rope_rung`, so sequences on different rungs share a launch
//! without sharing anything rung-dependent. Every rung's table exists from
//! load: at 768 KiB – 1.5 MiB per rung the whole set sits in L2, and nothing
//! ever rebuilds it for a longer sequence.

use std::sync::Arc;

use candle::{Device, Result, Tensor};
#[cfg(feature = "cuda")]
use candle_kernels::rope::RopeRungsFfi;

use super::angle::AngleArithmetic;
use super::schedule::RopeSchedule;
use super::select::rung_of;
use super::table::{build, lookup, ROPE_HI_DIM, ROPE_LO_DIM, ROPE_REACH};

/// Every rung of a schedule, uploaded, with the host copies the float
/// fallback and the oracles rotate from.
#[derive(Debug, Clone)]
pub struct RopeRungs {
    /// `f32[n_rungs · (HI + LO) · pairs · 2]`, `(sin, cos)` pairs.
    tables: Tensor,
    /// `f32[n_rungs]`.
    q_scale: Tensor,
    /// Shared, so a clone costs two reference counts, not the tables.
    host_tables: Arc<[Vec<f32>]>,
    host_q_scale: Vec<f32>,
    /// Each rung's frequencies, as the schedule gave them.
    inv_freq: Arc<[Vec<f32>]>,
    /// The `LO` rows' arithmetic, as the schedule named it — what a step table
    /// built beside these tables must share.
    lo_angle: AngleArithmetic,
    pairs: usize,
    ceilings: Vec<usize>,
}

impl RopeRungs {
    /// Build and upload every rung of `schedule`.
    pub fn new(schedule: &RopeSchedule, device: &Device) -> Result<Self> {
        let rungs = schedule.rungs();
        let pairs = schedule.pairs();
        if pairs == 0 {
            candle::bail!("rope rungs: a schedule with no rotary pairs");
        }
        let lo_angle = schedule.lo_angle();
        let host_tables: Arc<[Vec<f32>]> =
            rungs.iter().map(|r| build(&r.inv_freq, lo_angle)).collect();
        let host_q_scale: Vec<f32> = rungs.iter().map(|r| r.q_rot_scale).collect();
        let flat: Vec<f32> = host_tables.iter().flatten().copied().collect();
        let n = flat.len();
        Ok(Self {
            tables: Tensor::from_vec(flat, n, device)?,
            q_scale: Tensor::from_vec(host_q_scale.clone(), host_q_scale.len(), device)?,
            host_tables,
            host_q_scale,
            inv_freq: rungs.into_iter().map(|r| r.inv_freq).collect(),
            lo_angle,
            pairs,
            // No ceiling past the table's reach: the kernel lookup clamps a
            // position beyond it, so a schedule "without a ceiling" still
            // refuses one rather than rotating it wrongly.
            ceilings: schedule
                .ceilings()
                .into_iter()
                .map(|c| c.min(ROPE_REACH))
                .collect(),
        })
    }

    /// Rung `rung`'s frequencies — rung 0's are the trained RoPE, what a
    /// control schedule extrapolates.
    pub fn inv_freq(&self, rung: u32) -> &[f32] {
        &self.inv_freq[rung as usize]
    }

    /// The `LO` rows' arithmetic.
    pub fn lo_angle(&self) -> AngleArithmetic {
        self.lo_angle
    }

    /// Rotary pairs per position.
    pub fn pairs(&self) -> usize {
        self.pairs
    }

    /// Rungs in the set.
    pub fn n_rungs(&self) -> usize {
        self.host_tables.len()
    }

    /// Each rung's highest reach, ascending — what a header writer picks a
    /// sequence's rung by.
    pub fn ceilings(&self) -> &[usize] {
        &self.ceilings
    }

    /// The rung a sequence with `reach` positions rotates by.
    pub fn rung_for(&self, reach: usize) -> Result<u32> {
        rung_of(&self.ceilings, reach)
    }

    /// Rung `rung`'s Q rotary scale (`m²`).
    pub fn q_scale(&self, rung: u32) -> f32 {
        self.host_q_scale[rung as usize]
    }

    /// `(cos, sin)` of frequency `f` at `pos` on rung `rung`, as the kernels
    /// read it: the factored lookup for a rotary pair, the identity for a
    /// pass-through one.
    pub fn cos_sin(&self, rung: u32, pos: usize, f: usize) -> (f32, f32) {
        if f >= self.pairs {
            return (1.0, 0.0);
        }
        let (s, c) = lookup(&self.host_tables[rung as usize], self.pairs, pos, f);
        (c, s)
    }

    /// Rung `rung`'s table alone, `f32[(HI + LO) · pairs · 2]` — a view into the
    /// set, not a copy.
    pub fn table(&self, rung: u32) -> Result<Tensor> {
        if rung as usize >= self.n_rungs() {
            candle::bail!("rope rungs: rung {rung} of a {}-rung set", self.n_rungs());
        }
        let len = rung_table_len(self.pairs);
        self.tables.narrow(0, rung as usize * len, len)
    }

    /// The launch argument: device addresses of the tables and scales.
    #[cfg(feature = "cuda")]
    pub fn ffi(&self) -> Result<RopeRungsFfi> {
        Ok(RopeRungsFfi {
            tables: f32_ptr(&self.tables)?,
            q_scale: f32_ptr(&self.q_scale)?,
            n_rungs: self.n_rungs() as u32,
            pairs: self.pairs as u32,
        })
    }
}

/// Elements in one rung's table: `(HI + LO) · pairs` `(sin, cos)` pairs.
pub fn rung_table_len(pairs: usize) -> usize {
    (ROPE_HI_DIM + ROPE_LO_DIM) * pairs * 2
}

#[cfg(feature = "cuda")]
fn f32_ptr(t: &Tensor) -> Result<u64> {
    use candle::cuda_backend::cudarc::driver::DevicePtr;
    use candle::Storage;
    let Device::Cuda(dev) = t.device() else {
        candle::bail!("rope rungs: the tables live on a CUDA device")
    };
    let stream = dev.cuda_stream();
    let (s, l) = t.storage_and_layout();
    let slice = match &*s {
        Storage::Cuda(c) => c.as_cuda_slice::<f32>()?,
        _ => candle::bail!("rope rungs: expected CUDA f32 storage"),
    }
    .slice(l.start_offset()..);
    let (p, _g) = slice.device_ptr(&stream);
    Ok(p)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::rope_schedule::schedule::Rung;
    use crate::models::rope_schedule::table::plain_inv_freq;

    fn hybrid() -> RopeSchedule {
        RopeSchedule::yarn(
            64,
            1e7,
            262_144,
            vec![
                Rung {
                    ceiling: 262_144,
                    factor: 1.0,
                },
                Rung {
                    ceiling: 524_288,
                    factor: 2.0,
                },
                Rung {
                    ceiling: 1_010_000,
                    factor: 4.0,
                },
            ],
            true,
        )
        .unwrap()
    }

    /// The tables lie end to end, one rung's length apart, in rung order.
    #[test]
    fn rungs_lie_end_to_end() {
        let r = RopeRungs::new(&hybrid(), &Device::Cpu).unwrap();
        assert_eq!(r.n_rungs(), 3);
        assert_eq!(r.pairs(), 32);
        let flat = r.tables.to_vec1::<f32>().unwrap();
        assert_eq!(flat.len(), 3 * rung_table_len(32));
        let rung1 = &flat[rung_table_len(32)..2 * rung_table_len(32)];
        let want = build(&hybrid().rungs()[1].inv_freq, AngleArithmetic::Exact);
        assert_eq!(rung1, &want[..]);
        assert_eq!(r.q_scale.to_vec1::<f32>().unwrap(), r.host_q_scale);
        assert_eq!(r.q_scale(0), 1.0);
        assert_eq!(r.table(1).unwrap().to_vec1::<f32>().unwrap(), want);
        assert!(r.table(3).is_err());
        assert_eq!(r.inv_freq(1), &hybrid().rungs()[1].inv_freq[..]);
        assert_eq!(r.inv_freq(0), &plain_inv_freq(64, 1e7)[..]);
    }

    /// The host lookup is the kernels': rung 0 at a rotary pair is the
    /// factored lookup of the plain frequencies, a pass-through pair the
    /// identity.
    #[test]
    fn the_host_lookup_is_the_kernels() {
        let r = RopeRungs::new(&hybrid(), &Device::Cpu).unwrap();
        let t = build(&plain_inv_freq(64, 1e7), AngleArithmetic::Exact);
        let (s, c) = lookup(&t, 32, 123_457, 7);
        assert_eq!(r.cos_sin(0, 123_457, 7), (c, s));
        assert_eq!(r.cos_sin(2, 123_457, 32), (1.0, 0.0));
        assert_eq!(r.cos_sin(2, 123_457, 127), (1.0, 0.0));
    }

    /// Every rung's table takes the schedule's `LO` arithmetic, and the set
    /// reports it for the step tables built beside it.
    #[test]
    fn every_rung_takes_the_schedules_lo_arithmetic() {
        let s = hybrid().with_lo_angle(AngleArithmetic::F32Product);
        let r = RopeRungs::new(&s, &Device::Cpu).unwrap();
        assert_eq!(r.lo_angle(), AngleArithmetic::F32Product);
        for (i, rung) in s.rungs().iter().enumerate() {
            let want = build(&rung.inv_freq, AngleArithmetic::F32Product);
            assert_eq!(r.table(i as u32).unwrap().to_vec1::<f32>().unwrap(), want);
        }
        let a = 700f32 * r.inv_freq(0)[0];
        assert_eq!(r.cos_sin(0, 700, 0), (a.cos(), a.sin()));
    }

    /// A header writer's rung is the schedule's, and past the maximum refuses.
    #[test]
    fn the_rung_is_the_schedules() {
        let r = RopeRungs::new(&hybrid(), &Device::Cpu).unwrap();
        assert_eq!(r.ceilings(), &[262_144, 524_288, 1_010_000]);
        assert_eq!(r.rung_for(262_144).unwrap(), 0);
        assert_eq!(r.rung_for(262_145).unwrap(), 1);
        assert_eq!(r.rung_for(1_010_000).unwrap(), 2);
        assert!(r.rung_for(1_010_001).is_err());
    }

    /// A schedule with no ceiling of its own stops at the table's reach.
    #[test]
    fn no_rung_reaches_past_the_table() {
        let s = RopeSchedule::stated(plain_inv_freq(64, 1e7), usize::MAX).unwrap();
        let r = RopeRungs::new(&s, &Device::Cpu).unwrap();
        assert_eq!(r.ceilings(), &[ROPE_REACH]);
        assert_eq!(r.rung_for(ROPE_REACH).unwrap(), 0);
        assert!(r.rung_for(ROPE_REACH + 1).is_err());
    }
}
