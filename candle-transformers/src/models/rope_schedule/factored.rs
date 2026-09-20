//! The QSA indexer's rotation: every rung's factored table, and each rung's
//! step tables (`docs/progressive_yarn.md` §7.4).
//!
//! The indexer rotates with the attention's own schedule, rung for rung, as the
//! reference does (§12). Its queries are rotated in one launch over the wave,
//! each row at its sequence's rung, straight from the rung set; its keys are
//! rotated inside the paged scorer, which is launched per sequence span and so
//! takes that one sequence's rung table and the step table for its stride.

use candle::{Device, Result, Tensor};

use super::rungs::RopeRungs;
use super::schedule::RopeSchedule;
use super::table::{build_steps, MAX_STEP};

/// The indexer's tables on the device.
#[derive(Debug, Clone)]
pub struct FactoredRope {
    rungs: RopeRungs,
    /// `steps[r][k]` is rung `r`'s step table for a stride of `k + 1`.
    steps: Vec<Vec<Tensor>>,
}

impl FactoredRope {
    /// One rung of `inv_freq`, at every length — a schedule that never
    /// changes rung.
    pub fn new(inv_freq: &[f32], device: &Device) -> Result<Self> {
        let schedule = RopeSchedule::stated(inv_freq.to_vec(), usize::MAX)?;
        Self::over(&RopeRungs::new(&schedule, device)?, device)
    }

    /// The indexer's tables over `rungs`, the attention's own set — shared,
    /// not copied. Only the step tables are new, built from each rung's own
    /// frequencies in the set's `LO` arithmetic, so a warp term and a lane term
    /// always come from one rung and one arithmetic.
    pub fn over(rungs: &RopeRungs, device: &Device) -> Result<Self> {
        let steps = (0..rungs.n_rungs() as u32)
            .map(|r| {
                (1..=MAX_STEP)
                    .map(|step| {
                        let s = build_steps(rungs.inv_freq(r), step, rungs.lo_angle());
                        let n = s.len();
                        Tensor::from_vec(s, n, device)
                    })
                    .collect::<Result<Vec<_>>>()
            })
            .collect::<Result<_>>()?;
        Ok(Self {
            rungs: rungs.clone(),
            steps,
        })
    }

    /// Every rung, as the query rotation takes them.
    pub fn rungs(&self) -> &RopeRungs {
        &self.rungs
    }

    /// The rung a sequence with `reach` positions rotates by — the same the
    /// attention's header writers pick.
    pub fn rung_for(&self, reach: usize) -> Result<u32> {
        self.rungs.rung_for(reach)
    }

    /// Rung `rung`'s table, `f32[(HI + LO) · pairs · 2]`.
    pub fn table(&self, rung: u32) -> Result<Tensor> {
        self.rungs.table(rung)
    }

    /// Rung `rung`'s step table for a stride of `step` tokens,
    /// `f32[pairs · 32 · 2]`.
    pub fn steps(&self, rung: u32, step: usize) -> Result<&Tensor> {
        let Some(tables) = self.steps.get(rung as usize) else {
            candle::bail!("indexer rope: rung {rung} of {}", self.steps.len())
        };
        match step.checked_sub(1).and_then(|k| tables.get(k)) {
            Some(t) => Ok(t),
            None => {
                candle::bail!("indexer rope: no step table for a stride of {step} (1..={MAX_STEP})")
            }
        }
    }

    /// Rotary pairs per position — half the rotary width.
    pub fn pairs(&self) -> usize {
        self.rungs.pairs()
    }

    /// The rotary width, `2 × pairs`.
    pub fn rope_dim(&self) -> usize {
        self.pairs() * 2
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::rope_schedule::angle::AngleArithmetic;
    use crate::models::rope_schedule::schedule::Rung;
    use crate::models::rope_schedule::table::{build, plain_inv_freq, STEP_LANES};

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
            ],
            true,
        )
        .unwrap()
    }

    /// Each rung's table is that rung's frequencies, shared with the set; its
    /// step tables are built from the same frequencies.
    #[test]
    fn each_rung_has_its_own_table_and_steps() {
        for arith in [AngleArithmetic::Exact, AngleArithmetic::F32Product] {
            let s = hybrid().with_lo_angle(arith);
            let rungs = RopeRungs::new(&s, &Device::Cpu).unwrap();
            let f = FactoredRope::over(&rungs, &Device::Cpu).unwrap();
            let freqs = s.rungs();
            for r in 0..2u32 {
                let inv = &freqs[r as usize].inv_freq;
                assert_eq!(
                    f.table(r).unwrap().to_vec1::<f32>().unwrap(),
                    build(inv, arith)
                );
                let st = f.steps(r, 3).unwrap().to_vec1::<f32>().unwrap();
                assert_eq!(st, build_steps(inv, 3, arith));
                assert_eq!(st.len(), 32 * STEP_LANES * 2);
            }
        }
        let f = FactoredRope::over(
            &RopeRungs::new(&hybrid(), &Device::Cpu).unwrap(),
            &Device::Cpu,
        )
        .unwrap();
        assert_eq!(f.rung_for(262_145).unwrap(), 1);
        assert!(f.steps(2, 1).is_err());
        assert!(f.steps(0, MAX_STEP + 1).is_err());
    }

    /// A single-frequency-set table is one rung, to the table's reach.
    #[test]
    fn a_plain_table_is_one_rung_to_the_reach() {
        use crate::models::rope_schedule::table::ROPE_REACH;
        let f = FactoredRope::new(&plain_inv_freq(64, 1e7), &Device::Cpu).unwrap();
        assert_eq!(f.rungs().n_rungs(), 1);
        assert_eq!(f.rung_for(ROPE_REACH).unwrap(), 0);
        assert!(f.rung_for(ROPE_REACH + 1).is_err());
        assert_eq!(f.rope_dim(), 64);
    }
}
