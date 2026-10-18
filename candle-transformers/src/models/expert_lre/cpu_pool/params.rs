//! `SplitParams`: everything the CPU split decides with, as data
//! (`docs/cpu_expert_compute.md` §5.3.1).
//!
//! The measured costs come from calibration at load and the online loop; the
//! knobs from the daemon's config (`expert_split:`) and its admin route. The
//! forward thread turns the whole into the device's [`CpuSplit`] at every
//! invocation and passes it **by value** — a decode graph rewrites changed
//! parameters at each wave's recapture — so a value written between waves
//! takes effect on the next wave with no rebuild, no restart and no extra
//! kernel.
//!
//! One set per process ([`shared`]). Its default is **off**: an engine built
//! without a daemon's config — every test fixture — runs all-GPU, exactly as
//! reproducible as before the CPU class existed. A daemon turns it on from its
//! config before it loads the model.

use std::fmt::{self, Display, Formatter};
use std::str::FromStr;
use std::sync::{Arc, Mutex, OnceLock};

use candle_kernels::simple::moe_bucketize::{
    CpuSplit, CPU_MAX_N, CPU_PROMOTE_ALL, CPU_PROMOTE_COLD, CPU_PROMOTE_NONE,
};
use serde::{Deserialize, Serialize};
use serde_json::{to_value, Value};

/// The count buckets the cost tables are measured at; the device table holds
/// every count 1..=CPU_MAX_N, interpolated from these.
pub const BUCKETS: [usize; 10] = [1, 2, 3, 4, 6, 8, 12, 16, 24, 32];

/// Measured costs, all in nanoseconds.
#[derive(Debug, Clone, Copy, PartialEq, Serialize)]
pub struct Measured {
    /// One expert over the link.
    pub copy_ns: f32,
    /// The pool reading one expert with the link idle, and while it streams.
    pub cpu_read_alone_ns: f32,
    pub cpu_read_shared_ns: f32,
    /// The pool computing one expert at each of `BUCKETS` tokens.
    pub cpu_n_ns: [f32; BUCKETS.len()],
    /// The grouped GEMM for one expert at each of `BUCKETS` tokens.
    pub gpu_n_ns: [f32; BUCKETS.len()],
    /// The fixed handoff per layer with CPU work.
    pub handoff_ns: f32,
    /// The pool's own fixed cost per layer — waking its workers, the serial
    /// SwiGLU tail, the dataflow's last task — over what the per-expert
    /// throughput `cpu_n_ns` accounts for. A layer of one or two experts, the
    /// decode common case, is mostly this.
    pub pool_fixed_ns: f32,
    /// Per CPU row: its input out and its result back over the link.
    pub row_ns: f32,
}

/// The knobs (§5.3.1). Defaults are neutral: trust the measurements.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct Knobs {
    /// Multiplies every CPU cost; above 1 biases to the GPU.
    pub cpu_scale: f32,
    /// Multiplies the copy cost.
    pub gpu_scale: f32,
    /// Added to the handoff.
    pub handoff_extra_ns: f32,
    /// The cut: an expert stays on the GPU when its CPU compute exceeds this
    /// times what the GPU would spend on it.
    pub cut_scale: f32,
    /// The CPU path must beat all-GPU by this.
    pub min_gain_ns: f32,
    /// Most CPU rows a layer may take. 0 sends everything to the GPU — a value,
    /// not a flag: the same arithmetic with a cap of zero.
    pub max_cpu_assignments: u32,
    /// How long the device polls tightly for the pool's results.
    pub wait_tight_ns: u64,
    /// Invocations the pool may go without work before one explores: its
    /// split sends the cheapest miss when the model says it loses by at most
    /// `explore_ns` (§5.3.2). 0 never explores.
    pub explore_period: u32,
    pub explore_ns: f32,
    /// The online loop's time constant: how fast the measured CPU costs
    /// follow what the pool actually takes.
    pub online_tau_ms: f32,
    /// Residency, not the split: the weight of the pack read in what evicting
    /// a pack-only expert costs (§5.11). Above 1 keeps NVMe-backed experts
    /// resident ahead of RAM-backed ones of equal heat.
    pub cold_bias: f32,
    /// Which experts the pool served are copied into VRAM behind it, by the
    /// next layer's launch (§5.11).
    pub promote: Promote,
}

/// Which of the experts the CPU pool served the next launch promotes into
/// VRAM — claims a slot for and copies from the expert's pinned image, off the
/// path the pool's result is on (§5.11).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Promote {
    /// None: a pool-served expert stays where it is until a GPU miss or a
    /// read-ahead brings it in.
    None,
    /// The cold ones: read from the pack for the pool, they would be read
    /// again at their next miss.
    #[default]
    Cold,
    /// Every one, cold or pinned.
    All,
}

impl Promote {
    /// The device's value (`CPU_PROMOTE_*`).
    pub fn device(self) -> u32 {
        match self {
            Self::None => CPU_PROMOTE_NONE,
            Self::Cold => CPU_PROMOTE_COLD,
            Self::All => CPU_PROMOTE_ALL,
        }
    }
}

impl FromStr for Promote {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, String> {
        match s {
            "none" => Ok(Self::None),
            "cold" => Ok(Self::Cold),
            "all" => Ok(Self::All),
            other => Err(format!("promote: {other:?} is not none, cold or all")),
        }
    }
}

impl Display for Promote {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::None => "none",
            Self::Cold => "cold",
            Self::All => "all",
        })
    }
}

impl Default for Knobs {
    fn default() -> Self {
        Self {
            cpu_scale: 1.0,
            gpu_scale: 1.0,
            handoff_extra_ns: 0.0,
            cut_scale: 1.0,
            min_gain_ns: 0.0,
            max_cpu_assignments: 1024,
            wait_tight_ns: 200_000,
            // 64 tokens of a 48-layer model.
            explore_period: 3072,
            explore_ns: 20_000.0,
            online_tau_ms: 500.0,
            cold_bias: 1.0,
            promote: Promote::default(),
        }
    }
}

/// The process's split parameters. The default is on, with neutral knobs and
/// nothing measured — all-GPU until the pool has calibrated.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SplitParams {
    /// Whether the CPU class runs at all — the daemon's config.
    pub enabled: bool,
    pub knobs: Knobs,
    /// `None` until calibration has run.
    pub measured: Option<Measured>,
    /// Most experts a launch gives the pool (`CpuSplit::max_experts`): a
    /// multiple of its performance cores, set when the pool starts (`attach`).
    /// Its other misses stay on the GPU. 0 before the pool starts.
    pub max_experts: u32,
}

impl Default for SplitParams {
    fn default() -> Self {
        Self {
            enabled: true,
            knobs: Knobs::default(),
            measured: None,
            max_experts: 0,
        }
    }
}

/// The process's one set.
pub fn shared() -> Arc<Mutex<SplitParams>> {
    static GLOBAL: OnceLock<Arc<Mutex<SplitParams>>> = OnceLock::new();
    GLOBAL
        .get_or_init(|| Arc::new(Mutex::new(SplitParams::default())))
        .clone()
}

/// A table over `BUCKETS` read at count `n`: linear between buckets.
fn interpolate(t: &[f32; BUCKETS.len()], n: usize) -> f32 {
    if n <= BUCKETS[0] {
        return t[0] * n as f32 / BUCKETS[0] as f32;
    }
    for i in 1..BUCKETS.len() {
        if n <= BUCKETS[i] {
            let (n0, n1) = (BUCKETS[i - 1] as f32, BUCKETS[i] as f32);
            let w = (n as f32 - n0) / (n1 - n0);
            return t[i - 1] + w * (t[i] - t[i - 1]);
        }
    }
    t[BUCKETS.len() - 1] * n as f32 / BUCKETS[BUCKETS.len() - 1] as f32
}

impl SplitParams {
    /// The knobs with `patch`'s fields written over them: a JSON object naming
    /// any subset of [`Knobs`]' fields. A field it does not know, or a value of
    /// the wrong type, refuses the whole patch and leaves the knobs as they
    /// were.
    pub fn patch_knobs(&mut self, patch: &Value) -> Result<(), String> {
        let Value::Object(fields) = patch else {
            return Err("expected a JSON object of knobs".into());
        };
        let mut merged = to_value(self.knobs).map_err(|e| e.to_string())?;
        if let Value::Object(current) = &mut merged {
            for (k, v) in fields {
                current.insert(k.clone(), v.clone());
            }
        }
        self.knobs = serde_json::from_value(merged).map_err(|e| e.to_string())?;
        Ok(())
    }

    /// What the pool should take, by the measured costs (knobs not applied),
    /// to compute experts of these routed counts: their DRAM read at the
    /// pool's own rate, or their compute, whichever is longer. `None` until
    /// calibrated.
    pub fn predicted_pool_ns(&self, counts: &[u32]) -> Option<f32> {
        let m = self.measured?;
        let mem = counts.len() as f32 * m.cpu_read_alone_ns;
        let comp: f32 = counts
            .iter()
            .map(|&n| interpolate(&m.cpu_n_ns, n as usize))
            .sum();
        Some(m.pool_fixed_ns + mem.max(comp))
    }

    /// Scales every measured CPU cost by `factor`: what the online loop
    /// applies when the pool runs slower or faster than measured.
    pub fn rescale_cpu(&mut self, factor: f32) {
        if let Some(m) = &mut self.measured {
            m.pool_fixed_ns *= factor;
            m.cpu_read_alone_ns *= factor;
            m.cpu_read_shared_ns *= factor;
            for c in &mut m.cpu_n_ns {
                *c *= factor;
            }
        }
    }

    /// What a miss on an evicted expert costs over keeping it resident, in
    /// seconds, by its backing (§5.11), given one image's `copy` over the link
    /// and the pack's `read`. A resident expert pays its GEMM too, so a refill
    /// is priced by what it adds:
    /// - warm-backed: the copy — or, where the split runs and takes one-token
    ///   misses, the pool's handoff, read and compute less the GEMM it spares
    ///   the GPU, when that is cheaper: a warm miss goes to whichever is;
    /// - pack-only: `cold_bias` × the read, then the copy — the pool takes no
    ///   cold expert, so the read always lands ahead of the GPU.
    pub fn refill_secs(&self, copy: f64, read: f64) -> (f64, f64) {
        let s = self.split();
        let warm = if s.cap_rows > 0 && s.max_n >= 1 {
            let cpu = f64::from(s.handoff_ns + s.read_shared_ns + s.cpu_n_ns[1] - s.gpu_n_ns[1]);
            copy.min(cpu.max(0.0) * 1e-9)
        } else {
            copy
        };
        (warm, f64::from(self.knobs.cold_bias) * read + copy)
    }

    /// The device's split: off unless enabled and calibrated; otherwise the
    /// measured costs with the knobs applied, and the cut — the largest count
    /// whose CPU compute stays within `cut_scale` of the GPU's copy + GEMM,
    /// and never under one.
    ///
    /// **A lone token always passes the cut.** The cut keeps heavy experts on
    /// the GPU; it prices one expert alone, CPU against copy, with nothing
    /// running beside it. A one-token miss at the crossover is the layer
    /// model's to judge (`cpu_split_decide`), which prices the pool running
    /// beside the GPU's copies and holds the gain to `min_gain_ns`. A cut of
    /// zero also left nothing eligible to explore with, so a correction that
    /// pushed c(1) a microsecond past the copy shut the pool out for good:
    /// Flash-Next on the RTX 3090 (2026-10-18) served one config and then
    /// nothing, at c(1) 166 µs against 161 + 4.
    pub fn split(&self) -> CpuSplit {
        let Some(m) = self.measured.filter(|_| self.enabled) else {
            return CpuSplit::OFF;
        };
        let k = self.knobs;
        let mut s = CpuSplit::OFF;
        s.cap_rows = k.max_cpu_assignments;
        s.copy_ns = m.copy_ns * k.gpu_scale;
        s.read_shared_ns = m.cpu_read_shared_ns * k.cpu_scale;
        s.read_alone_ns = m.cpu_read_alone_ns * k.cpu_scale;
        // The device prices one fixed cost per layer with CPU work: the
        // handoff and the pool's own.
        s.handoff_ns = (m.handoff_ns + m.pool_fixed_ns) * k.cpu_scale + k.handoff_extra_ns;
        s.row_ns = m.row_ns;
        s.min_gain_ns = k.min_gain_ns;
        s.promote = k.promote.device();
        s.max_experts = self.max_experts;
        let mut max_n = 1u32;
        for n in 1..=CPU_MAX_N {
            s.cpu_n_ns[n] = interpolate(&m.cpu_n_ns, n) * k.cpu_scale;
            s.gpu_n_ns[n] = interpolate(&m.gpu_n_ns, n);
            if s.cpu_n_ns[n] <= k.cut_scale * (s.copy_ns + s.gpu_n_ns[n]) {
                max_n = n as u32;
            }
        }
        s.max_n = max_n;
        s
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn measured() -> Measured {
        Measured {
            copy_ns: 160_000.0,
            cpu_read_alone_ns: 25_000.0,
            cpu_read_shared_ns: 30_000.0,
            cpu_n_ns: [
                60e3, 70e3, 80e3, 90e3, 110e3, 130e3, 170e3, 210e3, 290e3, 370e3,
            ],
            gpu_n_ns: [2e3, 2e3, 2e3, 2e3, 3e3, 3e3, 4e3, 5e3, 7e3, 9e3],
            handoff_ns: 5_000.0,
            pool_fixed_ns: 0.0,
            row_ns: 1_000.0,
        }
    }

    /// On by default, but off until calibrated, and off when disabled even
    /// once calibrated.
    #[test]
    fn off_until_enabled_and_calibrated() {
        assert!(SplitParams::default().enabled);
        assert_eq!(SplitParams::default().split(), CpuSplit::OFF);
        let p = SplitParams {
            enabled: false,
            measured: Some(measured()),
            ..SplitParams::default()
        };
        assert_eq!(p.split(), CpuSplit::OFF);
    }

    /// The tables are interpolated between buckets, scaled by the knobs, and
    /// the cut is the last count the CPU stays within the GPU's cost.
    #[test]
    fn split_interpolates_scales_and_cuts() {
        let p = SplitParams {
            enabled: true,
            knobs: Knobs::default(),
            measured: Some(measured()),
            max_experts: 5,
        };
        let s = p.split();
        assert_eq!(s.cap_rows, 1024);
        assert_eq!(s.max_experts, 5, "the cap, copied as set");
        // Halfway between 4 (90) and 6 (110), and between 8 (130) and 12 (170).
        assert_eq!(s.cpu_n_ns[5], 100e3);
        assert_eq!(s.cpu_n_ns[10], 150e3);
        // The cut, CPU ≤ copy + GPU: n = 11 is 160k ≤ 163.75k; n = 12 is
        // 170k > 164k.
        assert_eq!(s.max_n, 11);
        let mut double = p;
        double.knobs.cpu_scale = 2.0;
        let d = double.split();
        assert_eq!(d.read_alone_ns, 50_000.0);
        // Doubled: n = 3 is 160k ≤ 162k; n = 4 is 180k > 162k.
        assert_eq!(d.max_n, 3);
        // A CPU slower than the copy at every count still takes a lone token.
        let mut slow = p;
        slow.knobs.cpu_scale = 100.0;
        assert_eq!(slow.split().max_n, 1);
    }

    /// The pool's prediction is the longer of its read and its compute, from
    /// the measured tables; a rescale moves every CPU cost and nothing else.
    #[test]
    fn the_prediction_and_its_rescale() {
        let mut p = SplitParams::default();
        assert_eq!(p.predicted_pool_ns(&[1]), None);
        p.measured = Some(measured());
        // Compute-bound: 60k + 100k (n = 5, between 4 and 6) > 2 × 25k.
        assert_eq!(p.predicted_pool_ns(&[1, 5]), Some(160e3));
        // Read-bound once a one-token compute is cheap: 3 × 25k > 3 × 1k.
        p.measured.as_mut().unwrap().cpu_n_ns[0] = 1e3;
        assert_eq!(p.predicted_pool_ns(&[1, 1, 1]), Some(75e3));
        p.rescale_cpu(2.0);
        let m = p.measured.unwrap();
        assert_eq!(m.cpu_read_alone_ns, 50e3);
        assert_eq!(m.cpu_read_shared_ns, 60e3);
        assert_eq!(m.cpu_n_ns[1], 140e3);
        assert_eq!(
            m.copy_ns, 160e3,
            "the GPU's costs are not the pool's to correct"
        );
        assert_eq!(m.gpu_n_ns, measured().gpu_n_ns);
    }

    /// Without the split a warm refill is the copy; with it, the pool's path
    /// when cheaper — handoff 5k + read 30k + compute 60k − GEMM 2k = 93 µs
    /// against a 160 µs copy. A pack-only refill is `cold_bias` × the read
    /// plus the copy, whatever the split.
    #[test]
    fn refills_are_priced_by_backing() {
        let (copy, read) = (160e-6, 800e-6);
        let off = SplitParams::default();
        assert_eq!(off.refill_secs(copy, read), (copy, read + copy));
        let mut on = SplitParams {
            enabled: true,
            knobs: Knobs::default(),
            measured: Some(measured()),
            max_experts: 5,
        };
        let (warm, pack) = on.refill_secs(copy, read);
        assert!((warm - 93e-6).abs() < 1e-12, "{warm}");
        assert_eq!(pack, read + copy);
        on.knobs.cold_bias = 2.5;
        assert_eq!(on.refill_secs(copy, read).1, 2.5 * read + copy);
    }

    /// A patch writes only the fields it names; an unknown field or a wrong
    /// type refuses it whole.
    /// `promote` reaches the device as `CPU_PROMOTE_*`, reads and writes as
    /// its lowercase name, and a patch sets it.
    #[test]
    fn promote_is_named_and_reaches_the_device() {
        let p = SplitParams {
            enabled: true,
            measured: Some(measured()),
            ..Default::default()
        };
        assert_eq!(p.split().promote, CPU_PROMOTE_COLD, "cold by default");
        for (v, name, dev) in [
            (Promote::None, "none", CPU_PROMOTE_NONE),
            (Promote::Cold, "cold", CPU_PROMOTE_COLD),
            (Promote::All, "all", CPU_PROMOTE_ALL),
        ] {
            assert_eq!(v.device(), dev);
            assert_eq!(v.to_string(), name);
            assert_eq!(name.parse::<Promote>(), Ok(v));
            assert_eq!(serde_json::to_value(v).unwrap(), serde_json::json!(name));
        }
        assert!("some".parse::<Promote>().is_err());
        let mut q = p;
        q.patch_knobs(&serde_json::json!({"promote": "all"}))
            .unwrap();
        assert_eq!(q.split().promote, CPU_PROMOTE_ALL);
        assert!(q
            .patch_knobs(&serde_json::json!({"promote": "most"}))
            .is_err());
    }

    #[test]
    fn a_patch_writes_its_fields_and_refuses_unknown_ones() {
        let mut p = SplitParams::default();
        p.patch_knobs(&serde_json::json!({"cpu_scale": 1.5, "max_cpu_assignments": 64}))
            .unwrap();
        assert_eq!(p.knobs.cpu_scale, 1.5);
        assert_eq!(p.knobs.max_cpu_assignments, 64);
        assert_eq!(p.knobs.gpu_scale, 1.0);
        let before = p.knobs;
        assert!(p
            .patch_knobs(&serde_json::json!({"cpu_scale": 2.0, "cpu_sclae": 3.0}))
            .is_err());
        assert!(p
            .patch_knobs(&serde_json::json!({"max_cpu_assignments": "lots"}))
            .is_err());
        assert!(p.patch_knobs(&serde_json::json!([1, 2])).is_err());
        assert_eq!(p.knobs, before);
    }

    /// A cap of zero is the all-GPU split.
    #[test]
    fn zero_cap_is_all_gpu() {
        let mut p = SplitParams {
            enabled: true,
            knobs: Knobs::default(),
            measured: Some(measured()),
            max_experts: 5,
        };
        p.knobs.max_cpu_assignments = 0;
        assert_eq!(p.split().cap_rows, 0);
    }
}
