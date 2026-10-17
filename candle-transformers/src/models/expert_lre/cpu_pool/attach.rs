//! Starting the CPU class for one expert cache (`docs/cpu_expert_compute.md`
//! §5.6): the rows' geometry as the kernel reads it, the pool's cores, the
//! rings, and the service thread.
//!
//! Returns no service — the dispatch then runs all-GPU — when the process's
//! split is off, or when a row's experts are not a KO width the CPU kernel
//! reads (Q3_KO, Q4_KO).

use std::sync::Arc;
use std::time::Duration;

use candle::quantized::GgmlDType;
use candle::Result;
use tracing::info;

use super::kernel::ko::Ko;
use super::params::shared;
use super::rings::CpuRings;
use super::service::{CpuService, PoolCores, RowShape};
use super::topology::{physical_cores, pool_cores};
use crate::models::expert_lre::pinned::LayerGeometry;

/// Performance cores left to the forward thread and the other threads that
/// must not be starved (§5.6).
const RESERVED_CORES: usize = 1;
/// Outside the warm window (`CpuService::wake`), how long the service waits on
/// the doorbell after a layer before it parks.
const SPIN: Duration = Duration::from_millis(5);
/// Outside the warm window, how long a pool worker waits for its next job
/// before it parks — and the only wait the slower cores' workers, which
/// inference never takes, ever have: calibration's jobs reach them, and then
/// they sleep. Kept waiting, fourteen workers held the laptop's CPU at ~75% of
/// base clock.
const WORKER_SPIN: Duration = Duration::from_micros(200);

/// Most experts a launch hands the pool, per performance core. Every layer
/// runs on all of them whatever its expert count (`layer::participants_for`),
/// so the cap is not a core count: under it the price (`cpu_n_ns`) decides,
/// and it stops only the rare launch the price would load past what the
/// device can wait for. Measured on the laptop (Flash-Next, split decode):
/// one a core 17.8–18.4 t/s with the cap binding at ~4.3 experts a layer; two
/// 19.4, the price settling at ~5.2; four no more experts on average, and
/// 17.6–18.6, the device waiting longer on the rare large layer.
const EXPERTS_PER_FAST_CORE: usize = 2;

fn ko_of(dt: GgmlDType) -> Option<Ko> {
    match dt {
        GgmlDType::Q3_KO => Some(Ko::Q3),
        GgmlDType::Q4_KO => Some(Ko::Q4),
        _ => None,
    }
}

/// The service for a cache over `geoms` (one per MoE row), or `None`.
/// `link_bps` is the link's measured rate.
pub(crate) fn start(geoms: &[LayerGeometry], link_bps: f64) -> Result<Option<CpuService>> {
    let (enabled, cap) = shared()
        .lock()
        .map(|p| (p.enabled, p.knobs.max_cpu_assignments))
        .unwrap_or((false, 0));
    if !enabled || cap == 0 || geoms.is_empty() || link_bps <= 0.0 {
        return Ok(None);
    }
    let mut shapes = Vec::with_capacity(geoms.len());
    for g in geoms {
        let (Some(ko), Some(down_ko)) = (ko_of(g.gate_dtype), ko_of(g.down_dtype)) else {
            info!(
                gate = ?g.gate_dtype,
                down = ?g.down_dtype,
                "cpu experts: off — the expert weights are not a KO width the CPU reads"
            );
            return Ok(None);
        };
        if ko != down_ko || g.up_dtype != g.gate_dtype {
            info!("cpu experts: off — one expert's projections mix KO widths");
            return Ok(None);
        }
        shapes.push(RowShape {
            hidden: g.gate_shape[1],
            ff: g.gate_shape[0],
            ko,
            bytes: [
                g.gate_repacked_size,
                g.up_repacked_size,
                g.down_repacked_size,
            ],
        });
    }
    let cores = pool_cores(
        &physical_cores().map_err(candle::Error::wrap)?,
        RESERVED_CORES,
    );
    let Some((first, rest)) = cores.split_first() else {
        info!("cpu experts: off — no core left for the pool");
        return Ok(None);
    };
    let workers: Vec<u32> = rest.iter().map(|c| c.primary()).collect();
    // The pool runs on its performance cores — the service thread's and the
    // workers' before the first slower one — so inference never waits on a
    // slower core, and the split's cap scales with them.
    let fast = cores
        .iter()
        .take_while(|c| c.efficiency_class == first.efficiency_class)
        .count();
    if let Ok(mut p) = shared().lock() {
        p.max_experts = (EXPERTS_PER_FAST_CORE * fast) as u32;
    }
    let rings = Arc::new(CpuRings::new(cap as usize, shapes[0].hidden)?);
    info!(
        pool = 1 + workers.len(),
        performance = fast,
        cap_rows = cap,
        "cpu experts: starting the pool"
    );
    Ok(Some(CpuService::start(
        rings,
        shapes,
        PoolCores {
            caller: first.primary(),
            workers,
            fast_workers: fast - 1,
        },
        SPIN,
        WORKER_SPIN,
        link_bps,
    )?))
}
