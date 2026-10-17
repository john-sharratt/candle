//! The card wave by wave through a batched gate — the bands the performance
//! page stacks (`candle_nn::kv_cache::VramBands`) — written to a CSV under the
//! workspace's `target/` as the gate runs.
//!
//! One row per forward. A thread watches the region pool's tier count
//! (`RegionStats::tiers_placed`, one per forward) and reads the bands each time
//! it moves. The span's partition carries the widest tier since the reading
//! before, so a forward the thread sees late still shows its activations; a row
//! covering more than one forward says how many in `forwards`.

use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use candle::{Device, DeviceLocation};
use candle_nn::kv_cache::{region_stats, span_partition, vram_bands, VramBands};

use crate::models::expert_lre::PipelineStats;

/// How often the thread looks for a new forward.
const POLL: Duration = Duration::from_millis(2);

const HEADER: &str = "wave,t_s,forwards,config,total,free,context,pool,dense,experts,\
                      expert_free,kv,state,transient,reserved,kv_free,unaccounted";

/// One row, the bands in MiB to one decimal.
fn row(wave: u64, t: f64, forwards: u64, config: &str, b: &VramBands) -> String {
    format!(
        "{wave},{t:.3},{forwards},{config},{:.1},{:.1},{:.1},{:.1},{:.1},{:.1},{:.1},{:.1},{:.1},\
         {:.1},{:.1},{:.1},{:.1}",
        b.total,
        b.free,
        b.context,
        b.pool,
        b.dense,
        b.experts,
        b.expert_free,
        b.kv,
        b.state,
        b.transient,
        b.reserved,
        b.kv_free,
        b.unaccounted,
    )
}

/// The span's device ordinal, for a CUDA device.
fn gpu_id(device: &Device) -> Option<usize> {
    match device.location() {
        DeviceLocation::Cuda { gpu_id } => Some(gpu_id),
        _ => None,
    }
}

/// The device's bands now, as the scheduler reads them for the performance page.
fn bands(
    device: &Device,
    gpu: usize,
    experts: Option<&Arc<Mutex<PipelineStats>>>,
) -> Option<VramBands> {
    let (free, total) = device.mem_get_info().ok()?;
    let pool = device
        .as_cuda_device()
        .ok()
        .and_then(|d| d.pool_reserved_bytes().ok())
        .unwrap_or(0);
    let context = region_stats(gpu).and_then(|s| s.pre_reservation_in_use_bytes);
    let resident = experts
        .and_then(|s| s.lock().ok().map(|s| s.resident_vram_bytes as u64))
        .unwrap_or(0);
    Some(vram_bands(
        total as u64,
        free as u64,
        context,
        pool as u64,
        span_partition(gpu),
        resident,
    ))
}

/// A running log. Dropping it stops the thread and closes the file.
pub struct VramWaveLog {
    stop: Arc<AtomicBool>,
    config: Arc<Mutex<String>>,
    thread: Option<JoinHandle<u64>>,
    path: PathBuf,
}

impl VramWaveLog {
    /// Start logging `device` to a fresh file under `target/`, or `None` when
    /// it has no span to read or the file cannot be made. `experts` splits the
    /// weight zone into resident experts and free slots.
    pub fn start(device: &Device, experts: Option<Arc<Mutex<PipelineStats>>>) -> Option<Self> {
        let gpu = gpu_id(device)?;
        region_stats(gpu)?;
        let stamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_or(0, |d| d.as_secs());
        let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("..")
            .join("target");
        let path = dir.join(format!("vram_waves_{stamp}.csv"));
        let mut out = BufWriter::new(File::create(&path).ok()?);
        writeln!(out, "{HEADER}").ok()?;
        let stop = Arc::new(AtomicBool::new(false));
        let config = Arc::new(Mutex::new(String::new()));
        let (flag, label, device) = (stop.clone(), config.clone(), device.clone());
        let thread = thread::Builder::new()
            .name("vram-wave-log".into())
            .spawn(move || {
                let origin = Instant::now();
                let mut seen = region_stats(gpu).map_or(0, |s| s.tiers_placed);
                let mut wave = 0u64;
                while !flag.load(Ordering::Relaxed) {
                    thread::sleep(POLL);
                    let Some(now) = region_stats(gpu).map(|s| s.tiers_placed) else {
                        continue;
                    };
                    if now == seen {
                        continue;
                    }
                    let Some(b) = bands(&device, gpu, experts.as_ref()) else {
                        continue;
                    };
                    wave += 1;
                    let config = label.lock().map(|c| c.clone()).unwrap_or_default();
                    let t = origin.elapsed().as_secs_f64();
                    let line = row(wave, t, now - seen, &config, &b);
                    if writeln!(out, "{line}").and_then(|()| out.flush()).is_err() {
                        break;
                    }
                    seen = now;
                }
                wave
            })
            .ok()?;
        println!("  VRAM wave log: {}", path.display());
        Some(Self {
            stop,
            config,
            thread: Some(thread),
            path,
        })
    }

    /// Label the rows that follow — the config now running.
    pub fn set_config(&self, label: &str) {
        if let Ok(mut c) = self.config.lock() {
            *c = label.to_string();
        }
    }
}

impl Drop for VramWaveLog {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Relaxed);
        if let Some(rows) = self.thread.take().and_then(|t| t.join().ok()) {
            println!(
                "  VRAM wave log: {rows} waves written to {}",
                self.path.display()
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A row carries the wave, its time, the forwards it covers, the config
    /// and every band in MiB to one decimal, in the header's order.
    #[test]
    fn a_row_is_the_headers_columns() {
        let b = VramBands {
            total: 16_375.5,
            free: 1_360.0,
            context: 1_361.5,
            pool: 614.0,
            dense: 6_016.0,
            experts: 3_955.2,
            expert_free: 0.0,
            kv: 240.0,
            state: 2_048.0,
            transient: 144.0,
            reserved: 520.25,
            kv_free: 128.0,
            unaccounted: -11.45,
        };
        assert_eq!(
            row(7, 1.23456, 2, "C10 x8", &b),
            "7,1.235,2,C10 x8,16375.5,1360.0,1361.5,614.0,6016.0,3955.2,0.0,240.0,2048.0,\
             144.0,520.2,128.0,-11.4"
        );
        assert_eq!(HEADER.split(',').count(), 17);
        assert_eq!(row(1, 0.0, 1, "x", &b).split(',').count(), 17);
    }
}
