//! The text model's vision tower, as a guest.
//!
//! An image the text model is to read becomes rows here, once, between the
//! engine's waves: the tower's weights are placed in guest ground from the
//! model's `mmproj` GGUF, each image runs through it on the guest arena, and
//! the rows come back on the host. The engine then reads them from the vision
//! registry by the virtual ids a prompt names them with
//! (`candle_transformers::models::qwen_vision`). The tower holds no VRAM
//! between images, and the engine's own load is untouched by it.
//!
//! Design: `docs/flash_next_vision.md` §3.

use std::path::PathBuf;
use std::sync::{Arc, Mutex};

use candle::quantized::Int8Mode;
use candle::{DType, Device};
use candle_transformers::models::qwen_vision::tower::ACTIVATION_BYTES_PER_PATCH;
use candle_transformers::models::qwen_vision::VisionTower;

use super::ground::GuestGround;
use super::groundgguf::{ground_bytes, place_gguf};
use super::model::GuestModel;
use super::progress::GuestSink;
use super::work::{Guest, GuestOutcome, GuestRequest, VisionRows};

/// What the activation arena leaves of the widest run, for the placements
/// after it.
const ARENA_HEADROOM_BYTES: usize = 32 << 20;
/// The smallest arena opened: a 64-token image's forward.
const MIN_ARENA_BYTES: usize = 64 << 20;

/// The tower, configured by where its `mmproj` GGUF is found.
///
/// `mmproj` answers from the local cache alone. The file is fetched by the
/// caller before it queues a job ([`crate::vision::encode`]), so a drain — every
/// wave stopped — never waits on the network.
pub struct VisionGuest {
    mmproj: fn() -> Option<PathBuf>,
    loaded: Option<(Device, VisionTower)>,
}

impl VisionGuest {
    pub fn new(mmproj: fn() -> Option<PathBuf>) -> Self {
        Self {
            mmproj,
            loaded: None,
        }
    }

    fn path(&self) -> Result<PathBuf, String> {
        (self.mmproj)().ok_or_else(|| {
            "vision guest: the tower's mmproj is not in the local cache — it is fetched before a \
             job is queued"
                .to_string()
        })
    }
}

/// The activation bytes the backlog's largest image needs.
fn activation_bytes(jobs: &[GuestRequest]) -> usize {
    jobs.iter()
        .filter_map(|j| match j {
            GuestRequest::Vision(r) => r
                .images
                .iter()
                .map(|i| (i.height as usize / 16) * (i.width as usize / 16))
                .max(),
            _ => None,
        })
        .max()
        .unwrap_or(0)
        * ACTIVATION_BYTES_PER_PATCH
}

impl GuestModel for VisionGuest {
    fn guest(&self) -> Guest {
        Guest::Vision
    }

    fn footprint_bytes(&self, jobs: &[GuestRequest]) -> usize {
        let mode = Int8Mode::auto(&Device::Cpu);
        let weights = self
            .path()
            .and_then(|p| ground_bytes(&p, mode, &[]))
            .unwrap_or(0);
        weights + activation_bytes(jobs).max(MIN_ARENA_BYTES) + ARENA_HEADROOM_BYTES
    }

    fn load(
        &mut self,
        device: &Device,
        ground: &Arc<Mutex<GuestGround>>,
        jobs: &[GuestRequest],
    ) -> Result<(), String> {
        let mode = Int8Mode::auto(device);
        // The arena first, on the widest run: it must be one stretch, and the
        // weights are tensors of a few megabytes that fit the runs it leaves.
        let wanted = activation_bytes(jobs).max(MIN_ARENA_BYTES);
        let arena = {
            let mut g = ground
                .lock()
                .map_err(|_| "vision guest: the ground lock was poisoned".to_string())?;
            let len = wanted.min(g.largest_free_run().saturating_sub(ARENA_HEADROOM_BYTES));
            if len < MIN_ARENA_BYTES {
                return Err(format!(
                    "vision guest: the ground's widest run leaves {} MiB for activations, \
                     {} MiB wanted",
                    len >> 20,
                    wanted >> 20
                ));
            }
            let at = g
                .place(len, 256)
                .map_err(|e| format!("vision guest: placing the arena: {e}"))?;
            (at, len)
        };
        let vb = place_gguf(&self.path()?, device, ground, mode, &[])?;
        let tower = VisionTower::load(&vb, mode).map_err(|e| format!("vision guest: {e}"))?;
        if let Device::Cuda(cuda) = device {
            candle_nn::kv_cache::open_guest_arena(&cuda.cuda_stream(), arena.0.ptr, arena.1);
        }
        tracing::info!(
            target: "candle_conversation::guest",
            arena_mib = arena.1 >> 20,
            "vision guest: loaded"
        );
        self.loaded = Some((device.clone(), tower));
        Ok(())
    }

    fn run(&mut self, request: &GuestRequest, _sink: &GuestSink) -> Result<GuestOutcome, String> {
        let GuestRequest::Vision(r) = request else {
            return Err(format!(
                "the vision guest was handed a {} job — the queue routed by kind and should not \
                 have",
                request.guest()
            ));
        };
        let (_, tower) = self
            .loaded
            .as_ref()
            .ok_or_else(|| "the vision guest was asked to run before it loaded".to_string())?;
        let mut rows = Vec::with_capacity(r.images.len());
        for image in &r.images {
            let out = tower
                .forward(&image.input())
                .and_then(|t| t.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>())
                .map_err(|e| format!("vision guest: {e}"))?;
            rows.push(out);
        }
        Ok(GuestOutcome::Vision(VisionRows { rows }))
    }

    fn unload(&mut self) {
        // Before the ground goes back: a ticket outliving the arena would carve
        // from regions that are attention state again.
        if let Some((device, _)) = self.loaded.as_ref() {
            if let Device::Cuda(cuda) = device {
                candle_nn::kv_cache::close_guest_arena(&cuda.cuda_stream());
            }
        }
        self.loaded = None;
    }
}
