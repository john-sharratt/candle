//! The text model's vision tower, as a guest.
//!
//! An image the text model is to read becomes rows here, once, between the
//! engine's waves: the tower's weights are placed in guest ground from its
//! twinned model pack, each image runs through it on the guest arena, and the
//! rows come back on the host. The engine then reads them from the vision
//! registry by the virtual ids a prompt names them with
//! (`candle_transformers::models::qwen_vision`). The tower holds no VRAM
//! between images, and the engine's own load is untouched by it.
//!
//! The pack stores each projection as the KO twin the tower's matmuls read
//! (`model_pack::twins`), so placement copies every tensor as it is: one DMA a
//! tensor from the pinned staging buffers, and nothing allocated off the
//! ground.
//!
//! Design: `docs/flash_next_vision.md` §3.

use std::path::PathBuf;
use std::sync::{Arc, Mutex};

use candle::quantized::Int8Mode;
use candle::{DType, Device};
use candle_nn::kv_cache::{close_guest_arena, open_guest_arena};
use candle_transformers::models::qwen_vision::tower::ACTIVATION_BYTES_PER_PATCH;
use candle_transformers::models::qwen_vision::VisionTower;

use super::gguf_pick::Pick;
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
/// The placement's mode: the pack already holds every twin, so none is built
/// as the tower goes down.
const PLACEMENT: Int8Mode = Int8Mode::Off;

/// The tower, configured by where its pack is found, and the int8 mode that
/// pack was built at.
///
/// `pack` answers from the local cache alone. The pack is fetched and built by
/// the caller before it queues a job ([`crate::vision::encode`]), so a drain —
/// every wave stopped — never waits on the network or a build.
pub struct VisionGuest {
    pack: fn() -> Option<PathBuf>,
    mode: Int8Mode,
    loaded: Option<(Device, VisionTower)>,
}

impl VisionGuest {
    pub fn new(pack: fn() -> Option<PathBuf>, mode: Int8Mode) -> Self {
        Self {
            pack,
            mode,
            loaded: None,
        }
    }

    fn path(&self) -> Result<PathBuf, String> {
        (self.pack)().ok_or_else(|| {
            "vision guest: the tower's pack is not on this machine — it is built before a job is \
             queued"
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
        let weights = self
            .path()
            .and_then(|p| ground_bytes(&p, PLACEMENT, Pick::default()))
            .unwrap_or(0);
        weights + activation_bytes(jobs).max(MIN_ARENA_BYTES) + ARENA_HEADROOM_BYTES
    }

    fn load(
        &mut self,
        device: &Device,
        ground: &Arc<Mutex<GuestGround>>,
        jobs: &[GuestRequest],
    ) -> Result<(), String> {
        // The pack's twins are read by the int8 matmul alone; a card without it
        // has no tower to load.
        if !Int8Mode::auto(device).is_int8() {
            return Err(format!(
                "vision guest: the tower's pack stores {:?} twins, which need the int8 matmul \
                 this device does not have",
                self.mode
            ));
        }
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
        let vb = place_gguf(&self.path()?, device, ground, PLACEMENT, Pick::default())?;
        let tower = VisionTower::load(&vb, self.mode).map_err(|e| format!("vision guest: {e}"))?;
        if let Device::Cuda(cuda) = device {
            open_guest_arena(&cuda.cuda_stream(), arena.0.ptr, arena.1);
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
        if let Some((Device::Cuda(cuda), _)) = self.loaded.as_ref() {
            close_guest_arena(&cuda.cuda_stream());
        }
        self.loaded = None;
    }
}

#[cfg(all(test, feature = "hub"))]
mod tests {
    use super::*;
    use crate::guest::progress::GuestSink;
    use crate::guest::work::{VisionImage, VisionRequest};
    use candle_transformers::models::quantized_qwen38_moe::{
        cached_vision_tower_pack, vision_tower_pack, VISION_TOWER_MODE,
    };
    use cudarc::driver::DevicePtr;

    fn tower_pack() -> Option<PathBuf> {
        cached_vision_tower_pack(VISION_TOWER_MODE)
    }

    /// **The guest reads an image from its own ground**, as a drain runs it:
    /// the footprint it asks for, the tower's pack placed into that ground —
    /// its twins as they are stored — the arena opened on it, one 768 × 512
    /// image run, and the guest unloaded: one finite 2 560-wide row per
    /// `<|image_pad|>`. Nothing to run on a machine without a card.
    #[test]
    fn the_vision_guest_reads_an_image_from_its_ground() {
        let Ok(device) = Device::new_cuda(0) else {
            return;
        };
        vision_tower_pack(VISION_TOWER_MODE).unwrap();
        let (width, height) = (768u32, 512u32);
        let rgb = (0..(width * height * 3) as usize)
            .map(|i| (i * 31 % 251) as u8)
            .collect();
        let jobs = vec![GuestRequest::Vision(VisionRequest {
            images: vec![VisionImage { rgb, height, width }],
        })];
        let mut guest = VisionGuest::new(tower_pack, VISION_TOWER_MODE);
        let bytes = guest.footprint_bytes(&jobs);
        let Device::Cuda(cuda) = &device else {
            unreachable!()
        };
        let stream = cuda.cuda_stream();
        let held = stream.alloc_zeros::<u8>(bytes).unwrap();
        let (base, _guard) = held.device_ptr(&stream);
        let ground = Arc::new(Mutex::new(GuestGround::for_test(vec![base], bytes)));

        guest.load(&device, &ground, &jobs).unwrap();
        let out = guest.run(&jobs[0], &GuestSink::none());
        guest.unload();

        let GuestOutcome::Vision(rows) = out.unwrap() else {
            panic!("the vision guest answered another guest's outcome")
        };
        let tokens = (width as usize / 32) * (height as usize / 32);
        assert_eq!(rows.rows.len(), 1);
        assert_eq!(rows.rows[0].len(), tokens * 2560);
        assert!(
            rows.rows[0].iter().all(|v| v.is_finite()),
            "a row is not finite"
        );
        assert!(rows.rows[0].iter().any(|v| *v != 0.0), "every row is zero");
    }
}
