//! The Qwen3-VL vision tower: an image's patches in, one row per
//! `<|image_pad|>` out, at the text trunk's width.
//!
//! Read from the llama.cpp `mmproj` GGUF (`clip.projector_type =
//! qwen3vl_merger`), by its tensor names:
//!
//! | step | tensors |
//! |---|---|
//! | patch embed | `v.patch_embd.weight`, `.weight.1` (the Conv3d's two time slices), `.bias` |
//! | position table | `v.position_embd.weight`, 48 × 48 |
//! | 27 blocks | `v.blk.N.{ln1, attn_qkv, attn_out, ln2, ffn_up, ffn_down}` |
//! | merger | `v.post_ln`, `mm.0`, `mm.2` |
//!
//! Plain candle ops throughout, so the one forward is both the CPU reference a
//! test runs and the CUDA path a guest drain runs. Under a guest arena the
//! whole forward is one stage and each block a stage of the chain inside it
//! (`candle_nn::kv_cache::{guest_stage, guest_chain}`), so the arena holds one
//! block's intermediates at a time; with no arena open both are the identity.
//!
//! Design: `docs/flash_next_vision.md` §1.

use std::result::Result as StdResult;

use candle::quantized::Int8Mode;
use candle::{DType, Device, IndexOp, Module, Result, Tensor};
use candle_nn::kv_cache::{guest_chain, guest_stage};
use candle_nn::rotary_emb::rope;
use candle_nn::LayerNorm;

use super::geometry::{
    position_taps, vit_rope_angles, Grid, MERGE, PATCH, POS_SIDE, VIT_ROPE_FREQS,
};
use crate::models::z_image::attention::{banded, scale_of};
use crate::quantized_nn::{linear_b_mode, Linear};
use crate::quantized_var_builder::VarBuilder;

/// The tower's width.
pub const HIDDEN: usize = 1152;
/// Attention heads; the head is 72 wide.
pub const HEADS: usize = 16;
pub const HEAD_DIM: usize = HIDDEN / HEADS;
/// The MLP's inner width.
pub const INTERMEDIATE: usize = 4304;
/// Blocks.
pub const DEPTH: usize = 27;
/// The rows the merger writes: the text trunk's width.
pub const OUT_HIDDEN: usize = 2560;
/// Every LayerNorm's epsilon.
pub const EPS: f64 = 1e-6;
/// Values in one patch row: three channels of 16 × 16 pixels.
pub const PATCH_VALUES: usize = 3 * PATCH * PATCH;
/// Values in one merged row: four patches side by side.
const MERGED_VALUES: usize = HIDDEN * MERGE * MERGE;
/// Query rows one attention band covers. A band's scores and probabilities are
/// `2 · 16 · 512 · patches` bf16 — 32 KiB a patch, 512 MiB at the largest image
/// (16 384 patches) — against twice that at z_image's 1 024, for one more
/// launch pair per band on an image the size of a screen.
pub const ATTN_BAND: usize = 512;
/// Activation bytes per patch a forward needs at its peak: one block's
/// intermediates (the residual, its norm, `qkv`, the rotated heads and the
/// context, about 13 hidden widths; the MLP's 4 304, twice), the band's scores,
/// and the chain's two carried hidden states — about 110 KiB in bf16, with the
/// margin a carve's alignment and the merger's tail take.
pub const ACTIVATION_BYTES_PER_PATCH: usize = 128 * 1024;

/// One image, resized to its grid: RGB8, row-major, `grid.height · grid.width · 3`
/// bytes.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VisionInput {
    pub rgb: Vec<u8>,
    pub grid: Grid,
}

impl VisionInput {
    /// Refuse pixels that do not fill the grid — the one malformed input that
    /// would otherwise reach the card, after a drain had evicted for it.
    pub fn check(&self) -> StdResult<(), String> {
        let want = self.grid.height * self.grid.width * 3;
        if self.rgb.len() != want {
            return Err(format!(
                "an image of {}×{} is {want} RGB bytes, not {}",
                self.grid.width,
                self.grid.height,
                self.rgb.len()
            ));
        }
        if !self.grid.height.is_multiple_of(PATCH * MERGE)
            || !self.grid.width.is_multiple_of(PATCH * MERGE)
        {
            return Err(format!(
                "{}×{} is not a whole number of merge blocks",
                self.grid.width, self.grid.height
            ));
        }
        Ok(())
    }
}

/// A LayerNorm from a GGUF, its weight and bias in the activations' type.
fn layer_norm(vb: VarBuilder, dtype: DType) -> Result<LayerNorm> {
    let w = vb.get(HIDDEN, "weight")?.dequantize(vb.device())?;
    let b = vb.get(HIDDEN, "bias")?.dequantize(vb.device())?;
    // A conversion at load, once per vector, for the norm's own operands.
    Ok(LayerNorm::new(w.to_dtype(dtype)?, b.to_dtype(dtype)?, EPS))
}

/// One pre-norm block.
struct Block {
    ln1: LayerNorm,
    qkv: Linear,
    proj: Linear,
    ln2: LayerNorm,
    up: Linear,
    down: Linear,
}

impl Block {
    fn load(vb: VarBuilder, mode: Int8Mode, dtype: DType) -> Result<Self> {
        Ok(Self {
            ln1: layer_norm(vb.pp("ln1"), dtype)?,
            qkv: linear_b_mode(HIDDEN, 3 * HIDDEN, true, mode, dtype, vb.pp("attn_qkv"))?,
            proj: linear_b_mode(HIDDEN, HIDDEN, true, mode, dtype, vb.pp("attn_out"))?,
            ln2: layer_norm(vb.pp("ln2"), dtype)?,
            up: linear_b_mode(HIDDEN, INTERMEDIATE, true, mode, dtype, vb.pp("ffn_up"))?,
            down: linear_b_mode(INTERMEDIATE, HIDDEN, true, mode, dtype, vb.pp("ffn_down"))?,
        })
    }

    /// `x += attn(ln1(x)); x += mlp(ln2(x))` over one image's `[n, 1152]`.
    fn forward(&self, x: &Tensor, cos: &Tensor, sin: &Tensor) -> Result<Tensor> {
        let n = x.dim(0)?;
        let qkv = self
            .qkv
            .forward(&self.ln1.forward(x)?)?
            .reshape((n, 3, HEADS, HEAD_DIM))?;
        // [1, heads, n, 72], each contiguous for the rotary kernel.
        let part = |i: usize| -> Result<Tensor> {
            qkv.i((.., i))?.transpose(0, 1)?.unsqueeze(0)?.contiguous()
        };
        let q = rope(&part(0)?, cos, sin)?;
        let k = rope(&part(1)?, cos, sin)?;
        let v = part(2)?;
        let q = (q * scale_of(HEAD_DIM))?;
        let kt = k.transpose(2, 3)?.contiguous()?;
        // Non-causal, every patch of the image against every other.
        let ctx = banded(&q, &kt, &v, ATTN_BAND)?
            .squeeze(0)?
            .transpose(0, 1)?
            .contiguous()?
            .reshape((n, HIDDEN))?;
        let x = (x + self.proj.forward(&ctx)?)?;
        let h = self.up.forward(&self.ln2.forward(&x)?)?.gelu()?;
        x + self.down.forward(&h)?
    }
}

/// The tower, loaded.
pub struct VisionTower {
    /// The patch embedding's two time slices summed, `[768, 1152]` for
    /// `patches · W`.
    patch_w: Tensor,
    patch_b: Tensor,
    /// `[2304, 1152]`.
    pos: Tensor,
    blocks: Vec<Block>,
    post_ln: LayerNorm,
    fc1: Linear,
    fc2: Linear,
    dtype: DType,
}

impl VisionTower {
    /// The activations' type on `device`: bf16 on a card, as the reference
    /// runs the tower, and f32 on the CPU, which serves as the test reference.
    pub fn dtype_for(device: &Device) -> DType {
        if device.is_cuda() {
            DType::BF16
        } else {
            DType::F32
        }
    }

    /// Read the tower from an mmproj GGUF's tensors.
    pub fn load(vb: &VarBuilder, mode: Int8Mode) -> Result<Self> {
        let dev = vb.device().clone();
        let dtype = Self::dtype_for(&dev);
        // A still image is the same frame twice, so the Conv3d over time is the
        // two slices summed over one frame — and, stride being the kernel, a
        // plain matmul over the patch rows. Folded here, once.
        let w0 = vb.get_no_shape("v.patch_embd.weight")?.dequantize(&dev)?;
        let w1 = vb.get_no_shape("v.patch_embd.weight.1")?.dequantize(&dev)?;
        let patch_w = (w0 + w1)?
            .reshape((HIDDEN, PATCH_VALUES))?
            .t()?
            .contiguous()?
            .to_dtype(dtype)?;
        let patch_b = vb
            .get(HIDDEN, "v.patch_embd.bias")?
            .dequantize(&dev)?
            .to_dtype(dtype)?;
        let pos = vb
            .get((POS_SIDE * POS_SIDE, HIDDEN), "v.position_embd.weight")?
            .dequantize(&dev)?
            .to_dtype(dtype)?;
        let blocks = (0..DEPTH)
            .map(|i| Block::load(vb.pp(format!("v.blk.{i}")), mode, dtype))
            .collect::<Result<Vec<_>>>()?;
        Ok(Self {
            patch_w,
            patch_b,
            pos,
            blocks,
            post_ln: layer_norm(vb.pp("v.post_ln"), dtype)?,
            fc1: linear_b_mode(
                MERGED_VALUES,
                MERGED_VALUES,
                true,
                mode,
                dtype,
                vb.pp("mm.0"),
            )?,
            fc2: linear_b_mode(MERGED_VALUES, OUT_HIDDEN, true, mode, dtype, vb.pp("mm.2"))?,
            dtype,
        })
    }

    /// One image's rows, `[tokens, 2560]` in the activations' type, in raster
    /// order over the merged grid — the order of its `<|image_pad|>`s.
    pub fn forward(&self, image: &VisionInput) -> Result<Tensor> {
        image.check().map_err(candle::Error::Msg)?;
        let dev = self.pos.device().clone();
        let grid = image.grid;
        let pixels = Tensor::from_vec(image.rgb.clone(), (grid.height, grid.width, 3), &dev)?;
        guest_stage(&pixels, |pixels| self.encode(pixels, grid))
    }

    fn encode(&self, pixels: &Tensor, grid: Grid) -> Result<Tensor> {
        let dev = pixels.device().clone();
        let n = grid.patches();
        let (bh, bw) = (grid.merged_rows(), grid.merged_cols());
        // Normalised to (x/255 − 0.5)/0.5, then permuted into patch rows in
        // merge-block order, each row channel-major then pixel row then column.
        let patches = pixels
            .to_dtype(DType::F32)?
            .affine(1.0 / 127.5, -1.0)?
            .reshape(vec![bh, MERGE, PATCH, bw, MERGE, PATCH, 3])?
            .permute(vec![0, 3, 1, 4, 6, 2, 5])?
            .contiguous()?
            .reshape((n, PATCH_VALUES))?
            .to_dtype(self.dtype)?;
        let x = patches
            .matmul(&self.patch_w)?
            .broadcast_add(&self.patch_b)?;

        let (idx, w) = position_taps(grid);
        let idx = Tensor::from_vec(idx, n * 4, &dev)?;
        let w = Tensor::from_vec(w, (n, 4, 1), &dev)?.to_dtype(self.dtype)?;
        let pos = self
            .pos
            .index_select(&idx, 0)?
            .reshape((n, 4, HIDDEN))?
            .broadcast_mul(&w)?
            .sum(1)?;
        let x = (x + pos)?;

        let angles = Tensor::from_vec(vit_rope_angles(grid), (n, 2 * VIT_ROPE_FREQS), &dev)?;
        let cos = angles.cos()?.to_dtype(self.dtype)?;
        let sin = angles.sin()?.to_dtype(self.dtype)?;
        let x = guest_chain(&x, &self.blocks, |x, block| block.forward(x, &cos, &sin))?;

        let merged = self
            .post_ln
            .forward(&x)?
            .reshape((n / (MERGE * MERGE), MERGED_VALUES))?;
        let h = self.fc1.forward(&merged)?.gelu_erf()?;
        self.fc2.forward(&h)
    }

    /// The device the tower's weights are on.
    pub fn device(&self) -> &Device {
        self.pos.device()
    }

    /// The type of the rows [`Self::forward`] returns.
    pub fn dtype(&self) -> DType {
        self.dtype
    }
}

#[cfg(test)]
mod cpu_tests {
    use super::*;
    use crate::models::quantized_qwen38_moe::cached_vision_tower_file;

    /// **The real tower runs on the CPU**, the reference its card run is held
    /// to: a 256 × 256 image — 256 patches, one attention band — into one
    /// finite 2 560-wide row per `<|image_pad|>`. Nothing to run on a machine
    /// that has not fetched the tower.
    #[test]
    fn the_pinned_tower_reads_an_image_on_the_cpu() {
        let Some(path) = cached_vision_tower_file() else {
            return;
        };
        let dev = Device::Cpu;
        let vb = VarBuilder::from_gguf(&path, &dev).unwrap();
        let tower = VisionTower::load(&vb, Int8Mode::auto(&dev)).unwrap();
        let grid = Grid {
            height: 256,
            width: 256,
        };
        let rgb = (0..grid.height * grid.width * 3)
            .map(|i| (i * 31 % 251) as u8)
            .collect();
        let rows = tower.forward(&VisionInput { rgb, grid }).unwrap();
        assert_eq!(rows.dims(), [grid.tokens(), OUT_HIDDEN]);
        let values = rows.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        assert!(values.iter().all(|v| v.is_finite()), "a row is not finite");
        assert!(values.iter().any(|v| *v != 0.0), "every row is zero");
    }
}

#[cfg(all(test, feature = "cuda"))]
mod tests {
    use super::*;
    use crate::models::gpu_test_lock::gpu_serial;
    use crate::models::quantized_qwen38_moe::cached_vision_tower_file;
    use candle::cuda_backend::cudarc::driver::DevicePtr;
    use candle_nn::kv_cache::{close_guest_arena, open_guest_arena};

    /// The pinned tower on the card, or `None` on a machine without a card or
    /// without the tower fetched.
    fn tower() -> Option<(Device, VisionTower)> {
        let dev = Device::new_cuda(0).ok()?;
        let path = cached_vision_tower_file()?;
        let vb = VarBuilder::from_gguf(&path, &dev).unwrap();
        let tower = VisionTower::load(&vb, Int8Mode::auto(&dev)).unwrap();
        Some((dev, tower))
    }

    /// A 768 × 512 image: 1 536 patches, two attention bands and a ragged
    /// third.
    fn image() -> VisionInput {
        let grid = Grid {
            height: 512,
            width: 768,
        };
        let rgb = (0..grid.height * grid.width * 3)
            .map(|i| (i * 31 % 251) as u8)
            .collect();
        VisionInput { rgb, grid }
    }

    /// One finite, non-zero 2 560-wide row per `<|image_pad|>`.
    fn assert_rows(rows: &Tensor, grid: Grid) {
        assert_eq!(rows.dims(), [grid.tokens(), OUT_HIDDEN]);
        let values = rows
            .to_dtype(DType::F32)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        assert!(values.iter().all(|v| v.is_finite()), "a row is not finite");
        assert!(values.iter().any(|v| *v != 0.0), "every row is zero");
    }

    /// **The real tower runs on the card**: Flash-Next's pinned mmproj reads
    /// an image into its rows. On a machine that has not fetched the tower
    /// there is nothing to run.
    #[test]
    fn the_pinned_tower_reads_an_image_on_the_card() {
        let _gpu = gpu_serial();
        let Some((_, tower)) = tower() else { return };
        let input = image();
        let grid = input.grid;
        assert_rows(&tower.forward(&input).unwrap(), grid);
    }

    /// **And on a guest arena**, sized as the vision guest sizes it: the
    /// image's patches at [`ACTIVATION_BYTES_PER_PATCH`] — the way the drain
    /// runs it, every stage and block carving from that one stretch.
    #[test]
    fn the_pinned_tower_reads_an_image_on_a_guest_arena() {
        let _gpu = gpu_serial();
        let Some((dev, tower)) = tower() else { return };
        let input = image();
        let grid = input.grid;
        let bytes = grid.patches() * ACTIVATION_BYTES_PER_PATCH;
        let Device::Cuda(cuda) = &dev else {
            unreachable!()
        };
        let stream = cuda.cuda_stream();
        let ground = stream.alloc_zeros::<u8>(bytes).unwrap();
        let (base, _guard) = ground.device_ptr(&stream);
        open_guest_arena(&stream, base, bytes);
        let rows = tower.forward(&input);
        close_guest_arena(&stream);
        assert_rows(&rows.unwrap(), grid);
    }
}
