//! W4A16 → Q4_KO expert import: rewrite the pinned Q8_0 split with the routed
//! experts replaced by AWQ-calibrated 4-bit weights in the KO matmul layout.
//!
//! The structural fact this rests on (`docs/qwen38_flash_next.md` §4, and the
//! reason no Q4-class *download* can be loaded directly): compressed-tensors
//! `W4A16 group_size=128 symmetric` is the **same quantization** as `Q4_KO` —
//! int4, one scale per 128 along K. So the import is a re-layout, not a
//! requant: unpack the source's own codes (`nibble − 8`), carry its own scale
//! as `(scale, min = −8·scale)`, and pack with [`pack_q4_ko`], which bypasses
//! `quantize_ko`'s min-max observer precisely so no value is re-rounded. The
//! result dequantizes bit-for-bit to what the AWQ checkpoint dequantizes to —
//! asserted per expert during conversion, not assumed.
//!
//! Output: a sibling GGUF split (`…-Q4KOEXP-…`). Shards that carry no expert
//! tensors are hard-linked (same bytes, no copy); shards that do are rewritten
//! streaming — every non-expert tensor byte-copied, every
//! `ffn_{gate,up,down}_exps` re-emitted as Q4_KO. Metadata passes through
//! verbatim, so the converted split loads through the same `GgufModel` path.
//!
//! Source layout facts, pinned from `compressed-tensors`'s `pack_to_int32`
//! (vllm-project, `pack_quantized/helpers.py`): int4 codes are offset by +8 to
//! unsigned, packed 8 per i32 along the **input** dimension, element `i` at
//! bits `[4i, 4i+4)` of its word — little-endian nibble order, no cross-word
//! splits at 4 bits. `weight_scale` is `[out, in/128]` in the model dtype
//! (BF16 here); a BF16 scale in the f16 normal range converts to the KO f16
//! `(scale, min)` pair exactly, and the converter refuses one that does not.

use std::collections::HashMap;
use std::fs::File;
use std::io::{BufWriter, Seek, Write};
use std::path::{Path, PathBuf};

use byteorder::{LittleEndian, WriteBytesExt};
use candle::cuda_backend::cudarc::driver::DevicePtr;
use candle::cuda_backend::WrapErr;
use candle::quantized::gguf_file::Content;
use candle::quantized::ko_quant::{pack_q4_ko, unpack_q4_ko};
use candle::quantized::GgmlDType;
use candle::{Device, Result};
use candle_kernels::simple::w4a16_repack::run_w4a16_repack_q4ko;
use half::{bf16, f16};
use rayon::prelude::*;

/// One expert matrix decoded from the W4A16 source: the packed codes shifted
/// to unsigned and the per-group affine pairs — exactly the bytes-of-record
/// the KO pack stores, so verification compares these rather than
/// materialising two 19.6 MB F32 dequantizations per expert (which was the
/// first cut's dominant cost; see `convert_bench`).
pub struct DecodedExpert {
    pub codes: Vec<u8>,
    pub dm: Vec<(f32, f32)>,
    pub nrows: usize,
    pub ncols: usize,
}

/// Decode a compressed-tensors `pack-quantized` int4 pair from raw bytes:
/// `packed` is the I32 words tensor (`[nrows, ncols/8]`, 8 nibbles per word,
/// little-endian nibble order, +8 offset already applied), `scales` the BF16
/// `[nrows, ncols/128]` group scales. Split out of [`W4a16Source::decode`] so
/// the conversion harness (`convert_bench`) drives the same code over
/// synthetic buffers.
pub fn decode_packed(
    packed: &[u8],
    scales: &[u8],
    nrows: usize,
    ncols: usize,
    what: &str,
) -> Result<DecodedExpert> {
    let k_groups = ncols / 128;
    if packed.len() != nrows * ncols / 2 || scales.len() != nrows * k_groups * 2 {
        candle::bail!(
            "w4a16 {what}: {} packed / {} scale bytes for [{nrows}, {ncols}]",
            packed.len(),
            scales.len()
        );
    }

    // Scales: BF16 → F32, and the f16 exactness gate. A scale outside the
    // f16 normal range would make the KO store lossy — refuse rather than
    // silently degrade the "bit-exact import" contract.
    let mut dm = Vec::with_capacity(nrows * k_groups);
    for (gi, sb) in scales.as_chunks::<2>().0.iter().enumerate() {
        let s = bf16::from_le_bytes(*sb).to_f32();
        let s16 = f16::from_f32(s).to_f32();
        if s16 != s {
            candle::bail!(
                "w4a16 {what}: scale[{gi}] = {s:e} is not f16-exact — the KO store \
                 would round it and the import would no longer be lossless"
            );
        }
        let mn = -8.0 * s;
        debug_assert_eq!(f16::from_f32(mn).to_f32(), mn);
        dm.push((s, mn));
    }

    // Codes: nibble i of word w is element 8w+i, +8 offset already in the
    // file — exactly the unsigned code Q4_KO stores. Chunked iteration keeps
    // the inner eight stores bounds-check-free.
    let mut codes = vec![0u8; nrows * ncols];
    for (cs, wb) in codes
        .as_chunks_mut::<8>()
        .0
        .iter_mut()
        .zip(packed.as_chunks::<4>().0.iter())
    {
        let word = u32::from_le_bytes(*wb);
        for (i, c) in cs.iter_mut().enumerate() {
            *c = ((word >> (4 * i)) & 0xF) as u8;
        }
    }

    Ok(DecodedExpert {
        codes,
        dm,
        nrows,
        ncols,
    })
}

/// Where one tensor's raw bytes live inside the mmapped shard set.
struct TensorSpan {
    shard: usize,
    start: usize,
    len: usize,
    dtype: String,
    shape: Vec<usize>,
}

/// Read-only view over the AWQ safetensors shards: name → raw byte span.
pub struct W4a16Source {
    maps: Vec<memmap2::Mmap>,
    index: HashMap<String, TensorSpan>,
}

impl W4a16Source {
    /// Open the shard files (any subset that covers the tensors the caller
    /// will ask for).
    pub fn open(paths: &[PathBuf]) -> Result<Self> {
        let mut maps = Vec::with_capacity(paths.len());
        let mut index = HashMap::new();
        for (si, p) in paths.iter().enumerate() {
            let file = File::open(p)?;
            let map = unsafe { memmap2::Mmap::map(&file)? };
            // Record raw byte spans while the parse borrows the mmap; the
            // spans are plain offsets, so they survive the borrow ending.
            let entries: Vec<(String, TensorSpan)> = {
                let st = safetensors::SafeTensors::deserialize(&map)
                    .map_err(|e| candle::Error::Msg(format!("safetensors {p:?}: {e}")))?;
                let base = map.as_ptr() as usize;
                st.tensors()
                    .into_iter()
                    .map(|(name, view)| {
                        let start = view.data().as_ptr() as usize - base;
                        (
                            name.to_string(),
                            TensorSpan {
                                shard: si,
                                start,
                                len: view.data().len(),
                                dtype: format!("{:?}", view.dtype()),
                                shape: view.shape().to_vec(),
                            },
                        )
                    })
                    .collect()
            };
            for (name, entry) in entries {
                index.insert(name, entry);
            }
            maps.push(map);
        }
        Ok(Self { maps, index })
    }

    fn bytes(&self, name: &str) -> Result<(&[u8], &str, &[usize])> {
        let t = self
            .index
            .get(name)
            .ok_or_else(|| candle::Error::Msg(format!("w4a16: missing tensor {name}")))?;
        Ok((
            &self.maps[t.shard][t.start..t.start + t.len],
            &t.dtype,
            &t.shape,
        ))
    }

    /// Decode one quantized linear (`{prefix}.weight_packed` + `.weight_scale`)
    /// to codes + affine pairs.
    fn decode(&self, prefix: &str) -> Result<DecodedExpert> {
        let (packed, pdt, pshape) = self.bytes(&format!("{prefix}.weight_packed"))?;
        let (scales, sdt, sshape) = self.bytes(&format!("{prefix}.weight_scale"))?;
        if pdt != "I32" {
            candle::bail!("w4a16 {prefix}: weight_packed is {pdt}, expected I32");
        }
        if sdt != "BF16" {
            candle::bail!("w4a16 {prefix}: weight_scale is {sdt}, expected BF16");
        }
        let (nrows, words) = (pshape[0], pshape[1]);
        let ncols = words * 8;
        if sshape != [nrows, ncols / 128] {
            candle::bail!(
                "w4a16 {prefix}: scale shape {sshape:?} does not pair with packed {pshape:?}"
            );
        }
        decode_packed(packed, scales, nrows, ncols, prefix)
    }
}

/// The HF name of expert `e`'s `proj` in layer `li`.
fn hf_expert_prefix(li: usize, e: usize, proj: &str) -> String {
    format!("model.language_model.layers.{li}.mlp.experts.{e}.{proj}_proj")
}

/// Map a GGUF expert-tensor name to its HF projection, or `None` for a
/// pass-through tensor.
fn expert_proj_of(gguf_name: &str) -> Option<(usize, &'static str)> {
    let rest = gguf_name.strip_prefix("blk.")?;
    let dot = rest.find('.')?;
    let li: usize = rest[..dot].parse().ok()?;
    let proj = match &rest[dot..] {
        ".ffn_gate_exps.weight" => "gate",
        ".ffn_up_exps.weight" => "up",
        ".ffn_down_exps.weight" => "down",
        _ => return None,
    };
    Some((li, proj))
}

/// Convert one expert to its Q4_KO image, verifying bit-exactness against the
/// source's own dequantization.
fn convert_expert(src: &W4a16Source, li: usize, e: usize, proj: &str) -> Result<Vec<u8>> {
    let d = src.decode(&hf_expert_prefix(li, e, proj))?;
    let packed = pack_q4_ko(&d.codes, &d.dm, d.nrows, d.ncols);
    verify_packed(&packed, &d, &format!("blk.{li} expert {e} {proj}"))?;
    Ok(packed)
}

/// Repack a whole expert tensor on the GPU — one fused launch over every
/// expert (`simple/w4a16_repack.cu`), a pure byte permutation. Returns the
/// concatenated Q4_KO images plus the kernel's f16-exactness violation count
/// (nonzero refuses the conversion, matching the CPU importer's per-group
/// bail).
///
/// Bytes per expert: `words` is `nrows·ncols/2`, `scales` `nrows·ncols/64`,
/// output `nrows·ncols/1024 · 544`.
pub fn gpu_repack_tensor(
    device: &Device,
    words: &[u8],
    scales: &[u8],
    n_experts: usize,
    nrows: usize,
    ncols: usize,
) -> Result<Vec<u8>> {
    let Device::Cuda(dev) = device else {
        candle::bail!("gpu_repack_tensor: the W4A16 import runs on a CUDA device");
    };
    if words.len() != n_experts * nrows * ncols / 2
        || scales.len() != n_experts * nrows * (ncols / 128) * 2
    {
        candle::bail!(
            "gpu_repack_tensor: {} word / {} scale bytes for {n_experts} × [{nrows}, {ncols}]",
            words.len(),
            scales.len()
        );
    }
    let out_bytes = n_experts * (nrows / 8) * (ncols / 128) * 544;
    let stream = dev.cuda_stream();
    let words_gpu = stream.memcpy_stod(words).w()?;
    let scales_gpu = stream.memcpy_stod(scales).w()?;
    let violations_gpu = stream.memcpy_stod(&[0i32]).w()?;
    // Fully overwritten by the kernel — allocate uninitialised.
    let out_gpu = unsafe { dev.alloc::<u8>(out_bytes)? };
    {
        let (wp, _gw) = words_gpu.device_ptr(&stream);
        let (sp, _gs) = scales_gpu.device_ptr(&stream);
        let (op, _go) = out_gpu.device_ptr(&stream);
        let (vp, _gv) = violations_gpu.device_ptr(&stream);
        unsafe {
            run_w4a16_repack_q4ko(
                wp as *const u32,
                sp as *const u16,
                op as *mut u8,
                vp as *mut i32,
                n_experts as i32,
                nrows as i32,
                ncols as i32,
                stream.cu_stream() as *mut std::ffi::c_void,
            );
        }
    }
    let mut out = vec![0u8; out_bytes];
    let mut violations = [0i32];
    stream.memcpy_dtoh(&out_gpu, &mut out[..]).w()?;
    stream
        .memcpy_dtoh(&violations_gpu, &mut violations[..])
        .w()?;
    stream.synchronize().w()?;
    if violations[0] != 0 {
        candle::bail!(
            "gpu_repack_tensor: {} scale groups are not f16-exact — the KO store \
             would round them and the import would no longer be lossless",
            violations[0]
        );
    }
    Ok(out)
}

/// The per-expert losslessness gate: the packed bytes must recover the
/// source's exact codes and `(scale, min)` pairs.
///
/// Byte-equivalent to comparing full dequantizations — `dequant_ko` is
/// precisely the affine `scale·code + min` over these bytes, pinned by the
/// `pack_q4_ko` unit tests against the untouched `dequant_ko` — but without
/// materialising two 19.6 MB F32 buffers per expert, which was over half the
/// first cut's conversion time (`convert_bench` has the measurements).
pub(crate) fn verify_packed(packed: &[u8], d: &DecodedExpert, what: &str) -> Result<()> {
    let (codes2, dm2) = unpack_q4_ko(packed, d.nrows, d.ncols);
    if codes2 != d.codes {
        let i = codes2
            .iter()
            .zip(d.codes.iter())
            .position(|(a, b)| a != b)
            .unwrap_or(0);
        candle::bail!(
            "w4a16 import {what}: code {i} round-tripped to {} against source {} — \
             the import is not lossless",
            codes2[i],
            d.codes[i]
        );
    }
    if dm2 != d.dm {
        let i = dm2
            .iter()
            .zip(d.dm.iter())
            .position(|(a, b)| a != b)
            .unwrap_or(0);
        candle::bail!(
            "w4a16 import {what}: group {i} (scale, min) stored as {:?} against source \
             {:?} — the import is not lossless",
            dm2[i],
            d.dm[i]
        );
    }
    Ok(())
}

/// GGUF v3 string write.
fn write_string<W: Write>(w: &mut W, s: &str) -> Result<()> {
    w.write_u64::<LittleEndian>(s.len() as u64)?;
    w.write_all(s.as_bytes())?;
    Ok(())
}

fn pad32(size: usize) -> usize {
    31 - (31 + size) % 32
}

/// Rewrite one GGUF shard: expert tensors re-emitted as Q4_KO from `src`,
/// everything else byte-copied; metadata verbatim.
fn rewrite_shard(
    src_path: &Path,
    dst_path: &Path,
    awq: &W4a16Source,
    device: &Device,
) -> Result<usize> {
    let mut f = File::open(src_path)?;
    let content = Content::read(&mut f)?;

    // Deterministic tensor order: by source offset, so the pass-through reads
    // are sequential.
    let mut names: Vec<&String> = content.tensor_infos.keys().collect();
    names.sort_by_key(|n| content.tensor_infos[*n].offset);

    // Sizes first, so the directory can be written before any data.
    let mut out_sizes = Vec::with_capacity(names.len());
    for name in &names {
        let info = &content.tensor_infos[*name];
        let bytes = if expert_proj_of(name).is_some() {
            info.shape.elem_count() / GgmlDType::Q4_KO.block_size() * GgmlDType::Q4_KO.type_size()
        } else {
            info.shape.elem_count() / info.ggml_dtype.block_size() * info.ggml_dtype.type_size()
        };
        out_sizes.push(bytes);
    }

    let out = File::create(dst_path)?;
    let mut w = BufWriter::with_capacity(1 << 20, out);
    w.write_u32::<LittleEndian>(0x4655_4747)?; // GGUF
    w.write_u32::<LittleEndian>(2)?; // version 2, as candle's own writer emits
    w.write_u64::<LittleEndian>(names.len() as u64)?;
    w.write_u64::<LittleEndian>(content.metadata.len() as u64)?;
    // Metadata verbatim (sorted for determinism; GGUF readers key by name).
    let mut md: Vec<_> = content.metadata.iter().collect();
    md.sort_by(|a, b| a.0.cmp(b.0));
    for (key, value) in md {
        write_string(&mut w, key)?;
        w.write_u32::<LittleEndian>(value.value_type().to_u32())?;
        value.write(&mut w)?;
    }
    // Tensor directory with recomputed offsets.
    let mut offset = 0usize;
    for (name, bytes) in names.iter().zip(&out_sizes) {
        let info = &content.tensor_infos[*name];
        write_string(&mut w, name)?;
        let dims = info.shape.dims();
        w.write_u32::<LittleEndian>(dims.len() as u32)?;
        for &d in dims.iter().rev() {
            w.write_u64::<LittleEndian>(d as u64)?;
        }
        let dtype = if expert_proj_of(name).is_some() {
            GgmlDType::Q4_KO
        } else {
            info.ggml_dtype
        };
        w.write_u32::<LittleEndian>(dtype.to_gguf_file_code())?;
        w.write_u64::<LittleEndian>(offset as u64)?;
        offset += bytes + pad32(*bytes);
    }
    // Align the data section.
    let pos = w.stream_position()? as usize;
    w.write_all(&vec![0u8; pad32(pos)])?;

    // Data: pass-through copies from the source, expert tensors converted on
    // the GPU — one fused launch over all 512 experts of a tensor, one tensor
    // in memory at a time.
    //
    // Verification splits by failure mode. The scale f16-exactness gate is
    // **data-dependent** and covers every group: the kernel counts violations
    // and `gpu_repack_tensor` refuses on any. The permutation is
    // **data-independent**, so byte-identity against the CPU reference
    // (`convert_expert`, which carries its own losslessness check) on a
    // sample of experts per tensor pins the layout without re-running the
    // whole tensor on the host — a wrong permutation garbles every expert,
    // not an unlucky one.
    let src_map = unsafe { memmap2::Mmap::map(&File::open(src_path)?)? };
    let data_base = content.tensor_data_offset as usize;
    let mut converted = 0usize;
    for (name, bytes) in names.iter().zip(&out_sizes) {
        let info = &content.tensor_infos[*name];
        match expert_proj_of(name) {
            Some((li, proj)) => {
                let n_experts = info.shape.dims()[0];
                let (nrows, ncols) = (info.shape.dims()[1], info.shape.dims()[2]);
                // Gather every expert's packed words + scales contiguously.
                let mut words = Vec::with_capacity(n_experts * nrows * ncols / 2);
                let mut scales = Vec::with_capacity(n_experts * nrows * (ncols / 128) * 2);
                for e in 0..n_experts {
                    let prefix = hf_expert_prefix(li, e, proj);
                    let (p, _, pshape) = awq.bytes(&format!("{prefix}.weight_packed"))?;
                    if pshape != [nrows, ncols / 8] {
                        candle::bail!(
                            "w4a16 import {name}: expert {e} packed shape {pshape:?} \
                             against gguf [{nrows}, {ncols}]"
                        );
                    }
                    let (s, _, _) = awq.bytes(&format!("{prefix}.weight_scale"))?;
                    words.extend_from_slice(p);
                    scales.extend_from_slice(s);
                }
                let out = gpu_repack_tensor(device, &words, &scales, n_experts, nrows, ncols)?;
                if out.len() != *bytes {
                    candle::bail!(
                        "w4a16 import {name}: emitted {} bytes, directory says {bytes}",
                        out.len()
                    );
                }
                // Sampled byte-identity against the CPU reference path.
                let per = out.len() / n_experts;
                let sample: Vec<usize> = (0..n_experts).step_by(32).collect();
                sample.into_par_iter().try_for_each(|e| -> Result<()> {
                    let cpu = convert_expert(awq, li, e, proj)?;
                    if cpu.as_slice() != &out[e * per..(e + 1) * per] {
                        candle::bail!(
                            "w4a16 import {name}: expert {e} GPU repack diverges from the \
                             CPU reference bytes"
                        );
                    }
                    Ok(())
                })?;
                w.write_all(&out)?;
                converted += n_experts;
            }
            None => {
                let src_bytes = info.shape.elem_count() / info.ggml_dtype.block_size()
                    * info.ggml_dtype.type_size();
                let start = data_base + info.offset as usize;
                w.write_all(&src_map[start..start + src_bytes])?;
            }
        }
        w.write_all(&vec![0u8; pad32(*bytes)])?;
    }
    w.flush()?;
    Ok(converted)
}

/// Merge a GGUF split into ONE engine file.
///
/// The production loader wants a single artifact twice over: the expert cache
/// consumes one `Content` + one mmap (`ExpertCacheSetup`), and the expert
/// tensors of a split live across several shards. This is the same prepare
/// step DeepSeek's merged `MXFP4_KO` file is. Metadata comes from the split's
/// first shard **minus the `split.*` keys** (which would lie about the merged
/// artifact); tensors stream verbatim in shard order, offset-sorted within
/// each shard.
///
/// Resumable the same way the converter is: written to a `.tmp` sibling and
/// renamed on success, so an existing final name is always a complete file.
pub fn merge_gguf_split(splits: &[PathBuf], dst: &Path) -> Result<PathBuf> {
    merge_gguf_files(splits, dst, &[], &Verbatim)
}

/// A per-tensor transform applied while a merge streams its data.
///
/// The directory is written before any data, so a rewrite first declares the
/// dtype a tensor will be written as ([`TensorRewrite::target`]) — its length
/// follows from the dtype's block geometry — and then produces exactly that many
/// bytes ([`TensorRewrite::emit`]), which the merge checks.
pub trait TensorRewrite {
    /// The dtype `name` is written as, or `None` to copy its bytes verbatim.
    fn target(&self, name: &str, dims: &[usize], dtype: GgmlDType) -> Option<GgmlDType>;

    /// `name`'s bytes re-encoded as `target`, from its raw source bytes.
    fn emit(
        &self,
        name: &str,
        dims: &[usize],
        dtype: GgmlDType,
        src: &[u8],
        target: GgmlDType,
    ) -> Result<Vec<u8>>;
}

/// The rewrite that rewrites nothing.
pub struct Verbatim;

impl TensorRewrite for Verbatim {
    fn target(&self, _: &str, _: &[usize], _: GgmlDType) -> Option<GgmlDType> {
        None
    }

    fn emit(
        &self,
        name: &str,
        _: &[usize],
        _: GgmlDType,
        _: &[u8],
        _: GgmlDType,
    ) -> Result<Vec<u8>> {
        candle::bail!("Verbatim rewrite asked to emit {name}, which it never targets")
    }
}

/// Byte length of a tensor of `elems` elements stored as `dtype`.
fn stored_bytes(elems: usize, dtype: GgmlDType) -> usize {
    elems / dtype.block_size() * dtype.type_size()
}

/// Whether `name` is a routed-expert projection (`blk.N.ffn_{gate,up,down}_exps.weight`).
pub fn is_expert_tensor(name: &str) -> bool {
    expert_proj_of(name).is_some()
}

/// [`merge_gguf_split`] over files that are not all shards of one split.
///
/// Two differences, both needed to fold a **draft head** into the engine
/// artifact — the head ships as its own GGUF, and `qwen35::mtp`'s reasoning is
/// that a NextN head "is a layer of the model, not a sidecar": as `blk.48` of a
/// 49-block file it holds its KV in the same paged cache, and the expert cache
/// covers its 512 experts by construction rather than by learning about a
/// second file.
///
/// * **Duplicate tensor names are dropped, first file wins.** A self-contained
///   head's GGUF carries its own `token_embd` / `output` so it can run
///   standalone, but `mtp_use_dedicated_embeddings: false` says those *are* the
///   trunk's — keeping both would write ~1.35 GiB twice and leave the loader
///   two answers for one name.
/// * **`overrides` replace metadata keys**, because the merged file's geometry
///   is not the first shard's: `block_count` gains the head's block and
///   `nextn_predict_layers` announces it, which is what makes the shared config
///   parser split trunk from head (`num_layers = block_count − nextn`).
///
/// `rewrite` re-encodes the tensors it targets as they stream — which is how a
/// requantized-expert artifact is written in one pass over the split, with no
/// intermediate converted copy on disk.
pub fn merge_gguf_files(
    splits: &[PathBuf],
    dst: &Path,
    overrides: &[(String, candle::quantized::gguf_file::Value)],
    rewrite: &dyn TensorRewrite,
) -> Result<PathBuf> {
    if dst.exists() {
        return Ok(dst.to_path_buf());
    }
    // Per shard: the parsed directory plus each tensor's absolute data start,
    // offset-sorted so the copy below is one forward pass per file.
    struct ShardDir {
        path: PathBuf,
        data_base: u64,
        tensors: Vec<MergedTensor>,
    }
    /// One tensor as read from its shard and as written to the merged file.
    struct MergedTensor {
        name: String,
        /// Outermost-first.
        dims: Vec<usize>,
        dtype: GgmlDType,
        src_offset: u64,
        src_bytes: usize,
        out_dtype: GgmlDType,
        out_bytes: usize,
    }
    let mut shards = Vec::with_capacity(splits.len());
    let mut metadata: Option<Vec<(String, candle::quantized::gguf_file::Value)>> = None;
    let mut seen: std::collections::HashSet<String> = std::collections::HashSet::new();
    for path in splits {
        let mut f = File::open(path)?;
        let content = Content::read(&mut f)?;
        if metadata.is_none() {
            let mut md: Vec<_> = content
                .metadata
                .iter()
                .filter(|(k, _)| !k.starts_with("split."))
                .map(|(k, v)| (k.clone(), v.clone()))
                .collect();
            md.sort_by(|a, b| a.0.cmp(&b.0));
            metadata = Some(md);
        }
        let mut tensors: Vec<MergedTensor> = content
            .tensor_infos
            .iter()
            .map(|(name, info)| {
                let dims = info.shape.dims().to_vec();
                let elems = info.shape.elem_count();
                let out_dtype = rewrite
                    .target(name, &dims, info.ggml_dtype)
                    .unwrap_or(info.ggml_dtype);
                MergedTensor {
                    name: name.clone(),
                    dims,
                    dtype: info.ggml_dtype,
                    src_offset: info.offset,
                    src_bytes: stored_bytes(elems, info.ggml_dtype),
                    out_dtype,
                    out_bytes: stored_bytes(elems, out_dtype),
                }
            })
            .collect();
        tensors.sort_by_key(|t| t.src_offset);
        // First file wins on a repeated name — see the doc on
        // [`merge_gguf_files`]. Filtering here rather than at write time keeps
        // the directory and the data pass counting the same tensors, which is
        // the invariant the recomputed offsets rest on.
        tensors.retain(|t| seen.insert(t.name.clone()));
        shards.push(ShardDir {
            path: path.clone(),
            data_base: content.tensor_data_offset,
            tensors,
        });
    }
    let mut metadata = metadata.ok_or_else(|| candle::Error::Msg("merge: empty split".into()))?;
    for (key, value) in overrides {
        match metadata.iter_mut().find(|(k, _)| k == key) {
            Some(slot) => slot.1 = value.clone(),
            None => metadata.push((key.clone(), value.clone())),
        }
    }
    metadata.sort_by(|a, b| a.0.cmp(&b.0));
    let n_tensors: usize = shards.iter().map(|s| s.tensors.len()).sum();

    let tmp = dst.with_extension("gguf.tmp");
    let out = File::create(&tmp)?;
    let mut w = BufWriter::with_capacity(1 << 22, out);
    w.write_u32::<LittleEndian>(0x4655_4747)?; // GGUF
    w.write_u32::<LittleEndian>(2)?;
    w.write_u64::<LittleEndian>(n_tensors as u64)?;
    w.write_u64::<LittleEndian>(metadata.len() as u64)?;
    for (key, value) in &metadata {
        write_string(&mut w, key)?;
        w.write_u32::<LittleEndian>(value.value_type().to_u32())?;
        value.write(&mut w)?;
    }
    // Directory with recomputed offsets, in the exact order the data section
    // will stream.
    let mut offset = 0usize;
    for shard in &shards {
        for t in &shard.tensors {
            write_string(&mut w, &t.name)?;
            w.write_u32::<LittleEndian>(t.dims.len() as u32)?;
            for &d in t.dims.iter().rev() {
                w.write_u64::<LittleEndian>(d as u64)?;
            }
            w.write_u32::<LittleEndian>(t.out_dtype.to_gguf_file_code())?;
            w.write_u64::<LittleEndian>(offset as u64)?;
            offset += t.out_bytes + pad32(t.out_bytes);
        }
    }
    let pos = w.stream_position()? as usize;
    w.write_all(&vec![0u8; pad32(pos)])?;

    // Data: one forward mmap pass per shard, 256 MiB windows so the page cache
    // is streamed through rather than filled. A rewritten tensor is emitted
    // whole and must be exactly the length its directory entry promised.
    for shard in &shards {
        let map = unsafe { memmap2::Mmap::map(&File::open(&shard.path)?)? };
        for t in &shard.tensors {
            let start = shard.data_base as usize + t.src_offset as usize;
            let src = &map[start..start + t.src_bytes];
            if t.out_dtype != t.dtype {
                let out = rewrite.emit(&t.name, &t.dims, t.dtype, src, t.out_dtype)?;
                if out.len() != t.out_bytes {
                    candle::bail!(
                        "merge: {} rewritten to {} bytes, directory says {}",
                        t.name,
                        out.len(),
                        t.out_bytes
                    );
                }
                w.write_all(&out)?;
            } else {
                for window in src.chunks(256 << 20) {
                    w.write_all(window)?;
                }
            }
            w.write_all(&vec![0u8; pad32(t.out_bytes)])?;
        }
    }
    w.flush()?;
    drop(w);
    std::fs::rename(&tmp, dst)?;
    Ok(dst.to_path_buf())
}

/// Rewrite the MTP draft head's GGUF into the form the engine artifact takes it
/// in: experts at the trunk's width, and no duplicate of anything the trunk
/// already has.
///
/// # Why the experts must match the trunk exactly
///
/// The head is merged in as `blk.{num_layers}` and its 512 experts join the same
/// grid as every trunk layer's — which is the whole benefit, because they then
/// stream over PCIe and offload through the same three tiers instead of sitting
/// resident. But the zone's slots are **uniformly sized to the widest layer**
/// (`expert_lre::slot_bytes_for` takes a `max` over every layer's geometry), so
/// a layer's width is not a local choice:
///
/// * **Wider** than the trunk — leaving these at the sidecar's `Q8_0` — inflates
///   every slot in the grid, 49 × 512 of them, roughly doubling the zone's
///   footprint for one block's benefit and halving how many experts stay
///   resident.
/// * **Narrower** frees nothing: a narrow expert simply under-fills a slot sized
///   for the widest. It buys PCIe bytes per draft step, not residency.
///
/// So the head takes the trunk's width — `experts`, whatever the rung chose — and
/// is requantized from `Q8_0` to it on the GPU
/// ([`super::prepare::requant::requant_experts`]). That cost is confined to
/// *proposals* — a worse draft is a rejected token, never a wrong one, because
/// the target's own logits decide — and the alternative halves trunk residency.
///
/// # What is dropped
///
/// The self-contained head ships `token_embd` and `output` so it can run
/// standalone, but `mtp_use_dedicated_embeddings: false` says those *are* the
/// trunk's. Writing them again would cost ~1.35 GiB and leave the loader two
/// answers for one tensor name; the merge would drop the second copy anyway.
///
/// Written to a `.tmp` sibling and renamed on success, so an existing
/// `dst_path` is always a complete file.
#[cfg(feature = "cuda")]
pub fn convert_mtp_sidecar(
    src_path: &Path,
    dst_path: &Path,
    experts: GgmlDType,
    device: &Device,
) -> Result<usize> {
    use super::prepare::requant::requant_experts;

    if dst_path.exists() {
        return Ok(0);
    }
    let mut f = File::open(src_path)?;
    let content = Content::read(&mut f)?;

    // **Only the embedding and the LM head are the trunk's**, which is what
    // `mtp_use_dedicated_embeddings: false` promises; the head's own output
    // mixer is published under its own block (`nextn.hc_head_*`), so every
    // other tensor keeps the name it came with.
    let is_trunks = |n: &str| n == "token_embd.weight" || n == "output.weight";

    let mut names: Vec<&String> = content
        .tensor_infos
        .keys()
        .filter(|n| !is_trunks(n))
        .collect();
    names.sort_by_key(|n| content.tensor_infos[*n].offset);

    // Requantize the experts up front: the directory needs their output sizes
    // before any data is written. Each one is its own `[out, in]` matrix — the
    // KO layout is per matrix, so the `[n_expert, out, in]` tensor is converted
    // expert by expert (`requant_experts`).
    let map = unsafe { memmap2::Mmap::map(&File::open(src_path)?)? };
    let data_base = content.tensor_data_offset as usize;
    let mut converted: std::collections::HashMap<String, Vec<u8>> =
        std::collections::HashMap::new();
    // An expert already at the target width is copied like any other tensor.
    for name in names
        .iter()
        .filter(|n| expert_proj_of(n).is_some() && content.tensor_infos[**n].ggml_dtype != experts)
    {
        let info = &content.tensor_infos[*name];
        let start = data_base + info.offset as usize;
        let len = stored_bytes(info.shape.elem_count(), info.ggml_dtype);
        let image = requant_experts(
            &map[start..start + len],
            info.shape.dims(),
            info.ggml_dtype,
            experts,
            device,
        )?;
        converted.insert((*name).clone(), image);
    }

    let out_size = |name: &str| -> usize {
        match converted.get(name) {
            Some(b) => b.len(),
            None => {
                let info = &content.tensor_infos[name];
                stored_bytes(info.shape.elem_count(), info.ggml_dtype)
            }
        }
    };
    let out_sizes: Vec<usize> = names.iter().map(|n| out_size(n)).collect();

    let tmp = dst_path.with_extension("gguf.tmp");
    let out = File::create(&tmp)?;
    let mut w = BufWriter::with_capacity(1 << 20, out);
    w.write_u32::<LittleEndian>(0x4655_4747)?; // GGUF
    w.write_u32::<LittleEndian>(2)?;
    w.write_u64::<LittleEndian>(names.len() as u64)?;
    w.write_u64::<LittleEndian>(content.metadata.len() as u64)?;
    let mut md: Vec<_> = content.metadata.iter().collect();
    md.sort_by(|a, b| a.0.cmp(b.0));
    for (key, value) in md {
        write_string(&mut w, key)?;
        w.write_u32::<LittleEndian>(value.value_type().to_u32())?;
        value.write(&mut w)?;
    }
    let mut offset = 0usize;
    for (name, bytes) in names.iter().zip(&out_sizes) {
        let info = &content.tensor_infos[*name];
        write_string(&mut w, name)?;
        let dims = info.shape.dims();
        w.write_u32::<LittleEndian>(dims.len() as u32)?;
        for &d in dims.iter().rev() {
            w.write_u64::<LittleEndian>(d as u64)?;
        }
        let dtype = if converted.contains_key(*name) {
            experts
        } else {
            info.ggml_dtype
        };
        w.write_u32::<LittleEndian>(dtype.to_gguf_file_code())?;
        w.write_u64::<LittleEndian>(offset as u64)?;
        offset += bytes + pad32(*bytes);
    }
    let pos = w.stream_position()? as usize;
    w.write_all(&vec![0u8; pad32(pos)])?;

    for (name, bytes) in names.iter().zip(&out_sizes) {
        match converted.get(*name) {
            Some(image) => w.write_all(image)?,
            None => {
                let info = &content.tensor_infos[*name];
                let start = data_base + info.offset as usize;
                w.write_all(&map[start..start + *bytes])?;
            }
        }
        w.write_all(&vec![0u8; pad32(*bytes)])?;
    }
    w.flush()?;
    drop(w);
    std::fs::rename(&tmp, dst_path)?;
    Ok(converted.len())
}

/// Whether a GGUF shard carries any expert tensor.
fn shard_has_experts(path: &Path) -> Result<bool> {
    let mut f = File::open(path)?;
    let content = Content::read(&mut f)?;
    Ok(content
        .tensor_infos
        .keys()
        .any(|n| expert_proj_of(n).is_some()))
}

/// Convert the pinned Q8_0 split into its Q4_KO-expert sibling.
///
/// `q8_splits` is the ordered Q8_0 shard list; `awq_shards` the safetensors
/// files covering the expert tensors; `out_dir` receives the sibling split
/// (expert-free shards hard-linked, expert shards rewritten). Returns the
/// ordered converted split paths.
///
/// Resumable, and **a final name is only ever a complete shard**: rewrites go
/// to a `.tmp` sibling and rename into place on success, so an interrupted
/// conversion leaves a `.tmp` to overwrite, never a truncated `.gguf` for the
/// resume check to mistake for done. (A killed first run produced exactly
/// that: a partial shard skipped as finished, surfacing 200 GB later as
/// `failed to fill whole buffer` at load.)
pub fn convert_w4a16_experts(
    q8_splits: &[PathBuf],
    awq_shards: &[PathBuf],
    out_dir: &Path,
    device: &Device,
) -> Result<Vec<PathBuf>> {
    std::fs::create_dir_all(out_dir)?;
    let awq = W4a16Source::open(awq_shards)?;
    let n = q8_splits.len();
    let mut out = Vec::with_capacity(n);
    for (i, src) in q8_splits.iter().enumerate() {
        let dst = out_dir.join(format!(
            "Qwen3.8-Flash-Next-Q4KOEXP-{:05}-of-{:05}.gguf",
            i + 1,
            n
        ));
        if dst.exists() {
            out.push(dst);
            continue;
        }
        if shard_has_experts(src)? {
            let tmp = dst.with_extension("gguf.tmp");
            let t = std::time::Instant::now();
            let converted = rewrite_shard(src, &tmp, &awq, device)?;
            std::fs::rename(&tmp, &dst)?;
            println!(
                "  converted {:?}: {} experts re-emitted as Q4_KO ({:.0}s)",
                dst.file_name().unwrap_or_default(),
                converted,
                t.elapsed().as_secs_f32()
            );
        } else {
            // Same bytes, no copy. Fall back to a copy across volumes. The link
            // is to the resolved file: a hub-cache source is itself a symlink
            // into `blobs/`, and on Linux `link(2)` links the symlink — a
            // relative one that dangles from any other directory.
            let resolved = std::fs::canonicalize(src)?;
            if std::fs::hard_link(&resolved, &dst).is_err() {
                std::fs::copy(&resolved, &dst)?;
            }
        }
        out.push(dst);
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle::quantized::ko_quant::dequant_ko;

    /// Build a minimal in-memory W4A16 pair (packed + scales) for one linear,
    /// returning the safetensors bytes and the expected dequantized values.
    fn synthetic_w4a16(nrows: usize, ncols: usize, seed: u64) -> (Vec<u8>, Vec<f32>) {
        let words = ncols / 8;
        let k_groups = ncols / 128;
        let mut lcg = seed;
        let mut next = move || {
            lcg = lcg.wrapping_mul(6364136223846793005).wrapping_add(1);
            (lcg >> 33) as u32
        };
        let codes: Vec<u8> = (0..nrows * ncols).map(|_| (next() % 16) as u8).collect();
        let scales: Vec<f32> = (0..nrows * k_groups)
            .map(|_| {
                // f16-exact scales, as bf16 weight scales in range are.
                let e = (next() % 8) as i32 - 10;
                2f32.powi(e)
            })
            .collect();
        let mut packed = Vec::with_capacity(nrows * words * 4);
        for r in 0..nrows {
            for wi in 0..words {
                let mut word = 0u32;
                for i in 0..8 {
                    word |= (codes[r * ncols + wi * 8 + i] as u32) << (4 * i);
                }
                packed.extend_from_slice(&word.to_le_bytes());
            }
        }
        let scale_bytes: Vec<u8> = scales
            .iter()
            .flat_map(|&s| bf16::from_f32(s).to_le_bytes())
            .collect();
        let expect: Vec<f32> = codes
            .iter()
            .enumerate()
            .map(|(i, &c)| {
                let row = i / ncols;
                let g = (i % ncols) / 128;
                scales[row * k_groups + g] * (c as f32 - 8.0)
            })
            .collect();

        // Assemble a one-tensor-pair safetensors buffer by hand.
        let header = format!(
            r#"{{"x.weight_packed":{{"dtype":"I32","shape":[{nrows},{words}],"data_offsets":[0,{p}]}},"x.weight_scale":{{"dtype":"BF16","shape":[{nrows},{k_groups}],"data_offsets":[{p},{end}]}}}}"#,
            p = packed.len(),
            end = packed.len() + scale_bytes.len(),
        );
        let mut buf = Vec::new();
        buf.extend_from_slice(&(header.len() as u64).to_le_bytes());
        buf.extend_from_slice(header.as_bytes());
        buf.extend_from_slice(&packed);
        buf.extend_from_slice(&scale_bytes);
        (buf, expect)
    }

    #[test]
    fn w4a16_decode_pack_roundtrip_is_bit_exact() {
        let (nrows, ncols) = (16usize, 256usize);
        let (buf, expect) = synthetic_w4a16(nrows, ncols, 42);
        let dir = std::env::temp_dir().join("qwen4exp_w4a16_test");
        std::fs::create_dir_all(&dir).unwrap();
        let p = dir.join("one.safetensors");
        std::fs::write(&p, &buf).unwrap();

        let src = W4a16Source::open(&[p]).unwrap();
        let d = src.decode("x").unwrap();
        assert_eq!(d.nrows, nrows);
        assert_eq!(d.ncols, ncols);
        // The decoded affine IS the source dequantization…
        let decoded: Vec<f32> = d
            .codes
            .iter()
            .enumerate()
            .map(|(i, &c)| {
                let (s, mn) = d.dm[(i / ncols) * (ncols / 128) + (i % ncols) / 128];
                s * c as f32 + mn
            })
            .collect();
        assert_eq!(decoded, expect);
        // …the KO pack reproduces it bit-for-bit through the INDEPENDENT
        // dequant path (this is the check that anchors the fast
        // codes-roundtrip verify used by the converter)…
        let packed = pack_q4_ko(&d.codes, &d.dm, nrows, ncols);
        let back = dequant_ko(&packed, nrows, ncols, GgmlDType::Q4_KO);
        assert_eq!(back, expect, "KO round-trip diverged from the source");
        // …and the fast verify agrees with it.
        verify_packed(&packed, &d, "test").unwrap();
    }

    #[test]
    fn nibble_order_is_little_endian_within_the_word() {
        // One row, eight elements: codes 0..7 → word 0x7654_3210. Element i
        // must decode from bits [4i, 4i+4).
        let word: u32 = 0x7654_3210;
        for i in 0..8 {
            assert_eq!((word >> (4 * i)) & 0xF, i as u32);
        }
    }

    #[test]
    fn gguf_expert_names_map_to_hf_projections() {
        assert_eq!(
            expert_proj_of("blk.7.ffn_gate_exps.weight"),
            Some((7, "gate"))
        );
        assert_eq!(expert_proj_of("blk.0.ffn_up_exps.weight"), Some((0, "up")));
        assert_eq!(
            expert_proj_of("blk.47.ffn_down_exps.weight"),
            Some((47, "down"))
        );
        assert_eq!(expert_proj_of("blk.7.ffn_gate_shexp.weight"), None);
        assert_eq!(expert_proj_of("per_layer_token_embd.weight"), None);
        assert_eq!(expert_proj_of("blk.7.ffn_gate_inp.weight"), None);
    }
}
