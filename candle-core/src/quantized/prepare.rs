//! Offline GGUF preparation: **merge** split GGUFs and **repack** matmul-weight tensors to
//! their lane-major KO twin, emitting a single file the engine loads *pre-repacked*. This moves
//! the KO repack out of every launch (no per-load reorder, no runtime double-storage) and lets
//! the loader stream KO chunks straight from the (pageable) mmap — the reusable core behind the
//! download-time `merge`/`repack` option.
//!
//! Pure CPU + the parallel [`mxfp4_native_to_ko_gpu_chunk`] reorder, so it runs headless. It
//! streams tensor-by-tensor (only one tensor's bytes are ever in memory) and reads each source
//! tensor sequentially, so a ~150 GB model is bounded by sequential disk I/O + the parallel
//! reorder — minutes, not the strided-random cost the runtime staging paid.
//!
//! First cut repacks **MXFP4 → MXFP4_KO** (the DeepSeek routed experts, the case that matters);
//! every other tensor passes through byte-for-byte, so a mixed-quant model still round-trips.
//! Extending to the affine KO twins (`Q4_K → Q4_KO`, …) is a matter of adding their repack in
//! [`repack_matrix`].

use crate::quantized::ko_quant::{mxfp4_native_to_ko_gpu_chunk, quantize_ko};
use crate::quantized::{gguf_file, GgmlDType, Int8Mode};
use crate::{Result, Shape};
use byteorder::{LittleEndian, WriteBytesExt};
use memmap2::MmapOptions;
use std::fs::File;
use std::io::{BufWriter, Cursor, Write};
use std::path::Path;

/// Bytes of a `dtype` tensor of the given `shape` (blocks × type size).
fn tensor_bytes(shape: &Shape, dtype: GgmlDType) -> usize {
    shape.elem_count() / dtype.block_size() * dtype.type_size()
}

/// GGUF pads each tensor's data to a 32-byte boundary. Bytes of padding after `size`.
fn pad32(size: usize) -> usize {
    31 - (31 + size) % 32
}

/// The KO twin this tensor repacks to, or `None` to pass it through unchanged.
///
/// Rule: a quantized **matmul weight** whose innermost 2-D matrix fits the KO tiling
/// (`nrows % 8 == 0`, `ncols % 128 == 0`). Embeddings are lookup tables, never matmul'd, so
/// they are excluded by the standard `token_embd` name (a llama.cpp convention, not a per-model
/// list). 2-D weights repack directly; 3-D MoE expert banks `[n_expert, nrows, ncols]` repack
/// per expert. Only MXFP4 is handled today; extend here for the affine twins.
fn ko_target(name: &str, shape: &Shape, dtype: GgmlDType, mode: Int8Mode) -> Option<GgmlDType> {
    // Embeddings are lookup tables, not matmul weights — never repack (llama.cpp convention).
    if name.contains("token_embd") {
        return None;
    }
    // `attn_output_a` (wo_a) is a per-group *batched* matmul, so its KO form isn't a single 2-D
    // weight — the engine keeps it Q8_0 and handles the grouped case itself. Leave it native.
    if name.contains("attn_output_a") {
        return None;
    }
    // Repack MXFP4 (routed experts), Q8_0 (attention/shared/dense), and BF16/F16 (the float
    // projection weights: compressor, indexer, router, mHC, lm_head) to their KO twin. A float
    // weight quantizes to Q8_KO (the int8 twin the q8a128 matmul reads); already-quantized sources
    // map via `to_ko`. Non-matmul float tensors are excluded by rank (norms/biases/scales are 1-D →
    // fail the 2-D/3-D check below) and by name (token_embd, attn_output_a) above.
    let float_src = matches!(dtype, GgmlDType::BF16 | GgmlDType::F16);
    if !float_src && !matches!(dtype, GgmlDType::MXFP4 | GgmlDType::Q8_0) {
        return None;
    }
    let d = shape.dims();
    let (nrows, ncols) = match d.len() {
        2 => (d[0], d[1]),
        3 => (d[1], d[2]),
        _ => return None,
    };
    // The q8a128/MXFP4 KO matmul kernel tiles N in blocks of 32 (K in blocks of 128), a stricter
    // bound than the KO *storage* chunk (8 rows). A weight with `nrows` in {8,16,24} mod 32 packs
    // fine but the matmul kernel rejects it — leave those dense (e.g. the tiny mHC `fn_w`,
    // `mix_hc=24`). See `repack_ko` / `q8a128_dense_matmul`.
    if nrows % 32 != 0 || ncols % 128 != 0 {
        return None;
    }
    if float_src {
        Some(GgmlDType::Q8_KO)
    } else {
        dtype.to_ko(mode).ok()
    }
}

/// Dequantize an affine-quant source matrix's bytes to F32 (byte-wise, no alignment assumptions),
/// for the KO repack. Only the dtypes `ko_target` admits need handling.
fn dequant_source(src: &[u8], n: usize, src_dtype: GgmlDType) -> Result<Vec<f32>> {
    match src_dtype {
        GgmlDType::Q8_0 => {
            // BlockQ8_0 = 2 B f16 scale + 32 × i8 → 34 B / 32 elems; value = d · q.
            let mut out = vec![0f32; n];
            for (bi, blk) in src.chunks_exact(34).enumerate() {
                let d = half::f16::from_le_bytes([blk[0], blk[1]]).to_f32();
                for j in 0..32 {
                    out[bi * 32 + j] = blk[2 + j] as i8 as f32 * d;
                }
            }
            Ok(out)
        }
        GgmlDType::BF16 => {
            let mut out = vec![0f32; n];
            for (i, ch) in src.chunks_exact(2).take(n).enumerate() {
                out[i] = half::bf16::from_le_bytes([ch[0], ch[1]]).to_f32();
            }
            Ok(out)
        }
        GgmlDType::F16 => {
            let mut out = vec![0f32; n];
            for (i, ch) in src.chunks_exact(2).take(n).enumerate() {
                out[i] = half::f16::from_le_bytes([ch[0], ch[1]]).to_f32();
            }
            Ok(out)
        }
        other => crate::bail!("prepare: no dequant_source codec for {other:?}"),
    }
}

/// Repack one native matrix (`[nrows, ncols]`) to its KO twin and write the KO bytes.
fn repack_matrix(
    src: &[u8],
    nrows: usize,
    ncols: usize,
    src_dtype: GgmlDType,
    ko: GgmlDType,
    out: &mut impl Write,
) -> Result<()> {
    match ko {
        // MXFP4 → MXFP4_KO: exact byte-reorder + collapse-dm bake (no requant).
        GgmlDType::MXFP4_KO => {
            let ko_bytes = mxfp4_native_to_ko_gpu_chunk(src, nrows, ncols);
            out.write_all(&ko_bytes)?;
            Ok(())
        }
        // Affine KO twins (Q8_KO/Q4_KO/…): dequantize the source, then quantize to the lane-major
        // per-128 KO layout the int8 kernel reads. Near-lossless for Q8→Q8_KO.
        GgmlDType::Q8_KO | GgmlDType::Q4_KO | GgmlDType::Q5_KO | GgmlDType::Q6_KO => {
            let f32 = dequant_source(src, nrows * ncols, src_dtype)?;
            let ko_bytes = quantize_ko(&f32, nrows, ncols, ko);
            out.write_all(&ko_bytes)?;
            Ok(())
        }
        other => crate::bail!("prepare: KO twin {other:?} not supported"),
    }
}

/// Merge `sources` (one file, or the ordered GGUF splits) and repack matmul weights to their KO
/// twin per `mode`, writing a single prepared GGUF to `dst`. Non-repacked tensors and all
/// metadata are copied verbatim (minus the now-meaningless `split.*` keys).
pub fn prepare_ko_gguf(sources: &[&Path], dst: &Path, mode: Int8Mode) -> Result<()> {
    if sources.is_empty() {
        crate::bail!("prepare_ko_gguf: no source files");
    }
    // ── Read every source header; keep the mmaps alive for the streaming data copy. ──
    let mut mmaps = Vec::with_capacity(sources.len());
    let mut contents = Vec::with_capacity(sources.len());
    for &s in sources {
        let f = File::open(s).map_err(crate::Error::wrap)?;
        let mmap = unsafe { MmapOptions::new().map(&f).map_err(crate::Error::wrap)? };
        let ct = gguf_file::Content::read(&mut Cursor::new(&mmap[..]))?;
        mmaps.push(mmap);
        contents.push(ct);
    }

    // ── Flatten tensors across sources, ordered by (source, source-offset) = file order, so the
    //    source reads below stay sequential. ──
    let mut tensors: Vec<(String, usize, gguf_file::TensorInfo)> = Vec::new();
    for (si, ct) in contents.iter().enumerate() {
        for (name, info) in ct.tensor_infos.iter() {
            tensors.push((name.clone(), si, info.clone()));
        }
    }
    tensors.sort_by_key(|(_, si, info)| (*si, info.offset));

    // ── Metadata: take the first split's, drop the split bookkeeping (output is one file). ──
    let metadata: Vec<(&str, &gguf_file::Value)> = contents[0]
        .metadata
        .iter()
        .filter(|(k, _)| !k.starts_with("split."))
        .map(|(k, v)| (k.as_str(), v))
        .collect();

    // ── Resolve each output tensor's dtype + byte size, then its data offset. ──
    struct OutTensor {
        name: String,
        src_idx: usize,
        src_offset: u64,
        src_dtype: GgmlDType,
        shape: Shape,
        out_dtype: GgmlDType,
        out_offset: usize,
        out_size: usize,
    }
    let mut out_tensors = Vec::with_capacity(tensors.len());
    let mut offset = 0usize;
    for (name, si, info) in tensors {
        let ko = ko_target(&name, &info.shape, info.ggml_dtype, mode);
        let out_dtype = ko.unwrap_or(info.ggml_dtype);
        let out_size = tensor_bytes(&info.shape, out_dtype);
        out_tensors.push(OutTensor {
            name,
            src_idx: si,
            src_offset: info.offset,
            src_dtype: info.ggml_dtype,
            shape: info.shape.clone(),
            out_dtype,
            out_offset: offset,
            out_size,
        });
        offset += out_size + pad32(out_size);
    }

    // ── Write the prepared GGUF (streaming). ──
    let f = File::create(dst).map_err(crate::Error::wrap)?;
    let mut w = BufWriter::with_capacity(64 << 20, f);

    // Header (GGUF v2).
    w.write_u32::<LittleEndian>(0x4655_4747)?;
    w.write_u32::<LittleEndian>(2)?;
    w.write_u64::<LittleEndian>(out_tensors.len() as u64)?;
    w.write_u64::<LittleEndian>(metadata.len() as u64)?;
    for (name, value) in metadata.iter() {
        gguf_file::write_string(&mut w, name)?;
        w.write_u32::<LittleEndian>(value.value_type().to_u32())?;
        value.write(&mut w)?;
    }
    // Tensor infos.
    for t in out_tensors.iter() {
        gguf_file::write_string(&mut w, &t.name)?;
        let dims = t.shape.dims();
        w.write_u32::<LittleEndian>(dims.len() as u32)?;
        for &dim in dims.iter().rev() {
            w.write_u64::<LittleEndian>(dim as u64)?;
        }
        w.write_u32::<LittleEndian>(t.out_dtype.to_gguf_file_code())?;
        w.write_u64::<LittleEndian>(t.out_offset as u64)?;
    }
    // Align to the tensor-data start (32 B), then stream each tensor's data.
    let pos = stream_pos(&mut w)?;
    w.write_all(&vec![0u8; pad32(pos)])?;

    for t in out_tensors.iter() {
        let td = contents[t.src_idx].tensor_data_offset as usize;
        let src_bytes = tensor_bytes(&t.shape, t.src_dtype);
        let src =
            &mmaps[t.src_idx][td + t.src_offset as usize..td + t.src_offset as usize + src_bytes];
        if t.out_dtype == t.src_dtype {
            w.write_all(src)?; // pass-through
        } else {
            // Repack per innermost 2-D matrix (1 for a weight, n_expert for a MoE bank).
            let d = t.shape.dims();
            let (n_mat, nrows, ncols) = match d.len() {
                2 => (1, d[0], d[1]),
                3 => (d[0], d[1], d[2]),
                _ => crate::bail!("prepare: repack of {}-D tensor {}", d.len(), t.name),
            };
            let src_mat = tensor_bytes(&Shape::from((nrows, ncols)), t.src_dtype);
            for m in 0..n_mat {
                repack_matrix(
                    &src[m * src_mat..(m + 1) * src_mat],
                    nrows,
                    ncols,
                    t.src_dtype,
                    t.out_dtype,
                    &mut w,
                )?;
            }
        }
        w.write_all(&vec![0u8; pad32(t.out_size)])?;
    }
    w.flush().map_err(crate::Error::wrap)?;
    Ok(())
}

/// `BufWriter` doesn't expose `stream_position` without flushing; track it via a seek on the
/// inner file after a flush. Cheap — called once, between the info and data sections.
fn stream_pos<W: Write + std::io::Seek>(w: &mut W) -> Result<usize> {
    Ok(w.stream_position().map_err(crate::Error::wrap)? as usize)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::quantized::ko_quant::mxfp4_native_to_ko_gpu_chunk;
    use crate::quantized::QTensor;
    use crate::{Device, Tensor};

    /// A synthetic GGUF with a 3-D MXFP4 expert bank and an MXFP4 `token_embd`. After prepare,
    /// the expert bank must be MXFP4_KO with bytes equal to the direct per-expert reorder, while
    /// `token_embd` (a lookup, excluded by name) must remain MXFP4 byte-for-byte.
    #[test]
    fn prepare_repacks_experts_excludes_embedding_bytes_exact() -> Result<()> {
        let dev = Device::Cpu;
        let tmp = std::env::temp_dir();
        let src_path = tmp.join("candle_prep_src.gguf");
        let dst_path = tmp.join("candle_prep_dst.gguf");

        let (n_expert, nrows, ncols) = (2usize, 32usize, 256usize);
        let experts = {
            let v: Vec<f32> = (0..n_expert * nrows * ncols)
                .map(|i| ((i % 13) as f32 - 6.0) * 0.1)
                .collect();
            let t = Tensor::from_vec(v, (n_expert, nrows, ncols), &dev)?;
            QTensor::quantize(&t, GgmlDType::MXFP4)?
        };
        let embd = {
            let v: Vec<f32> = (0..128 * 256)
                .map(|i| ((i % 7) as f32 - 3.0) * 0.1)
                .collect();
            let t = Tensor::from_vec(v, (128, 256), &dev)?;
            QTensor::quantize(&t, GgmlDType::MXFP4)?
        };

        let arch = gguf_file::Value::String("test".to_string());
        {
            let mut f = File::create(&src_path).map_err(crate::Error::wrap)?;
            gguf_file::write(
                &mut f,
                &[("general.architecture", &arch)],
                &[
                    ("blk.0.ffn_gate_exps.weight", &experts),
                    ("token_embd.weight", &embd),
                ],
            )?;
        }

        prepare_ko_gguf(&[&src_path], &dst_path, Int8Mode::Performance)?;

        let mut f = File::open(&dst_path).map_err(crate::Error::wrap)?;
        let ct = gguf_file::Content::read(&mut f)?;
        let exp = ct.tensor_infos.get("blk.0.ffn_gate_exps.weight").unwrap();
        let emb = ct.tensor_infos.get("token_embd.weight").unwrap();
        assert_eq!(exp.ggml_dtype, GgmlDType::MXFP4_KO, "experts must repack");
        assert_eq!(exp.shape.dims(), &[n_expert, nrows, ncols]);
        assert_eq!(
            emb.ggml_dtype,
            GgmlDType::MXFP4,
            "embedding must pass through"
        );

        // Byte-exact check: the KO bytes in the output == per-expert reorder of the source native.
        let src_native = experts.data()?;
        let per_expert_src = (nrows * ncols / 32) * 17;
        let mut want = Vec::new();
        for e in 0..n_expert {
            want.extend_from_slice(&mxfp4_native_to_ko_gpu_chunk(
                &src_native[e * per_expert_src..(e + 1) * per_expert_src],
                nrows,
                ncols,
            ));
        }
        let f2 = File::open(&dst_path).map_err(crate::Error::wrap)?;
        let mmap = unsafe { MmapOptions::new().map(&f2).map_err(crate::Error::wrap)? };
        let base = ct.tensor_data_offset as usize + exp.offset as usize;
        let got = &mmap[base..base + want.len()];
        assert_eq!(
            got,
            want.as_slice(),
            "prepared KO bytes must match the direct reorder"
        );

        let _ = std::fs::remove_file(&src_path);
        let _ = std::fs::remove_file(&dst_path);
        Ok(())
    }

    /// Regression for the Q8_KO `type_size` drift: a Q8_0 matmul weight repacks to Q8_KO, and a
    /// tensor written *after* it must land byte-exact at its declared offset. Q8_KO's GGUF
    /// `type_size` (used to lay out and slice tensor data) once returned the CPU `BlockQ8_KO`
    /// size (160 B/128) instead of the emitted `ko_chunk_bytes` (132 B/128), so the header
    /// over-reserved for every Q8_KO tensor and every later tensor drifted forward — the tail
    /// experts ended up past EOF. The size assertion pins the invariant; the trailing-expert
    /// byte match proves the full offset accounting stays aligned through a Q8_KO tensor.
    #[test]
    fn prepare_q8_to_ko_offsets_do_not_drift() -> Result<()> {
        let dev = Device::Cpu;
        let tmp = std::env::temp_dir();
        let src_path = tmp.join("candle_prep_q8ko_src.gguf");
        let dst_path = tmp.join("candle_prep_q8ko_dst.gguf");

        let (nrows, ncols) = (32usize, 256usize);
        let wq = {
            let v: Vec<f32> = (0..nrows * ncols)
                .map(|i| ((i % 11) as f32 - 5.0) * 0.05)
                .collect();
            QTensor::quantize(&Tensor::from_vec(v, (nrows, ncols), &dev)?, GgmlDType::Q8_0)?
        };
        let (ne, er, ec) = (2usize, 32usize, 256usize);
        let experts = {
            let v: Vec<f32> = (0..ne * er * ec)
                .map(|i| ((i % 13) as f32 - 6.0) * 0.1)
                .collect();
            QTensor::quantize(&Tensor::from_vec(v, (ne, er, ec), &dev)?, GgmlDType::MXFP4)?
        };

        let arch = gguf_file::Value::String("test".to_string());
        {
            let mut f = File::create(&src_path).map_err(crate::Error::wrap)?;
            gguf_file::write(
                &mut f,
                &[("general.architecture", &arch)],
                &[
                    ("blk.0.attn_q_b.weight", &wq),
                    ("blk.0.ffn_gate_exps.weight", &experts),
                ],
            )?;
        }

        prepare_ko_gguf(&[&src_path], &dst_path, Int8Mode::Performance)?;

        let mut f = File::open(&dst_path).map_err(crate::Error::wrap)?;
        let ct = gguf_file::Content::read(&mut f)?;
        let wqi = ct.tensor_infos.get("blk.0.attn_q_b.weight").unwrap();
        let exi = ct.tensor_infos.get("blk.0.ffn_gate_exps.weight").unwrap();
        assert_eq!(
            wqi.ggml_dtype,
            GgmlDType::Q8_KO,
            "Q8_0 matmul weight must repack to Q8_KO"
        );

        // The GGUF reader slices `tensor_bytes` from the offset; it MUST equal the bytes the
        // repack actually emitted, or the next tensor's offset is wrong.
        let emitted = quantize_ko(&vec![0f32; nrows * ncols], nrows, ncols, GgmlDType::Q8_KO).len();
        assert_eq!(
            tensor_bytes(&wqi.shape, GgmlDType::Q8_KO),
            emitted,
            "Q8_KO tensor_bytes must equal quantize_ko output length (no offset drift)"
        );

        // Trailing expert byte-exact at its declared offset ⇒ the Q8_KO tensor before it reserved
        // exactly what it wrote (no accumulated drift).
        let src_native = experts.data()?;
        let per_expert_src = (er * ec / 32) * 17;
        let mut want_exp = Vec::new();
        for e in 0..ne {
            want_exp.extend_from_slice(&mxfp4_native_to_ko_gpu_chunk(
                &src_native[e * per_expert_src..(e + 1) * per_expert_src],
                er,
                ec,
            ));
        }
        let f2 = File::open(&dst_path).map_err(crate::Error::wrap)?;
        let mmap = unsafe { MmapOptions::new().map(&f2).map_err(crate::Error::wrap)? };
        let exp_base = ct.tensor_data_offset as usize + exi.offset as usize;
        assert_eq!(
            &mmap[exp_base..exp_base + want_exp.len()],
            want_exp.as_slice(),
            "trailing expert must be byte-exact at its offset (no drift from the Q8_KO tensor)"
        );

        let _ = std::fs::remove_file(&src_path);
        let _ = std::fs::remove_file(&dst_path);
        Ok(())
    }

    /// Dump the dtype of every non-expert tensor in the prepared file that the engine loader
    /// DEQUANTIZES (compressor / indexer / gate / mHC / norms / wo_a). If prepare repacked any of
    /// these to a KO twin, the dequant path faults (KO has no CPU/dequant form) — this pins which.
    #[test]
    #[ignore]
    fn dump_deepseek_dequant_dtypes() -> Result<()> {
        let dir = Path::new(r"D:\models\deepseek-v4-flash-mxfp4");
        let dst = dir.join("DeepSeek-V4-Flash-0731-MXFP4_KO.gguf");
        if !dst.exists() {
            eprintln!("[skip] prepared file absent");
            return Ok(());
        }
        let mut f = File::open(&dst).map_err(crate::Error::wrap)?;
        let ct = gguf_file::Content::read(&mut f)?;
        let mut names: Vec<&String> = ct.tensor_infos.keys().collect();
        names.sort();
        for n in names {
            let d = ct.tensor_infos[n].ggml_dtype;
            let is_dequant_path = n.contains("compressor")
                || n.contains("indexer")
                || n.contains("ffn_gate_inp")
                || n.contains("_hc_")
                || n.contains("attn_output_a")
                || n.ends_with("_norm.weight")
                || n.contains("sinks");
            if (n.starts_with("blk.0.") || n.starts_with("blk.2.")) && is_dequant_path {
                eprintln!("{n}: {d:?}  (KO={})", d.is_ko());
            }
        }
        Ok(())
    }

    /// Produce the real DeepSeek MXFP4_KO file from the merged native GGUF and report the wall
    /// time — validates the transform at 146 GB scale and measures the offline prepare cost.
    #[test]
    #[ignore]
    fn prepare_deepseek_real() -> Result<()> {
        let dir = Path::new(r"D:\models\deepseek-v4-flash-mxfp4");
        let src = dir.join("DeepSeek-V4-Flash-0731-MXFP4-merged.gguf");
        let dst = dir.join("DeepSeek-V4-Flash-0731-MXFP4_KO.gguf");
        if !src.exists() {
            eprintln!("[skip] merged native absent: {src:?}");
            return Ok(());
        }
        let t0 = std::time::Instant::now();
        prepare_ko_gguf(&[&src], &dst, Int8Mode::Performance)?;
        let secs = t0.elapsed().as_secs_f64();
        let out_gb = std::fs::metadata(&dst).map(|m| m.len()).unwrap_or(0) as f64 / 1e9;
        eprintln!("[prepare] wrote {dst:?} ({out_gb:.1} GB) in {secs:.1}s");
        Ok(())
    }

    /// Header sanity of the produced MXFP4_KO file: routed experts must be MXFP4_KO, and a
    /// sampling of resident tensors (attn, embedding, norms) must be untouched.
    #[test]
    #[ignore]
    fn verify_deepseek_ko_header() -> Result<()> {
        let dst =
            Path::new(r"D:\models\deepseek-v4-flash-mxfp4\DeepSeek-V4-Flash-0731-MXFP4_KO.gguf");
        if !dst.exists() {
            eprintln!("[skip] prepared KO file absent");
            return Ok(());
        }
        let mut f = File::open(dst).map_err(crate::Error::wrap)?;
        let ct = gguf_file::Content::read(&mut f)?;
        let mut experts = 0usize;
        let mut mxfp4_left = 0usize;
        for (name, info) in ct.tensor_infos.iter() {
            if name.contains("_exps.weight") {
                assert_eq!(
                    info.ggml_dtype,
                    GgmlDType::MXFP4_KO,
                    "expert {name} should be MXFP4_KO, got {:?}",
                    info.ggml_dtype
                );
                experts += 1;
            }
            if info.ggml_dtype == GgmlDType::MXFP4 {
                mxfp4_left += 1;
            }
        }
        let embd = ct.tensor_infos.get("token_embd.weight");
        eprintln!(
            "[verify] {} MXFP4_KO expert tensors; {} tensors still MXFP4 (embedding etc.); \
             token_embd dtype = {:?}; total tensors = {}",
            experts,
            mxfp4_left,
            embd.map(|i| i.ggml_dtype),
            ct.tensor_infos.len(),
        );
        assert!(experts > 0, "no expert tensors were repacked");
        Ok(())
    }

    /// Break the resident (non-routed-expert) base down by category and size, both on-disk and as
    /// the F32 the engine currently dequantizes several of them to — answers "what's actually
    /// eating the resident VRAM?".
    #[test]
    #[ignore]
    fn analyze_deepseek_resident() -> Result<()> {
        let dst =
            Path::new(r"D:\models\deepseek-v4-flash-mxfp4\DeepSeek-V4-Flash-0731-MXFP4_KO.gguf");
        if !dst.exists() {
            eprintln!("[skip] prepared KO file absent");
            return Ok(());
        }
        let mut f = File::open(dst).map_err(crate::Error::wrap)?;
        let ct = gguf_file::Content::read(&mut f)?;
        // (category, on-disk bytes, F32-equivalent bytes).
        let mut cats: std::collections::BTreeMap<&str, (u64, u64)> = Default::default();
        let mut biggest: Vec<(String, u64, GgmlDType, Vec<usize>)> = Vec::new();
        for (name, info) in ct.tensor_infos.iter() {
            let elems = info.shape.elem_count() as u64;
            let disk =
                elems / info.ggml_dtype.block_size() as u64 * info.ggml_dtype.type_size() as u64;
            let f32b = elems * 4;
            let cat = if name.contains("_exps.weight") {
                "routed_experts (ExpertCache)"
            } else if name.contains("_shexp") {
                "shared_experts (resident)"
            } else if name == "output.weight" {
                "lm_head (resident)"
            } else if name.contains("token_embd") {
                "embedding (now host RAM)"
            } else if name.contains("attn") {
                "attention (resident)"
            } else if name.contains("ffn_gate_inp")
                || name.contains("exp_probs")
                || name.contains("tid2eid")
            {
                "router (resident)"
            } else if name.contains("_hc_") || name.contains(".hc_") {
                "mHC params (resident)"
            } else if name.contains("norm") {
                "norms (resident)"
            } else {
                "other (resident)"
            };
            let e = cats.entry(cat).or_default();
            e.0 += disk;
            e.1 += f32b;
            if !name.contains("_exps.weight") {
                biggest.push((
                    name.clone(),
                    disk,
                    info.ggml_dtype,
                    info.shape.dims().to_vec(),
                ));
            }
        }
        let gb = |b: u64| b as f64 / (1u64 << 30) as f64;
        eprintln!("=== resident-base breakdown (on-disk / F32-equivalent) ===");
        for (cat, (disk, f32b)) in cats.iter() {
            eprintln!(
                "  {cat:32} {:>7.2} GB disk  |  {:>7.2} GB as F32",
                gb(*disk),
                gb(*f32b)
            );
        }
        biggest.sort_by_key(|(_, d, _, _)| std::cmp::Reverse(*d));
        eprintln!("--- 12 largest resident tensors ---");
        for (name, disk, dt, dims) in biggest.iter().take(12) {
            eprintln!("  {:>7.1} MB  {dt:?}  {dims:?}  {name}", *disk as f64 / 1e6);
        }
        Ok(())
    }
}
