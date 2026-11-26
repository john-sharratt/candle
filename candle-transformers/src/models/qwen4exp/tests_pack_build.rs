//! The built trunk against a synthetic Flash-Next checkpoint: every dense
//! weight the engine assembles, built once by `build_trunk` and stored, is
//! byte for byte what the loader's own construction of it produces, and a pack
//! carrying them reads each back as stored, with the checkpoint tensors they
//! replace gone.

use std::collections::{HashMap, HashSet};
use std::fs::File;
use std::path::{Path, PathBuf};

use candle::quantized::gguf_file::Value;
use candle::quantized::gguf_writer::{GgufPlan, GgufStreamWriter, PlannedTensor};
use candle::quantized::{GgmlDType, Int8Mode, QTensor};
use candle::{Device, Result, Tensor};

use super::config::Qwen4ExpConfig;
use super::hyper::ko::HcWeightsKo;
use super::hyper::HcWeights;
use super::loader::OUTPUT;
use super::pack_build::build_trunk;
use super::trunk_names::replaced_sources;
use crate::models::gpu_test_lock::gpu_serial;
use crate::models::latent_moe::GgufModel;
use crate::models::model_pack::build::{write_pack, PackBuild, TokenizerSource};
use crate::models::model_pack::built::BuiltWriter;
use crate::models::model_pack::compose::{Composition, MappedSource};
use crate::models::model_pack::provenance::{Provenance, SourceRecord};
use crate::models::quantized_matmul::QMatMul;

const HIDDEN: usize = 128;
const HC: usize = 4;
const HC_DIM: usize = HC * HIDDEN;
const LOW_RANK: usize = 320;
const VOCAB: usize = 256;

/// A deterministic tensor of `dims`, small and free of NaN.
fn values(dims: &[usize], seed: u64) -> Vec<f32> {
    let n: usize = dims.iter().product();
    let mut s = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
    (0..n)
        .map(|_| {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((s >> 33) as f32 / (1u64 << 31) as f32) - 0.5
        })
        .collect()
}

/// `dims` of `dtype`'s bytes, quantized on the host.
fn bytes(dims: &[usize], dtype: GgmlDType, seed: u64) -> Result<Vec<u8>> {
    let t = Tensor::from_vec(values(dims, seed), dims, &Device::Cpu)?;
    Ok(QTensor::quantize(&t, dtype)?.data()?.into_owned())
}

/// The metadata of a three-block model — a DeltaNet layer, an attention layer
/// and the draft head — at widths the int8 matmul tiles.
fn metadata() -> Vec<(String, Value)> {
    let u = Value::U32;
    let arr = |v: Vec<Value>| Value::Array(v);
    let mut md: Vec<(&str, Value)> = vec![
        ("block_count", u(3)),
        ("nextn_predict_layers", u(1)),
        ("context_length", u(4096)),
        ("embedding_length", u(HIDDEN as u32)),
        ("full_attention_interval", u(2)),
        ("attention.head_count", u(2)),
        ("attention.head_count_kv", u(1)),
        ("attention.key_length", u(64)),
        ("attention.value_length", u(64)),
        ("attention.layer_norm_rms_epsilon", Value::F32(1e-6)),
        (
            "attention.compress_ratios",
            arr(vec![Value::I32(0), Value::I32(4), Value::I32(4)]),
        ),
        ("attention.indexer.head_count", u(1)),
        ("attention.indexer.key_length", u(64)),
        ("attention.indexer.top_k", u(64)),
        ("rope.dimension_count", u(64)),
        (
            "rope.dimension_sections",
            arr(vec![
                Value::I32(11),
                Value::I32(11),
                Value::I32(10),
                Value::I32(0),
            ]),
        ),
        ("rope.freq_base", Value::F32(1e7)),
        ("ssm.state_size", u(16)),
        ("ssm.group_count", u(2)),
        ("ssm.time_step_rank", u(4)),
        ("ssm.inner_size", u(64)),
        ("ssm.conv_kernel", u(4)),
        ("expert_count", u(32)),
        ("expert_used_count", u(2)),
        ("expert_feed_forward_length", u(64)),
        ("expert_shared_feed_forward_length", u(HIDDEN as u32)),
        ("hyper_connection.count", u(HC as u32)),
        ("hyper_connection.low_rank", u(LOW_RANK as u32)),
        ("ple.layers", arr(vec![Value::I32(0)])),
        ("ple.ngram_size", u(3)),
        ("ple.heads_per_ngram", u(8)),
        ("ple.conv_kernel", u(4)),
        ("ple.eos_token_id", u(1)),
        ("ple.image_token_id", u(2)),
        (
            "ple.layer_multipliers",
            arr(vec![Value::U64(3), Value::U64(5), Value::U64(7)]),
        ),
        (
            "ple.head_offsets",
            arr((0..16u64).map(|h| Value::U64(h * 64)).collect()),
        ),
        (
            "ple.head_vocab_sizes",
            arr((0..16).map(|_| Value::U64(64)).collect()),
        ),
        ("embedding_length_per_layer_input", u(8)),
        ("vocab_size", u(VOCAB as u32)),
    ];
    let mut out: Vec<(String, Value)> = md
        .drain(..)
        .map(|(k, v)| (format!("qwen4exp.{k}"), v))
        .collect();
    out.push((
        "general.architecture".into(),
        Value::String("qwen4exp".into()),
    ));
    out
}

/// One hyper-connection module's checkpoint tensors.
fn hc_tensors(prefix: &str, with_inject: bool) -> Vec<(String, GgmlDType, Vec<usize>)> {
    let f = GgmlDType::F32;
    let mut out = vec![
        (format!("{prefix}_norm.weight"), f, vec![HC_DIM]),
        (format!("{prefix}_down.weight"), f, vec![LOW_RANK, HC_DIM]),
        (format!("{prefix}_up.weight"), f, vec![HC_DIM, LOW_RANK]),
    ];
    if with_inject {
        out.push((format!("{prefix}_inject.weight"), f, vec![HC, HC_DIM]));
    }
    out
}

/// Every tensor the trunk build reads, by name, type and shape.
fn tensors() -> Vec<(String, GgmlDType, Vec<usize>)> {
    let (f, q) = (GgmlDType::F32, GgmlDType::Q8_0);
    let mut out = vec![
        (OUTPUT.to_string(), q, vec![VOCAB, HIDDEN]),
        ("token_embd.weight".into(), q, vec![VOCAB, HIDDEN]),
    ];
    out.extend(hc_tensors("output_hc", false));
    for li in 0..3 {
        let p = format!("blk.{li}");
        let t = |s: &str| format!("{p}.{s}.weight");
        out.extend(hc_tensors(&format!("{p}.hc_attn"), true));
        out.extend(hc_tensors(&format!("{p}.hc_ffn"), true));
        if li == 0 {
            out.extend([
                (t("attn_qkv"), q, vec![96, HIDDEN]),
                (t("attn_gate"), q, vec![64, HIDDEN]),
                (t("ssm_beta"), q, vec![16, HIDDEN]),
                (t("ssm_alpha"), q, vec![16, HIDDEN]),
                (t("ssm_out"), q, vec![HIDDEN, HIDDEN]),
                (t("ple_key"), f, vec![HC_DIM, HIDDEN]),
                (t("ple_value"), f, vec![HIDDEN, HIDDEN]),
                (t("ple_norm_key"), f, vec![HC_DIM]),
                (t("ple_norm_query"), f, vec![HC_DIM]),
                (t("ple_norm_conv"), f, vec![HC_DIM]),
                (t("ple_conv1d"), f, vec![HC_DIM, 4]),
            ]);
        } else {
            out.extend([
                (t("attn_q"), q, vec![256, HIDDEN]),
                (t("attn_k"), q, vec![64, HIDDEN]),
                (t("attn_v"), q, vec![64, HIDDEN]),
                (t("attn_output"), q, vec![HIDDEN, HIDDEN]),
            ]);
        }
        out.extend([
            (t("ffn_gate_inp_shexp"), f, vec![HIDDEN]),
            (t("ffn_gate_inp"), f, vec![32, HIDDEN]),
            (t("ffn_gate_shexp"), q, vec![HIDDEN, HIDDEN]),
            (t("ffn_up_shexp"), q, vec![HIDDEN, HIDDEN]),
            (t("ffn_down_shexp"), q, vec![HIDDEN, HIDDEN]),
        ]);
    }
    out.push((
        "blk.2.nextn.eh_proj.weight".into(),
        q,
        vec![HIDDEN, 2 * HIDDEN],
    ));
    out.extend(hc_tensors("blk.2.nextn.hc_head", false));
    out
}

/// The synthetic checkpoint, written to `dir`.
fn checkpoint(dir: &Path) -> Result<PathBuf> {
    let mut plan = GgufPlan::new(32)?;
    for (k, v) in metadata() {
        plan.push_metadata(k, v)?;
    }
    let all = tensors();
    for (name, dtype, dims) in &all {
        plan.push_tensor(PlannedTensor {
            name: name.clone(),
            dtype: *dtype,
            dims: dims.clone(),
        })?;
    }
    let path = dir.join("fixture.gguf");
    let mut w = GgufStreamWriter::new(plan, File::create(&path)?)?;
    for (i, (_, dtype, dims)) in all.iter().enumerate() {
        w.write_tensor_bytes(&bytes(dims, *dtype, i as u64 + 1)?)?;
    }
    w.finish()?;
    Ok(path)
}

/// A built weight's bytes, read back from the device.
fn weight_bytes(m: &QMatMul) -> Vec<u8> {
    m.inner().qtensor().unwrap().data().unwrap().into_owned()
}

/// The loader's construction of hyper-connection module `prefix`.
fn hc_as_loaded(g: &mut GgufModel, prefix: &str, with_inject: bool, dev: &Device) -> HcWeightsKo {
    let mut f32t = |n: String| g.qtensor(&n, dev).unwrap().dequantize(dev).unwrap();
    let down = f32t(format!("{prefix}_down.weight"));
    let down = if with_inject {
        let inject = f32t(format!("{prefix}_inject.weight"));
        Tensor::cat(&[&down, &inject], 0).unwrap()
    } else {
        down
    };
    let w = HcWeights::from_checkpoint(
        f32t(format!("{prefix}_norm.weight")),
        down,
        f32t(format!("{prefix}_up.weight")),
        HC,
    )
    .unwrap();
    HcWeightsKo::from_weights(&w, Int8Mode::Precision).unwrap()
}

/// **The built trunk is the loader's build, stored.** Every weight the
/// engine assembles is built under the name the load reads it by, as a KO
/// twin, in build order; the ones checked here byte for byte — a padded
/// hyper-connection module, a stacked input projection, the padded shared
/// gate — equal what the loader's own construction of them writes; and a
/// pack composed of them stores each exactly as built, with the checkpoint
/// tensors they replace gone.
#[test]
fn the_built_trunk_is_the_loaders_build_and_reads_back_as_stored() -> Result<()> {
    let _gpu = gpu_serial();
    let Ok(dev) = Device::new_cuda(0) else {
        return Ok(());
    };
    let dir = tempfile::tempdir()?;
    let path = checkpoint(dir.path())?;
    let mut g = GgufModel::open(std::slice::from_ref(&path))?;
    let cfg = Qwen4ExpConfig::from_gguf_metadata(&g.metadata)?;
    assert_eq!((cfg.num_layers, cfg.num_mtp_layers), (2, 1));
    let mode = Int8Mode::Precision;

    let mut w = BuiltWriter::create(dir.path())?;
    let pairs = build_trunk(&mut g, &cfg, mode, OUTPUT, &dev, &mut w)?;
    let built = w.finish()?;

    let block = |p: &str, mix: [&str; 2], ple: bool| -> Vec<String> {
        let mut v = vec![
            format!("{p}.hc_attn_down.ko.weight"),
            format!("{p}.hc_attn_up.ko.weight"),
            format!("{p}.hc_ffn_down.ko.weight"),
            format!("{p}.hc_ffn_up.ko.weight"),
            format!("{p}.{}.weight", mix[0]),
            format!("{p}.{}.weight", mix[1]),
        ];
        if ple {
            v.push(format!("{p}.ple_key_value.weight"));
        }
        v.extend([
            format!("{p}.ffn_gate_inp_shexp.padded.weight"),
            format!("{p}.ffn_gate_inp.weight"),
            format!("{p}.ffn_gate_up_shexp.weight"),
            format!("{p}.ffn_down_shexp.weight"),
        ]);
        v
    };
    let mut want: Vec<String> = vec![
        "output.weight".into(),
        "output_hc_down.ko.weight".into(),
        "output_hc_up.ko.weight".into(),
    ];
    want.extend(block("blk.0", ["ssm_in", "ssm_out"], true));
    want.extend(block("blk.1", ["attn_q_k_v", "attn_output"], false));
    want.extend(block("blk.2", ["attn_q_k_v", "attn_output"], false));
    want.extend([
        "blk.2.nextn.eh_proj.embedding.weight".to_string(),
        "blk.2.nextn.eh_proj.hidden.weight".into(),
        "blk.2.nextn.hc_head_down.ko.weight".into(),
        "blk.2.nextn.hc_head_up.ko.weight".into(),
        "output.draft.weight".into(),
    ]);
    let names: Vec<&str> = built.tensors().iter().map(|t| t.name.as_str()).collect();
    assert_eq!(names, want);
    for t in built.tensors() {
        assert!(t.dtype.is_ko(), "{} built as {:?}", t.name, t.dtype);
    }
    assert_eq!(
        built
            .tensors()
            .iter()
            .find(|t| t.name == "output.draft.weight")
            .map(|t| t.dtype),
        Some(GgmlDType::Q4_KO)
    );
    let by_name: HashMap<&str, &[u8]> = built
        .tensors()
        .iter()
        .map(|t| (t.name.as_str(), built.bytes(t).unwrap()))
        .collect();

    // A padded module, its gate's `1/hc` folded in: the loader's construction.
    let hc = hc_as_loaded(&mut g, "blk.0.hc_attn", true, &dev);
    let (down, up) = hc.stored_projections();
    assert_eq!(by_name["blk.0.hc_attn_down.ko.weight"], weight_bytes(down));
    assert_eq!(by_name["blk.0.hc_attn_up.ko.weight"], weight_bytes(up));

    // A stacked input projection: the parts row-concatenated, then repacked.
    let parts: Vec<QTensor> = ["attn_q", "attn_k", "attn_v"]
        .iter()
        .map(|s| g.qtensor(&format!("blk.1.{s}.weight"), &dev))
        .collect::<Result<_>>()?;
    let refs: Vec<&QTensor> = parts.iter().collect();
    let stack = QMatMul::from_qtensor_with_mode(QTensor::concat_rows_cuda(&refs)?, mode)?;
    assert_eq!(by_name["blk.1.attn_q_k_v.weight"], weight_bytes(&stack));

    // The shared gate's one row, padded to a KO tile.
    let row = g
        .qtensor("blk.1.ffn_gate_inp_shexp.weight", &dev)?
        .dequantize(&dev)?
        .reshape((1, HIDDEN))?;
    let pad = Tensor::zeros((31, HIDDEN), row.dtype(), &dev)?;
    let gate = QMatMul::from_qtensor_with_mode(
        QTensor::quantize(&Tensor::cat(&[&row, &pad], 0)?, GgmlDType::F32)?,
        mode,
    )?;
    assert_eq!(
        by_name["blk.1.ffn_gate_inp_shexp.padded.weight"],
        weight_bytes(&gate)
    );

    // The repack pairs, each recorded once.
    let mut got = pairs.clone();
    got.dedup();
    assert_eq!(got, pairs);
    assert!(pairs.contains(&(GgmlDType::Q8_0, GgmlDType::Q8_KO)));
    assert!(pairs.contains(&(GgmlDType::Q8_0, GgmlDType::Q4_KO)));

    // A pack composed of them stores each as built, and the sources they
    // replace are gone.
    drop(g);
    let sources = [MappedSource::open(&path)?];
    let replaced: HashSet<String> = replaced_sources(&cfg).into_iter().collect();
    let mut composition = Composition::from_checkpoint(&sources, &|n| !replaced.contains(n));
    composition.add_built(&built);
    let out = dir.path().join("fixture.precision.pack.gguf");
    write_pack(
        PackBuild {
            sources: &sources,
            provenance: Provenance::complete(vec![SourceRecord {
                role: "checkpoint".into(),
                repo: "org/fixture".into(),
                rev: "r".into(),
                file: "fixture.gguf".into(),
                len: 1,
                sha256: "00".repeat(32),
            }]),
            composition,
            built: Some(&built),
            int8_mode: mode,
            narrow: None,
            checkpoint_bytes: 1,
            tokenizer: TokenizerSource {
                repo: "org/tok".into(),
                rev: "t".into(),
                json: "{}".into(),
            },
            experts: None,
            layers: None,
        },
        &out,
    )?;
    let pack = GgufModel::open(std::slice::from_ref(&out))?;
    for t in built.tensors() {
        let (stored, info) = pack.host_bytes(&t.name)?;
        assert_eq!(info.ggml_dtype, t.dtype, "{}", t.name);
        assert_eq!(stored, by_name[t.name.as_str()], "{}", t.name);
    }
    for name in &replaced {
        assert!(pack.info(name).is_none(), "{name} is still in the pack");
    }
    // The norms and the embedding stay as the checkpoint has them.
    assert_eq!(
        pack.info("blk.0.hc_attn_norm.weight").map(|i| i.ggml_dtype),
        Some(GgmlDType::F32)
    );
    assert_eq!(
        pack.info("token_embd.weight").map(|i| i.ggml_dtype),
        Some(GgmlDType::Q8_0)
    );
    Ok(())
}
