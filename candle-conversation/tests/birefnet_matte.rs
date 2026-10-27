//! BiRefNet HR matting, whole graph, once, on the card: what it costs.
//!
//! Before the matte guest can carry this model it has to be known whether its
//! activations fit where a guest's come from. The figures this prints are the
//! answer — the pool's high-water mark across one 2048² forward, the weights
//! apart, and the wall time — and the alpha's range, which must be a
//! probability for `Family::BiRefNet` to clamp rather than normalise.
//!
//! Run with the card to itself (no daemon beside it):
//!
//! ```text
//! cargo test --release -p candle-conversation --test birefnet_matte -- --ignored --nocapture
//! ```
//!
//! The checkpoint is the fp32 export, `PinkPixel/birefnet-hr-matting-onnx`
//! `model.onnx`, in the Hugging Face cache as downloaded.

use std::collections::HashMap;
use std::path::PathBuf;
use std::time::Instant;

use candle::{DType, Device, Tensor};

/// One export, at the revision this was measured against.
fn checkpoint(file: &str) -> PathBuf {
    dirs_home()
        .join(
            ".cache/huggingface/hub/models--PinkPixel--birefnet-hr-matting-onnx/snapshots/\
             792518b9d4dfe9793891712677de25e554f948c6",
        )
        .join(file)
}

fn dirs_home() -> PathBuf {
    PathBuf::from(
        std::env::var_os("USERPROFILE")
            .or_else(|| std::env::var_os("HOME"))
            .unwrap(),
    )
}

#[test]
#[ignore = "loads BiRefNet HR (0.9 GB) and runs one 2048² forward on the card; needs it to itself"]
fn birefnet_hr_matting_fp32_costs_this_much() {
    measure("model.onnx");
}

#[test]
#[ignore = "loads BiRefNet HR fp16 (0.45 GB) and runs one 2048² forward on the card; needs it to itself"]
fn birefnet_hr_matting_fp16_costs_this_much() {
    measure("model_fp16.onnx");
}

/// One forward of `file`: its weights, wall time, activation peak and alpha.
fn measure(file: &str) {
    let path = checkpoint(file);
    assert!(path.exists(), "the checkpoint is not at {path:?}");
    let device = Device::new_cuda(0).unwrap();
    let cuda = device.as_cuda_device().unwrap();

    let t = Instant::now();
    let model = candle_onnx::read_file(&path).unwrap();
    let graph = model.graph.as_ref().unwrap();
    let mut values = HashMap::new();
    for init in graph.initializer.iter() {
        let host = candle_onnx::get_tensor(init, &init.name).unwrap();
        values.insert(init.name.clone(), host.to_device(&device).unwrap());
    }
    device.synchronize().unwrap();
    let weights = cuda.pool_used_bytes().unwrap();
    println!(
        "loaded {} weights, {:.0} MiB in the pool, in {:.1} s",
        values.len(),
        weights as f64 / (1u64 << 20) as f64,
        t.elapsed().as_secs_f64()
    );

    // A picture with structure in it — a bright disc on a dark field — so the
    // alpha has somewhere to change, normalised as the export documents.
    let side = 2048usize;
    let mean = [0.485f32, 0.456, 0.406];
    let std = [0.229f32, 0.224, 0.225];
    let mut chw = vec![0f32; 3 * side * side];
    for y in 0..side {
        for x in 0..side {
            let (dx, dy) = (x as f32 - 1024.0, y as f32 - 1024.0);
            let v = if dx * dx + dy * dy < 600.0 * 600.0 {
                0.9
            } else {
                0.1
            };
            for c in 0..3 {
                chw[c * side * side + y * side + x] = (v - mean[c]) / std[c];
            }
        }
    }
    // In the dtype the graph declares: an fp16 export may take f16 or keep f32.
    let declared = graph
        .input
        .iter()
        .find(|i| i.name == "image")
        .and_then(|i| match i.r#type.as_ref()?.value.as_ref()? {
            candle_onnx::onnx::type_proto::Value::TensorType(t) => {
                candle_onnx::onnx::tensor_proto::DataType::try_from(t.elem_type).ok()
            }
            _ => None,
        })
        .and_then(candle_onnx::dtype)
        .unwrap_or(DType::F32);
    let input = Tensor::from_vec(chw, (1, 3, side, side), &device)
        .unwrap()
        .to_dtype(declared)
        .unwrap();
    println!("input dtype {declared:?}");
    values.insert("image".to_string(), input);

    cuda.reset_pool_used_peak().unwrap();
    let t = Instant::now();
    let out = candle_onnx::simple_eval_on(&model, values, &device).unwrap();
    device.synchronize().unwrap();
    let secs = t.elapsed().as_secs_f64();
    let peak = cuda.pool_used_peak_bytes().unwrap();

    let alpha = out["alpha"]
        .flatten_all()
        .unwrap()
        .to_dtype(DType::F32)
        .unwrap();
    let (lo, hi) = (
        alpha.min(0).unwrap().to_scalar::<f32>().unwrap(),
        alpha.max(0).unwrap().to_scalar::<f32>().unwrap(),
    );
    println!(
        "forward {secs:.2} s; pool peak {:.0} MiB, of which activations {:.0} MiB; alpha in [{lo:.4}, {hi:.4}]",
        peak as f64 / (1u64 << 20) as f64,
        peak.saturating_sub(weights) as f64 / (1u64 << 20) as f64,
    );
    assert_eq!(out["alpha"].dims(), &[1, 1, side, side]);
    assert!(
        lo >= 0.0 && hi <= 1.0,
        "the export's alpha is not a probability"
    );
}
