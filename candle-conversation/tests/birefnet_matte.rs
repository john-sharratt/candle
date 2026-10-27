//! BiRefNet, whole graph, once, on the card: what it costs.
//!
//! Before the matte guest can carry a model it has to be known whether its
//! activations fit where a guest's come from. The figures this prints are the
//! answer — the pool's high-water mark across one forward at the graph's own
//! side, the weights apart, and the wall time — and the output's range, which
//! says whether the export ends in its sigmoid (a probability, which
//! `Family::BiRefNet` clamps) or in logits.
//!
//! Run with the card to itself (no daemon beside it):
//!
//! ```text
//! cargo test --release -p candle-conversation --test birefnet_matte -- --ignored --nocapture
//! ```
//!
//! The HR matting exports are `PinkPixel/birefnet-hr-matting-onnx` (2048²) in
//! the Hugging Face cache as downloaded; the general one is
//! `onnx-community/BiRefNet-ONNX` `onnx/model_fp16.onnx` (1024²) in the repack
//! folder.

use std::collections::HashMap;
use std::path::PathBuf;
use std::time::Instant;

use candle::{DType, Device, Tensor};
use candle_onnx::onnx::{tensor_shape_proto::dimension::Value as Dim, type_proto};

fn home() -> PathBuf {
    PathBuf::from(
        std::env::var_os("USERPROFILE")
            .or_else(|| std::env::var_os("HOME"))
            .unwrap(),
    )
}

/// One HR export, at the revision this was measured against.
fn hr(file: &str) -> PathBuf {
    home()
        .join(
            ".cache/huggingface/hub/models--PinkPixel--birefnet-hr-matting-onnx/snapshots/\
             792518b9d4dfe9793891712677de25e554f948c6",
        )
        .join(file)
}

#[test]
#[ignore = "loads BiRefNet HR (0.9 GB) and runs one 2048² forward on the card; needs it to itself"]
fn birefnet_hr_matting_fp32_costs_this_much() {
    measure(hr("model.onnx"));
}

#[test]
#[ignore = "loads BiRefNet HR fp16 (0.45 GB) and runs one 2048² forward on the card; needs it to itself"]
fn birefnet_hr_matting_fp16_costs_this_much() {
    measure(hr("model_fp16.onnx"));
}

#[test]
#[ignore = "loads BiRefNet general fp16 (0.49 GB) and runs one forward on the card; needs it to itself"]
fn birefnet_general_fp16_costs_this_much() {
    measure(general());
}

/// **In a guest heap, the forward leaves the pool alone.** The matte guest's
/// shape: the activations' heap open over ground, the input seeded into it,
/// and the forward run under the guest's ambient seed so every chain — the
/// graph's constants and sampling grids as well as the input's — carves there.
/// Without the seed, 36 GB of a 1024² forward's allocations missed the heap and
/// a busy card's pool could not hold them (WhiteWorld, 2026-10-27).
#[test]
#[ignore = "loads BiRefNet general fp16 and runs one forward in a 5 GiB guest heap; needs the card to itself"]
fn birefnet_general_in_a_guest_heap_leaves_the_pool_alone() {
    let escaped = measure_in(general(), Some(5 << 30));
    assert!(
        escaped < 256 << 20,
        "{} MiB of the forward still came from the pool",
        escaped >> 20
    );
}

fn general() -> PathBuf {
    home().join(".cache/zend/models/onnx-community--BiRefNet-ONNX/model_fp16.onnx")
}

/// One forward of the graph at `path`: its weights, wall time, activation
/// peak and output range.
fn measure(path: PathBuf) {
    measure_in(path, None);
}

/// [`measure`], optionally with the activations' heap open over `heap` bytes
/// as the matte guest runs it. Returns what the pool grew by over the forward.
fn measure_in(path: PathBuf, heap: Option<usize>) -> usize {
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

    // The graph's own input: its name, its dtype, and the side it declares.
    let initialised: std::collections::HashSet<&str> =
        graph.initializer.iter().map(|i| i.name.as_str()).collect();
    let input = graph
        .input
        .iter()
        .find(|i| !initialised.contains(i.name.as_str()))
        .expect("a graph input that is not a weight");
    let Some(type_proto::Value::TensorType(tt)) =
        input.r#type.as_ref().and_then(|t| t.value.as_ref())
    else {
        panic!("the input is not a tensor");
    };
    let dims: Vec<i64> = tt
        .shape
        .as_ref()
        .unwrap()
        .dim
        .iter()
        .map(|d| match d.value {
            Some(Dim::DimValue(v)) => v,
            _ => 1,
        })
        .collect();
    let side = dims[3] as usize;
    let declared = candle_onnx::onnx::tensor_proto::DataType::try_from(tt.elem_type)
        .ok()
        .and_then(candle_onnx::dtype)
        .unwrap_or(DType::F32);
    let output = graph.output[0].name.clone();
    println!(
        "input `{}` {dims:?} {declared:?}, output `{output}`",
        input.name
    );

    // A picture with structure in it — a bright disc on a dark field — so the
    // output has somewhere to change, normalised with ImageNet's statistics.
    let mean = [0.485f32, 0.456, 0.406];
    let std = [0.229f32, 0.224, 0.225];
    let mut chw = vec![0f32; 3 * side * side];
    let (c0, r) = (side as f32 / 2.0, side as f32 * 0.3);
    for y in 0..side {
        for x in 0..side {
            let (dx, dy) = (x as f32 - c0, y as f32 - c0);
            let v = if dx * dx + dy * dy < r * r { 0.9 } else { 0.1 };
            for c in 0..3 {
                chw[c * side * side + y * side + x] = (v - mean[c]) / std[c];
            }
        }
    }
    let image = Tensor::from_vec(chw, (1, 3, side, side), &device)
        .unwrap()
        .to_dtype(declared)
        .unwrap();

    // The heap's ground is one pool allocation made before the measurement,
    // so whatever the pool grows by during the forward escaped the heap.
    let stream = cuda.cuda_stream();
    let ground = heap.map(|bytes| {
        let g = stream.alloc_zeros::<u8>(bytes).unwrap();
        let base = {
            use candle::cuda_backend::cudarc::driver::DevicePtr;
            g.device_ptr(&stream).0
        };
        candle_nn::kv_cache::open_guest_heap(&stream, &[(base, bytes)]);
        g
    });
    let image = if heap.is_some() {
        candle_conversation::guest::varground::into_arena(&image, &device).unwrap()
    } else {
        image
    };
    values.insert(input.name.clone(), image);
    device.synchronize().unwrap();
    let before = cuda.pool_used_bytes().unwrap();

    cuda.reset_pool_used_peak().unwrap();
    let t = Instant::now();
    let seed =
        candle::cuda_backend::wave_provenance::WaveTicket::guest(stream.context().ordinal() as u32);
    let out = match heap {
        Some(_) => candle::cuda_backend::wave_provenance::with_ambient(seed, || {
            candle_onnx::simple_eval_on(&model, values, &device)
        })
        .unwrap(),
        None => candle_onnx::simple_eval_on(&model, values, &device).unwrap(),
    };
    device.synchronize().unwrap();
    let secs = t.elapsed().as_secs_f64();
    let peak = cuda.pool_used_peak_bytes().unwrap();
    let escaped = peak.saturating_sub(before) as usize;
    if heap.is_some() {
        println!(
            "pool grew {} MiB over the forward, outside the heap",
            escaped >> 20
        );
    }

    let o = out[&output]
        .flatten_all()
        .unwrap()
        .to_dtype(DType::F32)
        .unwrap();
    let (lo, hi) = (
        o.min(0).unwrap().to_scalar::<f32>().unwrap(),
        o.max(0).unwrap().to_scalar::<f32>().unwrap(),
    );
    println!(
        "forward {secs:.2} s; pool peak {:.0} MiB, of which activations {:.0} MiB; output {:?} in [{lo:.4}, {hi:.4}] ({})",
        peak as f64 / (1u64 << 20) as f64,
        peak.saturating_sub(weights) as f64 / (1u64 << 20) as f64,
        out[&output].dims(),
        if lo >= 0.0 && hi <= 1.0 { "a probability" } else { "logits" },
    );
    assert_eq!(out[&output].dims(), &[1, 1, side, side]);
    drop((o, out));
    if let Some(g) = ground {
        if let Some(s) = candle_nn::kv_cache::close_guest_heap(&stream) {
            println!(
                "heap {} MiB, peak {} MiB, {} carves",
                s.capacity >> 20,
                s.peak >> 20,
                s.carves
            );
        }
        drop(g);
    }
    escaped
}
