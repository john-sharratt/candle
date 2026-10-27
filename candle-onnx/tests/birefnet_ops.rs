//! The ops BiRefNet's ONNX export needs beyond the rest of the interpreter:
//! `GridSample`, `LayerNormalization`, `GlobalAveragePool`, `Mod` and
//! `Reciprocal`. Each is run as a one-node graph against values worked out by
//! hand from the ONNX operator definitions.

use candle::{DType, Device, Result, Tensor};
use candle_onnx::onnx::attribute_proto::AttributeType;
use candle_onnx::onnx::{AttributeProto, GraphProto, ModelProto, NodeProto, ValueInfoProto};
use candle_onnx::simple_eval;
use std::collections::HashMap;

/// An attribute holding one int, float or string.
enum Attr {
    Int(i64),
    Float(f32),
    Str(&'static str),
}

fn attribute(name: &str, value: Attr) -> AttributeProto {
    let mut a = AttributeProto {
        name: name.to_string(),
        ..Default::default()
    };
    match value {
        Attr::Int(i) => {
            a.i = i;
            a.r#type = AttributeType::Int as i32;
        }
        Attr::Float(f) => {
            a.f = f;
            a.r#type = AttributeType::Float as i32;
        }
        Attr::Str(s) => {
            a.s = s.as_bytes().to_vec();
            a.r#type = AttributeType::String as i32;
        }
    }
    a
}

fn value_info(name: &str) -> ValueInfoProto {
    ValueInfoProto {
        name: name.to_string(),
        ..Default::default()
    }
}

/// One `op` over `inputs` (in order), with `attrs`, answering every name in
/// `outputs`.
fn run(
    op: &str,
    inputs: Vec<(&str, Tensor)>,
    attrs: Vec<(&str, Attr)>,
    outputs: &[&str],
) -> Result<HashMap<String, Tensor>> {
    let model = ModelProto {
        graph: Some(GraphProto {
            node: vec![NodeProto {
                op_type: op.to_string(),
                input: inputs.iter().map(|(n, _)| n.to_string()).collect(),
                output: outputs.iter().map(|n| n.to_string()).collect(),
                attribute: attrs.into_iter().map(|(n, v)| attribute(n, v)).collect(),
                ..Default::default()
            }],
            input: inputs.iter().map(|(n, _)| value_info(n)).collect(),
            output: outputs.iter().map(|n| value_info(n)).collect(),
            ..Default::default()
        }),
        ..Default::default()
    };
    simple_eval(
        &model,
        inputs
            .into_iter()
            .map(|(n, t)| (n.to_string(), t))
            .collect(),
    )
}

fn floats(t: &Tensor) -> Vec<f32> {
    t.flatten_all()
        .unwrap()
        .to_dtype(DType::F32)
        .unwrap()
        .to_vec1()
        .unwrap()
}

/// Bilinear sampling of a 2×2 image at its centre is the mean of its four
/// pixels; `align_corners` 1 puts (-1, -1) on pixel 0 itself.
#[test]
fn grid_sample_reads_between_and_on_pixels() -> Result<()> {
    let x = Tensor::new(&[[[[0f32, 1.0], [2.0, 3.0]]]], &Device::Cpu)?;
    let grid = Tensor::new(&[[[[0f32, 0.0], [-1.0, -1.0]]]], &Device::Cpu)?;
    let out = run(
        "GridSample",
        vec![("x", x), ("grid", grid)],
        vec![
            ("mode", Attr::Str("linear")),
            ("padding_mode", Attr::Str("zeros")),
            ("align_corners", Attr::Int(1)),
        ],
        &["y"],
    )?;
    let y = &out["y"];
    assert_eq!(y.dims(), &[1, 1, 1, 2]);
    assert_eq!(floats(y), vec![1.5, 0.0]);
    Ok(())
}

/// The defaults are `linear`, `zeros` and unaligned corners: (-1, -1) is then
/// the outer corner, a quarter of pixel 0 and nothing else.
#[test]
fn grid_sample_defaults() -> Result<()> {
    let x = Tensor::new(&[[[[4f32, 1.0], [2.0, 3.0]]]], &Device::Cpu)?;
    let grid = Tensor::new(&[[[[-1f32, -1.0]]]], &Device::Cpu)?;
    let out = run("GridSample", vec![("x", x), ("grid", grid)], vec![], &["y"])?;
    assert_eq!(floats(&out["y"]), vec![1.0]);
    Ok(())
}

/// Each row of the last axis is centred and scaled to unit variance, then
/// scaled and shifted: [1, 3] → [-1, 1] (mean 2, variance 1), ×2 + 0.5.
#[test]
fn layer_normalization_over_the_last_axis() -> Result<()> {
    let x = Tensor::new(&[[1f32, 3.0], [10.0, 10.0]], &Device::Cpu)?;
    let scale = Tensor::new(&[2f32, 2.0], &Device::Cpu)?;
    let bias = Tensor::new(&[0.5f32, 0.5], &Device::Cpu)?;
    let out = run(
        "LayerNormalization",
        vec![("x", x), ("scale", scale), ("bias", bias)],
        vec![("epsilon", Attr::Float(0.0))],
        &["y"],
    )?;
    // The constant row has zero variance: with epsilon 0 it divides 0 by 0, so
    // only the first row is checked exactly; see the epsilon case below.
    let y = floats(&out["y"]);
    assert_eq!(&y[..2], &[-1.5, 2.5]);
    Ok(())
}

/// Epsilon keeps a constant row finite: it centres to 0 and stays there, so
/// the answer is the bias.
#[test]
fn layer_normalization_epsilon_and_axis() -> Result<()> {
    let x = Tensor::new(&[[[1f32, 1.0], [1.0, 1.0]]], &Device::Cpu)?;
    let scale = Tensor::new(&[[3f32, 3.0], [3.0, 3.0]], &Device::Cpu)?;
    let bias = Tensor::new(&[[7f32, 7.0], [7.0, 7.0]], &Device::Cpu)?;
    let out = run(
        "LayerNormalization",
        vec![("x", x), ("scale", scale), ("bias", bias)],
        vec![("axis", Attr::Int(1)), ("epsilon", Attr::Float(1e-5))],
        &["y", "mean", "inv_std"],
    )?;
    assert_eq!(floats(&out["y"]), vec![7.0; 4]);
    assert_eq!(out["mean"].dims(), &[1, 1, 1]);
    assert_eq!(floats(&out["mean"]), vec![1.0]);
    Ok(())
}

/// The mean over every spatial position, per channel, keeping the rank.
#[test]
fn global_average_pool() -> Result<()> {
    let x = Tensor::new(
        &[[[[1f32, 2.0], [3.0, 6.0]], [[0.0, 0.0], [0.0, 8.0]]]],
        &Device::Cpu,
    )?;
    let out = run("GlobalAveragePool", vec![("x", x)], vec![], &["y"])?;
    assert_eq!(out["y"].dims(), &[1, 2, 1, 1]);
    assert_eq!(floats(&out["y"]), vec![3.0, 2.0]);
    Ok(())
}

/// `fmod` 0 is an integer mod with the divisor's sign; 1 is C's fmod with the
/// dividend's — the operator definition's own examples.
#[test]
fn mod_takes_the_sign_the_attribute_asks_for() -> Result<()> {
    let a = Tensor::new(&[-4i64, 7, 5, 4, -7, 8], &Device::Cpu)?;
    let b = Tensor::new(&[2i64, -3, 8, -2, 3, 5], &Device::Cpu)?;
    let int = run(
        "Mod",
        vec![("a", a.clone()), ("b", b.clone())],
        vec![],
        &["y"],
    )?;
    assert_eq!(int["y"].to_vec1::<i64>()?, vec![0, -2, 5, 0, 2, 3]);
    assert_eq!(int["y"].dtype(), DType::I64);

    let a = Tensor::new(&[-4.3f32, 7.2, 5.0, 4.3, -7.2, 8.0], &Device::Cpu)?;
    let b = Tensor::new(&[2.1f32, -3.4, 8.0, -2.1, 3.4, 5.0], &Device::Cpu)?;
    let f = run(
        "Mod",
        vec![("a", a), ("b", b)],
        vec![("fmod", Attr::Int(1))],
        &["y"],
    )?;
    let got: Vec<f32> = floats(&f["y"])
        .iter()
        .map(|v| (v * 100.0).round() / 100.0)
        .collect();
    assert_eq!(got, vec![-0.1, 0.4, 5.0, 0.1, -0.4, 3.0]);
    Ok(())
}

/// `constant` pads with the third input's value; the pads are all the axes'
/// befores, then all their afters.
#[test]
fn pad_constant_takes_its_value() -> Result<()> {
    let x = Tensor::new(&[[1f32, 2.0]], &Device::Cpu)?;
    let pads = Tensor::new(&[0i64, 1, 0, 1], &Device::Cpu)?;
    let value = Tensor::new(9f32, &Device::Cpu)?;
    let out = run(
        "Pad",
        vec![("x", x), ("pads", pads), ("v", value)],
        vec![],
        &["y"],
    )?;
    assert_eq!(out["y"].to_vec2::<f32>()?, vec![vec![9.0, 1.0, 2.0, 9.0]]);
    Ok(())
}

/// `edge` repeats the outermost element; a negative pad crops instead.
#[test]
fn pad_edge_and_a_negative_pad() -> Result<()> {
    let x = Tensor::new(&[1f32, 2.0], &Device::Cpu)?;
    let pads = Tensor::new(&[2i64, 1], &Device::Cpu)?;
    let out = run(
        "Pad",
        vec![("x", x), ("pads", pads)],
        vec![("mode", Attr::Str("edge"))],
        &["y"],
    )?;
    assert_eq!(floats(&out["y"]), vec![1.0, 1.0, 1.0, 2.0, 2.0]);

    let x = Tensor::new(&[1f32, 2.0, 3.0, 4.0], &Device::Cpu)?;
    let pads = Tensor::new(&[-1i64, 1], &Device::Cpu)?;
    let out = run("Pad", vec![("x", x), ("pads", pads)], vec![], &["y"])?;
    assert_eq!(floats(&out["y"]), vec![2.0, 3.0, 4.0, 0.0]);
    Ok(())
}

/// With `axes`, the pads name only those axes.
#[test]
fn pad_along_named_axes() -> Result<()> {
    let x = Tensor::new(&[[1f32, 2.0], [3.0, 4.0]], &Device::Cpu)?;
    let pads = Tensor::new(&[1i64, 0], &Device::Cpu)?;
    let value = Tensor::new(0f32, &Device::Cpu)?;
    let axes = Tensor::new(&[0i64], &Device::Cpu)?;
    let out = run(
        "Pad",
        vec![("x", x), ("pads", pads), ("v", value), ("axes", axes)],
        vec![],
        &["y"],
    )?;
    assert_eq!(
        out["y"].to_vec2::<f32>()?,
        vec![vec![0.0, 0.0], vec![1.0, 2.0], vec![3.0, 4.0]]
    );
    Ok(())
}

#[test]
fn reciprocal() -> Result<()> {
    let x = Tensor::new(&[-4f32, 2.0, 0.5], &Device::Cpu)?;
    let out = run("Reciprocal", vec![("x", x)], vec![], &["y"])?;
    assert_eq!(floats(&out["y"]), vec![-0.25, 0.5, 2.0]);
    Ok(())
}
