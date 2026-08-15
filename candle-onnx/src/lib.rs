//! ONNX model loading and evaluation on Candle tensors.
//!
//! [`onnx`] is the prost-generated protobuf schema for `ModelProto` (built
//! from the `.proto` definitions at build time via `OUT_DIR`). [`eval`]
//! interprets a parsed `ModelProto` graph node-by-node against Candle
//! tensors, mapping ONNX ops and dtypes to their Candle equivalents; use
//! [`read_file`] to load a `.onnx` file and [`simple_eval`] to run it.
use candle::Result;
use prost::Message;

pub mod onnx {
    include!(concat!(env!("OUT_DIR"), "/onnx.rs"));
}

pub mod eval;
pub use eval::{dtype, simple_eval};

pub fn read_file<P: AsRef<std::path::Path>>(p: P) -> Result<onnx::ModelProto> {
    let buf = std::fs::read(p)?;
    onnx::ModelProto::decode(buf.as_slice()).map_err(candle::Error::wrap)
}
