//! Browser demo: YOLOv8 object detection and pose estimation, compiled to
//! WebAssembly.
//!
//! This crate root wires up a pure-Rust Yew UI (`App`, built with Trunk)
//! backed by a `yew_agent` `Worker` that owns the `model` submodule's
//! `YoloV8`/`YoloV8Pose` heads, with `coco_classes::NAMES` for label lookup.
//! A second, JS-driven path (`src/bin/m.rs`) wraps `worker::Model` and
//! `worker::ModelPose` in `#[wasm_bindgen]` types `Model`/`ModelPose`, each
//! exposing `run(image_bytes, conf_threshold, iou_threshold) -> JSON`
//! (class-labeled boxes, or pose keypoints) for a plain WebWorker built via
//! `build-lib.sh`. Weights (`yolov8s.safetensors`) are `wget`-ed for the
//! Trunk build or fetched by JS and passed in for the WebWorker build.
mod app;
pub mod coco_classes;
pub mod model;
pub mod worker;
pub use app::App;
pub use worker::Worker;
