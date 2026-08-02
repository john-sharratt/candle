//! Thin re-export surface over the raw Metal API types used to build and run
//! compute pipelines: `buffer` (`Buffer`), `device` (`Device`),
//! `command_buffer`/`commands` (command buffer + queue), `encoder` (compute/
//! blit command encoders), `compute_pipeline` (`ComputePipeline`), and
//! `library` (`Library`/`Function`, i.e. compiled kernel lookup). Everything
//! here is re-exported flat so `kernels/` and `metal_backend` can use short
//! names without reaching into each submodule.

pub mod buffer;
pub mod command_buffer;
pub mod commands;
pub mod compute_pipeline;
pub mod device;
pub mod encoder;
pub mod library;

pub use buffer::*;
pub use command_buffer::*;
pub use commands::*;
pub use compute_pipeline::*;
pub use device::*;
pub use encoder::*;
pub use library::*;
