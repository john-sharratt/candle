//! Opt-in post-layer residual capture for offline activation analysis.
//!
//! The recorder never changes an activation or the forward result. It is
//! disabled by default; when enabled, selected prefill rows are copied to host
//! memory after each completed transformer layer.

use std::fs::{File, OpenOptions};
use std::io::{self, BufWriter, Write};
use std::path::Path;

use candle::{DType, Tensor};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CapturePhase {
    Prefill,
    Decode,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ActivationRecord {
    pub sequence_id: usize,
    pub position: usize,
    pub layer: usize,
    pub hidden: usize,
    pub values: Vec<f32>,
    pub phase: CapturePhase,
}

pub struct ActivationSink {
    writer: BufWriter<File>,
}

impl ActivationSink {
    pub fn open(path: impl AsRef<Path>) -> io::Result<Self> {
        let file = OpenOptions::new().create(true).append(true).open(path)?;
        Ok(Self {
            writer: BufWriter::new(file),
        })
    }

    pub fn record_prefill_row(
        &mut self,
        sequence_id: usize,
        position: usize,
        layer: usize,
        row: &Tensor,
        phase: CapturePhase,
    ) -> candle::Result<()> {
        let values = row
            .flatten_all()?
            .to_dtype(DType::F32)?
            .to_vec1::<f32>()?;
        let record = ActivationRecord {
            sequence_id,
            position,
            layer,
            hidden: values.len(),
            values,
            phase,
        };
        let payload = bincode::serialize(&record)
            .map_err(|error| candle::Error::Msg(format!("activation capture: {error}")))?;
        let length = u32::try_from(payload.len())
            .map_err(|_| candle::Error::Msg("activation capture record is too large".into()))?;
        self.writer
            .write_all(&length.to_le_bytes())
            .map_err(|error| candle::Error::Msg(format!("activation capture write: {error}")))?;
        self.writer
            .write_all(&payload)
            .map_err(|error| candle::Error::Msg(format!("activation capture write: {error}")))?;
        self.writer
            .flush()
            .map_err(|error| candle::Error::Msg(format!("activation capture flush: {error}")))?;
        Ok(())
    }
}

