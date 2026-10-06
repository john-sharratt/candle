//! Eager sections of a forward recorded as a wave capture.
//!
//! While a thread records a wave (`CudaDevice::begin_wave_capture`), its
//! launches are recorded rather than executed. Host work that meets the device
//! directly — building and uploading a table, waiting on a result — has to see
//! everything issued before it executed, so it runs inside an [`Eager`]
//! section: the recorded launches go to the driver first, and recording
//! resumes when the guard drops. On every other device, and on a thread that is
//! not recording, the guard does nothing.

#[cfg(feature = "cuda")]
use crate::cuda_backend::graph::Paused;
use crate::{Device, Result};
use std::marker::PhantomData;

/// Recording is suspended on this thread until this drops. See [`Device::eager`].
#[must_use = "the section is eager only while the guard lives"]
pub struct Eager<'a> {
    #[cfg(feature = "cuda")]
    _paused: Option<Paused<'a>>,
    _device: PhantomData<&'a Device>,
}

impl Device {
    /// Run what follows eagerly, in issue order behind every launch this
    /// thread has recorded, until the guard drops.
    pub fn eager(&self) -> Result<Eager<'_>> {
        match self {
            #[cfg(feature = "cuda")]
            Device::Cuda(d) => Ok(Eager {
                _paused: d.pause_capture()?,
                _device: PhantomData,
            }),
            _ => Ok(Eager {
                #[cfg(feature = "cuda")]
                _paused: None,
                _device: PhantomData,
            }),
        }
    }

    /// Mark where a forward's recorded launches begin: a wave capture opens
    /// held, so everything before this ran eagerly. Does nothing on another
    /// device or on a thread with no wave capture open.
    pub fn record_launches(&self) -> Result<()> {
        match self {
            #[cfg(feature = "cuda")]
            Device::Cuda(d) => d.record_launches(),
            _ => Ok(()),
        }
    }

    /// Hand what this thread has recorded to the device now, so it starts
    /// executing while the rest of the forward records. Does nothing on
    /// another device or on a thread with no wave capture open.
    pub fn flush_launches(&self) -> Result<()> {
        match self {
            #[cfg(feature = "cuda")]
            Device::Cuda(d) => d.flush_launches(),
            _ => Ok(()),
        }
    }
}
