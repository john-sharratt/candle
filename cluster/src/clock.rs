//! The wall clock, in milliseconds since the Unix epoch: what `HELLO.now_ms`,
//! claim epochs and the coordination log read.

use std::time::{SystemTime, UNIX_EPOCH};

pub fn now_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_millis() as u64)
}
