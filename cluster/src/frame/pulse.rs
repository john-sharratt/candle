//! `PULSE` (§9.1): a transient world event — never persisted, no `oseq`, no
//! retransmit, the first thing dropped under pressure.

use crate::wire::{Reader, WireError, Writer};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Pulse {
    pub world: String,
    pub kind: String,
    pub body: Vec<u8>,
}

impl Pulse {
    pub fn write(&self, w: &mut Writer) {
        w.str(&self.world).str(&self.kind).blob(&self.body);
    }

    pub fn read(r: &mut Reader) -> Result<Self, WireError> {
        Ok(Self {
            world: r.str()?,
            kind: r.str()?,
            body: r.blob_vec()?,
        })
    }
}
