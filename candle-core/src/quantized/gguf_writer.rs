//! A GGUF file laid out before it is written, then streamed.
//!
//! [`super::gguf_file::write`] needs every tensor materialised as a [`QTensor`], and
//! [`super::gguf_file::merge_split_ggufs`] fixes the alignment at 32 and computes
//! offsets as it goes. A model pack needs neither: its tensors are copied byte for
//! byte out of source files far larger than RAM, and it needs to know, **before
//! the first byte is written**, exactly where the GGUF part ends — because the
//! sections that follow it are placed by offsets recorded in the GGUF's own
//! metadata.
//!
//! So the layout is a value ([`GgufPlan`]): metadata and tensor directory in, header
//! bytes and per-tensor offsets out, all computed without touching a file. A
//! [`GgufStreamWriter`] then takes tensors in plan order and checks every one lands
//! where the plan said.
//!
//! [`QTensor`]: super::QTensor

use super::gguf_file::{write_string, Value, DEFAULT_ALIGNMENT};
use super::GgmlDType;
use crate::Result;
use byteorder::{LittleEndian, WriteBytesExt};
use std::io::Write;

/// The metadata key that carries a non-default alignment.
const ALIGNMENT_KEY: &str = "general.alignment";

/// One tensor in the directory: its name, type and shape (outermost dimension
/// first, as candle's `Shape` orders it — the file stores them reversed).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PlannedTensor {
    pub name: String,
    pub dtype: GgmlDType,
    pub dims: Vec<usize>,
}

impl PlannedTensor {
    /// Bytes of tensor data, unpadded.
    pub fn byte_len(&self) -> Result<u64> {
        let elems: usize = self.dims.iter().product();
        let block = self.dtype.block_size();
        if !elems.is_multiple_of(block) {
            crate::bail!(
                "gguf plan: {} has {elems} elements, not a multiple of {:?}'s block of {block}",
                self.name,
                self.dtype
            );
        }
        Ok((elems / block * self.dtype.type_size()) as u64)
    }
}

/// A GGUF file's layout: metadata, tensor directory and alignment, with every
/// offset derived from them.
#[derive(Debug, Clone)]
pub struct GgufPlan {
    alignment: u64,
    metadata: Vec<(String, Value)>,
    tensors: Vec<PlannedTensor>,
}

impl GgufPlan {
    /// An empty plan aligned to `alignment` bytes. Anything but the GGUF default
    /// is recorded as `general.alignment`, which is how a reader learns it.
    pub fn new(alignment: u64) -> Result<Self> {
        if alignment == 0 || !alignment.is_power_of_two() {
            crate::bail!("gguf plan: alignment {alignment} is not a power of two");
        }
        let metadata = if alignment == DEFAULT_ALIGNMENT {
            Vec::new()
        } else {
            vec![(ALIGNMENT_KEY.to_string(), Value::U32(alignment as u32))]
        };
        Ok(Self {
            alignment,
            metadata,
            tensors: Vec::new(),
        })
    }

    pub fn alignment(&self) -> u64 {
        self.alignment
    }

    /// Add a metadata entry. `general.alignment` is the plan's own and is
    /// refused here: a source file's value would contradict the layout.
    pub fn push_metadata(&mut self, key: impl Into<String>, value: Value) -> Result<()> {
        let key = key.into();
        if key == ALIGNMENT_KEY {
            crate::bail!("gguf plan: {ALIGNMENT_KEY} is set by the plan, not by a caller");
        }
        if self.metadata.iter().any(|(k, _)| *k == key) {
            crate::bail!("gguf plan: metadata key {key} given twice");
        }
        self.metadata.push((key, value));
        Ok(())
    }

    /// Add a tensor to the directory. Tensors are written in the order pushed.
    pub fn push_tensor(&mut self, tensor: PlannedTensor) -> Result<()> {
        if self.tensors.iter().any(|t| t.name == tensor.name) {
            crate::bail!("gguf plan: tensor {} given twice", tensor.name);
        }
        tensor.byte_len()?;
        self.tensors.push(tensor);
        Ok(())
    }

    pub fn tensors(&self) -> &[PlannedTensor] {
        &self.tensors
    }

    fn pad(&self, n: u64) -> u64 {
        n.div_ceil(self.alignment) * self.alignment - n
    }

    /// Each tensor's offset from the start of the data section, in plan order.
    pub fn tensor_offsets(&self) -> Result<Vec<u64>> {
        let mut at = 0u64;
        let mut out = Vec::with_capacity(self.tensors.len());
        for t in &self.tensors {
            out.push(at);
            let len = t.byte_len()?;
            at += len + self.pad(len);
        }
        Ok(out)
    }

    /// The header — magic through tensor directory — padded to the alignment, so
    /// the data section starts at `header_bytes().len()`.
    pub fn header_bytes(&self) -> Result<Vec<u8>> {
        let mut w = Vec::new();
        w.write_u32::<LittleEndian>(0x46554747)?;
        w.write_u32::<LittleEndian>(3)?;
        w.write_u64::<LittleEndian>(self.tensors.len() as u64)?;
        w.write_u64::<LittleEndian>(self.metadata.len() as u64)?;
        for (key, value) in &self.metadata {
            write_string(&mut w, key)?;
            w.write_u32::<LittleEndian>(value.value_type().to_u32())?;
            value.write(&mut w)?;
        }
        for (t, offset) in self.tensors.iter().zip(self.tensor_offsets()?) {
            write_string(&mut w, &t.name)?;
            w.write_u32::<LittleEndian>(t.dims.len() as u32)?;
            for &d in t.dims.iter().rev() {
                w.write_u64::<LittleEndian>(d as u64)?;
            }
            w.write_u32::<LittleEndian>(t.dtype.to_gguf_file_code())?;
            w.write_u64::<LittleEndian>(offset)?;
        }
        let pad = self.pad(w.len() as u64) as usize;
        w.resize(w.len() + pad, 0);
        Ok(w)
    }

    /// The whole file's length: header, every tensor, and the padding after the
    /// last one — so whatever follows a GGUF written from this plan starts aligned.
    pub fn total_len(&self) -> Result<u64> {
        let header = self.header_bytes()?.len() as u64;
        let data: u64 = self
            .tensors
            .iter()
            .map(|t| t.byte_len().map(|l| l + self.pad(l)))
            .sum::<Result<u64>>()?;
        Ok(header + data)
    }
}

/// Writes a [`GgufPlan`]'s file: the header at construction, then each tensor's
/// bytes in plan order, checked against the plan's lengths.
pub struct GgufStreamWriter<W: Write> {
    plan: GgufPlan,
    w: W,
    next: usize,
    /// Bytes of the current tensor written so far.
    in_tensor: u64,
}

impl<W: Write> GgufStreamWriter<W> {
    /// Write `plan`'s header to `w`.
    pub fn new(plan: GgufPlan, mut w: W) -> Result<Self> {
        w.write_all(&plan.header_bytes()?)?;
        Ok(Self {
            plan,
            w,
            next: 0,
            in_tensor: 0,
        })
    }

    /// The tensor the next bytes belong to, or `None` once every tensor is done.
    pub fn current(&self) -> Option<&PlannedTensor> {
        self.plan.tensors.get(self.next)
    }

    /// Append bytes of the current tensor. A tensor may arrive in any number of
    /// pieces; the piece that completes it also writes its padding.
    pub fn write_tensor_bytes(&mut self, bytes: &[u8]) -> Result<()> {
        let Some(t) = self.plan.tensors.get(self.next) else {
            crate::bail!("gguf writer: {} bytes past the last tensor", bytes.len());
        };
        let len = t.byte_len()?;
        let after = self.in_tensor + bytes.len() as u64;
        if after > len {
            crate::bail!(
                "gguf writer: {} is {len} bytes, {after} were written to it",
                t.name
            );
        }
        self.w.write_all(bytes)?;
        self.in_tensor = after;
        if after == len {
            let pad = self.plan.pad(len) as usize;
            self.w.write_all(&vec![0u8; pad])?;
            self.next += 1;
            self.in_tensor = 0;
        }
        Ok(())
    }

    /// Finish the file, refusing one with tensors still unwritten, and hand the
    /// writer back — positioned at [`GgufPlan::total_len`], for whatever follows.
    pub fn finish(self) -> Result<W> {
        if let Some(t) = self.plan.tensors.get(self.next) {
            crate::bail!(
                "gguf writer: finished with {} at {} of {} bytes",
                t.name,
                self.in_tensor,
                t.byte_len()?
            );
        }
        Ok(self.w)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::quantized::gguf_file::Content;
    use std::io::Cursor;

    fn tiny(alignment: u64) -> GgufPlan {
        let mut plan = GgufPlan::new(alignment).unwrap();
        plan.push_metadata("a", Value::U32(7)).unwrap();
        plan.push_tensor(PlannedTensor {
            name: "t".into(),
            dtype: GgmlDType::F32,
            dims: vec![3],
        })
        .unwrap();
        plan
    }

    /// The header, byte for byte, at the default alignment — no
    /// `general.alignment` entry, a 12-byte tensor at offset 0.
    #[test]
    fn the_header_is_these_bytes() {
        let got = tiny(32).header_bytes().unwrap();
        let mut want: Vec<u8> = Vec::new();
        want.extend_from_slice(b"GGUF");
        want.extend_from_slice(&3u32.to_le_bytes());
        want.extend_from_slice(&1u64.to_le_bytes()); // tensors
        want.extend_from_slice(&1u64.to_le_bytes()); // metadata
        want.extend_from_slice(&1u64.to_le_bytes());
        want.extend_from_slice(b"a");
        want.extend_from_slice(&4u32.to_le_bytes()); // U32
        want.extend_from_slice(&7u32.to_le_bytes());
        want.extend_from_slice(&1u64.to_le_bytes());
        want.extend_from_slice(b"t");
        want.extend_from_slice(&1u32.to_le_bytes()); // one dimension
        want.extend_from_slice(&3u64.to_le_bytes());
        want.extend_from_slice(&0u32.to_le_bytes()); // F32
        want.extend_from_slice(&0u64.to_le_bytes()); // offset
        assert_eq!(want.len(), 74);
        want.resize(96, 0);
        assert_eq!(got, want);
        // 96 of header, then 12 of tensor padded to 32.
        assert_eq!(tiny(32).total_len().unwrap(), 128);
    }

    /// A non-default alignment is recorded as the first metadata entry, and
    /// pads both the header and every tensor to it.
    #[test]
    fn a_wide_alignment_is_recorded_and_applied() {
        let plan = tiny(4096);
        let header = plan.header_bytes().unwrap();
        assert_eq!(header.len(), 4096);
        assert_eq!(&header[16..24], &2u64.to_le_bytes()[..]); // two metadata entries
        assert_eq!(&header[24..32], &17u64.to_le_bytes()[..]);
        assert_eq!(&header[32..49], ALIGNMENT_KEY.as_bytes());
        assert_eq!(&header[49..53], &4u32.to_le_bytes()[..]); // U32
        assert_eq!(&header[53..57], &4096u32.to_le_bytes()[..]);
        assert_eq!(plan.total_len().unwrap(), 8192);
    }

    /// What the writer emits reads back through the GGUF reader with the
    /// plan's offsets, data and alignment.
    #[test]
    fn a_streamed_file_reads_back() {
        let mut plan = GgufPlan::new(4096).unwrap();
        plan.push_metadata("k", Value::String("v".into())).unwrap();
        for (name, n) in [("x", 4usize), ("y", 2)] {
            plan.push_tensor(PlannedTensor {
                name: name.into(),
                dtype: GgmlDType::F32,
                dims: vec![n],
            })
            .unwrap();
        }
        let total = plan.total_len().unwrap();
        let mut w = GgufStreamWriter::new(plan, Cursor::new(Vec::new())).unwrap();
        let x: Vec<u8> = [1f32, 2., 3., 4.]
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        w.write_tensor_bytes(&x[..5]).unwrap();
        w.write_tensor_bytes(&x[5..]).unwrap();
        let y: Vec<u8> = [5f32, 6.].iter().flat_map(|v| v.to_le_bytes()).collect();
        w.write_tensor_bytes(&y).unwrap();
        let bytes = w.finish().unwrap().into_inner();
        assert_eq!(bytes.len() as u64, total);

        let content = Content::read(&mut Cursor::new(&bytes)).unwrap();
        assert_eq!(content.tensor_data_offset, 4096);
        assert_eq!(
            content.metadata["general.alignment"].to_u32().unwrap(),
            4096
        );
        assert_eq!(content.metadata["k"].to_string().unwrap(), "v");
        let start = (content.tensor_data_offset + content.tensor_infos["y"].offset) as usize;
        assert_eq!(content.tensor_infos["y"].offset, 4096);
        assert_eq!(&bytes[start..start + 8], y.as_slice());
    }

    #[test]
    fn a_short_or_long_tensor_is_refused() {
        let mut w = GgufStreamWriter::new(tiny(32), Vec::new()).unwrap();
        assert!(w.write_tensor_bytes(&[0u8; 13]).is_err());
        let mut w = GgufStreamWriter::new(tiny(32), Vec::new()).unwrap();
        w.write_tensor_bytes(&[0u8; 8]).unwrap();
        assert!(w.finish().is_err());
    }

    #[test]
    fn the_alignment_key_belongs_to_the_plan() {
        let mut plan = GgufPlan::new(4096).unwrap();
        assert!(plan.push_metadata(ALIGNMENT_KEY, Value::U32(32)).is_err());
        assert!(plan.push_metadata("a", Value::U8(1)).is_ok());
        assert!(plan.push_metadata("a", Value::U8(2)).is_err());
    }
}
