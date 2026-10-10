//! A weight file's identity, read without loading it.
//!
//! A GGUF header — the metadata and the tensor-info table — changes with any
//! quant, merge or recipe. A fine-tune that keeps every tensor's name, shape and
//! type and every metadata value would leave the header as it was, so the
//! fingerprint also takes [`SAMPLE_COUNT`] evenly spaced windows of the tensor
//! data. Reading a 125 GB checkpoint's header and 1 MiB of its data costs
//! milliseconds; hashing all of it would cost minutes on every start.

use std::fs::File;
use std::io::{self, Read, Seek, SeekFrom};
use std::path::Path;

use candle::quantized::gguf_file::Content;
use sha2::{Digest, Sha256};

/// How many windows of tensor data the fingerprint reads.
pub const SAMPLE_COUNT: u64 = 16;

/// How many bytes each window holds.
pub const SAMPLE_BYTES: u64 = 64 * 1024;

/// The fingerprint of the GGUF checkpoint at `path`.
///
/// SHA-256 over, in order: the domain tag `gguf-fingerprint v1\0`, the file
/// length (u64 little-endian), every header byte up to the tensor data, and the
/// tensor data — all of it when it is no longer than the windows would read,
/// otherwise [`SAMPLE_COUNT`] windows of [`SAMPLE_BYTES`], the first at the
/// start of the data and the last ending at the end of the file.
pub fn gguf_fingerprint(path: &Path) -> io::Result<[u8; 32]> {
    let mut file = File::open(path)?;
    let len = file.metadata()?.len();
    let content = Content::read(&mut file).map_err(|e| {
        io::Error::new(
            io::ErrorKind::InvalidData,
            format!("{} is not a readable GGUF: {e}", path.display()),
        )
    })?;
    let data_start = content.tensor_data_offset;
    if data_start > len {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!(
                "{} says its tensor data starts at byte {data_start}, past its end at {len}",
                path.display()
            ),
        ));
    }

    let mut hasher = Sha256::new();
    hasher.update(b"gguf-fingerprint v1\0");
    hasher.update(len.to_le_bytes());
    hash_range(&mut file, 0, data_start, &mut hasher)?;

    let data_len = len - data_start;
    if data_len <= SAMPLE_COUNT * SAMPLE_BYTES {
        hash_range(&mut file, data_start, data_len, &mut hasher)?;
    } else {
        let stride_span = data_len - SAMPLE_BYTES;
        for i in 0..SAMPLE_COUNT {
            let offset = data_start + stride_span * i / (SAMPLE_COUNT - 1);
            hash_range(&mut file, offset, SAMPLE_BYTES, &mut hasher)?;
        }
    }
    Ok(hasher.finalize().into())
}

/// SHA-256 of every byte of the file at `path` — for the small files a model
/// is also made of (an adapter's weights and config), where a sample would
/// save nothing worth the doubt.
pub fn file_digest(path: &Path) -> io::Result<[u8; 32]> {
    let mut file = File::open(path)?;
    let len = file.metadata()?.len();
    let mut hasher = Sha256::new();
    hash_range(&mut file, 0, len, &mut hasher)?;
    Ok(hasher.finalize().into())
}

/// Feed `len` bytes of `file` from `offset` into `hasher`.
fn hash_range(file: &mut File, offset: u64, len: u64, hasher: &mut Sha256) -> io::Result<()> {
    file.seek(SeekFrom::Start(offset))?;
    let mut buf = vec![0u8; 1 << 20];
    let mut left = len;
    while left > 0 {
        let take = left.min(buf.len() as u64) as usize;
        file.read_exact(&mut buf[..take])?;
        hasher.update(&buf[..take]);
        left -= take as u64;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    use std::io::Write;

    use candle::quantized::gguf_file::{self, Value};
    use candle::quantized::{GgmlDType, QTensor};
    use candle::{Device, Tensor};

    /// A one-tensor GGUF whose tensor is `values`, F32, so every data byte is
    /// known.
    fn write_gguf(path: &Path, name: &str, values: &[f32]) {
        let t = Tensor::from_slice(values, values.len(), &Device::Cpu).unwrap();
        let q = QTensor::quantize(&t, GgmlDType::F32).unwrap();
        let arch = Value::String("fixture".into());
        let mut f = File::create(path).unwrap();
        gguf_file::write(&mut f, &[("general.architecture", &arch)], &[(name, &q)]).unwrap();
        f.flush().unwrap();
    }

    fn hex(d: &[u8; 32]) -> String {
        d.iter().map(|b| format!("{b:02x}")).collect()
    }

    /// The fingerprint of a fixed small GGUF is a fixed value: the
    /// definition — tag, length, header, whole data section — is pinned, so a
    /// change to it shows here before it makes every substrate refuse its own
    /// model.
    #[test]
    fn a_small_ggufs_fingerprint_is_a_fixed_value() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("m.gguf");
        write_gguf(&path, "w", &[1.0, 2.0, 3.0, 4.0]);
        assert_eq!(
            hex(&gguf_fingerprint(&path).unwrap()),
            "9e53d2f0e28bcf8a4aa17b6954aa286d05bcf0fc11834119c47e401abd6b37f1",
        );
    }

    /// The data is read, not only the header: two files that differ in one
    /// tensor value and nothing else have different fingerprints.
    #[test]
    fn a_changed_tensor_value_changes_the_fingerprint() {
        let dir = tempfile::tempdir().unwrap();
        let a = dir.path().join("a.gguf");
        let b = dir.path().join("b.gguf");
        write_gguf(&a, "w", &[1.0, 2.0, 3.0, 4.0]);
        write_gguf(&b, "w", &[1.0, 2.0, 3.0, 5.0]);
        assert_ne!(gguf_fingerprint(&a).unwrap(), gguf_fingerprint(&b).unwrap());
    }

    /// A data section longer than the windows is sampled: a value inside the
    /// last window changes the fingerprint, and one between windows does not —
    /// which is the trade the sampling makes, and the reason the header is
    /// hashed whole.
    #[test]
    fn a_large_data_section_is_sampled_by_windows() {
        let n = ((SAMPLE_COUNT * SAMPLE_BYTES * 2) / 4) as usize;
        let base: Vec<f32> = (0..n).map(|i| i as f32).collect();
        let dir = tempfile::tempdir().unwrap();
        let p0 = dir.path().join("base.gguf");
        write_gguf(&p0, "w", &base);
        let f0 = gguf_fingerprint(&p0).unwrap();

        let mut in_last = base.clone();
        in_last[n - 1] = -1.0;
        let p1 = dir.path().join("last.gguf");
        write_gguf(&p1, "w", &in_last);
        assert_ne!(
            gguf_fingerprint(&p1).unwrap(),
            f0,
            "the last window is read"
        );

        // The first window covers data bytes [0, 64 KiB); the second starts
        // past 128 KiB here, so float 20,000 (byte 80,000) lies between them.
        let mut between = base;
        between[20_000] = -1.0;
        let p2 = dir.path().join("between.gguf");
        write_gguf(&p2, "w", &between);
        assert_eq!(
            gguf_fingerprint(&p2).unwrap(),
            f0,
            "bytes between windows are not read"
        );
    }

    #[test]
    fn a_file_that_is_not_gguf_is_refused() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("not.gguf");
        std::fs::write(&path, b"plain text").unwrap();
        let err = gguf_fingerprint(&path).unwrap_err();
        assert_eq!(err.kind(), io::ErrorKind::InvalidData);
    }

    #[test]
    fn a_files_digest_is_its_sha256() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("adapter_config.json");
        std::fs::write(&path, b"abc").unwrap();
        assert_eq!(
            hex(&file_digest(&path).unwrap()),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad",
        );
    }
}
