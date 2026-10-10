//! Every file a model's weights are loaded from, and their joint digest.
//!
//! A model is not always one file. The AntiLoop + StyleTune fine-tune reads its
//! trunk from one checkpoint and its `output.weight` from another; a fine-tune
//! that stored its recurrent gates too narrow reads them from the base
//! checkpoint; a dense Qwen3.5 can carry adapters. The builder resolves exactly
//! the files its loader will read into a [`WeightFiles`], and the same value
//! feeds both the load and the key, so the key names what was loaded.

use std::io;
use std::path::{Path, PathBuf};

use candle_transformers::models::qwen35::TensorOverride;
use sha2::{Digest, Sha256};

use super::fingerprint::{file_digest, gguf_fingerprint};

/// The files a model is loaded from.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WeightFiles {
    /// The checkpoint itself.
    pub primary: PathBuf,
    /// Tensors read from another checkpoint instead of the primary, in the
    /// order the spec names them.
    pub tensor_overrides: Vec<TensorOverride>,
    /// The base checkpoint the recurrent gates are read from, when the primary
    /// stores them below F32.
    pub gate_donor: Option<PathBuf>,
    /// Adapters, as `(name, directory)`.
    pub adapters: Vec<(String, PathBuf)>,
}

impl WeightFiles {
    /// A model loaded from one checkpoint and nothing else.
    pub fn single(primary: impl Into<PathBuf>) -> Self {
        Self {
            primary: primary.into(),
            tensor_overrides: Vec::new(),
            gate_donor: None,
            adapters: Vec::new(),
        }
    }

    /// The digest of the weights: SHA-256 over the tag `weights v1\0`, then
    /// each part tagged with its role — `primary\0` and the primary's
    /// [`gguf_fingerprint`]; per override `override\0<tensor>\0` and its
    /// file's fingerprint; `gate-donor\0` and the donor's; per adapter
    /// `adapter\0<name>\0`, then each regular file in its directory, by name,
    /// as `<file name>\0` and its [`file_digest`].
    pub fn digest(&self) -> io::Result<[u8; 32]> {
        let mut hasher = Sha256::new();
        hasher.update(b"weights v1\0");
        hasher.update(b"primary\0");
        hasher.update(gguf_fingerprint(&self.primary)?);
        for o in &self.tensor_overrides {
            hasher.update(b"override\0");
            hasher.update(o.tensor.as_bytes());
            hasher.update(b"\0");
            hasher.update(gguf_fingerprint(&o.path)?);
        }
        if let Some(donor) = &self.gate_donor {
            hasher.update(b"gate-donor\0");
            hasher.update(gguf_fingerprint(donor)?);
        }
        for (name, dir) in &self.adapters {
            hasher.update(b"adapter\0");
            hasher.update(name.as_bytes());
            hasher.update(b"\0");
            for file in adapter_files(dir)? {
                let file_name = file.file_name().map(|n| n.to_string_lossy().into_owned());
                hasher.update(file_name.unwrap_or_default().as_bytes());
                hasher.update(b"\0");
                hasher.update(file_digest(&file)?);
            }
        }
        Ok(hasher.finalize().into())
    }
}

/// The regular files directly in `dir`, sorted by name.
fn adapter_files(dir: &Path) -> io::Result<Vec<PathBuf>> {
    let mut files = Vec::new();
    for entry in std::fs::read_dir(dir)? {
        let entry = entry?;
        if entry.file_type()?.is_file() {
            files.push(entry.path());
        }
    }
    files.sort();
    Ok(files)
}

#[cfg(test)]
mod tests {
    use super::*;

    use std::fs::File;
    use std::io::Write;

    use candle::quantized::gguf_file::{self, Value};
    use candle::quantized::{GgmlDType, QTensor};
    use candle::{Device, Tensor};

    fn write_gguf(path: &Path, values: &[f32]) {
        let t = Tensor::from_slice(values, values.len(), &Device::Cpu).unwrap();
        let q = QTensor::quantize(&t, GgmlDType::F32).unwrap();
        let arch = Value::String("fixture".into());
        let mut f = File::create(path).unwrap();
        gguf_file::write(&mut f, &[("general.architecture", &arch)], &[("w", &q)]).unwrap();
        f.flush().unwrap();
    }

    /// Each part of a model counts: an override's file, the donor and an
    /// adapter's bytes each move the digest, and so does which tensor an
    /// override replaces.
    #[test]
    fn every_file_a_model_is_made_of_moves_the_digest() {
        let dir = tempfile::tempdir().unwrap();
        let primary = dir.path().join("trunk.gguf");
        let head = dir.path().join("head.gguf");
        let head2 = dir.path().join("head2.gguf");
        write_gguf(&primary, &[1.0, 2.0]);
        write_gguf(&head, &[3.0, 4.0]);
        write_gguf(&head2, &[3.0, 5.0]);

        let base = WeightFiles::single(&primary);
        let d0 = base.digest().unwrap();

        let with_head = WeightFiles {
            tensor_overrides: vec![TensorOverride::new("output.weight", &head)],
            ..base.clone()
        };
        let d1 = with_head.digest().unwrap();
        assert_ne!(d1, d0);

        let other_head = WeightFiles {
            tensor_overrides: vec![TensorOverride::new("output.weight", &head2)],
            ..base.clone()
        };
        assert_ne!(other_head.digest().unwrap(), d1);

        let other_tensor = WeightFiles {
            tensor_overrides: vec![TensorOverride::new("token_embd.weight", &head)],
            ..base.clone()
        };
        assert_ne!(other_tensor.digest().unwrap(), d1);

        let with_donor = WeightFiles {
            gate_donor: Some(head.clone()),
            ..base.clone()
        };
        assert_ne!(with_donor.digest().unwrap(), d0);

        let adapter = dir.path().join("lora");
        std::fs::create_dir(&adapter).unwrap();
        std::fs::write(adapter.join("adapter_config.json"), b"{}").unwrap();
        std::fs::write(adapter.join("adapter_model.safetensors"), b"v1").unwrap();
        let with_adapter = WeightFiles {
            adapters: vec![("style".into(), adapter.clone())],
            ..base.clone()
        };
        let d2 = with_adapter.digest().unwrap();
        assert_ne!(d2, d0);
        std::fs::write(adapter.join("adapter_model.safetensors"), b"v2").unwrap();
        assert_ne!(with_adapter.digest().unwrap(), d2);

        // And the digest is a function of the files: computed again, the same.
        assert_eq!(base.digest().unwrap(), d0);
    }
}
