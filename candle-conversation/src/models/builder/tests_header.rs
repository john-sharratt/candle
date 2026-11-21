//! Test that the builder reads its GGUF header through the process-wide cache.
//!
//! The builder asks for its header three times per engine build. The test checks
//! that a path the cache has not seen is parsed once and then held, so the repeated
//! reads do not each go back to the file.

use super::*;

#[cfg(test)]
pub(super) mod header_read_tests {
    use super::ModelBuilder;
    use crate::guest::checkpoint::cached_headers_for;
    use crate::models::ModelArch;
    use std::fs::File;
    use std::io::Write;
    use std::path::Path;

    /// A GGUF v3 header: no tensors, one metadata entry,
    /// `general.architecture = "qwen3"`.
    fn write_gguf(path: &Path) {
        let mut f = File::create(path).unwrap();
        f.write_all(b"GGUF").unwrap();
        f.write_all(&3u32.to_le_bytes()).unwrap(); // version
        f.write_all(&0u64.to_le_bytes()).unwrap(); // tensor count
        f.write_all(&1u64.to_le_bytes()).unwrap(); // metadata count
        let key = b"general.architecture";
        f.write_all(&(key.len() as u64).to_le_bytes()).unwrap();
        f.write_all(key).unwrap();
        f.write_all(&8u32.to_le_bytes()).unwrap(); // value type: string
        let value = b"qwen3";
        f.write_all(&(value.len() as u64).to_le_bytes()).unwrap();
        f.write_all(value).unwrap();
        f.sync_all().unwrap();
    }

    /// **The builder reads its header through the process-wide cache.** It asks
    /// for the header three times per engine build; read straight off an
    /// unbuffered `File`, that was a syscall per field of a 248k-token table,
    /// three times over, on every boot. A path the cache has never seen holds no
    /// entry, so one after the call is the builder's own parse, cached.
    #[test]
    fn metadata_is_read_through_the_header_cache() {
        let dir = std::env::temp_dir().join(format!(
            "builder-header-{}-{:?}",
            std::process::id(),
            std::thread::current().id()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        let p = dir.join("m.gguf");
        write_gguf(&p);
        assert_eq!(cached_headers_for(&p), 0);

        let info = ModelBuilder::detect_sampling_from_gguf(&p).unwrap();
        assert!(matches!(info.arch, Some(ModelArch::Qwen3)));
        assert_eq!(info.name, "<unknown>");
        assert_eq!(cached_headers_for(&p), 1);

        std::fs::remove_dir_all(&dir).ok();
    }
}
