use std::io::Result;

/// Compile `onnx.proto3` into the Rust types `eval` walks.
///
/// # Why the compiler is vendored
///
/// `prost-build` shells out to `protoc`, and upstream leaves finding it to the
/// machine — so building this crate meant installing a protobuf compiler by
/// hand, on every developer box and every CI runner, or the whole workspace
/// failed to build the moment anything depended on it. That is a poor trade for
/// one generated file: `protoc-bin-vendored` ships the binaries and hands over
/// a path, which makes the build hermetic and the dependency invisible.
///
/// `PROTOC` is still honoured if it is already set, so a deployment that wants
/// its own compiler keeps it.
fn main() -> Result<()> {
    if std::env::var_os("PROTOC").is_none() {
        match protoc_bin_vendored::protoc_bin_path() {
            Ok(p) => std::env::set_var("PROTOC", p),
            // Not fatal: an unusual target may have no vendored binary, and the
            // machine's own `protoc` is then the remaining hope. Saying so beats
            // failing later with "could not find protoc" and no explanation.
            Err(e) => println!(
                "cargo:warning=no vendored protoc for this target ({e}); falling back to PATH"
            ),
        }
    }
    prost_build::compile_protos(&["src/onnx.proto3"], &["src/"])?;
    Ok(())
}
