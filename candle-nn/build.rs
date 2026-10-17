//! Refuses sources that reach the CUDA legacy null stream — see
//! `lints/null_stream.rs` for the rule and why it holds here.

include!("../lints/null_stream.rs");

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=../lints/null_stream.rs");
    forbid_null_stream(&[
        std::path::Path::new("src"),
        std::path::Path::new("tests"),
        std::path::Path::new("examples"),
        std::path::Path::new("benches"),
    ]);
}
