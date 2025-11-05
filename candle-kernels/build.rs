use std::env;
use std::path::PathBuf;

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=src/compatibility.cuh");
    println!("cargo:rerun-if-changed=src/cuda_utils.cuh");
    println!("cargo:rerun-if-changed=src/binary_op_macros.cuh");
    println!("cargo:rerun-if-changed=src/add_at_indices.cu");
    println!("cargo:rerun-if-changed=src/sub_at_indices.cu");
    println!("cargo:rerun-if-changed=src/sub_at_indices_with_values.cu");
    println!("cargo:rerun-if-changed=src/div_at_indices.cu");
    println!("cargo:rerun-if-changed=src/mul_at_indices.cu");
    println!("cargo:rerun-if-changed=src/multinomial.cu");
    println!("cargo:rerun-if-changed=src/max_abs_in_range.cu");

    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
    let ptx_path = out_dir.join("ptx.rs");
    let builder = bindgen_cuda::Builder::default();
    println!("cargo:info={builder:?}");
    let bindings = builder.build_ptx().unwrap();
    bindings.write(ptx_path).unwrap();
}
