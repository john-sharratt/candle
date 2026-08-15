//! Placeholder library for the `candle-wasm-tests` crate.
//!
//! This crate has no library code of its own; it exists so that
//! `tests/quantized_tests.rs` can run under `wasm-bindgen-test` in a real
//! browser, verifying quantized (GGML) matmul kernels compiled to
//! `wasm32-unknown-unknown` against fixed expected output. `add` below is an
//! unused scaffold left by `cargo new`.
pub fn add(left: usize, right: usize) -> usize {
    left + right
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn it_works() {
        let result = add(2, 2);
        assert_eq!(result, 4);
    }
}
