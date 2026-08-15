# candle-wasm-tests

Browser-executed test crate for candle's WASM target. Workspace member with
no library code of its own beyond a placeholder (`src/lib.rs` exports a
trivial `add`); the real content is `tests/quantized_tests.rs`, which runs
under `wasm-bindgen-test` (`#[wasm_bindgen_test] wasm_bindgen_test_configure!(run_in_browser)`).

## What it verifies

Quantized (GGML) matmul correctness compiled to `wasm32` and executed inside
a real browser: `quantized_matmul_neg` builds a `BlockQ4_0`-quantized RHS via
`candle::quantized::k_quants`, compares the raw `k_quants::matmul` kernel
output, `Tensor::matmul` on the dequantized float path, and
`QMatMul::from_qtensor(...).forward(...)` against fixed expected values
(exact rounded output, not error-tolerance). This guards against
architecture-specific SIMD/codegen divergence in the quantized kernels when
targeting `wasm32-unknown-unknown` instead of native.

## Running

```bash
RUST_LOG=wasm_bindgen_test_runner wasm-pack test --chrome --headless
```

or interactively (keeps the browser open):

```bash
wasm-pack test --chrome
```

Requires `wasm-pack` and a Chrome/Chromedriver pair at matching versions
(`webdriver.json` configures the driver); an "invalid session id" failure in
headless mode usually means a ChromeDriver/Chrome version mismatch — check
`chromedriver`'s logs first.
