# candle-book

mdBook source for "Candle Documentation" (`book.toml`: title "Candle
Documentation", author Nicolas Patry, `src = "src"`). Excluded from the
Cargo workspace (`Cargo.toml` `[workspace] exclude`) since it is a book
project, not a Rust crate that other code depends on — `src/lib.rs` exists
only so the book's embedded code samples (`src/simplified.rs`) type-check as
doctests.

**This is upstream Candle documentation, inherited as-is.** It documents
general candle usage (getting started, MNIST training, generic model
inference, CUDA porting, WASM/REST/desktop apps) and predates this fork's
unbounded-context architecture — it does not cover provenance-selected
attention, the three-tier KV cache, adaptive quantization, or Markov expert
prediction. For this fork's actual design, read `docs/*.md` and the root
`README.md` instead.

## Chapter layout (`src/SUMMARY.md`)

| Path | Covers |
|---|---|
| `guide/` | Installation, hello-world, cheatsheet |
| `guide/mnist/` | MNIST intro, modeling, training, saving/loading |
| `inference/` | Generic model inference, Hub loading, CUDA porting/writing |
| `training/` | Training loop, fine-tuning, serialization, simplified example |
| `apps/` | Desktop, REST server, WASM app walkthroughs |
| `cuda/`, `advanced/mkl.md` | CUDA kernel porting/writing, MKL backend |
| `error_manage.md`, `tracing.md` | Error handling, tracing/profiling |

## Building

```bash
cargo install mdbook
mdbook build candle-book      # renders to candle-book/book/
mdbook serve candle-book      # live-reload local server
```
