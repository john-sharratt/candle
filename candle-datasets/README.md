# candle-datasets

Dataset loaders and batching helpers for training/evaluation with candle;
consumed by training-oriented examples such as `mnist-training` and
`llama2-c`, not by the inference engine.

## What it does

Four modules, re-exported from `src/lib.rs`:

- **`batcher`** — `Batcher<I>`, an iterator adapter that groups an
  `Iterator<Item = Tensor>` (`Batcher::new1`) or
  `Iterator<Item = (Tensor, Tensor)>` (`Batcher::new2`) into fixed-size
  batches, with `.batch_size(n)` and `.return_last_incomplete_batch(bool)`
  builder methods. Re-exported at the crate root as `candle_datasets::Batcher`.
- **`hub`** — `from_hub(&Api, dataset_id)` fetches every `.parquet` sibling
  file for a Hugging Face Hub dataset (via the `refs/convert/parquet` ref)
  and returns `parquet::file::reader::SerializedFileReader<File>` handles;
  the `parquet::file::reader::FileReader` trait is re-exported so callers
  don't need `parquet` as a direct dependency.
- **`nlp::tinystories`** — loader for the TinyStories dataset (used by
  `llama2-c`).
- **`vision`** — `Dataset { train_images, train_labels, test_images,
  test_labels, labels }` plus loaders in `vision::mnist`,
  `vision::fashion_mnist`, and `vision::cifar` (used by `mnist-training`).

## How it's used

Depended on optionally by `candle-examples` (feature `candle-datasets`);
training examples pull `Dataset`/`Batcher` to iterate minibatches over
Tensors. Not on the inference (`zend`/`candle-conversation`) code path.
