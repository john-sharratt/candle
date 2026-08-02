//! Image classification dataset loaders.
//!
//! Defines the shared [`Dataset`] struct (train/test image and label
//! tensors, plus the label count) returned by each loader in [`cifar`],
//! [`fashion_mnist`], and [`mnist`].
use candle::Tensor;

pub struct Dataset {
    pub train_images: Tensor,
    pub train_labels: Tensor,
    pub test_images: Tensor,
    pub test_labels: Tensor,
    pub labels: usize,
}

pub mod cifar;
pub mod fashion_mnist;
pub mod mnist;
