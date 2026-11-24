//! A wave's images, as this lineage's engine takes them (`docs/vision_models.md`
//! §2.3).
//!
//! The embedding is host-mapped: it gathers quantized rows across PCIe by id,
//! and has no second table to read an image's rows from. So the wave's inputs
//! go to it with every virtual id replaced by a placeholder, and the image rows
//! are copied over the placeholders' rows afterwards — the trunk's residual and
//! the draft head's embedding alike — by one `rows_scatter` launch each.

use candle::{Device, Result, Tensor};

use crate::models::qwen_vision::wave_images::{count_image_rows, take_wave_images, WaveImages};

/// The id an image position is gathered as before its row is copied over it.
/// Row 0 of every vocabulary, and never read: the copy replaces it whole
/// before any layer runs.
const PLACEHOLDER: u32 = 0;

/// The wave's images and the inputs the embedding reads in their place, or
/// `None` for a wave with none — whose inputs are used as they stand.
pub(super) struct WaveImageInputs {
    pub images: WaveImages,
    /// Each input as the embedding gathers it: a host input with its virtual
    /// ids replaced by [`PLACEHOLDER`], a device input — a sampled token, never
    /// an image's — as it was.
    pub inputs: Vec<Tensor>,
}

/// Resolve the virtual ids among `inputs`, rows `width` wide.
pub(super) fn wave_image_inputs(
    inputs: &[Tensor],
    width: usize,
) -> Result<Option<WaveImageInputs>> {
    let refs: Vec<&Tensor> = inputs.iter().collect();
    if count_image_rows(&refs)? == 0 {
        return Ok(None);
    }
    // Every input's ids in wave order, so each run's row is its row in the
    // wave; a device input's are not read and stand as placeholders.
    let mut seq_tokens: Vec<Vec<u32>> = inputs
        .iter()
        .map(|t| {
            if t.device().is_cuda() {
                Ok(vec![PLACEHOLDER; t.elem_count()])
            } else {
                t.flatten_all()?.to_vec1::<u32>()
            }
        })
        .collect::<Result<_>>()?;
    let Some(images) = take_wave_images(&mut seq_tokens, 0, PLACEHOLDER, width)? else {
        return Ok(None);
    };
    let inputs = inputs
        .iter()
        .zip(seq_tokens)
        .map(|(t, toks)| {
            if t.device().is_cuda() {
                Ok(t.clone())
            } else {
                Tensor::from_vec(toks, t.shape(), &Device::Cpu)
            }
        })
        .collect::<Result<_>>()?;
    Ok(Some(WaveImageInputs { images, inputs }))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::qwen_vision::geometry::FACTOR;
    use crate::models::qwen_vision::registry::{ImageHandle, ImageRows};
    use crate::models::qwen_vision::Grid;

    /// A text wave is left alone; a wave with an image has its virtual ids
    /// replaced by the placeholder in its host inputs, and its runs counted
    /// over every input in wave order.
    #[test]
    fn virtual_ids_become_placeholders_in_the_host_inputs() {
        let text = [Tensor::new(&[[5u32, 6]], &Device::Cpu).unwrap()];
        assert!(wave_image_inputs(&text, 2).unwrap().is_none());

        let grid = Grid {
            height: FACTOR,
            width: 2 * FACTOR,
        };
        let a = ImageHandle::register(ImageRows {
            grid,
            width: 2,
            values: vec![1.0, 2.0, 3.0, 4.0],
        })
        .unwrap();
        let inputs = [
            Tensor::new(&[[9u32]], &Device::Cpu).unwrap(),
            Tensor::new(&[[7u32, a.id(0), a.id(1), 8]], &Device::Cpu).unwrap(),
        ];
        let got = wave_image_inputs(&inputs, 2).unwrap().unwrap();
        assert_eq!(got.inputs[0].to_vec2::<u32>().unwrap(), [[9]]);
        assert_eq!(got.inputs[1].to_vec2::<u32>().unwrap(), [[7, 0, 0, 8]]);
        assert_eq!(got.images.values, [1.0, 2.0, 3.0, 4.0]);
        assert_eq!(got.images.runs.len(), 1);
        assert_eq!((got.images.runs[0].row, got.images.runs[0].len), (2, 2));
    }
}
