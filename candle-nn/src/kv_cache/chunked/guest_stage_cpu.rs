//! [`guest_stage`] for a build with no GPU backend.
//!
//! A guest arena is carved out of device span — `open_guest_arena` takes a CUDA
//! stream — so without the `cuda` feature no arena can ever be open, and a stage
//! is what the CUDA build's `guest_stage` is for every caller that is not a
//! guest: `f(x)` and nothing else. The model code that marks its stages
//! (`stable_diffusion`'s resnets and decoder, `z_image`'s blocks) is shared by
//! both configurations, so the function is too.

use candle::{Result, Tensor};

/// Run `f` as one stage of a guest's pipeline. There is no guest arena without a
/// GPU backend, so the stage is `f(x)`.
pub fn guest_stage<F>(x: &Tensor, f: F) -> Result<Tensor>
where
    F: FnOnce(&Tensor) -> Result<Tensor>,
{
    f(x)
}

#[cfg(test)]
mod tests {
    use super::guest_stage;
    use candle::{Device, Result, Tensor};

    #[test]
    fn a_stage_is_its_closure_applied_to_its_input() -> Result<()> {
        let x = Tensor::new(&[1.0f32, -2.0, 3.5], &Device::Cpu)?;
        let out = guest_stage(&x, |x| x.affine(2.0, 1.0))?;
        assert_eq!(out.to_vec1::<f32>()?, vec![3.0, -3.0, 8.0]);
        Ok(())
    }

    #[test]
    fn a_stage_returns_its_closures_error() {
        let x = Tensor::new(&[0u32], &Device::Cpu).unwrap();
        let err = guest_stage(&x, |_| candle::bail!("stage failed")).unwrap_err();
        assert_eq!(err.to_string(), "stage failed");
    }
}
