//! Model-side RoPE (cos, sin) tables, built on first read.
//!
//! The paged attention kernels rotate Q and K themselves, at each sequence's
//! own positions under its rung, so a forward whose caches are paged never
//! reads model-side tables. Only the non-paged fallback rotates on the model
//! side. Building the tables up front cost every wave a position upload and a
//! gather per attention group for tensors nothing consumed; a `LazyRope` holds
//! the recipe instead and runs it the first time a layer asks, then serves the
//! same tables to every later layer of the forward.

use std::cell::OnceCell;

use candle::{Result, Tensor};

/// Builds the (cos, sin) pair for one attention group of one forward.
type BuildTables<'a> = Box<dyn Fn() -> Result<(Tensor, Tensor)> + 'a>;

/// One attention group's model-side RoPE tables, built at most once per
/// forward and only if a layer takes the non-paged path.
pub struct LazyRope<'a> {
    build: BuildTables<'a>,
    tables: OnceCell<(Tensor, Tensor)>,
}

impl<'a> LazyRope<'a> {
    /// A lazy pair whose tables `build` produces on first read.
    pub fn new(build: impl Fn() -> Result<(Tensor, Tensor)> + 'a) -> Self {
        Self {
            build: Box::new(build),
            tables: OnceCell::new(),
        }
    }

    /// The (cos, sin) tables, building them on the first call.
    pub fn tables(&self) -> Result<(&Tensor, &Tensor)> {
        let (cos, sin) = match self.tables.get() {
            Some(tables) => tables,
            None => {
                let built = (self.build)()?;
                self.tables.get_or_init(|| built)
            }
        };
        Ok((cos, sin))
    }

    /// Whether the tables have been built.
    pub fn is_built(&self) -> bool {
        self.tables.get().is_some()
    }
}

#[cfg(test)]
mod tests {
    use std::cell::Cell;

    use candle::{Device, Tensor};

    use super::LazyRope;

    #[test]
    fn nothing_is_built_until_a_layer_reads_the_tables() {
        let calls = Cell::new(0u32);
        let rope = LazyRope::new(|| {
            calls.set(calls.get() + 1);
            Ok((
                Tensor::new(&[1f32, 2.0], &Device::Cpu)?,
                Tensor::new(&[3f32, 4.0], &Device::Cpu)?,
            ))
        });
        assert!(!rope.is_built());
        assert_eq!(calls.get(), 0);
        drop(rope);
        assert_eq!(calls.get(), 0);
    }

    #[test]
    fn the_tables_are_built_once_and_served_to_every_later_read() {
        let calls = Cell::new(0u32);
        let rope = LazyRope::new(|| {
            calls.set(calls.get() + 1);
            Ok((
                Tensor::new(&[1f32, 2.0], &Device::Cpu)?,
                Tensor::new(&[3f32, 4.0], &Device::Cpu)?,
            ))
        });
        for _ in 0..3 {
            let (cos, sin) = rope.tables().unwrap();
            assert_eq!(cos.to_vec1::<f32>().unwrap(), vec![1.0, 2.0]);
            assert_eq!(sin.to_vec1::<f32>().unwrap(), vec![3.0, 4.0]);
        }
        assert!(rope.is_built());
        assert_eq!(calls.get(), 1);
    }

    #[test]
    fn a_failed_build_is_retried_on_the_next_read() {
        let calls = Cell::new(0u32);
        let rope = LazyRope::new(|| {
            calls.set(calls.get() + 1);
            if calls.get() == 1 {
                candle::bail!("first build fails");
            }
            Ok((
                Tensor::new(&[5f32], &Device::Cpu)?,
                Tensor::new(&[6f32], &Device::Cpu)?,
            ))
        });
        assert!(rope.tables().is_err());
        assert!(!rope.is_built());
        let (cos, sin) = rope.tables().unwrap();
        assert_eq!(cos.to_vec1::<f32>().unwrap(), vec![5.0]);
        assert_eq!(sin.to_vec1::<f32>().unwrap(), vec![6.0]);
        assert_eq!(calls.get(), 2);
    }
}
