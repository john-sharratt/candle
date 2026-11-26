//! Which model an expert belongs to, on a card whose expert cache every model on
//! it shares (`docs/parasite_model.md` §11.2).
//!
//! An expert is named `(model, row, expert)`: the model's index among those the
//! card has registered, its MoE row, and its expert in that row. Every map and
//! score the cache keeps is keyed that way, and every flat table is laid out one
//! model after another: a model's rows begin at its own base, so two models' row
//! 0 never share an entry.
//!
//! Models register in order and keep their index for the life of the card. Each
//! has its own row count and experts per row, since the models a card carries are
//! not the same shape.

use std::ops::Range;

/// A model's index among those the card's expert cache has registered.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct ModelIx(u16);

impl ModelIx {
    /// The first model a card registers.
    pub const FIRST: ModelIx = ModelIx(0);

    /// The index as a position in the card's registration order.
    pub fn index(self) -> usize {
        usize::from(self.0)
    }
}

/// One expert of one model on the card: its model, its MoE row and its expert.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct ExpertKey {
    pub model: ModelIx,
    pub row: usize,
    pub expert: usize,
}

impl ExpertKey {
    pub fn new(model: ModelIx, row: usize, expert: usize) -> Self {
        Self { model, row, expert }
    }
}

/// A registered model's place in the card's flat tables: its first entry, its
/// rows and its experts per row.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ModelShape {
    pub base: usize,
    pub rows: usize,
    pub experts: usize,
}

impl ModelShape {
    /// Entries the model takes in a flat table.
    pub fn len(&self) -> usize {
        self.rows * self.experts
    }

    /// Whether the model has no expert at all.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// The models a card's expert cache has registered, in order, and where each
/// one's experts lie in a flat table.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct CardModels {
    shapes: Vec<ModelShape>,
}

impl CardModels {
    /// A card carrying one model of `rows` MoE rows of `experts` each.
    pub fn one(rows: usize, experts: usize) -> Self {
        let mut models = Self::default();
        models.register(rows, experts);
        models
    }

    /// Register a model of `rows` rows of `experts` each, after every model
    /// already registered; its index is the next one.
    pub fn register(&mut self, rows: usize, experts: usize) -> ModelIx {
        let ix = ModelIx(
            u16::try_from(self.shapes.len()).expect("a card registers at most 65,536 models"),
        );
        self.shapes.push(ModelShape {
            base: self.total(),
            rows,
            experts,
        });
        ix
    }

    /// The shape `model` registered with.
    ///
    /// Panics for a model the card never registered: every key the cache is
    /// handed names a registered model.
    pub fn shape(&self, model: ModelIx) -> ModelShape {
        self.shapes[model.index()]
    }

    /// Entries a flat table over every registered model takes.
    pub fn total(&self) -> usize {
        self.shapes.last().map_or(0, |s| s.base + s.len())
    }

    /// `key`'s entry in a flat table over every registered model.
    #[inline]
    pub fn flat(&self, key: ExpertKey) -> usize {
        let s = self.shape(key.model);
        debug_assert!(
            key.row < s.rows && key.expert < s.experts,
            "{key:?} is outside its model's {} rows of {}",
            s.rows,
            s.experts
        );
        s.base + key.row * s.experts + key.expert
    }

    /// The entries of `model`'s row `row` in a flat table.
    pub fn row_span(&self, model: ModelIx, row: usize) -> Range<usize> {
        let s = self.shape(model);
        let start = s.base + row * s.experts;
        start..start + s.experts
    }

    /// Every registered model, in order.
    pub fn iter(&self) -> impl Iterator<Item = (ModelIx, ModelShape)> + '_ {
        self.shapes
            .iter()
            .enumerate()
            .map(|(i, s)| (ModelIx(i as u16), *s))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// **One model lays out exactly as the single-model tables did**: entry
    /// `row * experts + expert`, so a card of one model indexes as before.
    #[test]
    fn one_model_indexes_as_row_times_experts_plus_expert() {
        let m = CardModels::one(48, 128);
        assert_eq!(m.total(), 6_144);
        assert_eq!(m.flat(ExpertKey::new(ModelIx::FIRST, 0, 0)), 0);
        assert_eq!(m.flat(ExpertKey::new(ModelIx::FIRST, 3, 5)), 389);
        assert_eq!(m.flat(ExpertKey::new(ModelIx::FIRST, 47, 127)), 6_143);
        assert_eq!(m.row_span(ModelIx::FIRST, 2), 256..384);
    }

    /// **A second model begins after the first**, in its own shape, so the two
    /// models' row 0 never share an entry.
    #[test]
    fn a_second_model_lies_after_the_first_in_its_own_shape() {
        let mut m = CardModels::one(48, 512);
        let second = m.register(40, 256);
        assert_eq!(second.index(), 1);
        assert_eq!(
            m.shape(second),
            ModelShape {
                base: 24_576,
                rows: 40,
                experts: 256
            }
        );
        assert_eq!(m.total(), 24_576 + 10_240);
        assert_eq!(m.flat(ExpertKey::new(second, 0, 0)), 24_576);
        assert_eq!(m.flat(ExpertKey::new(second, 1, 3)), 24_835);
        assert_eq!(m.row_span(second, 39), 34_560..34_816);
        let all: Vec<usize> = m.iter().map(|(ix, _)| ix.index()).collect();
        assert_eq!(all, [0, 1]);
    }

    /// Keys order by model, then row, then expert.
    #[test]
    fn keys_order_by_model_row_then_expert() {
        let a = ExpertKey::new(ModelIx::FIRST, 9, 9);
        let b = ExpertKey::new(ModelIx(1), 0, 0);
        assert!(a < b);
        assert!(ExpertKey::new(ModelIx::FIRST, 1, 0) > ExpertKey::new(ModelIx::FIRST, 0, 127));
    }
}
