mod ptx {
    include!(concat!(env!("OUT_DIR"), "/ptx.rs"));
}

#[repr(u32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Id {
    AddAtIndices,
    Affine,
    Binary,
    Cast,
    Conv,
    DivAtIndices,
    Fill,
    Indexing,
    Multinomial,
    MulAtIndices,
    Quantized,
    Reduce,
    RepeatPenalty,
    Sort,
    SubAtIndices,
    SubAtIndicesWithValues,
    Ternary,
    Unary,
}

pub const ALL_IDS: [Id; 18] = [
    Id::AddAtIndices,
    Id::Affine,
    Id::Binary,
    Id::Cast,
    Id::Conv,
    Id::DivAtIndices,
    Id::Fill,
    Id::Indexing,
    Id::Multinomial,
    Id::MulAtIndices,
    Id::Quantized,
    Id::Reduce,
    Id::RepeatPenalty,
    Id::Sort,
    Id::SubAtIndices,
    Id::SubAtIndicesWithValues,
    Id::Ternary,
    Id::Unary,
];

pub struct Module {
    index: usize,
    ptx: &'static str,
}

impl Module {
    pub fn index(&self) -> usize {
        self.index
    }

    pub fn ptx(&self) -> &'static str {
        self.ptx
    }
}

const fn module_index(id: Id) -> usize {
    let mut i = 0;
    while i < ALL_IDS.len() {
        if ALL_IDS[i] as u32 == id as u32 {
            return i;
        }
        i += 1;
    }
    panic!("id not found")
}

macro_rules! mdl {
    ($cst:ident, $id:ident) => {
        pub const $cst: Module = Module {
            index: module_index(Id::$id),
            ptx: ptx::$cst,
        };
    };
}

mdl!(ADD_AT_INDICES, AddAtIndices);
mdl!(AFFINE, Affine);
mdl!(BINARY, Binary);
mdl!(CAST, Cast);
mdl!(CONV, Conv);
mdl!(DIV_AT_INDICES, DivAtIndices);
mdl!(FILL, Fill);
mdl!(INDEXING, Indexing);
mdl!(MULTINOMIAL, Multinomial);
mdl!(MUL_AT_INDICES, MulAtIndices);
mdl!(QUANTIZED, Quantized);
mdl!(REDUCE, Reduce);
mdl!(REPEAT_PENALTY, RepeatPenalty);
mdl!(SORT, Sort);
mdl!(SUB_AT_INDICES, SubAtIndices);
mdl!(SUB_AT_INDICES_WITH_VALUES, SubAtIndicesWithValues);
mdl!(TERNARY, Ternary);
mdl!(UNARY, Unary);
