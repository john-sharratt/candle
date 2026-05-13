use crate::kernel::{Kernel, Source};

pub const MULTINOMIAL: Kernel = Kernel {
    name: "multinomial",
    source: Source::Custom {
        src: include_str!("../multinomial.metal"),
    },
};
