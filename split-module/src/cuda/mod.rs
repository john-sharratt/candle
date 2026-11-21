//! Splitting a CUDA or C++ kernel source into headers it includes.
//!
//! A kernel file is one translation unit, and its definitions reach each other
//! by being in it, in order. So a split leaves the file where it is and moves
//! runs of its units into headers under `<stem>/`, each `#include`d where its
//! run stood: the text the compiler sees is the source as it was, the parts'
//! own header comments aside. The `#include` lines stay where they are, so no
//! include path changes what it names; the build already hashes a kernel with
//! every header it reaches, so a change to a part rebuilds what includes it.
//!
//! `scan` cuts the source into units, `layout` opens the big blocks and writes
//! the map, and `split` builds and checks the parts.

pub mod layout;
pub mod scan;
pub mod split;

/// Whether `path` is a kernel source this mode splits.
pub fn is_kernel(path: &std::path::Path) -> bool {
    matches!(
        path.extension().and_then(|e| e.to_str()),
        Some("cu") | Some("cuh")
    )
}
