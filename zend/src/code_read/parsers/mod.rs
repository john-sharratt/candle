//! Per-language scope extractors.
//!
//! Each parser walks a tree-sitter syntax tree (or a line-based
//! fallback) and emits a `Vec<Scope>` in declaration order.  Carving
//! never panics — a parse failure downgrades to the fixed-window
//! fallback.

pub mod bash;
pub mod c;
pub mod cpp;
pub mod css;
pub mod fallback;
pub mod go;
pub mod html;
pub mod java;
pub mod javascript;
pub mod markdown;
pub mod php;
pub mod python;
pub mod ruby;
pub mod rust;
pub mod structured_config;
pub mod tree_sitter_util;
pub mod typescript;
