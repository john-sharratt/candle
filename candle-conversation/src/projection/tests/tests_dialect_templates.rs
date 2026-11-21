//! Tests for dialect templates that a system prompt references by name.
//!
//! A template item resolves to its catalog entry, and one the dialect does not
//! have is filtered out at build time, so projection never sees an empty section.
//! Template sections seal like any other, and a template declared after its
//! collection still emits when that collection materialises.
