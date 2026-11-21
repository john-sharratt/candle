//! Tests for section trees nested inside one another.
//!
//! A nested tree's selector sits below static content, so it multiplies only the
//! dimensions above it, and switching it leaves the content above sealed as it
//! was. These cases are kept apart from the basic section tree tests.
