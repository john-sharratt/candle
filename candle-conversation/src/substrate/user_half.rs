//! A turn's user half, borrowed from the substrate.

/// A sealed turn's user body, where its layout places it in the turn's grid.
#[derive(Clone, Copy, Debug)]
pub struct UserHalf<'a> {
    /// Where the body starts in the turn's ids.
    pub offset: usize,
    /// The body's ids.
    pub ids: &'a [u32],
    /// The body's text, as the turn was submitted.
    pub text: &'a str,
}
