//! What lets a drawn picture leave: a [`Clearance`], and the [`Held`] picture
//! that opens only with one (`docs/guest_draw_orchestration.md` §2).
//!
//! A draw starts before its prompt is cleared, so what it makes is held until
//! a clearance for that prompt arrives. Storing and answering a picture take
//! only what [`Held::open`] gives back, so a picture whose prompt was refused,
//! or never cleared, cannot leave by construction.

use axum::http::HeaderMap;
use axum::response::Response;
use web::auth::{Role, Roles};

use super::token::{prompt_sha256, Authorization};
use crate::identity::require;

/// Why a draw may be released.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Clearance {
    /// An authorization already verified for the prompt it names.
    Authorized(Authorization),
    /// An Admin's own judgement, in place of the check: the restricted
    /// checkpoint's rule.
    AdminWaiver(AdminWaiver),
}

/// Proof that the caller is an Admin. Made only by [`AdminWaiver::of`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AdminWaiver(());

impl AdminWaiver {
    /// The waiver, when `headers` name an Admin; the refusal to answer with
    /// otherwise.
    pub fn of(headers: &HeaderMap, roles: &Roles) -> Result<Self, Box<Response>> {
        require(headers, roles, Role::Admin)?;
        Ok(Self(()))
    }
}

/// A drawn picture, held until its prompt is cleared.
#[derive(Debug)]
pub struct Held<T> {
    value: T,
    prompt_sha256: String,
}

/// A clearance that is not for the held picture's prompt.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NotForThis;

impl<T> Held<T> {
    /// `value`, drawn for `prompt`, held.
    pub fn new(value: T, prompt: &str) -> Self {
        Self {
            value,
            prompt_sha256: prompt_sha256(prompt),
        }
    }

    /// The picture, given a clearance for its prompt. An authorization must
    /// name this prompt; it was verified when the clearance was made.
    pub fn open(self, clearance: &Clearance) -> Result<T, NotForThis> {
        match clearance {
            Clearance::Authorized(a) if a.prompt_sha256 == self.prompt_sha256 => Ok(self.value),
            Clearance::Authorized(_) => Err(NotForThis),
            Clearance::AdminWaiver(_) => Ok(self.value),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::authorize::token::{Signer, IMAGE};

    /// **A held picture opens only for its own prompt's authorization**, and
    /// for an Admin's waiver.
    #[test]
    fn a_held_picture_opens_only_for_its_prompt() {
        let signer = Signer::new([1; 32]);
        let mine = Clearance::Authorized(signer.sign(IMAGE, "a fox", 0));
        let other = Clearance::Authorized(signer.sign(IMAGE, "a wolf", 0));
        assert_eq!(Held::new(5, "a fox").open(&mine), Ok(5));
        assert_eq!(Held::new(5, "a fox").open(&other), Err(NotForThis));
        assert_eq!(
            Held::new(5, "a fox").open(&Clearance::AdminWaiver(AdminWaiver(()))),
            Ok(5)
        );
    }

    /// **A waiver needs an Admin**: anyone else is refused.
    #[test]
    fn a_waiver_needs_an_admin() {
        assert!(AdminWaiver::of(&HeaderMap::new(), &Roles::default()).is_err());
    }
}
