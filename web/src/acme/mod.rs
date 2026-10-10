//! Certificates from Let's Encrypt, issued and renewed by the gateway itself.
//!
//! Every public name in the site table is covered, one certificate per site
//! ([`names`]). Challenges are HTTP-01, answered on every hostname ahead of the
//! site table ([`challenges`]) — on the plain entrance, which is where the CA's
//! request arrives, through the tunnel or directly.
//!
//! The CA's rate limits are the danger: they are counted per name, per domain
//! and per account, they last up to a week, and a gateway that re-orders on
//! every restart reaches one within the hour. Three things stand between this
//! code and them:
//!
//! 1. **The [`ledger`]**, on disk, which refuses an order before the CA sees it
//!    — caps well under the CA's, and exponential back-off after a failure
//!    (that site) or a `rateLimited` answer (every site).
//! 2. **The [`preflight`]**, which proves a name reaches this gateway before it
//!    goes into an order, so a misrouted name costs nothing at the CA.
//! 3. **Renewal by the CA's own schedule** ([`manager`]), sent with `replaces`,
//!    which Let's Encrypt exempts from its limits.
//!
//! The [`store`] holds the account, the certificates and the ledger on disk.

pub mod challenges;
pub mod issue;
pub mod ledger;
pub mod manager;
pub mod names;
pub mod preflight;
pub mod store;

pub use challenges::Challenges;
pub use manager::Manager;

/// The CA every certificate comes from: Let's Encrypt.
pub const LETS_ENCRYPT: &str = "https://acme-v02.api.letsencrypt.org/directory";
