//! This binary's build version (§3.5), stamped at compile time by `build.rs`,
//! and what it says of itself to `--build-info` (§3.9).

pub mod compute;
pub mod info;

use crate::frame::BuildId;

/// The build this binary was compiled from.
pub fn this_build() -> BuildId {
    BuildId {
        count: env!("TOKERA_BUILD_COUNT")
            .parse()
            .expect("build.rs stamps a number"),
        hash: env!("TOKERA_BUILD_HASH").to_string(),
        local: env!("TOKERA_BUILD_LOCAL").to_string(),
    }
}

#[cfg(test)]
mod tests;
