//! The link that adds a character's bot to a server.
//!
//! Discord's OAuth2 authorize page, for the `bot` scope, asking for exactly
//! the permissions a character's bot uses in its world's server and no more.

/// What a character's bot is granted in a server, bit by bit:
///
/// | Bit | Permission | For |
/// |---|---|---|
/// | `1 << 6` | Add Reactions | answering a message without a message |
/// | `1 << 10` | View Channels | seeing the channels it is a member of |
/// | `1 << 11` | Send Messages | speaking in them |
/// | `1 << 14` | Embed Links | a link it posts showing as a preview |
/// | `1 << 15` | Attach Files | a picture it posts |
/// | `1 << 16` | Read Message History | what was said while it was away |
/// | `1 << 29` | Manage Webhooks | the persona webhook characters without a bot speak through |
/// | `1 << 36` | Create Private Threads | a "DM" with a character that has no bot |
/// | `1 << 38` | Send Messages in Threads | speaking in those threads |
pub const PERMISSIONS: u64 = (1 << 6)
    | (1 << 10)
    | (1 << 11)
    | (1 << 14)
    | (1 << 15)
    | (1 << 16)
    | (1 << 29)
    | (1 << 36)
    | (1 << 38);

/// The invite for the bot of application `application_id`.
pub fn url(application_id: &str) -> String {
    format!(
        "https://discord.com/oauth2/authorize?client_id={application_id}&scope=bot&permissions={PERMISSIONS}"
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_permissions_are_exactly_these_bits() {
        assert_eq!(PERMISSIONS, 344_134_372_416);
    }

    #[test]
    fn an_invite_names_the_application_the_scope_and_the_permissions() {
        assert_eq!(
            url("1234567890"),
            "https://discord.com/oauth2/authorize?client_id=1234567890&scope=bot&permissions=344134372416"
        );
    }
}
