//! What a character is told when its bot joins or leaves a server
//! (`docs/npcd_discord.md` §3.4), and the name its channels' threads go by.

/// What the thread that mirrors channel `channel` of server `server` is
/// called: `zenworld #general`.
pub fn thread_name(server: &str, channel: &str) -> String {
    format!("{server} #{channel}")
}

/// The channel a thread called `thread` mirrors in server `server`, read
/// back from its name.
pub fn channel_of<'a>(thread: &'a str, server: &str) -> Option<&'a str> {
    thread.strip_prefix(server)?.strip_prefix(" #")
}

/// What the character is told of being added to server `server`, whose
/// text channels are `channels`: the first by its thread's whole name, the
/// rest by `#` alone.
pub fn joined(server: &str, channels: &[String]) -> String {
    let told = format!("Someone added you to the Discord server {server}.");
    let Some((first, rest)) = channels.split_first() else {
        return told;
    };
    let mut names = vec![thread_name(server, first)];
    names.extend(rest.iter().map(|c| format!("#{c}")));
    format!("{told} You can talk in its channels: {}", names.join(", "))
}

/// What the character is told of being removed from server `server`.
pub fn removed(server: &str) -> String {
    format!("You were removed from the Discord server {server}.")
}

#[cfg(test)]
mod tests {
    use super::*;

    /// **The texts are exactly the design's.**
    #[test]
    fn a_join_and_a_removal_are_told_in_these_words() {
        assert_eq!(
            joined(
                "zenworld",
                &["general".into(), "lore".into(), "news".into()]
            ),
            "Someone added you to the Discord server zenworld. You can talk in its channels: \
             zenworld #general, #lore, #news"
        );
        assert_eq!(
            joined("zenworld", &[]),
            "Someone added you to the Discord server zenworld."
        );
        assert_eq!(
            removed("zenworld"),
            "You were removed from the Discord server zenworld."
        );
    }

    #[test]
    fn a_thread_is_named_by_its_server_and_channel_and_read_back() {
        assert_eq!(thread_name("zenworld", "general"), "zenworld #general");
        assert_eq!(channel_of("zenworld #general", "zenworld"), Some("general"));
        assert_eq!(channel_of("zenworld #general", "zen"), None);
        assert_eq!(channel_of("other #general", "zenworld"), None);
    }
}
