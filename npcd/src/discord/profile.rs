//! What a bot is named and pictured as, from its character.
//!
//! A bot's username is the character's name, held to Discord's bounds — 2 to
//! 32 characters — and its avatar the character's portrait, as the data URI
//! `PATCH /users/@me` takes.

use base64::engine::general_purpose::STANDARD;
use base64::Engine;

/// The longest username Discord takes, in characters.
const MAX: usize = 32;

/// The shortest.
const MIN: usize = 2;

/// The username for a character called `name`: trimmed, and cut to
/// [`MAX`] characters on a character boundary. `None` for a name shorter
/// than [`MIN`], which Discord refuses — the bot keeps the name it was given
/// in the portal.
pub fn username(name: &str) -> Option<String> {
    let cut: String = name.trim().chars().take(MAX).collect();
    let cut = cut.trim_end();
    (cut.chars().count() >= MIN).then(|| cut.to_string())
}

/// An image as the avatar `PATCH /users/@me` takes: `data:<mime>;base64,…`.
pub fn avatar(bytes: &[u8], mime: &str) -> String {
    format!("data:{mime};base64,{}", STANDARD.encode(bytes))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_username_is_the_name_trimmed_and_cut_to_fit() {
        assert_eq!(username("  Zen  "), Some("Zen".to_string()));
        assert_eq!(username("Zo"), Some("Zo".to_string()));
        assert_eq!(username("Z"), None);
        assert_eq!(username("   "), None);
        assert_eq!(
            username("Drone-Kettle of the Southern Reaches and Beyond"),
            Some("Drone-Kettle of the Southern Rea".to_string())
        );
        // Cut on a character, not a byte, and never left ending in a space.
        assert_eq!(username(&"日".repeat(40)), Some("日".repeat(32)));
        assert_eq!(
            username("Thirty-one characters long here x"),
            Some("Thirty-one characters long here".to_string())
        );
    }

    #[test]
    fn an_avatar_is_a_base64_data_uri() {
        assert_eq!(
            avatar(b"\x89PNG", "image/png"),
            "data:image/png;base64,iVBORw=="
        );
    }
}
