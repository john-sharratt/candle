//! Linking a Discord account to an account here, and unlinking it
//! (`docs/npcd_discord.md` §3.2): the account's `discord` field and the
//! `discord/user/<id>` index written together.
//!
//! A Discord account links to one account. Linking it again moves it, and
//! the account it leaves has its field cleared; an account linking a second
//! Discord account lets go of the first.

use candle_conversation::persistence::SharedSubstrate;

use super::index;
use crate::accounts::{Accounts, DiscordUser};

/// Link `user` to account `account`.
pub fn link(
    accounts: &Accounts,
    shared: &SharedSubstrate,
    account: &str,
    user: &DiscordUser,
) -> anyhow::Result<()> {
    if accounts.identity(account).is_none() {
        anyhow::bail!("there is no account {account} to link");
    }
    if let Some(before) = index::account_of(shared, &user.id).filter(|a| a != account) {
        accounts.set_discord(&before, None)?;
    }
    if let Some(old) = accounts.discord(account).filter(|old| old.id != user.id) {
        index::remove(shared, &old.id)?;
    }
    accounts.set_discord(account, Some(user))?;
    index::put(shared, &user.id, account)
}

/// Unlink whatever Discord account is linked to `account`: what was linked,
/// if anything was.
pub fn unlink(
    accounts: &Accounts,
    shared: &SharedSubstrate,
    account: &str,
) -> anyhow::Result<Option<DiscordUser>> {
    let Some(old) = accounts.discord(account) else {
        return Ok(None);
    };
    if index::account_of(shared, &old.id).as_deref() == Some(account) {
        index::remove(shared, &old.id)?;
    }
    accounts.set_discord(account, None)?;
    Ok(Some(old))
}

#[cfg(test)]
mod tests {
    use std::path::Path;

    use tempfile::TempDir;
    use web::auth::session::Identity;

    use super::*;
    use crate::accounts::key_of;
    use crate::npcs::Npcs;

    fn ident(sub: &str) -> Identity {
        Identity {
            provider: "google".into(),
            sub: sub.into(),
            email: format!("{sub}@example.com"),
            name: sub.into(),
            picture: String::new(),
            exp: 9_999_999_999,
        }
    }

    fn user(id: &str) -> DiscordUser {
        DiscordUser {
            id: id.into(),
            username: format!("user{id}"),
            linked_ms: 5,
        }
    }

    /// Two accounts, `g1` and `g2`, and their keys.
    fn two(dir: &Path) -> (Accounts, SharedSubstrate, String, String) {
        let shared = Npcs::load(dir).unwrap().substrate();
        let mut accounts = Accounts::open(shared.clone(), &dir.join("accounts")).unwrap();
        accounts.upsert(&ident("g1"), 1).unwrap();
        accounts.upsert(&ident("g2"), 1).unwrap();
        (accounts, shared, key_of(&ident("g1")), key_of(&ident("g2")))
    }

    /// **A link writes both halves**: the account's field and the index.
    #[test]
    fn a_link_writes_the_field_and_the_index() {
        let dir = TempDir::new().unwrap();
        let (accounts, shared, g1, _) = two(dir.path());
        link(&accounts, &shared, &g1, &user("77")).unwrap();
        assert_eq!(accounts.discord(&g1), Some(user("77")));
        assert_eq!(index::account_of(&shared, "77"), Some(g1.clone()));
        assert!(link(
            &accounts,
            &shared,
            "0123456789abcdef0123456789abcdef",
            &user("78")
        )
        .is_err());
    }

    /// **Linking a Discord account again moves it**: the account it leaves
    /// is unlinked.
    #[test]
    fn relinking_moves_the_discord_account() {
        let dir = TempDir::new().unwrap();
        let (accounts, shared, g1, g2) = two(dir.path());
        link(&accounts, &shared, &g1, &user("77")).unwrap();
        link(&accounts, &shared, &g2, &user("77")).unwrap();
        assert_eq!(accounts.discord(&g1), None);
        assert_eq!(accounts.discord(&g2), Some(user("77")));
        assert_eq!(index::account_of(&shared, "77"), Some(g2));
    }

    /// An account that links a second Discord account lets go of the first.
    #[test]
    fn an_account_holds_one_discord_account() {
        let dir = TempDir::new().unwrap();
        let (accounts, shared, g1, _) = two(dir.path());
        link(&accounts, &shared, &g1, &user("77")).unwrap();
        link(&accounts, &shared, &g1, &user("88")).unwrap();
        assert_eq!(index::account_of(&shared, "77"), None);
        assert_eq!(index::account_of(&shared, "88"), Some(g1.clone()));
        assert_eq!(accounts.discord(&g1), Some(user("88")));
    }

    #[test]
    fn unlinking_clears_both_halves() {
        let dir = TempDir::new().unwrap();
        let (accounts, shared, g1, _) = two(dir.path());
        assert_eq!(unlink(&accounts, &shared, &g1).unwrap(), None);
        link(&accounts, &shared, &g1, &user("77")).unwrap();
        assert_eq!(unlink(&accounts, &shared, &g1).unwrap(), Some(user("77")));
        assert_eq!(accounts.discord(&g1), None);
        assert_eq!(index::account_of(&shared, "77"), None);
    }
}
