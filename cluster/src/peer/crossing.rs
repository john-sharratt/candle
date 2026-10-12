//! Two gateways that dial each other at once end with two links; both keep the
//! one dialled by the lower node, so they agree on which without a word
//! (§13.4). A link dialled the same way as the one held replaces it: the held
//! one is a dead socket the dialler has already given up on.

/// Which link a gateway keeps.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Keep {
    /// The new link: the held one is closed.
    New,
    /// The held link: the new one is closed.
    Held,
}

/// Given the dialler of the link held and of the new one.
pub fn keep(held_dialler: u64, new_dialler: u64) -> Keep {
    if new_dialler <= held_dialler {
        Keep::New
    } else {
        Keep::Held
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Both ends compute the same answer: the link the lower node dialled.
    #[test]
    fn the_lower_dialler_wins_on_both_ends() {
        // Gateways 3 and 7 dial each other. Gateway 3 holds its own link
        // (dialled by 3) and receives 7's; gateway 7 holds 7's and receives 3's.
        assert_eq!(keep(3, 7), Keep::Held);
        assert_eq!(keep(7, 3), Keep::New);
        // A redial in the same direction replaces the stale link.
        assert_eq!(keep(3, 3), Keep::New);
        assert_eq!(keep(7, 7), Keep::New);
    }
}
