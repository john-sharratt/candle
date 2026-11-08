//! The crates (§4.1, step 3): the TIDAL playlists ZenJ owns, which rekordbox
//! shows and stores offline. A track is filed in each crate it belongs to;
//! its profile names the first.

use crate::music::Profile;

pub const WARM_UP: &str = "ZenJ Warm-up";
pub const BUILD: &str = "ZenJ Build";
pub const PEAK: &str = "ZenJ Peak";
pub const CLASSICS: &str = "ZenJ Classics";
pub const MIDNIGHT: &str = "ZenJ Midnight";
pub const BOTH: &str = "ZenJ Liked by both";
/// The next tracks, kept short, which the conductor loads from (§6.2).
pub const NEXT: &str = "NYE Next";
/// What Automix plays when ZenJ steps down (§7.4).
pub const LIVE: &str = "NYE Live";

/// Every crate, in the order rekordbox lists them.
pub const ALL: [&str; 6] = [WARM_UP, BUILD, PEAK, CLASSICS, MIDNIGHT, BOTH];

/// A liking past which a person counts as liking a track for *Liked by
/// both*.
const LIKES: f32 = 0.3;

/// The crates `t` belongs in, given `people`.
pub fn crates_of(t: &Profile, people: &[String]) -> Vec<&'static str> {
    let mut out = vec![if t.energy < 0.45 {
        WARM_UP
    } else if t.energy < 0.65 {
        BUILD
    } else {
        PEAK
    }];
    if t.year.is_some_and(|y| y < 2000) || t.tags.iter().any(|x| x == "classic") {
        out.push(CLASSICS);
    }
    if t.energy >= 0.7 && t.tags.iter().any(|x| x == "singalong" || x == "anthem") {
        out.push(MIDNIGHT);
    }
    if !people.is_empty() && people.iter().all(|p| t.liked_by(p) >= LIKES) {
        out.push(BOTH);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::music::track::fixture::club;
    use crate::music::Liked;

    #[test]
    fn a_track_is_filed_by_energy_and_in_every_crate_it_fits() {
        let people = vec!["john".to_string(), "yen".to_string()];
        assert_eq!(crates_of(&club("a", 120.0, "8A", 0.3), &people), [WARM_UP]);
        assert_eq!(crates_of(&club("a", 120.0, "8A", 0.5), &people), [BUILD]);
        let mut t = club("a", 128.0, "8A", 0.8);
        t.year = Some(1994);
        t.tags = vec!["anthem".into()];
        t.liked = vec![
            Liked {
                person: "john".into(),
                weight: 0.9,
            },
            Liked {
                person: "yen".into(),
                weight: 0.4,
            },
        ];
        assert_eq!(crates_of(&t, &people), [PEAK, CLASSICS, MIDNIGHT, BOTH]);
        t.liked.pop();
        assert_eq!(crates_of(&t, &people), [PEAK, CLASSICS, MIDNIGHT]);
    }
}
