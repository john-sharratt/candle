//! The night's balance (§5.1): each person's share of the music, and the
//! requests', against what has played. A person under their share is owed
//! tracks, and the shortlist leans their way.

use std::collections::BTreeMap;

use crate::music::Profile;

/// Whose a played track counts as.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub enum Whose {
    Person(String),
    Request,
}

/// The shares the config names, `0..=1`, and what has played so far.
#[derive(Debug, Clone, Default)]
pub struct Budget {
    shares: BTreeMap<String, f32>,
    request_share: f32,
    played: BTreeMap<Whose, f32>,
    total: f32,
}

impl Budget {
    pub fn new(shares: BTreeMap<String, f32>, request_share: f32) -> Budget {
        Budget {
            shares,
            request_share,
            ..Budget::default()
        }
    }

    /// Whose `track` is: a request's, else split between those who like it,
    /// by how much.
    pub fn count(&mut self, track: &Profile, requested: bool) {
        self.total += 1.0;
        if requested {
            *self.played.entry(Whose::Request).or_default() += 1.0;
            return;
        }
        let likes: Vec<(String, f32)> = self
            .shares
            .keys()
            .map(|p| (p.clone(), track.liked_by(p)))
            .filter(|(_, w)| *w > 0.0)
            .collect();
        let sum: f32 = likes.iter().map(|(_, w)| w).sum();
        for (p, w) in likes {
            *self.played.entry(Whose::Person(p)).or_default() += w / sum;
        }
    }

    /// How far under their share `person` is, `0..=1`; nothing is owed
    /// before anything has played.
    pub fn owed(&self, person: &str) -> f32 {
        if self.total == 0.0 {
            return 0.0;
        }
        let share = self.shares.get(person).copied().unwrap_or(0.0);
        let had = self
            .played
            .get(&Whose::Person(person.to_string()))
            .copied()
            .unwrap_or(0.0)
            / self.total;
        (share - had).max(0.0)
    }

    /// How far under their share the requests are.
    pub fn requests_owed(&self) -> f32 {
        if self.total == 0.0 {
            return 0.0;
        }
        let had = self.played.get(&Whose::Request).copied().unwrap_or(0.0) / self.total;
        (self.request_share - had).max(0.0)
    }

    pub fn people(&self) -> impl Iterator<Item = &String> {
        self.shares.keys()
    }

    /// Each person's and the requests' fraction of what has played.
    pub fn fractions(&self) -> Vec<(String, f32)> {
        let total = self.total.max(1.0);
        self.played
            .iter()
            .map(|(w, n)| {
                let name = match w {
                    Whose::Person(p) => p.clone(),
                    Whose::Request => "requests".into(),
                };
                (name, n / total)
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::music::track::fixture::club;
    use crate::music::Liked;

    fn liked(id: &str, by: &[(&str, f32)]) -> Profile {
        let mut t = club(id, 128.0, "8A", 0.5);
        t.liked = by
            .iter()
            .map(|(p, w)| Liked {
                person: p.to_string(),
                weight: *w,
            })
            .collect();
        t
    }

    fn budget() -> Budget {
        Budget::new(
            [("john".to_string(), 0.4), ("yen".to_string(), 0.4)].into(),
            0.2,
        )
    }

    /// **A person under their share is owed**; a track both like counts to
    /// both by how much each likes it, and a request to the requests.
    #[test]
    fn the_owed_are_those_under_their_share() {
        let mut b = budget();
        assert_eq!(b.owed("yen"), 0.0, "nothing played");
        b.count(&liked("a", &[("john", 1.0)]), false);
        b.count(&liked("b", &[("john", 0.6), ("yen", 0.2)]), false);
        b.count(&liked("c", &[]), true);
        b.count(&liked("d", &[("john", 1.0)]), false);
        // john 1 + 0.75 + 1 = 2.75 of 4; yen 0.25 of 4; requests 1 of 4.
        assert_eq!(b.owed("john"), 0.0);
        assert!((b.owed("yen") - (0.4 - 0.0625)).abs() < 1e-6);
        assert_eq!(b.requests_owed(), 0.0);
        assert_eq!(b.owed("stranger"), 0.0);
    }
}
