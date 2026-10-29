//! Rule 4's fence (`docs/cooperative_admission.md` §4.4): the cards with the
//! least latency load, taken until their power reaches the NPC floor, and the
//! share of each the floor needs.

/// The share of the cluster's NPC hosting power the floor keeps for the worlds,
/// in percent.
pub const NPC_FLOOR_PERCENT: u64 = 15;

/// One card the fence may take: its index, latency load and hosting power.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Candidate {
    pub index: usize,
    pub latency: u32,
    pub power: u32,
}

/// The fenced cards, each with the thousandths of its power the floor needs:
/// the least latency load first, then the weaker card — a fence on the card
/// latency least wants — until the floor is covered. Nothing when no card
/// hosts any power.
pub fn fence(candidates: &[Candidate]) -> Vec<(usize, u32)> {
    let total: u64 = candidates.iter().map(|c| u64::from(c.power)).sum();
    let mut need = (total * NPC_FLOOR_PERCENT).div_ceil(100);
    let mut order: Vec<Candidate> = candidates.iter().copied().filter(|c| c.power > 0).collect();
    order.sort_by_key(|c| (c.latency, c.power, c.index));
    let mut fenced = Vec::new();
    for c in order {
        if need == 0 {
            break;
        }
        let power = u64::from(c.power);
        let take = need.min(power);
        fenced.push((c.index, (take * 1000).div_ceil(power) as u32));
        need -= take;
    }
    fenced
}

#[cfg(test)]
mod tests {
    use super::*;

    fn card(index: usize, latency: u32, power: u32) -> Candidate {
        Candidate {
            index,
            latency,
            power,
        }
    }

    /// **The floor sits on the card latency least wants.** BlackWorld 300 and
    /// WhiteWorld 140: the floor is 15% of 440, 66, which WhiteWorld covers
    /// with 472 thousandths of its power — whichever card is the quieter, the
    /// weaker first on a tie.
    #[test]
    fn the_floor_sits_on_the_quietest_weakest_card() {
        assert_eq!(fence(&[card(0, 0, 300), card(1, 0, 140)]), vec![(1, 472)]);
        assert_eq!(fence(&[card(0, 0, 300), card(1, 3, 140)]), vec![(0, 220)]);
    }

    /// A floor larger than the quietest card spills onto the next; with no
    /// power hosted there is nothing to fence.
    #[test]
    fn a_floor_spills_onto_the_next_card() {
        assert_eq!(
            fence(&[card(0, 0, 10), card(1, 1, 100)]),
            vec![(0, 1000), (1, 70)]
        );
        assert_eq!(fence(&[card(0, 0, 0)]), vec![]);
        assert_eq!(fence(&[]), vec![]);
    }
}
