use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashMap, HashSet};

/// (rank, proposer)
#[derive(Debug, Eq)]
pub struct EngagementTuple(u32, String);

/// Allows us to store the engagement tuple in a binary heap.
impl Ord for EngagementTuple {
    fn cmp(&self, other: &Self) -> Ordering {
        self.0.cmp(&other.0)
    }
}

impl PartialOrd for EngagementTuple {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl PartialEq for EngagementTuple {
    fn eq(&self, other: &Self) -> bool {
        self.0 == other.0
    }
}

type ProposerId = String;
type AcceptorId = String;
#[derive(Debug)]
pub struct Acceptor {
    /// The name of this acceptor
    id: AcceptorId,

    /// When a proposal comes in, we need to quickly identify how
    /// we ranked the proposer.
    preference_map: HashMap<ProposerId, u32>,

    /// We accept proposals, provisionally,until
    /// capacity is exhausted. Afterward, if we receive a
    /// proposal from someone that we ranked higher,
    /// we have to reject one of the past proposals by
    /// removing the lowest ranked person from
    /// our provisional engagement list.
    capacity: i32,

    /// Note that since this is a max heap, we'll need to
    /// negate our priorities since we want extract max to
    /// give us the lowest ranked member.
    ///
    /// Note that because the current implementation only has
    /// acceptors with capacity=1, we could forego the heap and use
    /// a sinple string to keep track.
    provisional_engagements: BinaryHeap<EngagementTuple>,
}

impl Acceptor {
    /// To process a proposal, the acceptor first figures out how they
    /// ranked the proposer p1. If their rank is higher than that of some
    /// provisionally accepted proposer p2, p2 is removed and P1's request is accepted.
    /// The request is rejected if the acceptor ranked P1 lower that
    /// all the currently accepted proposers (and capacity is exhausted -- all proposals
    /// are accepted regardless of rank when capacity has yet to be exhausted)
    pub fn process_proposal(&mut self, proposer: String) -> ProposalResult {
        let proposers_rank = *self.preference_map.get(&proposer).unwrap();
        if self.capacity > 0 {
            self.capacity -= 1;
            self.provisional_engagements
                .push(EngagementTuple(proposers_rank, proposer));
            ProposalResult::Accept(None)
        } else {
            // If we are here, we have no capacity. We need to check if
            // the current proposer's rank is better than the lowest ranked
            // person we accepted -- remember, the lowest ranked will have a
            // higher number associated with them. Therefore, they'll be
            // at the root of the max heap.
            let lowest_ranked = self.provisional_engagements.peek().unwrap();
            if proposers_rank > lowest_ranked.0 {
                ProposalResult::Reject
            } else {
                let broken_heart = self.provisional_engagements.pop().unwrap();
                self.provisional_engagements
                    .push(EngagementTuple(proposers_rank, proposer));
                ProposalResult::Accept(Some(broken_heart.1))
            }
        }
    }
}

/// An acceptor can either accept or reject a proposal.
/// They can also break a previous engagement.
#[derive(Debug)]
pub enum ProposalResult {
    /// An acceptance can come with an associated breakup
    Accept(Option<String>),
    Reject,
}

#[derive(Debug)]
pub struct Proposer<'gale_shapely> {
    /// The name of this proposer.
    id: ProposerId,

    /// The proposers go through their preference list
    /// from top to bottom, proposing to each of their
    /// possible matches. This is to be used as a stack
    preference_list: &'gale_shapely Vec<String>,

    /// An index into preference list that points to the
    /// next person to propose to. It is incremented
    /// on each call to propose.
    current: usize,
}

impl<'gale_shapely> Proposer<'gale_shapely> {
    /// Propose to the next acceptor in the prefrence list
    pub fn propose(&mut self, acceptors: &mut HashMap<String, Acceptor>) -> ProposalResult {
        let proposal_result = acceptors
            .get_mut(&self.preference_list[self.current])
            .and_then(|acceptor| Some(acceptor.process_proposal(self.id.to_string())));
        proposal_result.unwrap()
    }

    /// This procedure is called by an the gale shapely procedure
    /// to inform a given Proposer that, their proposal -- which
    /// had been provisionally accepted, is being revoked because
    /// a suitor that is better ranked has come along.
    pub fn move_on(&mut self) {
        self.current += 1;
    }
}

type StableMatchings = HashMap<String, String>;

#[derive(Debug)]
pub struct DefferedAcceptance<'gale_shapely> {
    proposers: HashMap<ProposerId, Proposer<'gale_shapely>>,
    acceptors: HashMap<AcceptorId, Acceptor>,

    /// The set of the String IDs of proposers that are not yet engaged
    unmatched_proposers: HashSet<ProposerId>,
}

impl<'gale_shapely> DefferedAcceptance<'gale_shapely> {
    /// Create a new instance of the Gale-Shapely procedure
    /// proposers is the set of all string IDs of proposers in the market
    /// acceptors is the set of all string IDs of acceptors in the markets
    /// rankings is a mapping from the string ID of a market participant,
    /// either a proposer of acceptor, to a list of other market participants.
    /// The order of the list is assumed to be the ranking of the associated
    /// participant.
    ///
    /// Note that the way we consume inputs could be improved by taking advantage of
    /// Rust's type system. In particular, currently we have to manually check that
    /// a proposer can only rank an aceptor and vice versa. Using the type system,
    /// this could be checked by the compiler.
    pub fn new(
        proposer_ids: HashSet<ProposerId>,
        acceptor_ids: HashSet<AcceptorId>,
        rankings: &'gale_shapely HashMap<String, Vec<String>>,
    ) -> Self {
        let proposers = Self::create_proposers(&proposer_ids, rankings);
        let acceptors = Self::create_acceptors(&acceptor_ids, rankings);
        DefferedAcceptance {
            proposers,
            acceptors,
            unmatched_proposers: proposer_ids,
        }
    }

    /// pub struct Acceptor {
    ///     id: String,
    ///     preference_map: HashMap<String, i32>, // <ProposerId, rank>
    ///     capacity: i32,
    ///     provisional_engagements: BinaryHeap<EngagementTuple>,
    /// }
    fn create_acceptors(
        ids: &HashSet<String>,
        rankings: &HashMap<String, Vec<String>>,
    ) -> HashMap<String, Acceptor> {
        let mut acceptors = HashMap::new();
        for acceptor_id in ids {
            let mut preference_map = HashMap::<String, u32>::new();
            let current_acceptors_rankings = rankings.get(acceptor_id).unwrap();
            for (rank, proposer_id) in current_acceptors_rankings.iter().enumerate() {
                preference_map.insert(proposer_id.to_string(), rank as u32);
            }
            acceptors.insert(
                acceptor_id.to_string(),
                Acceptor {
                    id: acceptor_id.to_string(),
                    preference_map,
                    capacity: 1,
                    provisional_engagements: BinaryHeap::new(),
                },
            );
        }
        acceptors
    }

    fn create_proposers(
        ids: &HashSet<String>,
        rankings: &'gale_shapely HashMap<String, Vec<String>>,
    ) -> HashMap<String, Proposer<'gale_shapely>> {
        let mut proposers = HashMap::new();
        for acceptor_id in ids {
            let preference_list = rankings.get(acceptor_id).unwrap();
            proposers.insert(
                acceptor_id.to_string(),
                Proposer {
                    id: acceptor_id.to_string(),
                    current: 0,
                    preference_list,
                },
            );
        }
        proposers
    }

    /// While there is still an unmatched Proposer:
    ///     1. Every unassigned Proposes to the Acceptor at the top of
    ///        their preference stack.
    ///     2. If this proposal is accepted, then the proposer is removed from
    ///        the unmatched list. Note that, as a result of this, some
    ///        poor Proposer may have their heart broken, so they go back to the
    ///        unmatched list.
    /// After we exit the loop, all unrejected proposals are made final. The
    /// resulting match is a stable match.
    pub fn run(&mut self) -> StableMatchings {
        while self.unmatched_proposers.len() > 0 {
            let mut broken_hearts = HashSet::new();
            let mut matched = HashSet::new();
            for proposer in &self.unmatched_proposers {
                let current_proposer = self.proposers.get_mut(proposer).unwrap();
                match current_proposer.propose(&mut self.acceptors) {
                    ProposalResult::Accept(heart_breakee) => match heart_breakee {
                        Some(heart_breakee_id) => {
                            broken_hearts.insert(heart_breakee_id.clone());
                            self.proposers.get_mut(&heart_breakee_id).unwrap().move_on();
                            matched.insert(proposer.to_string());
                        }
                        None => {
                            matched.insert(proposer.to_string());
                        }
                    },
                    ProposalResult::Reject => current_proposer.move_on(),
                }
            }
            for matched_proposer in &matched {
                self.unmatched_proposers.remove(matched_proposer);
            }
            self.unmatched_proposers.extend(broken_hearts)
        }
        let mut stable_matchings = HashMap::new();
        for (acceptor_id, acceptor) in &mut self.acceptors {
            stable_matchings.insert(
                acceptor_id.to_string(),
                acceptor.provisional_engagements.pop().unwrap().1,
            );
        }
        stable_matchings
    }
}

#[cfg(test)]
mod test {
    use crate::algebra::randomize_in_place;
    use std::collections::{HashMap, HashSet};
    #[test]
    fn test_gale_shapely() {
        let proposers: HashSet<String> = (vec!["A", "B", "C", "D"])
            .iter_mut()
            .map(|s| s.to_string())
            .collect();
        let acceptors: HashSet<String> = (vec!["Alpha", "Beta", "Gamma", "Delta"])
            .iter_mut()
            .map(|s| s.to_string())
            .collect();
        let mut rankings = HashMap::new();
        for p in &proposers {
            rankings.insert(p.to_string(), random_permutation(&acceptors));
        }
        for a in &acceptors {
            rankings.insert(a.to_string(), random_permutation(&proposers));
        }
        let mut procedure = super::DefferedAcceptance::new(proposers, acceptors, &rankings);
        let stable_match = procedure.run();
        println!("{:?}", stable_match)
    }

    fn random_permutation(s: &HashSet<String>) -> Vec<String> {
        let mut v = s.clone().into_iter().collect();
        randomize_in_place(&mut v);
        v
    }
}
