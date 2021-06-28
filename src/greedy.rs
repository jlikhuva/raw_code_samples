//! Algorithms for discrete optimization iterate through a sequence of steps.
//! At each iteration/time-step, they are faced with a set of k choices.
//! The goal is to select a single choice (or action) that will lead to an
//! optimal solution. Choosing an action often subdivides the problem space
//! into smaller problem spaces (Or, if you like RL speak —
//! after selecting an action, the algorithms transition to a new reduced state space).
//! Greedy algorithms differ from  DP algorithms discussed in the preceding section in
//! the way they select the optimal action out of the $$k$$ available choices.
//! Whereas DP methods fully examine each of the $$k$$ options before selecting one
//! (that is, they fully expand and evaluate the resultant sub-problems),
//! greedy algorithms simply pick the action that looks optimal at the moment.
//! At each step, they make an irrevocable decision and move on to the next step
//! hoping that the greedy choice will lead to an optimal solution. This makes for
//! simpler algorithms. The catch is that greedy methods don’t always yield the
//! optimal solution. Greedy algorithms have a top-down design — we make a choice
//! and then solve the resultant sub-problems. Contrast this with DP methods which
//! always have a bottom up structure (even when implemented recursively with memoization)
//! Another thing to note is that because greedy algorithms need to access locally optimal
//! elements, they make extensive use of sorting and heaps.
use crate::sorting;
use bitvec::prelude::{BitVec, LocalBits};
use std::{
    cmp::{Ord, Ordering, Reverse},
    collections::{BinaryHeap, HashMap},
};

#[derive(Debug, Eq)]
pub struct Activity {
    id: &'static str,
    start_time: u32,
    end_time: u32,
}

impl Ord for Activity {
    fn cmp(&self, other: &Self) -> Ordering {
        self.end_time.cmp(&other.end_time)
    }
}

impl PartialOrd for Activity {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl PartialEq for Activity {
    fn eq(&self, other: &Self) -> bool {
        self.end_time == other.end_time
    }
}

pub fn activity_selection(mut activities: &mut Vec<Activity>) -> Vec<&'static str> {
    let mut largest_disjoint_set = Vec::new();
    let end_idx = activities.len() - 1;

    // Sort in increasing order of end_time
    sorting::quick_sort(&mut activities, (0, end_idx));

    // First activity will always be selected.
    largest_disjoint_set.push(activities[0].id);
    let mut cur_latest_finish_time = activities[0].end_time;
    for k in 1..activities.len() {
        // Only add an activity that does not overlap with any of the
        // ones already chosen.
        if activities[k].start_time >= cur_latest_finish_time {
            cur_latest_finish_time = activities[k].end_time;
            largest_disjoint_set.push(activities[k].id)
        }
    }
    largest_disjoint_set
}

/// An entry of the Huffman Tree over some specified AlphabetType.
#[derive(Debug, Eq)]
pub enum HuffmanTreeNode<AlphabetType: Ord> {
    /// Each leaf holds a single character in our alphabet
    /// along with that character's frequency.
    Leaf {
        character: AlphabetType,
        freq: Reverse<u64>,
    },

    /// Internal nodes are formed via the merge of other internal nodes
    /// or via the fusion of 2 Leaf nodes. Each Internal node's frequency
    /// is simply the sum of the frequency values stored in the left
    /// subtree and the right subtree. Note that since a huffman tree is
    /// built from the ground up, internal nodes will always have left and right
    /// pointers.
    Internal {
        left: usize,
        right: usize,
        freq: Reverse<u64>,
    },
}

impl<AlphabetType: Ord> Ord for HuffmanTreeNode<AlphabetType> {
    fn cmp(&self, other: &Self) -> Ordering {
        use HuffmanTreeNode::{Internal, Leaf};
        match (self, other) {
            (Leaf { freq: f, .. }, Leaf { freq: other_freq, .. })
            | (Leaf { freq: f, .. }, Internal { freq: other_freq, .. })
            | (Internal { freq: f, .. }, Leaf { freq: other_freq, .. })
            | (Internal { freq: f, .. }, Internal { freq: other_freq, .. }) => f.cmp(other_freq),
        }
    }
}

impl<AlphabetType: Ord> PartialOrd for HuffmanTreeNode<AlphabetType> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl<AlphabetType: Ord> PartialEq for HuffmanTreeNode<AlphabetType> {
    fn eq(&self, other: &Self) -> bool {
        use HuffmanTreeNode::{Internal, Leaf};
        match (self, other) {
            (Leaf { freq: f, .. }, Leaf { freq: other_freq, .. })
            | (Leaf { freq: f, .. }, Internal { freq: other_freq, .. })
            | (Internal { freq: f, .. }, Leaf { freq: other_freq, .. })
            | (Internal { freq: f, .. }, Internal { freq: other_freq, .. }) => f == other_freq,
        }
    }
}
#[derive(Debug)]
pub struct HuffmanTree<AlphabetType: Ord>(HuffmanTreeNode<AlphabetType>);

pub type Counter<AlphabetType> = HashMap<AlphabetType, u64>;
#[derive(Debug)]
pub struct HuffmanCode(BitVec<LocalBits, usize>);

impl<AlphabetType: Ord + Copy> HuffmanTree<AlphabetType> {
    pub fn build_huffman_tree(alphabet: &Counter<AlphabetType>) -> Self {
        let mut queue = BinaryHeap::new();
        // let mut chars = vec![];
        // TODO
        for (char, count) in alphabet.iter() {
            let leaf_node = HuffmanTreeNode::Leaf {
                character: *char,
                freq: Reverse(*count),
            };
            queue.push(leaf_node);
        }
        for _ in 1..alphabet.len() - 1 {
            // TODO: keep the alphabet keys in an vector
        }
        unimplemented! {}
    }

    pub fn extract_huffman_code() -> HashMap<AlphabetType, HuffmanCode> {
        unimplemented! {}
    }
}

pub fn fractional_knapsack() {
    unimplemented!()
}

pub fn coin_changing() {
    unimplemented!()
}

pub fn huffman_tree() {
    unimplemented!()
}

#[cfg(test)]
mod test {
    #[test]
    fn test_activity_selection() {
        unimplemented!();
    }
}
