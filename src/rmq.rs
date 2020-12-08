//! The Range Min Query (RMQ) is the task of finding the minimal value
//! in some underlying array in a give range (i, j). It is usually
//! assummed that the underlying array is static and that we plan to make
//! a large number of RMQ queries over time. Therefore, the naive solution
//! of doing a linear search over the range in question every time a query
//! comes in may be suboptimal. Since the array is static, we can get better
//! runtimes by first preprocessing it. This module preprocesses
//! the array using the Fisher-Heun Structure which uses the method of four russians.
//! Specifically, it preprocesses the array by first partitioning the input
//! array into blocks of some size `b=0.5(log_base_four n)`. it then computes
//! the minimal value in each of these partitions, storing the result in a
//! a new auxilliarry summary array S. It then builds a sparse table of RMQ
//! answer for S. Finally, on each of the partitions, it precomputes
//! RMQ values for each possible range, using Cartesian tree numbers to avoid
//! recomputing values for blocks that share the same answers.
use lazy_static::lazy_static;
use std::collections::{HashMap, LinkedList as Stack};

/// Each our our blocks will be of size 64
static MAX_BLOCK_SIZE: usize = 4;

/// Enum to aid in bit-level binary search
enum Bits {
    Top32(u64),
    Bot32(u64),
    Top16(u64),
    Bot16(u64),
    Top8(u64),
    Bot8(u64),
    Top4(u64),
    Bot4(u64),
    Top2(u64),
    Bot2(u64),
    Top1(u64),
    Bot1(u64),
}

lazy_static! {
    /// Bit masks that allow us to do bit level binary
    /// search on an unsigned 64 bit number in order to
    /// figure out the index of the most significant bit
    static ref MASKS: HashMap<&'static str, u64> = {
        let mut map = HashMap::new();
        map.insert(
            "top32",
            0b11111111_11111111_11111111_11111111_00000000_00000000_00000000_00000000,
        );
        map.insert(
            "bot32",
            0b00000000_00000000_00000000_00000000_11111111_11111111_11111111_11111111,
        );
        map.insert("top16", 0b11111111_11111111_00000000_00000000);
        map.insert("bot16", 0b00000000_00000000_11111111_11111111);
        map.insert("top8", 0b11111111_00000000);
        map.insert("bot8", 0b00000000_11111111);
        map.insert("top4", 0b1111_0000);
        map.insert("bot4", 0b0000_1111);
        map.insert("bot2", 0b00_11);
        map.insert("top2", 0b11_00);
        map.insert("top1", 0b10);
        map.insert("bot1", 0b01);
        map
    };
}

/// A range in our underlying array. This is assumed to be
/// zero indexed. For example, Range(2, 5) should represent
/// the range starting at the 3rd element up to the 6th element
/// inclusive.
#[derive(Hash, Debug, Eq, PartialEq)]
pub struct Range {
    /// The start index i
    start: usize,

    /// The ending index j.
    end: usize,
}
#[derive(Debug)]
struct CartesianTreeNode {
    /// This is the location of the value being
    /// represented by this node in the underlying
    /// static array. Note that for simplicity,
    /// this should not be the index within a
    /// block. Rather, it should be the index within
    /// the global static array.
    index_of_value: usize,

    /// The locations of the children of this node.
    index_of_left_child: Option<usize>,
    index_of_right_child: Option<usize>,
}

impl CartesianTreeNode {
    pub fn new(index_of_value: usize) -> Self {
        CartesianTreeNode {
            index_of_value,
            index_of_left_child: None,
            index_of_right_child: None,
        }
    }
}

/// We use this to store an retrieve entries in our
/// our sparse table. Index.0 is the starting index
/// while Index.1 is the size of the precomputed
/// length. This length is always a power of 2.
type SparseTableIndex = (usize, usize);

/// A table mapping powes of 2 k = (0, 1, ...) such that
/// 2^k fits withing an underlying array to precomputed RMQ
/// answers in intervals of length k.
type SparseTable = HashMap<SparseTableIndex, HashMap<Range, InBlockOffset>>;

/// In order to effectively reuse RMQ structures between
/// different blocks without having to treat each block
/// as an independent array with start index at 0, in the
/// BlockLeveRMQ structure, we store the offest instead
/// of the actual index. for instance, suppose RMQ(i, j)
/// where i and j are indexes in the global static_array
/// returns rmq  = i + k, we should store k instead of
/// rmq. To make this distinction clear, we should wrap k
/// so that it is different from other indexes
#[derive(Debug)]
pub struct InBlockOffset {
    /// The index at which this offset's block begins in the global static array
    block_start: usize,

    /// The index of this item in the global array
    base_index: usize,

    /// The offset in the block. Two different items
    /// can have different base_index values but the same inblock_index
    /// value.
    inblock_index: usize,
}

impl InBlockOffset {
    pub fn new(block_start: usize, base_index: usize) -> Self {
        InBlockOffset {
            block_start,
            base_index,
            inblock_index: base_index - block_start,
        }
    }
}

/// A table mapping the cartesian number of a block to
/// the precomputed RMQ answers in that range.
type BlockLevelRMQ = HashMap<u128, HashMap<Range, InBlockOffset>>;

impl Range {
    pub fn new(i: usize, j: usize) -> Self {
        Range { start: i, end: j }
    }
}
#[derive(Debug)]
pub struct FischerHeunRMQ<T: PartialOrd + Copy> {
    /// This is the array that we are building the
    /// RMQ structures for. It's assumed that this
    /// array is static. If it changes, all the RMQ
    /// structures have to be rebuilt.
    static_array: Vec<T>,

    /// The list of minimal values in each block. We build
    /// the sparse table on top of this.
    minimums: Vec<T>,

    /// A mapping from a cartesian number of a given array of size b
    /// to a matrix of precomputed RMQ values for all subarrays
    block_level_rmq: BlockLevelRMQ,

    /// A sparse table for the RMQ queries on the array
    /// of block level minimum values.
    summary_rmq_sparse_table: SparseTable,

    /// A cache of each block's cartesian tree number so that we
    /// dont have to compute them at query time again
    cartesian_number_cache: HashMap<usize, u128>,
}

impl<T: Ord + Copy> FischerHeunRMQ<T> {
    /// Creates a new FischerHeunRMQ structure and initializes
    /// all of its caches. The expectation is that the array
    /// on which we are building the structure is static. In the process
    /// of creation, it initializes all the data structures needed
    /// to answer ad hoc range queries
    pub fn new(static_array: Vec<T>) -> Self {
        let minimums = Self::generate_macro_array(&static_array);
        let (block_level_rmq, cartesian_number_cache) =
            Self::compute_cached_dense_tables(&static_array);
        let summary_rmq_sparse_table = Self::compute_sparse_table(&minimums);
        FischerHeunRMQ {
            static_array,
            minimums,
            block_level_rmq,
            summary_rmq_sparse_table,
            cartesian_number_cache,
        }
    }

    /// Returns the smallest value [range.start, range.end] in constant time
    pub fn query(&self, range: Range) -> T {
        let start_block = range.start / MAX_BLOCK_SIZE;
        let end_block = range.end / MAX_BLOCK_SIZE;
        let min_at_ends = self.get_min_value_at_ends(start_block, end_block);
        let min_in_intermediate = self.get_intermediate_min(range);
        std::cmp::min(min_at_ends, min_in_intermediate)
    }

    /// Finds and returns the smallest value in the blocks that the range passed
    /// into query starts and ends. To do that, it uses the per-block dense
    /// tables that have been precomputed. With this scheme, we are able to answer
    /// this portion of the query in O(1)
    fn get_min_value_at_ends(&self, start_block: usize, end_block: usize) -> T {
        let start_ct_number = *self.cartesian_number_cache.get(&start_block).unwrap();
        let end_ct_number = *self.cartesian_number_cache.get(&end_block).unwrap();
        let start_min = self
            .block_level_rmq
            .get(&start_ct_number)
            .and_then(|m| m.get(&Range::new(start_block, start_block + MAX_BLOCK_SIZE)))
            .unwrap();
        let end_min = self
            .block_level_rmq
            .get(&end_ct_number)
            .and_then(|m| m.get(&Range::new(end_block, end_block + MAX_BLOCK_SIZE)))
            .unwrap();
        let (l, r) = (
            self.static_array[start_block + start_min.inblock_index],
            self.static_array[end_block + end_min.inblock_index],
        );
        std::cmp::min(l, r)
    }

    /// Finds and returns the smallest value in the blocks between
    /// the start_block and the end_block. To do this, we use
    /// the sparse table of RMQ answers that we precomputed
    /// using the table of minimums of each block. Because any
    /// range can be formed as the union of 2 ranges whose size is a
    /// power of 2 (and we precomputed the answers to all ranges
    /// whose size is s power of 2), we are able to answer this part of the
    /// query in O(1) as well
    fn get_intermediate_min(&self, range: Range) -> T {
        let k = self.get_msb(range.end - range.start + 1);
        let summary_answers = self
            .summary_rmq_sparse_table
            .get(&(range.start, 2 ^ k))
            .unwrap();
        let left = Range::new(range.start, range.start + (1 << k) - 1);
        let right = Range::new(range.end - (2 << k) + 1, range.end);
        let left_min = summary_answers.get(&left).unwrap().base_index;
        let right_min = summary_answers.get(&right).unwrap().base_index;
        std::cmp::min(self.static_array[left_min], self.static_array[right_min])
    }

    /// Does bit level binary search to find the most significant bit
    /// of `n`. This takes O(lg w) where w is the number of bits in usize
    /// Theroretically speaking, this is not O(1) since w, the word size
    /// is platform dependent and in general depends on n, the largest
    /// input we'd like to  represent. It is O(lg lg n). Practically, however
    /// this is O(1). Even for the largest values of n, O(lg lg n) is a small
    /// constant factor. On 64 bit platforms this is O(lg 64) = 6 = O(1)
    fn get_msb(&self, n: usize) -> usize {
        let top32 = MASKS
            .get("top32")
            .and_then(|mask| Some((n as u64 & mask) >> 32))
            .unwrap();
        if top32 > 0 {
            Self::msb_helper(top32, Bits::Top32(32)) as usize
        } else {
            let bot32 = n as u64 & MASKS.get("bot32").unwrap();
            Self::msb_helper(bot32, Bits::Bot32(0)) as usize
        }
    }

    // fn get_bits_in_portion(n: usize, upper: &'static str, lower: &'static str,  k: u64, count: u64) -> u64 {
    //     let top = MASKS
    //         .get(upper)
    //         .and_then(|mask| Some((n as u64 & mask) >> k))
    //         .unwrap();
    //     if top > 0 {
    //         Self::msb_helper(top, Bits::Top16(count + k))
    //     } else {
    //         let bot = n as u64 & MASKS.get(lower).unwrap();
    //         Self::msb_helper(bot, Bits::Bot16(count))
    //     }
    // }

    fn msb_helper(n: u64, portion: Bits) -> u64 {
        match portion {
            Bits::Top32(l) | Bits::Bot32(l) => {
                let top16 = MASKS
                    .get("top16")
                    .and_then(|mask| Some((n as u64 & mask) >> 16))
                    .unwrap();
                if top16 > 0 {
                    Self::msb_helper(top16, Bits::Top16(l + 16))
                } else {
                    let bot16 = n as u64 & MASKS.get("bot16").unwrap();
                    Self::msb_helper(bot16, Bits::Bot16(l))
                }
            }
            Bits::Top16(l) | Bits::Bot16(l) => {
                let top8 = MASKS
                    .get("top8")
                    .and_then(|mask| Some((n as u64 & mask) >> 8))
                    .unwrap();
                if top8 > 0 {
                    Self::msb_helper(top8, Bits::Top8(l + 8))
                } else {
                    let bot8 = n as u64 & MASKS.get("bot8").unwrap();
                    Self::msb_helper(bot8, Bits::Bot8(l))
                }
            }
            Bits::Top8(l) | Bits::Bot8(l) => {
                let top4 = MASKS
                    .get("top4")
                    .and_then(|mask| Some((n as u64 & mask) >> 4))
                    .unwrap();
                if top4 > 0 {
                    Self::msb_helper(top4, Bits::Top4(l + 4))
                } else {
                    let bot4 = n as u64 & MASKS.get("bot4").unwrap();
                    Self::msb_helper(bot4, Bits::Bot4(l))
                }
            }
            Bits::Top4(l) | Bits::Bot4(l) => {
                let top2 = MASKS
                    .get("top2")
                    .and_then(|mask| Some((n as u64 & mask) >> 4))
                    .unwrap();
                if top2 > 0 {
                    Self::msb_helper(top2, Bits::Top2(l + 4))
                } else {
                    let bot2 = n as u64 & MASKS.get("bot2").unwrap();
                    Self::msb_helper(bot2, Bits::Bot2(l))
                }
            }
            Bits::Top2(l) | Bits::Bot2(l) => {
                let top1 = MASKS
                    .get("top1")
                    .and_then(|mask| Some((n as u64 & mask) >> 2))
                    .unwrap();
                if top1 > 0 {
                    Self::msb_helper(top1, Bits::Top1(l + 2))
                } else {
                    let bot1 = n as u64 & MASKS.get("bot1").unwrap();
                    Self::msb_helper(bot1, Bits::Bot1(l))
                }
            }
            Bits::Top1(l) | Bits::Bot1(l) => {
                if n == 1 {
                    l + 1
                } else {
                    l
                }
            }
        }
    }

    /// Generates the array for min values in each block. The generated
    /// array forms the basis of our 'macro' problem portion of the
    /// method of four russians.
    pub fn generate_macro_array(static_array: &Vec<T>) -> Vec<T> {
        let mut minimums = Vec::new();
        for i in (0..static_array.len()).step_by(MAX_BLOCK_SIZE - 1) {
            let cur_range = Range::new(i, i + MAX_BLOCK_SIZE - 1);
            let cur_min_idx = Self::min_index_in_range(static_array, cur_range);
            minimums.push(static_array[cur_min_idx])
        }
        minimums
    }

    /// For each index i, compute RMQ for ranges starting at i of
    /// size 1, 2, 4, 8, 16, …, 2k as long as they fit in the array.
    /// For each array element, we compute lg n ranges. Therefore,
    /// the total cost of the procedure is O(n lg n)
    pub fn compute_sparse_table(minimums: &Vec<T>) -> SparseTable {
        let mut table = HashMap::new();
        let array = minimums;
        for i in 0..array.len() {
            let mut k = 0;
            while 2 ^ k < array.len() {
                let end_idx = i + (2 ^ k - 1);
                let range_len = (end_idx - i) + 1;
                let query_range = Range::new(i, end_idx);
                let rmq_answers_in_range = Self::compute_rmq_all_ranges(&query_range, array);
                table.insert((i, range_len), rmq_answers_in_range);
                k += 1;
            }
        }
        table
    }

    /// This procedure uses the dp algorithm `compute_rmq_all_ranges` to preccompute
    /// the index of the minimal element in all n*n range of a given block.
    /// However, if any 2 blocks have the same structure -- as indicated by
    /// their cartesian tree number, the procedure uses only a single
    /// precomputed dictionary for both.
    pub fn compute_cached_dense_tables(
        static_array: &Vec<T>,
    ) -> (BlockLevelRMQ, HashMap<usize, u128>) {
        let mut block_level_rmq = HashMap::new();
        let mut cartesian_cache = HashMap::new();
        for i in (0..static_array.len()).step_by(MAX_BLOCK_SIZE - 1) {
            let cur_range = Range::new(i, i + MAX_BLOCK_SIZE - 1);
            let cur_number = Self::cartesian_tree_number(&cur_range, static_array);
            cartesian_cache.insert(i, cur_number);
            if block_level_rmq.contains_key(&cur_number) {
                continue;
            }
            let rmq = Self::compute_rmq_all_ranges(&cur_range, static_array);
            block_level_rmq.insert(cur_number, rmq);
        }
        (block_level_rmq, cartesian_cache)
    }

    /// Computes the cartesian tree number for the block in the given
    /// range. This number gives us the shape of the cartesian tree
    /// formed from the elements in that range. If two blocks
    /// have isomorphic cartesian trees then they have the same
    /// cartesian tree number. This means that minimal values
    /// appear at the same index for all corresponding intervals
    /// in the two blocks -- therefore, we can use a single
    /// precomputed table to answer RMQ queries for both blocks.
    pub fn cartesian_tree_number(block_range: &Range, static_array: &Vec<T>) -> u128 {
        let mut cartesian_tree_number = 0;
        let mut stack = Stack::<usize>::new();

        // Technically, we do not even need to build the certesian tree, all
        // we need to know is the series of stack push and pop operations
        // needed to create the tree.
        let mut cartesian_tree = Vec::with_capacity(MAX_BLOCK_SIZE);

        // We initialize each item in the block as its own
        // cartesian tree with no children
        for i in block_range.start..=block_range.end {
            cartesian_tree.push(CartesianTreeNode::new(i));
        }

        // To create the cartesian tree, we pop the stack until either
        // it's empty or the element atop the stack has a smaller value
        // than the element we are currently trying to add to the stack.
        // Once we break out of the `pop` loop, we make the item we popped
        // a left child of the new item we are adding. Additionally, we make
        // this new item a right/left child of the item atop the stack
        for i in block_range.start..=block_range.end {
            let mut last_popped = None;
            let mut offset_in_number = 0;
            loop {
                match stack.front() {
                    None => break,
                    Some(&top_node_index) => {
                        if static_array[top_node_index] < static_array[i] {
                            cartesian_tree[top_node_index].index_of_right_child = Some(i);
                            break;
                        }
                        last_popped = stack.pop_front();
                        offset_in_number += 1;
                    }
                }
            }
            if let Some(last_popped_idx) = last_popped {
                cartesian_tree[i].index_of_left_child = Some(last_popped_idx);
            }
            stack.push_front(i);
            offset_in_number += 1;

            // The cartesian tree number is simply a profile of the push/pop
            // sequences needed to create a cartesian tree for the current block
            cartesian_tree_number |= 1 << offset_in_number;
        }
        cartesian_tree_number
    }

    /// Return the index of the minimal element in
    /// the given range.
    fn min_index_in_range(static_array: &Vec<T>, range: Range) -> usize {
        let end = range.end;
        let len = static_array.len();
        let cur_block = if end < len {
            &static_array[range.start..=end]
        } else {
            &static_array[range.start..]
        };
        cur_block
            .iter()
            .enumerate()
            .fold(0, |cur_min_idx, (cur_idx, &x)| {
                match x.cmp(&cur_block[cur_min_idx]) {
                    std::cmp::Ordering::Less => cur_idx,
                    _ => cur_min_idx,
                }
            })
    }

    /// This procedure is used to compute the answers to all
    /// n^2 RMQ queries that can ever be asked in the range passed in.
    /// This procedure is used to pre-compute the RMQ values on
    /// individual blocks. It returns a mapping from a Range
    /// to the index of the minimal value in that range
    pub fn compute_rmq_all_ranges(block: &Range, array: &Vec<T>) -> HashMap<Range, InBlockOffset> {
        let mut all_range_answers = HashMap::new();
        let (i, j) = (block.start, block.end);
        for start_index in i..=j {
            for end_index in start_index..=j {
                if start_index == end_index {
                    // RMQ(i, i) = i
                    all_range_answers.insert(
                        Range::new(start_index, end_index),
                        InBlockOffset::new(i, start_index),
                    );
                } else {
                    let prev_range = Range::new(start_index, end_index - 1);
                    let cur_range = Range::new(start_index, end_index);
                    let prev_range_min = all_range_answers.get(&prev_range).unwrap();

                    // RMQ(i, j) = min(RMQ(i, j-1), A[j])
                    if array[prev_range_min.base_index] > array[end_index] {
                        all_range_answers.insert(cur_range, InBlockOffset::new(i, end_index));
                    } else {
                        let prev_idx = prev_range_min.base_index;
                        all_range_answers.insert(cur_range, InBlockOffset::new(i, prev_idx));
                    }
                }
            }
        }
        all_range_answers
    }
}

#[cfg(test)]
mod test {
    #[test]
    fn test_cartesian_tree() {
        use super::{FischerHeunRMQ, Range};
        let array = vec![2, 43, 45, 12, 34, 54, 6, 7, 1, 2, 89, 89, 78, 2, 3, 5, 62];
        let last = array.len() - 1;
        let rmq = FischerHeunRMQ::new(array);
        let ans = rmq.query(Range::new(0, last));
    }
}
