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

use std::collections::{HashMap, LinkedList as Stack};

/// Each our our blocks will be of size 64
static MAX_BLOCK_SIZE: usize = 64;

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

impl Range {
    pub fn new(i: usize, j: usize) -> Self {
        Range { start: i, end: j }
    }
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

/// A table mapping powers of 2 k = (0, 1, ...) such that
/// 2 << k fits withing an underlying array to precomputed RMQ
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

    /// A precomputed list of MSB values for a 16 bit number. We use this to
    /// speed up msb checks
    msb_16: [u8; 1 << 16],
}

impl<T: Ord + Copy> FischerHeunRMQ<T> {
    /// Creates a new FischerHeunRMQ structure and initializes
    /// all of its caches. The expectation is that the array
    /// on which we are building the structure is static. In the process
    /// of creation, it initializes all the data structures needed
    /// to answer ad hoc range queries
    pub fn new(static_array: Vec<T>) -> Self {
        let minimums = Self::generate_macro_array(&static_array, MAX_BLOCK_SIZE);
        let (block_level_rmq, cartesian_number_cache) =
            Self::compute_cached_dense_tables(&static_array, MAX_BLOCK_SIZE);
        let summary_rmq_sparse_table = Self::compute_sparse_table(&minimums);
        let msb_16 = Self::compute_msb_values();
        FischerHeunRMQ {
            static_array,
            minimums,
            block_level_rmq,
            summary_rmq_sparse_table,
            cartesian_number_cache,
            msb_16,
        }
    }

    /// Returns the smallest value in [range.start, range.end] in constant time
    pub fn query(&self, range: Range) -> T {
        let start_block = range.start / MAX_BLOCK_SIZE;
        let end_block = range.end / MAX_BLOCK_SIZE;
        let min_at_ends = self.get_min_value_at_ends(start_block, end_block);
        let min_in_intermediate = self.get_intermediate_min(range);
        std::cmp::min(min_at_ends.unwrap(), min_in_intermediate) // TODO
    }

    /// Finds and returns the smallest value in the blocks that the range passed
    /// into query starts and ends. To do that, it uses the per-block dense
    /// tables that have been precomputed. With this scheme, we are able to answer
    /// this portion of the query in O(1)
    fn get_min_value_at_ends(&self, start_block: usize, end_block: usize) -> Option<T> {
        let start_ct_number = *self.cartesian_number_cache.get(&start_block).unwrap();
        let end_ct_number = *self.cartesian_number_cache.get(&end_block).unwrap();
        let start_min = self
            .block_level_rmq
            .get(&start_ct_number)
            .and_then(|m| m.get(&Range::new(start_block, start_block + MAX_BLOCK_SIZE)));
        let end_min = self
            .block_level_rmq
            .get(&end_ct_number)
            .and_then(|m| m.get(&Range::new(end_block, end_block + MAX_BLOCK_SIZE)));
        let lopt = start_min.and_then(|l| Some(self.static_array[start_block + l.inblock_index]));
        let ropt = end_min.and_then(|r| Some(self.static_array[end_block + r.inblock_index]));
        if let (Some(l), Some(r)) = (lopt, ropt) {
            Some(std::cmp::min(l, r))
        } else {
            None
        }
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
        let k = Self::get_msb(&self.msb_16, (range.end - range.start) + 1);
        let summary_answers = self.summary_rmq_sparse_table.get(&(range.start, 1 << k)).unwrap();
        let left = Range::new(range.start, range.start + (1 << k) - 1);
        let right = Range::new(range.end - (1 << k) + 1, range.end);
        let left_min = summary_answers.get(&left).unwrap().base_index;
        let right_min = summary_answers.get(&right).unwrap().base_index;
        std::cmp::min(self.static_array[left_min], self.static_array[right_min])
    }

    /// Finds the index of the most significant bit in the given number.
    /// n is assumed to be a 64 bit unsigned integer.
    fn get_msb(lookup_16: &[u8], n: usize) -> usize {
        debug_assert!(n != 0);
        let mask = 0b1111_1111_1111_1111;
        let bot_16 = lookup_16[n & mask];
        let lmid_16 = lookup_16[(n >> 16) & mask];
        let rmid_16 = lookup_16[(n >> 32) & mask];
        let top_16 = lookup_16[n >> 48];
        let mut val = bot_16;
        val += if lmid_16 != 0 { lmid_16 + 16 } else { lmid_16 };
        val += if rmid_16 != 0 { rmid_16 + 32 } else { rmid_16 };
        val += if top_16 != 0 { top_16 + 48 } else { top_16 };
        (val - 1) as usize
    }

    /// Precomputes a lookup table of msb values for all
    /// 16 bit unsigned integers. This table is used to do
    /// O(1) msb lookups, using 4 array accesses, when
    /// computing msb values for unsigned 64 bit integers
    pub fn compute_msb_values() -> [u8; 1 << 16] {
        let mut v = [0; 1 << 16];
        for k in 0..16 {
            let start = 1 << k;
            let end = start << 1;
            for i in start..end {
                v[i] = k + 1;
            }
        }
        v
    }

    /// Generates the array for min values in each block. The generated
    /// array forms the basis of our 'macro' problem portion of the
    /// method of four russians.
    ///
    /// Suppose static_array.len() = 7 and block_size=3
    /// i = 0, 1, 2, 3, 4, 5, 6
    /// cur_range = {0, 2}, {3, 5}, {6, 6}
    pub fn generate_macro_array(static_array: &Vec<T>, block_size: usize) -> Vec<T> {
        let mut minimums = Vec::new();
        for i in (0..static_array.len()).step_by(block_size) {
            let cur_range = Range::new(i, i + block_size - 1);
            let cur_min_idx = Self::min_index_in_range(static_array, cur_range);
            minimums.push(static_array[cur_min_idx])
        }
        minimums
    }

    /// For each index i, compute RMQ for ranges starting at i of
    /// size 1, 2, 4, 8, 16, …, 2^k as long as they fit in the array.
    /// For each array element, we compute lg n ranges. Therefore,
    /// the total cost of the procedure is O(n lg n)
    pub fn compute_sparse_table(minimums: &Vec<T>) -> SparseTable {
        let mut table = HashMap::new();
        let array = minimums;
        for i in 0..array.len() {
            let mut k = 0;
            while 1 << k < array.len() {
                let end_idx = i + (1 << k) - 1;
                let range_len = (end_idx - i) + 1;
                let query_range = Range::new(i, end_idx);
                let rmq_answers_in_range = Self::compute_rmq_all_ranges(array, &query_range);
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
        block_size: usize,
    ) -> (BlockLevelRMQ, HashMap<usize, u128>) {
        let mut block_level_rmq = HashMap::new();
        let mut cartesian_cache = HashMap::new();
        for i in (0..static_array.len()).step_by(block_size) {
            let cur_range = Range::new(i, i + block_size - 1);
            // todo!(): Last portion
            let cur_number = Self::cartesian_tree_number(&cur_range, static_array);
            cartesian_cache.insert(i, cur_number);
            if block_level_rmq.contains_key(&cur_number) {
                continue;
            }
            let rmq = Self::compute_rmq_all_ranges(static_array, &cur_range);
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
        let (end, len) = (block_range.end, static_array.len());
        let j = if end >= len { len - 1 } else { end };

        // Technically, we do not even need to build the certesian tree, all
        // we need to know is the series of stack push and pop operations
        // needed to create the tree.
        let mut cartesian_tree = Vec::with_capacity(MAX_BLOCK_SIZE);

        // We initialize each item in the block as its own
        // cartesian tree with no children
        for i in block_range.start..=j {
            cartesian_tree.push(CartesianTreeNode::new(i));
        }

        // To create the cartesian tree, we pop the stack until either
        // it's empty or the element atop the stack has a smaller value
        // than the element we are currently trying to add to the stack.
        // Once we break out of the `pop` loop, we make the item we popped
        // a left child of the new item we are adding. Additionally, we make
        // this new item a right/left child of the item atop the stack
        let mut offset_in_number = 0;
        for i in block_range.start..=j {
            let mut last_popped = None;
            loop {
                match stack.front() {
                    None => break,
                    Some(&top_node_index) => {
                        if static_array[top_node_index] < static_array[i] {
                            cartesian_tree[top_node_index - block_range.start].index_of_right_child = Some(i);
                            break;
                        }
                        last_popped = stack.pop_front();
                        offset_in_number += 1;
                    }
                }
            }
            if let Some(last_popped_idx) = last_popped {
                cartesian_tree[i - block_range.start].index_of_left_child = Some(last_popped_idx);
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
    pub fn min_index_in_range(static_array: &Vec<T>, range: Range) -> usize {
        let end = range.end;
        let len = static_array.len();
        let cur_block = if end < len {
            &static_array[range.start..=end]
        } else {
            &static_array[range.start..]
        };
        let inblock_min_idx =
            cur_block
                .iter()
                .enumerate()
                .fold(0, |cur_min_idx, (cur_idx, &x)| match x.cmp(&cur_block[cur_min_idx]) {
                    std::cmp::Ordering::Less => cur_idx,
                    _ => cur_min_idx,
                });
        range.start + inblock_min_idx
    }

    /// This procedure is used to compute the answers to all
    /// n << 2 RMQ queries that can ever be asked in the range passed in.
    /// This procedure is used to pre-compute the RMQ values on
    /// individual blocks. It returns a mapping from a Range
    /// to the index of the minimal value in that range
    pub fn compute_rmq_all_ranges(array: &Vec<T>, block: &Range) -> HashMap<Range, InBlockOffset> {
        let mut all_range_answers = HashMap::new();
        let i = block.start;
        let (end, len) = (block.end, array.len());
        let j = if end >= len { len - 1 } else { end };
        for start_index in i..=j {
            for end_index in start_index..=j {
                if start_index == end_index {
                    // RMQ(i, i) = i
                    all_range_answers.insert(Range::new(start_index, end_index), InBlockOffset::new(i, start_index));
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
    fn get_msb() {
        use super::FischerHeunRMQ;
        let v = FischerHeunRMQ::<u64>::compute_msb_values();
        let msb = FischerHeunRMQ::<u64>::get_msb(&v, 16);
        assert_eq!(4, msb);
        let msb = FischerHeunRMQ::<u64>::get_msb(&v, 256);
        assert_eq!(8, msb);
        let msb = FischerHeunRMQ::<u64>::get_msb(&v, 873);
        assert_eq!(9, msb);
        let msb = FischerHeunRMQ::<u64>::get_msb(&v, 25);
        assert_eq!(4, msb);
        let msb = FischerHeunRMQ::<u64>::get_msb(&v, 91);
        assert_eq!(6, msb);
        let msb = FischerHeunRMQ::<u64>::get_msb(&v, 255);
        assert_eq!(7, msb);
        let msb = FischerHeunRMQ::<u64>::get_msb(&v, 1);
        assert_eq!(0, msb);
        let base: usize = 2;
        let msb = FischerHeunRMQ::<u64>::get_msb(&v, base.pow(63));
        assert_eq!(63, msb);
        let msb = FischerHeunRMQ::<u64>::get_msb(&v, base.pow(32));
        assert_eq!(32, msb);
        let msb = FischerHeunRMQ::<u64>::get_msb(&v, base.pow(16));
        assert_eq!(16, msb);
        let msb = FischerHeunRMQ::<u64>::get_msb(&v, 1 << 18);
        assert_eq!(18, msb);
    }

    #[test]
    fn min_index_in_range() {
        use super::{FischerHeunRMQ, Range};
        let v = vec![2, 32, 45, 64, 21, 78, 36, 27, 8, 21];
        let min_idx = FischerHeunRMQ::<u64>::min_index_in_range(&v, Range::new(0, 311));
        assert_eq!(2, v[min_idx])
    }

    #[test]
    fn generate_macro_array() {
        use super::FischerHeunRMQ;
        let v = vec![2, 32, 45, 64, 21, 78, 36, 27, 8, 21];
        let mins = FischerHeunRMQ::<u64>::generate_macro_array(&v, 4);
        assert_eq!(vec![2, 21, 8], mins)
    }

    #[test]
    fn compute_rmq_all_ranges() {
        use super::{FischerHeunRMQ, Range};
        let v = vec![2, 32, 45, 64, 21, 78, 36, 27, 8, 21];
        let ans = FischerHeunRMQ::<u64>::compute_rmq_all_ranges(&v, &Range::new(0, 9));
        assert_eq!(ans.get(&Range::new(0, 5)).unwrap().base_index, 0);
        assert_eq!(ans.get(&Range::new(1, 5)).unwrap().base_index, 4);
        assert_eq!(ans.get(&Range::new(7, 9)).unwrap().base_index, 8);
    }

    #[test]
    fn compute_sparse_table() {
        use super::{FischerHeunRMQ, Range};
        let v = vec![2, 32, 45, 64, 21, 78, 36, 27, 8, 21, 1, 34, 43];
        let mins = FischerHeunRMQ::<u64>::generate_macro_array(&v, 3);
        assert_eq!(vec![2, 21, 8, 1, 43], mins);
        let sparse = FischerHeunRMQ::<u64>::compute_sparse_table(&mins);
        println!("{:?}", sparse);
        todo!()
    }

    // #[test]
    // fn cartesian_tree() {
    //     use super::{FischerHeunRMQ, Range};
    //     let v = vec![2, 32, 45, 64, 21, 78, 36, 27, 8, 21, 1, 34, 43];
    //     todo!()
    // }

    #[test]
    fn compute_cached_dense() {
        todo!()
    }

    #[test]
    fn query() {
        use super::{FischerHeunRMQ, Range};
        let v = vec![2, 32, 45, 64, 21, 78, 36, 27, 8, 21, 1, 34, 43];
        let rmq = FischerHeunRMQ::new(v);
        assert_eq!(rmq.query(Range::new(0, 4)), 2);
    }
}
