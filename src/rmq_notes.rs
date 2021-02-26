use itertools::Itertools;
use std::{cmp::Ordering, collections::HashMap, hash::Hash, todo};

/// An inclusive ([i, j]), 0 indexed range for specifying a range
/// query.
#[derive(Debug, Eq, PartialEq, Hash)]
pub struct RMQRange<'a, T> {
    /// The starting index, i
    start_idx: usize,

    /// The last index, j
    end_idx: usize,

    /// The array to which the indexes above refer. Keeping
    /// a reference here ensures that some key invariants are
    /// not violated. Since it is expected that the underlying
    /// array will be static, we'll never make a mutable reference
    /// to it. As such, storing shared references in many
    /// different RMQRange objects should be fine
    underlying: &'a [T],
}

impl<'a, T> From<(usize, usize, &'a [T])> for RMQRange<'a, T> {
    fn from(block: (usize, usize, &'a [T])) -> Self {
        let start_idx = block.0;
        let end_idx = block.1;
        let len = block.2.len();
        if start_idx > end_idx {
            panic!("Range start cannot be larger than the range's end")
        }
        if end_idx >= len {
            panic!("Range end cannot be >= the len of underlying array")
        }
        RMQRange {
            start_idx,
            end_idx,
            underlying: block.2,
        }
    }
}

type LookupTable<'a, T> = HashMap<RMQRange<'a, T>, usize>;

fn compute_rmq_all_ranges_old<'a, T: Hash + Eq + Ord>(array: &'a [T]) -> LookupTable<'a, T> {
    let len = array.len();
    let mut lookup_table = HashMap::with_capacity((len * len) / 2);
    for start in 0..len {
        for end in start..len {
            if start == end {
                lookup_table.insert((start, end, array).into(), start);
            } else {
                let prev_range = (start, end - 1, array).into();
                let new_range = (start, end, array).into();
                let mut min_idx = *lookup_table.get(&prev_range).unwrap();
                if array[min_idx] > array[end] {
                    min_idx = end;
                }
                lookup_table.insert(new_range, min_idx);
            }
        }
    }
    lookup_table
}

impl From<(usize, usize)> for SparseTableIdx {
    fn from(idx_tuple: (usize, usize)) -> Self {
        let start_idx = idx_tuple.0;
        let len = idx_tuple.1;
        if !len.is_power_of_two() {
            panic!("Expected the length to be a power or 2")
        }
        SparseTableIdx { start_idx, len }
    }
}

///
type SparseTable<'a, T> = HashMap<SparseTableIdx, RMQResult<'a, T>>;

/// For each index `i`, compute RMQ answers for ranges starting at `i` of
/// size `1, 2, 4, 8, 16, …, 2^k` as long as the resultant ending index
/// fits in the underlying array in the array.
/// For each array index, we compute lg n ranges. Therefore,
/// the total cost of the procedure is O(n lg n)
fn compute_rmq_sparse_table_old<'a, T: Hash + Eq + Ord>(array: &'a [T]) -> SparseTable<'a, T> {
    let len = array.len();
    let mut sparse_table = HashMap::new();
    for start_idx in 0..len {
        let mut power = 0;
        let mut end_idx = start_idx + (1 << power) - 1;
        while end_idx < len {
            let cur_range_idx: SparseTableIdx = (start_idx, 1 << power).into();
            let cur_range_lookup_table = get_min_by_scanning(&array[start_idx..=end_idx]);
            sparse_table.insert(cur_range_idx, cur_range_lookup_table);
            power += 1;
            end_idx = start_idx + (1 << power) - 1;
        }
    }
    sparse_table
}

/// Here's the scheme we shall use to implement the lookup table:
///     - First, we shall assume that the values we get are 8 bytes (64bits) wide
///     - We shall precompute all MSB(n) values for all n <= 2^16. This will use
///       65536 bytes which is approximately 66Kb.
///     - To find the MSB of any value, we combine answers from the 4 16 bit
///       portions using logical shifts and masks
#[derive(Debug)]
pub struct MSBLookupTable([u8; 1 << 16]);

impl MSBLookupTable {
    /// Build the lookup table. To fill up the table, we simply subdivide
    /// it into segements whose sizes are powers of two. The MSB for a segment
    /// are the same for instance:
    ///   ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    /// n       |1|2|3|4|5|6|7|8|9|10|11|12|13|14|15|16|17|18|19|...
    /// msb(n)  |0|1|1|2|2|2|2|3|3|3 |3 |3 |3 |3 |3 |4 |4 |4 |4 |...
    ///   ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    pub fn build() -> Self {
        let mut lookup_table = [0; 1 << 16];
        let mut msb_idx = 0;
        for (i, cur_msb) in lookup_table.iter_mut().enumerate() {
            let n = i + 1;
            if n > 1 && n.is_power_of_two() {
                msb_idx += 1;
            }
            *cur_msb = msb_idx;
        }
        MSBLookupTable(lookup_table)
    }

    /// Get the most significant bit of the given 64 bit value. A 64 bit
    /// bit number can be subdivided into 4 16 bit portions. Since we have
    /// pre-calculated the msb values for all 16 possible 16 bit integers,
    /// we can find the msb of the number by combining answers to each segment
    pub fn get_msb_idx_of(&self, n: usize) -> u8 {
        debug_assert!(n != 0);
        let bit_mask = 0xFFFF;
        if n >> 48 > 0 {
            let d_idx = (n >> 48) - 1;
            self.0[d_idx] + 48
        } else if n >> 32 > 0 {
            let c_idx = ((n >> 32) & bit_mask) - 1;
            self.0[c_idx] + 32
        } else if n >> 16 > 0 {
            let b_idx = ((n >> 16) & bit_mask) - 1;
            self.0[b_idx] + 16
        } else {
            let a_idx = (n & bit_mask) - 1;
            self.0[a_idx]
        }
    }
}

#[test]
fn get_msb() {
    let lookup_table = MSBLookupTable::build();
    assert_eq!(4, lookup_table.get_msb_idx_of(16));
    assert_eq!(8, lookup_table.get_msb_idx_of(256));
    assert_eq!(9, lookup_table.get_msb_idx_of(873));
    assert_eq!(4, lookup_table.get_msb_idx_of(25));
    assert_eq!(6, lookup_table.get_msb_idx_of(91));
    assert_eq!(7, lookup_table.get_msb_idx_of(255));
    assert_eq!(0, lookup_table.get_msb_idx_of(1));
    let base: usize = 2;
    assert_eq!(63, lookup_table.get_msb_idx_of(base.pow(63)));
    assert_eq!(32, lookup_table.get_msb_idx_of(base.pow(32)));
    assert_eq!(16, lookup_table.get_msb_idx_of(base.pow(16)));
    assert_eq!(18, lookup_table.get_msb_idx_of(1 << 18));
}

/// The abstraction for a single block.
pub struct MedianBlock<'a, T> {
    /// The starting index. This is 0-indexed and should be
    /// less than or equal to the end_idx
    start_idx: usize,

    /// The ending index. This should be strictly less than the
    /// length of the underlying array. Further, end_idx - start_idx should
    /// be 5 for all except possibly the last block
    end_idx: usize,

    /// The index of the median value in the given range. To move from this
    /// index to an indx in the underlying, we simply calculate
    /// `start_idx + median_idx`
    median_idx: usize,

    /// The median of this block
    median: &'a T,
}

// impl <'a, T: Ord> Ord for MedianBlock<'a, T>  {
//     fn cmp(&self, other: &Self) -> Ordering {
//         self.median.cmp(&other.median)
//     }
// }

// impl <'a, T: Ord> Eq for MedianBlock<'a, T>  {
// }

// impl <'a, T: Ord> PartialOrd for MedianBlock<'a, T>  {
//     fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
//         Some(self.median.cmp(other.median))
//     }
// }

// impl <'a, T: Ord> PartialEq for MedianBlock<'a, T>  {
//     fn eq(&self, other: &Self) -> bool {
//         self.median == other.median
//     }
// }

/// For quick construction and error checking. (start_idx, end_idx, underlying, median_idx)
impl<'a, T> From<(usize, usize, &'a [T], usize)> for MedianBlock<'a, T> {
    fn from(block: (usize, usize, &'a [T], usize)) -> Self {
        let start_idx = block.0;
        let mut end_idx = block.1;
        let len = block.2.len();
        let median_idx = block.3;
        if end_idx >= len {
            end_idx = len - 1;
        }
        debug_assert!(start_idx < end_idx);
        debug_assert!(median_idx >= start_idx && median_idx <= end_idx);
        if end_idx < len {
            debug_assert!(end_idx - start_idx == 5);
        }
        MedianBlock {
            start_idx,
            end_idx,
            median_idx,
            median: &block.2[median_idx],
        }
    }
}

/// Computes the index of the k-th smallest element in the `array`. This is sometimes
/// referred to as the k-th order statistic. This procedure computes this value in
/// O(n).
fn kth_order_statistic<'a, T: Ord + Clone>(array: &'a mut [T], k: usize) -> &T {
    match 5.cmp(&array.len()) {
        Ordering::Less | Ordering::Equal => &array[get_kth_by_sorting(array, k)],
        Ordering::Greater => {
            let mut macro_array = generate_macro_array(array);
            let approx_median_idx = get_approx_median_idx(macro_array.as_mut_slice());
            let left_range_size = partition_at_pivot(array, approx_median_idx);
            match k.cmp(&left_range_size) {
                Ordering::Equal => &array[k - 1],
                Ordering::Less => kth_order_statistic(&mut array[..left_range_size], k),
                Ordering::Greater => kth_order_statistic(&mut array[left_range_size..], k - left_range_size),
            }
        }
    }
}

#[test]
fn test_mom() {
    let mut a = [
        1, 7, 4, 8, 5, 8, 2, 5, 6, 2, 43, 1, 43, 56, 5, 7, 12, 34, 54, 76, 89, 1, 32, 4, 56, 12, 5, 67, 89,
    ];
    let mut b = a.clone();
    b.sort();
    for (i, val) in b.iter().enumerate() {
        let ith = kth_order_statistic(&mut a, i);
        assert_eq!(ith, val)
    }
}

fn get_approx_median_idx<'a, T: Ord + Clone>(macro_array: &mut [MedianBlock<'a, T>]) -> usize {
    let median_pos = macro_array.len() / 2;
    let mut medians: Vec<_> = macro_array.iter().map(|x| x.median.clone()).collect();
    let approx_median = kth_order_statistic(&mut medians, median_pos);
    let mut median_idx = 0;
    for block in macro_array {
        if block.median == approx_median {
            median_idx = block.start_idx + block.median_idx;
        }
    }
    median_idx
}

/// Reorient the elements of the array around the element at `pivot_idx` and
/// return the length of the left partition
fn partition_at_pivot<T: Ord>(array: &mut [T], pivot_idx: usize) -> usize {
    let last_idx = array.len() - 1;
    array.swap(pivot_idx, last_idx);
    let mut less_than_tail = 0;
    for cur_idx in 0..last_idx {
        if array[cur_idx] <= array[last_idx] {
            array.swap(less_than_tail, cur_idx);
            less_than_tail += 1;
        }
    }
    array.swap(less_than_tail, last_idx);
    less_than_tail + 1
}

#[test]
fn partition() {
    let mut a = [1, 7, 4, 8, 5, 8, 2, 5, 6];
    assert_eq!(partition_at_pivot(&mut a, 8), 6);
    println!("{:?}", a)
}

fn generate_macro_array<'a, T: Ord>(array: &'a [T]) -> Vec<MedianBlock<'a, T>> {
    let mut blocks = Vec::with_capacity(array.len() / 5);
    for start_idx in (0..array.len()).step_by(5) {
        let end_idx = (start_idx + 5) - 1;
        let block = &array[start_idx..=end_idx];
        let median_idx = get_kth_by_sorting(block, block.len() / 2);
        blocks.push((start_idx, end_idx, array, median_idx).into())
    }
    blocks
}

fn get_kth_by_sorting<T: Ord>(block: &[T], k: usize) -> usize {
    let kth = block.iter().enumerate().sorted_by_key(|x| x.1).map(|x| x.0).nth(k);
    kth.unwrap()
}

#[test]
fn median_by_sorting() {
    let even = [3, 7, 1, 3];
    let odd = [2, 3, 9, 1, 3];
    let even_median_idx = get_kth_by_sorting(&even, even.len() / 2);
    let odd_median_idx = get_kth_by_sorting(&odd, odd.len() / 2);
    assert_eq!(3, even[even_median_idx]);
    assert_eq!([3, 7, 1, 3], even);

    assert_eq!(3, odd[odd_median_idx]);
    assert_eq!([2, 3, 9, 1, 3], odd);
}

/// The abstraction for a single block.
#[derive(Debug, Eq, Hash, Clone)]
pub struct RMQBlock<'a, T: Ord> {
    /// The starting index. This is 0-indexed and should be
    /// less than or equal to the end_idx
    start_idx: usize,

    /// The ending index. This should be strictly less than the
    /// length of the underlying array. Further, end_idx - start_idx should
    /// be 5 for all except possibly the last block
    end_idx: usize,

    /// The index of the median value in the given range. To move from this
    /// index to an idx in the underlying, we simply calculate
    /// `start_idx + median_idx`
    min_idx: usize,

    /// The median of this block
    min: &'a T,
}

impl<'a, T: Ord> From<(usize, usize, RMQResult<'a, T>)> for RMQBlock<'a, T> {
    fn from((start_idx, end_idx, res): (usize, usize, RMQResult<'a, T>)) -> Self {
        RMQBlock {
            start_idx,
            end_idx,
            min_idx: res.min_idx,
            min: res.min_value,
        }
    }
}

// pub trait RMQSolver<T: Ord> {
//     /// Create a solver capable of answering Range Min Queries
//     /// in the given array slice
//     fn build(array: &[T]) -> Self;

//     /// Returns the smallest element in the given range
//     fn solve(&self, range: (usize, usize)) -> &T;
// }

#[derive(Debug)]
pub struct RMQBlockDecomposition<'a, T: Ord> {
    /// An aggregation of the minimal values in each of our
    /// (n/b) blocks.
    macro_array: Vec<RMQBlock<'a, T>>,

    /// The size of each block
    block_size: usize,

    /// The static array onto which this data structure is layered
    underlying: &'a [T],
}

impl<'a, T: Ord> RMQBlockDecomposition<'a, T> {}

#[derive(Debug, PartialEq, Eq, Hash, Clone, PartialOrd, Ord)]
pub struct SuffixIndex(usize);

/// To make things even more ergonomic, we implement the `Index` trait to allow
/// us to use our new type without retrieving the wrapped index. For now,
/// We assume that our string will be a collection of bytes. That is of course
/// the case for the ascii alphabet
impl std::ops::Index<SuffixIndex> for [u8] {
    type Output = u8;
    fn index(&self, index: SuffixIndex) -> &Self::Output {
        &self[index.0]
    }
}

/// This is an index into the suffix array
#[derive(Debug, PartialEq, Eq, Hash, Clone, Ord, PartialOrd)]
pub struct SuffixArrayIndex(usize);

impl<'a> std::ops::Index<SuffixArrayIndex> for SuffixArray<'a> {
    type Output = str;
    fn index(&self, index: SuffixArrayIndex) -> &Self::Output {
        let suffix_idx = &self.suffix_array[index.0];
        &self.underlying[suffix_idx.0..]
    }
}

#[derive(Debug)]
pub struct SuffixArray<'a> {
    /// The string over which we are building this suffix array
    underlying: &'a str,

    /// The suffix array is simply all the suffixes of the
    /// underlying string in sorted order
    suffix_array: Vec<SuffixIndex>,
}

impl<'a> SuffixArray<'a> {
    /// Construct the suffix array by sorting. This has worst case performance
    /// of O(n log n)
    pub fn make_sa_naive(s: &'a str) -> Self {
        let mut suffixes = vec![];
        for i in 0..s.len() {
            suffixes.push(&s[i..]);
        }
        suffixes.sort();
        let mut suffix_array = vec![];
        for suffix in suffixes {
            let cur = SuffixIndex(s.len() - suffix.len());
            suffix_array.push(cur);
        }
        Self {
            underlying: s,
            suffix_array,
        }
    }

    fn len(&self) -> usize {
        self.suffix_array.len()
    }
}

impl<'a> std::fmt::Display for SuffixArray<'a> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut res = String::new();
        res += "  sa_id  |  suffix_id | suffix             \n";
        res += "---------|------------|--------------------\n";
        for id in 0..self.len() {
            let idx = SuffixArrayIndex(id);
            if id <= 9 {
                res += &format!(
                    " {:?}       |{:?}        |{}",
                    id,
                    self.suffix_array[idx.0].0,
                    self[idx].to_string()
                );
            } else {
                res += &format!(
                    " {:?}      |{:?}      |{}",
                    id,
                    self.suffix_array[idx.0].0,
                    self[idx].to_string()
                );
            }
            res += "\n";
        }
        writeln!(f, "{}", res)
    }
}

#[test]
fn suff_idx() {
    let idx = SuffixIndex(4);
    let slice = &[21u8, 3, 4, 56, 76, 34, 21, 90];
    assert_eq!(76, slice[idx])
}

/// The length of the longest common prefix between the
/// suffixes that start at `left` and `right`. These
/// suffixes are adjacent to each other in the suffix array
#[derive(Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct LCPHeight {
    left: SuffixIndex,
    right: SuffixIndex,
    height: usize,
}

impl From<(SuffixIndex, SuffixIndex, usize)> for LCPHeight {
    fn from((l, r, h): (SuffixIndex, SuffixIndex, usize)) -> Self {
        LCPHeight {
            left: l,
            right: r,
            height: h,
        }
    }
}

impl<'a> SuffixArray<'a> {
    /// Retrieve the index of the siffic array stored at this location
    /// in the suffix array. Put another way, we retrieve the id
    /// of the (idx + 1) smallest suffix in the string
    pub fn get_suffix_idx_at(&self, idx: usize) -> SuffixIndex {
        // These clones are quite cheap
        self.suffix_array[idx].clone()
    }
}

pub fn make_lcp_by_scanning(sa: &SuffixArray) -> Vec<LCPHeight> {
    let mut lcp_len_array = Vec::with_capacity(sa.len());
    for i in 1..sa.len() {
        let prev_sa_idx = SuffixArrayIndex(i - 1);
        let cur_sa_idx = SuffixArrayIndex(i);
        let lcp_len = calculate_lcp_len(&sa[prev_sa_idx], &sa[cur_sa_idx]);
        lcp_len_array.push((sa.get_suffix_idx_at(i - 1), sa.get_suffix_idx_at(i), lcp_len).into());
    }
    lcp_len_array
}

/// Calculate the length of the longest common prefix
/// between the two string slices in linear time
fn calculate_lcp_len(left: &str, right: &str) -> usize {
    let mut len = 0;
    for (l, r) in left.as_bytes().iter().zip(right.as_bytes()) {
        if l != r {
            break;
        }
        len += 1;
    }
    len
}

impl std::fmt::Display for SuffixIndex {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl std::fmt::Display for LCPHeight {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "({} * {} = {}) ", self.left, self.right, self.height)
    }
}

#[test]
fn test_lcp_len() {
    let l = "banana";
    let r = "banaftgshkksl";
    assert_eq!(calculate_lcp_len(l, r), 4);
}

#[test]
fn lcp_naive() {
    let test_str = "bananabandana$";
    let sa = SuffixArray::make_sa_naive(&test_str);
    println!("{}", sa);
    let lcp = make_lcp_by_scanning(&sa);
    println!("{:?}", lcp)
}

/// Computes the lcp array in O(n) using Kasai's algorithm. This procedure assumes that
/// the sentinel character has been appended onto `s`.
pub fn make_lcp_by_kasai(s: &str, sa: &SuffixArray) -> Vec<LCPHeight> {
    let s_ascii = s.as_bytes();
    let mut lcp_array = Vec::with_capacity(s.len());
    // We need a quick way to move from the index of the suffix in the
    // string (the `SuffixIndex`) to the index of that same string in the
    // suffix array (the `SuffixArrayIndex` aka the rank). This map
    // will allow us to do that once populated.
    let mut suffix_index_to_rank = HashMap::with_capacity(s.len());
    for i in 1..s.len() {
        suffix_index_to_rank.insert(sa.get_suffix_idx_at(i), SuffixArrayIndex(i));
        lcp_array.push((sa.get_suffix_idx_at(i - 1), sa.get_suffix_idx_at(i), 0).into())
    }
    let mut h = 0;
    // We then loop over all the suffixes one by one. One thing to note that we
    // are looping over the suffixes in the order they occur in the underlying
    // string. This order is different from the order in which they occur in
    // the suffix array. This means that we will not fill the `lcp_array`
    // in order
    for i in 0..s.len() - 1 {
        let cur_suffix_index = SuffixIndex(i);
        // We are currently processing the suffix that starts at index `i` in
        // the underlying string. We'd like to know where this suffix
        // is located in the lexicographically sorted suffix array
        let location_in_sa = suffix_index_to_rank.get(&cur_suffix_index).unwrap();

        // grab a hold of the id of the suffix that is just before `location_in_sa`
        // in the suffix array
        let left_adjacent_suffix = sa.get_suffix_idx_at(location_in_sa.0 - 1);

        // Here, we compute the length of the longest common prefix between
        // the current suffix and the suffix that is left of it in the suffix
        // array
        while s_ascii[cur_suffix_index.0 + h] == s_ascii[left_adjacent_suffix.0 + h] {
            h += 1
        }
        lcp_array[location_in_sa.0 - 1] = (left_adjacent_suffix, cur_suffix_index, h).into();

        // When we move from i to i+1, we are effectively moving from processing the suffix
        // A[i..] to processing the suffix A[i+1..]. Notice how this is the same as moving
        // to processing a suffix formed by dropping the first character of A[i..]. Therefore,
        // Theorem 1 from Kasai et al. tells us that the lcp between the new shorter suffix
        // and the suffix adjacent to its left in the suffix array is at least `h-1`
        if h > 0 {
            h -= 1
        }
    }
    lcp_array
}

#[test]
fn lcp_kasai() {
    let test_str = "bananabandana$";
    let sa = SuffixArray::make_sa_naive(&test_str);
    println!("{}", sa);
    let lcp = make_lcp_by_scanning(&sa);
    let naive_len = lcp.len();
    let lcp_kasai = make_lcp_by_kasai(test_str, &sa);
    assert_eq!(naive_len, lcp_kasai.len());
    for (naive, kasai) in lcp.iter().zip(lcp_kasai.iter()) {
        assert_eq!(naive, kasai)
    }
}

#[derive(Debug)]
pub struct Suffix<'a> {
    start: SuffixIndex,
    suffix_type: SuffixType,
    underlying: &'a str,
}

impl<'a> Suffix<'a> {
    /// Is this suffix a left most `S-type` suffix?
    pub fn is_lms(&self) -> bool {
        match self.suffix_type {
            SuffixType::L => false,
            SuffixType::S(lms) => lms,
        }
    }
}

impl<'a> From<(SuffixIndex, SuffixType, &'a str)> for Suffix<'a> {
    /// Turn a 3-tuple into a Suffix object
    fn from((start, suffix_type, underlying): (SuffixIndex, SuffixType, &'a str)) -> Self {
        Suffix {
            start,
            suffix_type,
            underlying,
        }
    }
}

#[derive(Debug, PartialEq, Eq, Clone)]
enum SuffixType {
    /// A variant for S-type suffixes. The associated boolen
    /// indicates whether this suffix is an `LMS` suffix. We
    /// discuss what that means below.
    S(bool),

    /// A variant for L-type suffixes
    L,
}

/// The first character of the suffixes in a bucket
/// uniquely identifies that bucket
#[derive(Eq, PartialEq, Hash, Debug)]
pub struct BucketId<T>(T);

#[derive(Debug)]
pub struct Bucket {
    /// Index in suffix array where this bucket begins
    start: SuffixArrayIndex,

    /// Region in suffix array where this bucket ends.
    /// Note that these indexes are 0-based and inclusive
    end: SuffixArrayIndex,

    /// Tells us how many suffixes have been inserted at the start
    /// of this bucket. Doing `start + offset_from_start` gives us
    /// the next empty index within this bucket from the start
    /// The utility of this will become clear once we look at
    /// the mechanics of induced sorting
    offset_from_start: usize,

    /// Used in a similar manner as `offset_from_start` just
    /// from the end index
    offset_from_end: usize,

    /// Used for inserting `lms` suffixes in a bucket
    lms_offset_from_end: usize,
}

impl<'a> Suffix<'a> {
    pub fn get_bucket(&self) -> BucketId<u8> {
        let first_char = self.underlying.chars().nth(self.start.0);
        debug_assert!(first_char.is_some());
        BucketId(first_char.unwrap() as u8)
    }
}

impl<'a> SuffixArray<'a> {
    /// Create a list of the suffixes in the string. We scan the underlying string left to right and
    /// mark each corresponding suffix as either `L` or `S(false)`. Then we do a second right to left
    /// scan and mark all LMS suffixes as `S(true)`. We also keep track of the locations of the lms
    /// suffixes
    fn create_suffixes(underlying: &'a str) -> (Vec<Suffix>, Vec<SuffixIndex>) {
        let s_len = underlying.len();
        let mut tags = vec![SuffixType::S(false); s_len];
        let mut suffixes = Vec::with_capacity(s_len);
        let mut lms_locations = Vec::with_capacity(s_len / 2);
        let s_ascii = underlying.as_bytes();

        // We tag each chatacter as either `S` type or `L` type. Since
        // we initialized everything as `S` type, we only need to mark
        // the `L` type suffixes
        for i in (0..s_len - 1).rev() {
            let (cur, next) = (s_ascii[i], s_ascii[i + 1]);
            if (cur > next) || (cur == next && tags[i + 1] == SuffixType::L) {
                tags[i] = SuffixType::L;
            }
        }
        // The first character can never be an `lms` suffix, so we skip it
        // Similary, the last character, which is the sentinel `$` is definitely
        // an lms suffix, so we deal with it outside the loop
        for i in 1..s_len - 1 {
            match (tags[i - 1].clone(), &mut tags[i]) {
                // If the left character is the start of an `L-type` suffix
                // and the current character is an `S-type` suffix, then
                // we mark the current suffix as an `lms` suffix. We
                // ignore all other cases
                (SuffixType::L, SuffixType::S(is_lms)) => {
                    *is_lms = true;
                    lms_locations.push(SuffixIndex(i));
                }
                _ => {}
            }
        }
        tags[s_len - 1] = SuffixType::S(true);
        lms_locations.push(SuffixIndex(s_len - 1));

        // Now that we have all the suffix tags in place, we can construct
        // the list of suffixes in the string
        for (i, suffix_type) in tags.into_iter().enumerate() {
            suffixes.push((SuffixIndex(i), suffix_type, underlying).into())
        }
        (suffixes, lms_locations)
    }
}

pub struct AlphabetCounter([usize; 1 << 8]);

impl AlphabetCounter {
    /// Initialize the counter from a string slice. This procedure expects
    /// the give slice to only have ascii characters
    pub fn from_ascii_str(s: &str) -> Self {
        let mut alphabet = [0_usize; 1 << 8];
        for byte in s.as_bytes() {
            alphabet[*byte as usize] += 1;
        }
        Self(alphabet)
    }

    pub fn create_buckets<'a>(&self, array: &'a [SuffixIndex]) -> HashMap<BucketId<u8>, Bucket> {
        let mut buckets = HashMap::new();
        let mut start_location = 0;
        let alphabet_counter = self.0;
        for i in 0..alphabet_counter.len() {
            if alphabet_counter[i] > 0 {
                let end_location = start_location + alphabet_counter[i] - 1;
                let bucket = Bucket {
                    start: SuffixArrayIndex(start_location),
                    end: SuffixArrayIndex(end_location),
                    offset_from_end: 0,
                    offset_from_start: 0,
                    lms_offset_from_end: 0,
                };
                buckets.insert(BucketId(i as u8), bucket);
                start_location = end_location + 1;
            }
        }
        buckets
    }
}

// #[test]
// fn test_alphabet() {
//     let s = "bananabandana$";
//     let alph = AlphabetCounter::from_ascii_str(s);
//     let bkt = alph.create_buckets();
//     for (id, bucket) in bkt {
//         println!(
//             "{} from {} to {}",
//             id.0 as char, bucket.start.0, bucket.end.0
//         );
//     }
// }

#[derive(Debug)]
pub struct RMQResult<'a, T> {
    min_idx: usize,
    min_value: &'a T,
}

impl<'a, T> From<(usize, &'a T)> for RMQResult<'a, T> {
    fn from((min_idx, min_value): (usize, &'a T)) -> Self {
        RMQResult { min_idx, min_value }
    }
}

#[derive(Debug, Eq, PartialEq, Hash)]
pub struct SparseTableIdx {
    /// The index where the range in question begins
    start_idx: usize,

    /// The length of the range. This has to always be a power of
    /// two
    len: usize,
}

type DenseTable<'a, T> = HashMap<RMQRange<'a, T>, RMQResult<'a, T>>;

/// All structures capable of answering range min queries should
/// expose the solve method.
pub trait RMQSolver<'a, T: Ord> {
    fn solve(&self, range: &RMQRange<'a, T>) -> RMQResult<T>;
}

/// A solver that answers range min queries by  doing no preprocessing. At query time, it
/// simply does a linear scan of the range in question to get the answer. This is an
/// <O(1), O(n)> solver
#[derive(Debug)]
pub struct ScanningSolver<'a, T> {
    underlying: &'a [T],
}

impl<'a, T> ScanningSolver<'a, T> {
    pub fn new(underlying: &'a [T]) -> Self {
        ScanningSolver { underlying }
    }
}

/// A solver that answers `rmq` queries by first pre-computing
/// the answers to all possible ranges. At query time, it simply
/// makes a table lookup. This is the <O(n*n), O(1)> solver
#[derive(Debug)]
pub struct DenseTableSolver<'a, T> {
    underlying: &'a [T],
    lookup_table: DenseTable<'a, T>,
}

impl<'a, T: Ord + Hash> DenseTableSolver<'a, T> {
    pub fn new(underlying: &'a [T]) -> Self {
        let lookup_table = compute_rmq_all_ranges(underlying);
        DenseTableSolver {
            underlying,
            lookup_table,
        }
    }
}

/// A solver that answers rmq queries by first precomputing
/// the answers to ranges whose length is a power of 2
/// At query time, it uses a lookup table of `msb(n)` values to
/// factor the length of the requested query into powers of
/// 2 and then looks up the answers in the sparse table.
/// This is the <O(n lg n), O(1)> solver
#[derive(Debug)]
pub struct SparseTableSolver<'a, T> {
    underlying: &'a [T],
    sparse_table: SparseTable<'a, T>,

    /// The precomputed and cached array of msb(n)
    /// answers for all n between 1 and 1 << 16
    msb_sixteen_lookup: MSBLookupTable,
}

impl<'a, T: Ord + Hash> SparseTableSolver<'a, T> {
    pub fn new(underlying: &'a [T]) -> Self {
        let sparse_table = compute_rmq_sparse(underlying);
        let msb_sixteen_lookup = MSBLookupTable::build();
        SparseTableSolver {
            underlying,
            sparse_table,
            msb_sixteen_lookup,
        }
    }
}

fn get_min_by_scanning<T: Ord>(block: &[T]) -> RMQResult<T> {
    let (min_idx, min_value) = block.iter().enumerate().min_by_key(|x| x.1).unwrap();
    RMQResult { min_idx, min_value }
}

#[test]
fn test_min_by_scanning() {
    let arr = [28, 6, 38, 5, 8, 6, 78, 2, 1];
    let res = get_min_by_scanning(&arr);
    assert_eq!((8, &1), (res.min_idx, res.min_value));
}

impl<'a, T: Ord> RMQSolver<'a, T> for ScanningSolver<'a, T> {
    fn solve(&self, range: &RMQRange<'a, T>) -> RMQResult<T> {
        let range_slice = &self.underlying[range.start_idx..=range.end_idx];
        get_min_by_scanning(range_slice)
    }
}

fn compute_rmq_all_ranges<'a, T: Hash + Eq + Ord>(array: &'a [T]) -> DenseTable<'a, T> {
    let len = array.len();
    let mut lookup_table = HashMap::with_capacity((len * len) / 2);
    for start in 0..len {
        for end in start..len {
            if start == end {
                lookup_table.insert((start, end, array).into(), (start, &array[start]).into());
            } else {
                let prev_range = (start, end - 1, array).into();
                let new_range = (start, end, array).into();
                let prev_min_res: &RMQResult<T> = lookup_table.get(&prev_range).unwrap();
                let cur_min_res: RMQResult<T>;
                if prev_min_res.min_value <= &array[end] {
                    cur_min_res = (prev_min_res.min_idx, prev_min_res.min_value.clone()).into();
                } else {
                    cur_min_res = (end, &array[end]).into();
                }
                lookup_table.insert(new_range, cur_min_res);
            }
        }
    }
    lookup_table
}

impl<'a, T: Ord + Eq + Hash> RMQSolver<'a, T> for DenseTableSolver<'a, T> {
    fn solve(&self, range: &RMQRange<'a, T>) -> RMQResult<T> {
        let res = self.lookup_table.get(&range).unwrap();
        (res.min_idx, res.min_value.clone()).into()
    }
}

fn get_prev_min<'a, T: Hash + Eq + Ord>(
    array: &'a [T],
    left_res: &RMQResult<T>,
    right_res: &RMQResult<T>,
) -> RMQResult<'a, T> {
    if left_res.min_value < right_res.min_value {
        (left_res.min_idx, &array[left_res.min_idx]).into()
    } else {
        (right_res.min_idx, &array[right_res.min_idx]).into()
    }
}

fn compute_rmq_sparse<'a, T: Hash + Eq + Ord>(array: &'a [T]) -> SparseTable<'a, T> {
    let len = array.len();
    let mut sparse_table = HashMap::new();
    let mut power = 0;
    while 1 << power <= len {
        let mut start_idx = 0;
        let mut end_idx = start_idx + (1 << power) - 1;
        while end_idx < len {
            if start_idx == end_idx {
                let idx = (start_idx, 1 << power).into();
                let rmq_res = (start_idx, &array[start_idx]).into();
                sparse_table.insert(idx, rmq_res);
            } else {
                let idx = (start_idx, 1 << power).into();
                let prev_len = 1 << (power - 1);
                let left: SparseTableIdx = (start_idx, prev_len).into();
                let right: SparseTableIdx = (start_idx + prev_len, prev_len).into();
                let left_res = sparse_table.get(&left).unwrap();
                let right_res = sparse_table.get(&right).unwrap();
                let rmq_res = get_prev_min(array, left_res, right_res);
                sparse_table.insert(idx, rmq_res);
            }
            println!("({}, {}, {})", start_idx, end_idx, 1 << power);
            start_idx += 1;
            end_idx += 1;
        }
        power += 1;
    }
    sparse_table
}

#[test]
fn compute_sparse_ftd() {
    let array = [1, 2, 3, 4, 5, 6, 7, 8];
    let sparse_table = compute_rmq_sparse(&array);
    assert_eq!(sparse_table.len(), 21);
    for (k, v) in sparse_table {
        println!(
            "(start {}, end {}, len {} min {})",
            k.start_idx,
            k.start_idx + k.len,
            k.len,
            v.min_value
        );
    }
}

impl<'a, T: Ord + Eq + Hash> RMQSolver<'a, T> for SparseTableSolver<'a, T> {
    fn solve(&self, range: &RMQRange<'a, T>) -> RMQResult<T> {
        let (i, j) = (range.start_idx, range.end_idx);
        let range_len = (j - i) + 1;
        let k = self.msb_sixteen_lookup.get_msb_idx_of(range_len);
        let right_start = j - (1 << k) + 1;
        let left: SparseTableIdx = (i, 1 << k).into();
        let right: SparseTableIdx = (right_start, 1 << k).into();
        let left_res = self.sparse_table.get(&left).unwrap();
        let right_res = self.sparse_table.get(&right).unwrap();
        get_prev_min(self.underlying, left_res, right_res)
    }
}

/// The primary solvers available.
#[derive(Debug)]
pub enum RMQSolverKind {
    ScanningSolver,
    DenseTableSolver,
    SparseTableSolver,
}

struct Block<'a, T> {
    start_idx: usize,
    end_idx: usize,
    values: &'a [T],
}
type BlockLevelSolvers<'a, T> = HashMap<RMQBlock<'a, T>, Box<dyn RMQSolver<'a, T>>>;
/// Since we unified our various solve, we can succinctly represent
/// a solver that follows the method of four russians scheme.
/// Notice how we allow one to set the block_size, and solvers
pub struct FourRussiansRMQ<'a, T: Ord> {
    /// This is the entire array. The solvers only operate on slices of
    /// this array
    static_array: &'a [T],

    /// As discussed already, block decomposition is at the
    /// heart of the method of four russians. This
    /// fields keeps track of all the blocks
    blocks: Vec<RMQBlock<'a, T>>,

    /// The size of each block. The last block may be smaller
    /// than this
    block_size: usize,

    /// We call the solve method of this object when we want to
    /// answer an `rmq` query over the macro array
    macro_level_solver: Box<dyn RMQSolver<'a, RMQBlock<'a, T>> + 'a>,

    /// We call the solve method of this object when we want to
    /// answer an `rmq` query over a single block (ie a micro array)
    block_level_solvers: BlockLevelSolvers<'a, T>,
}

#[derive(Debug, Default)]
pub struct FourRussiansRMQBuilder<'a, T: Ord + Default> {
    static_array: Option<&'a [T]>,
    block_size: Option<usize>,
    macro_solver: Option<RMQSolverKind>,
    micro_solver: Option<RMQSolverKind>,
}

impl<'a, T: Ord + Default> FourRussiansRMQBuilder<'a, T> {
    pub fn new() -> Self {
        FourRussiansRMQBuilder::default()
    }

    pub fn with_static_array(mut self, array: &'a [T]) -> Self {
        self.static_array = Some(array);
        self
    }

    pub fn with_block_size(mut self, b: usize) -> Self {
        self.block_size = Some(b);
        self
    }

    pub fn with_macro_solver(mut self, macro_solver: RMQSolverKind) -> Self {
        self.macro_solver = Some(macro_solver);
        self
    }

    pub fn with_micro_solver(mut self, micro_solver: RMQSolverKind) -> Self {
        self.micro_solver = Some(micro_solver);
        self
    }

    pub fn build(mut self) -> FourRussiansRMQ<'a, T> {
        todo!()
    }
}

impl<'a, T: Ord> PartialOrd for RMQBlock<'a, T> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.min.cmp(other.min))
    }
}

impl<'a, T: Ord> Ord for RMQBlock<'a, T> {
    fn cmp(&self, other: &Self) -> Ordering {
        self.min.cmp(&other.min)
    }
}

impl<'a, T: Ord> PartialEq for RMQBlock<'a, T> {
    fn eq(&self, other: &Self) -> bool {
        self.min == other.min
    }
}

impl<'a, T: Ord + Hash> FourRussiansRMQ<'a, T> {
    /// Create a new RMQ solver for `static_array` that uses block decomposition
    /// with a block size of `b`. The solver will use the `macro_solver` to
    /// solve the instance of the problem on the array of aggregate solutions from
    /// the blocks and `micro_solver` to solve the solution in each individual block
    pub fn new(
        static_array: &'a [T],
        block_size: usize,
        macro_solver: RMQSolverKind,
        micro_solver: RMQSolverKind,
    ) -> Self {
        let blocks = {
            let mut blocks = Vec::<RMQBlock<T>>::with_capacity(static_array.len() / block_size);
            for start_idx in (0..static_array.len()).step_by(block_size) {
                let end_idx = (start_idx + block_size) - 1;
                let block = &static_array[start_idx..=end_idx];
                let rmq_res = get_min_by_scanning(block);
                blocks.push((start_idx, end_idx, rmq_res).into())
            }
            blocks
        };
        let macro_level_solver: Box<dyn RMQSolver<'_, RMQBlock<'_, T>>> = match macro_solver {
            RMQSolverKind::ScanningSolver => Box::new(ScanningSolver::new(&blocks)),
            RMQSolverKind::DenseTableSolver => Box::new(DenseTableSolver::new(&blocks)),
            RMQSolverKind::SparseTableSolver => Box::new(SparseTableSolver::new(&blocks)),
        };
        let block_level_solvers = Self::create_micro_solvers(static_array, block_size, micro_solver);
        // FourRussiansRMQ {
        //     blocks,
        //     block_size,
        //     static_array,
        //     macro_level_solver,
        //     block_level_solvers,
        // }
        todo!()
    }

    fn create_micro_solvers(array: &[T], b: usize, kinds: RMQSolverKind) -> BlockLevelSolvers<'_, T> {
        todo!()
    }

    /// Find the smallest element in the range provided (i, j). This works by finding the
    /// minimum among three answers:
    ///     (a) The smallest value in the valid portion of i's block
    ///     (b) The smallest value in the valid portion of j's block
    ///     (c) The smallest value in the intermediate blocks
    pub fn rmq(&self, range: RMQRange<'a, T>) -> RMQResult<'a, T> {
        let start_block = range.start_idx / self.block_size;
        let end_block = range.end_idx / self.block_size;
        // Retrieve the solvers for the start block and the end block from
        //      the HashMap of block level solvers -- Should we have BlockId?
        // Construct the ranges [i..start_block_end], [end_block_start..j]
        //      and get the RMQResult from the solvers
        // Construct a range for the intermediate blocks
        //  and query for the RMQResult in the intermediate solver
        //Return the smallest of these three

        todo!()
    }
}

impl Bucket {
    /// Put the provided s_type suffix into its rightful location.
    /// In a bucket, S-type suffixes appear after all L-Type suffixes
    /// because they are lexicographically larger. Furthermore, in a given bucket,
    /// `lms` suffixes are larger than all other suffixes.
    fn insert_stype_suffix(&mut self, suffix: &Suffix, sa: &mut Vec<SuffixIndex>) {
        sa[self.end.0 - self.offset_from_end] = suffix.start.clone();
        self.offset_from_end += 1;
    }

    /// Put the provided lms_suffix at its  correct position within
    /// a bucket
    fn insert_lms_suffix(&mut self, suffix: &Suffix, sa: &mut Vec<SuffixIndex>) {
        sa[self.end.0 - self.lms_offset_from_end] = suffix.start.clone();
        self.lms_offset_from_end += 1;
    }

    /// Put the provided l_type suffix in its approximate location in this
    /// bucket
    fn insert_ltype_suffix(&mut self, suffix: &Suffix, sa: &mut Vec<SuffixIndex>) {
        sa[self.start.0 + self.offset_from_start] = suffix.start.clone();
        self.offset_from_start += 1;
    }
}

type Buckets<'a> = HashMap<BucketId<u8>, Bucket>;

impl<'a> SuffixArray<'a> {
    // TODO: This is wrong -- reimplement
    fn induced_lms_sort(s: &'a str, buckets: &mut Buckets, sa: &mut Vec<SuffixIndex>) {
        let (suffixes, lms_locations) = Self::create_suffixes(s);
        // Place LMS suffixes in position.
        for lms_idx in lms_locations {
            let cur_lms_suffix: Suffix = (lms_idx, SuffixType::S(true), s).into();
            let id = cur_lms_suffix.get_bucket();
            buckets.get_mut(&id).unwrap().insert_lms_suffix(&cur_lms_suffix, sa);
        }

        // 2. Place L-type suffixes in position
        for suffix in &suffixes {
            if suffix.suffix_type == SuffixType::L {
                let id = suffix.get_bucket();
                buckets.get_mut(&id).unwrap().insert_ltype_suffix(&suffix, sa);
            }
        }

        // // 3. Place S-type suffixes in position. This operation
        // //    may change the location of the `lms` suffixes that
        // //    (1) ordered.
        // for suffix in suffixes.iter().rev() {
        //     if suffix.suffix_type != SuffixType::L {
        //         let id = suffix.get_bucket();
        //         buckets.get_mut(&id).unwrap().insert_stype_suffix(&suffix, sa);
        //     }
        // }
    }

    fn make_reduced_str(sa_: &Vec<SuffixIndex>) {
        todo!()
    }

    fn induce_sa_from_reduced_sa() {
        todo!()
    }

    fn make_reduced_sa() {
        todo!()
    }

    pub fn make_sa_by_sais(underlying: &'a str, sigma: AlphabetCounter) -> Self {
        // We initialize all slots in our suffix array with an invalid suffix index.
        let mut suffix_array = vec![SuffixIndex(underlying.len()); underlying.len()];
        let mut buckets = sigma.create_buckets(&suffix_array);
        Self::induced_lms_sort(underlying, &mut buckets, &mut suffix_array);

        // 4. Substring Renaming: Create a summary string using the buckets of the LMS
        //    suffixes
        // 5. If the substring contains unique characters -- create suffix array directly.
        //    otherwise, recursively create the s suffix array for the reduced string.
        // 6. Finally, induced sorting again.
        SuffixArray {
            suffix_array,
            underlying,
        }
    }
}

#[test]
fn test_sais() {
    let s = "mmiissiissiippii$";
    let sigma = AlphabetCounter::from_ascii_str(s);
    let sa = SuffixArray::make_sa_by_sais(s, sigma);
    println!("{}", sa)
}

/// An index into a collection of cartesian tree nodes
#[derive(Debug, Ord, PartialOrd, Eq, PartialEq, Clone)]
struct CartesianNodeIdx(usize);

#[derive(Debug)]
struct CartesianTreeNode<'a, T: Ord> {
    /// A reference to the array value that this node represents
    value: &'a T,

    /// The locations of the children and parent of this node.
    left_child_idx: Option<CartesianNodeIdx>,
    right_child_idx: Option<CartesianNodeIdx>,
}

impl<'a, T: Ord> std::ops::Index<CartesianNodeIdx> for Vec<CartesianTreeNode<'a, T>> {
    type Output = CartesianTreeNode<'a, T>;
    fn index(&self, index: CartesianNodeIdx) -> &Self::Output {
        &self[index.0]
    }
}

impl<'a, T: Ord> std::ops::IndexMut<CartesianNodeIdx> for Vec<CartesianTreeNode<'a, T>> {
    fn index_mut(&mut self, index: CartesianNodeIdx) -> &mut Self::Output {
        &mut self[index.0]
    }
}
/// A cartesian tree is a heap ordered binary tree
/// derived from some underlying array. An in-order
/// traversal of the tree yields the underlying tree.
#[derive(Debug)]
struct CartesianTree<'a, T: Ord> {
    nodes: Vec<CartesianTreeNode<'a, T>>,
    root_idx: Option<CartesianNodeIdx>,
    action_profile: Vec<CartesianTreeAction>,
}

/// When constructing a cartesian tree, we either
/// push a node to or pop a node from a stack.
/// We keep track of these actions because we can
/// use them to generate the cartesian tree number.
#[derive(Debug, Eq, PartialEq)]
enum CartesianTreeAction {
    Push,
    Pop,
}

impl<'a, T: Ord> From<&'a T> for CartesianTreeNode<'a, T> {
    fn from(value: &'a T) -> Self {
        CartesianTreeNode {
            value,
            left_child_idx: None,
            right_child_idx: None,
        }
    }
}

// To create the cartesian tree, we pop the stack until either
// it's empty or the element atop the stack has a smaller value
// than the element we are currently trying to add to the stack.
// Once we break out of the `pop` loop, we make the item we popped
// a left child of the new item we are adding. Additionally, we make
// this new item a right/left child of the item atop the stack
impl<'a, T: Ord> From<&'a [T]> for CartesianTree<'a, T> {
    fn from(underlying: &'a [T]) -> Self {
        let len = underlying.len();
        let mut nodes = Vec::with_capacity(len);
        let mut stack = Vec::<CartesianNodeIdx>::with_capacity(len);
        let mut action_profile = Vec::with_capacity(len * 2);
        for (idx, value) in underlying.iter().enumerate() {
            nodes.push(value.into());
            let node_idx = CartesianNodeIdx(idx);
            add_node_to_cartesian_tree(&mut nodes, &mut stack, &mut action_profile, node_idx);
        }
        let root_idx = stack.first().map(|min| min.clone());
        CartesianTree {
            nodes,
            root_idx,
            action_profile,
        }
    }
}

type Nodes<'a, T> = Vec<CartesianTreeNode<'a, T>>;
type Stack = Vec<CartesianNodeIdx>;
type Actions = Vec<CartesianTreeAction>;
/// Adds the node at the given idx into the tree by wiring up the
/// child and parent pointers. it is assumed that the
/// node has already been added to `nodes` the list of nodes.
/// This procedure returns an optional index value
/// that is populated if the root changed.
fn add_node_to_cartesian_tree<T: Ord>(
    nodes: &mut Nodes<T>,
    stack: &mut Stack,
    actions: &mut Actions,
    new_idx: CartesianNodeIdx,
) {
    let mut last_popped = None;
    loop {
        match stack.last() {
            None => break,
            Some(top_node_idx) => {
                // If the new node is greater than the value atop the stack,
                // we make the new node a right child of that value
                if nodes[top_node_idx.clone()].value < nodes[new_idx.clone()].value {
                    nodes[top_node_idx.clone()].right_child_idx = Some(new_idx.clone());
                    break;
                }
                last_popped = stack.pop();
                actions.push(CartesianTreeAction::Pop);
            }
        }
    }
    // We make the last item we popped a left child of the
    // new node
    if let Some(last_popped_idx) = last_popped {
        nodes[new_idx.clone()].left_child_idx = Some(last_popped_idx);
    }
    stack.push(new_idx);
    actions.push(CartesianTreeAction::Push);
}

impl<'a, T: Ord> CartesianTree<'a, T> {
    fn in_order_traversal(&self) -> Vec<&T> {
        let mut res = Vec::with_capacity(self.nodes.len());
        self.traversal_helper(&self.root_idx, &mut res);
        res
    }

    fn traversal_helper(&self, cur_idx: &Option<CartesianNodeIdx>, res: &mut Vec<&'a T>) {
        let nodes = &self.nodes;
        match cur_idx {
            None => {}
            Some(cur_sub_root) => {
                self.traversal_helper(&nodes[cur_sub_root.clone()].left_child_idx, res);
                res.push(&nodes[cur_sub_root.clone()].value);
                self.traversal_helper(&nodes[cur_sub_root.clone()].right_child_idx, res);
            }
        }
    }
}

#[test]
fn test_cartesian_tree() {
    let v = [93, 84, 33, 64, 62, 83, 63];
    let tree: CartesianTree<'_, _> = v.as_ref().into();
    assert!(tree.root_idx.is_some());
    assert_eq!(tree.nodes[tree.root_idx.clone().unwrap()].value, &33);
    for (&l, &r) in tree.in_order_traversal().into_iter().zip(v.iter()) {
        assert_eq!(l, r);
    }
}

impl<'a, T: Ord> CartesianTree<'a, T> {
    /// Calculates the cartesian tree number of this tree
    /// using the sequence of `push` and `pop` operations
    /// stored in the `action_profile`. Note that calculating this
    /// value only makes sense when the underlying array is small.
    /// More specifically, this procedure assumes that the underlying
    /// array has at most 32 items. This makes sense in our context
    /// since we're mostly interested in the cartesian tree numbers
    /// of RMQ blocks
    fn cartesian_tree_number(&self) -> u64 {
        let mut number = 0;
        let mut offset = 0;
        for action in &self.action_profile {
            if action == &CartesianTreeAction::Push {
                number |= 1 << offset;
            }
            offset += 1;
        }
        number
    }
}
