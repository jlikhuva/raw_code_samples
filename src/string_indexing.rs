//! Implementation of string indexing algorithms and data structures
//! ## Contents
//! 1. A patricia trie for fast prefix matching
//! 2. A suffix tree for fast substring matching. The suffix tree is constructed using
//!    Ukonnen's procedure
//! 3. A suffix array constructed using the SA-IS procedure
//! 4. An LCP array constructed using Kasai's procedure

/// A suffix of some string is uniquely defined by
/// its starting index. We use this to index into
/// the string over which we are building our suffix
/// and lcp array
#[derive(Debug, PartialEq, Eq, Hash, Clone)]
pub struct SuffixId(usize);

impl std::fmt::Display for SuffixId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

mod suffix_array {
    use super::SuffixId;
    use std::fmt;

    #[derive(Debug)]
    pub struct Alphabet<const ALPHABET_SIZE: usize> {
        characters: [char; ALPHABET_SIZE],
    }

    /// We use this too index into the siffix
    /// array
    #[derive(Debug)]
    struct SAIdx(usize);

    /// A bucket in the suffix array is a contiguous
    /// slice in whcih all suffixes begin
    /// with the same character
    #[derive(Debug)]
    struct BucketIdx {
        /// This is the starting index of this bucket
        head: SAIdx,

        /// The ending index of this bucket. Do note that
        /// a bucket is range inclusive i.e [head, tail]
        tail: SAIdx,

        /// The label of a bucket is the first character of the
        /// suffixes in this bucket
        label: char,
    }

    #[derive(Debug, Eq, PartialEq, Clone)]
    enum SuffixType {
        /// A suffix starting at some position `k` in some text `T` is an
        /// S-type suffix if:
        ///
        /// ```latex
        /// T[k] < T[k + 1]
        ///     OR
        /// T[k] == T[K + 1] AND k + 1 is an S-type suffix
        ///     OR
        /// T[k] = $, the sentinel character
        /// ```
        S,

        /// A suffix starting at some position `k` in some text `T` is an
        /// L-type suffix if:
        ///
        /// ```latex
        /// T[k] > T[k + 1]
        ///     OR
        /// T[k] == T[K + 1] AND k + 1 is an L-type suffix
        /// ```
        L,
    }

    impl fmt::Display for SuffixType {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            match self {
                SuffixType::L => write!(f, "L"),
                SuffixType::S => write!(f, "S"),
            }
        }
    }
    #[derive(Debug)]
    pub struct SuffixArray<'a> {
        /// The string over which we are building this suffix array
        underlying: &'a str,

        /// The suffix array is simply all the suffixes of the
        /// underlying string in sorted order
        suffix_array: Vec<SuffixId>,
    }
    impl<'a> fmt::Display for SuffixArray<'a> {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            let mut res = String::new();
            res += "  id  |  suffix   \n";
            res += "------|-------------------\n";
            for id in &self.suffix_array {
                if id.0 <= 9 {
                    res += &format!(" {}    |{}", id, self.underlying[id.0..].to_string());
                } else {
                    res += &format!(" {}   |{}", id, self.underlying[id.0..].to_string());
                }
                res += "\n";
            }
            writeln!(f, "{}", res)
        }
    }
    impl<'a> SuffixArray<'a> {
        pub fn get(&self, i: usize) -> SuffixId {
            self.suffix_array[i].clone()
        }

        /// Construct the suffix array by sorting. This has worst case performance
        /// of O(n log n)
        pub fn make_sa_naive(s: &'a str) -> Self {
            let mut suffixes = vec![];
            for i in 0..s.len() {
                suffixes.push(s[i..].to_string());
            }
            suffixes.sort();
            let mut suffix_array = vec![];
            for suffix in suffixes {
                let cur = SuffixId(s.len() - suffix.len());
                suffix_array.push(cur);
            }
            Self {
                underlying: s,
                suffix_array,
            }
        }

        pub fn make_sa_sais(s: &'a str) -> Self {
            // Create abstraction for LMS Substring.
            // Explore using const generics in codeg
            // 1. Tag each character in s as either `L` type or `S` type
            //    The method returns an array of Vec<SuffixType>
            // 2. Create a new array of LMS suffixes -- preserving positional
            //    order
            // 3. Create an array of buckets: Vec<BucketId>
            // 4. Put the LMS suffixes in their approximate locations
            // 5. Use two passes (R-L, L-R) to put all the other suffixes in place
            // 6. Create a reduced substring (This is the substring renaming step)
            // 7. If the renamed string contains only unique characters: create SA of
            //    reduced string directly. otherwise, recurse
            // 8. Use induced sorting to create suffix array from the suffix array of
            //    the reduced string
            let s_tags = Self::tag_sequence(s);
            let lms_suffixes = Self::extract_lms_suffixes(s_tags);
            todo!()
        }

        fn tag_sequence(s: &'a str) -> Vec<SuffixType> {
            let s_len = s.len();
            let mut tags = vec![SuffixType::S; s_len];
            let s_ascii = s.as_bytes();
            for i in (1..s_len - 1).rev() {
                let (cur, next) = (s_ascii[i], s_ascii[i + 1]);
                if (cur > next) || (cur == next && tags[i + 1] == SuffixType::L) {
                    tags[i] = SuffixType::L;
                }
            }
            tags
        }

        fn extract_lms_suffixes(s_tags: Vec<SuffixType>) {
            todo!()
        }
    }
}

mod lcp {
    use super::suffix_array::SuffixArray;
    use super::SuffixId;
    use std::{collections::HashMap, fmt};

    #[derive(Debug)]
    pub struct Height(usize);

    impl fmt::Display for Height {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            write!(f, "{}", self.0)
        }
    }

    #[derive(Debug, PartialEq, Eq, PartialOrd, Ord)]
    struct Rank(usize);

    #[derive(Debug)]
    pub struct LCPHeightArray<'a> {
        /// The string over which we are building this suffix array
        underlying: &'a str,

        /// The suffix array of the underlying string
        suffix_array: &'a SuffixArray<'a>,

        /// The lcp height array is an array giving us the length
        /// of the longest common prefix over all pairs of adjacent
        /// suffixes in the suffix array
        lcp_array: Vec<Height>,
    }

    impl<'a> fmt::Display for LCPHeightArray<'a> {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            let mut res = String::new();
            res += "  id  |  suffix   \n";
            res += "------|-------------------\n";
            for h in &self.lcp_array {
                res += &format!(" {}   ", h);
                res += "\n";
            }
            writeln!(f, "{}", res)
        }
    }

    impl<'a> LCPHeightArray<'a> {
        /// Creates a new lcp array from the provided string and suffix array
        pub fn new(s: &'a str, sa: &'a SuffixArray) -> Self {
            let lcp_array = Self::make_lcp(s, sa);
            Self {
                underlying: s,
                suffix_array: sa,
                lcp_array,
            }
        }

        /// Computes the lcp array in O(n) using Kasai's algorithm.
        /// The Crux of this procedure is the observation that when LCP(A[i-1 ..], A[Rank[sa[i-1]] - 1] ...) = h,
        /// that is the lcp between the suffix that begins at i-1 and the suffix to its left in the suffix array, sa,  is h,
        /// We know that the LCP between the suffix that starts at i and the suffix adjacent to it is at least h - 1. Therefore,
        /// to calculate the actual length efficiently, it suffices to only compare characters starting at h instead of 0
        fn make_lcp(s: &str, sa: &SuffixArray) -> Vec<Height> {
            let s_ascii = s.as_bytes();
            // We begin by computing the rank array. This is an array indexed
            // by the suffix ids. rank[id] tells the index we can find the suffix A[id ..]
            // in the suffix array. That is sa[rank[id]] = id
            let mut rank = HashMap::with_capacity(s.len());
            let mut lcp_array = Vec::with_capacity(s.len());
            for i in 0..s.len() {
                rank.insert(sa.get(i), Rank(i));
                lcp_array.push(Height(0));
            }
            let mut h = 0;
            // We then loop over all the suffixes one by one. One thing to note that we
            // are looping over the suffixes in the order they occur in the underlying
            // string. This order is different from the order in which they occur in
            // the suffix array. This means that we fill up the lcp_array in "random"
            // order
            for i in 0..s.len() {
                let id = SuffixId(i);

                // We are currently processing the suffix that starts at index `i` in
                // the underlying string. We'd like to know where this suffix
                // is located in the lexicographically sorted suffix array
                let location_in_sa = rank.get(&id).unwrap();

                // We skip over the sentinel character.Since it's smallest character in the string
                // by definition, it has rank = 0. This means that it has no suffix adjacent to
                // it from the left
                if location_in_sa > &Rank(0) {
                    // grab a hold of the id of the suffix that is just before `location_in_sa`
                    // in the suffix array
                    let left_adjacent_suffix = sa.get(location_in_sa.0 - 1);

                    // Here, we compute the length of the longest common prefix between
                    // the current suffix and the suffix that is left of it in the suffix
                    // array
                    while s_ascii[id.0 + h] == s_ascii[left_adjacent_suffix.0 + h] {
                        h += 1
                    }
                    lcp_array[location_in_sa.0] = Height(h);

                    // When we move from i to i+1, we are effectively moving from processing the suffix
                    // A[i..] to processing the suffix A[i+1..]. Notice how this is the same as moving
                    // to processing a suffix formed by dropping the first character of A[i..]. Therefore,
                    // Theorem 1 from Kasai et al. tells us that the lcp between the new shorter suffix
                    // and the suffix adjacent to its left in the suffix array is at least `h-1`
                    if h > 0 {
                        h -= 1
                    }
                }
            }
            lcp_array
        }
    }
}

#[cfg(test)]
mod test {
    use super::{lcp::LCPHeightArray, suffix_array::SuffixArray};

    #[test]
    fn lcp() {
        let test_str = "bananabandana$";
        let sa = SuffixArray::make_sa_naive(&test_str);
        println!("{}", sa);
        let lcp = LCPHeightArray::new(&test_str, &sa);
        println!("{}", lcp)
    }

    #[test]
    fn tag_sequence() {
        unimplemented!();
    }
}
