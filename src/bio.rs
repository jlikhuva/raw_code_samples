//! How do we find the origin of replication?
pub mod ori {
    use std::{collections::BTreeMap, usize};

    use crate::rmq_notes::{make_lcp_by_kasai, SuffixArray};
    /// Counts and returns the number of times `pattern`
    /// appears in `str`, The runtime of this procedure is
    /// O(M * N) where `M = s.len()` and `N = pattern.len()`
    pub fn pattern_count(s: &str, pattern: &str) -> usize {
        let mut count = 0;
        for window in s.as_bytes().windows(pattern.len()) {
            if pattern_matches(pattern, window) {
                count += 1;
            }
        }
        count
    }

    /// A linear time procedure to check if `pattern` and `window`
    /// have the same characters in the same order.
    fn pattern_matches(pattern: &str, window: &[u8]) -> bool {
        for (a, b) in pattern.as_bytes().iter().zip(window) {
            if a != b {
                return false;
            }
        }
        true
    }

    /// Returns a count of each k-mer in the given string. This procedure calculates
    /// all counts in linear time
    pub fn count_of_kmers_in(s: &str, k: usize) -> BTreeMap<&str, usize> {
        let mut counter = BTreeMap::new();
        for cur_kmer in s.as_bytes().windows(k) {
            let as_str = std::str::from_utf8(cur_kmer).unwrap();
            let cur_count = counter.entry(as_str).or_insert(0);
            *cur_count += 1;
        }
        counter
    }

    /// Given a map of counts, returns a list of the most frequent ones
    pub fn find_most_frequent_kmers<'a, 'b>(
        kmers_counter: &'a BTreeMap<&'b str, usize>,
    ) -> Vec<(&'a &'b str, &'a usize)> {
        let &max = kmers_counter.values().max().unwrap();
        kmers_counter.iter().filter(|(_, &v)| v == max).collect()
    }

    pub fn reverse_complement(dna_strand: &str, complement_mapper: impl Fn(char) -> char) -> String {
        dna_strand.chars().map(complement_mapper).rev().collect::<String>()
    }

    pub fn deoxyribose_complement(c: char) -> char {
        match c {
            'A' => 'T',
            'a' => 't',
            'T' => 'A',
            't' => 'a',
            'G' => 'C',
            'g' => 'c',
            'C' => 'G',
            'c' => 'g',
            _ => panic!("{} is an invalid dna nucleotide character", c),
        }
    }

    /// Finds all index locations where `patten` occurs in `s`. This is
    /// an O(|s| lg |s|) procedure.
    pub fn find_all_occurrences(s: &str, pattern: &str) -> Vec<usize> {
        let mut locations = Vec::new();
        let sa = SuffixArray::make_sa_naive(s);
        let lcp = make_lcp_by_kasai(s, &sa);

        // 2. Do a binary search to find an index within the matching region -- ln g
        // todo!();
        locations
    }

    /// Finds all the distinct kmers that form (interval_size, t) clumps in genome.
    /// That is, it finds all kmers that are repeated more than `t` times in any
    /// interval of `interval_size` in `genome`.
    pub fn find_clumping_kmers(genome: &str, interval_size: usize, t: usize, k: usize) -> Vec<&str> {
        todo!()
    }

    #[derive(Debug)]
    pub struct Genome(String);
    #[derive(Debug)]
    pub struct Skew {
        gc_diff_profile: Vec<i32>,
    }

    /// Construct the skew diagram for the provided genome
    impl From<&Genome> for Skew {
        fn from(genome: &Genome) -> Self {
            todo!()
        }
    }

    impl Skew {
        /// Returns the list of indices where the min
        /// of the skew diagram is located. This is the approximate location
        /// of the origin of replication
        pub fn min(&self) -> Vec<usize> {
            todo!()
        }
    }

    /// The hamming distance between two sequences is simply the count
    /// of the number of positions where the two sequences differ.
    /// If one sequence is shorter than the other, ...
    pub fn hamming_distance(a: &str, b: &str) -> usize {
        a.chars().zip(b.chars()).filter(|(ac, bc)| ac != bc).count()
    }

    /// This is an approximate version of the `find_all_occurrences` procedure.
    /// Instead of only looking for exact matches, it also retrieves positions
    /// where pattern appears with at most (>=) d mismatches. That is, the Hamming
    /// distance is less than or equal to d
    pub fn find_approx_all_occurrences(s: &str, pattern: &str, d: usize) -> Vec<usize> {
        todo!()
    }
}

#[cfg(test)]
mod test_ori {
    use super::ori;
    const VC_ORI: &str = "atcaatgatcaacgtaagcttctaagcatgatcaaggtgctcacacagtttatccacaac
    ctgagtggatgacatcaagataggtcgttgtatctccttcctctcgtactctcatgacca
    cggaaagatgatcaagagaggatgatttcttggccatatcgcaatgaatacttgtgactt
    gtgcttccaattgacatcttcagcgccatattgcgctggccaaggtgacggagcgggatt
    acgaaagcatgatcatggctgttgttctgtttatcttgttttgactgagacttgttagga
    tagacggtttttcatcactgactagccaaagccttactctgcctgacatcgaccgtaaat
    tgataatgaatttacatgcttccgcgacgatttacctcttgatcatcgatccgattgaag
    atcttcaattgttaattctcttgcctcgactcatagccatgatgagctcttgatcatgtt
    tccttaaccctctattttttacggaagaatgatcaagctgctgctcttgatcatcgtttc";
    #[test]
    fn pattern_count() {
        let s = "ATCATGATTTTGGCTACTGTAGCTGAT";
        let pattern = "TT";
        let cnt = ori::pattern_count(s, pattern);
        println!("{}", cnt)
    }

    #[test]
    fn find_most_frequent_kmers() {
        // let vc_genome = std::fs::read_to_string("data/vibrio_cholerae.txt");
        // let genome = match vc_genome {
        //     Err(err) => panic!(err),
        //     Ok(genome) => genome,
        // }; use mmap/ buffered reader
        let counter = ori::count_of_kmers_in(VC_ORI, 9);
        let most_frequent_nine_mers = ori::find_most_frequent_kmers(&counter);
        assert_eq!(most_frequent_nine_mers.len(), 4);
        for (k, &v) in most_frequent_nine_mers {
            assert_eq!(v, 3);
            println!("{} appears {} times", k, v);
        }
    }

    #[test]
    fn reverse_complement() {
        let base_strand = "ATGATCAAG";
        let complement = "CTTGATCAT";
        assert_eq!(
            complement,
            ori::reverse_complement(base_strand, ori::deoxyribose_complement)
        );
    }

    #[test]
    fn find_all_occurrences() {
        let s = "panamabanana$";
        let p = "ana";
        let _ = ori::find_all_occurrences(s, p);
    }
}

mod genome_assembly {
    use super::ori;

    /// A k-mer is simply a string of length k
    #[derive(Debug)]
    pub struct KMer(String);

    impl From<String> for KMer {
        fn from(s: String) -> Self {
            KMer(s)
        }
    }

    impl KMer {
        pub fn len(&self) -> usize {
            self.0.len()
        }

        // TODO: Impl slicing for KMer
    }

    #[derive(Debug)]
    pub struct KMerComposition {
        /// The length of each substring
        k: usize,

        /// A lexicographically ordered list of all the substrings
        /// of length `k` in the underlying
        reads: Vec<KMer>,
    }

    /// The k-mer composition of a sequence `s` the collection of all
    /// k-mer substrings (substrings of len=k) of `s` including duplicates.
    /// These substrings are returned as a list of lexicographically ordered
    /// strings
    pub fn kmer_composition_of(s: &ori::Genome, k: usize) -> KMerComposition {
        todo!()
    }

    /// Reconstruct a Genome from its k-mer decomposition
    pub fn kmer_reconstruction_of(c: &KMerComposition) -> ori::Genome {
        todo!()
    }
    #[derive(Debug)]
    pub struct NodeHandle(usize);
    #[derive(Debug)]
    pub struct Node {
        /// A string of len = k
        kmer: KMer,

        /// Links to nodes in the graph whose (k-1)-prefix
        /// is the same as this node's (k-1)-suffix
        neighbors: Option<Vec<NodeHandle>>,
    }

    /// A directed graph
    #[derive(Debug)]
    pub struct OverlapGraph {
        /// An arena holding all the nodes in the graph
        /// This should only be accessed via the `NodeHandle`
        nodes: Vec<Node>,
    }

    /// A genome path is a sequence of all the nodes in the graph
    /// such that consecutive nodes overlap
    #[derive(Debug)]
    pub struct GenomePath<'a> {
        path: Vec<&'a Node>,
    }

    /// Reconstructs a string from its genome path.
    impl From<&GenomePath<'_>> for ori::Genome {
        fn from(path: &GenomePath) -> Self {
            todo!()
        }
    }

    /// Construct the overlap graph of a collection of kmers
    impl From<&[KMer]> for OverlapGraph {
        fn from(kmers: &[KMer]) -> Self {
            todo!()
        }
    }

    /// An adjacency matrix for an unweighted graph
    /// stored in row major
    #[derive(Debug)]
    pub struct AdjacencyMatrix<T> {
        /// A bit matrix that tells us if two nodes are connected
        matrix: Vec<u8>,

        /// The identifiers for our nodes since we cannot keep this in the
        /// matrix
        num_nodes: Vec<T>,
    }
    #[derive(Debug)]
    pub struct DeBruijnGraph {
        // 1. Construct the de Bruijn graph of a string
    // 2. Construct an Eulerian path in a graph
    // 3. Construct the de Bruin graph of a collection of kmers
    // 4. Find an Eulerian cycle in a graph
    }
}

#[cfg(test)]
mod test_genome_assembly {}
