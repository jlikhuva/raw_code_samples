//! Implementation of data structures for working with integers.
//! ## Contents
//! 1. Utilities for implementing word-level parallelism
//! 2. An X-Fast Trie
//! 3. A Y-Fast Trie
//! 4. A Fusion Tree
/// A structure for storing small, byte sized integers by packing them in two
/// words (128 bits). This allows us to implement some
/// really cool word-level parallelism  operations
/// Data(storage_container, count) where count is
/// the number of ints currently in the container
#[derive(Debug, Default)]
pub struct SardineCan {
    can: u128,
    count: u8,
}

impl SardineCan {
    pub fn new() -> Self {
        SardineCan::default()
    }

    pub fn add(&mut self, v: u8) {
        todo! {}
    }

    pub fn par_compare(&self, other: u8) {
        todo!()
    }

    pub fn par_tile(&self, other: u8) {
        todo!()
    }

    pub fn par_add(&self, other: u8) {}

    pub fn par_rank(&self, other: u8) {
        todo!()
    }

    pub fn msb(n: u64) {
        todo!()
    }

    pub fn lcp(m: u64, n: u64) {
        todo!()
    }
}
