use std::{
    collections::hash_map::RandomState,
    hash::{BuildHasher, Hash, Hasher},
    marker::PhantomData,
};

use rand::{thread_rng, Rng};

/// A sampler will be anything that can observe a possibly infinite
/// number of items and produce a finite random sample from that
/// stream
pub trait Sampler<T> {
    /// We observe each item as it comes in
    fn observe(&mut self, item: T);

    /// Produce a random uniform sample of the all the items that
    /// have been observed so far.
    fn sample(&self) -> &[T];
}

#[derive(Debug)]
pub struct ReservoirSampler<T> {
    /// This is what we produce whenever someone calls sample. It
    /// maintains this invariant: at any time-step `t`, reservoir
    /// contains a uniform random sample of the elements seen thus far
    reservoir: Vec<T>,

    /// This determines the size of the reservoir. The
    /// client sets this value
    sample_size: usize,

    /// The number of items we have seen so far. We use this
    /// to efficiently update the reservoir in constant time
    /// while maintaining its invariant.
    count: usize,
}
impl<T> ReservoirSampler<T> {
    /// Create a new reservoir sampler that will produce a random sample
    /// of the given size.
    pub fn new(sample_size: usize) -> Self {
        ReservoirSampler {
            reservoir: Vec::with_capacity(sample_size),
            sample_size,
            count: 0,
        }
    }
}

impl<T> Sampler<T> for ReservoirSampler<T> {
    fn observe(&mut self, item: T) {
        // To make sure we that maintain the reservoir invariant,
        // we have to ensure that each incoming item has an equal
        // probability of being included in the sample. We do so by
        // generating a random index `k`, and if `k` falls within
        // our reservoir, we replace the item at `k` with the
        // new item
        if self.reservoir.len() == self.sample_size {
            let rand_idx = thread_rng().gen_range(0..self.count);
            if rand_idx < self.sample_size {
                self.reservoir[rand_idx] = item;
            }
        } else {
            // If the reservoir is not full, no need
            // to evict items
            self.reservoir.push(item)
        }
        self.count += 1;
    }

    fn sample(&self) -> &[T] {
        // Since we always have a random sample ready to go, we
        // simply return it
        self.reservoir.as_ref()
    }
}

#[test]
fn test_reservoir_sampler() {
    let mut sampler = ReservoirSampler::new(10);
    for i in 0..1000 {
        sampler.observe(i);
    }
    println!("{:?}", sampler.sample())
}

/// Indicates that the filter has probably seen a given
/// item before
pub struct ProbablyYes;
/// indicates that a filter hasn't seen a given item before.
pub struct DefinitelyNot;
/// A filter will be any object that is able to observe a possibly
/// infinite stream of items and, at any point, answer if a given
/// item has been seen before
pub trait Filter<T> {
    /// We observe each item as it comes in. We do not use terminology such as
    /// `insert` because we do not store any of the items.
    fn observe(&mut self, item: T);

    /// Produce a random uniform sample of the all the items that
    /// have been observed so far.
    fn has_been_observed_before(&self, item: &T) -> Result<ProbablyYes, DefinitelyNot>;
}

#[derive(Debug)]
pub struct BloomFilter<T: Hash> {
    /// The bit vector
    buckets: Vec<bool>,

    /// The list of hash functions
    hash_functions: Vec<RandomState>,

    _marker: PhantomData<T>,
}

impl<T: Hash> BloomFilter<T> {
    /// Creates a new bloom filter `m` buckets and `k` hash functions.
    /// Each hash function is randomly initialized and is independent
    /// of the other hash functions
    pub fn new(m: usize, k: usize) -> Self {
        let mut buckets = Vec::with_capacity(m);
        for _ in 0..m {
            buckets.push(false);
        }
        let mut hash_functions = Vec::with_capacity(k);
        for _ in 0..k {
            hash_functions.push(RandomState::new());
        }

        BloomFilter {
            buckets,
            hash_functions,
            _marker: PhantomData,
        }
    }

    /// ..
    fn get_index(&self, state: &RandomState, item: &T) -> usize {
        let mut hasher = state.build_hasher();
        item.hash(&mut hasher);
        let idx = hasher.finish() % self.buckets.len() as u64;
        idx as usize
    }
}

impl<T: Hash> Filter<T> for BloomFilter<T> {
    /// ...
    fn observe(&mut self, item: T) {
        for state in &self.hash_functions {
            let index = self.get_index(state, &item);
            self.buckets[index] = true;
        }
    }

    /// ...
    fn has_been_observed_before(&self, item: &T) -> Result<ProbablyYes, DefinitelyNot> {
        for state in &self.hash_functions {
            let index = self.get_index(state, &item);
            if !self.buckets[index] {
                return Err(DefinitelyNot);
            }
        }
        Ok(ProbablyYes)
    }
}

#[derive(Debug)]
pub struct CountMinSketch<T: Hash, const M: usize, const N: usize> {
    /// A count sketch is defined by `N` hash functions
    /// and `M` bucket groups. Each of the `N` hash functions
    /// maps an item into a single slot in a corresponding bucket group
    sketch_matrix: [[u64; M]; N],

    ///
    hash_functions: [RandomState; N],

    /// As with the Bloom Filter, we'd like to
    _marker: PhantomData<T>,
}

impl<T: Hash, const M: usize, const N: usize> CountMinSketch<T, M, N> {
    /// ...
    pub fn inc(&mut self, item: &T) {
        for (i, state) in self.hash_functions.iter().enumerate() {
            let idx = self.get_index(state, item);
            self.sketch_matrix[i][idx] += 1;
        }
    }

    /// ...
    pub fn count(&mut self, item: &T) -> u64 {
        let mut cur_min = u64::MIN;
        for (i, state) in self.hash_functions.iter().enumerate() {
            let idx = self.get_index(state, item);
            let cur_value = self.sketch_matrix[i][idx];
            if cur_value < cur_min {
                cur_min = cur_value;
            }
        }
        cur_min
    }

    /// Hashes the given item and maps it to the appropriate index location within
    /// a single bucket
    fn get_index(&self, state: &RandomState, item: &T) -> usize {
        let mut hasher = state.build_hasher();
        item.hash(&mut hasher);
        let idx = hasher.finish() % self.sketch_matrix[0].len() as u64;
        idx as usize
    }
}
