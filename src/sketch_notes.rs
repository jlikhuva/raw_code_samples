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
