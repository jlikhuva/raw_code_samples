use rand::{random, thread_rng, Rng};

pub trait Sampler<T> {
    fn observe(&mut self, item: T);
    fn sample(&self) -> &[T];
}

#[derive(Debug, Default)]
pub struct ReservoirSampler<T> {
    reservoir: Vec<T>,
    sample_size: usize,
    count: usize,
}
impl<T: Default> ReservoirSampler<T> {
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
        if self.reservoir.len() >= self.sample_size {
            let keep_probability = (self.sample_size / self.count) as f32;
            if keep_probability < thread_rng().gen_range(0.0, self.sample_size as f32) {
                let rand_idx = thread_rng().gen_range(0, self.sample_size);
                self.reservoir[rand_idx] = item;
            }
        } else {
            self.reservoir.push(item)
        }
        self.count += 1;
    }

    fn sample(&self) -> &[T] {
        self.reservoir.as_ref()
    }
}

#[test]
fn test_reservoir_sampler() {
    let mut sampler = ReservoirSampler::new(100);
    for i in 0..1000 {
        sampler.observe(i);
    }
    println!("{:#?}", sampler.sample())
}
