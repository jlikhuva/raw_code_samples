use bitvec::prelude::{BitVec, LocalBits};
use core::hash::BuildHasher;
use std::hash::Hasher;
use std::marker::PhantomData;
use twox_hash::RandomXxHashBuilder64;

unsafe fn as_u8_slice<T: Sized>(p: &T) -> &[u8] {
    std::slice::from_raw_parts((p as *const T) as *const u8, std::mem::size_of::<T>())
}
pub struct BloomFilter<T> {
    buckets: BitVec,
    hash_functions: Vec<RandomXxHashBuilder64>,
    _marker: PhantomData<T>,
}
impl<T> BloomFilter<T> {
    pub fn new(num_buckets: usize, num_hash_funcs: usize) -> Self {
        let mut buckets = BitVec::<LocalBits, usize>::with_capacity(num_buckets);
        let mut hash_functions = Vec::new();
        for _ in 0..num_buckets {
            buckets.push(false);
        }
        for _ in 0..num_hash_funcs {
            hash_functions.push(RandomXxHashBuilder64::default())
        }
        BloomFilter {
            buckets,
            hash_functions,
            _marker: PhantomData,
        }
    }

    pub fn insert(&mut self, item: T) {
        for f in &mut self.hash_functions {
            let mut hasher = f.build_hasher();
            unsafe {
                hasher.write(as_u8_slice(&item));
            }
            let idx = hasher.finish() % self.buckets.len() as u64;
            self.buckets.set(idx as usize, true);
        }
    }

    pub fn contains(&mut self, item: T) -> bool {
        for f in &mut self.hash_functions {
            let mut hasher = f.build_hasher();
            unsafe {
                hasher.write(as_u8_slice(&item));
            }
            let idx = hasher.finish() % self.buckets.len() as u64;
            if !self.buckets.get(idx as usize).unwrap() {
                return false;
            }
        }
        true
    }
}

pub struct CountMinSketch<T> {
    sketch_matrix: Vec<Vec<u64>>,
    hash_functions: Vec<RandomXxHashBuilder64>,
    _marker: PhantomData<T>,
}

impl<T> CountMinSketch<T> {
    pub fn new(num_buckets: usize, num_hash_funcs: usize) -> Self {
        let mut sketch_matrix = Vec::with_capacity(num_hash_funcs);
        let mut hash_functions = Vec::with_capacity(num_hash_funcs);
        for _ in 0..num_hash_funcs {
            sketch_matrix.push(vec![0; num_buckets]);
            hash_functions.push(RandomXxHashBuilder64::default());
        }
        CountMinSketch {
            sketch_matrix,
            hash_functions,
            _marker: PhantomData,
        }
    }

    pub fn inc(&mut self, item: T) {
        for (i, f) in self.hash_functions.iter().enumerate() {
            unsafe {
                let mut hasher = f.build_hasher();
                hasher.write(as_u8_slice(&item));
                let idx = hasher.finish() % self.sketch_matrix[0].len() as u64;
                self.sketch_matrix[i][idx as usize] += 1;
            }
        }
    }

    pub fn count(&self, item: T) -> u64 {
        let mut cur_min = u64::MIN;
        for (i, f) in self.hash_functions.iter().enumerate() {
            unsafe {
                let mut hasher = f.build_hasher();
                hasher.write(as_u8_slice(&item));
                let idx = hasher.finish() % self.sketch_matrix[0].len() as u64;
                let cur_value = self.sketch_matrix[i][idx as usize];
                if cur_value < cur_min {
                    cur_min = cur_value;
                }
            }
        }
        cur_min
    }
}

#[cfg(test)]
mod test {
    struct Point {
        _x: String,
        _y: usize,
    }
    #[test]
    fn test_from_raw_parts() {
        use super::as_u8_slice;
        let p = Point {
            _x: "Joseph".to_string(),
            _y: 33,
        };
        unsafe {
            let bytes = as_u8_slice(&p);
            println!("{:?}", bytes)
        }
    }
}
