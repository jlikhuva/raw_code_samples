/// A structure for storing small integers
/// of up to 7 bits by packing them in a single
/// word (64 bits). This allows us to implement
/// some really cool word-level parallelism
/// operations
///
/// Data(storage_container, count) where count is
/// the number of ints currently in the container
#[derive(Debug)]
pub struct SardineCan(u64, u8);

impl SardineCan {
    pub fn new() -> Self {
        SardineCan(0, 0)
    }

    pub fn add(&mut self, v: i8) {
        todo! {}
    }

    pub fn par_compare(&self, other: i8) {
        todo!()
    }

    pub fn par_tile(&self, other: i8) {
        todo!()
    }

    pub fn par_add(&self, other: i8) {}

    pub fn par_rank(&self, other: i8) {
        todo!()
    }

    pub fn msb(n: u64) {
        todo!()
    }

    pub fn lcp(m: u64, n: u64) {
        todo!()
    }
}
