//! Implementation of data structures for working with integers.
//! ## Contents
//! 1. Utilities for implementing word-level parallelism
//! 2. An X-Fast Trie
//! 3. A Y-Fast Trie
//! 4. A Fusion Tree

mod wlp {

    const USIZE_BITS: usize = 64;
    pub fn top_k_bits_of(x: usize, k: usize) -> usize {
        assert!(k != 0);
        let mut mask: usize = 1;

        // Shift the 1 to the index that is `k`
        // positions from the last index location.
        // That is `k` away from 64
        mask <<= USIZE_BITS - k;

        // Turn that one into a zero. And all
        // the other 63 zeros into ones. This
        // basically introduces a hole. in the next
        // step, we'll use this hole to trap a cascade
        // of carries
        mask = !mask;

        // I think this is the most interesting/entertaining part.
        // Adding a one triggers a cascade of carries that flip all
        // the bits (all ones) before the location of the zero from above into
        // zeros. The cascade stops when they reach the zero from
        // above. Since it is a zero, adding a 1 does not trigger a carry
        //
        // In the end, we have a mask where the top k bits are ones
        mask += 1;

        // This is straightforward
        x & mask
    }

    /// The abstraction for a single node in our b-tree
    /// that is specialized for holding small integers
    /// that can be packed into a single machine word
    #[derive(Debug, Default)]
    pub struct SardineCan {
        /// The actual storage container
        buckets: u64,

        /// The count of items in this node.
        count: u8,
    }

    impl std::fmt::Display for SardineCan {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            let res = format!("{:b}", self.buckets);
            writeln!(f, "{}", res)
        }
    }

    impl SardineCan {
        /// Procedure to store a single small integer in a given node
        /// Note that we do not handle the case where a can could be full.
        /// We ignore that because, ideally, this data structure would be part
        /// of a larger B-Tree implementation that would take care of such details
        pub fn add(&mut self, mut x: u8) {
            // Add the sentinel bit. It is set to 0
            x &= 0b0111_1111;

            // Make space in the bucket for the new item
            self.buckets <<= 8;

            // Add the new item into the bucket
            self.buckets |= x as u64;

            // Increment the count of items
            self.count += 1;
        }

        /// Produces a number that is the result of replicating `x`
        /// as many times to produce a value with as many bits as
        /// the bits in `buckets`
        pub fn parallel_tile_64(query: u8) -> u64 {
            // This carefully chosen multiplier will have the desired effect of replicating `x`
            // seven times, interspersing each instance of `x` with a 0
            let multiplier: u64 = 0b10000000_10000000_10000000_10000000_10000000_10000000_100000001;

            // Produce the provisional tiled number. We still need to set its
            // sentinel bits to 1
            let tiled_x = query as u64 * multiplier;

            // The bitmask to turn on  the sentinel bits
            let sentinel_mask: u64 = 0b10000000_10000000_10000000_10000000_10000000_10000000_1000000010000000;

            // Set the sentinel bits to 1 and return the tiled number
            tiled_x | sentinel_mask
        }

        /// Calculate how many items in this can are less than or
        /// equal to `x`
        pub fn parallel_rank(&self, x: u8) -> u8 {
            Self::parallel_rank_helper(self.buckets, x)
        }

        fn parallel_rank_helper(packed_keys: u64, query: u8) -> u8 {
            // Perform the parallel comparison
            let mut difference = Self::parallel_tile_64(query) - packed_keys;

            // Ultimately, we're only interested in whether the spacer sentinel bits
            // are turned on or off. In particular, we just need to know how many are
            // turned on. Here we use the mask from `parallel_tile` to isolate them
            let sentinel_mask: u64 = 0b10000000_10000000_10000000_10000000_10000000_10000000_1000000010000000;
            difference &= sentinel_mask;

            // There's an alternative method of counting up how many spacer bits are set to 1.
            // That method involves using a well chosen multiplier. To check it out look in
            // at the  `parallel_count` method below
            difference.count_ones() as u8
        }

        /// Counts up how many of the sentinel bits of `difference` are turned on
        fn parallel_count(difference: u64) -> u8 {
            let stacker = 0b10000000_10000000_10000000_10000000_10000000_10000000_100000001u64;
            let mut stacked = difference as u128 * stacker as u128;
            stacked >>= 63;
            stacked &= 0b111;
            stacked as u8
        }
    }

    #[derive(Debug)]
    struct FourRussiansMSB {
        /// The secondary routing bit array
        macro_bit_array: u8,

        /// This is simply the number whose `msb` we'd like to find.
        /// It is logically split into blocks of 8 bits
        micro_arrays: u64,
    }

    impl FourRussiansMSB {
        pub fn build(query: u64) -> Self {
            let macro_bit_array = Self::generate_macro_bit_array(query);
            FourRussiansMSB {
                macro_bit_array,
                micro_arrays: query,
            }
        }

        /// Generates the routing macro array. To do so, it
        /// relies on the observation that a block contains a
        /// 1 bit if it's highest bit is a 1 or if its
        /// lower 7 bits' numeric value is greater than 0.
        fn generate_macro_bit_array(query: u64) -> u8 {
            // The first step is to extract information about the highest bit in each block.
            let high_bit_mask = 0b10000000_10000000_10000000_10000000_10000000_10000000_10000000_10000000u64;
            let is_high_bit_set = query & high_bit_mask;

            // The second step is to extract information about the lower seven bits
            // in each block. To do so, we use parallel_compare, which is basically
            // subtraction.
            let packed_ones = 0b00000001_00000001_00000001_00000001_00000001_00000001_00000001_00000001u64;
            let mut are_lower_bits_set = query | high_bit_mask;
            are_lower_bits_set -= packed_ones;
            are_lower_bits_set &= high_bit_mask;

            // We unify the information from the first two steps into a single value
            // that tells us if a block could conceivably contain the MSB
            let is_block_active = is_high_bit_set | are_lower_bits_set;

            // To generate the macro array, we need to form an 8-bit number out of the
            // per-block highest bits from the last step. To pack them together, we simply use
            // an appropriate multiplier which does the work of a series of bitshifts
            let packer = 0b10000001_00000010_00000100_00001000_00010000_00100000_010000001u64;
            let mut macro_bit_array = is_block_active as u128 * packer as u128;
            macro_bit_array >>= 49;
            if is_block_active >> 56 == 0 {
                macro_bit_array &= 0b0111_1111;
            } else {
                macro_bit_array |= 0b1000_0000;
                macro_bit_array &= 0b1111_1111;
            }
            macro_bit_array as u8
        }

        pub fn get_msb(&self) -> u8 {
            let block_id = self.msb_by_rank(self.macro_bit_array);
            let block_start = (block_id - 1) * 8;
            let msb_block = self.get_msb_block(block_start); // msb block is wrong!!
            let msb = self.msb_by_rank(msb_block);
            let in_block_location = msb - 1;
            block_start + in_block_location
        }

        /// Given a block id -- which is the msb value in the macro routing array,
        /// this method retrieves the 8 bits that represent that block
        /// from the `micro_arrays`. `block_id 0 refers to the highest
        fn get_msb_block(&self, block_start: u8) -> u8 {
            let block_mask = 0b1111_1111u64;
            let mut block = self.micro_arrays >> block_start;
            block &= block_mask;
            block as u8
        }

        /// Finds the index of the most significant bit in the
        /// provided 8-bit number by finding its rank among the
        /// 8 possible powers of 2: <1, 2, 4, 8, 16, 32, 64, 128>.
        /// To do so in constant time, it employs techniques from
        /// our discussion of `parallel_rank`
        fn msb_by_rank(&self, query: u8) -> u8 {
            // Perform the parallel comparison
            let tiled_query = Self::parallel_tile_128(query);
            let packed_keys = 0b000000001_000000010_000000100_000001000_000010000_000100000_001000000_010000000u128;
            let mut difference = tiled_query - packed_keys;

            // Isolate the spacer sentinel bits
            let sentinel_mask = 0b100000000_100000000_100000000_100000000_100000000_100000000_100000000_100000000u128;
            difference &= sentinel_mask;

            // Count the number of spacer bits that are turned on
            difference.count_ones() as u8
        }

        /// Produces a number that is a result of replicating the query
        /// eight times. This uses 72 bits of space.
        pub fn parallel_tile_128(query: u8) -> u128 {
            let multiplier = 0b100000000_100000000_100000000_100000000_100000000_100000000_100000000_1000000001u128;

            // Produce the provisional tiled number. We still need to set its
            // sentinel bits to 1
            let tiled_query = query as u128 * multiplier;

            // The bitmask to turn on  the sentinel bits
            let sentinel_mask = 0b100000000_100000000_100000000_100000000_100000000_100000000_100000000_100000000u128;

            // Set the sentinel bits to 1 and return the tiled number
            tiled_query | sentinel_mask
        }
    }

    pub fn get_msb_idx_of(query: u64) -> u8 {
        FourRussiansMSB::build(query).get_msb()
    }

    pub fn lcp_len_of(a: u64, b: u64) -> u64 {
        63 - get_msb_idx_of(a ^ b) as u64
    }
}

#[cfg(test)]
mod test_wlp {
    use super::wlp;
    use rand::Rng;

    #[test]
    fn sardine_add() {
        let mut rng = rand::thread_rng();
        let mut can = wlp::SardineCan::default();
        for _ in 0..8 {
            let small_int = rng.gen_range(0..=1 << 7);
            can.add(small_int);
            println!("{:b}, can is {}", small_int, can)
        }
        //_11101110_10101110_11111000_11001101_10101111_10001101_11110111_11100001
        //_01010110_00111110_00111110_01000011_00011011_00101111_00100011_01111010
        //1100111
    }

    #[test]
    fn sardine_tile() {
        let tiled = wlp::SardineCan::parallel_tile_64(0b1100111);
        println!("{:b}", tiled)
        // 1100111_01100111_01100111_01100111
        // 01100111_01100111_01100111_01100111_01100111_01100111_01100111_01100111
        // 11100111_11100111_11100111_11100111_11100111_11100111_11100111_11100111
    }

    #[test]
    fn test_stacker() {
        // Test alternative method of computing rank
        let a = 0b10000000_10000000_10000000_10000000_10000000_10000000_100000001u64;
        let b = 0b10000000_00000000_10000000_10000000_00000000_10000000_00000000_10000000u64;
        let mut c = a as u128 * b as u128;
        println!("{:b}", c);
        c >>= 63;
        println!("{:b}", c);
        println!("{}", c & 0b111);
    }

    #[test]
    fn sardine_rank() {
        let mut rng = rand::thread_rng();
        let mut can = wlp::SardineCan::default();
        for _ in 0..8 {
            let small_int = rng.gen_range(0..=1 << 7);
            can.add(small_int);
        }
        println!("{}", can.parallel_rank(0b1100111));
        // _10000000_00000000_10000000_10000000_00000000_10000000_00000000_10000000
        // 10000000_10000000_10000000_10000000_10000000_10000000_100000001
    }

    #[test]
    fn pack() {
        let tt = 0b00010000_10000000_10000000_10000000_10000000_00000000_00000000_00000000u64;
        let m = 0b10000001_00000010_00000100_00001000_00010000_00100000_010000001u64;
        let mut c = tt as u128 * m as u128;
        println!("{:b}", c);
        c >>= 49;
        println!("{:b}", c);
        if tt >> 56 == 0 {
            c &= 0b0111_1111;
        } else {
            c |= 0b1000_0000;
            c &= 0b1111_1111;
        }
        println!("{:b}", c);
        // 100000_01100000_11100001_11100011_11000111_10001111_00011110_001111000
        // 100000_10000001_01000010_11000101_11001011_10010111_00101110_01011100_01111000
    }

    #[test]
    fn get_msb() {
        use super::wlp;
        let msb = wlp::get_msb_idx_of(873);
        assert_eq!(9, msb);
        let base: usize = 2;
        let msb = wlp::get_msb_idx_of(base.pow(32) as u64);
        assert_eq!(32, msb);
        let msb = wlp::get_msb_idx_of(base.pow(55) as u64);
        assert_eq!(55, msb);
        let msb = wlp::get_msb_idx_of((base.pow(56) + 13) as u64);
        assert_eq!(56, msb);
        let msb = wlp::get_msb_idx_of((base.pow(61) + 31) as u64);
        assert_eq!(61, msb);
        let msb = wlp::get_msb_idx_of((2u128.pow(64) - 1) as u64);
        assert_eq!(63, msb);
        let msb = wlp::get_msb_idx_of(base.pow(48) as u64);
        assert_eq!(48, msb);
        let msb = wlp::get_msb_idx_of(base.pow(63) as u64);
        assert_eq!(63, msb);
        let msb = wlp::get_msb_idx_of(255);
        assert_eq!(7, msb);
        let msb = wlp::get_msb_idx_of(1);
        assert_eq!(0, msb);
        let msb = wlp::get_msb_idx_of(16);
        assert_eq!(4, msb);
        let msb = wlp::get_msb_idx_of(256);
        assert_eq!(8, msb);
        let msb = wlp::get_msb_idx_of(25);
        assert_eq!(4, msb);
        let msb = wlp::get_msb_idx_of(91);
        assert_eq!(6, msb);
        let msb = wlp::get_msb_idx_of(base.pow(16) as u64);
        assert_eq!(16, msb);
        let msb = wlp::get_msb_idx_of(1 << 18);
        assert_eq!(18, msb);
    }
}
