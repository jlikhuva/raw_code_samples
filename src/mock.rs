mod bloomberg_i {
    use std::collections::HashSet;

    struct Solution;

    impl Solution {
        /// Input: Sorted Array of distinct integers
        ///        A target value
        ///
        /// Output: Index of the target value in the array
        ///         If the target is not in the array, return
        ///         location where it would have occurred
        pub fn search_insert(nums: Vec<i32>, target: i32) -> i32 {
            Self::search_logarithmic(&nums, target)
        }

        /// The naive procedure does not utilize the fact that the array
        /// is sorted. That fact allows us to use binary search
        fn search_logarithmic(haystack: &[i32], needle: i32) -> i32 {
            match haystack.binary_search(&needle) {
                Ok(location) | Err(location) => location as i32,
            }
        }

        // Province --> A group of directly or indirectly connected cities
        ///          --> This is a connected component of a graph
        /// Input: An n x n adjacency matrix.
        /// Output: The total number of provinces
        ///
        /// This is basically asking us to calculate the number of connected components
        /// in the given graph
        pub fn find_circle_num(is_connected: Vec<Vec<i32>>) -> i32 {
            let mut already_explored = HashSet::with_capacity(is_connected.len());
            let mut num_components = 0;

            // Each city could be its own component.
            for city in 0..is_connected.len() {
                // Skip over cities that are already part of some component
                if already_explored.contains(&city) {
                    continue;
                }
                num_components += 1;
                Self::explore_from(city, &is_connected, &mut already_explored);
            }
            num_components
        }

        /// Depth first search explore from the given src city
        fn explore_from(src: usize, graph: &Vec<Vec<i32>>, seen: &mut HashSet<usize>) {
            seen.insert(src);
            let src_neighbors = &graph[src];
            for (neighbor, &is_connected) in src_neighbors.iter().enumerate() {
                if is_connected == 0 || seen.contains(&neighbor) {
                    continue;
                }
                Self::explore_from(neighbor as usize, graph, seen)
            }
        }
    }

    impl Solution {
        /// 1. The i-th row has i elements
        /// 2. The first two rows are trivially defined: [1], [1, 1]
        pub fn generate(num_rows: i32) -> Vec<Vec<i32>> {
            let n = num_rows as usize;
            let mut triangle = Vec::with_capacity(n);
            triangle.push(vec![1]);
            triangle.push(vec![1, 1]);
            for row_id in 3..=n {
                let mut cur_row = Vec::with_capacity(row_id);
                let prev_row = triangle.last().unwrap().as_slice();
                cur_row.push(1);
                for tuple_window in prev_row.windows(2) {
                    cur_row.push(tuple_window[0] + tuple_window[1]);
                }
                cur_row.push(1);
                triangle.push(cur_row);
            }
            triangle
        }
    }

    #[test]
    fn test_components() {
        let t = vec![vec![1, 1, 0], vec![1, 1, 0], vec![0, 0, 1]];
        Solution::find_circle_num(t);
    }
}

mod bloomberg_ii {
    use std::collections::HashSet;

    pub fn max_area_of_island(grid: Vec<Vec<i32>>) -> i32 {
        let mut max_area = 0;
        let mut visited = HashSet::with_capacity(grid.len());

        for x in 0..grid.len() {
            for y in 0..grid[0].len() {
                if !visited.contains(&(x, y)) && grid[x][y] == 1 {
                    let cur_max = max_area_from((x, y), &grid, &mut visited);
                    max_area = std::cmp::max(max_area, cur_max);
                }
            }
        }
        max_area
    }

    fn max_area_from((x, y): (usize, usize), grid: &Vec<Vec<i32>>, visited: &mut HashSet<(usize, usize)>) -> i32 {
        visited.insert((x, y));
        let neighbors = get_neighbors_of((x, y), grid);

        let mut area = 0;
        for (x_n, y_n) in neighbors {
            if !visited.contains(&(x_n, y_n)) && grid[x_n][y_n] == 1 {
                area += max_area_from((x_n, y_n), grid, visited);
            }
        }

        // Remember to add in the area of this piece
        area += 1;
        area
    }

    // Returns all neighboring squares that are within the grid
    fn get_neighbors_of((x, y): (usize, usize), grid: &Vec<Vec<i32>>) -> Vec<(usize, usize)> {
        let mut neighbors = Vec::with_capacity(4);
        if x > 0 {
            neighbors.push((x - 1, y));
        }
        if y > 0 {
            neighbors.push((x, y - 1));
        }
        if x < grid.len() - 1 {
            neighbors.push((x + 1, y));
        }
        if y < grid[0].len() - 1 {
            neighbors.push((x, y + 1))
        }
        neighbors
    }

    #[test]
    fn area() {
        let inp = vec![
            vec![1, 1, 0, 0, 0],
            vec![1, 1, 0, 0, 0],
            vec![0, 0, 0, 1, 1],
            vec![0, 0, 0, 1, 1],
        ];
        max_area_of_island(inp);
    }
}

mod facebook {
    use std::{collections::HashMap, usize};

    struct SparseVector {
        /// We represent our sparse vector as a mapping from
        /// an index to the value stored at that index
        entries: HashMap<usize, i32>,
    }

    impl SparseVector {
        fn new(nums: Vec<i32>) -> Self {
            let mut entries = HashMap::new();

            // Scan over the input and only add non-zero entries
            for (idx, &value) in nums.iter().enumerate() {
                if value != 0 {
                    entries.insert(idx, value);
                }
            }

            SparseVector { entries }
        }

        // Return the dotProduct of two sparse vectors
        fn dot_product(&self, vec: SparseVector) -> i32 {
            let mut res = 0;
            for (idx, value) in &self.entries {
                if vec.entries.contains_key(&idx) {
                    res += value * vec.entries.get(&idx).unwrap();
                }
            }
            res
        }
    }

    pub fn bulb_switch(n: i32) -> i32 {
        if n == 0 {
            return 0;
        }

        let mut bulbs = Vec::with_capacity(n as usize);

        // Round 1: Turn on all the bulbs
        for i in 0..n as usize {
            bulbs.push(1);
        }

        // Round 2 - n: toggle every i-th bulb
        let mut round_id = 2;
        while round_id <= n as usize {
            for i in 1..n as usize {
                if (i + 1) % round_id == 0 {
                    if bulbs[i] == 1 {
                        bulbs[i] = 0;
                    } else {
                        bulbs[i] = 1;
                    }
                }
            }
            round_id += 1;
        }

        bulbs.iter().sum()
    }

    #[test]
    fn bulbs() {
        bulb_switch(1);
    }

    struct NumMatrix {
        matrix: Vec<Vec<i32>>,
    }

    /**
     * `&self` means the method takes an immutable reference.
     * If you need a mutable reference, change it to `&mut self` instead.
     */
    impl NumMatrix {
        fn new(matrix: Vec<Vec<i32>>) -> Self {
            NumMatrix { matrix }
        }

        fn sum_region(&self, row1: i32, col1: i32, row2: i32, col2: i32) -> i32 {
            let mut sum = 0;
            for row_id in row1..=row2 {
                for col_id in col1..=col2 {
                    sum += self.matrix[row_id as usize][col_id as usize];
                }
            }
            sum
        }
    }
}

mod google_i {
    use std::collections::HashMap;

    pub fn unique_occurrences(arr: Vec<i32>) -> bool {
        let mut counter = HashMap::with_capacity(arr.len());

        // count up how many times each element occur
        for val in arr {
            let cur_count = counter.entry(val).or_insert(0_usize);
            *cur_count += 1;
        }

        let mut occurrences: Vec<&usize> = counter.values().collect();
        occurrences.sort();
        for i in 1..occurrences.len() {
            if occurrences[i - 1] == occurrences[i] {
                return false;
            }
        }
        true
    }

    pub fn min_operations(boxes: String) -> Vec<i32> {
        let mut box_locations = Vec::with_capacity(boxes.len());
        let mut min_moves = Vec::with_capacity(boxes.len());

        // Process the input to figure out the location of all the boxes
        for (idx, c) in boxes.char_indices() {
            if c == '1' {
                box_locations.push(idx as i32);
            }
        }

        for sink in 0..boxes.len() as i32 {
            let mut num_moves = 0;
            for box_location in &box_locations {
                num_moves += (sink - box_location).abs();
            }
            min_moves.push(num_moves);
        }
        min_moves
    }
}

mod google_ii {
    use std::collections::HashMap;

    /// We first calculate the total wealth of each individual and select the max
    pub fn maximum_wealth(accounts: Vec<Vec<i32>>) -> i32 {
        accounts
            .iter()
            .map(|wealth_profile| wealth_profile.iter().sum())
            .max()
            .unwrap()
    }
    struct Solution;

    impl Solution {
        pub fn count_squares(matrix: Vec<Vec<i32>>) -> i32 {
            let mut count = 0;
            for square_size in 1..=matrix.len() {
                for x_start in 0..matrix.len() {
                    for y_start in 0..matrix[0].len() {
                        if Self::square_fits((x_start, y_start), square_size, &matrix) {
                            count += Self::count_of_ones_in((x_start, y_start), square_size, &matrix);
                        }
                    }
                }
            }
            count
        }

        fn count_of_ones_in((x_start, y_start): (usize, usize), square_size: usize, matrix: &Vec<Vec<i32>>) -> i32 {
            for i in x_start..(x_start + square_size) {
                for j in y_start..(y_start + square_size) {
                    if matrix[i][j] != 1 {
                        return 0;
                    }
                }
            }
            return 1;
        }

        fn square_fits((x, y): (usize, usize), square_size: usize, matrix: &Vec<Vec<i32>>) -> bool {
            let x_fits = x + square_size <= matrix.len();
            let y_fits = y + square_size <= matrix[0].len();
            x_fits && y_fits
        }
    }

    #[test]
    fn square() {
        let v = vec![vec![0, 1, 1, 1], vec![1, 1, 1, 1], vec![0, 1, 1, 1]];
        Solution::count_squares(v);
    }

    pub fn decompress_rl_elist(nums: Vec<i32>) -> Vec<i32> {
        let mut out = Vec::new();
        for window in nums.chunks(2) {
            for _ in 0..window[0] as usize {
                out.push(window[1]);
            }
        }
        out
    }

    pub fn is_alien_sorted(words: Vec<String>, order: String) -> bool {
        // We'll first map the words in the alien language to
        // their equivalent words in our alphabet. For instance,
        // if in the alien alphabet the first character is `h` then
        // we should change all occurrences of `h` to `a`
        let mut alien_to_eng = HashMap::with_capacity(26);
        let english = "abcdefghijklmnopqrstuvwxyz";
        for (a, b) in order.chars().zip(english.chars()) {
            alien_to_eng.insert(a, b);
        }
        let mut transformed = Vec::with_capacity(words.len());
        for word in words {
            transformed.push(map_to_english(word, &alien_to_eng));
        }

        // check that all are sorted
        for tuple in transformed.windows(2) {
            if tuple[0] > tuple[1] {
                return false;
            }
        }
        true
    }

    fn map_to_english(word: String, alien_to_eng: &HashMap<char, char>) -> String {
        let mut transformed = String::with_capacity(word.len());
        for ch in word.chars() {
            transformed.push(*alien_to_eng.get(&ch).unwrap());
        }
        transformed
    }

    pub fn check_inclusion(s1: String, s2: String) -> bool {
        let (mut small, larger): (Vec<_>, Vec<_>) = (s1.as_bytes().into(), s2.as_bytes().into());
        small.sort();
        for window in larger.windows(small.len()) {
            let mut clone: Vec<_> = window.clone().into();
            clone.sort();
            if clone == small.as_slice() {
                return true;
            }
        }
        false
    }
}

mod google_iii {
    fn horner(array_form: &[u8]) -> u128 {
        let n = array_form.len();
        if n == 0 {
            return 0;
        }
        array_form[n - 1] as u128 + 10 * horner(&array_form[..n - 1])
    }

    #[test]
    fn test_horner() {
        let a = [9, 1, 4, 5, 9, 7, 5, 6];
        println!("{}", horner(&a))
    }

    pub fn merge(mut intervals: Vec<Vec<i32>>) -> Vec<Vec<i32>> {
        let mut merged = Vec::with_capacity(intervals.len());
        intervals.sort_by_key(|interval| interval[0]);
        merged.push(intervals[0].clone());

        let mut cur_idx = 1;
        while cur_idx < intervals.len() {
            let last_added = merged.last_mut().unwrap();
            let new_interval = &intervals[cur_idx];
            if last_added[1] >= new_interval[0] {
                if last_added[1] < new_interval[1] {
                    last_added[1] = new_interval[1];
                }
                cur_idx += 1;
            } else {
                merged.push(new_interval.clone());
            }
            cur_idx += 1;
        }

        merged
    }

    pub fn remove_duplicates(s: String) -> String {
        let mut res = String::new();
        let mut stack = Vec::with_capacity(s.len());
        let mut chars = s.chars();
        stack.push(chars.next().unwrap());
        for ch in s.chars() {
            let top_char = stack.last();
            match top_char {
                None => stack.push(ch),
                Some(&top) => {
                    if top == ch {
                        stack.pop();
                    } else {
                        stack.push(ch);
                    }
                }
            }
        }
        for ch in stack {
            res.push(ch);
        }
        res
    }
}

mod hrt {
    struct StackEntry {
        count: i32,
        character: char,
    }

    impl StackEntry {
        fn new(count: i32, character: char) -> Self {
            Self { count, character }
        }
    }

    pub fn remove_duplicates(s: &str, k: i32) -> String {
        let mut res = String::new();
        let mut stack: Vec<StackEntry> = Vec::with_capacity(s.len());
        let chars = s.chars();

        for ch in chars {
            let top_entry = stack.last_mut();
            match top_entry {
                None => stack.push(StackEntry::new(1, ch)),
                Some(top) => {
                    if top.character == ch {
                        top.count += 1;
                        if top.count == k {
                            stack.pop();
                        }
                    } else {
                        stack.push(StackEntry::new(1, ch));
                    }
                }
            }
        }

        for entry in stack {
            for _ in 0..entry.count as usize {
                res.push(entry.character);
            }
        }

        res
    }

    #[test]
    fn duplicate_rem() {
        remove_duplicates("deeedbbcccbdaa", 3);
    }

    pub fn min_elements(nums: Vec<i32>, limit: i32, goal: i32) -> i32 {
        let cur_sum: i32 = nums.iter().sum();
        let remainder = goal - cur_sum;
        count_num_to_add(remainder, limit)
    }

    ///
    fn count_num_to_add(remainder: i32, limit: i32) -> i32 {
        todo!()
    }
}
