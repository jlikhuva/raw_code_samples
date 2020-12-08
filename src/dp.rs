//! Dynamic Programming is a discrete optimization technique. In this algorithmic realm,
//! the goal is not to find the one correct answer to a given problem, but to find the
//! one answer — among many possible answers, that maximizes or minimizes some predefined cost.
//! Note that in continuous optimization, we usually solve such problems using differential calculus.
//! DP applies to problems that exhibit optimal substructure, that is, an optimal solution to the
//! problem contains within it and have the problems have overlapping subproblems.
//! When faced with such problems, other techniques, such as DnC, do more work
//! than is necessary, leading to extremely poor runtimes. DP on the other hand
//! is able to run in reasonable time by ensuring that each subproblem is only
//! ever solved once. It achieves this either by ordering the subproblems such
//! that all subproblems are solved before problems that depend on them, or by
//! keeping a cache (a memo) of all the solutions to subproblems and reusing them later on.
//!
//! DP algorithms leverage optimal substructure to build an optimal solution to the given macro problem
//! by combining optimal solutions to micro subproblems. The way it usually goes is: at every timestep
//! during the algorithms execution, we are faced with a choice that we have to make. Making this choice
//! as the effect of splitting our problem space into two or more subproblems with a reduced problem space.
//! Our goal is then to figure out which choice leads to an optimal solution to the global problem.
//! To figure this out, DP algorithms examine all possible choices and select the best one. The process
//! of examining each choice involves finding optimal solutions to subproblems that result from making that choice.
//! Thus, the runtime of DP algorithms is often a product of 2 factors: the number of choices one has to examine
//! and the number of subproblems that result from each choice.
//!
//! One additional thing to note is that DP algorithms, in their vanilla form,
//! simply compute the value of an optimal solution. To recover the optimal solution itself,
//! we have to augment them with by adding additional storage space to track the choices
//! associated with optimal values to subproblems.

/// The rod cutting operation returns the maximum revenue along with
/// the positions to cut in order to achieve that revenue.
pub type RodCuttingResult = anyhow::Result<u32>;

pub fn rod_cutting(prices: Vec<u32>, rod_len: u32) -> RodCuttingResult {
    let mut optimal_rev = Vec::new();
    optimal_rev.push(0);
    for j in 0..rod_len {
        let mut max_rev = u32::MIN;
        // Given a rod of length j, we are faced with a choice -- where should we cut?
        // We solve this by trying out all the j - 1 places we could cut. Each cut at each
        // of point, i, gives us a piece of length i and another one of length j - i.
        // Note that this formulation also the captures the option of not cutting when
        // i == j.
        for i in 1..=j {
            let rev_from_cut_i = prices[i as usize] + optimal_rev[(j - i) as usize];
            if max_rev < rev_from_cut_i {
                max_rev = rev_from_cut_i;
            }
        }
        optimal_rev.push(max_rev);
    }
    Ok(*optimal_rev.last().unwrap())
}

/// The dimesions of some matrix
#[derive(Debug)]
pub struct MatrixDims {
    ncols: u32,
    nrows: u32,
}

impl MatrixDims {
    pub fn new(nrows: u32, ncols: u32) -> Self {
        MatrixDims { nrows, ncols }
    }
}

/// Determine a parenthesization of the matrices that minimizes the
/// number of scalar multiplications needed to compute the product.
pub fn maxtrix_chain_parenthesization(dims: Vec<MatrixDims>) -> Matrix<u32> {
    let n = dims.len();
    let mut cost_matrix = vec![];

    // We begin by populating our bottom up table with initial costs.
    // At this point, all we know is that it costs nothing to
    // parenthesize a singleton matrix sequence (i == j). All
    // other costs are initialized with infinity (here represented
    // by u32::MAX which is 2^32 + 2^31 + ... 1)
    for i in 0..n {
        let mut v = vec![];
        for j in 0..n {
            if i == j {
                v.push(0);
            } else {
                v.push(u32::MAX);
            }
        }
        cost_matrix.push(v);
    }

    // Here, we consider all possible starting ane ending indexes i, and j
    // We ignore the case where i == j, because we already know the answers
    // for that to be 0.
    for l in 1..n {
        for i in 0..n - l + 1 {
            // We consider all the places, k, we could split the matrix
            // sunsequence starting at i and ending at j
            let j = i + l - 1;
            for k in i..j {
                let cost_to_combine = dims[i].nrows * dims[k].nrows * dims[j].ncols;
                let cur_split_cost = cost_matrix[i][k] + cost_matrix[k + 1][j] + cost_to_combine;
                if cur_split_cost < cost_matrix[i][j] {
                    cost_matrix[i][j] = cur_split_cost;
                }
            }
        }
    }
    Matrix(cost_matrix)
}

type Sequence<T> = Vec<T>;
pub fn longest_common_subsequence<T>(x: Sequence<T>, y: Sequence<T>) -> Sequence<T> {
    todo!()
}

/// We may choose from among six transformation operations
#[derive(Debug)]
pub enum EditAction {
    /// Copy a character from x to y.
    Copy(usize, usize),

    /// character from x by another character c
    Replace(usize, usize),

    /// a character from x
    Erase(usize),

    /// the character c into z by setting ´z\[j\] = c and then incrementing j,
    /// but leaving i alone. This operation examines no characters of x.
    Insert(usize),

    /// exchange the next two characters by copying them from x to z -- the output sequence
    /// but in the opposite order;
    Exchange(usize, usize),

    /// This operation examines all characters in x that have not
    /// yet been examined. This operation, if performed,
    /// must be the final operation
    Kill,
}

type EditDistanceResult = anyhow::Result<Vec<EditAction>>;

/// In order to transform one source string of text x to a target string y,
/// we can perform various transformation operations. Our goal is, given x and y,
/// to produce a series of transformations that change x to y. We use an array
/// assumed to be large enough to hold all the characters it will need—to hold
/// the intermediate results.
///
/// impl matrix
pub fn edit_distance<T>(x: Sequence<T>, y: Sequence<T>) -> EditDistanceResult {
    todo!()
}

pub fn optimal_bst() {
    todo!()
}

pub fn longest_palindomic_sequence() {
    todo!()
}

pub fn breaking_a_string() {
    todo!()
}

pub fn inventory_planning() {
    todo!()
}

pub fn investment_strategy() {
    todo!()
}

type KnapsackResult = anyhow::Result<Vec<KnapsackItem>>;
/// KnapsackItem(value, weight)
#[derive(Debug)]
pub struct KnapsackItem(u32, u32);

pub fn zero_one_knapsack(items: &mut [KnapsackItem], n: u32) -> KnapsackResult {
    todo!()
}

pub fn moneyball() {
    todo!()
}

pub fn viterbi() {
    todo!()
}

pub fn bitonic_tsp() {
    todo!()
}

pub fn longest_path_dag() {
    todo!()
}

pub fn printing_neatly() {
    todo!()
}

pub fn needleman_wunsch() {
    todo!()
}

use std::fmt;
/// A matrix that knows how to print itself in
/// a pretty fashion.
#[derive(Debug)]
pub struct Matrix<T: fmt::Display>(Vec<Vec<T>>);

impl<T: fmt::Display> fmt::Display for Matrix<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut string_seq = String::new();
        for v in &self.0 {
            string_seq.push('|');
            for val in v {
                string_seq.extend(val.to_string().chars().into_iter());
                string_seq.push(' ');
            }
            string_seq.push('|');
            string_seq.push('\n')
        }
        write!(f, "{}", string_seq)
    }
}

#[cfg(test)]
mod test {
    use super::{maxtrix_chain_parenthesization, rod_cutting, Matrix, MatrixDims as dims};
    #[test]
    fn test_matrix() {
        let v = vec![vec![1, 2, 3, 4], vec![4, 1, 6, 8], vec![9, 12, 12, 9]];
        let m = Matrix(v);
        println!("{}", m);
    }

    #[test]
    fn test_rod_cutting() {
        let prices = vec![2, 4, 1, 2, 3, 9, 4];
        let res = rod_cutting(prices, 6);
        println!("{:?}", res)
    }

    #[test]
    fn test_matrix_parenthesization() {
        let m = vec![dims::new(10, 100), dims::new(100, 5), dims::new(5, 50)];
        let res = maxtrix_chain_parenthesization(m);
        println!("{}", res)
    }
}
