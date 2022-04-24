use crate::sorting::{insertion_sort, partition, Range, RangeTuple};
use rand::{thread_rng, Rng};

/// We can select the maximum or minimum element from
/// a collection of n elements in time linear in the number of
/// items. An additional optimization can be used to reduce the
/// number of comparisons needed even further and to find both
/// extrema simultaneously. We could iterate over two elements at
/// a time, and keep track of two counters, one for each
/// extrema. Under this scheme, we can find the extrema using
/// 3*floor(n/2) comparisons.
fn find_extremum<T: PartialOrd>(elements: &Vec<T>, extremum_type: ExtremumType) -> &T {
    let mut extremum = &elements[0];
    for elem in elements {
        match extremum_type {
            ExtremumType::MAX => {
                if *elem > *extremum {
                    extremum = elem;
                }
            }
            ExtremumType::MIN => {
                if *elem < *extremum {
                    extremum = elem;
                }
            }
        };
    }
    extremum
}

pub enum ExtremumType {
    MIN,
    MAX,
}

/// Finding the ith order statistic in linear time hinges on our
/// ability to effectively reduce the search space. Randomized
/// select takes inspiration from quicksort in that it partitions
/// the search space using a randomly chosen pivot. Once this
/// partition is done, it is able to discard one of the portions
/// by checking if the statistic asked for i.e i can possibly
/// be contained there.
///
fn randomized_select<T: PartialOrd>(elements: &mut Vec<T>, range: Range, i: usize) -> &T {
    if range.0 < range.1 {
        let (left_range, right_range) = randomized_partition(elements, range);
        let left_range_size = (left_range.1 - left_range.0) + 1;
        if i == left_range_size - 1 {
            &elements[left_range.1]
        } else if i < left_range_size {
            let new_range = (range.0, left_range.1 - 1);
            randomized_select(elements, new_range, i)
        } else {
            randomized_select(elements, right_range, i - left_range_size)
        }
    } else {
        &elements[range.0]
    }
}

/// Partition the the sub array of `elements` starting at range.0 and ending at range.1
/// at a randomly chosen pivot. Return the ranges of the `less_than` and `greater_than`
/// partitions. Note that the pivot element is in the `less_than` range.
///
pub fn randomized_partition<T: PartialOrd>(elements: &mut Vec<T>, range: Range) -> RangeTuple {
    let pivot = thread_rng().gen_range(range.0..=range.1 + 1);
    partition(elements, pivot, range)
}

/// Randomize select only guarantees linear time performance on average.
/// To come up with a procedure that guarantees O(n) in the worst case,
/// we need to figure out a way of splitting the search space into
/// roughly equal portions at each iteration. The median of medians is
/// one such way. It works by first decomposing the input sequence into blocks
/// of size K, where k is a small number like 5. It then finds the median in
/// each block by first sorting each block and selecting the middle element.
/// Because the blocks are small, sorting takes constant time. It then recursiely
/// finds the approximate median of the medians of the blocks. The input sequence is
/// then split around this approximate median. The rest of the procedure
/// proceeds as randomized_select would.
fn median_of_medians_select<T: PartialOrd + Copy + std::fmt::Debug>(
    elements: &mut Vec<T>,
    range: Range,
    i: usize,
) -> (&T, usize) {
    if range.0 < range.1 {
        let mid = get_approximate_median(elements, range);
        let (left_range, right_range) = approximate_median_partition(elements, mid, range);
        let left_range_size = (left_range.1 - left_range.0) + 1;
        if i == left_range_size - 1 {
            (&elements[left_range.1], range.1)
        } else if i < left_range_size {
            let new_range = (range.0, left_range.1 - 1);
            median_of_medians_select(elements, new_range, i)
        } else {
            median_of_medians_select(elements, right_range, i - left_range_size)
        }
    } else {
        (&elements[range.0], range.0)
    }
}

static MEDIAN_BLOCK_SIZE: usize = 5;
fn get_approximate_median<T: PartialOrd + Copy + std::fmt::Debug>(elements: &mut Vec<T>, range: Range) -> T {
    let mut medians = Vec::new();
    let mut range_start = 0;
    let mut range_end = MEDIAN_BLOCK_SIZE - 1;
    while range_end < range.1 {
        insertion_sort(elements, (range_start, range_end));
        let mid = ((range_end - range_start) + 1) / 2; // The lower median
        medians.push(elements[range_start + mid]);
        range_start += MEDIAN_BLOCK_SIZE;
        range_end += MEDIAN_BLOCK_SIZE;
    }
    if ((range.1 - range.0) + 1) % MEDIAN_BLOCK_SIZE > 0 {
        insertion_sort(elements, (range_start, range.1));
        let mid = ((range.1 - range_start) + 1) / 2; // The lower median
        medians.push(elements[range_start + mid]);
    }
    let mid = (medians.len() + 1) / 2;
    let medians_end = medians.len() - 1;
    println!("{:?}", medians);
    let (median_of_medians, _) = median_of_medians_select(&mut medians, (0, medians_end), mid);
    println!("\t {:?}", median_of_medians);
    *median_of_medians
}

fn approximate_median_partition<T: PartialOrd>(a: &mut Vec<T>, mid: T, range: Range) -> RangeTuple {
    let (mut lt, mut eq, mut gt) = (Vec::new(), Vec::new(), Vec::new());
    for i in range.0..=range.1 {
        if a[i] == mid {
            eq.push(&a[i]);
        } else if a[i] > mid {
            gt.push(&a[i]);
        } else {
            lt.push(&a[i]);
        }
    }

    let mut mid_idx = range.0;
    if lt.len() >= 1 {
        mid_idx += lt.len() - 1
    }
    if eq.len() >= 1 {
        mid_idx += eq.len() - 1;
    }
    return ((range.0, mid_idx), (mid_idx + 1, range.1));
}

#[cfg(test)]
mod test {
    #[test]
    fn test_find_extremum() {
        let v = vec![13, 19, 9, 5, 12, 8, 7, 4, 21, 2, 6, 11];
        assert_eq!(21, *super::find_extremum(&v, super::ExtremumType::MAX));
        assert_eq!(2, *super::find_extremum(&v, super::ExtremumType::MIN));
    }

    #[test]
    fn test_randomized_select() {
        let mut v = vec![13, 19, 9, 5, 12, 8, 7, 4, 21, 2, 6, 11];
        let range_end = v.len() - 1;
        assert_eq!(2, *super::randomized_select(&mut v, (0, range_end), 0));
        assert_eq!(4, *super::randomized_select(&mut v, (0, range_end), 1));
        assert_eq!(21, *super::randomized_select(&mut v, (0, range_end), range_end));
        assert_eq!(9, *super::randomized_select(&mut v, (0, range_end), 6));
        assert_eq!(8, *super::randomized_select(&mut v, (0, range_end), 5));
    }

    #[test]
    fn test_median_of_medians() {
        let mut v = vec![13, 19, 9, 5, 12, 8, 7, 4, 21, 2, 6, 11];
        let range_end = v.len() - 1;
        assert_eq!(2, *super::median_of_medians_select(&mut v, (0, range_end), 0).0);
        assert_eq!(4, *super::median_of_medians_select(&mut v, (0, range_end), 1).0);
        assert_eq!(
            21,
            *super::median_of_medians_select(&mut v, (0, range_end), range_end).0
        );
        assert_eq!(9, *super::median_of_medians_select(&mut v, (0, range_end), 6).0);
        assert_eq!(8, *super::median_of_medians_select(&mut v, (0, range_end), 5).0);
    }
}
