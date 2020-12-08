use rand::{thread_rng, Rng};

/// Insertion sort is a procedure that  sorts
/// a sequence of items incrementaly. It works
/// by maintaining two portions of the input sequence
/// One portion in which all entries are sorted and
/// another that is yet to be ordered. The sorted
/// portion begins with only one item, and is incremented
/// iteratively. j, the procedure below, points to the end of the sorted section,
/// while i points to the start of the unsorted portion. To find the right position
/// for the first unsorted element A[i], we iterate backwards over the sorted section
/// until we find a value that is larger than A[i]
pub fn insertion_sort<T: PartialOrd + Copy + std::fmt::Debug>(items: &mut Vec<T>, range: Range) {
    let mut sorted_end = range.0;
    let mut unsorted_start = range.0 + 1;
    while unsorted_start <= range.1 {
        let unsorted_candidate = items[unsorted_start];
        let mut i = sorted_end;
        while unsorted_candidate < items[i] && i > range.0 {
            items.swap(i, i + 1);
            i -= 1;
        }
        if i == range.0 && unsorted_candidate < items[i] {
            items.swap(i, i + 1);
        }
        sorted_end += 1;
        unsorted_start += 1;
    }
}

/// How many slices of pizza can a person obtain by making
/// n straight cuts with a pizza knife? More academically,
/// What is the maximum number L_n of regions defined by n
/// lines in a plane?
/// We have this recurrence:
///     L(0) = 1; L(n) = L(n-1) + n
/// Note that this problem can be solved in O(1)
/// To find out how, we simply expand the recurrence above.
/// After doing so, we'll realize that L(n) = (n(n-1)/2) + 1
/// Solving the recurrence involves using Gauss' trick in evaluating
/// S(n): the sum of integers up to n
fn intersections(n: i64) -> i64 {
    let mut num_slices = 1;
    for i in 1..n + 1 {
        num_slices = num_slices + i;
    }
    num_slices
}

enum Primality {
    Even,
    Odd,
}

fn primality(i: i64) -> Primality {
    if i % 2 == 0 {
        Primality::Even
    } else {
        Primality::Odd
    }
}

/// If n people are seated in a circle, and we go around killing
/// every other person until only one survivor is left, can
/// we find the starting position, J(n) of the survivor?
///
/// This problem can be solved in a number of ways. First, after
/// playing around with a few examples, we can come up with this
/// recurrence relations: J(1) = 1; J(2n) = 2J(n) - 1; J(2n + 1) = 2J(n) + 1
/// Further manipulation of the recurrence (along with key insights such as
/// representing n as 2^m + l) gives us a second method: J(n) = J(2^m + l) = 2l + 1
/// In this method, 2^m is the highest power of 2 less than n. The third method
/// can be found by writing n is its binary represntation.
/// Doing so, and after a few manipulations, you realize that J(n) is simply the
/// cyclic left shift by 1 of n
fn josephus_problem(n: i64) -> i64 {
    if n == 1 {
        1
    } else {
        match primality(n) {
            Primality::Even => 2 * josephus_problem(n / 2) - 1,
            Primality::Odd => 2 * josephus_problem((n - 1) / 2) + 1,
        }
    }
}

/// Merge sort is a divide and conquer algorithms that
/// sorts a sequence in O(n log n). It works by first
/// splitting the input sequence into two subarrays. It
/// then recursively applies itself on the subarrays
/// after which it combines the results of the two calls
/// in a merge step.
fn merge_sort<T: PartialOrd + Copy + std::fmt::Debug>(a: &mut Vec<T>, begin: usize, end: usize) {
    if begin < end {
        let mid = (begin + end) / 2;
        merge_sort(a, begin, mid);
        merge_sort(a, mid + 1, end);
        merge(a, begin, mid, end);
    }
}

fn merge<T>(a: &mut Vec<T>, begin: usize, mid: usize, end: usize)
where
    T: PartialOrd + Copy + std::fmt::Debug,
{
    let mut left_array = Vec::with_capacity((mid - begin) + 1);
    let mut right_array = Vec::with_capacity((end - mid) + 1);
    for i in begin..mid + 1 {
        left_array.push(a[i]);
    }
    for i in mid + 1..end + 1 {
        right_array.push(a[i]);
    }

    let mut left_index = 0;
    let mut right_index = 0;
    for main_index in begin..end + 1 {
        // How do we handle the times when we exhaust
        // one array before the other is finished? CLRS circumvents this by
        // having both the left and right arrays hold an INF value. This
        // way, when one of the arrays is 'exhausted', its index lingers on
        // the INF value. Here, we instead have a pre-check step that ensures that
        // the indexes of our two arrays are always well defined
        if left_index >= left_array.len() {
            a[main_index] = right_array[right_index];
            right_index += 1;
            continue;
        }

        if right_index >= right_array.len() {
            a[main_index] = left_array[left_index];
            left_index += 1;
            continue;
        }

        if left_array[left_index] <= right_array[right_index] {
            a[main_index] = left_array[left_index];
            left_index += 1;
        } else {
            a[main_index] = right_array[right_index];
            right_index += 1;
        }
    }
}

type MaxSubArrayResult = (usize, usize, i64);

/// Given sequence of n item <a1, a2, a3, a4 ..., an>, return the
/// starting and the ending index of the subarray with the largest
/// sum in the sequence. We assume that the smallest sequence size
/// is that containing a single item
fn maximum_subarray(a: &Vec<i64>, array_begin: usize, array_end: usize) -> MaxSubArrayResult {
    if array_begin < array_end {
        let midpoint = (array_begin + array_end) / 2;
        let (array_begin_l, array_end_l, max_l) = maximum_subarray(a, midpoint + 1, array_end);
        let (array_begin_r, array_end_r, max_r) = maximum_subarray(a, array_begin, midpoint);
        let (array_begin_c, array_end_c, max_c) =
            max_crossing_subarray(a, array_begin, array_end, midpoint);
        if max_l >= max_r && max_l >= max_c {
            (array_begin_l, array_end_l, max_l)
        } else if max_r >= max_l && max_r >= max_c {
            (array_begin_r, array_end_r, max_r)
        } else {
            (array_begin_c, array_end_c, max_c)
        }
    } else {
        (array_begin, array_end, a[array_begin])
    }
}

fn max_crossing_subarray(
    a: &Vec<i64>,
    array_begin: usize,
    array_end: usize,
    mid: usize,
) -> MaxSubArrayResult {
    let mut left_sum = i64::MIN;
    let mut sum = 0;
    let mut max_left = mid;
    for i in (array_begin..mid + 1).rev() {
        sum += a[i];
        if sum > left_sum {
            left_sum = sum;
            max_left = i;
        }
    }

    let mut right_sum = i64::MIN;
    sum = 0;
    let mut max_right = mid;
    for j in mid + 1..array_end + 1 {
        sum += a[j];
        if sum > right_sum {
            right_sum = sum;
            max_right = j;
        }
    }
    (max_left, max_right, left_sum + right_sum)
}

fn maximum_subarray_kadane(a: &Vec<i64>) -> MaxSubArrayResult {
    let mut best_sum = i64::MIN;
    let (mut best_start, mut best_end) = (0, 0);
    let (mut cur_sum, mut cur_start) = (0, 0);

    let mut best_sums = Vec::new();
    let mut cur_sums = Vec::new();
    for (cur_end, cur_val) in a.iter().enumerate() {
        // Why is it that when the sum of the sub_array ending at cur_end-1
        // falls below or to 0, we begin a new sub array? That seems to
        // be the key insight to this procedure
        if cur_sum <= 0 {
            cur_sum = *cur_val;
            cur_start = cur_end;
        } else {
            cur_sum += cur_val;
        }

        if cur_sum > best_sum {
            best_sum = cur_sum;
            best_start = cur_start;
            best_end = cur_end;
        }
        best_sums.push(best_sum);
        cur_sums.push(cur_sum);
    }
    println!("{:?}", a);
    println!("{:?}", cur_sums);
    println!("{:?}", best_sums);
    (best_start, best_end, best_sum)
}

/// Range.0 = low, Range.1 = high
pub type Range = (usize, usize);
pub type RangeTuple = (Range, Range);

/// Randomized quick sort
pub fn quick_sort<T: PartialEq + PartialOrd + std::fmt::Debug>(a: &mut Vec<T>, range: Range) {
    if range.0 < range.1 {
        let pivot = thread_rng().gen_range(range.0, range.1 + 1);
        let (left_range, right_range) = partition(a, pivot, range);
        println!("{:?}, {:?}", left_range, right_range);
        println!("\t {:?}", a);
        quick_sort(a, left_range);
        quick_sort(a, right_range);
    }
}

pub fn partition<T: PartialEq + PartialOrd>(
    a: &mut Vec<T>,
    pivot: usize,
    range: Range,
) -> RangeTuple {
    a.swap(pivot, range.1); // Pivot is now at the end of the current sub_array
    let mut less_than_end = range.0;
    for j in range.0..range.1 {
        if a[j] <= a[range.1] {
            a.swap(less_than_end, j);
            less_than_end += 1;
        }
    }
    a.swap(less_than_end, range.1);
    ((range.0, less_than_end), (less_than_end + 1, range.1))
}
/// Counting Sort is able to sort in O(n) by assuming that the items
/// to be sorted are integers between 0 and some upper bound k that is: 0 <= i <= k
/// To determine the location of an element, i,  we first figure out how many elements are less that i
/// and then use this to place the i at its appropriate position.
/// Counting Sort is a stable sorting procedure, meaning that if two items a, b, a=b
/// appear is a certain order in the input sequence, that order is preserved in the sorted  sequence.
///
fn counting_sort(a: Vec<u64>, k: u64) -> Vec<u64> {
    let mut sorted = Vec::new();
    let mut counters = Vec::new();
    for _ in 0..k {
        counters.push(0);
    }
    for val in &a {
        sorted.push(0);
        counters[*val as usize] += 1;
    }
    for i in 1..k as usize {
        counters[i] = counters[i] + counters[i - 1];
    }
    println!("{:?}", counters);
    for j in (0..a.len()).rev() {
        let position = counters[a[j] as usize] - 1;
        sorted[position] = a[j];
        counters[a[j] as usize] -= 1;
    }
    sorted
}

#[derive(Debug, Clone, Copy, Default)]
struct Date {
    /// The max value is 31
    day: u32,

    /// the max value is 12
    month: u32,

    /// This will be used to deternime the size of the bucket used
    /// to sort the date field. Currently, we'll use earliest=1900
    /// This will
    earliest_year: u32,

    /// The max value is 200. This means that we can handle
    /// dates between 1900 and 2100. This could be made
    /// tighter by setting the approporiate value of earliest year
    year: u32,
}

pub enum DateField {
    Day,
    Month,
    Year,
}

fn map_date_to_slot(d: &Date, on: &DateField) -> usize {
    match on {
        DateField::Day => d.day as usize,
        DateField::Month => d.month as usize,
        DateField::Year => (d.year - d.earliest_year) as usize,
    }
}

fn radix_sort(dates: Vec<Date>) -> Vec<Date> {
    sort_by_year(sort_by_month(sort_by_day(dates)))
}

fn sort_by_month(a: Vec<Date>) -> Vec<Date> {
    counting_sort_dates(a, DateField::Month)
}

fn sort_by_year(a: Vec<Date>) -> Vec<Date> {
    counting_sort_dates(a, DateField::Year)
}

fn sort_by_day(a: Vec<Date>) -> Vec<Date> {
    counting_sort_dates(a, DateField::Day)
}

fn counting_sort_dates(a: Vec<Date>, on: DateField) -> Vec<Date> {
    let mut sorted = Vec::new();
    let mut counters = Vec::new();
    let k = match on {
        DateField::Day => 31,
        DateField::Month => 12,
        DateField::Year => 200,
    };
    for _ in 0..k {
        counters.push(0);
    }
    for val in &a {
        sorted.push(Date::default());
        counters[map_date_to_slot(val, &on)] += 1;
    }
    for i in 1..k as usize {
        counters[i] = counters[i] + counters[i - 1];
    }
    for j in (0..a.len()).rev() {
        let position = counters[map_date_to_slot(&a[j], &on)] - 1;
        sorted[position] = a[j];
        counters[map_date_to_slot(&a[j], &on)] -= 1;
    }
    sorted
}

#[cfg(test)]
mod test {
    use super::{
        counting_sort, insertion_sort, intersections, josephus_problem, maximum_subarray,
        maximum_subarray_kadane, merge_sort, quick_sort, radix_sort,
    };

    #[test]
    fn test_max_subarray() {
        let v = vec![
            13, -3, -25, 20, -3, -16, -23, 18, 20, -7, 12, -5, -22, 15, -4, 7,
        ];
        let r = maximum_subarray(&v, 0, v.len() - 1);
        assert_eq!(r.0, 7);
        assert_eq!(r.1, 10);
        assert_eq!(r.2, 43);
    }

    #[test]
    fn test_kadane() {
        let v = vec![
            13, -3, -25, 20, -3, -16, -23, 18, 20, -7, 12, -5, -22, 15, -4, 7,
        ];
        let r = maximum_subarray_kadane(&v);
        assert_eq!(r.0, 7);
        assert_eq!(r.1, 10);
        assert_eq!(r.2, 43);
    }

    #[test]
    fn test_insertion_sort() {
        let mut v = vec![4, 3, 5, 8, 9, 1, 0];
        let end = v.len() - 1;
        insertion_sort(&mut v, (0, end));
        assert_eq!(vec![0, 1, 3, 4, 5, 8, 9], v);

        let mut v = vec![13, 19, 9, 5, 12, 8, 7, 4, 21, 2, 6, 11];
        let range_end = v.len() - 1;
        insertion_sort(&mut v, (0, range_end));
        assert_eq!(vec![2, 4, 5, 6, 7, 8, 9, 11, 12, 13, 19, 21], v);
    }

    #[test]
    fn test_josephus_problem() {
        assert_eq!(1, josephus_problem(1));
        assert_eq!(1, josephus_problem(2));
        assert_eq!(1, josephus_problem(16));
        assert_eq!(73, josephus_problem(100));
    }

    #[test]
    fn test_intersections() {
        assert_eq!(1, intersections(0));
        assert_eq!(2, intersections(1));
        assert_eq!(4, intersections(2));
    }

    #[test]
    fn test_merge_sort() {
        let mut v = vec![4, 3, 5, 8, 9, 1, 0];
        let end_idx = v.len() - 1;
        merge_sort(&mut v, 0, end_idx);
        assert_eq!(vec![0, 1, 3, 4, 5, 8, 9], v);
    }

    #[test]
    fn test_quick_sort() {
        let mut v = vec![4, 3, 5, 8, 9, 1, 0];
        let end_idx = v.len() - 1;
        quick_sort(&mut v, (0, end_idx));
        assert_eq!(vec![0, 1, 3, 4, 5, 8, 9], v);
    }
    #[test]
    fn test_counting_sort() {
        let v = vec![4, 3, 5, 8, 9, 1, 2, 0];
        let v = counting_sort(v, 10);
        assert_eq!(vec![0, 1, 2, 3, 4, 5, 8, 9], v);
    }

    #[test]
    // fn radix_sort(mut dates: Vec<Date>) -> Vec<Date>
    fn test_radix_sort() {
        use rand::{thread_rng, Rng};
        let mut dates = Vec::with_capacity(10);
        let earliest_year = 1900;
        for _ in 0..10 {
            let day = thread_rng().gen_range(1, 31);
            let month = thread_rng().gen_range(1, 12);
            let year = thread_rng().gen_range(earliest_year, 2100);
            dates.push(super::Date {
                earliest_year,
                day,
                month,
                year,
            })
        }
        let sorted_dates = radix_sort(dates);
        for d in sorted_dates {
            println!("{}-{}-{}", d.year, d.month, d.day)
        }
    }
}
