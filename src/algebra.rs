use rand::{thread_rng, Rng};

/// Naïve exponentiation. On the order of O(N) where N is the value of the exponent
/// Does not deal with negative exponents for now (extending to negative exponents is
/// indeed quite easy, just divide 1 by the result before returning. We ignore that here
/// so as not to deal with traits)
pub fn naive_exp(a: i64, n: i64) -> i64 {
    let mut result = 1;
    for _ in 0..n {
        result *= a;
    }
    result
}

/// Exponentiation in O(log n). The Binary representation
/// of any integer n has exactly floor(log n) + 1 digits.
/// The idea os that we split the work, by using the binary
/// representation of the exponent. This way, we only need to do
/// log n multiplications.
pub fn binary_exp(a: i64, mut n: i64) -> i64 {
    if n == 0 {
        1
    } else {
        let mut prev_square = a;
        let mut result = a;
        while n != 0 {
            n = n >> 1;
            prev_square = prev_square * prev_square;
            if n & 1 == 1 {
                result *= prev_square;
            }
        }
        result
    }
}

fn naive_gcd(a: i64, b: i64) -> i64 {
    if b == 0 {
        a
    } else {
        naive_gcd(b, a % b)
    }
}

pub struct GCDArgs {
    a: i64,
    b: i64,
}

pub struct GCDRes {
    gcd: i64,
    x: i64,
    y: i64,
}

impl GCDArgs {
    fn new(a: i64, b: i64) -> GCDArgs {
        GCDArgs { a, b }
    }
}

impl GCDRes {
    fn new(gcd: i64, x: i64, y: i64) -> GCDRes {
        GCDRes { gcd, x, y }
    }
}
/// Extened GCD where in addition to finding the largest
/// common divisor of (a, b), it also finds coefficients
/// x, y such that gcd(a, b) = a*x + b*y
fn extended_gcd(args: GCDArgs) -> GCDRes {
    if args.b == 0 {
        GCDRes::new(args.a, 1, 0)
    } else {
        let new_args = GCDArgs::new(args.b, args.a % args.b);
        let prev_res = extended_gcd(new_args);
        let new_y = prev_res.x - prev_res.y * (args.a / args.b);
        GCDRes::new(prev_res.gcd, prev_res.y, new_y)
    }
}

/// Consider the problem of adding two n-bit binary integers
/// stored in two n-element arrays A and B. The sum of the two
/// should be stored in binary form in an (n+1) element array C
fn binary_addition(A: Vec<usize>, B: Vec<usize>) -> Vec<usize> {
    let mut C = Vec::with_capacity(A.len() + 1);
    for _ in 0..A.len() + 1 {
        C.push(0);
    }
    let mut carry = 0;
    for i in (0..A.len()).rev() {
        let pattern = (A[i], B[i], carry);
        match pattern {
            (0, 0, 0) => {
                carry = 0;
                C[i + 1] = 0
            }
            (1, 1, 1) => {
                carry = 1;
                C[i + 1] = 1;
            }
            (0, 0, 1) | (0, 1, 0) | (1, 0, 0) => {
                carry = 0;
                C[i + 1] = 1;
            }
            (0, 1, 1) | (1, 0, 1) | (1, 1, 0) => {
                carry = 1;
                C[i + 1] = 0;
            }
            (_, _, _) => panic!(pattern),
        }
    }
    C[0] = carry;
    C.to_vec()
}

/// Produces a uniform random permutation of the
/// input sequence in place and in liner time and
/// constant, O(1) space.
pub fn randomize_in_place<T>(items: &mut Vec<T>) {
    for i in 0..items.len() {
        let random_index = thread_rng().gen_range(i, items.len());
        items.swap(i, random_index);
    }
}

/// One simple way to calculate all primes <= x is to form
/// the so callsed sieve of Eratosthenes: First write down all
/// integers from to to x. Next, circle 2 marking it as prime,
/// and cross out all other multiples of 2. then repeatedly
/// circle the smallest uncircled, uncrossed number and
/// cross out its other multiples. When everything has been
/// circled or crossed out, the circled numbers are the primes.
///
/// Ref: Concrete Mathematics Page 111
fn sieve_of_eratosthenes(x: u64) -> Vec<bool> {
    let mut primes = Vec::new();
    primes.push(false); // 1 is not Prime
    for _ in 1..x {
        primes.push(true);
    }
    let sqrt = (x as f64).sqrt() as u64;
    for i in 2..=sqrt {
        let mut multiplier = 2;
        let mut composite = i * multiplier;
        while composite <= x {
            primes[(composite - 1) as usize] = false;
            multiplier += 1;
            composite = i * multiplier;
        }
    }
    primes
}

#[cfg(test)]
mod test {
    use super::{binary_addition, binary_exp, extended_gcd, naive_exp, naive_gcd, randomize_in_place, GCDArgs};

    #[test]
    fn test_randomize_in_place() {
        let mut v = vec![13, -3, -25, 20, -3, -16, -23, 18, 20, -7];
        randomize_in_place(&mut v);
        println!("{:?}", v);
    }

    #[test]
    fn test_naive_exp() {
        assert_eq!(1, naive_exp(2, 0));
        assert_eq!(8, naive_exp(2, 3));
    }

    #[test]
    fn test_binary_exp() {
        assert_eq!(1, binary_exp(2, 0));
        assert_eq!(10, binary_exp(10, 1));
        assert_eq!(8, binary_exp(2, 3));
        assert_eq!(1594323, binary_exp(3, 13));
    }
    #[test]
    fn test_naive_gcd() {
        assert_eq!(5, naive_gcd(55, 80));
    }

    #[test]
    fn test_extended_gcd() {
        let res = extended_gcd(GCDArgs::new(55, 80));
        assert_eq!(3, res.x);
        assert_eq!(-2, res.y);
        assert_eq!(5, res.gcd);
    }

    #[test]
    fn test_binary_addition() {
        let res = binary_addition(vec![1, 1, 1], vec![1, 1, 1]);
        assert_eq!(res, vec![1, 1, 1, 0]);
    }

    #[test]
    fn sieve() {
        let res = super::sieve_of_eratosthenes(10);
        assert_eq!(
            res,
            vec![false, true, true, false, true, false, true, false, false, false]
        );
    }
}
