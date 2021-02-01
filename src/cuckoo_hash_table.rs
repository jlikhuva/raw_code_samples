//! Cuckoo hashing is yet another open addressing scheme.
//! In this scheme, we maintain **K** hash tables. Suppose K = 2,
//! in that case, we'd have two tables H_1 and H_2
//! with corresponding universal hash functions h_1 and h_2.
//! Each item ~x~ from our universe can be at two positions, one in each table.
//! To search for an item, we first apply the h1 and use the value it gives us
//! to index H1 If the item we're looking for is not there, we apply h2
//! and look in H2. If the item is not at both positions, then we conclude t
//! hat the item is not in the hash table. Therefore, search is `O(1)` since
//! we only need to look at two slots at most. Similary deletion from a
//! cuckoo hash table is `O(1)` since it relies on search.
//! Insertion into a cuckoo hash table is interesting (it's also how this method gets its name).  
//! To insert `x`, we begin by trying to place it in `H_1` at index `i = h_1(x)`.
//! if there's no occupant at `i`, we place `x` there and we are done.
//! If `H_1[i]` is occupied by another key, call it `y`,  we evict  `y`,
//! place `x` there and try inserting `y` in `H_1`. Note that this is exactly
//! as if we're inserting `y` into the hash table, only that we start by examining `H_2`.
//! This is a nice recursive structure that lends itself well to such elegant implementation.
//! It is possible that inserting $y$ in `H_2` sill result in another eviction.
//! When that happens, we continues evicting keys and ping-ponging between
//! the two tables until we stabilize. The act of pushing current occupants out of
//! their slots is similar to the behavior of cuckoo birds, hence the name.
//! It is possible for an insertion to fail. This happens is we never
//! stabilize out of our ping-pong loop. When that happens, we rehash everything with
//! two new hash functions. We may have to rehash multiple times. For the runtime and
//! correctness analysis of cuckoo hashing,
//! check out [these slides from Keith Schwarz]
//! (http://web.stanford.edu/class/archive/cs/cs166/cs166.1196/lectures/13/Small13.pdf).
//! For now, it suffices to say that the expected cost of an insertion into a cuckoo hash table is `O(1)`
//! and that the expected number of rehashes for any insertion is `O(1/m^2)`. Therefore,
//! cuckoo hash maps give us worst case constant lookups and insertion and expected, amortized constant insertion
//!
//! Note that some documentation and doctests of the fucntions is heavily borrowed from std::collections::HashMaps's
//! documentation. This serves two main pursposes. First, I was aiming for API parity so it helps
//! to use the same doctests. Second,

use fxhash::FxHashMap;
use std::{collections::hash_map::RandomState, hash::Hash};

// const MAX_EVICTIONS: usize = 500;
// use raw arrays -- how to grow them?
// use vec.
const NUM_TABLES: usize = 4;

#[derive(Debug)]
pub struct CuckooHashMap<K, V: Hash> {
    cuckoo_table: RawCuckooTable<K, V>,
}
impl<K, V: Hash> CuckooHashMap<K, V> {
    /// Creates a new, empty `CuckooHashMap`.
    ///
    /// # Examples
    ///
    /// ```
    /// use crate::cuckoo_hash_table::CuckooHashMap;
    /// let mut map: CuckooHashMap<&str, usize> = CuckooHashMap::new();
    /// ```
    pub fn new() -> CuckooHashMap<K, V> {
        todo!()
    }

    /// Creates a new, empty `CuckooHashmap` with the specified capacity
    /// As with `std::collections::HashMap`, the map will not allocate if
    /// `capacity == 0`.
    ///
    /// # Examples
    ///
    /// ```
    /// use crate::cuckoo_hash_table::CuckooHashMap;
    /// let mut map: CuckooHashMap<&str, usize> = CuckooHashMap::with_capacity(10);
    /// ```
    pub fn with_capacity(capacity: usize) -> CuckooHashMap<K, V> {
        todo!()
    }

    /// Returns the number of elements the map can hold without reallocating.
    /// This number includes the capacity of the overflow hash table
    ///
    /// # Example
    /// ```
    /// use crate::cuckoo_hash_table::CuckooHashMap;
    /// let mut map: CuckooHashMap<&str, usize> = CuckooHashMap::with_capacity(10);
    /// assert_eq!(map.capacity() >= 10)
    /// ```
    pub fn capacity() -> usize {
        todo!()
    }

    /// As with the std-lib API, this method produces
    /// an iterator visiting all keys in arbitrary order.
    /// The iterator element type is `&'a K`.
    ///
    /// # Examples
    ///
    /// ```
    /// use crate::cuckoo_hash_table::CuckooHashMap;
    /// ```
    pub fn keys(&self) {
        todo!()
    }

    /// An iterator visiting all values in arbitrary order.
    /// The iterator element type is `&'a V`.
    ///
    /// # Examples
    ///
    /// ```
    /// use crate::cuckoo_hash_table::CuckooHashMap;
    /// ```
    pub fn values(&self) {
        todo!()
    }

    /// An iterator visiting all values mutably in arbitrary order.
    /// The iterator element type is `&'a mut V`.
    ///
    /// # Examples
    ///
    /// ```
    /// use crate::cuckoo_hash_table::CuckooHashMap;
    /// ```
    pub fn values_mut(&mut self) {
        todo!()
    }

    /// An iterator visiting all key-value pairs in arbitrary order.
    /// The iterator element type is `(&'a K, &'a V)`.
    ///
    /// # Examples
    ///
    /// ```
    /// use crate::cuckoo_hash_table::CuckooHashMap;
    ///
    /// let mut map = CuckooHashMap::new();
    /// map.insert("a", 1);
    /// map.insert("b", 2);
    /// map.insert("c", 3);
    ///
    /// for (key, val) in map.iter() {
    ///     println!("key: {} val: {}", key, val);
    /// }
    /// ```
    pub fn iter(&self) {
        todo!()
    }

    /// An iterator visiting all key-value pairs in arbitrary order,
    /// with mutable references to the values.
    /// The iterator element type is `(&'a K, &'a mut V)`.
    ///
    /// # Examples
    ///
    /// ```
    /// use crate::cuckoo_hash_table::CuckooHashMap;
    ///
    /// let mut map = CuckooHashMap::new();
    /// map.insert("a", 1);
    /// map.insert("b", 2);
    /// map.insert("c", 3);
    ///
    /// // Update all values
    /// for (_, val) in map.iter_mut() {
    ///     *val *= 2;
    /// }
    ///
    /// for (key, val) in &map {
    ///     println!("key: {} val: {}", key, val);
    /// }
    /// ```
    pub fn iter_mut(&mut self) {
        todo!()
    }

    /// Returns the number of elements in the map.
    ///
    /// # Examples
    ///
    /// ```
    /// use crate::cuckoo_hash_table::CuckooHashMap;
    ///
    /// let mut a = CuckooHashMap::new();
    /// assert_eq!(a.len(), 0);
    /// a.insert(1, "a");
    /// assert_eq!(a.len(), 1);
    /// ```
    pub fn len(&self) -> usize {
        todo!()
    }

    /// Returns `true` if the map contains no elements.
    ///
    /// # Examples
    ///
    /// ```
    /// use crate::cuckoo_hash_table::CuckooHashMap;
    ///
    /// let mut a = CuckooHashMap::new();
    /// assert!(a.is_empty());
    /// a.insert(1, "a");
    /// assert!(!a.is_empty());
    /// ```
    #[inline]
    pub fn is_empty(&self) -> bool {
        todo!()
    }

    /// Clears the map, returning all key-value pairs as an iterator. Keeps the
    /// allocated memory for reuse.
    ///
    /// # Examples
    ///
    /// ```
    /// use crate::cuckoo_hash_table::CuckooHashMap;
    ///
    /// let mut a = CuckooHashMap::new();
    /// a.insert(1, "a");
    /// a.insert(2, "b");
    ///
    /// for (k, v) in a.drain().take(1) {
    ///     assert!(k == 1 || k == 2);
    ///     assert!(v == "a" || v == "b");
    /// }
    ///
    /// assert!(a.is_empty());
    /// ```
    #[inline]
    pub fn drain(&mut self) {
        todo!()
    }

    /// Creates an iterator which uses a closure to determine if an element should be removed.
    ///
    /// If the closure returns true, the element is removed from the map and yielded.
    /// If the closure returns false, or panics, the element remains in the map and will not be
    /// yielded.
    ///
    /// Note that `drain_filter` lets you mutate every value in the filter closure, regardless of
    /// whether you choose to keep or remove it.
    ///
    /// If the iterator is only partially consumed or not consumed at all, each of the remaining
    /// elements will still be subjected to the closure and removed and dropped if it returns true.
    ///
    /// It is unspecified how many more elements will be subjected to the closure
    /// if a panic occurs in the closure, or a panic occurs while dropping an element,
    /// or if the `DrainFilter` value is leaked.
    ///
    /// # Examples
    ///
    /// Splitting a map into even and odd keys, reusing the original map:
    ///
    /// ```
    /// use crate::cuckoo_hash_table::CuckooHashMap;
    ///
    /// let mut map: CuckooHashMap<i32, i32> = (0..8).map(|x| (x, x)).collect();
    /// let drained: CuckooHashMap<i32, i32> = map.drain_filter(|k, _v| k % 2 == 0).collect();
    ///
    /// let mut evens = drained.keys().copied().collect::<Vec<_>>();
    /// let mut odds = map.keys().copied().collect::<Vec<_>>();
    /// evens.sort();
    /// odds.sort();
    ///
    /// assert_eq!(evens, vec![0, 2, 4, 6]);
    /// assert_eq!(odds, vec![1, 3, 5, 7]);
    /// ```
    #[inline]
    pub fn drain_filter<F>(&mut self, pred: F) {
        todo!()
    }

    /// Clears the map, removing all key-value pairs. Keeps the allocated memory
    /// for reuse.
    ///
    /// # Examples
    ///
    /// ```
    /// ```
    #[inline]
    pub fn clear(&mut self) {
        todo!()
    }

    /// Returns a reference to the the [`BuidHasher`]s used.
    /// for each ouf our [`NUM_TABLES`] tables.
    ///
    /// # Examples
    ///
    /// ```
    /// ```
    #[inline]
    pub fn hashers(&self) -> [RandomState; NUM_TABLES] {
        todo!()
    }

    /// Gets the given key's corresponding entry in the map for in-place manipulation.
    #[inline]
    pub fn entry(&mut self, key: K) {
        todo!()
    }

    /// Returns a reference to the value corresponding to the key.`
    #[inline]
    pub fn get<Q: ?Sized>(&self, k: &Q) -> Option<&V> {
        todo!()
    }

    /// Returns the key-value pair corresponding to the supplied key.
    #[inline]
    pub fn get_key_value<Q: ?Sized>(&self, k: &Q) -> Option<(&K, &V)> {
        todo!()
    }

    /// Returns `true` if the map contains a value for the specified key.```
    #[inline]
    pub fn contains_key<Q: ?Sized>(&self, k: &Q) -> bool {
        todo!()
    }

    /// Returns a mutable reference to the value corresponding to the key.
    #[inline]
    pub fn get_mut<Q: ?Sized>(&mut self, k: &Q) -> Option<&mut V> {
        todo!()
    }

    /// Inserts a key-value pair into the map.
    ///
    /// If the map did not have this key present, [`None`] is returned.
    ///
    /// If the map did have this key present, the value is updated, and the old
    /// value is returned. The key is not updated, though; this matters for
    /// types that can be `==` without being identical. See the [module-level
    /// documentation] for more.
    #[inline]
    pub fn insert(&mut self, k: K, v: V) -> Option<V> {
        todo!()
    }

    /// Removes a key from the map, returning the value at the key if the key
    /// was previously in the map.
    #[inline]
    pub fn remove<Q: ?Sized>(&mut self, k: &Q) -> Option<V> {
        todo!()
    }

    /// Removes a key from the map, returning the stored key and value if the
    /// key was previously in the map.
    #[inline]
    pub fn remove_entry<Q: ?Sized>(&mut self, k: &Q) -> Option<(K, V)> {
        todo!()
    }

    /// Retains only the elements specified by the predicate.
    ///
    /// In other words, remove all pairs `(k, v)` such that `f(&k,&mut v)` returns `false`.
    #[inline]
    pub fn retain<F>(&mut self, f: F)
    where
        F: FnMut(&K, &mut V) -> bool,
    {
        todo!()
    }

    /// Creates a consuming iterator visiting all the keys in arbitrary order.
    /// The map cannot be used after calling this.
    /// The iterator element type is `K`.
    #[inline]
    pub fn into_keys(self) {
        todo!()
    }

    /// Creates a consuming iterator visiting all the values in arbitrary order.
    /// The map cannot be used after calling this.
    /// The iterator element type is `V`.
    #[inline]
    pub fn into_values(self) {
        todo!()
    }
}

#[derive(Debug)]
struct RawCuckooTable<K, V> {
    /// We use 4 hash tables of equal size
    hash_tables: Option<[Box<[CuckooTableSlot<K, V>]>; NUM_TABLES]>,

    /// hash functions for each of our 4 hash tables
    hash_functions: Option<[RandomState; NUM_TABLES]>,

    /// During a single insert episode, if we evict
    /// more that `M=MAX_EVICTION` we stop evicting
    /// and store the item that still doesn't have a home in
    /// the overflow hashmap.
    /// TODO -> Change this to a closed addressed hash table with double hashing.
    overflow: Option<FxHashMap<K, V>>,
}

impl<K, V> RawCuckooTable<K, V> {
    /// create a new empty cuckoo hash table
    pub fn new() -> RawCuckooTable<K, V> {
        RawCuckooTable {
            hash_tables: None,
            overflow: None,
            hash_functions: None,
        }
    }
}

/// A single bucket in the hash table. A single `Slot` is capable of holding
/// up to 8 slot entries. When collisions happen, we insert colliding
/// keys in a single slot until `size=8`. At that point, to resolve collisions
/// for a single slot, we pick a random element to evict.
#[derive(Debug)]
struct CuckooTableSlot<K, V> {
    /// The number of slot entries in this slot.
    size: u8,

    /// The entries in this slot. This has a maximum size
    /// of 8
    entries: Box<[SlotEntry<K, V>]>,
}

impl<K, V> CuckooTableSlot<K, V> {
    /// Create a new single bucket capable of holding a fixed number of
    /// entries. Each slot starts out empty
    pub fn new() -> Self {
        CuckooTableSlot {
            size: 0,
            entries: Box::new([]),
        }
    }

    /// Inserts the provided Key-Value pair in this slot. This
    /// procedure can either find space in this slot of find that
    /// this slot is filled up. If there is space, we create a new `SlotEntry`
    /// for `k` and `v` and store it here. If this slot is filled up,
    /// we randomly pick and index and evict the current occupant of that index.
    /// The return type indicates if we succeded without eviction, or
    /// if we succeded via eviction
    pub fn insert(k: K, v: V) -> Option<SlotEntry<K, V>> {
        todo!()
    }
}

/// A single entry in the hash table.
#[derive(Debug)]
struct SlotEntry<K, V> {
    key: K,
    value: V,
}

impl<K, V> SlotEntry<K, V> {
    /// Create a new slot entry with the provided key and value
    pub fn new(key: K, value: V) -> Self {
        SlotEntry { key, value }
    }
}
