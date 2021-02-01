//! A red-black-tree is a binary search tree in which each node is marked as either RED or BLACK.
//! This extra bit of information if used to maintain the tree's balance. This balance is obtained
//! by ensuring that all operations on the tree maintain the following RBT invariants.
//!
//!  ## Red-Black Properties
//! 1. Every node ie either Red or Black
//! 2. The root is black
//! 3. Every Nil node is black
//! 4. If a node is red, then both of its children are black
//! 5. For each node, a simple (ie no cycles) path from the node to
//!    descendant leaves contains the same number of black nodes
//!
//! These invariants are maintained by using rotations. These are local structural
//! changes to the tree that move nodes around while maintaining the search tree
//! properties.
use std::cmp::Ordering::{Equal, Greater, Less};

/// The color of a given node
///
/// # Example
///
/// ```
/// let color: Color = Color::Red;
/// ```
#[derive(Debug, Eq, PartialEq)]
pub enum Color {
    Red,
    Black,
}

/// Indicates whether a given node is a left or right child of its
/// parent
pub enum ChildType {
    Left,
    Right,
}

/// We use the new type index pattern to make our code more
/// understandable. Using indexes to simulate pointers
/// can lead to opaqueness. Using a concrete type instead of
/// raw indexes ameliorates this. The 'insane' derive ensures
/// that the new type has all the crucial properties of the
/// underlying index
#[derive(Debug, Copy, Clone, Ord, PartialOrd, Eq, PartialEq, Hash)]
struct RbtNodeIdx(usize);

impl From<usize> for RbtNodeIdx {
    /// Allows us to quickly construct tree indexes from
    /// a raw index
    /// # Example
    ///
    /// ```
    /// let idx: RbtNodeIdx = 5.into();
    /// ```
    fn from(idx: usize) -> Self {
        RbtNodeIdx(idx)
    }
}

impl<K: Ord, V> std::ops::Index<RbtNodeIdx> for Vec<RbtNode<K, V>> {
    type Output = RbtNode<K, V>;

    /// This allows us to use RbtNodeIdx directly as an index
    ///
    /// # Example
    /// ```
    /// let v: Vec<RbtNode<K, V>> = ... // snip
    /// let idx: RbtNodeIdx = 0.into();
    /// let first_node = v[idx]
    /// ```
    fn index(&self, index: RbtNodeIdx) -> &Self::Output {
        &self[index.0]
    }
}

impl<K: Ord, V> std::ops::IndexMut<RbtNodeIdx> for Vec<RbtNode<K, V>> {
    fn index_mut(&mut self, index: RbtNodeIdx) -> &mut Self::Output {
        &mut self[index.0]
    }
}
/// A single entry in the tree
#[derive(Debug)]
pub struct Entry<K, V> {
    key: K,
    value: V,
}

impl<K: Ord, V> From<(K, V)> for Entry<K, V> {
    /// Allows us to quickly construct an entry from
    /// a key value tuple
    ///
    /// # Example
    ///
    /// ```
    /// let entry: Entry<&str, usize> = ("usa", 245).into();
    /// ```
    fn from(e: (K, V)) -> Self {
        Entry { key: e.0, value: e.1 }
    }
}

/// A single node in the tree. This is the main unit of
/// computation in the tree. That is, all operations
/// operate on nodes. It is parameterized by a key which
/// should be orderable and an arbitrary value
#[derive(Debug)]
struct RbtNode<K: Ord, V> {
    entry: Entry<K, V>,
    left: Option<RbtNodeIdx>,
    right: Option<RbtNodeIdx>,
    parent: Option<RbtNodeIdx>,
    color: Color,
}

impl<K: Ord, V> RbtNode<K, V> {
    /// Create a new red black tree node with the given entry.
    ///
    /// # Example
    /// ```
    /// let entry: Entry<&str, usize> = ("usa", 245).into();
    /// let node = RbtNode::new(entry);
    /// ```
    pub fn new(entry: Entry<K, V>) -> Self {
        RbtNode {
            entry,
            left: None,
            right: None,
            parent: None,
            color: Color::Black,
        }
    }

    /// Create a new red black tree node with the given entry.
    /// and color
    ///
    /// # Example
    /// ```
    /// let entry: Entry<&str, usize> = ("usa", 245).into();
    /// let node = RbtNode::with_color(entry, Color::Black);
    /// #[test]
    /// assert_eq!(node.color, Color::Black);
    /// ```
    pub fn with_color(entry: Entry<K, V>, color: Color) -> Self {
        RbtNode {
            entry,
            left: None,
            right: None,
            parent: None,
            color,
        }
    }

    /// Retrieve the key in this node
    pub fn key(&self) -> &K {
        &self.entry.key
    }

    /// Retrieves the value in this node
    pub fn value(&self) -> &V {
        &self.entry.value
    }
}

/// A type alias for ergonomic reasons
type Nodes<K, V> = Vec<RbtNode<K, V>>;

/// A red black tree implemented using indexes
#[derive(Debug)]
pub struct RBT<K: Ord, V> {
    /// A growable container of all the nodes in the tree
    elements: Option<Nodes<K, V>>,

    // The location of the root. Optional because we first create
    // an empty tree. We keep track of it because its location
    // can change as we make structural changes to the tree
    root: Option<RbtNodeIdx>,
}

/// Implementation of Read operations. These procedures
/// do not lead to structural changes in the tree
impl<K: Ord, V> RBT<K, V> {
    /// Create a new, empty red black tree
    ///
    /// # Example
    /// ```
    /// let rbt: RBT<&str, usize> = RBT::new();
    /// ```
    pub fn new() -> Self {
        RBT {
            elements: None,
            root: None,
        }
    }

    /// Retrieves the Key-Value pair associated with
    /// the provided key id it exists in the tree.
    ///
    /// # Examples
    /// ```
    /// use crate::red_black_tree::RBT;
    /// let mut tree = RBT::new();
    /// assert!(tree.get("cat").is_none())
    ///
    /// tree.insert(("cat", "tabby").into());
    /// assert!(tree.get("cat").is_some());
    /// assert_eq!(tree.get("cat").unwrap(), ("cat", "tabby").into());
    /// ```
    pub fn get(&self, k: K) -> Option<&Entry<K, V>> {
        self.elements
            .as_ref()
            .and_then(|nodes| Self::get_helper(nodes, self.root, k))
            .and_then(|key_index| match &self.elements {
                None => None,
                Some(nodes) => Some(&nodes[key_index].entry),
            })
    }

    /// Retrieves the Key-Value pair associated with
    /// the largest key smaller than the provides key
    /// if it exists. Such a value does not exist if the given
    /// key is the smallest element in the tree
    ///
    /// # Examples
    /// ```
    ///
    /// ```
    pub fn pred(&self, k: K) -> Option<&Entry<K, V>> {
        self.elements
            .as_ref()
            .and_then(|nodes| Self::get_helper(nodes, self.root, k))
            .and_then(|key_idx| {
                self.elements
                    .as_ref()
                    .and_then(|nodes| {
                        let left = nodes[key_idx].left;
                        left.map_or_else(
                            || Self::lra(nodes, nodes[key_idx].parent),
                            |left_idx| Self::max_helper(nodes, Some(left_idx)),
                        )
                    })
                    .and_then(|pred_idx| self.elements.as_ref().and_then(|nodes| Some(&nodes[pred_idx].entry)))
            })
    }

    /// Retrieves the Key-Value pair associated with
    /// the smallest key larger than the provides key
    /// if it exists. Such a value does not exist if the given
    /// key is the largest element in the tree
    ///
    /// # Examples
    /// ```
    ///
    /// ```
    pub fn succ(&self, k: K) -> Option<&Entry<K, V>> {
        self.elements
            .as_ref()
            .and_then(|nodes| Self::get_helper(nodes, self.root, k))
            .and_then(|key_idx| {
                self.elements
                    .as_ref()
                    .and_then(|nodes| {
                        let right = nodes[key_idx].right;
                        right.map_or_else(
                            || Self::lla(nodes, Some(key_idx)),
                            |right_idx| Self::min_helper(nodes, Some(right_idx)),
                        )
                    })
                    .and_then(|succ_idx| self.elements.as_ref().and_then(|nodes| Some(&nodes[succ_idx].entry)))
            })
    }

    /// Retrieves the Key-Value pair associated with
    /// the largest key in the tree
    ///
    /// # Examples
    /// ```
    ///
    /// ```
    pub fn max(&self) -> Option<&Entry<K, V>> {
        self.elements
            .as_ref()
            .and_then(|nodes| Self::max_helper(nodes, self.root))
            .and_then(|max_idx| self.elements.as_ref().and_then(|nodes| Some(&nodes[max_idx].entry)))
    }

    /// Retrieves the Key-Value pair associated with
    /// the smallest key in the tree
    ///
    /// # Examples
    /// ```
    ///
    /// ```
    pub fn min(&self) -> Option<&Entry<K, V>> {
        match &self.elements {
            None => None,
            Some(nodes) => Self::min_helper(nodes, self.root).and_then(move |min_idx| Some(&nodes[min_idx].entry)),
        }
    }
}

/// Implementation of associated helper functions to read operations
impl<K: Ord, V> RBT<K, V> {
    /// Searches for the location of the entry with the provided key starting
    /// at the specified location index. If such an entry exists, we return its index.
    /// If not, we return [`None`]
    fn get_helper(nodes: &Nodes<K, V>, start: Option<RbtNodeIdx>, key: K) -> Option<RbtNodeIdx> {
        start.and_then(|cur_idx| match nodes[cur_idx].key().cmp(&key) {
            Equal => start,
            Less => Self::get_helper(nodes, nodes[cur_idx].left, key),
            Greater => Self::get_helper(nodes, nodes[cur_idx].right, key),
        })
    }

    /// Searches for the location of the entry with the largest key value
    fn max_helper(nodes: &Nodes<K, V>, start: Option<RbtNodeIdx>) -> Option<RbtNodeIdx> {
        start.and_then(|cur_idx| match nodes[cur_idx].right {
            None => start,
            Some(right_child_idx) => Self::max_helper(nodes, Some(right_child_idx)),
        })
    }

    /// Searches for the location of the entry with the smallest key value
    fn min_helper(nodes: &Nodes<K, V>, start: Option<RbtNodeIdx>) -> Option<RbtNodeIdx> {
        start.and_then(|cur_idx| match nodes[cur_idx].left {
            None => start,
            Some(left_child_idx) => Self::min_helper(nodes, Some(left_child_idx)),
        })
    }

    /// Searches for the location of the lowest ancestor node that is a right child of
    /// its parent. This is a sub-procedure used when computing the predecessor of a node
    /// `lra` stands for lowest right ancestor. It assumes that start is the parent
    /// of the node whose predecessor we are interested in
    fn lra(nodes: &Nodes<K, V>, start: Option<RbtNodeIdx>) -> Option<RbtNodeIdx> {
        start.and_then(|cur_idx| {
            Self::child_type(nodes, start).and_then(|child_type| match child_type {
                ChildType::Left => Self::lra(nodes, nodes[cur_idx].parent),
                ChildType::Right => nodes[cur_idx].parent,
            })
        })
    }

    /// Searches for the location of the lowest ancestor node that is a left child of
    /// its parent. This is a sub-procedure used when computing the successor of a node
    /// `lla` stands for lowest left ancestor. It assumes that start is the parent
    /// of the node whose predecessor we are interested in
    fn lla(nodes: &Nodes<K, V>, start: Option<RbtNodeIdx>) -> Option<RbtNodeIdx> {
        start.and_then(|cur_idx| {
            Self::child_type(nodes, start).and_then(|child_type| match child_type {
                ChildType::Right => Self::lla(nodes, nodes[cur_idx].parent),
                ChildType::Left => nodes[cur_idx].parent,
            })
        })
    }

    /// Is this node a left child or right child of its parent
    fn child_type(nodes: &Nodes<K, V>, cur: Option<RbtNodeIdx>) -> Option<ChildType> {
        cur.and_then(|cur_idx| {
            let parent = nodes[cur_idx].parent;
            parent.and_then(|parent_idx| match nodes[parent_idx].left.cmp(&cur) {
                Equal => Some(ChildType::Left),
                _ => Some(ChildType::Right),
            })
        })
    }
}

/// Implementation of Write operations. These procedures
/// lead to structural changes in the tree
impl<K: Ord, V> RBT<K, V> {
    /// Adds a new entry into the red black tree in O(lg n) time.
    /// This procedure uses tree rotations to maintain the Red-Black
    /// Invariants.
    ///
    /// # Examples
    /// ```
    ///
    /// ```
    pub fn insert(&mut self, e: Entry<K, V>) {
        match &mut self.elements {
            None => {
                let mut elements = Vec::new();
                elements.push(RbtNode::new(e));
                self.elements = Some(elements);
                self.root = Some(0.into());
            }
            Some(elts) => match Self::insert_helper(elts, self.root, e) {
                None => {}
                Some(new_root_idx) => self.root = Some(new_root_idx),
            },
        }
    }

    /// Removes the entry associated with the provided key
    /// from the red black tree in O(lg n) time.
    /// This procedure uses tree rotations to maintain the Red-Black
    /// Invariants.
    ///
    /// # Examples
    /// ```
    ///
    /// ```
    pub fn delete(&mut self, k: K) -> Option<Entry<K, V>> {
        self.elements
            .as_ref()
            .and_then(|elts| Self::get_helper(elts, self.root, k).and_then(|delete_idx| todo!()))
    }
}

/// Implementation of associated helper functions
/// to write operation
impl<K: Ord, V> RBT<K, V> {
    fn insert_helper(nodes: &mut Nodes<K, V>, mut cur_idx: Option<RbtNodeIdx>, e: Entry<K, V>) -> Option<RbtNodeIdx> {
        let mut prev_idx = None;
        while cur_idx.is_some() {
            prev_idx = cur_idx;
            let cur_node = &nodes[cur_idx.unwrap()];
            if &e.key <= cur_node.key() {
                cur_idx = cur_node.left
            } else {
                cur_idx = cur_node.right
            }
        }
        let mut new_node = RbtNode::with_color(e, Color::Red);
        new_node.parent = prev_idx;
        let new_node_idx = Some(nodes.len().into());
        let mut parent_node = &mut nodes[prev_idx.unwrap()];
        if parent_node.key() < new_node.key() {
            parent_node.right = new_node_idx;
        } else {
            parent_node.left = new_node_idx;
        }
        nodes.push(new_node);
        Self::insert_balance(nodes, new_node_idx)
    }

    /// Restores the red-black-tree invariants after an insert has been made by
    /// applying rotations as needed
    fn insert_balance(nodes: &mut Nodes<K, V>, mut z_idx: Option<RbtNodeIdx>) -> Option<RbtNodeIdx> {
        todo!()
    }

    // fn insert_balance_helper(nodes: &mut Nodes<K, V>, )

    /// Restores the red-black-tree invariants after an entry has been deleted by
    /// applying rotations as needed
    fn delete_balance(nodes: &mut Nodes<K, V>, y_idx: RbtNodeIdx) -> Option<RbtNodeIdx> {
        todo!()
    }

    /// Given the root `x` of some subtree, we would like to switch this root with its
    /// right child `y`. Let `α` be the left child of `x`. Further, let `Ω` be the right
    /// child of `y` and `χ` be the left child of `y`. Right off the bat, we know that:
    /// ** Both `χ` and `Ω` -- the children of `y` are larger than `x`
    /// ** `α` is the only value smaller than `x`.
    /// ** `χ` is the only value smaller than `y`
    /// ** `Ω` is larger than both `x` and `y`
    /// Since the task is to swap `x` and `y`, we can use the observations above to maintain
    /// the search tree invariant using the following steps:
    /// 1. Swap `x` and `y`.
    /// 2. This breaks the bst invariant because `x` is smaller than `y`
    ///    yet its to the right of `y`. To remedy this, make `x` the left
    ///    child of `y`.
    /// 3. Make `Ω` the right child of `y`. It's the only value larger than `y`. No modifications are
    ///    needed for this.
    /// 4. Now all that remains is to handle `α` and `χ`. This is simple because only
    ///    `α` is less than `x`, so it stays as `x`'s left child. Since `x` < `χ` < `Ω`
    ///     it goes to the right of `x`.
    /// We also have to take care when `x` is the root. In that case, `y` becomes the new root
    /// Since we need to communicate this to the [`RBT`] object, we return the index of the
    /// new root if it changed.
    fn rotate_left(nodes: &mut Nodes<K, V>, x_idx: RbtNodeIdx) -> Option<RbtNodeIdx> {
        nodes[x_idx].right.and_then(|y_idx| {
            // Since `x` < `χ` < `Ω`, it goes to the right of `x`
            nodes[x_idx].right = nodes[y_idx].left;
            if let Some(chi_idx) = nodes[y_idx].left {
                nodes[chi_idx].parent = Some(x_idx);
            }
            // Swap x and y
            nodes[y_idx].parent = nodes[x_idx].parent;
            let result = {
                match Self::child_type(nodes, Some(x_idx)) {
                    None => Some(y_idx),
                    Some(ChildType::Left) => {
                        nodes[x_idx]
                            .parent
                            .and_then(|x_parent| Some(nodes[x_parent].left = Some(y_idx)));
                        None
                    }
                    Some(ChildType::Right) => {
                        nodes[x_idx]
                            .parent
                            .and_then(|x_parent| Some(nodes[x_parent].right = Some(y_idx)));
                        None
                    }
                }
            };

            // Make `x` the left child of `y`.
            nodes[x_idx].parent = Some(y_idx);
            nodes[y_idx].left = Some(x_idx);
            result
        })
    }

    /// This is similar to [`left_rotate`] only that we begin with the final state of that procedure.
    /// That is, we are given a root `y` of some subtree. We would like to switch this root
    /// with its left child `x`. Let alpha be the left child of `x` and `chi` the right child of `x`.
    /// Further, let `Omega` be the right child of `y`. The reset of the procedure proceeds analogously
    fn rotate_right(nodes: &mut Nodes<K, V>, y_idx: RbtNodeIdx) -> Option<RbtNodeIdx> {
        nodes[y_idx].left.and_then(|x_idx| {
            nodes[y_idx].left = nodes[x_idx].right;
            if let Some(chi_idx) = nodes[x_idx].right {
                nodes[chi_idx].parent = Some(y_idx);
            }
            nodes[x_idx].parent = nodes[y_idx].parent;
            let result = {
                match Self::child_type(nodes, Some(y_idx)) {
                    None => Some(x_idx),
                    Some(ChildType::Left) => {
                        nodes[y_idx]
                            .parent
                            .and_then(|y_parent| Some(nodes[y_parent].left = Some(x_idx)));
                        None
                    }
                    Some(ChildType::Right) => {
                        nodes[y_idx]
                            .parent
                            .and_then(|y_parent| Some(nodes[y_parent].right = Some(x_idx)));
                        None
                    }
                }
            };
            nodes[y_idx].parent = Some(x_idx);
            nodes[x_idx].right = Some(y_idx);
            result
        })
    }

    fn transplant(nodes: &mut Nodes<K, V>, left: Option<RbtNodeIdx>) {
        todo!()
    }

    fn swap_remove(nodes: &mut Nodes<K, V>, delete_idx: Option<RbtNodeIdx>) -> Option<V> {
        todo!()
    }
}
