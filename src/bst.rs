use std::marker::PhantomData;

#[derive(Debug, PartialEq, Eq)]
pub struct BSTNodeHandle<K, V>(usize, PhantomData<K>, PhantomData<V>);

/// A single node in the tree
#[derive(Debug)]
struct BstNode<Key: Ord, Value> {
    key: Key,
    value: Value,
    left: Option<usize>,
    right: Option<usize>,
    parent: Option<usize>,
    child_type: Option<ChildType>,
}

#[derive(Debug, Clone, PartialOrd, Eq, PartialEq)]
enum ChildType {
    Right,
    Left,
}

impl<Key: Ord, Value> BstNode<Key, Value> {
    pub fn new(k: Key, v: Value) -> Self {
        BstNode {
            key: k,
            value: v,
            left: None,
            right: None,
            parent: None,
            child_type: None,
        }
    }
}

/// An adjacency list representation of an associative tree
#[derive(Debug)]
pub struct BST<Key: Ord, Value> {
    root: Option<Vec<BstNode<Key, Value>>>,
    root_index: Option<usize>,
}

impl<Key: Ord, Value> BST<Key, Value> {
    pub fn new() -> Self {
        BST {
            root: None,
            root_index: None,
        }
    }

    pub fn insert(&mut self, k: Key, v: Value) {
        match &mut self.root {
            None => {
                let mut root = Vec::new();
                root.push(BstNode::new(k, v));
                self.root = Some(root);
                self.root_index = Some(0);
            }
            Some(elements) => {
                let mut prev = None;
                let mut cur = self.root_index;
                while cur.is_some() {
                    prev = cur;
                    let cur_node = &elements[cur.unwrap()];
                    if k <= cur_node.key {
                        cur = cur_node.left;
                    } else {
                        cur = cur_node.right;
                    }
                }
                let mut new_node = BstNode::new(k, v);
                new_node.parent = prev;
                let new_node_idx = Some(elements.len());
                let mut parent_node = &mut elements[prev.unwrap()];
                if parent_node.key < new_node.key {
                    parent_node.right = new_node_idx;
                    new_node.child_type = Some(ChildType::Right);
                } else {
                    parent_node.left = new_node_idx;
                    new_node.child_type = Some(ChildType::Left);
                }
                elements.push(new_node);
            }
        }
    }

    pub fn delete(&mut self, k: Key) -> Option<Value> {
        match &mut self.root {
            None => None,
            Some(elements) => {
                let delete_idx_option = Self::get_helper(&elements, self.root_index, k);
                match delete_idx_option {
                    None => None,
                    Some(delete_idx) => {
                        if elements[delete_idx].left.is_none() {
                            Self::transplant(elements, delete_idx, ChildType::Right);
                            if elements[delete_idx].parent.is_none() {
                                self.root_index = elements[delete_idx].right;
                            }
                        } else if elements[delete_idx].right.is_none() {
                            Self::transplant(elements, delete_idx, ChildType::Left);
                            if elements[delete_idx].parent.is_none() {
                                self.root_index = elements[delete_idx].left;
                            }
                        } else {
                            if let Some(succ_idx) = Self::min_helper(&elements, elements[delete_idx].right) {
                                if elements[succ_idx].parent != delete_idx_option {
                                    Self::transplant(elements, succ_idx, ChildType::Right);
                                    elements[succ_idx].right = elements[delete_idx].right;
                                    let delete_node_right_idx = elements[succ_idx].right;
                                    elements[delete_node_right_idx.unwrap()].parent = Some(succ_idx);
                                }
                                // if we are here, then succ_idx is actually delete_idx's right child.
                                Self::transplant(elements, delete_idx, ChildType::Right);
                                elements[succ_idx].left = elements[delete_idx].left;
                                let delete_node_left_idx = elements[succ_idx].left;
                                elements[delete_node_left_idx.unwrap()].parent = Some(succ_idx);
                                if elements[delete_idx].parent.is_none() {
                                    self.root_index = Some(succ_idx);
                                }
                            }
                        }
                        let return_value = Self::swap_remove(elements, delete_idx);
                        if self.root_index.is_none() {
                            // Assert that the tree should be empty any time we descend into this path
                            if let Some(elements) = &self.root {
                                debug_assert!(elements.len() == 0)
                            }
                            self.root = None;
                        }
                        return_value
                    }
                }
            }
        }
    }

    /// By the time this is called, the node at delete_idx has effectively been removed from the
    /// tree structure. That is, its parent disowned them and all of its children now point to
    /// its former parents. What we do here is to remove it from the underlying vector.
    /// We'd like to so so in O(1), however. So, we swap it with the last node `last_node` and set the
    /// pointers of `last_node`'s parent and children to point to delete_idx.
    fn swap_remove(elements: &mut Vec<BstNode<Key, Value>>, delete_idx: usize) -> Option<Value> {
        let deleted_node = elements.swap_remove(delete_idx);
        if delete_idx < elements.len() {
            // After the call to swap_remove, delete_idx now holds the node that was the
            // last node. We now need to inform its parents and children that it was
            // moved.
            if let Some(left_child_idx) = elements[delete_idx].left {
                elements[left_child_idx].parent = Some(delete_idx);
            }
            if let Some(right_child_idx) = elements[delete_idx].right {
                elements[right_child_idx].parent = Some(delete_idx);
            }
            match elements[delete_idx].parent {
                None => {
                    // self.root_idx = delete_idx. handled by caller
                }
                Some(parent_idx) => match elements[delete_idx].child_type {
                    Some(ChildType::Left) => elements[parent_idx].left = Some(delete_idx),
                    Some(ChildType::Right) => elements[parent_idx].right = Some(delete_idx),
                    None => {}
                },
            }
        }
        Some(deleted_node.value)
    }

    /// Link a node's parent to one of its children, effectively removing the node from the
    /// tree structure.
    fn transplant(elements: &mut Vec<BstNode<Key, Value>>, delete_idx: usize, direction: ChildType) {
        let delete_node_child = match direction {
            ChildType::Left => elements[delete_idx].left,
            ChildType::Right => elements[delete_idx].right,
        };
        if let Some(child_idx) = delete_node_child {
            elements[child_idx].parent = elements[delete_idx].parent;
            elements[child_idx].child_type = elements[delete_idx].child_type.clone();
        }
        match elements[delete_idx].parent {
            Some(parent_idx) => match elements[delete_idx].child_type {
                Some(ChildType::Left) => elements[parent_idx].left = delete_node_child,
                Some(ChildType::Right) => elements[parent_idx].right = delete_node_child,
                None => {}
            },
            None => {
                // self.root_index = delete_node_child
                // Handled by caller because we do not have a handle to &mut self
            }
        }
    }

    pub fn get(&self, k: Key) -> Option<&Value> {
        match &self.root {
            None => None,
            Some(elements) => match Self::get_helper(elements, self.root_index, k) {
                None => None,
                Some(k_idx) => Some(&elements[k_idx].value),
            },
        }
    }

    pub fn pred(&self, k: Key) -> Option<&Value> {
        match &self.root {
            None => None,
            Some(elements) => match Self::get_helper(elements, self.root_index, k) {
                None => None,
                Some(q_idx) => {
                    let mut query_node = &elements[q_idx];
                    if query_node.left.is_some() {
                        match Self::max_helper(elements, query_node.left) {
                            None => None,
                            Some(min_idx) => Some(&elements[min_idx].value),
                        }
                    } else {
                        let mut ancestor = query_node.parent;
                        while ancestor.is_some() && query_node.child_type == Some(ChildType::Left) {
                            let ancestor_node = &elements[ancestor.unwrap()];
                            query_node = ancestor_node;
                            ancestor = ancestor_node.parent;
                        }
                        match ancestor {
                            None => None,
                            Some(ancestor_idx) => Some(&elements[ancestor_idx].value),
                        }
                    }
                }
            },
        }
    }

    pub fn succ(&self, k: Key) -> Option<&Value> {
        match &self.root {
            None => None,
            Some(elements) => match Self::get_helper(elements, self.root_index, k) {
                None => None,
                Some(q_idx) => {
                    let mut query_node = &elements[q_idx];
                    if query_node.right.is_some() {
                        match Self::min_helper(elements, query_node.right) {
                            None => None,
                            Some(max_idx) => Some(&elements[max_idx].value),
                        }
                    } else {
                        let mut ancestor = query_node.parent;
                        while ancestor.is_some() && query_node.child_type == Some(ChildType::Right) {
                            let ancestor_node = &elements[ancestor.unwrap()];
                            query_node = ancestor_node;
                            ancestor = ancestor_node.parent;
                        }
                        match ancestor {
                            None => None,
                            Some(ancestor_idx) => Some(&elements[ancestor_idx].value),
                        }
                    }
                }
            },
        }
    }

    pub fn max(&self) -> Option<&Value> {
        match &self.root {
            None => None,
            Some(elements) => match Self::max_helper(elements, self.root_index) {
                None => None,
                Some(max_idx) => Some(&elements[max_idx].value),
            },
        }
    }

    pub fn min(&self) -> Option<&Value> {
        match &self.root {
            None => None,
            Some(elements) => match Self::min_helper(elements, self.root_index) {
                None => None,
                Some(min_idx) => Some(&elements[min_idx].value),
            },
        }
    }

    fn get_helper(elements: &Vec<BstNode<Key, Value>>, cur_subtree_root: Option<usize>, k: Key) -> Option<usize> {
        match cur_subtree_root {
            None => None,
            Some(subtree_idx) => {
                let cur_node = &elements[subtree_idx];
                if k == cur_node.key {
                    cur_subtree_root
                } else if k <= cur_node.key {
                    Self::get_helper(elements, cur_node.left, k)
                } else {
                    Self::get_helper(elements, cur_node.right, k)
                }
            }
        }
    }

    fn max_helper(elements: &Vec<BstNode<Key, Value>>, mut cur: Option<usize>) -> Option<usize> {
        while elements[cur.unwrap()].right.is_some() {
            cur = elements[cur.unwrap()].right;
        }
        cur
    }

    fn min_helper(elements: &Vec<BstNode<Key, Value>>, mut cur: Option<usize>) -> Option<usize> {
        while elements[cur.unwrap()].left.is_some() {
            cur = elements[cur.unwrap()].left;
        }
        cur
    }
}

#[cfg(test)]
mod test {
    use super::BST;

    #[test]
    fn test_create() {
        let t = BST::<String, String>::new();
        assert_eq!(None, t.get("Georgie".into()));
        assert_eq!(None, t.pred("Georgie".into()));
        assert_eq!(None, t.succ("Georgie".into()));
        assert_eq!(None, t.min());
        assert_eq!(None, t.max());
    }

    #[test]
    fn test_insert() {
        let mut t = BST::<u32, String>::new();
        let values = [
            (1, "Jean"),
            (12, "FT"),
            (8, "SCMP"),
            (11, "WSJ"),
            (0, "MA"),
            (1, "Norman"),
        ];
        for (k, v) in &values {
            t.insert(*k, v.to_string());
        }
        assert_eq!(Some("FT".to_string()).as_ref(), t.get(12));
        assert_eq!(None, t.pred(0));
        assert_eq!(Some("WSJ".to_string()).as_ref(), t.succ(8));
        assert_eq!(Some("FT".to_string()).as_ref(), t.succ(11));
        assert_eq!(Some("MA".to_string()).as_ref(), t.min());
        assert_eq!(Some("FT".to_string()).as_ref(), t.max());
        assert_eq!(None, t.succ(12));
        assert_eq!(Some("SCMP".to_string()).as_ref(), t.pred(11));
        assert_eq!(None, t.get(14));
    }

    #[test]
    fn test_delete() {
        let mut t = BST::<u32, String>::new();
        let values = [
            (1, "Jean"),
            (12, "FT"),
            (8, "SCMP"),
            (11, "WSJ"),
            (0, "MA"),
            (1, "Norman"),
        ];
        for (k, v) in &values {
            t.insert(*k, v.to_string());
        }

        assert_eq!(Some("FT".to_string()).as_ref(), t.get(12));
        let _ft = t.delete(12);
        assert_eq!(None, t.get(12));
        assert_eq!(Some("WSJ".to_string()).as_ref(), t.max());
        assert_eq!(None, t.succ(11));
        assert_eq!(Some("WSJ".to_string()).as_ref(), t.succ(8));
        assert_eq!(Some("MA".to_string()).as_ref(), t.min());
        let _ma = t.delete(0); // TEST this. Why panic
        assert_eq!(Some("Norman".to_string()).as_ref(), t.min());
        assert_eq!(None, t.get(0));

        // Test use after delete: The compiler wont allow it as expected
        // let scmp_ref = t.get(8);
        // let scmp = t.delete(8);
        // assert_eq!(None, t.get(8));
        // println!("before: {}, after: {}", scmp_ref.unwrap(), scmp.unwrap());
    }
}
