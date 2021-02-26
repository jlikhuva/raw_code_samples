//! Module implementing all foundational procedures for graphs

use std::{
    collections::{HashMap, HashSet},
    hash::Hash,
    ops::{Index, IndexMut},
};

/// We will be using a proto-arena, in the form of a vector,
/// to store our graph. We'll use the wrapped index pattern
/// when indexing the arena
#[derive(Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Clone)]
pub struct NodeHandle(usize);

/// A node in an undirected graph
#[derive(Debug)]
struct Node<T: Hash> {
    /// The value stored at this node
    entry: T,

    /// The collection of nodes that
    /// are adjacent to this node
    neighbors: Option<HashSet<NodeHandle>>,
}

/// A collection of all the nodes in the graph
type Nodes<T> = Vec<Node<T>>;

impl<T: Hash> Index<NodeHandle> for Nodes<T> {
    type Output = Node<T>;
    fn index(&self, idx: NodeHandle) -> &Self::Output {
        &self[idx.0]
    }
}

impl<T: Hash> IndexMut<NodeHandle> for Nodes<T> {
    fn index_mut(&mut self, idx: NodeHandle) -> &mut Self::Output {
        &mut self[idx.0]
    }
}

#[derive(Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Clone)]
pub struct EdgeHandle(usize);

#[derive(Debug)]
struct Edge<K: Hash> {
    /// The arbitrary label on this edge
    label: K,

    /// The left and right ends of this edge
    left: NodeHandle,
    right: NodeHandle,
}

/// A Collection of all the edges in the graph
type Edges<K> = Vec<Edge<K>>;

impl<K: Hash> Index<EdgeHandle> for Edges<K> {
    type Output = Edge<K>;
    fn index(&self, idx: EdgeHandle) -> &Self::Output {
        &self[idx.0]
    }
}

impl<K: Hash> IndexMut<EdgeHandle> for Edges<K> {
    fn index_mut(&mut self, idx: EdgeHandle) -> &mut Self::Output {
        &mut self[idx.0]
    }
}

/// A graph is simply a collection of nodes
#[derive(Debug)]
pub struct UndirectedGraph<NodeLabel: Hash, EdgeLabel: Hash> {
    nodes: Option<Nodes<NodeLabel>>,
    nodes_mapper: Option<HashMap<NodeLabel, NodeHandle>>,

    edges: Option<Edges<EdgeLabel>>,
}
