//! A graph G = (V, E) is a collection of vertices and edges between
//! those vertices. Each Node has an associated key. For instance,
//! a graph of computers in a network could have the IP address of
//! each machine as the key of a node. This key identifies a node
//! In this implementation, we do not include the payload as part
//! of the node. Thus, it is assumed that the key can be used to refer
//! to some value external to the graph. All edges in the graph have the same
//! type - either directed or undirected.

use std::collections::{HashMap, LinkedList};

#[derive(Debug, PartialOrd, PartialEq)]
pub enum EdgeKind {
    Directed,
    Undirected,
}

/// The list of nodes in the entire graph
type NodeList<T> = Vec<Node<T>>;

/// A path in the graph is a sequence of nodes (their keys)
/// ordered from src to dest
pub type GraphPath<T> = Vec<T>;

#[derive(Debug)]
pub struct Graph<NodeType: Ord + Copy> {
    /// We keep all the nodes in the graph in a single list
    nodes: Option<NodeList<NodeType>>,

    /// A map from the id of a node to the index at which
    /// it is stored in the list of nodes
    nodes_mapper: Option<HashMap<NodeType, usize>>,

    /// Similarly, we keed all our edges in a single list
    edges: Option<Vec<Edge>>,

    /// A graph can either be directed or undirected
    kind: EdgeKind,
}

impl<NodeType: Ord + Copy> Graph<NodeType> {
    /// Create a new graph instance of the type
    /// `graph_kind`
    pub fn new(graph_kind: EdgeKind) -> Self {
        Graph {
            nodes: None,
            edges: None,
            kind: graph_kind,
            nodes_mapper: None,
        }
    }

    /// Adds a new node with the given id into the graph. If the graph
    /// already contains a node with that id, this method will return an
    /// error (It will not insert the duplicate node)
    pub fn add_node(&mut self, node_id: NodeType) {
        // TODO: Result and Error Object
        todo! {}
    }

    /// Add an edge between the two nodes. If either one of the nodes is
    /// or both of the nodes are not in the graph, this method will
    /// create the missing node(s) with the given node values and add an
    /// edge between them. The kind of edge that is added depends
    /// on the kind that was defined during the graph's creation.
    /// In case of EdgeKind::Directed, we assume that `left` is the
    /// source vertex while `right` is the destination vartex. That is,
    /// the added edge will be from left to right
    pub fn add_edge(&mut self, left: NodeType, right: NodeType) {
        todo! {}
    }

    /// Find the unweightes shortest path from the source to every other
    /// node in the graph
    pub fn bfs_path_from(&self, src: NodeType) -> GraphPath<NodeType> {
        todo!()
    }

    pub fn bellman_ford(&self, src: NodeType) -> GraphPath<NodeType> {
        todo!()
    }

    pub fn dijkstra(&self, src: NodeType) -> GraphPath<NodeType> {
        todo!()
    }

    pub fn matrix_ap_shortest_path(&self, src: NodeType) -> GraphPath<NodeType> {
        todo!()
    }

    pub fn floyd_warshall(&self, src: NodeType) -> GraphPath<NodeType> {
        todo!()
    }

    pub fn johnsons(&self, src: NodeType) -> GraphPath<NodeType> {
        todo!()
    }

    /// Uses Depth First Search to find the conneted components of the graph
    pub fn strongly_connected_components(&self) {
        // Tarjan or Kosoraju
        todo!()
    }

    /// Uses DFS to find a topological sorting if the nodes in the graph provided that the
    /// graph is a directed acyclic graph.
    pub fn topological_sort(&self) {
        todo!()
    }

    pub fn prim(&self) {
        todo!()
    }

    pub fn kruskal(&self) {
        todo!()
    }

    pub fn ford_fulkerson(&self, s: NodeType, t: NodeType) {
        todo!()
    }

    pub fn edmonds_karp(&self, s: NodeType, t: NodeType) {
        todo!()
    }

    pub fn dinics(&self, s: NodeType, t: NodeType) {
        todo!()
    }

    pub fn relabel_to_front(&self, s: NodeType, t: NodeType) {
        todo!()
    }
}

#[derive(Debug)]
struct Node<NodeType: Ord> {
    /// The id of this node
    key: NodeType,

    /// The list of other nodes in the graph to which
    /// this node has a direct connection
    neighbors: Option<LinkedList<usize>>,
}

impl<NodeType: Ord> Node<NodeType> {
    pub fn new(key: NodeType) -> Self {
        Node {
            key,
            neighbors: None,
        }
    }
}

#[derive(Debug)]
struct Edge {
    /// The left end of this edge. This is an index into `NodeMap<T>` the global
    /// list of vertices. If this Edge is directed, this is assumed to be the
    /// source node, that is this vertex is from u to v
    u: usize,

    /// The right end of this edge. This is an index into `NodeMap<T>` the global
    /// list of vertices. If this Edge is directed, this is assumed to be the
    /// destination node, that is this vertex is from u to v
    v: usize,

    /// Whether this edge is directed or undirected. Note that
    /// most graph algorithms assume that all edges have a uniform
    /// EdgeKind. That is either the graph is directed or undirected
    kind: EdgeKind,

    /// This value is used if we are dealing with a weighted graph.
    weight: Option<f64>,
}

impl Edge {
    pub fn new(u: usize, v: usize, kind: EdgeKind) -> Self {
        Edge {
            u,
            v,
            kind,
            weight: None,
        }
    }

    pub fn set_weight(&mut self, weight: f64) {
        self.weight = Some(weight)
    }
}
