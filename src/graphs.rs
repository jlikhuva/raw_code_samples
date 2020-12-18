//! A graph G = (V, E) is a collection of vertices and edges between
//! those vertices. Each Node has an associated key. For instance,
//! a graph of computers in a network could have the IP address of
//! each machine as the key of a node. This key identifies a node
//! In this implementation, we do not include the payload as part
//! of the node. Thus, it is assumed that the key can be used to refer
//! to some value external to the graph. All edges in the graph have the same
//! type - either directed or undirected.
use std::{
    collections::{HashMap, LinkedList},
    hash::Hash,
};

pub type GraphPath<T> = Vec<T>;

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
pub enum Edge {
    /// An undirected, unweighted edge
    /// (left_node_idx, right_node_idx)
    UnweightedUndirected(usize, usize),

    /// An undirected, weighted edge
    /// (left_node_idx, right_node_idx, weight)
    WeightedUndirected(usize, usize, f32),

    /// A directed, unweighted edge
    /// (src_node_idx, dest_node_idx)
    UnweightedDirected(usize, usize),

    /// A weighted, directed edge
    /// (src_node_idx, dest_node_idx, weight)
    WeightedDirected(usize, usize, f32),
}

#[derive(Debug)]
pub struct Graph<NodeType: Ord + Hash + Clone> {
    /// We keep all the nodes in the graph in a single list
    nodes: Option<Vec<Node<NodeType>>>,

    /// A map from the id of a node to the index at which
    /// it is stored in the list of nodes
    nodes_mapper: Option<HashMap<NodeType, usize>>,

    /// Similarly, we keed all our edges in a single list
    /// is this necessary?
    edges: Option<Vec<Edge>>,
}

impl<NodeType: Ord + Hash + Clone> Graph<NodeType> {
    /// Create a new graph instance of the type
    /// `graph_kind`
    pub fn new() -> Self {
        Graph {
            nodes: None,
            edges: None,
            nodes_mapper: None,
        }
    }

    /// Adds a new node with the given id into the graph. If the graph
    /// already contains a node with that id, this method will return an
    /// error (It will not insert the duplicate node)
    pub fn add_node(&mut self, node_id: NodeType) {
        let new_node = Node::new(node_id.clone());
        match &mut self.nodes_mapper {
            None => {
                self.nodes = Some(vec![new_node]);
                let mut mapper = HashMap::new();
                mapper.insert(node_id, 0);
                self.nodes_mapper = Some(mapper);
            }
            Some(mapper) => {
                match mapper.get(&node_id) {
                    Some(_idx) => {
                        // Duplicate node Error
                    }
                    None => {
                        let nodes = self.nodes.as_mut().unwrap();
                        let idx = nodes.len();
                        nodes.push(new_node);
                        mapper.insert(node_id, idx);
                    }
                }
            }
        }
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
