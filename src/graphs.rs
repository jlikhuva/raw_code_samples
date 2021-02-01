//! A graph G = (V, E) is a collection of vertices and edges between
//! those vertices. Each Node has an associated key. For instance,
//! a graph of computers in a network could have the IP address of
//! each machine as the key of a node. This key identifies a node
//! In this implementation, we do not include the payload as part
//! of the node. Thus, it is assumed that the key can be used to refer
//! to some value external to the graph. All edges in the graph have the same
//! type - either directed or undirected.
use edge::*;
use graph_states::*;
use handles::{EdgeHandle, Entry, NodeHandle};
use std::hash::Hash;
use std::{collections::HashMap, marker::PhantomData};

/// We use the new type index pattern to make our code more
/// understandable. Using indexes to simulate pointers
/// can lead to opaqueness. Using a concrete type instead of
/// raw indexes ameliorates this. The 'insane' derives ensures
/// that the new type has all the crucial properties of the
/// underlying index
pub mod handles {
    use super::{Edge, Node};
    use std::marker::PhantomData;

    /// We use the new type index pattern to make our code more
    /// understandable. This is a handle that can be used to access
    /// nodes
    #[derive(Debug, Clone, Ord, PartialOrd, Eq, PartialEq, Hash)]
    pub struct NodeHandle<T: Ord>(pub(super) usize, pub(super) PhantomData<T>);

    impl<K: Ord, V> std::ops::Index<NodeHandle<K>> for Vec<Node<K, V>> {
        type Output = Node<K, V>;

        /// This allows us to use NodeHandle directly as an index
        fn index(&self, index: NodeHandle<K>) -> &Self::Output {
            &self[index.0]
        }
    }

    impl<K: Ord, V> std::ops::IndexMut<NodeHandle<K>> for Vec<Node<K, V>> {
        fn index_mut(&mut self, index: NodeHandle<K>) -> &mut Self::Output {
            &mut self[index.0]
        }
    }

    /// We use the new type index pattern to make our code more
    /// understandable. This is a handle that can be used to access
    /// edges
    #[derive(Debug, Clone, Ord, PartialOrd, Eq, PartialEq, Hash)]
    pub struct EdgeHandle(pub(super) usize);

    impl<K> std::ops::Index<EdgeHandle> for Vec<Box<dyn Edge<K>>> {
        type Output = Box<dyn Edge<K>>;

        /// This allows us to use NodeHandle directly as an index
        fn index(&self, index: EdgeHandle) -> &Self::Output {
            &self[index.0]
        }
    }

    impl<K> std::ops::IndexMut<EdgeHandle> for Vec<Box<dyn Edge<K>>> {
        fn index_mut(&mut self, index: EdgeHandle) -> &mut Self::Output {
            &mut self[index.0]
        }
    }

    /// A single entry in the tree
    #[derive(Debug)]
    pub struct Entry<K, V> {
        pub(super) key: K,
        pub(super) value: V,
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
        fn from((key, value): (K, V)) -> Self {
            Entry { key, value }
        }
    }
}

/// turn to iterator
pub type GraphPath<T> = Vec<T>;

#[derive(Debug)]
struct Node<K: Ord, V> {
    entry: Entry<K, V>,

    /// The start of the list of other nodes in the graph to which
    /// this node has a direct connection
    neighbors_head: Option<EdgeHandle>,
}

impl<K: Ord, V> Node<K, V> {
    pub fn new(entry: Entry<K, V>) -> Self {
        Node {
            entry,
            neighbors_head: None,
        }
    }

    pub fn set_entry(&mut self, entry: Entry<K, V>) {
        self.entry = entry;
    }
}

pub mod edge {
    use super::{EdgeHandle, NodeHandle};
    use std::fmt::Debug;
    pub trait Edge<K: Ord>: Debug {
        /// The first end of this edge. If the edge is directed
        /// then this is the source node. If not, then this is
        /// simply one of the nodes forming this edge
        fn first_end(&self) -> &NodeHandle<K>;

        /// The second end of this edge. If the graph is directed, then
        /// this is the destination node.
        fn second_end(&self) -> &NodeHandle<K>;

        /// Since we represent the neighbor list implicitly
        /// in the edges, this method gives us a handle to
        /// the next edge that either emanates from (if directed)
        /// or is connected to (if undirected) the handle returned
        /// by `first_end`
        fn next_neighbor(&self) -> Option<&EdgeHandle>;
    }

    pub trait WeightedEdge<K: Ord>: Edge<K> {
        /// For weighted edges, retrieve their weight
        fn weight(&self) -> f32;
    }

    /// An undirected, unweighted edge
    /// (left_node_idx, right_node_idx)
    #[derive(Debug)]
    pub struct UnweightedUndirected<K: Ord> {
        pub(super) left: NodeHandle<K>,
        pub(super) right: NodeHandle<K>,
        pub(super) next: Option<EdgeHandle>,
    }

    impl<K: Ord + Debug> Edge<K> for UnweightedUndirected<K> {
        fn first_end(&self) -> &NodeHandle<K> {
            &self.left
        }

        fn second_end(&self) -> &NodeHandle<K> {
            &self.right
        }

        fn next_neighbor(&self) -> Option<&EdgeHandle> {
            self.next.as_ref()
        }
    }

    /// A directed, unweighted edge
    /// (src_node_idx, dest_node_idx)
    #[derive(Debug)]
    pub struct UnweightedDirected<K: Ord> {
        pub(super) src: NodeHandle<K>,
        pub(super) dest: NodeHandle<K>,
        pub(super) next: Option<EdgeHandle>,
    }

    impl<K: Ord + Debug> Edge<K> for UnweightedDirected<K> {
        fn first_end(&self) -> &NodeHandle<K> {
            &self.src
        }

        fn second_end(&self) -> &NodeHandle<K> {
            &self.dest
        }

        fn next_neighbor(&self) -> Option<&EdgeHandle> {
            self.next.as_ref()
        }
    }

    /// An undirected, weighted edge
    /// (left_node_idx, right_node_idx, weight)
    #[derive(Debug)]
    pub struct WeightedUndirected<K: Ord> {
        pub(super) left: NodeHandle<K>,
        pub(super) right: NodeHandle<K>,
        pub(super) weight: f32,
        pub(super) next: Option<EdgeHandle>,
    }

    impl<K: Ord + Debug> Edge<K> for WeightedUndirected<K> {
        fn first_end(&self) -> &NodeHandle<K> {
            &self.left
        }

        fn second_end(&self) -> &NodeHandle<K> {
            &self.right
        }

        fn next_neighbor(&self) -> Option<&EdgeHandle> {
            self.next.as_ref()
        }
    }
    impl<K: Ord + Debug> WeightedEdge<K> for WeightedUndirected<K> {
        fn weight(&self) -> f32 {
            self.weight
        }
    }

    /// A weighted, directed edge
    /// (src_node_idx, dest_node_idx, weight)
    #[derive(Debug)]
    pub struct WeightedDirected<K: Ord> {
        pub(super) src: NodeHandle<K>,
        pub(super) dest: NodeHandle<K>,
        pub(super) weight: f32,
        pub(super) next: Option<EdgeHandle>,
    }
    impl<K: Ord + Debug> Edge<K> for WeightedDirected<K> {
        fn first_end(&self) -> &NodeHandle<K> {
            &self.src
        }

        fn second_end(&self) -> &NodeHandle<K> {
            &self.dest
        }

        fn next_neighbor(&self) -> Option<&EdgeHandle> {
            self.next.as_ref()
        }
    }
    impl<K: Ord + Debug> WeightedEdge<K> for WeightedDirected<K> {
        fn weight(&self) -> f32 {
            self.weight
        }
    }
}

/// Types representing the different types
/// of graphs that we can build
pub mod graph_states {
    pub trait GraphWeight {}
    pub trait GraphDirection {}
    #[derive(Debug)]
    pub struct Directed;
    impl GraphDirection for Directed {}
    #[derive(Debug)]
    pub struct Weighted;
    impl GraphWeight for Weighted {}
    #[derive(Debug)]
    pub struct Undirected;
    impl GraphDirection for Undirected {}
    #[derive(Debug)]
    pub struct Unweighted;
    impl GraphWeight for Unweighted {}
}

#[derive(Debug, Default)]
pub struct Graph<K: Ord + Hash, V, D: GraphDirection, W: GraphWeight> {
    /// We keep all the nodes in the graph in a single list
    nodes: Option<Vec<Node<K, V>>>,

    /// A map from the id of a node to the index at which
    /// it is stored in the list of nodes
    nodes_mapper: Option<HashMap<K, NodeHandle<K>>>,

    /// Similarly, we need all our edges in a single list
    edges: Option<Vec<Box<dyn Edge<K>>>>,

    /// The set of all edges in the graph
    edges_set: Option<HashMap<(NodeHandle<K>, NodeHandle<K>), EdgeHandle>>,

    /// type states
    _kind: (PhantomData<D>, PhantomData<W>),
}

/// Implementations of procedures that apply to all graph
/// types
impl<K: Ord + Hash + Clone, V, D: GraphDirection, W: GraphWeight> Graph<K, V, D, W> {
    /// Create a new graph instance
    pub fn new() -> Self {
        Graph {
            nodes: None,
            edges: None,
            nodes_mapper: None,
            edges_set: None,
            _kind: (PhantomData, PhantomData),
        }
    }

    /// Adds a new node with the given id into the graph. If the graph
    /// already contains a node with that id, it replaces the old entry.
    /// The procedure returns a handle to the added node.
    pub fn add_node(&mut self, entry: (K, V)) -> NodeHandle<K> {
        let entry = entry.into();
        match &mut self.nodes_mapper {
            None => {
                // The graph is empty. Let's create a new node and add it to the graph.
                let nodes = vec![Node::new(entry)];
                let mut mapper = HashMap::new();
                let new_handle = NodeHandle(mapper.len(), PhantomData);
                mapper.insert(nodes[0].entry.key.clone(), new_handle.clone());
                self.nodes_mapper = Some(mapper);
                self.nodes = Some(nodes);
                new_handle.clone()
            }
            Some(mapper) => {
                match mapper.get(&entry.key) {
                    Some(handle) => {
                        // The graph already contains this node. Lets replace its entry with the new entry
                        self.nodes
                            .as_mut()
                            .and_then(|nodes| Some(nodes[handle.clone()].set_entry(entry)));
                        handle.clone()
                    }
                    None => {
                        // This node is not in the graph, lets add it to both the list of nodes
                        // and the nodes mapper
                        let new_handle = NodeHandle(mapper.len(), PhantomData);
                        mapper.insert(entry.key.clone(), new_handle.clone());
                        let new_node = Node::new(entry);
                        self.nodes.as_mut().and_then(|nodes| Some(nodes.push(new_node)));
                        new_handle.clone()
                    }
                }
            }
        }
    }
}

impl<K: Ord + Hash + Clone, V> Graph<K, V, Directed, Weighted> {
    /// Add an edge between the two nodes in a weighted directed graph.
    /// Both ends of the edge have to be in the graph. If an edge already
    /// exists between the two nodes, we update the weight without
    /// adding in a new edge
    pub fn add_edge(&mut self, ends: (K, K)) -> EdgeHandle {
        todo!()
    }
}

/// Add an edge between the two nodes in a weighted undirected graph
impl<K: Ord + Hash + Clone, V> Graph<K, V, Undirected, Weighted> {
    pub fn add_edge(&mut self, edge: WeightedUndirected<K>) -> EdgeHandle {
        todo!()
    }
}

/// Add an edge between the two nodes in an unweighted directed graph
impl<K: Ord + Hash + Clone, V> Graph<K, V, Directed, Unweighted> {
    pub fn add_edge(&mut self, edge: UnweightedDirected<K>) -> EdgeHandle {
        todo!()
    }
}

/// Add an edge between the two nodes in an unweighted undirected graph
impl<K: Ord + Hash + Clone, V> Graph<K, V, Undirected, Unweighted> {
    pub fn add_edge(&mut self, edge: UnweightedUndirected<K>) -> EdgeHandle {
        todo!()
    }
}

impl<K: Ord + Hash + Clone, V> Graph<K, V, Undirected, Unweighted> {
    /// Find the unweighted shortest path from the source to every other
    /// node in the graph
    pub fn bfs_path_from(&self, src: K) -> GraphPath<K> {
        todo!()
    }

    pub fn bellman_ford(&self, src: K) -> GraphPath<K> {
        todo!()
    }

    pub fn dijkstra(&self, src: K) -> GraphPath<K> {
        todo!()
    }

    pub fn matrix_ap_shortest_path(&self, src: K) -> GraphPath<K> {
        todo!()
    }

    pub fn floyd_warshall(&self, src: K) -> GraphPath<K> {
        todo!()
    }

    pub fn johnsons(&self, src: K) -> GraphPath<K> {
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

    pub fn ford_fulkerson(&self, s: K, t: K) {
        todo!()
    }

    pub fn edmonds_karp(&self, s: K, t: K) {
        todo!()
    }

    pub fn dinics(&self, s: K, t: K) {
        todo!()
    }

    pub fn relabel_to_front(&self, s: K, t: K) {
        todo!()
    }
}
