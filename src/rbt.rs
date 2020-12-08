/// The nodes of a red blck tree can take on either
/// one of two values, Red, or Black.
/// By constraining the node colors on any
/// simple path from the root to a leaf, red-black trees
/// ensure that no such path is more
/// than twice as long as any other, so that the
/// tree is approximately balanced
#[derive(Debug)]
pub enum Color {
    Red,
    Black,
}
