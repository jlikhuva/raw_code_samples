//! Algorithms for solving geometric problems take as input a description of a set of
//! geometric objects, the most primitive of which is a point in some d-dimensional
//! space. With such a description,  such algorithms are able to answer several ad-hoc
//! queries about the set of objects.

use num_traits::Num;
use std::ops::Sub;
#[derive(Debug)]
pub enum Direction {
    Clockwise,
    CounterClockwise,
    Colinear,
}

#[derive(Debug)]
pub enum Turn {
    /// To move from the first line segment to the second line
    /// segment, take a right turn at the hared point
    Right,

    /// To move from the first line segment to the second line
    /// segment, take a left turn at the hared point
    Left,

    /// This means that the two line segments are colinear.
    /// Therefore, when a person arrives at the shared point,
    /// they simply continue walking straight
    NoTurn,

    /// This is technically an error. We can only answer the
    /// `turn_at_common_point` query if the two points in question
    /// do have a common point.
    NoCommonPoint,
}

/// The available algorithms for computing the convex hull
/// of a set of 2-dimensional points. Useful for benchmarking
#[derive(Debug)]
pub enum ConvexHullMethod {
    Incremental,
    /// PruneAndSearch,
    KirkpatrickSeidel,
    GrahamScan,
    JarvisMarch,
}

/// A point in 2-dimensional space
#[derive(Debug)]
pub struct Point<T: Num> {
    x: T,
    y: T,
}

/// A group of points
#[derive(Debug)]
pub struct PointCollection<T: Num>(Vec<Point<T>>);

impl<T: Num> Sub for Point<T> {
    type Output = Self;

    fn sub(self, other: Self) -> Self {
        Self {
            x: self.x - other.x,
            y: self.y - other.y,
        }
    }
}

impl<T: Num + Copy> Point<T> {
    /// Create a new point with the provided x, and y coordinates
    pub fn new(x: T, y: T) -> Self {
        Point { x, y }
    }

    /// The cross product of two 2-dimensional points
    pub fn cross(&self, other: &Point<T>) -> T {
        self.x * other.y - other.x * self.y
    }
}

/// A line segment in 2-dimensional space is defined by
/// its two endpoints.
#[derive(Debug)]
pub struct LineSegment<T: Num> {
    left: Point<T>,
    right: Point<T>,
}

/// A collection of line segments. Such an abstraction is useful
/// when we want to ask queries that operate on some group of line
/// segments
#[derive(Debug)]
pub struct SegmentCollection<T: Num>(Vec<LineSegment<T>>);

/// Fundamental Algorithms on Line Segments
impl<T: Num> LineSegment<T> {
    pub fn new(left: Point<T>, right: Point<T>) -> Self {
        LineSegment { left, right }
    }

    // suppose that this segment, and the `other` segment are directed,
    // that is, they are vectors from a common origin point, what is the direction
    // of this segment from the other segment?
    pub fn direction(&self, other: &LineSegment<T>) -> Direction {
        todo!()
    }

    pub fn turn_at_common_point(&self, other: &LineSegment<T>) -> Turn {
        todo!()
    }

    pub fn intersect(&self, other: &LineSegment<T>) -> bool {
        todo!()
    }
}

impl<T: Num> PointCollection<T> {
    pub fn new() -> Self {
        PointCollection(Vec::new())
    }

    /// Add a point into the collection. Since we expect points in the
    /// collection to be unique, we'll only add a point if it's not
    /// already in the collection
    pub fn insert(&mut self, p: Point<T>) {
        todo!()
    }

    /// Compute the convex hull for the points in this collection using
    /// the stated algorithm
    pub fn convex_hull(&self, method: ConvexHullMethod) -> PointCollection<T> {
        todo!()
    }

    pub fn closest_pair(&self) -> (Point<T>, Point<T>) {
        todo!()
    }
}

impl<T: Num> SegmentCollection<T> {
    pub fn new() -> Self {
        SegmentCollection(Vec::new())
    }

    /// Add a new segment into the collection.
    pub fn insert(&mut self, s: LineSegment<T>) {
        todo!()
    }

    pub fn any_pair_intersect(&self) -> bool {
        todo!()
    }
}
