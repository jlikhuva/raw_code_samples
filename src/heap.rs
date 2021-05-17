/// A binary heap data structure implemented using
/// a list. The heap has two key properties: heap_size and
/// heap_array.len(). The former gives us the number
/// of items in the heap -- note that some items can be in the
/// heap array, but not in the heap
/// (for instance, when doing extract operation we move deleted
/// items to the end of the array and decrement heap_size in order to ignore them).
/// The max heap property states that the children must always be greater than
/// the parent.
#[derive(Debug)]
pub struct MaxHeap<T>
where
    T: PartialEq + PartialOrd + std::fmt::Debug + Copy,
{
    heap_array: Vec<T>,
    heap_size: usize,
}

/// A d-array heap is  like a binary heap with one possible exception -
/// non-leaf nodes can have a maximum of d children instead of 2 children.
#[derive(Debug)]
pub struct DArrayMaxHeap<T>
where
    T: PartialEq + PartialOrd + std::fmt::Debug + Copy,
{
    heap_array: Vec<T>,
    heap_size: usize,
    d: usize,
}

pub enum Relative {
    Parent,
    LeftChild,
    RightChild,
}

#[derive(Debug)]
pub enum MaxHeapError {
    HeapUnderflowError,
    SmallerNewKeyError,
}

type Result<T> = std::result::Result<T, MaxHeapError>;

macro_rules! relative {
    ($i: ident, Relative::Parent) => {
        (($i + 1) >> 1) - 1
    };
    ($i: ident, Relative::LeftChild) => {
        ((($i + 1) << 1) + 1) - 1
    };
    ($i: ident, Relative::RightChild) => {
        (2 * ($i + 1)) - 1
    };
}

impl<T: PartialEq + PartialOrd + std::fmt::Debug + Copy> MaxHeap<T> {
    pub fn new_heap(items: Vec<T>) -> Result<Self> {
        let mut heap = MaxHeap {
            heap_size: items.len() - 1,
            heap_array: items,
        };
        heap.build_max_heap();
        Ok(heap)
    }

    fn build_max_heap(&mut self) {
        let midpoint = self.heap_array.len() / 2;
        for i in (0..midpoint + 1).rev() {
            self.max_heapify(i);
        }
    }

    // The children of i are the roots of valid MaxHeaps
    // However, i, may be smaller than at least one of the children
    // thus breaking the MaxHeap Invariant. To fix this, we let it
    // bubble down to its rightful place.
    fn max_heapify(&mut self, i: usize) {
        let left_child = relative!(i, Relative::LeftChild);
        let right_child = relative!(i, Relative::RightChild);
        let mut largest = i;
        let items = &mut self.heap_array;

        if (left_child <= self.heap_size) && (items[left_child] > items[largest]) {
            largest = left_child;
        }
        if (right_child <= self.heap_size) && (items[right_child] > items[largest]) {
            largest = right_child;
        }

        if largest != i {
            items.swap(i, largest);
            self.max_heapify(largest);
        }
    }

    /// Produces the items in the heap in sorted order (ascending)
    /// Note that this procedure consumes the heap, meaning that
    /// the heap can no longer be used afterwards.
    pub fn heap_sort(mut self) -> Result<Vec<T>> {
        for i in (1..=self.heap_size).rev() {
            self.heap_array.swap(i, 0);
            self.heap_size -= 1;
            self.max_heapify(0);
        }
        Ok(self.heap_array)
    }

    #[inline(always)]
    fn get_max_value(&self) -> Result<T> {
        match self.heap_array.get(0) {
            None => Err(MaxHeapError::HeapUnderflowError),
            Some(&val) => Ok(val),
        }
    }

    pub fn extract_max(&mut self) -> Result<T> {
        let max = self.get_max_value()?;
        self.heap_array.swap(0, self.heap_size);
        self.heap_size -= 1;
        self.max_heapify(0);
        Ok(max)
    }

    pub fn insert(&mut self, value: T, new_val_min: T) -> Result<()> {
        self.heap_size += 1;
        if self.heap_size == self.heap_array.len() {
            self.heap_array.push(new_val_min);
        } else {
            self.heap_array[self.heap_size] = new_val_min;
        }
        self.increase_key(self.heap_size, value)
    }

    pub fn increase_key(&mut self, mut at: usize, new_val: T) -> Result<()> {
        if new_val < self.heap_array[at] {
            Err(MaxHeapError::SmallerNewKeyError)
        } else {
            let items = &mut self.heap_array;
            items[at] = new_val;
            while at >= 1 && items[relative!(at, Relative::Parent)] < items[at] {
                items.swap(at, relative!(at, Relative::Parent));
                at = relative!(at, Relative::Parent);
            }
            Ok(())
        }
    }
}

impl<T> DArrayMaxHeap<T> where T: PartialEq + PartialOrd + std::fmt::Debug + Copy {}

#[cfg(test)]
mod test {
    use super::MaxHeap;
    #[test]
    fn test_macros() {
        let mut r = 1;
        assert_eq!(0, relative!(r, Relative::Parent));
        assert_eq!(3, relative!(r, Relative::RightChild));
        assert_eq!(4, relative!(r, Relative::LeftChild));

        r = 5;
        assert_eq!(2, relative!(r, Relative::Parent));
        assert_eq!(11, relative!(r, Relative::RightChild));
        assert_eq!(12, relative!(r, Relative::LeftChild))
    }

    #[test]
    fn test_heap() {
        // Test Create and HeapSort
        let v = vec![4, 3, 5, 8, 9, 1, 0, 2];
        let heap = MaxHeap::new_heap(v);
        assert_eq!(
            vec![0, 1, 2, 3, 4, 5, 8, 9],
            heap.and_then(|heap| heap.heap_sort()).unwrap()
        );

        // Test Extract Max
        let v = vec![4, 3, 5, 8, 9, 1, 0];
        let mut heap = MaxHeap::new_heap(v).unwrap();
        assert_eq!(9, heap.extract_max().unwrap_or(0));
        assert_eq!(8, heap.extract_max().unwrap_or(0));
        println!("{:?}", heap);

        // Test insert
        let v = vec![4, 3, 5, 8, 9, 1, 0];
        let mut heap = MaxHeap::new_heap(v).unwrap();
        heap.insert(78, i32::MIN).unwrap();
        assert_eq!(78, heap.extract_max().unwrap_or(0));
        println!("{:?}", heap);

        // Test insert after extract
        let v = vec![4, 3, 5, 8, 9, 1, 0];
        let mut heap = MaxHeap::new_heap(v).unwrap();
        heap.insert(78, i32::MIN).unwrap();
        assert_eq!(78, heap.extract_max().unwrap_or(0));
        heap.insert(78, i32::MIN).unwrap();
        assert_eq!(78, heap.extract_max().unwrap_or(0));
        heap.insert(44, i32::MIN).unwrap();
        heap.insert(15, i32::MIN).unwrap();
        assert_eq!(44, heap.extract_max().unwrap_or(0));
        println!("{:?}", heap);
    }
}
