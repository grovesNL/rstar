use crate::algorithm::selection_functions::*;
use crate::node::{ParentNode, RTreeNode};
use crate::object::RTreeObject;
use core::convert::Infallible;
use core::mem::{replace, take};
use core::ops::ControlFlow;
use core::slice::{Iter, IterMut};

#[cfg(doc)]
use crate::RTree;

use smallvec::SmallVec;

pub use super::intersection_iterator::IntersectionIterator;
pub use super::nearest_neighbor::{NearestNeighborDistance2Iterator, NearestNeighborIterator};
pub use super::removal::{DrainIterator, IntoIter};

/// Iterator returned by [`RTree::locate_all_at_point`].
pub type LocateAllAtPoint<'a, T> = SelectionIterator<'a, T, SelectAtPointFunction<T>>;
/// Iterator returned by [`RTree::locate_all_at_point_mut`].
pub type LocateAllAtPointMut<'a, T> = SelectionIteratorMut<'a, T, SelectAtPointFunction<T>>;

/// Iterator returned by [`RTree::locate_in_envelope`].
pub type LocateInEnvelope<'a, T> = SelectionIterator<'a, T, SelectInEnvelopeFunction<T>>;
/// Iterator returned by [`RTree::locate_in_envelope_mut`].
pub type LocateInEnvelopeMut<'a, T> = SelectionIteratorMut<'a, T, SelectInEnvelopeFunction<T>>;

/// Iterator returned by [`RTree::locate_in_envelope_intersecting`].
pub type LocateInEnvelopeIntersecting<'a, T> =
    SelectionIterator<'a, T, SelectInEnvelopeFuncIntersecting<T>>;
/// Iterator returned by [`RTree::locate_in_envelope_intersecting_mut`].
pub type LocateInEnvelopeIntersectingMut<'a, T> =
    SelectionIteratorMut<'a, T, SelectInEnvelopeFuncIntersecting<T>>;

/// Iterator returned by [`RTree::iter`].
pub type RTreeIterator<'a, T> = SelectionIterator<'a, T, SelectAllFunc>;
/// Iterator returned by [`RTree::iter_mut`].
pub type RTreeIteratorMut<'a, T> = SelectionIteratorMut<'a, T, SelectAllFunc>;

/// Iterator returned by [`RTree::locate_within_distance`].
pub type LocateWithinDistanceIterator<'a, T> =
    SelectionIterator<'a, T, SelectWithinDistanceFunction<T>>;

/// The number of ancestors of the current node that the selection iterators store inline.
const INLINE_ANCESTORS: usize = 11;

/// Iterator returned by `RTree::locate_*` methods.
pub struct SelectionIterator<'a, T, Func>
where
    T: RTreeObject + 'a,
    Func: SelectionFunction<T>,
{
    func: Func,
    /// The remaining children of the current node.
    ///
    /// Each child is only tested when the iterator reaches it. `should_unpack_leaf` can be far more
    /// expensive than testing an envelope, and it must not be called for elements that the iterator
    /// never reaches (for example, because it's dropped before the end).
    current: Iter<'a, RTreeNode<T>>,
    /// The remaining children of each ancestor of the current node.
    ancestors: SmallVec<[Iter<'a, RTreeNode<T>>; INLINE_ANCESTORS]>,
}

impl<'a, T, Func> SelectionIterator<'a, T, Func>
where
    T: RTreeObject,
    Func: SelectionFunction<T>,
{
    #[inline]
    pub(crate) fn new(root: &'a ParentNode<T>, func: Func) -> Self {
        // Do not call `should_unpack_parent` on an empty root as
        // its AABB is pathological and might make that function panic.
        let current = if !root.children.is_empty() && func.should_unpack_parent(&root.envelope) {
            root.children.iter()
        } else {
            Default::default()
        };

        Self {
            func,
            current,
            ancestors: SmallVec::new(),
        }
    }
}

impl<'a, T, Func> Iterator for SelectionIterator<'a, T, Func>
where
    T: RTreeObject,
    Func: SelectionFunction<T>,
{
    type Item = &'a T;

    #[inline]
    fn next(&mut self) -> Option<&'a T> {
        // Taken out of `self` so the compiler can keep it in registers, and put back on return. If
        // a selection function panics, the iterator then skips the rest of the current node.
        let mut current = take(&mut self.current);
        loop {
            while let Some(child) = current.next() {
                match child {
                    RTreeNode::Leaf(t) => {
                        if self.func.should_unpack_leaf(t) {
                            self.current = current;
                            return Some(t);
                        }
                    }
                    RTreeNode::Parent(data) => {
                        if self.func.should_unpack_parent(&data.envelope) {
                            let siblings = replace(&mut current, data.children.iter());
                            self.ancestors.push(siblings);
                        }
                    }
                }
            }

            current = self.ancestors.pop()?;
        }
    }

    #[inline]
    fn fold<B, F>(self, init: B, mut f: F) -> B
    where
        F: FnMut(B, Self::Item) -> B,
    {
        let mut f = |acc: B, t: &'a T| ControlFlow::<Infallible, B>::Continue(f(acc, t));
        // The rest of the current node first, then the rest of each ancestor, innermost first.
        let ControlFlow::Continue(mut acc) =
            try_fold_children(self.current.as_slice(), &self.func, init, &mut f);
        for siblings in self.ancestors.into_iter().rev() {
            let ControlFlow::Continue(next) =
                try_fold_children(siblings.as_slice(), &self.func, acc, &mut f);
            acc = next;
        }

        acc
    }
}

/// Folds the elements selected by `func` in `children` and below them with `f`, in order, until `f`
/// breaks.
///
/// Shared by [`SelectionIterator::fold`] and [`select_nodes`].
fn try_fold_children<'a, T, Func, B, R, F>(
    children: &'a [RTreeNode<T>],
    func: &Func,
    mut acc: B,
    f: &mut F,
) -> ControlFlow<R, B>
where
    T: RTreeObject,
    Func: SelectionFunction<T>,
    F: FnMut(B, &'a T) -> ControlFlow<R, B>,
{
    for child in children {
        match child {
            RTreeNode::Leaf(t) => {
                if func.should_unpack_leaf(t) {
                    acc = f(acc, t)?;
                }
            }
            RTreeNode::Parent(data) => {
                if func.should_unpack_parent(&data.envelope) {
                    acc = try_fold_children(&data.children, func, acc, f)?;
                }
            }
        }
    }

    ControlFlow::Continue(acc)
}

/// Internal iteration variant of [`SelectionIterator`]
pub fn select_nodes<'a, T, Func, V, B>(
    root: &'a ParentNode<T>,
    func: &Func,
    visitor: &mut V,
) -> ControlFlow<B>
where
    T: RTreeObject,
    Func: SelectionFunction<T>,
    V: FnMut(&'a T) -> ControlFlow<B>,
{
    if !root.children.is_empty() && func.should_unpack_parent(&root.envelope) {
        try_fold_children(&root.children, func, (), &mut |(), t| visitor(t))?;
    }

    ControlFlow::Continue(())
}

/// Iterator type returned by `RTree::locate_*_mut` methods.
pub struct SelectionIteratorMut<'a, T, Func>
where
    T: RTreeObject + 'a,
    Func: SelectionFunction<T>,
{
    func: Func,
    /// See [`SelectionIterator::current`].
    current: IterMut<'a, RTreeNode<T>>,
    /// See [`SelectionIterator::ancestors`].
    ancestors: SmallVec<[IterMut<'a, RTreeNode<T>>; INLINE_ANCESTORS]>,
}

impl<'a, T, Func> SelectionIteratorMut<'a, T, Func>
where
    T: RTreeObject,
    Func: SelectionFunction<T>,
{
    #[inline]
    pub(crate) fn new(root: &'a mut ParentNode<T>, func: Func) -> Self {
        let current = if !root.children.is_empty() && func.should_unpack_parent(&root.envelope) {
            root.children.iter_mut()
        } else {
            Default::default()
        };

        Self {
            func,
            current,
            ancestors: SmallVec::new(),
        }
    }
}

impl<'a, T, Func> Iterator for SelectionIteratorMut<'a, T, Func>
where
    T: RTreeObject,
    Func: SelectionFunction<T>,
{
    type Item = &'a mut T;

    #[inline]
    fn next(&mut self) -> Option<&'a mut T> {
        // Like in `SelectionIterator::next`.
        let mut current = take(&mut self.current);
        loop {
            while let Some(child) = current.next() {
                match child {
                    RTreeNode::Leaf(t) => {
                        if self.func.should_unpack_leaf(t) {
                            self.current = current;
                            return Some(t);
                        }
                    }
                    RTreeNode::Parent(data) => {
                        if self.func.should_unpack_parent(&data.envelope) {
                            let siblings = replace(&mut current, data.children.iter_mut());
                            self.ancestors.push(siblings);
                        }
                    }
                }
            }

            current = self.ancestors.pop()?;
        }
    }

    #[inline]
    fn fold<B, F>(self, init: B, mut f: F) -> B
    where
        F: FnMut(B, Self::Item) -> B,
    {
        let mut f = |acc: B, t: &'a mut T| ControlFlow::<Infallible, B>::Continue(f(acc, t));
        // See `SelectionIterator::fold`.
        let ControlFlow::Continue(mut acc) =
            try_fold_children_mut(self.current.into_slice(), &self.func, init, &mut f);
        for siblings in self.ancestors.into_iter().rev() {
            let ControlFlow::Continue(next) =
                try_fold_children_mut(siblings.into_slice(), &self.func, acc, &mut f);
            acc = next;
        }

        acc
    }
}

/// See [`try_fold_children`].
fn try_fold_children_mut<'a, T, Func, B, R, F>(
    children: &'a mut [RTreeNode<T>],
    func: &Func,
    mut acc: B,
    f: &mut F,
) -> ControlFlow<R, B>
where
    T: RTreeObject,
    Func: SelectionFunction<T>,
    F: FnMut(B, &'a mut T) -> ControlFlow<R, B>,
{
    for child in children {
        match child {
            RTreeNode::Leaf(t) => {
                if func.should_unpack_leaf(t) {
                    acc = f(acc, t)?;
                }
            }
            RTreeNode::Parent(data) => {
                if func.should_unpack_parent(&data.envelope) {
                    acc = try_fold_children_mut(&mut data.children, func, acc, f)?;
                }
            }
        }
    }

    ControlFlow::Continue(acc)
}

/// Internal iteration variant of [`SelectionIteratorMut`]
pub fn select_nodes_mut<'a, T, Func, V, B>(
    root: &'a mut ParentNode<T>,
    func: &Func,
    visitor: &mut V,
) -> ControlFlow<B>
where
    T: RTreeObject,
    Func: SelectionFunction<T>,
    V: FnMut(&'a mut T) -> ControlFlow<B>,
{
    if !root.children.is_empty() && func.should_unpack_parent(&root.envelope) {
        try_fold_children_mut(&mut root.children, func, (), &mut |(), t| visitor(t))?;
    }

    ControlFlow::Continue(())
}

#[cfg(test)]
mod test {
    use crate::aabb::AABB;
    use crate::envelope::Envelope;
    use crate::object::RTreeObject;
    use crate::rtree::RTree;
    use crate::test_utilities::{create_random_points, create_random_rectangles, SEED_1};
    use crate::SelectionFunction;

    /// Selects points in the bottom half, but only in nodes that reach into the left half.
    struct SelectFunc;

    impl SelectionFunction<[f64; 2]> for SelectFunc {
        fn should_unpack_parent(&self, envelope: &AABB<[f64; 2]>) -> bool {
            envelope.lower()[0] < 0.5
        }

        fn should_unpack_leaf(&self, point: &[f64; 2]) -> bool {
            point[1] < 0.5
        }
    }

    fn sorted(mut points: Vec<[f64; 2]>) -> Vec<[f64; 2]> {
        points.sort_unstable_by(|lhs, rhs| lhs.partial_cmp(rhs).unwrap());
        points
    }

    #[test]
    fn test_root_node_is_not_always_unpacked() {
        use super::{select_nodes, select_nodes_mut};
        use core::ops::ControlFlow;

        struct SelectNoneFunc {}

        impl SelectionFunction<[i32; 2]> for SelectNoneFunc {
            fn should_unpack_parent(&self, _: &AABB<[i32; 2]>) -> bool {
                false
            }
        }

        /// Panics when it tests a parent.
        struct UntestedRootFunc {}

        impl SelectionFunction<[i32; 2]> for UntestedRootFunc {
            fn should_unpack_parent(&self, _: &AABB<[i32; 2]>) -> bool {
                panic!("the root of an empty tree was tested");
            }
        }

        let mut tree = RTree::bulk_load(vec![[0i32, 0]]);

        let mut elements = tree.locate_with_selection_function(SelectNoneFunc {});
        assert!(elements.next().is_none());
        drop(elements);

        let mut elements = tree.locate_with_selection_function_mut(SelectNoneFunc {});
        assert!(elements.next().is_none());
        drop(elements);

        let result = select_nodes(tree.root(), &SelectNoneFunc {}, &mut ControlFlow::Break);
        assert_eq!(result, ControlFlow::Continue(()));
        let result = select_nodes_mut(tree.root_mut(), &SelectNoneFunc {}, &mut ControlFlow::Break);
        assert_eq!(result, ControlFlow::Continue(()));

        // The root of an empty tree isn't tested at all.
        let mut tree = RTree::new();
        assert!(tree
            .locate_with_selection_function(UntestedRootFunc {})
            .next()
            .is_none());
        assert!(tree
            .locate_with_selection_function_mut(UntestedRootFunc {})
            .next()
            .is_none());
        let result = select_nodes(tree.root(), &UntestedRootFunc {}, &mut ControlFlow::Break);
        assert_eq!(result, ControlFlow::Continue(()));
        let result = select_nodes_mut(
            tree.root_mut(),
            &UntestedRootFunc {},
            &mut ControlFlow::Break,
        );
        assert_eq!(result, ControlFlow::Continue(()));
    }

    /// `should_unpack_leaf` may be expensive, so it must only be called for the leaves an
    /// iterator actually gets to.
    #[test]
    fn test_leaves_are_tested_lazily() {
        use core::cell::Cell;

        struct CountingFunc<'a>(&'a Cell<usize>);

        impl SelectionFunction<[f64; 2]> for CountingFunc<'_> {
            fn should_unpack_parent(&self, _: &AABB<[f64; 2]>) -> bool {
                true
            }

            fn should_unpack_leaf(&self, _: &[f64; 2]) -> bool {
                self.0.set(self.0.get() + 1);
                true
            }
        }

        let mut tree = RTree::bulk_load(create_random_points(1000, SEED_1));

        let tested_leaves = Cell::new(0);
        let mut elements = tree.locate_with_selection_function(CountingFunc(&tested_leaves));
        assert!(elements.next().is_some());
        assert_eq!(tested_leaves.get(), 1);
        assert!(elements.next().is_some());
        assert_eq!(tested_leaves.get(), 2);
        drop(elements);

        tested_leaves.set(0);
        let mut elements = tree.locate_with_selection_function_mut(CountingFunc(&tested_leaves));
        assert!(elements.next().is_some());
        assert_eq!(tested_leaves.get(), 1);
    }

    #[test]
    fn test_locate_all() {
        const NUM_RECTANGLES: usize = 400;
        let rectangles = create_random_rectangles(NUM_RECTANGLES, SEED_1);
        let tree = RTree::bulk_load(rectangles.clone());

        let query_points = create_random_points(20, SEED_1);

        for p in query_points {
            let contained_sequential: Vec<_> = rectangles
                .iter()
                .filter(|rectangle| rectangle.envelope().contains_point(&p))
                .cloned()
                .collect();

            let contained_rtree: Vec<_> = tree.locate_all_at_point(p).cloned().collect();

            contained_sequential
                .iter()
                .all(|r| contained_rtree.contains(r));
            contained_rtree
                .iter()
                .all(|r| contained_sequential.contains(r));
        }
    }

    #[test]
    fn test_locate_in_envelope() {
        let points = create_random_points(100, SEED_1);
        let tree = RTree::bulk_load(points.clone());
        let envelope = AABB::from_corners([0.5, 0.5], [1.0, 1.0]);
        let contained_in_envelope: Vec<_> = points
            .iter()
            .filter(|point| envelope.contains_point(point))
            .cloned()
            .collect();
        let len = contained_in_envelope.len();
        assert!(10 < len && len < 90, "unexpected point distribution");
        let located: Vec<_> = tree.locate_in_envelope(envelope).cloned().collect();
        assert_eq!(len, located.len());
        for point in &contained_in_envelope {
            assert!(located.contains(point));
        }
    }

    #[test]
    fn test_locate_with_selection_func() {
        use crate::SelectionFunction;

        struct SelectLeftOfZeroPointFiveFunc;

        impl SelectionFunction<[f64; 2]> for SelectLeftOfZeroPointFiveFunc {
            fn should_unpack_parent(&self, parent_envelope: &AABB<[f64; 2]>) -> bool {
                parent_envelope.lower()[0] < 0.5 || parent_envelope.upper()[0] < 0.5
            }

            fn should_unpack_leaf(&self, child: &[f64; 2]) -> bool {
                child[0] < 0.5
            }
        }

        let func = SelectLeftOfZeroPointFiveFunc;

        let points = create_random_points(100, SEED_1);
        let tree = RTree::bulk_load(points.clone());
        let iterative_count = points
            .iter()
            .filter(|leaf| func.should_unpack_leaf(leaf))
            .count();
        let selected = tree
            .locate_with_selection_function(func)
            .collect::<Vec<_>>();

        assert_eq!(iterative_count, selected.len());
        assert!(iterative_count > 20); // Make sure that we do test something interesting
        for point in &selected {
            assert!(point[0] < 0.5);
        }
    }

    #[test]
    fn test_iteration() {
        const NUM_POINTS: usize = 1000;
        let points = create_random_points(NUM_POINTS, SEED_1);
        let mut tree = RTree::new();
        for p in &points {
            tree.insert(*p);
        }
        let mut count = 0usize;
        for p in tree.iter() {
            assert!(points.iter().any(|q| q == p));
            count += 1;
        }
        assert_eq!(count, NUM_POINTS);
        count = 0;
        for p in tree.iter_mut() {
            assert!(points.iter().any(|q| q == p));
            count += 1;
        }
        assert_eq!(count, NUM_POINTS);
        for p in &points {
            assert!(tree.iter().any(|q| q == p));
            assert!(tree.iter_mut().any(|q| q == p));
        }
    }

    /// The iterators test the children of each node in order, and return its elements in order,
    /// whether they're consumed with `next`, `fold` or both.
    #[test]
    fn test_children_are_tested_in_order() {
        use crate::node::{ParentNode, RTreeNode};
        use core::cell::RefCell;

        #[derive(Clone, Copy, Debug, PartialEq)]
        enum Call {
            Parent(AABB<[f64; 2]>),
            Leaf([f64; 2]),
        }

        /// Selects every element, and records the calls it gets.
        struct RecordingFunc<'a>(&'a RefCell<Vec<Call>>);

        impl SelectionFunction<[f64; 2]> for RecordingFunc<'_> {
            fn should_unpack_parent(&self, envelope: &AABB<[f64; 2]>) -> bool {
                self.0.borrow_mut().push(Call::Parent(*envelope));
                true
            }

            fn should_unpack_leaf(&self, point: &[f64; 2]) -> bool {
                self.0.borrow_mut().push(Call::Leaf(*point));
                true
            }
        }

        /// Checks that `calls` tests the children of `parent`, and of each node below it, in order,
        /// and that `returned` has their elements in order.
        fn assert_in_order(parent: &ParentNode<[f64; 2]>, calls: &[Call], returned: &[[f64; 2]]) {
            let children: Vec<_> = parent
                .children()
                .iter()
                .map(|child| match child {
                    RTreeNode::Leaf(point) => Call::Leaf(*point),
                    RTreeNode::Parent(data) => Call::Parent(data.envelope()),
                })
                .collect();
            let tested: Vec<_> = calls
                .iter()
                .copied()
                .filter(|call| children.contains(call))
                .collect();
            assert_eq!(tested, children);

            let elements: Vec<_> = children
                .iter()
                .filter_map(|child| match child {
                    Call::Leaf(point) => Some(*point),
                    Call::Parent(_) => None,
                })
                .collect();
            let in_returned: Vec<_> = returned
                .iter()
                .copied()
                .filter(|point| elements.contains(point))
                .collect();
            assert_eq!(in_returned, elements);

            for child in parent.children() {
                if let RTreeNode::Parent(data) = child {
                    assert_in_order(data, calls, returned);
                }
            }
        }

        let mut tree = RTree::bulk_load(create_random_points(1000, SEED_1));
        let calls = RefCell::new(Vec::new());

        // `fold` continues where `next` left off.
        for consumed in [0, 1, tree.size() / 2, tree.size() - 1, tree.size()] {
            let mut returned = Vec::new();
            let mut iter = tree.locate_with_selection_function(RecordingFunc(&calls));
            for point in iter.by_ref().take(consumed) {
                returned.push(*point);
            }

            iter.for_each(|point| returned.push(*point));
            assert_in_order(tree.root(), &calls.take(), &returned);

            let mut returned = Vec::new();
            let mut iter = tree.locate_with_selection_function_mut(RecordingFunc(&calls));
            for point in iter.by_ref().take(consumed) {
                returned.push(*point);
            }

            iter.for_each(|point| returned.push(*point));
            assert_in_order(tree.root(), &calls.take(), &returned);
        }
    }

    /// `fold` returns the same elements as `next`, also when it continues where `next` left off.
    #[test]
    fn test_fold_returns_the_same_elements_as_next() {
        let points = create_random_points(1000, SEED_1);
        let mut tree = RTree::bulk_load(points.clone());
        let mut selected = Vec::new();
        for point in tree.locate_with_selection_function(SelectFunc) {
            selected.push(*point);
        }

        let selected = sorted(selected);
        // Unpacking every parent would select more.
        let in_bottom_half = points
            .iter()
            .filter(|point| SelectFunc.should_unpack_leaf(point))
            .count();
        assert!(!selected.is_empty() && selected.len() < in_bottom_half);

        for consumed in [0, 1, selected.len() / 2, selected.len() - 1, selected.len()] {
            let mut returned = Vec::new();
            let mut iter = tree.locate_with_selection_function(SelectFunc);
            for point in iter.by_ref().take(consumed) {
                returned.push(*point);
            }

            iter.for_each(|point| returned.push(*point));
            assert_eq!(sorted(returned), selected);

            let mut returned = Vec::new();
            let mut iter = tree.locate_with_selection_function_mut(SelectFunc);
            for point in iter.by_ref().take(consumed) {
                returned.push(*point);
            }

            iter.for_each(|point| returned.push(*point));
            assert_eq!(sorted(returned), selected);
        }
    }

    /// `select_nodes` and `select_nodes_mut` visit the elements that the iterators return, and stop
    /// when the visitor breaks.
    #[test]
    fn test_select_nodes() {
        use super::{select_nodes, select_nodes_mut};
        use core::ops::ControlFlow;

        let mut tree = RTree::bulk_load(create_random_points(1000, SEED_1));
        let mut selected = Vec::new();
        for point in tree.locate_with_selection_function(SelectFunc) {
            selected.push(*point);
        }

        let selected = sorted(selected);
        let mut visited = Vec::new();
        let result = select_nodes(tree.root(), &SelectFunc, &mut |point: &[f64; 2]| {
            visited.push(*point);
            ControlFlow::<()>::Continue(())
        });
        assert_eq!(result, ControlFlow::Continue(()));
        assert_eq!(sorted(visited), selected);

        let mut visited = Vec::new();
        let result = select_nodes_mut(tree.root_mut(), &SelectFunc, &mut |point: &mut [f64; 2]| {
            visited.push(*point);
            ControlFlow::<()>::Continue(())
        });
        assert_eq!(result, ControlFlow::Continue(()));
        assert_eq!(sorted(visited), selected);

        for stop in [1, selected.len() / 2, selected.len()] {
            let mut visited = 0;
            let result = select_nodes(tree.root(), &SelectFunc, &mut |_| {
                visited += 1;
                if visited == stop {
                    ControlFlow::Break(())
                } else {
                    ControlFlow::Continue(())
                }
            });
            assert_eq!((result, visited), (ControlFlow::Break(()), stop));

            let mut visited = 0;
            let result = select_nodes_mut(tree.root_mut(), &SelectFunc, &mut |_| {
                visited += 1;
                if visited == stop {
                    ControlFlow::Break(())
                } else {
                    ControlFlow::Continue(())
                }
            });
            assert_eq!((result, visited), (ControlFlow::Break(()), stop));
        }
    }

    /// After a selection function panics, the iterators may skip elements. They never return an
    /// element twice.
    #[test]
    fn test_no_element_is_returned_twice_after_a_panic() {
        use core::cell::Cell;
        use std::panic::{catch_unwind, resume_unwind, AssertUnwindSafe};

        /// Panics when it tests its first leaf.
        struct PanicOnceFunc(Cell<bool>);

        impl SelectionFunction<[f64; 2]> for PanicOnceFunc {
            fn should_unpack_parent(&self, _: &AABB<[f64; 2]>) -> bool {
                true
            }

            fn should_unpack_leaf(&self, _: &[f64; 2]) -> bool {
                if self.0.replace(false) {
                    // Unlike `panic!`, this doesn't print a message.
                    resume_unwind(Box::new(()));
                }

                true
            }
        }

        fn assert_distinct(mut points: Vec<[f64; 2]>) {
            let len = points.len();
            points.sort_by(|a, b| a.partial_cmp(b).unwrap());
            points.dedup();
            assert_eq!(points.len(), len);
        }

        let mut tree = RTree::bulk_load(create_random_points(1000, SEED_1));

        let mut iter = tree.locate_with_selection_function(PanicOnceFunc(Cell::new(true)));
        assert!(catch_unwind(AssertUnwindSafe(|| iter.next().is_some())).is_err());
        assert_distinct(iter.copied().collect());

        let mut iter = tree.locate_with_selection_function_mut(PanicOnceFunc(Cell::new(true)));
        assert!(catch_unwind(AssertUnwindSafe(|| iter.next().is_some())).is_err());
        assert_distinct(iter.map(|p| *p).collect());

        // The same, continuing with `fold`.
        let mut iter = tree.locate_with_selection_function(PanicOnceFunc(Cell::new(true)));
        assert!(catch_unwind(AssertUnwindSafe(|| iter.next().is_some())).is_err());
        let mut points = Vec::new();
        iter.for_each(|point| points.push(*point));
        assert_distinct(points);

        let mut iter = tree.locate_with_selection_function_mut(PanicOnceFunc(Cell::new(true)));
        assert!(catch_unwind(AssertUnwindSafe(|| iter.next().is_some())).is_err());
        let mut points = Vec::new();
        iter.for_each(|point| points.push(*point));
        assert_distinct(points);
    }

    #[test]
    fn test_locate_within_distance() {
        use crate::primitives::Line;

        let points = create_random_points(100, SEED_1);
        let tree = RTree::bulk_load(points.clone());
        let circle_radius_2 = 0.3;
        let circle_origin = [0.2, 0.6];
        let contained_in_circle: Vec<_> = points
            .iter()
            .filter(|point| Line::new(circle_origin, **point).length_2() <= circle_radius_2)
            .cloned()
            .collect();
        let located: Vec<_> = tree
            .locate_within_distance(circle_origin, circle_radius_2)
            .cloned()
            .collect();

        assert_eq!(located.len(), contained_in_circle.len());
        for point in &contained_in_circle {
            assert!(located.contains(point));
        }
    }

    #[test]
    fn test_locate_within_distance_on_empty_tree() {
        let tree: RTree<[f64; 3]> = RTree::new();
        tree.locate_within_distance([0.0, 0.0, 0.0], 10.0);

        let tree: RTree<[i64; 3]> = RTree::new();
        tree.locate_within_distance([0, 0, 0], 10);
    }
}
