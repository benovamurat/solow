//! Empirical distribution utilities: a monotone [`StepFunction`] and the
//! [`Ecdf`] (empirical cumulative distribution function).
//!
//! These mirror the reference's `empirical_distribution` module exactly,
//! including the internal `[-inf]`-prefixed knot vector and the
//! `searchsorted(..) - 1` lookup rule, so evaluations agree to the last bit.

/// Which side of a tie a [`StepFunction`] (and hence [`Ecdf`]) is continuous on.
///
/// `Right` is right-continuous (the value jumps *at* a knot and holds for
/// larger arguments); `Left` jumps just *after* a knot.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Side {
    /// Right-continuous: `searchsorted` uses the right side.
    Right,
    /// Left-continuous: `searchsorted` uses the left side. This is the
    /// reference's `StepFunction` default.
    Left,
}

/// A monotone step function defined by knots `x` and step heights `y`.
///
/// The function is piecewise-constant. Internally the knot vector is prefixed
/// with `-inf` carrying the initial value `ival`, exactly as the reference
/// does, so that arguments below the first knot return `ival`.
#[derive(Clone, Debug)]
pub struct StepFunction {
    /// Knot abscissae, prefixed with `-inf`. Strictly the reference's `self.x`.
    x: Vec<f64>,
    /// Step heights, prefixed with `ival`. The reference's `self.y`.
    y: Vec<f64>,
    side: Side,
    n: usize,
}

/// Binary search returning the number of elements in `a` that compare
/// strictly-less (`Side::Left`) or less-or-equal (`Side::Right`) than `v`.
///
/// Equivalent to NumPy's `searchsorted(a, v, side)` on a sorted slice. `NaN`
/// is treated as larger than every value, matching NumPy.
fn searchsorted(a: &[f64], v: f64, side: Side) -> usize {
    // Standard bisection on a sorted slice.
    let mut lo = 0usize;
    let mut hi = a.len();
    while lo < hi {
        let mid = lo + (hi - lo) / 2;
        let go_right = match side {
            // side='left': first index i with a[i] >= v  => move right while a[mid] < v
            Side::Left => a[mid] < v,
            // side='right': first index i with a[i] > v  => move right while a[mid] <= v
            Side::Right => a[mid] <= v,
        };
        if go_right {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }
    lo
}

impl StepFunction {
    /// Build a step function from knots `x` and step heights `y`.
    ///
    /// `ival` is the value returned for arguments below the smallest knot.
    /// If `sorted` is false the pairs are sorted by `x` first. `side` selects
    /// left- or right-continuity.
    ///
    /// # Panics
    /// Panics if `x` and `y` have different lengths.
    pub fn new(x: &[f64], y: &[f64], ival: f64, sorted: bool, side: Side) -> Self {
        assert_eq!(x.len(), y.len(), "x and y must have the same length");
        let n = x.len();
        let (xs, ys): (Vec<f64>, Vec<f64>) = if sorted || n <= 1 {
            (x.to_vec(), y.to_vec())
        } else {
            let mut idx: Vec<usize> = (0..n).collect();
            idx.sort_by(|&i, &j| x[i].total_cmp(&x[j]));
            (
                idx.iter().map(|&i| x[i]).collect(),
                idx.iter().map(|&i| y[i]).collect(),
            )
        };
        let mut xx = Vec::with_capacity(n + 1);
        let mut yy = Vec::with_capacity(n + 1);
        xx.push(f64::NEG_INFINITY);
        yy.push(ival);
        xx.extend_from_slice(&xs);
        yy.extend_from_slice(&ys);
        StepFunction {
            x: xx,
            y: yy,
            side,
            n,
        }
    }

    /// The knot vector (prefixed with `-inf`), matching the reference's `.x`.
    pub fn x(&self) -> &[f64] {
        &self.x
    }

    /// The step heights (prefixed with the initial value), matching `.y`.
    pub fn y(&self) -> &[f64] {
        &self.y
    }

    /// Number of (user-supplied) knots, excluding the `-inf` sentinel.
    pub fn len(&self) -> usize {
        self.n
    }

    /// Whether the step function has no user-supplied knots.
    pub fn is_empty(&self) -> bool {
        self.n == 0
    }

    /// Evaluate the step function at a single point.
    pub fn eval(&self, v: f64) -> f64 {
        // tind = searchsorted(x, v, side) - 1, clipped to a valid index.
        let s = searchsorted(&self.x, v, self.side);
        let idx = s.saturating_sub(1).min(self.y.len() - 1);
        self.y[idx]
    }

    /// Evaluate the step function at many points.
    pub fn eval_many(&self, vs: &[f64]) -> Vec<f64> {
        vs.iter().map(|&v| self.eval(v)).collect()
    }
}

/// The empirical cumulative distribution function of a data sample.
///
/// `Ecdf(t)` is the proportion of observations `<= t`. It is built as a
/// right-continuous [`StepFunction`] over the sorted data with heights
/// `1/n, 2/n, ..., 1`, matching the reference's `ECDF`.
#[derive(Clone, Debug)]
pub struct Ecdf {
    step: StepFunction,
}

impl Ecdf {
    /// Build the ECDF of `data`. Defaults to right-continuity (`Side::Right`),
    /// matching the reference default.
    pub fn new(data: &[f64]) -> Self {
        Ecdf::with_side(data, Side::Right)
    }

    /// Build the ECDF with an explicit continuity side.
    ///
    /// # Panics
    /// Panics if `data` is empty.
    pub fn with_side(data: &[f64], side: Side) -> Self {
        let n = data.len();
        assert!(n > 0, "ECDF requires at least one observation");
        let mut xs = data.to_vec();
        xs.sort_by(|a, b| a.total_cmp(b));
        // y = linspace(1/n, 1, n)
        let nn = n as f64;
        let ys: Vec<f64> = (0..n).map(|i| (i as f64 + 1.0) / nn).collect();
        let step = StepFunction::new(&xs, &ys, 0.0, true, side);
        Ecdf { step }
    }

    /// The knot vector (prefixed with `-inf`), the reference's `.x`.
    pub fn x(&self) -> &[f64] {
        self.step.x()
    }

    /// The cumulative proportions (prefixed with `0`), the reference's `.y`.
    pub fn y(&self) -> &[f64] {
        self.step.y()
    }

    /// Evaluate the ECDF at a single point.
    pub fn eval(&self, v: f64) -> f64 {
        self.step.eval(v)
    }

    /// Evaluate the ECDF at many points.
    pub fn eval_many(&self, vs: &[f64]) -> Vec<f64> {
        self.step.eval_many(vs)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn stepfunction_left_default() {
        let sf = StepFunction::new(
            &[1.0, 2.0, 3.0],
            &[10.0, 20.0, 30.0],
            0.0,
            false,
            Side::Left,
        );
        assert_eq!(sf.x()[0], f64::NEG_INFINITY);
        assert_eq!(sf.y(), &[0.0, 10.0, 20.0, 30.0]);
        assert_abs_diff_eq!(sf.eval(0.5), 0.0);
        assert_abs_diff_eq!(sf.eval(1.0), 0.0);
        assert_abs_diff_eq!(sf.eval(1.5), 10.0);
        assert_abs_diff_eq!(sf.eval(3.0), 20.0);
        assert_abs_diff_eq!(sf.eval(3.5), 30.0);
    }

    #[test]
    fn stepfunction_right_ival() {
        let sf = StepFunction::new(
            &[1.0, 2.0, 3.0],
            &[10.0, 20.0, 30.0],
            -5.0,
            false,
            Side::Right,
        );
        assert_abs_diff_eq!(sf.eval(0.0), -5.0);
        assert_abs_diff_eq!(sf.eval(1.0), 10.0);
        assert_abs_diff_eq!(sf.eval(2.5), 20.0);
    }

    #[test]
    fn ecdf_basic() {
        let e = Ecdf::new(&[3.0, 1.0, 2.0, 1.0, 5.0]);
        assert_abs_diff_eq!(e.eval(-1.0), 0.0);
        assert_abs_diff_eq!(e.eval(1.0), 0.4, epsilon = 1e-15);
        assert_abs_diff_eq!(e.eval(2.0), 0.6, epsilon = 1e-15);
        assert_abs_diff_eq!(e.eval(3.0), 0.8, epsilon = 1e-15);
        assert_abs_diff_eq!(e.eval(5.0), 1.0, epsilon = 1e-15);
        assert_abs_diff_eq!(e.eval(6.0), 1.0, epsilon = 1e-15);
    }
}
