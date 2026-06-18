//! B-spline smooth basis for a single covariate, with a curvature penalty.
//!
//! This mirrors the spline construction used by the reference GAM
//! implementation: quantile-spaced interior knots with `degree + 1`
//! multiplicity at the boundaries, the constant column dropped (so the basis
//! does not collide with an explicit intercept), and a penalty matrix equal to
//! the integral of the cross-product of the basis' second derivative.

use ndarray::{Array1, Array2};
use solow_core::error::{Error, Result};

/// A penalized B-spline basis built from one explanatory variable.
///
/// The basis reproduces the reference's `BSplines` smoother for a single
/// component with `include_intercept=False` and no centering constraint. The
/// first raw B-spline column (the only one carrying a constant) is dropped, so
/// the design has `df - 1` columns where `df` is the requested degrees of
/// freedom.
#[derive(Clone, Debug)]
pub struct BSplines {
    /// Full augmented knot vector (sorted, with boundary multiplicity).
    knots: Array1<f64>,
    /// Spline degree (`3` is cubic).
    degree: usize,
    /// Design matrix of basis columns, one row per observation.
    basis: Array2<f64>,
    /// Curvature penalty `S = integral B''(x) B''(x)' dx` (square, `dim_basis`).
    cov_der2: Array2<f64>,
}

impl BSplines {
    /// Build the basis for `x` with `df` degrees of freedom and spline `degree`.
    ///
    /// Interior knots are the `df - (degree + 1)` evenly-spaced empirical
    /// quantiles of `x`; the boundary knots are the data range repeated
    /// `degree + 1` times. The constant column is dropped, yielding
    /// `df - 1` basis columns.
    pub fn new(x: &Array1<f64>, df: usize, degree: usize) -> Result<Self> {
        if x.is_empty() {
            return Err(Error::Shape("BSplines: empty input".into()));
        }
        let order = degree + 1;
        if df < order {
            return Err(Error::Shape(format!(
                "BSplines: df={df} is too small for degree={degree} (need df >= {order})"
            )));
        }
        let knots = compute_all_knots(x, df, degree);
        // Basis with the constant column dropped (include_intercept = false).
        let basis = eval_bspline_basis(x, &knots, degree, 0, false);
        let cov_der2 = compute_cov_der2(&knots, degree);
        Ok(BSplines {
            knots,
            degree,
            basis,
            cov_der2,
        })
    }

    /// The spline design matrix (`n_obs` x `dim_basis`).
    pub fn basis(&self) -> &Array2<f64> {
        &self.basis
    }

    /// The curvature penalty matrix `S` (`dim_basis` x `dim_basis`).
    pub fn cov_der2(&self) -> &Array2<f64> {
        &self.cov_der2
    }

    /// The full augmented knot vector.
    pub fn knots(&self) -> &Array1<f64> {
        &self.knots
    }

    /// Number of basis columns (`df - 1`).
    pub fn dim_basis(&self) -> usize {
        self.basis.ncols()
    }

    /// Spline degree.
    pub fn degree(&self) -> usize {
        self.degree
    }
}

/// Compute the augmented knot vector: boundary knots repeated `degree + 1`
/// times plus the interior quantile knots, all sorted ascending.
fn compute_all_knots(x: &Array1<f64>, df: usize, degree: usize) -> Array1<f64> {
    let order = degree + 1;
    let n_inner = df - order;
    let (x_min, x_max) = min_max(x);

    let mut all: Vec<f64> = Vec::with_capacity(2 * order + n_inner);
    for _ in 0..order {
        all.push(x_min);
        all.push(x_max);
    }
    // Interior knots: empirical quantiles at evenly spaced probabilities.
    // probs = linspace(0, 1, n_inner + 2)[1..-1].
    for i in 1..=n_inner {
        let q = i as f64 / (n_inner + 1) as f64;
        all.push(quantile_linear(x, q));
    }
    all.sort_by(|a, b| a.total_cmp(b));
    Array1::from_vec(all)
}

/// Linear-interpolation quantile matching NumPy's default `percentile`
/// (method "linear"). `q` is in `[0, 1]`.
fn quantile_linear(x: &Array1<f64>, q: f64) -> f64 {
    let mut sorted: Vec<f64> = x.to_vec();
    sorted.sort_by(|a, b| a.total_cmp(b));
    let n = sorted.len();
    if n == 1 {
        return sorted[0];
    }
    let pos = q * (n - 1) as f64;
    let lo = pos.floor() as usize;
    let hi = pos.ceil() as usize;
    if lo == hi {
        return sorted[lo];
    }
    let frac = pos - lo as f64;
    sorted[lo] * (1.0 - frac) + sorted[hi] * frac
}

fn min_max(x: &Array1<f64>) -> (f64, f64) {
    let mut lo = f64::INFINITY;
    let mut hi = f64::NEG_INFINITY;
    for &v in x {
        if v < lo {
            lo = v;
        }
        if v > hi {
            hi = v;
        }
    }
    (lo, hi)
}

/// Evaluate the degree-0 (indicator) B-spline functions over each knot
/// interval `[t[i], t[i+1])`, closing the final non-empty interval at the
/// rightmost knot so the upper boundary point is included (matching `splev`).
fn degree0_basis(x: &Array1<f64>, knots: &Array1<f64>) -> Array2<f64> {
    let nx = x.len();
    let m = knots.len();
    let t_max = knots[m - 1];
    let mut n0 = Array2::<f64>::zeros((nx, m - 1));
    for i in 0..m - 1 {
        let left = knots[i];
        let right = knots[i + 1];
        if right > left {
            for r in 0..nx {
                let xi = x[r];
                let inside = xi >= left && xi < right;
                let at_end = right == t_max && xi == t_max;
                if inside || at_end {
                    n0[[r, i]] = 1.0;
                }
            }
        }
    }
    n0
}

/// Run one level of the Cox-de Boor recursion, raising the basis from degree
/// `deg - 1` (`prev`) to degree `deg`.
fn cox_de_boor_step(
    prev: &Array2<f64>,
    x: &Array1<f64>,
    knots: &Array1<f64>,
    deg: usize,
) -> Array2<f64> {
    let nx = x.len();
    let m = knots.len();
    let n = m - deg - 1;
    let mut out = Array2::<f64>::zeros((nx, n));
    for i in 0..n {
        let den1 = knots[i + deg] - knots[i];
        let den2 = knots[i + deg + 1] - knots[i + 1];
        for r in 0..nx {
            let mut v = 0.0;
            if den1 != 0.0 {
                v += (x[r] - knots[i]) / den1 * prev[[r, i]];
            }
            if den2 != 0.0 {
                v += (knots[i + deg + 1] - x[r]) / den2 * prev[[r, i + 1]];
            }
            out[[r, i]] = v;
        }
    }
    out
}

/// All B-spline bases up to the requested degree: `out[d]` holds the
/// degree-`d` basis evaluated at `x`.
fn all_bases(x: &Array1<f64>, knots: &Array1<f64>, degree: usize) -> Vec<Array2<f64>> {
    let mut bases = Vec::with_capacity(degree + 1);
    bases.push(degree0_basis(x, knots));
    for d in 1..=degree {
        let next = cox_de_boor_step(&bases[d - 1], x, knots, d);
        bases.push(next);
    }
    bases
}

/// First derivative of the degree-`deg` basis, computed from the degree-`deg-1`
/// basis via the standard B-spline derivative recurrence.
fn deriv_one_level(lower: &Array2<f64>, knots: &Array1<f64>, deg: usize) -> Array2<f64> {
    let nx = lower.nrows();
    let m = knots.len();
    let n = m - deg - 1;
    let mut out = Array2::<f64>::zeros((nx, n));
    let degf = deg as f64;
    for i in 0..n {
        let den1 = knots[i + deg] - knots[i];
        let den2 = knots[i + deg + 1] - knots[i + 1];
        for r in 0..nx {
            let mut v = 0.0;
            if den1 != 0.0 {
                v += lower[[r, i]] / den1;
            }
            if den2 != 0.0 {
                v -= lower[[r, i + 1]] / den2;
            }
            out[[r, i]] = degf * v;
        }
    }
    out
}

/// Evaluate the B-spline basis (or a derivative) at `x`.
///
/// `deriv` selects the derivative order (`0`, `1`, or `2`). When
/// `include_intercept` is false the first (constant-carrying) column is
/// dropped, matching the reference's `include_intercept=False` basis.
fn eval_bspline_basis(
    x: &Array1<f64>,
    knots: &Array1<f64>,
    degree: usize,
    deriv: usize,
    include_intercept: bool,
) -> Array2<f64> {
    let bases = all_bases(x, knots, degree);
    let full = match deriv {
        0 => bases[degree].clone(),
        1 => deriv_one_level(&bases[degree - 1], knots, degree),
        2 => {
            // Second derivative: differentiate once to get the degree-(d-1)
            // first derivative, then apply the derivative recurrence again.
            let d1_lower = deriv_one_level(&bases[degree - 2], knots, degree - 1);
            deriv_one_level(&d1_lower, knots, degree)
        }
        _ => panic!("unsupported derivative order {deriv}"),
    };
    if include_intercept {
        full
    } else {
        // Drop the first column (k_const = 1).
        let nx = full.nrows();
        let n = full.ncols();
        full.slice(ndarray::s![.., 1..n])
            .to_owned()
            .into_shape_with_order((nx, n - 1))
            .unwrap()
    }
}

/// Integration points: insert `k_points` evenly between each pair of unique
/// knots, then append the rightmost knot. With `k_points = 3` this yields four
/// points per interval plus the final endpoint (always an odd count).
fn integration_points(knots: &Array1<f64>, k_points: usize) -> Array1<f64> {
    let uniq = unique_sorted(knots);
    let kp = k_points + 1;
    let mut pts: Vec<f64> = Vec::new();
    for w in 0..uniq.len() - 1 {
        let lo = uniq[w];
        let hi = uniq[w + 1];
        let dx = hi - lo;
        for j in 0..kp {
            pts.push(lo + dx * (j as f64) / (kp as f64));
        }
    }
    pts.push(uniq[uniq.len() - 1]);
    Array1::from_vec(pts)
}

fn unique_sorted(x: &Array1<f64>) -> Vec<f64> {
    let mut v: Vec<f64> = x.to_vec();
    v.sort_by(|a, b| a.total_cmp(b));
    v.dedup();
    v
}

/// Composite Simpson integration of the matrix-valued integrand `B''B''^T`
/// over the (possibly non-uniform) integration grid, matching SciPy's
/// `simpson` for an odd number of sample points.
fn compute_cov_der2(knots: &Array1<f64>, degree: usize) -> Array2<f64> {
    let xi = integration_points(knots, 3);
    // Second-derivative basis with the constant column dropped.
    let d2 = eval_bspline_basis(&xi, knots, degree, 2, false);
    let nx = d2.nrows();
    let nb = d2.ncols();
    let mut cov = Array2::<f64>::zeros((nb, nb));

    // Composite Simpson over consecutive pairs of intervals; `nx` is odd so
    // `nx - 1` is even and the loop covers the whole grid exactly.
    let mut i = 0;
    while i + 2 < nx {
        let h0 = xi[i + 1] - xi[i];
        let h1 = xi[i + 2] - xi[i + 1];
        let hsum = h0 + h1;
        let hprod = h0 * h1;
        let w0 = hsum / 6.0 * (2.0 - h1 / h0);
        let w1 = hsum / 6.0 * (hsum * hsum / hprod);
        let w2 = hsum / 6.0 * (2.0 - h0 / h1);
        for a in 0..nb {
            let ya0 = d2[[i, a]];
            let ya1 = d2[[i + 1, a]];
            let ya2 = d2[[i + 2, a]];
            for b in 0..nb {
                let contrib =
                    w0 * ya0 * d2[[i, b]] + w1 * ya1 * d2[[i + 1, b]] + w2 * ya2 * d2[[i + 2, b]];
                cov[[a, b]] += contrib;
            }
        }
        i += 2;
    }
    cov
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn knot_multiplicity_and_count() {
        let x = array![0.0, 0.1, 0.25, 0.4, 0.6, 0.75, 0.9, 1.0];
        let bs = BSplines::new(&x, 6, 3).unwrap();
        // order = 4, n_inner = 2, total knots = 2*order + n_inner = 10.
        assert_eq!(bs.knots().len(), 10);
        // Boundary knots have multiplicity order = 4.
        assert_eq!(bs.knots().iter().filter(|&&k| k == 0.0).count(), 4);
        assert_eq!(bs.knots().iter().filter(|&&k| k == 1.0).count(), 4);
        // dim_basis = df - 1.
        assert_eq!(bs.dim_basis(), 5);
        assert_eq!(bs.basis().ncols(), 5);
        assert_eq!(bs.basis().nrows(), x.len());
    }

    #[test]
    fn basis_partition_of_unity_with_constant() {
        // The *full* (intercept-included) basis sums to one at every point.
        let x = array![0.0, 0.2, 0.35, 0.5, 0.65, 0.8, 1.0];
        let knots = compute_all_knots(&x, 7, 3);
        let full = eval_bspline_basis(&x, &knots, 3, 0, true);
        for r in 0..x.len() {
            let s: f64 = full.row(r).sum();
            assert!((s - 1.0).abs() < 1e-12, "row {r} sums to {s}");
        }
    }

    #[test]
    fn penalty_is_symmetric_psd() {
        let x = Array1::linspace(0.0, 1.0, 40);
        let bs = BSplines::new(&x, 9, 3).unwrap();
        let s = bs.cov_der2();
        assert_eq!(s.dim(), (8, 8));
        for a in 0..8 {
            for b in 0..8 {
                assert!((s[[a, b]] - s[[b, a]]).abs() < 1e-9);
            }
        }
        // Quadratic form is non-negative for a few random vectors.
        for seed in 0..5u64 {
            let v: Array1<f64> =
                Array1::from_shape_fn(8, |i| ((seed + i as u64 * 7) % 11) as f64 - 5.0);
            let q = v.dot(&s.dot(&v));
            assert!(q >= -1e-9, "quadratic form negative: {q}");
        }
    }

    #[test]
    fn quantile_matches_numpy_linear() {
        let x = array![1.0, 2.0, 3.0, 4.0];
        // numpy.percentile(x, 50) == 2.5 (linear interpolation).
        assert!((quantile_linear(&x, 0.5) - 2.5).abs() < 1e-12);
        assert!((quantile_linear(&x, 0.25) - 1.75).abs() < 1e-12);
        assert!((quantile_linear(&x, 0.0) - 1.0).abs() < 1e-12);
        assert!((quantile_linear(&x, 1.0) - 4.0).abs() < 1e-12);
    }
}
