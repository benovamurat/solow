//! Cyclic cubic regression spline basis for a single covariate, with the
//! associated wiggliness penalty.
//!
//! This mirrors the `CyclicCubicSplines` smoother of the reference GAM
//! implementation, whose basis is produced by the cyclic cubic regression
//! spline (`cc`) construction of Wood, *Generalized Additive Models* (2006),
//! pp. 145-147. A cyclic spline ties the value and first/second derivatives at
//! the two boundary knots together, so the fitted curve joins up smoothly into
//! a loop. With `df` degrees of freedom the construction places `df + 1`
//! distinct knots (two boundary knots plus `df - 1` interior quantile knots)
//! and yields a basis with exactly `df` columns.

use ndarray::{Array1, Array2};
use solow_core::error::{Error, Result};
use solow_linalg::{inv, solve};

/// A cyclic cubic regression spline basis built from one explanatory variable.
///
/// Reproduces the reference `CyclicCubicSplines` smoother for a single
/// component with no extra constraints: the design has `df` columns and the
/// penalty `cov_der2` is the `s = d' B^{-1} d` wiggliness matrix of the cyclic
/// spline (Wood, 2006, p. 146).
#[derive(Clone, Debug)]
pub struct CyclicCubicSplines {
    /// All sorted knots (`df + 1` distinct values: boundaries plus interior).
    knots: Array1<f64>,
    /// Design matrix of basis columns, one row per observation (`df` columns).
    basis: Array2<f64>,
    /// Wiggliness penalty `s = d' B^{-1} d` (`df` x `df`).
    cov_der2: Array2<f64>,
}

impl CyclicCubicSplines {
    /// Build the unconstrained cyclic cubic spline basis for `x` with `df`
    /// degrees of freedom.
    ///
    /// `df` must be at least `4` (cyclic cubic splines need at least four
    /// distinct knots). Interior knots are the `df - 1` evenly-spaced empirical
    /// percentiles of `x`; the two boundary knots are `min(x)` and `max(x)`.
    /// The basis has `df` columns.
    ///
    /// Note: the unconstrained cyclic basis is a partition of unity (its rows
    /// sum to one), so the design `[1, basis]` is rank-deficient. Use
    /// [`CyclicCubicSplines::with_centering`] to absorb a centering constraint
    /// and obtain a full-rank, identifiable smooth term.
    pub fn new(x: &Array1<f64>, df: usize) -> Result<Self> {
        Self::build(x, df, false)
    }

    /// Build the cyclic cubic spline basis with a centering constraint absorbed.
    ///
    /// The reference's `constraints='center'` reparameterization removes the
    /// constant component of the smooth by a linear transform `T` whose columns
    /// span the null space of the mean of each (unconstrained) basis column.
    /// The basis becomes `basis @ T` (with `df - 1` columns) and the penalty
    /// `T' S T`. This makes `[1, basis]` full rank, so the fit is identifiable.
    pub fn with_centering(x: &Array1<f64>, df: usize) -> Result<Self> {
        Self::build(x, df, true)
    }

    fn build(x: &Array1<f64>, df: usize, center: bool) -> Result<Self> {
        if x.is_empty() {
            return Err(Error::Shape("CyclicCubicSplines: empty input".into()));
        }
        if df < 4 {
            return Err(Error::Shape(format!(
                "CyclicCubicSplines: df={df} too small (need df >= 4)"
            )));
        }
        // The cyclic basis needs `n_inner_knots = df - 2 + 1 = df - 1` interior
        // knots so that, after dropping one knot for the cyclic wrap, the basis
        // has exactly `df` columns.
        let n_inner = df - 1;
        let knots = all_sorted_knots(x, n_inner)?;
        let mut basis = free_cyclic_dmatrix(x, &knots)?;
        let mut cov_der2 = cyclic_penalty(&knots)?;
        if center {
            // Centering constraint c = column means of the basis (1 x k); the
            // transform T spans the null space of c (k x (k-1)).
            let k = basis.ncols();
            let n = basis.nrows();
            let mut c = vec![0.0f64; k];
            for j in 0..k {
                let mut s = 0.0;
                for i in 0..n {
                    s += basis[[i, j]];
                }
                c[j] = s / n as f64;
            }
            let transf = centering_transform(&c);
            basis = basis.dot(&transf);
            cov_der2 = transf.t().dot(&cov_der2).dot(&transf);
        }
        Ok(CyclicCubicSplines {
            knots,
            basis,
            cov_der2,
        })
    }

    /// The cyclic spline design matrix (`n_obs` x `df`).
    pub fn basis(&self) -> &Array2<f64> {
        &self.basis
    }

    /// The wiggliness penalty matrix `s` (`df` x `df`).
    pub fn cov_der2(&self) -> &Array2<f64> {
        &self.cov_der2
    }

    /// All sorted knots (`df + 1` distinct values).
    pub fn knots(&self) -> &Array1<f64> {
        &self.knots
    }

    /// Number of basis columns (`df`).
    pub fn dim_basis(&self) -> usize {
        self.basis.ncols()
    }
}

/// Boundary knots plus `n_inner` interior percentile knots, all sorted and
/// de-duplicated. Mirrors patsy's `_get_all_sorted_knots`: interior knots are
/// the percentiles of the unique data values at probabilities
/// `linspace(0, 100, n_inner + 2)[1..-1]`.
fn all_sorted_knots(x: &Array1<f64>, n_inner: usize) -> Result<Array1<f64>> {
    let (lo, hi) = min_max(x);
    // Unique sorted data values (the percentile is computed over these).
    let mut uniq: Vec<f64> = x.to_vec();
    uniq.sort_by(|a, b| a.total_cmp(b));
    uniq.dedup();

    let mut inner: Vec<f64> = Vec::with_capacity(n_inner);
    for i in 1..=n_inner {
        // probabilities are linspace(0, 100, n_inner + 2)[1..-1], i.e.
        // q = 100 * i / (n_inner + 1) in percent => fraction i/(n_inner+1).
        let q = i as f64 / (n_inner + 1) as f64;
        inner.push(percentile_linear(&uniq, q));
    }

    let mut all: Vec<f64> = vec![lo, hi];
    all.extend(inner);
    all.sort_by(|a, b| a.total_cmp(b));
    all.dedup();
    if all.len() != n_inner + 2 {
        return Err(Error::Shape(format!(
            "CyclicCubicSplines: could not form {} distinct knots (got {})",
            n_inner + 2,
            all.len()
        )));
    }
    Ok(Array1::from_vec(all))
}

/// NumPy `percentile` with method "linear" over an already-sorted slice; `q`
/// is the fraction in `[0, 1]`.
fn percentile_linear(sorted: &[f64], q: f64) -> f64 {
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

/// Map values cyclically into `[lo, hi]` (Wood's cyclic wrap, patsy
/// `_map_cyclic`).
fn map_cyclic(x: f64, lo: f64, hi: f64) -> f64 {
    let span = hi - lo;
    if x > hi {
        lo + (x - hi).rem_euclid(span)
    } else if x < lo {
        hi - (lo - x).rem_euclid(span)
    } else {
        x
    }
}

/// Lower-bound knot index for each `x`, matching patsy `_find_knots_lower_bounds`:
/// `lb = searchsorted(knots, x) - 1`, clamped to `[0, knots.len()-2]`.
fn find_knot_lb(xi: f64, knots: &[f64]) -> usize {
    // searchsorted(side='left'): first index i with knots[i] >= xi.
    let m = knots.len();
    let mut idx = m;
    for (i, &k) in knots.iter().enumerate() {
        if k >= xi {
            idx = i;
            break;
        }
    }
    // lb = idx - 1, with lb == -1 -> 0 and lb == m-1 -> m-2.
    let lb = if idx == 0 { 0usize } else { idx - 1 };
    lb.min(m - 2)
}

/// The four base functions `(ajm, ajp, cjm, cjp)` and the lower-bound index `j`
/// for a single value (Wood, 2006, p. 146; patsy `_compute_base_functions`).
fn base_functions(xi: f64, knots: &[f64]) -> (f64, f64, f64, f64, usize) {
    let kmin = knots[0];
    let kmax = knots[knots.len() - 1];
    let j = find_knot_lb(xi, knots);
    let hj = knots[j + 1] - knots[j];
    let xj1_x = knots[j + 1] - xi;
    let x_xj = xi - knots[j];

    let ajm = xj1_x / hj;
    let ajp = x_xj / hj;

    let mut cjm_3 = xj1_x * xj1_x * xj1_x / (6.0 * hj);
    if xi > kmax {
        cjm_3 = 0.0;
    }
    let cjm_1 = hj * xj1_x / 6.0;
    let cjm = cjm_3 - cjm_1;

    let mut cjp_3 = x_xj * x_xj * x_xj / (6.0 * hj);
    if xi < kmin {
        cjp_3 = 0.0;
    }
    let cjp_1 = hj * x_xj / 6.0;
    let cjp = cjp_3 - cjp_1;

    (ajm, ajp, cjm, cjp, j)
}

/// The cyclic mapping matrix `F = B^{-1} D` (Wood, 2006, p. 146;
/// patsy `_get_cyclic_f`). `B` and `D` are `n x n` with `n = len(knots) - 1`.
fn cyclic_b_and_d(knots: &Array1<f64>) -> (Array2<f64>, Array2<f64>) {
    let m = knots.len();
    let n = m - 1;
    let h: Vec<f64> = (0..m - 1).map(|i| knots[i + 1] - knots[i]).collect();

    let mut b = Array2::<f64>::zeros((n, n));
    let mut d = Array2::<f64>::zeros((n, n));

    b[[0, 0]] = (h[n - 1] + h[0]) / 3.0;
    b[[0, n - 1]] = h[n - 1] / 6.0;
    b[[n - 1, 0]] = h[n - 1] / 6.0;

    d[[0, 0]] = -1.0 / h[0] - 1.0 / h[n - 1];
    d[[0, n - 1]] = 1.0 / h[n - 1];
    d[[n - 1, 0]] = 1.0 / h[n - 1];

    for i in 1..n {
        b[[i, i]] = (h[i - 1] + h[i]) / 3.0;
        b[[i, i - 1]] = h[i - 1] / 6.0;
        b[[i - 1, i]] = h[i - 1] / 6.0;

        d[[i, i]] = -1.0 / h[i - 1] - 1.0 / h[i];
        d[[i, i - 1]] = 1.0 / h[i - 1];
        d[[i - 1, i]] = 1.0 / h[i - 1];
    }
    (b, d)
}

/// The unconstrained cyclic cubic regression spline design matrix
/// (patsy `_get_free_crs_dmatrix` with `cyclic=True`). Has `len(knots) - 1`
/// columns.
fn free_cyclic_dmatrix(x: &Array1<f64>, knots: &Array1<f64>) -> Result<Array2<f64>> {
    let m = knots.len();
    let n = m - 1; // cyclic: drop one column.
    let lo = knots[0];
    let hi = knots[m - 1];
    let kslice = knots
        .as_slice()
        .ok_or_else(|| Error::Value("knots must be contiguous".into()))?;

    // F = B^{-1} D.
    let (b, d) = cyclic_b_and_d(knots);
    let f = solve_matrix(&b, &d)?; // F = B^{-1} D, n x n.

    let nx = x.len();
    let mut dm = Array2::<f64>::zeros((nx, n));
    for r in 0..nx {
        let xi = map_cyclic(x[r], lo, hi);
        let (ajm, ajp, cjm, cjp, j) = base_functions(xi, kslice);
        // j1 = j + 1, wrapped to 0 if it hits n (cyclic).
        let mut j1 = j + 1;
        if j1 == n {
            j1 = 0;
        }
        // dmt column = ajm*e_j + ajp*e_j1 + cjm*F[j,:] + cjp*F[j1,:].
        // Identity contributions:
        dm[[r, j]] += ajm;
        dm[[r, j1]] += ajp;
        // F-row contributions:
        for c in 0..n {
            dm[[r, c]] += cjm * f[[j, c]] + cjp * f[[j1, c]];
        }
    }
    Ok(dm)
}

/// The cyclic wiggliness penalty `s = d' B^{-1} d` (patsy/Wood, p. 146).
fn cyclic_penalty(knots: &Array1<f64>) -> Result<Array2<f64>> {
    let (b, d) = cyclic_b_and_d(knots);
    let binv = inv(&b)?;
    let s = d.t().dot(&binv).dot(&d);
    Ok(s)
}

/// Centering reparameterization transform `T` (`k` x `k-1`) whose columns span
/// the null space of the constraint row `c` (length `k`).
///
/// This reproduces the reference's `transf_constraints`, which takes the full
/// QR of `c'` and keeps columns `1..k` of `Q`. For a single constraint the
/// QR is one Householder reflector `H = I - 2 v v'`, so `Q = H` exactly (signs
/// included) and `T = H[:, 1..]`.
fn centering_transform(c: &[f64]) -> Array2<f64> {
    let k = c.len();
    let nrm: f64 = c.iter().map(|v| v * v).sum::<f64>().sqrt();
    // alpha = -sign(c[0]) * ||c||  (LAPACK/scipy Householder convention).
    let sign = if c[0] >= 0.0 { 1.0 } else { -1.0 };
    let alpha = -sign * nrm;
    let mut v = c.to_vec();
    v[0] -= alpha;
    let vnorm: f64 = v.iter().map(|x| x * x).sum::<f64>().sqrt();
    // Full Householder reflector H = I - 2 (v/|v|)(v/|v|)'.
    let mut h = Array2::<f64>::eye(k);
    if vnorm > 0.0 {
        for a in 0..k {
            for b in 0..k {
                h[[a, b]] -= 2.0 * (v[a] / vnorm) * (v[b] / vnorm);
            }
        }
    }
    // T = columns 1..k of H.
    let mut t = Array2::<f64>::zeros((k, k - 1));
    for a in 0..k {
        for b in 1..k {
            t[[a, b - 1]] = h[[a, b]];
        }
    }
    t
}

/// Solve `B X = D` for a matrix right-hand side `D`, column by column.
fn solve_matrix(b: &Array2<f64>, d: &Array2<f64>) -> Result<Array2<f64>> {
    let n = d.ncols();
    let m = d.nrows();
    let mut out = Array2::<f64>::zeros((m, n));
    for c in 0..n {
        let rhs = d.column(c).to_owned();
        let sol = solve(b, &rhs)?;
        for r in 0..m {
            out[[r, c]] = sol[r];
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array1;

    #[test]
    fn knot_count_and_basis_shape() {
        let x = Array1::linspace(0.0, 1.0, 30);
        let df = 6;
        let cc = CyclicCubicSplines::new(&x, df).unwrap();
        // df + 1 distinct knots, df basis columns.
        assert_eq!(cc.knots().len(), df + 1);
        assert_eq!(cc.dim_basis(), df);
        assert_eq!(cc.basis().nrows(), x.len());
        assert_eq!(cc.basis().ncols(), df);
        assert_eq!(cc.cov_der2().dim(), (df, df));
    }

    #[test]
    fn penalty_symmetric_psd() {
        let x = Array1::linspace(-1.0, 2.0, 50);
        let cc = CyclicCubicSplines::new(&x, 7).unwrap();
        let s = cc.cov_der2();
        for a in 0..7 {
            for b in 0..7 {
                assert!(
                    (s[[a, b]] - s[[b, a]]).abs() < 1e-9,
                    "asymmetric at {a},{b}"
                );
            }
        }
        for seed in 0..6u64 {
            let v: Array1<f64> =
                Array1::from_shape_fn(7, |i| ((seed + i as u64 * 5) % 13) as f64 - 6.0);
            let q = v.dot(&s.dot(&v));
            assert!(q >= -1e-8, "penalty quadratic form negative: {q}");
        }
    }

    #[test]
    fn map_cyclic_wraps() {
        // Inside the range is unchanged.
        assert!((map_cyclic(0.5, 0.0, 1.0) - 0.5).abs() < 1e-15);
        // Just above the upper bound wraps near the lower bound.
        assert!((map_cyclic(1.25, 0.0, 1.0) - 0.25).abs() < 1e-12);
        // Just below the lower bound wraps near the upper bound.
        assert!((map_cyclic(-0.25, 0.0, 1.0) - 0.75).abs() < 1e-12);
    }
}
