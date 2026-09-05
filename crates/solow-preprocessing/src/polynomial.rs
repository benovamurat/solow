//! [`PolynomialFeatures`] — expand a feature matrix into all monomials
//! up to a given total degree.
//!
//! The column ordering matches `preprocessing.PolynomialFeatures`
//! and follows a **graded lexicographic** enumeration: first ascending
//! total degree, then ascending lexicographic on the exponent vector.
//!
//! # Complexity
//!
//! For `d` input features and total degree `p`:
//!
//! * Output columns: `include_bias`: `C(d + p, p)`,
//!   `interaction_only`: `Σ_{k=0..=p} C(d, k)`.
//! * Time: `O(n · m)` where `m` is the output-column count.
//! * Space: `O(m + n · m)` for the returned matrix.
//!
//! # Example
//!
//! Degree 3, `include_bias = true`, `interaction_only = false`, `d = 2`
//! produces the columns
//! `[1, x0, x1, x0², x0·x1, x1², x0³, x0²·x1, x0·x1², x1³]` — the same
//! ordering the reference returns.

use ndarray::{Array2, ArrayView2};
use solow_core::{Error, Result};

/// Build the full monomial-index vector for the given configuration.
///
/// Each entry is the exponent vector (length `d`, sums to `≤ degree`).
/// The vector is graded-lexicographic; the first entry is the all-zeros
/// bias when `include_bias`.
fn monomials(d: usize, degree: usize, include_bias: bool, interaction_only: bool) -> Vec<Vec<u32>> {
    let mut out: Vec<Vec<u32>> = Vec::new();
    for total in 0..=degree {
        if total == 0 {
            if include_bias {
                out.push(vec![0u32; d]);
            }
            continue;
        }
        // Enumerate every compositions-of-`total`-into-`d`-nonnegative-parts.
        let mut current = vec![0u32; d];
        enumerate(d, total as u32, 0, &mut current, &mut out, interaction_only);
    }
    out
}

fn enumerate(
    d: usize,
    remaining: u32,
    idx: usize,
    current: &mut Vec<u32>,
    out: &mut Vec<Vec<u32>>,
    interaction_only: bool,
) {
    if idx == d - 1 {
        current[idx] = remaining;
        if !interaction_only || current.iter().all(|&e| e <= 1) {
            out.push(current.clone());
        }
        current[idx] = 0;
        return;
    }
    let cap = if interaction_only { 1 } else { remaining };
    // Iterate the leading-variable exponent from high to low so the overall
    // enumeration matches the reference graded-lexicographic order
    // (`(2, 0), (1, 1), (0, 2)` at total degree 2 rather than the reverse).
    for k in (0..=cap.min(remaining)).rev() {
        current[idx] = k;
        enumerate(d, remaining - k, idx + 1, current, out, interaction_only);
    }
    current[idx] = 0;
}

/// Polynomial-feature expansion.
///
/// See the [module docs](crate::polynomial) for the column ordering and
/// complexity.
#[derive(Clone, Debug, PartialEq)]
pub struct PolynomialFeatures {
    /// Total polynomial degree.
    pub degree: usize,
    /// If `true`, only produce monomials in which every exponent is `0`
    /// or `1` (no `x_j²` etc.).
    pub interaction_only: bool,
    /// If `true`, prepend the all-ones bias column.
    pub include_bias: bool,
    /// Cached monomial exponents populated at `fit` time.
    exponents: Option<Vec<Vec<u32>>>,
}

impl PolynomialFeatures {
    /// New expansion at the given degree (`include_bias = true`,
    /// `interaction_only = false`).
    pub fn new(degree: usize) -> Self {
        Self {
            degree,
            interaction_only: false,
            include_bias: true,
            exponents: None,
        }
    }

    /// Set `interaction_only`.
    pub fn interaction_only(mut self, flag: bool) -> Self {
        self.interaction_only = flag;
        self
    }

    /// Set `include_bias`.
    pub fn include_bias(mut self, flag: bool) -> Self {
        self.include_bias = flag;
        self
    }

    /// Cache the monomial list.
    pub fn fit(&mut self, x: ArrayView2<'_, f64>) -> Result<()> {
        if x.ncols() == 0 {
            return Err(Error::Value(
                "PolynomialFeatures::fit: x must have at least one column".into(),
            ));
        }
        self.exponents = Some(monomials(
            x.ncols(),
            self.degree,
            self.include_bias,
            self.interaction_only,
        ));
        Ok(())
    }

    /// Number of output columns for a `d`-column input.
    pub fn output_columns(&self, d: usize) -> usize {
        monomials(d, self.degree, self.include_bias, self.interaction_only).len()
    }

    /// Apply the expansion.
    pub fn transform(&self, x: ArrayView2<'_, f64>) -> Result<Array2<f64>> {
        let d = x.ncols();
        let exps = monomials(d, self.degree, self.include_bias, self.interaction_only);
        let m = exps.len();
        let mut out = Array2::<f64>::zeros((x.nrows(), m));
        for i in 0..x.nrows() {
            for (k, exp) in exps.iter().enumerate() {
                let mut acc = 1.0_f64;
                for j in 0..d {
                    let e = exp[j];
                    if e == 0 {
                        continue;
                    }
                    let v = x[[i, j]];
                    // fast integer powers up to a small degree
                    let mut p = 1.0;
                    for _ in 0..e {
                        p *= v;
                    }
                    acc *= p;
                }
                out[[i, k]] = acc;
            }
        }
        Ok(out)
    }

    /// One-call fit + transform.
    pub fn fit_transform(&self, x: ArrayView2<'_, f64>) -> Result<Array2<f64>> {
        // Fit is stateless w.r.t. transform (the monomial list is
        // recomputed) but we keep the reference signature.
        let mut clone = self.clone();
        clone.fit(x)?;
        clone.transform(x)
    }

    /// Exponent vectors, one per output column.
    ///
    /// Returned only when `fit` has run; otherwise use
    /// [`PolynomialFeatures::exponents_for`].
    pub fn exponents(&self) -> Option<&[Vec<u32>]> {
        self.exponents.as_deref()
    }

    /// Compute the exponent vectors for a `d`-column input without
    /// consuming state.
    pub fn exponents_for(&self, d: usize) -> Vec<Vec<u32>> {
        monomials(d, self.degree, self.include_bias, self.interaction_only)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use ndarray::array;

    #[test]
    fn degree_2_matches_hand_derivation() {
        let x = array![[2.0, 3.0]];
        let poly = PolynomialFeatures::new(2).include_bias(true);
        let out = poly.fit_transform(x.view()).unwrap();
        // Columns: 1, x0, x1, x0², x0·x1, x1².
        assert_eq!(out.dim(), (1, 6));
        assert_abs_diff_eq!(out[[0, 0]], 1.0, epsilon = 1e-12);
        assert_abs_diff_eq!(out[[0, 1]], 2.0, epsilon = 1e-12);
        assert_abs_diff_eq!(out[[0, 2]], 3.0, epsilon = 1e-12);
        assert_abs_diff_eq!(out[[0, 3]], 4.0, epsilon = 1e-12);
        assert_abs_diff_eq!(out[[0, 4]], 6.0, epsilon = 1e-12);
        assert_abs_diff_eq!(out[[0, 5]], 9.0, epsilon = 1e-12);
    }

    #[test]
    fn interaction_only_drops_pure_powers() {
        let x = array![[2.0, 3.0, 5.0]];
        let poly = PolynomialFeatures::new(2)
            .include_bias(false)
            .interaction_only(true);
        let out = poly.fit_transform(x.view()).unwrap();
        // Columns: x0, x1, x2, x0·x1, x0·x2, x1·x2.
        assert_eq!(out.dim(), (1, 6));
        assert_abs_diff_eq!(out[[0, 3]], 6.0, epsilon = 1e-12); // x0·x1
        assert_abs_diff_eq!(out[[0, 4]], 10.0, epsilon = 1e-12); // x0·x2
        assert_abs_diff_eq!(out[[0, 5]], 15.0, epsilon = 1e-12); // x1·x2
    }

    #[test]
    fn column_count_matches_binomial_identity() {
        for d in 1..6 {
            for degree in 0..5 {
                let full = PolynomialFeatures::new(degree)
                    .include_bias(true)
                    .output_columns(d);
                // C(d + p, p)
                let expected: usize = (1..=degree).fold(1usize, |acc, k| acc * (d + k) / k);
                assert_eq!(full, expected, "d = {d}, p = {degree}");
            }
        }
    }
}
