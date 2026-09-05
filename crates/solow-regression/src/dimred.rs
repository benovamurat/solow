//! Sufficient dimension reduction by Sliced Inverse Regression (SIR).
//!
//! [`SlicedInverseReg`] estimates the *effective dimension reduction* (EDR)
//! subspace of K.-C. Li (1991): the directions `β` such that the response
//! depends on the covariates only through `xβ`. The estimator is:
//!
//! 1. sort the rows of `X` by the response `y` (ascending),
//! 2. centre the covariates and *whiten* them with the Cholesky factor of
//!    their covariance, `Σ̂ₓ = LLᵀ`, `z = L⁻¹(x − x̄)`,
//! 3. partition the whitened rows into contiguous slices (so each slice holds
//!    observations with adjacent `y`),
//! 4. form the between-slice covariance of the slice means
//!    `M = Σ_k (n_k / n) m_k m_kᵀ`,
//! 5. take the eigendecomposition of `M`; the leading eigenvectors `b`, mapped
//!    back through `Lᵀβ = b`, are the estimated EDR directions and the
//!    eigenvalues measure their importance.
//!
//! This matches the reference `SlicedInverseReg(y, X).fit(slice_n)`: `params`
//! are the EDR directions (columns) and the eigenvalues are returned in
//! descending order.

use ndarray::{Array1, Array2};
use solow_core::error::{Error, Result};
use solow_linalg::{cholesky, eigh, solve_matrix};

/// A Sliced Inverse Regression model awaiting estimation.
#[derive(Clone, Debug)]
pub struct SlicedInverseReg {
    endog: Array1<f64>,
    exog: Array2<f64>,
}

/// The fitted result of a [`SlicedInverseReg`] model.
#[derive(Clone, Debug)]
pub struct SirResults {
    /// Estimated EDR directions, one per column (`k_vars × k_vars`), ordered by
    /// decreasing eigenvalue. Each column is determined only up to sign.
    pub params: Array2<f64>,
    /// Eigenvalues of the between-slice covariance, in descending order.
    pub eigenvalues: Array1<f64>,
}

impl SlicedInverseReg {
    /// Create a SIR model from a response and a covariate matrix.
    pub fn new(endog: Array1<f64>, exog: Array2<f64>) -> Result<Self> {
        if endog.len() != exog.nrows() {
            return Err(Error::Shape(format!(
                "endog length {} != exog rows {}",
                endog.len(),
                exog.nrows()
            )));
        }
        if exog.nrows() == 0 {
            return Err(Error::Value("empty design matrix".into()));
        }
        Ok(SlicedInverseReg { endog, exog })
    }

    /// Estimate the EDR space, targeting roughly `slice_n` observations per
    /// slice (the reference default is 20).
    pub fn fit(&self, slice_n: usize) -> Result<SirResults> {
        let n = self.exog.nrows();
        let p = self.exog.ncols();
        if slice_n == 0 {
            return Err(Error::Value("slice_n must be >= 1".into()));
        }
        let n_slice = n / slice_n;
        if n_slice < 1 {
            return Err(Error::Value(
                "too few observations for even a single slice".into(),
            ));
        }

        // Sort rows of exog by endog ascending. `argsort` uses a stable sort to
        // match the reference's default (numpy argsort, stable for ties here via
        // index as secondary key).
        let mut idx: Vec<usize> = (0..n).collect();
        idx.sort_by(|&a, &b| {
            self.endog[a]
                .partial_cmp(&self.endog[b])
                .unwrap_or(std::cmp::Ordering::Equal)
                .then(a.cmp(&b))
        });
        let mut x = Array2::<f64>::zeros((n, p));
        for (newi, &oldi) in idx.iter().enumerate() {
            for j in 0..p {
                x[[newi, j]] = self.exog[[oldi, j]];
            }
        }

        // Centre columns.
        let nf = n as f64;
        let mut means = Array1::<f64>::zeros(p);
        for j in 0..p {
            means[j] = x.column(j).sum() / nf;
        }
        for i in 0..n {
            for j in 0..p {
                x[[i, j]] -= means[j];
            }
        }

        // covx = xᵀ x / n; Cholesky lower factor covxr (covx = L Lᵀ).
        let covx = x.t().dot(&x) / nf;
        let covxr = cholesky(&covx)?;

        // Whiten: solve L z = xᵀ columnwise, i.e. z = (L⁻¹ xᵀ)ᵀ.
        let xt = x.t().to_owned();
        let zt = solve_matrix(&covxr, &xt)?; // p × n
        let z = zt.t().to_owned(); // n × p (whitened rows)

        // Split whitened rows into `n_slice` contiguous slices, numpy
        // array_split semantics: the first `n % n_slice` slices get one extra.
        let base = n / n_slice;
        let rem = n % n_slice;
        let mut start = 0usize;
        let mut slice_means = Array2::<f64>::zeros((n_slice, p));
        let mut slice_counts = Array1::<f64>::zeros(n_slice);
        for s in 0..n_slice {
            let len = base + if s < rem { 1 } else { 0 };
            let end = start + len;
            for j in 0..p {
                let mut acc = 0.0;
                for i in start..end {
                    acc += z[[i, j]];
                }
                slice_means[[s, j]] = acc / len as f64;
            }
            slice_counts[s] = len as f64;
            start = end;
        }

        // Between-slice covariance: M = mnᵀ (n[:,None] * mn) / Σ n.
        let total: f64 = slice_counts.sum();
        let mut weighted = slice_means.clone();
        for s in 0..n_slice {
            for j in 0..p {
                weighted[[s, j]] *= slice_counts[s];
            }
        }
        let mnc = slice_means.t().dot(&weighted) / total; // p × p, symmetric

        // Eigendecomposition (ascending), then sort descending.
        let (evals_asc, evecs_asc) = eigh(&mnc)?;
        let mut order: Vec<usize> = (0..p).collect();
        order.sort_by(|&a, &b| {
            evals_asc[b]
                .partial_cmp(&evals_asc[a])
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        let mut eigenvalues = Array1::<f64>::zeros(p);
        let mut b = Array2::<f64>::zeros((p, p));
        for (newj, &oldj) in order.iter().enumerate() {
            eigenvalues[newj] = evals_asc[oldj];
            for i in 0..p {
                b[[i, newj]] = evecs_asc[[i, oldj]];
            }
        }

        // params = solve(covxrᵀ, b): Lᵀ params = b.
        let covxr_t = covxr.t().to_owned();
        let params = solve_matrix(&covxr_t, &b)?;

        Ok(SirResults {
            params,
            eigenvalues,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sir_runs_and_orders_eigenvalues() {
        // Deterministic data with a single strong direction along x0.
        let n = 100;
        let p = 3;
        let mut exog = Array2::<f64>::zeros((n, p));
        let mut endog = Array1::<f64>::zeros(n);
        for i in 0..n {
            let t = i as f64 / n as f64;
            exog[[i, 0]] = (t * 5.5).sin();
            exog[[i, 1]] = (t * 2.5).cos();
            exog[[i, 2]] = t - 0.5;
            endog[i] = 2.0 * exog[[i, 0]] + 0.01 * exog[[i, 2]];
        }
        let res = SlicedInverseReg::new(endog, exog).unwrap().fit(20).unwrap();
        // Eigenvalues descending.
        for w in res.eigenvalues.windows(2) {
            assert!(w[0] >= w[1] - 1e-12);
        }
        assert_eq!(res.params.dim(), (p, p));
        // Leading eigenvalue should dominate.
        assert!(res.eigenvalues[0] > res.eigenvalues[1]);
    }
}
