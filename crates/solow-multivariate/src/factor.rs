//! Factor analysis via the iterative principal-axis (`method='pa'`) method.
//!
//! The implementation mirrors the reference's
//! `multivariate.factor.Factor` with `method='pa'` and no rotation. Given a
//! correlation matrix `R` (or raw data, from which the correlation matrix is
//! formed) and a requested number of factors, communalities are estimated
//! iteratively:
//!
//! 1. Initialise communalities. With squared multiple correlations
//!    (`smc = true`, the default) the initial communality of variable `j` is
//!    `1 - 1 / (R^{-1})_{jj}`; otherwise communalities start at one.
//! 2. Replace the diagonal of `R` with the current communalities, take the
//!    symmetric eigendecomposition, and keep the `n` leading eigenvectors with
//!    positive eigenvalues (`n = min(n_factor, #positive eigenvalues)`).
//! 3. Form the loadings `A = V diag(sqrt(L))` and the new communalities as the
//!    row sums of `A^2`.
//! 4. Stop when `‖c_last - c‖ < tol`, otherwise repeat.
//!
//! The fitted [`FactorResults`] expose the loadings, communalities,
//! uniquenesses and the eigenvalues of the final adjusted correlation matrix.

use ndarray::{Array1, Array2};
use solow_core::error::{Error, Result};
use solow_linalg::{eigh, inv, matrix_rank};

/// A principal-axis factor analysis awaiting estimation.
///
/// Construct from a correlation matrix with [`Factor::from_corr`] or from a raw
/// `nobs × nvar` data matrix with [`Factor::from_data`] (which standardises the
/// columns into a Pearson correlation matrix), set the number of factors, then
/// call [`Factor::fit`].
#[derive(Clone, Debug)]
pub struct Factor {
    corr: Array2<f64>,
    n_factor: usize,
    smc: bool,
}

impl Factor {
    /// A factor analysis on a correlation (or covariance) matrix `corr`.
    ///
    /// `n_factor` is the number of factors to extract; `smc` selects squared
    /// multiple correlations as the initial communality estimate (matching the
    /// reference default `smc=True`).
    pub fn from_corr(corr: Array2<f64>, n_factor: usize, smc: bool) -> Self {
        Factor {
            corr,
            n_factor,
            smc,
        }
    }

    /// A factor analysis on a raw `nobs × nvar` data matrix. The columns are
    /// turned into a Pearson correlation matrix (population covariance scaled by
    /// the standard deviations) before estimation.
    pub fn from_data(data: &Array2<f64>, n_factor: usize, smc: bool) -> Self {
        let corr = corrcoef(data);
        Factor {
            corr,
            n_factor,
            smc,
        }
    }

    /// Estimate the factor model with the iterative principal-axis method.
    ///
    /// `maxiter` bounds the number of communality-update iterations and `tol`
    /// is the convergence threshold on the change in communalities (the
    /// reference requires `0 < tol <= 0.01`).
    pub fn fit(&self, maxiter: usize, tol: f64) -> Result<FactorResults> {
        let p = self.corr.nrows();
        if self.corr.ncols() != p {
            return Err(Error::Shape("corr must be square".into()));
        }
        if maxiter == 0 {
            return Err(Error::Shape("maxiter must be larger than 0".into()));
        }
        if tol <= 0.0 || tol > 0.01 {
            return Err(Error::Shape(
                "tol must be larger than 0 and smaller than 0.01".into(),
            ));
        }
        let n_comp = matrix_rank(&self.corr)?;
        if self.n_factor > n_comp {
            return Err(Error::Shape(
                "n_factor must be smaller or equal to the rank of the data".into(),
            ));
        }
        if self.n_factor == 0 {
            return Err(Error::Shape("n_factor must be positive".into()));
        }

        let mut r = self.corr.clone();

        // Initial communality estimate.
        let mut c = if self.smc {
            let rinv = inv(&self.corr)?;
            Array1::from_iter((0..p).map(|j| 1.0 - 1.0 / rinv[[j, j]]))
        } else {
            Array1::<f64>::ones(p)
        };

        let mut eigenvals = Array1::<f64>::zeros(p);
        let mut loadings = Array2::<f64>::zeros((p, self.n_factor));

        for _ in 0..maxiter {
            // Replace the diagonal of R with the current communalities.
            for j in 0..p {
                r[[j, j]] = c[j];
            }

            // Symmetric eigendecomposition; `eigh` returns ascending order.
            let (l_asc, v_asc) = eigh(&r)?;
            let c_last = c.clone();

            // Sort eigenvalues / eigenvectors descending.
            let mut order: Vec<usize> = (0..p).collect();
            order.sort_by(|&a, &b| l_asc[b].total_cmp(&l_asc[a]));

            let l_desc = Array1::from_iter(order.iter().map(|&i| l_asc[i]));
            eigenvals = l_desc.clone();

            let n_pos = l_desc.iter().filter(|&&x| x > 0.0).count();
            let n = n_pos.min(self.n_factor);

            // A = V[:, :n] diag(sqrt(L[:n])).
            let mut a = Array2::<f64>::zeros((p, self.n_factor));
            for k in 0..n {
                let col = order[k];
                let sl = l_desc[k].sqrt();
                for row in 0..p {
                    a[[row, k]] = v_asc[[row, col]] * sl;
                }
            }
            // Columns beyond `n` (only possible when n_factor exceeds the number
            // of positive eigenvalues) stay zero, matching the reference, which
            // keeps fewer columns; here the trailing columns contribute nothing
            // to the communalities.
            loadings = a.clone();

            // New communalities: row sums of A^2.
            for row in 0..p {
                let mut s = 0.0;
                for k in 0..self.n_factor {
                    s += a[[row, k]] * a[[row, k]];
                }
                c[row] = s;
            }

            // Convergence: ‖c_last - c‖ < tol.
            let diff: f64 = (0..p)
                .map(|i| (c_last[i] - c[i]).powi(2))
                .sum::<f64>()
                .sqrt();
            if diff < tol {
                break;
            }
        }

        let uniqueness = c.mapv(|x| 1.0 - x);

        Ok(FactorResults {
            eigenvals,
            communality: c,
            uniqueness,
            loadings,
            n_factor: self.n_factor,
        })
    }
}

/// Pearson correlation matrix of the columns of `data` (population convention,
/// matching `numpy.corrcoef` on `rowvar=False`).
fn corrcoef(data: &Array2<f64>) -> Array2<f64> {
    let (nobs, nvar) = data.dim();
    let mut means = Array1::<f64>::zeros(nvar);
    for j in 0..nvar {
        means[j] = data.column(j).sum() / nobs as f64;
    }
    // Centred data.
    let mut centred = data.clone();
    for j in 0..nvar {
        for i in 0..nobs {
            centred[[i, j]] -= means[j];
        }
    }
    // Covariance (un-normalised cross products) then scale by column norms.
    let mut cov = Array2::<f64>::zeros((nvar, nvar));
    for a in 0..nvar {
        for b in a..nvar {
            let mut s = 0.0;
            for i in 0..nobs {
                s += centred[[i, a]] * centred[[i, b]];
            }
            cov[[a, b]] = s;
            cov[[b, a]] = s;
        }
    }
    let mut std = Array1::<f64>::zeros(nvar);
    for j in 0..nvar {
        std[j] = cov[[j, j]].sqrt();
    }
    let mut corr = Array2::<f64>::zeros((nvar, nvar));
    for a in 0..nvar {
        for b in 0..nvar {
            let denom = std[a] * std[b];
            corr[[a, b]] = if denom > 0.0 {
                cov[[a, b]] / denom
            } else {
                0.0
            };
        }
    }
    corr
}

/// A fitted principal-axis factor model.
#[derive(Clone, Debug)]
pub struct FactorResults {
    /// Eigenvalues of the final adjusted correlation matrix, descending,
    /// length `nvar`.
    pub eigenvals: Array1<f64>,
    /// Estimated communalities (variance of each variable explained by the
    /// retained factors), length `nvar`.
    pub communality: Array1<f64>,
    /// Uniquenesses `1 - communality`, length `nvar`.
    pub uniqueness: Array1<f64>,
    /// Factor loadings, `nvar × n_factor`.
    pub loadings: Array2<f64>,
    /// Number of factors extracted.
    pub n_factor: usize,
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn communality_plus_uniqueness_is_one() {
        let corr = array![[1.0, 0.6, 0.5], [0.6, 1.0, 0.4], [0.5, 0.4, 1.0],];
        let res = Factor::from_corr(corr, 1, true).fit(500, 1e-10).unwrap();
        for i in 0..3 {
            assert!((res.communality[i] + res.uniqueness[i] - 1.0).abs() < 1e-12);
        }
    }

    #[test]
    fn loadings_reproduce_communalities() {
        let corr = array![[1.0, 0.6, 0.5], [0.6, 1.0, 0.4], [0.5, 0.4, 1.0],];
        let res = Factor::from_corr(corr, 2, true).fit(500, 1e-10).unwrap();
        for i in 0..3 {
            let s: f64 = (0..res.n_factor)
                .map(|k| res.loadings[[i, k]].powi(2))
                .sum();
            assert!((s - res.communality[i]).abs() < 1e-10);
        }
    }

    #[test]
    fn eigenvals_descending() {
        let corr = array![[1.0, 0.6, 0.5], [0.6, 1.0, 0.4], [0.5, 0.4, 1.0],];
        let res = Factor::from_corr(corr, 2, true).fit(500, 1e-10).unwrap();
        for i in 1..res.eigenvals.len() {
            assert!(res.eigenvals[i - 1] >= res.eigenvals[i]);
        }
    }

    #[test]
    fn from_data_matches_corr_construction() {
        let data = array![
            [1.0, 2.0, 0.5],
            [2.0, 1.0, 1.5],
            [3.0, 0.0, 2.5],
            [4.0, 1.0, 0.0],
            [2.0, 3.0, 1.0],
        ];
        let corr = corrcoef(&data);
        let a = Factor::from_data(&data, 1, true).fit(500, 1e-10).unwrap();
        let b = Factor::from_corr(corr, 1, true).fit(500, 1e-10).unwrap();
        for i in 0..3 {
            assert!((a.communality[i] - b.communality[i]).abs() < 1e-12);
        }
    }
}
