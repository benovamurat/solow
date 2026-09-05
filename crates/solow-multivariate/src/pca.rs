//! Principal component analysis (PCA).
//!
//! The implementation mirrors the reference's `multivariate.pca.PCA` with the
//! singular-value-decomposition method, supporting the `standardize`, `demean`
//! and `normalize` options. Given an `nobs × nvar` data matrix it produces, for
//! a requested number of components `ncomp`:
//!
//! * [`PcaResults::eigenvals`] — the eigenvalues of the (transformed) cross
//!   product, sorted in descending order (squared singular values);
//! * [`PcaResults::eigenvecs`] / [`PcaResults::loadings`] — the right singular
//!   vectors (one per column), the principal axes;
//! * [`PcaResults::factors`] / [`PcaResults::scores`] — the projection of the
//!   transformed data onto the components;
//! * [`PcaResults::coeff`] — the loadings used to reconstruct the data from the
//!   factors;
//! * [`PcaResults::rsquare`] — the cumulative fraction of variance explained as
//!   components are added (length `ncomp + 1`, starting at `0`);
//! * [`PcaResults::explained_variance_ratio`] — the per-component fraction of
//!   total variance, derived from the eigenvalues.

use ndarray::{s, Array1, Array2};
use solow_core::error::{Error, Result};
use solow_linalg::svd;

/// A principal component analysis awaiting estimation.
///
/// Construct with [`Pca::new`] (which mirrors the reference defaults:
/// `standardize = true`, `demean = true`, `normalize = true`, all components),
/// optionally adjust the builder options, then call [`Pca::fit`].
#[derive(Clone, Debug)]
pub struct Pca {
    data: Array2<f64>,
    ncomp: Option<usize>,
    standardize: bool,
    demean: bool,
    normalize: bool,
}

impl Pca {
    /// A PCA on `data` (an `nobs × nvar` matrix) with the reference defaults:
    /// standardized columns, all components, normalized factors.
    pub fn new(data: Array2<f64>) -> Self {
        Pca {
            data,
            ncomp: None,
            standardize: true,
            demean: true,
            normalize: true,
        }
    }

    /// Set the number of components to retain. Defaults to `min(nobs, nvar)`.
    pub fn ncomp(mut self, ncomp: usize) -> Self {
        self.ncomp = Some(ncomp);
        self
    }

    /// Whether to standardize each column to unit variance (population, ddof 0)
    /// before decomposition. When `true`, this implies demeaning. Default `true`.
    pub fn standardize(mut self, standardize: bool) -> Self {
        self.standardize = standardize;
        self
    }

    /// Whether to subtract each column's mean before decomposition. Ignored when
    /// `standardize` is `true` (standardizing already demeans). Default `true`.
    pub fn demean(mut self, demean: bool) -> Self {
        self.demean = demean;
        self
    }

    /// Whether to normalize the factors to unit inner product, scaling the
    /// coefficients by `sqrt(eigenvalue)`. Default `true`.
    pub fn normalize(mut self, normalize: bool) -> Self {
        self.normalize = normalize;
        self
    }

    /// Estimate the PCA via SVD.
    pub fn fit(&self) -> Result<PcaResults> {
        let (nobs, nvar) = self.data.dim();
        if nobs == 0 || nvar == 0 {
            return Err(Error::Shape("data must be non-empty".into()));
        }
        let min_dim = nobs.min(nvar);
        let ncomp = match self.ncomp {
            None => min_dim,
            Some(c) => {
                if c == 0 {
                    return Err(Error::Shape("ncomp must be positive".into()));
                }
                c.min(min_dim)
            }
        };

        // --- Prepare data: standardize / demean (population statistics). ---
        let mut mu = Array1::<f64>::zeros(nvar);
        let mut sigma = Array1::<f64>::ones(nvar);
        for j in 0..nvar {
            let col = self.data.column(j);
            let m = col.sum() / nobs as f64;
            mu[j] = m;
            let var = col.iter().map(|&x| (x - m) * (x - m)).sum::<f64>() / nobs as f64;
            sigma[j] = var.sqrt();
        }

        let mut transformed = self.data.clone();
        if self.standardize {
            for j in 0..nvar {
                let s = sigma[j];
                for i in 0..nobs {
                    transformed[[i, j]] = (transformed[[i, j]] - mu[j]) / s;
                }
            }
        } else if self.demean {
            for j in 0..nvar {
                for i in 0..nobs {
                    transformed[[i, j]] -= mu[j];
                }
            }
        }

        // --- SVD: eigenvals = s^2, eigenvecs = V (columns are axes). ---
        let (_u, s, vt) = svd(&transformed)?;
        let eig_all = s.mapv(|x| x * x);
        // `vt` is k × nvar (k = min(nobs, nvar)); columns of `v = vt.T` are the
        // right singular vectors. Sort by eigenvalue descending (SVD already
        // returns descending, but sort explicitly to match the reference).
        let k = s.len();
        let mut order: Vec<usize> = (0..k).collect();
        order.sort_by(|&a, &b| eig_all[b].total_cmp(&eig_all[a]));

        // Build sorted, truncated eigenvalues and eigenvectors.
        let mut eigenvals = Array1::<f64>::zeros(ncomp);
        let mut eigenvecs = Array2::<f64>::zeros((nvar, ncomp));
        for (c, &idx) in order.iter().take(ncomp).enumerate() {
            eigenvals[c] = eig_all[idx];
            for r in 0..nvar {
                eigenvecs[[r, c]] = vt[[idx, r]];
            }
        }

        // --- factors / loadings / coeff. ---
        // factors = scores = transformed @ vecs   (nobs × ncomp)
        let mut factors = transformed.dot(&eigenvecs);
        let loadings = eigenvecs.clone();
        // coeff = vecs.T   (ncomp × nvar)
        let mut coeff = eigenvecs.t().to_owned();

        if self.normalize {
            // coeff = (coeff.T * sqrt(vals)).T  -> scale row c by sqrt(vals[c]).
            for c in 0..ncomp {
                let sc = eigenvals[c].sqrt();
                for j in 0..nvar {
                    coeff[[c, j]] *= sc;
                }
                let inv = 1.0 / sc;
                for i in 0..nobs {
                    factors[[i, c]] *= inv;
                }
            }
        }
        let scores = factors.clone();

        // --- R-square (cumulative) and explained variance ratio. ---
        // TSS uses the transformed data (weights are unity).
        let tss_indiv: Array1<f64> = (0..nvar)
            .map(|j| transformed.column(j).iter().map(|&x| x * x).sum::<f64>())
            .collect();
        let tss: f64 = tss_indiv.sum();

        let mut rsquare = Array1::<f64>::zeros(ncomp + 1);
        for i in 0..=ncomp {
            // Projection onto the first i factors, in transformed space.
            let proj = project(&factors, &coeff, i);
            let rss: f64 = proj.iter().map(|&x| x * x).sum::<f64>();
            let ess = tss - rss;
            rsquare[i] = 1.0 - ess / tss;
        }

        // Per-component fraction of total variance from the eigenvalues.
        let explained_variance_ratio = eigenvals.mapv(|v| v / tss);

        Ok(PcaResults {
            eigenvals,
            eigenvecs,
            loadings,
            factors,
            scores,
            coeff,
            rsquare,
            explained_variance_ratio,
            mu,
            sigma,
            ncomp,
            nobs,
            nvar,
            standardize: self.standardize,
            demean: self.demean,
        })
    }
}

/// Reconstruct `factors[:, :ncomp] @ coeff[:ncomp, :]` (in transformed space).
fn project(factors: &Array2<f64>, coeff: &Array2<f64>, ncomp: usize) -> Array2<f64> {
    let f = factors.slice(s![.., 0..ncomp]);
    let c = coeff.slice(s![0..ncomp, ..]);
    f.dot(&c)
}

/// Fitted principal component analysis.
#[derive(Clone, Debug)]
pub struct PcaResults {
    /// Eigenvalues (squared singular values), descending, length `ncomp`.
    pub eigenvals: Array1<f64>,
    /// Eigenvectors / principal axes, `nvar × ncomp` (one axis per column).
    pub eigenvecs: Array2<f64>,
    /// Loadings, identical to [`Self::eigenvecs`] (`nvar × ncomp`).
    pub loadings: Array2<f64>,
    /// Factor scores, `nobs × ncomp` (normalized when `normalize` is set).
    pub factors: Array2<f64>,
    /// Alias of [`Self::factors`].
    pub scores: Array2<f64>,
    /// Reconstruction coefficients, `ncomp × nvar`.
    pub coeff: Array2<f64>,
    /// Cumulative variance explained, length `ncomp + 1` (starts at `0`).
    pub rsquare: Array1<f64>,
    /// Per-component fraction of total variance, length `ncomp`.
    pub explained_variance_ratio: Array1<f64>,
    /// Column means of the input data.
    pub mu: Array1<f64>,
    /// Column population standard deviations of the input data.
    pub sigma: Array1<f64>,
    /// Number of components retained.
    pub ncomp: usize,
    /// Number of observations.
    pub nobs: usize,
    /// Number of variables.
    pub nvar: usize,
    standardize: bool,
    demean: bool,
}

impl PcaResults {
    /// Project the (possibly fewer) factors back into the original data space,
    /// undoing the standardize/demean transformation. Returns an `nobs × nvar`
    /// reconstruction using the first `ncomp` components.
    pub fn project(&self, ncomp: usize) -> Result<Array2<f64>> {
        if ncomp > self.ncomp {
            return Err(Error::Shape(
                "ncomp must not exceed the number of fitted components".into(),
            ));
        }
        let mut projection = project(&self.factors, &self.coeff, ncomp);
        if self.standardize {
            for j in 0..self.nvar {
                for i in 0..self.nobs {
                    projection[[i, j]] *= self.sigma[j];
                }
            }
        }
        if self.standardize || self.demean {
            for j in 0..self.nvar {
                for i in 0..self.nobs {
                    projection[[i, j]] += self.mu[j];
                }
            }
        }
        Ok(projection)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    fn check_close(a: &Array1<f64>, b: &[f64], tol: f64) {
        assert_eq!(a.len(), b.len());
        for i in 0..a.len() {
            assert!((a[i] - b[i]).abs() <= tol, "idx {i}: {} vs {}", a[i], b[i]);
        }
    }

    #[test]
    fn eigenvals_descending() {
        let data = array![
            [1.0, 2.0, 0.5],
            [2.0, 1.0, 1.5],
            [3.0, 0.0, 2.5],
            [4.0, 1.0, 0.0],
            [2.0, 3.0, 1.0],
        ];
        let res = Pca::new(data).fit().unwrap();
        for c in 1..res.eigenvals.len() {
            assert!(res.eigenvals[c - 1] >= res.eigenvals[c]);
        }
    }

    #[test]
    fn rsquare_monotone_and_full() {
        let data = array![
            [1.0, 2.0, 0.5],
            [2.0, 1.0, 1.5],
            [3.0, 0.0, 2.5],
            [4.0, 1.0, 0.0],
            [2.0, 3.0, 1.0],
        ];
        let res = Pca::new(data).fit().unwrap();
        // First entry is 0, monotone increasing, last reaches 1 (full rank).
        assert!((res.rsquare[0]).abs() < 1e-12);
        for i in 1..res.rsquare.len() {
            assert!(res.rsquare[i] >= res.rsquare[i - 1] - 1e-12);
        }
        let last = res.rsquare[res.rsquare.len() - 1];
        assert!((last - 1.0).abs() < 1e-9);
    }

    #[test]
    fn explained_variance_sums_to_one_full_rank() {
        let data = array![
            [1.0, 2.0, 0.5],
            [2.0, 1.0, 1.5],
            [3.0, 0.0, 2.5],
            [4.0, 1.0, 0.0],
            [2.0, 3.0, 1.0],
        ];
        let res = Pca::new(data).fit().unwrap();
        let total: f64 = res.explained_variance_ratio.sum();
        assert!((total - 1.0).abs() < 1e-9);
    }

    #[test]
    fn standardize_demeans_and_scales() {
        // With standardize, transformed columns have zero mean, unit variance,
        // so eigenvalues sum to nobs * nvar (trace of correlation * nobs).
        let data = array![[1.0, 5.0], [2.0, 3.0], [3.0, 4.0], [4.0, 1.0],];
        let res = Pca::new(data).fit().unwrap();
        let sum: f64 = res.eigenvals.sum();
        // trace of X'X where each column standardized (population) = nobs per col.
        assert!((sum - (res.nobs * res.nvar) as f64).abs() < 1e-9);
    }

    #[test]
    fn reconstruction_recovers_data_full_rank() {
        let data = array![
            [1.0, 2.0, 0.5],
            [2.0, 1.0, 1.5],
            [3.0, 0.0, 2.5],
            [4.0, 1.0, 0.0],
            [2.0, 3.0, 1.0],
        ];
        let res = Pca::new(data.clone()).fit().unwrap();
        let recon = res.project(res.ncomp).unwrap();
        for i in 0..data.nrows() {
            for j in 0..data.ncols() {
                assert!((recon[[i, j]] - data[[i, j]]).abs() < 1e-7);
            }
        }
    }

    #[test]
    fn no_standardize_no_demean() {
        let data = array![[2.0, 0.0], [0.0, 2.0], [1.0, 1.0]];
        let res = Pca::new(data)
            .standardize(false)
            .demean(false)
            .normalize(false)
            .fit()
            .unwrap();
        // mu computed but not applied; check eigenvecs orthonormal.
        let v = &res.eigenvecs;
        let mut dot = 0.0;
        for r in 0..v.nrows() {
            dot += v[[r, 0]] * v[[r, 0]];
        }
        check_close(&array![dot], &[1.0], 1e-9);
    }
}
