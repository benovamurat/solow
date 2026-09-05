//! DictionaryLearning — sparse dictionary learning à la Mairal-Bach-
//! Ponce-Sapiro (2009). Alternating scheme:
//!   1. Fix the dictionary `D`, solve LASSO for the sparse codes `α`.
//!   2. Fix the codes, update `D` column-wise via a projected gradient
//!      step that renormalises to the unit ball.

use ndarray::{Array1, Array2, ArrayView2};
use solow_core::{Error, Result};

/// Fitted DictionaryLearning.
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Clone, Debug, PartialEq)]
pub struct DictionaryLearning {
    /// Dictionary `(n_components × d)`.
    pub components: Array2<f64>,
    /// Sparse codes `(n × n_components)` at fit time.
    pub codes: Array2<f64>,
    /// Kept rank.
    pub n_components: usize,
    /// L1 penalty α used.
    pub alpha: f64,
    /// Convergence iterations run.
    pub n_iter: usize,
}

impl DictionaryLearning {
    /// Fit with defaults `alpha = 1.0`, `max_iter = 100`, `tol = 1e-6`.
    pub fn fit(x: ArrayView2<'_, f64>, n_components: usize) -> Result<Self> {
        Self::fit_with(x, n_components, 1.0, 100, 1e-6)
    }

    /// Full-configuration fit.
    pub fn fit_with(
        x: ArrayView2<'_, f64>,
        n_components: usize,
        alpha: f64,
        max_iter: usize,
        tol: f64,
    ) -> Result<Self> {
        let n = x.nrows();
        let d = x.ncols();
        if n_components == 0 {
            return Err(Error::Value("DictionaryLearning: n_components must be ≥ 1".into()));
        }
        if alpha < 0.0 {
            return Err(Error::Value("DictionaryLearning: alpha must be ≥ 0".into()));
        }
        // Init dictionary: first `n_components` rows of X, normalised.
        let mut dict = Array2::<f64>::zeros((n_components, d));
        for k in 0..n_components.min(n) {
            for j in 0..d {
                dict[[k, j]] = x[[k, j]];
            }
        }
        for k in 0..n_components {
            let mut nrm = 0.0_f64;
            for j in 0..d {
                nrm += dict[[k, j]] * dict[[k, j]];
            }
            let nrm = nrm.sqrt().max(1e-30);
            for j in 0..d {
                dict[[k, j]] /= nrm;
            }
        }
        let mut codes = Array2::<f64>::zeros((n, n_components));
        let mut iters = 0_usize;
        for it in 0..max_iter {
            iters = it + 1;
            // Sparse coding: coordinate descent Lasso per row.
            for i in 0..n {
                for _ in 0..50 {
                    let mut delta = 0.0_f64;
                    for k in 0..n_components {
                        // Residual r_k = xᵢ − Σ_{k'≠k} α_{i,k'} d_{k'}
                        //           = xᵢ − Σ_{k'} α_{i,k'} d_{k'} + α_{i,k} d_k
                        let mut num = 0.0_f64;
                        for j in 0..d {
                            let mut recon = 0.0_f64;
                            for kk in 0..n_components {
                                if kk == k {
                                    continue;
                                }
                                recon += codes[[i, kk]] * dict[[kk, j]];
                            }
                            num += dict[[k, j]] * (x[[i, j]] - recon);
                        }
                        // ‖d_k‖² = 1 after normalisation.
                        let new = soft_threshold(num, alpha);
                        let dd = (new - codes[[i, k]]).abs();
                        if dd > delta {
                            delta = dd;
                        }
                        codes[[i, k]] = new;
                    }
                    if delta < tol {
                        break;
                    }
                }
            }
            // Dictionary update — projected gradient with unit-norm reset.
            let mut new_dict = Array2::<f64>::zeros((n_components, d));
            for k in 0..n_components {
                let mut num = Array1::<f64>::zeros(d);
                let mut denom = 0.0_f64;
                for i in 0..n {
                    denom += codes[[i, k]] * codes[[i, k]];
                    for j in 0..d {
                        // r_k = xᵢ − Σ_{k'} α_{i,k'} d_{k'} + α_{i,k} d_k
                        let mut recon = 0.0_f64;
                        for kk in 0..n_components {
                            recon += codes[[i, kk]] * dict[[kk, j]];
                        }
                        num[j] += codes[[i, k]] * (x[[i, j]] - recon + codes[[i, k]] * dict[[k, j]]);
                    }
                }
                if denom > 1e-30 {
                    let mut nrm2 = 0.0_f64;
                    for j in 0..d {
                        new_dict[[k, j]] = num[j] / denom;
                        nrm2 += new_dict[[k, j]] * new_dict[[k, j]];
                    }
                    let nrm = nrm2.sqrt().max(1e-30);
                    for j in 0..d {
                        new_dict[[k, j]] /= nrm;
                    }
                } else {
                    for j in 0..d {
                        new_dict[[k, j]] = dict[[k, j]];
                    }
                }
            }
            let mut delta = 0.0_f64;
            for k in 0..n_components {
                for j in 0..d {
                    let dd = new_dict[[k, j]] - dict[[k, j]];
                    delta += dd * dd;
                }
            }
            dict = new_dict;
            if delta.sqrt() < tol {
                break;
            }
        }
        Ok(Self {
            components: dict,
            codes,
            n_components,
            alpha,
            n_iter: iters,
        })
    }
}

fn soft_threshold(z: f64, alpha: f64) -> f64 {
    if z > alpha {
        z - alpha
    } else if z < -alpha {
        z + alpha
    } else {
        0.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn dict_learning_returns_dict_of_the_right_shape() {
        let x = array![
            [1.0_f64, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0],
            [1.0, 1.0, 0.0], [0.0, 1.0, 1.0]
        ];
        let m = DictionaryLearning::fit_with(x.view(), 3, 0.1, 20, 1e-4).unwrap();
        assert_eq!(m.components.shape(), &[3, 3]);
        assert_eq!(m.codes.shape(), &[5, 3]);
    }
}
