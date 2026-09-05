//! MiniBatchDictionaryLearning — online update of the (D, α) pair on
//! random mini-batches (Mairal-Bach-Ponce-Sapiro 2010).

use ndarray::{Array2, ArrayView2};
use solow_core::{Error, Result};

/// Fitted MiniBatchDictionaryLearning.
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Clone, Debug, PartialEq)]
pub struct MiniBatchDictionaryLearning {
    /// Dictionary `(n_components × d)`.
    pub components: Array2<f64>,
    /// Kept rank.
    pub n_components: usize,
    /// L1 penalty α used.
    pub alpha: f64,
    /// Batch size.
    pub batch_size: usize,
    /// Iterations run.
    pub n_iter: usize,
}

impl MiniBatchDictionaryLearning {
    /// Fit with the reference defaults `alpha = 1.0`, `batch_size = 3`,
    /// `max_iter = 100`, `tol = 1e-3`.
    pub fn fit(x: ArrayView2<'_, f64>, n_components: usize) -> Result<Self> {
        Self::fit_with(x, n_components, 1.0, 3, 100, 1e-3, 0)
    }

    /// Full-configuration fit.
    pub fn fit_with(
        x: ArrayView2<'_, f64>,
        n_components: usize,
        alpha: f64,
        batch_size: usize,
        max_iter: usize,
        tol: f64,
        seed: u64,
    ) -> Result<Self> {
        let n = x.nrows();
        let d = x.ncols();
        if n_components == 0 {
            return Err(Error::Value("MiniBatchDictionaryLearning: n_components must be ≥ 1".into()));
        }
        if alpha < 0.0 {
            return Err(Error::Value("MiniBatchDictionaryLearning: alpha must be ≥ 0".into()));
        }
        // Init from the first `n_components` rows, normalised.
        let mut dict = Array2::<f64>::zeros((n_components, d));
        for k in 0..n_components.min(n) {
            for j in 0..d {
                dict[[k, j]] = x[[k, j]];
            }
            let nrm = (0..d).map(|j| dict[[k, j]] * dict[[k, j]]).sum::<f64>().sqrt().max(1e-30);
            for j in 0..d {
                dict[[k, j]] /= nrm;
            }
        }
        let mut a_stat = Array2::<f64>::zeros((n_components, n_components));
        let mut b_stat = Array2::<f64>::zeros((n_components, d));
        let mut state = seed.wrapping_add(0xF00D_C0DE);
        let mut iters = 0_usize;
        let mut prev_dict = dict.clone();
        for it in 0..max_iter {
            iters = it + 1;
            let batch = batch_size.min(n).max(1);
            for _ in 0..batch {
                let i = uniform_index(&mut state, n as u64);
                // Sparse code x_i under current dictionary via coordinate-descent LASSO.
                let alpha_i = sparse_code(x.row(i), &dict, alpha);
                // Update running statistics A += αα^T, B += x α^T.
                for k in 0..n_components {
                    for l in 0..n_components {
                        a_stat[[k, l]] += alpha_i[k] * alpha_i[l];
                    }
                    for j in 0..d {
                        b_stat[[k, j]] += x[[i, j]] * alpha_i[k];
                    }
                }
            }
            // Dictionary update — block coordinate descent on rows.
            for k in 0..n_components {
                let akk = a_stat[[k, k]].max(1e-30);
                let mut u = vec![0.0_f64; d];
                for j in 0..d {
                    let mut acc = b_stat[[k, j]];
                    for l in 0..n_components {
                        if l == k {
                            continue;
                        }
                        acc -= a_stat[[l, k]] * dict[[l, j]];
                    }
                    u[j] = dict[[k, j]] + (acc - dict[[k, j]] * akk) / akk;
                }
                let nrm = u.iter().map(|v| v * v).sum::<f64>().sqrt().max(1e-30);
                for j in 0..d {
                    dict[[k, j]] = u[j] / nrm.max(1.0);
                }
            }
            let mut delta = 0.0_f64;
            for k in 0..n_components {
                for j in 0..d {
                    delta += (dict[[k, j]] - prev_dict[[k, j]]).powi(2);
                }
            }
            prev_dict = dict.clone();
            if delta.sqrt() < tol {
                break;
            }
        }
        Ok(Self {
            components: dict,
            n_components,
            alpha,
            batch_size,
            n_iter: iters,
        })
    }
}

fn sparse_code(x: ndarray::ArrayView1<'_, f64>, dict: &Array2<f64>, alpha: f64) -> Vec<f64> {
    let n_components = dict.nrows();
    let d = dict.ncols();
    let mut a = vec![0.0_f64; n_components];
    for _ in 0..50 {
        let mut delta = 0.0_f64;
        for k in 0..n_components {
            let mut num = 0.0_f64;
            let mut denom = 0.0_f64;
            for j in 0..d {
                let mut resid = x[j];
                for l in 0..n_components {
                    if l == k {
                        continue;
                    }
                    resid -= a[l] * dict[[l, j]];
                }
                num += dict[[k, j]] * resid;
                denom += dict[[k, j]] * dict[[k, j]];
            }
            let z = if denom > 1e-30 {
                soft_threshold(num, alpha) / denom
            } else {
                0.0
            };
            let d_val = (z - a[k]).abs();
            if d_val > delta {
                delta = d_val;
            }
            a[k] = z;
        }
        if delta < 1e-6 {
            break;
        }
    }
    a
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

fn uniform_index(state: &mut u64, n: u64) -> usize {
    *state = state
        .wrapping_mul(6_364_136_223_846_793_005)
        .wrapping_add(1_442_695_040_888_963_407);
    let max = u64::MAX - (u64::MAX % n);
    if *state < max {
        (*state % n) as usize
    } else {
        (state.wrapping_mul(3) % n) as usize
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn minibatch_dict_learning_returns_dict_of_the_right_shape() {
        let x = array![
            [1.0_f64, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0],
            [1.0, 1.0, 0.0], [0.0, 1.0, 1.0]
        ];
        let m = MiniBatchDictionaryLearning::fit_with(x.view(), 3, 0.1, 3, 20, 1e-4, 42).unwrap();
        assert_eq!(m.components.shape(), &[3, 3]);
    }
}
