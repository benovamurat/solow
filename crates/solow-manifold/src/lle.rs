//! Locally Linear Embedding (Roweis-Saul 2000).

use ndarray::{Array1, Array2, ArrayView2};
use solow_core::{Error, Result};

use crate::isomap::jacobi_symmetric;

/// Fitted LLE embedding.
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Clone, Debug, PartialEq)]
pub struct LocallyLinearEmbedding {
    /// Low-dimensional embedding, `n × n_components`.
    pub embedding: Array2<f64>,
    /// Number of neighbours used.
    pub n_neighbors: usize,
    /// Number of components.
    pub n_components: usize,
}

impl LocallyLinearEmbedding {
    /// Fit onto `x`.
    pub fn fit(x: ArrayView2<'_, f64>, n_neighbors: usize, n_components: usize) -> Result<Self> {
        if x.nrows() == 0 || x.ncols() == 0 {
            return Err(Error::Value(
                "LocallyLinearEmbedding::fit: x must be non-empty".into(),
            ));
        }
        if n_neighbors < 1 || n_neighbors >= x.nrows() {
            return Err(Error::Value(format!(
                "LocallyLinearEmbedding::fit: n_neighbors in [1, n-1] (got {n_neighbors})"
            )));
        }
        if n_components < 1 || n_components >= x.nrows() {
            return Err(Error::Value(format!(
                "LocallyLinearEmbedding::fit: n_components in [1, n-1] (got {n_components})"
            )));
        }
        let n = x.nrows();
        // Nearest neighbours (Euclidean).
        let mut neighbours = vec![vec![0usize; n_neighbors]; n];
        for i in 0..n {
            let mut idx: Vec<(usize, f64)> = (0..n)
                .filter(|&j| j != i)
                .map(|j| {
                    let mut s = 0.0_f64;
                    for k in 0..x.ncols() {
                        let dd = x[[i, k]] - x[[j, k]];
                        s += dd * dd;
                    }
                    (j, s)
                })
                .collect();
            idx.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
            for kk in 0..n_neighbors {
                neighbours[i][kk] = idx[kk].0;
            }
        }
        // Solve for per-row reconstruction weights via a small linear system.
        let mut w = Array2::<f64>::zeros((n, n));
        for i in 0..n {
            let k = n_neighbors;
            // Local Gram matrix of centred neighbours.
            let mut z = Array2::<f64>::zeros((k, x.ncols()));
            for (kk, &j) in neighbours[i].iter().enumerate() {
                for c in 0..x.ncols() {
                    z[[kk, c]] = x[[j, c]] - x[[i, c]];
                }
            }
            let mut g = Array2::<f64>::zeros((k, k));
            for a in 0..k {
                for b in 0..k {
                    let mut s = 0.0_f64;
                    for c in 0..x.ncols() {
                        s += z[[a, c]] * z[[b, c]];
                    }
                    g[[a, b]] = s;
                }
            }
            // Regularise Gram matrix.
            let trace: f64 = (0..k).map(|a| g[[a, a]]).sum();
            let reg = (1e-3 * trace / k as f64).max(1e-6);
            for a in 0..k {
                g[[a, a]] += reg;
            }
            // Solve g w = 1 by Gauss-Jordan on the small system.
            let ones = vec![1.0_f64; k];
            let solved = solve_small(&g, &ones);
            let sum: f64 = solved.iter().sum();
            for (kk, &j) in neighbours[i].iter().enumerate() {
                w[[i, j]] = solved[kk] / sum;
            }
        }
        // Build M = (I - W)ᵀ (I - W); eigendecompose for the smallest
        // non-zero `n_components` eigenvectors.
        let mut m = Array2::<f64>::zeros((n, n));
        for i in 0..n {
            m[[i, i]] += 1.0;
        }
        for i in 0..n {
            for j in 0..n {
                m[[i, j]] -= w[[i, j]];
            }
        }
        // Symmetric product mᵀm.
        let mut mtm = Array2::<f64>::zeros((n, n));
        for i in 0..n {
            for j in 0..n {
                let mut s = 0.0_f64;
                for k in 0..n {
                    s += m[[k, i]] * m[[k, j]];
                }
                mtm[[i, j]] = s;
            }
        }
        // Jacobi eigendecomposition.
        let (eigvals, eigvecs) = jacobi_symmetric(&mtm, 300, 1e-12);
        let mut order: Vec<usize> = (0..n).collect();
        order.sort_by(|&a, &b| eigvals[a].partial_cmp(&eigvals[b]).unwrap());
        // Skip the smallest (~zero) eigenvalue that corresponds to the
        // trivial constant vector.
        let mut embedding = Array2::<f64>::zeros((n, n_components));
        for k in 0..n_components {
            let idx = order[k + 1];
            for i in 0..n {
                embedding[[i, k]] = eigvecs[[i, idx]];
            }
        }
        Ok(Self {
            embedding,
            n_neighbors,
            n_components,
        })
    }
}

fn solve_small(m: &Array2<f64>, rhs: &[f64]) -> Array1<f64> {
    let n = m.nrows();
    let mut a = vec![vec![0.0_f64; n + 1]; n];
    for i in 0..n {
        for j in 0..n {
            a[i][j] = m[[i, j]];
        }
        a[i][n] = rhs[i];
    }
    for i in 0..n {
        // Partial pivot.
        let mut best = i;
        for r in (i + 1)..n {
            if a[r][i].abs() > a[best][i].abs() {
                best = r;
            }
        }
        if best != i {
            a.swap(i, best);
        }
        let piv = a[i][i];
        if piv.abs() < 1e-300 {
            return Array1::<f64>::zeros(n);
        }
        for c in 0..(n + 1) {
            a[i][c] /= piv;
        }
        for r in 0..n {
            if r == i {
                continue;
            }
            let f = a[r][i];
            for c in 0..(n + 1) {
                a[r][c] -= f * a[i][c];
            }
        }
    }
    let mut out = Array1::<f64>::zeros(n);
    for i in 0..n {
        out[i] = a[i][n];
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lle_returns_the_requested_shape() {
        let n = 20usize;
        let mut rows: Vec<[f64; 3]> = Vec::new();
        for i in 0..n {
            let t = i as f64 * 0.3;
            rows.push([t.cos(), t.sin(), 0.05 * t]);
        }
        let flat: Vec<f64> = rows.into_iter().flatten().collect();
        let x = Array2::from_shape_vec((n, 3), flat).unwrap();
        let lle = LocallyLinearEmbedding::fit(x.view(), 4, 2).unwrap();
        assert_eq!(lle.embedding.dim(), (n, 2));
        // Non-degenerate.
        let (mut mn, mut mx) = (f64::INFINITY, f64::NEG_INFINITY);
        for &v in lle.embedding.iter() {
            if v < mn {
                mn = v;
            }
            if v > mx {
                mx = v;
            }
        }
        assert!(mx - mn > 1e-6);
    }
}
