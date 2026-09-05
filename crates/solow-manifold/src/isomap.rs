//! Isomap — geodesic-distance MDS via a k-NN graph.

use ndarray::{Array1, Array2, ArrayView2};
use solow_core::{Error, Result};

/// Fitted Isomap embedding.
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Clone, Debug, PartialEq)]
pub struct Isomap {
    /// Low-dimensional embedding, `n × n_components`.
    pub embedding: Array2<f64>,
    /// Number of nearest neighbours used to build the graph.
    pub n_neighbors: usize,
    /// Number of components.
    pub n_components: usize,
    /// Recovered eigenvalues.
    pub eigenvalues: Array1<f64>,
}

impl Isomap {
    /// Fit onto `x` with `n_neighbors` and target `n_components`.
    pub fn fit(x: ArrayView2<'_, f64>, n_neighbors: usize, n_components: usize) -> Result<Self> {
        if x.nrows() == 0 || x.ncols() == 0 {
            return Err(Error::Value("Isomap::fit: x must be non-empty".into()));
        }
        if n_neighbors < 1 || n_neighbors >= x.nrows() {
            return Err(Error::Value(format!(
                "Isomap::fit: n_neighbors must be in [1, n-1] (got {n_neighbors}, n={})",
                x.nrows()
            )));
        }
        if n_components < 1 || n_components > x.nrows() {
            return Err(Error::Value(format!(
                "Isomap::fit: n_components must be in [1, n] (got {n_components})"
            )));
        }
        let n = x.nrows();
        // Pairwise distance matrix.
        let mut d = Array2::<f64>::zeros((n, n));
        for i in 0..n {
            for j in (i + 1)..n {
                let mut s = 0.0_f64;
                for k in 0..x.ncols() {
                    let dd = x[[i, k]] - x[[j, k]];
                    s += dd * dd;
                }
                let dij = s.sqrt();
                d[[i, j]] = dij;
                d[[j, i]] = dij;
            }
        }
        // k-NN sparsification.
        let mut adj = Array2::<f64>::from_elem((n, n), f64::INFINITY);
        for i in 0..n {
            adj[[i, i]] = 0.0;
            let mut idx: Vec<usize> = (0..n).filter(|&j| j != i).collect();
            idx.sort_by(|&a, &b| d[[i, a]].partial_cmp(&d[[i, b]]).unwrap());
            for &j in idx.iter().take(n_neighbors) {
                adj[[i, j]] = d[[i, j]];
                adj[[j, i]] = d[[i, j]];
            }
        }
        // Shortest paths — Floyd-Warshall (O(n³); fine for the target size).
        for k in 0..n {
            for i in 0..n {
                for j in 0..n {
                    let via = adj[[i, k]] + adj[[k, j]];
                    if via < adj[[i, j]] {
                        adj[[i, j]] = via;
                    }
                }
            }
        }
        // Any infinities → the graph is disconnected → error.
        for i in 0..n {
            for j in 0..n {
                if !adj[[i, j]].is_finite() {
                    return Err(Error::Value(
                        "Isomap::fit: k-NN graph is disconnected; increase n_neighbors".into(),
                    ));
                }
            }
        }
        // Classical MDS on the squared geodesic distances.
        let mut b = Array2::<f64>::zeros((n, n));
        for i in 0..n {
            for j in 0..n {
                b[[i, j]] = -0.5 * adj[[i, j]] * adj[[i, j]];
            }
        }
        // Double centring.
        let row_means: Vec<f64> = (0..n)
            .map(|i| b.row(i).iter().sum::<f64>() / n as f64)
            .collect();
        let col_means: Vec<f64> = (0..n)
            .map(|j| b.column(j).iter().sum::<f64>() / n as f64)
            .collect();
        let grand_mean: f64 = row_means.iter().sum::<f64>() / n as f64;
        for i in 0..n {
            for j in 0..n {
                b[[i, j]] += grand_mean - row_means[i] - col_means[j];
            }
        }
        // Eigendecomposition via Jacobi (dense O(n³), fine for these sizes).
        let (eigvals, eigvecs) = jacobi_symmetric(&b, 200, 1e-12);
        // Sort by descending eigenvalue.
        let mut order: Vec<usize> = (0..n).collect();
        order.sort_by(|&a, &b| eigvals[b].partial_cmp(&eigvals[a]).unwrap());
        let mut embedding = Array2::<f64>::zeros((n, n_components));
        let mut chosen_eig = Array1::<f64>::zeros(n_components);
        for k in 0..n_components {
            let idx = order[k];
            let sqrt_ev = eigvals[idx].max(0.0).sqrt();
            chosen_eig[k] = eigvals[idx];
            for i in 0..n {
                embedding[[i, k]] = eigvecs[[i, idx]] * sqrt_ev;
            }
        }
        Ok(Self {
            embedding,
            n_neighbors,
            n_components,
            eigenvalues: chosen_eig,
        })
    }
}

/// Cyclic Jacobi eigendecomposition for a symmetric matrix.
///
/// Returns `(eigenvalues, eigenvectors)` where `eigenvectors[:, i]`
/// is the eigenvector paired with `eigenvalues[i]`. `max_iter` bounds
/// the number of sweeps; `tol` is the off-diagonal absolute threshold
/// at which the iteration terminates.
///
/// Exposed publicly so downstream crates (e.g.
/// [`solow-decomposition`](https://docs.rs/solow-decomposition)) can
/// reuse the same symmetric eigendecomposition without pulling in
/// solow-linalg.
pub fn jacobi_symmetric(m: &Array2<f64>, max_iter: usize, tol: f64) -> (Vec<f64>, Array2<f64>) {
    // Cyclic Jacobi eigendecomposition — works for any symmetric matrix.
    let n = m.nrows();
    let mut a = m.clone();
    let mut v = Array2::<f64>::eye(n);
    for _ in 0..max_iter {
        // Find largest off-diagonal.
        let (mut p, mut q, mut best) = (0usize, 1usize, 0.0_f64);
        for i in 0..n {
            for j in (i + 1)..n {
                if a[[i, j]].abs() > best {
                    best = a[[i, j]].abs();
                    p = i;
                    q = j;
                }
            }
        }
        if best < tol {
            break;
        }
        let app = a[[p, p]];
        let aqq = a[[q, q]];
        let apq = a[[p, q]];
        let theta = (aqq - app) / (2.0 * apq);
        let t = if theta >= 0.0 {
            1.0 / (theta + (1.0 + theta * theta).sqrt())
        } else {
            1.0 / (theta - (1.0 + theta * theta).sqrt())
        };
        let c = 1.0 / (1.0 + t * t).sqrt();
        let s = t * c;
        // Update A.
        for i in 0..n {
            let aip = a[[i, p]];
            let aiq = a[[i, q]];
            a[[i, p]] = c * aip - s * aiq;
            a[[i, q]] = s * aip + c * aiq;
        }
        for j in 0..n {
            let apj = a[[p, j]];
            let aqj = a[[q, j]];
            a[[p, j]] = c * apj - s * aqj;
            a[[q, j]] = s * apj + c * aqj;
        }
        // Update V.
        for i in 0..n {
            let vip = v[[i, p]];
            let viq = v[[i, q]];
            v[[i, p]] = c * vip - s * viq;
            v[[i, q]] = s * vip + c * viq;
        }
    }
    let eig: Vec<f64> = (0..n).map(|i| a[[i, i]]).collect();
    (eig, v)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn isomap_recovers_2d_from_3d_swiss_roll_like_shape() {
        // A gentle 3-D curve. Isomap should recover a 1-D backbone.
        let n = 30usize;
        let mut rows: Vec<[f64; 3]> = Vec::with_capacity(n);
        for i in 0..n {
            let t = i as f64 * 0.2;
            rows.push([t.cos(), t.sin(), 0.05 * t]);
        }
        let flat: Vec<f64> = rows.into_iter().flatten().collect();
        let x = Array2::from_shape_vec((n, 3), flat).unwrap();
        let iso = Isomap::fit(x.view(), 4, 2).unwrap();
        assert_eq!(iso.embedding.dim(), (n, 2));
        // The two chosen eigenvalues are non-negative.
        for v in iso.eigenvalues.iter() {
            assert!(*v >= -1e-6);
        }
    }
}
