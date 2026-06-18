//! # solow-linalg
//!
//! Pure-Rust dense linear algebra for the Solow statistical stack. No system
//! LAPACK/BLAS is required: the decompositions are implemented here and validated
//! against an authoritative reference.
//!
//! - [`cholesky`] — `A = L Lᵀ`
//! - [`lu_factor`] — `P A = L U` (powers [`solve`], [`inv`], [`det`])
//! - [`qr`] — economy Householder QR
//! - [`svd`] — economy one-sided Jacobi SVD
//! - [`eigh`] — symmetric eigendecomposition (cyclic Jacobi)
//! - [`pinv`] / [`lstsq`] / [`matrix_rank`] — SVD-based

mod decomp;
mod solve;

pub use decomp::{cholesky, eigh, lu_factor, qr, svd, Lu};
pub use solve::{
    cholesky_solve, det, inv, lstsq, lstsq_qr, matrix_rank, pinv, pinv_rcond, solve, solve_matrix,
};

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use ndarray::{array, Array2};

    fn frob(a: &Array2<f64>) -> f64 {
        a.iter().map(|&x| x * x).sum::<f64>().sqrt()
    }

    fn norm1(a: &ndarray::Array1<f64>) -> f64 {
        a.iter().map(|&x| x * x).sum::<f64>().sqrt()
    }

    #[test]
    fn cholesky_reconstructs() {
        let a = array![[4.0, 2.0, 2.0], [2.0, 5.0, 1.0], [2.0, 1.0, 6.0]];
        let l = cholesky(&a).unwrap();
        let recon = l.dot(&l.t());
        assert_abs_diff_eq!(frob(&(&recon - &a)), 0.0, epsilon = 1e-12);
    }

    #[test]
    fn lu_solve_and_inv() {
        let a = array![[2.0, 1.0, 1.0], [4.0, -6.0, 0.0], [-2.0, 7.0, 2.0]];
        let b = array![5.0, -2.0, 9.0];
        let x = solve(&a, &b).unwrap();
        let recon = a.dot(&x);
        assert_abs_diff_eq!(norm1(&(&recon - &b)), 0.0, epsilon = 1e-10);

        let ainv = inv(&a).unwrap();
        let id = a.dot(&ainv);
        let eye = Array2::<f64>::eye(3);
        assert_abs_diff_eq!(frob(&(&id - &eye)), 0.0, epsilon = 1e-10);
    }

    #[test]
    fn det_matches_known() {
        let a = array![[1.0, 2.0], [3.0, 4.0]];
        assert_abs_diff_eq!(det(&a).unwrap(), -2.0, epsilon = 1e-12);
        let singular = array![[1.0, 2.0], [2.0, 4.0]];
        assert_abs_diff_eq!(det(&singular).unwrap(), 0.0, epsilon = 1e-12);
    }

    #[test]
    fn qr_reconstructs_and_orthonormal() {
        let a = array![
            [12.0, -51.0, 4.0],
            [6.0, 167.0, -68.0],
            [-4.0, 24.0, -41.0],
            [-1.0, 1.0, 0.0],
            [2.0, 0.0, 3.0]
        ];
        let (q, r) = qr(&a).unwrap();
        let recon = q.dot(&r);
        assert_abs_diff_eq!(frob(&(&recon - &a)), 0.0, epsilon = 1e-9);
        // Qᵀ Q = I_n
        let qtq = q.t().dot(&q);
        let eye = Array2::<f64>::eye(3);
        assert_abs_diff_eq!(frob(&(&qtq - &eye)), 0.0, epsilon = 1e-9);
        // R upper triangular.
        for i in 0..3 {
            for j in 0..i {
                assert_abs_diff_eq!(r[[i, j]], 0.0, epsilon = 1e-9);
            }
        }
    }

    #[test]
    fn svd_reconstructs_tall_and_wide() {
        let tall = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]];
        let (u, s, vt) = svd(&tall).unwrap();
        let sd = Array2::from_diag(&s);
        let recon = u.dot(&sd).dot(&vt);
        assert_abs_diff_eq!(frob(&(&recon - &tall)), 0.0, epsilon = 1e-9);
        // Descending singular values.
        assert!(s[0] >= s[1]);

        let wide = tall.t().to_owned();
        let (u2, s2, vt2) = svd(&wide).unwrap();
        let sd2 = Array2::from_diag(&s2);
        let recon2 = u2.dot(&sd2).dot(&vt2);
        assert_abs_diff_eq!(frob(&(&recon2 - &wide)), 0.0, epsilon = 1e-9);
    }

    #[test]
    fn lstsq_qr_matches_pinv_and_detects_rank_deficiency() {
        // Tall full-rank design: QR least squares == pinv least squares.
        let a = array![
            [1.0, 0.2, -0.5],
            [1.0, 1.1, 0.3],
            [1.0, 2.0, 1.7],
            [1.0, 3.2, -0.8],
            [1.0, 4.1, 2.2],
            [1.0, 5.0, 0.1]
        ];
        let b = array![1.0, 2.2, 3.1, 3.8, 5.3, 6.0];
        let (x_qr, ncp_qr) = lstsq_qr(&a, &b).unwrap().expect("full rank → Some");
        let x_pinv = lstsq(&a, &b).unwrap();
        for i in 0..3 {
            assert_abs_diff_eq!(x_qr[i], x_pinv[i], epsilon = 1e-10);
        }
        // ncp == (AᵀA)⁻¹.
        let ata = a.t().dot(&a);
        let ata_inv = inv(&ata).unwrap();
        assert_abs_diff_eq!(frob(&(&ncp_qr - &ata_inv)), 0.0, epsilon = 1e-9);

        // Rank-deficient design (col 3 = 2·col 2) → None (caller falls back to SVD).
        let rd = array![
            [1.0, 1.0, 2.0],
            [1.0, 2.0, 4.0],
            [1.0, 3.0, 6.0],
            [1.0, 4.0, 8.0]
        ];
        let b2 = array![1.0, 2.0, 3.0, 4.0];
        assert!(lstsq_qr(&rd, &b2).unwrap().is_none());
    }

    #[test]
    fn cholesky_solve_matches_general_solve() {
        let a = array![[4.0, 2.0, 2.0], [2.0, 5.0, 1.0], [2.0, 1.0, 6.0]];
        let b = array![1.0, -2.0, 3.0];
        let x = cholesky_solve(&a, &b).unwrap();
        assert_abs_diff_eq!(norm1(&(&a.dot(&x) - &b)), 0.0, epsilon = 1e-10);
    }

    #[test]
    fn pinv_left_inverse() {
        let a = array![[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]];
        let (p, _s) = pinv(&a).unwrap();
        // For full column rank, pinv(A) A = I.
        let prod = p.dot(&a);
        let eye = Array2::<f64>::eye(2);
        assert_abs_diff_eq!(frob(&(&prod - &eye)), 0.0, epsilon = 1e-9);
    }

    #[test]
    fn eigh_reconstructs_symmetric() {
        let a = array![[2.0, 1.0, 0.0], [1.0, 2.0, 1.0], [0.0, 1.0, 2.0]];
        let (w, v) = eigh(&a).unwrap();
        // Ascending eigenvalues; known spectrum 2 - sqrt2, 2, 2 + sqrt2.
        assert!(w[0] <= w[1] && w[1] <= w[2]);
        assert_abs_diff_eq!(w[0], 2.0 - 2.0_f64.sqrt(), epsilon = 1e-9);
        assert_abs_diff_eq!(w[2], 2.0 + 2.0_f64.sqrt(), epsilon = 1e-9);
        // Reconstruction V diag(w) Vᵀ = A.
        let recon = v.dot(&Array2::from_diag(&w)).dot(&v.t());
        assert_abs_diff_eq!(frob(&(&recon - &a)), 0.0, epsilon = 1e-9);
        // Orthonormal eigenvectors.
        let vtv = v.t().dot(&v);
        let eye = Array2::<f64>::eye(3);
        assert_abs_diff_eq!(frob(&(&vtv - &eye)), 0.0, epsilon = 1e-9);
    }
}
